"""First authenticated URL, with an inline stdlib handler and no model permissions."""

import base64
import json
import lzma
from pathlib import Path

from aws_cdk import App, Aws, BootstraplessSynthesizer, CfnOutput, Fn, RemovalPolicy, Stack, Tags
from aws_cdk import aws_apigatewayv2 as api
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

STACK_NAME = "kg-r72-tracer"
ROOT = Path(__file__).resolve().parents[1]


def handler_source() -> str:
    files = {"/": ("bootstrap.html", "text/html; charset=utf-8"),
             "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
             "/assets/app.js": ("bootstrap.mjs", "text/javascript; charset=utf-8"),
             "/assets/auth.mjs": ("auth.mjs", "text/javascript; charset=utf-8")}
    # Bootstrap is only the sign-in/setup preview, never an active document workspace.
    # Workspace and corpus projection belong to the query ZIP, not CFN's 51,200-byte inline budget.
    assets = {path: {"body": (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8"),
                     "content_type": mime} for path, (name, mime) in files.items()}
    if any(len(asset["body"].encode()) > 512 * 1024 for asset in assets.values()):
        raise ValueError("Static asset exceeds configured limit")
    source = (ROOT / "app/tracer.py").read_text(encoding="utf-8")
    # Compress the trusted handler and its exact asset bodies together. Leaving
    # Python source outside the envelope wastes inline CFN headroom as UI grows.
    # Lambda's stdlib supplies all three modules; the ZIP runtime is unchanged.
    payload = {"source": source, "assets": assets}
    compressed = lzma.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    # Base85 saves ~6.25% over Base64, preserving CFN's 51,200-byte inline template limit.
    encoded = base64.b85encode(compressed).decode("ascii")
    return ("import base64, json, lzma\n_payload = json.loads(lzma.decompress(base64.b85decode("
            + repr(encoded) + ")))\nexec(_payload['source'])\nSTATIC_ASSETS = _payload['assets']\ndel _payload\n")


class TracerStack(Stack):
    def __init__(self, scope: Construct, construct_id=STACK_NAME, *, runtime=None, registration_enabled=False):
        if registration_enabled and runtime is None:
            raise ValueError("Registration requires the query runtime enrollment handler")
        super().__init__(scope, construct_id, stack_name=STACK_NAME,
                         synthesizer=BootstraplessSynthesizer(),
                         description="KG R72 authenticated setup preview; no inference or RAG answers")
        Tags.of(self).add("Project", "kg-r72")
        Tags.of(self).add("Owner", "portfolio")
        http = api.CfnApi(self, "HttpApi", name="kg-r72", protocol_type="HTTP")
        endpoint = Fn.join("", ["https://", http.ref, ".execute-api.", Aws.REGION, ".", Aws.URL_SUFFIX])
        pool = cognito.CfnUserPool(
            self, "ReviewerPool", user_pool_name="kg-r72-reviewers", user_pool_tier="LITE",
            admin_create_user_config=cognito.CfnUserPool.AdminCreateUserConfigProperty(
                allow_admin_create_user_only=not registration_enabled),
            auto_verified_attributes=["email"],
            account_recovery_setting=cognito.CfnUserPool.AccountRecoverySettingProperty(
                recovery_mechanisms=[cognito.CfnUserPool.RecoveryOptionProperty(name="verified_email", priority=1)]),
            verification_message_template=cognito.CfnUserPool.VerificationMessageTemplateProperty(
                default_email_option="CONFIRM_WITH_CODE"),
            username_configuration=cognito.CfnUserPool.UsernameConfigurationProperty(case_sensitive=False),
            policies=cognito.CfnUserPool.PoliciesProperty(password_policy=
                cognito.CfnUserPool.PasswordPolicyProperty(
                    minimum_length=14, require_lowercase=True, require_uppercase=True,
                    require_numbers=True, require_symbols=True, temporary_password_validity_days=3,
                )),
            mfa_configuration="OFF",  # No SMS charges; owner AWS-account MFA is separate.
        )
        pool.apply_removal_policy(RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE)
        server = cognito.CfnUserPoolResourceServer(
            self, "Scopes", identifier="kg", name="Knowledge Graph API", user_pool_id=pool.ref,
            scopes=[cognito.CfnUserPoolResourceServer.ResourceServerScopeTypeProperty(
                scope_name=name, scope_description=description,
            ) for name, description in (("read", "Read own demo results"), ("query", "Ask the demo"))],
        )
        client = cognito.CfnUserPoolClient(
            self, "BrowserClient", user_pool_id=pool.ref, client_name="kg-r72-browser",
            generate_secret=False, supported_identity_providers=["COGNITO"],
            allowed_o_auth_flows_user_pool_client=True, allowed_o_auth_flows=["code"],
            allowed_o_auth_scopes=["openid", "kg/read", "kg/query"],
            callback_ur_ls=[endpoint + "/auth/callback"], logout_ur_ls=[endpoint + "/"],
            explicit_auth_flows=["ALLOW_REFRESH_TOKEN_AUTH"], enable_token_revocation=True,
            prevent_user_existence_errors="ENABLED", access_token_validity=5, id_token_validity=5,
            refresh_token_validity=1,
            token_validity_units=cognito.CfnUserPoolClient.TokenValidityUnitsProperty(
                access_token="minutes", id_token="minutes", refresh_token="hours"),
        )
        client.add_resource_dependency(server)
        prefix = Fn.join("", ["kg-r72-", Fn.select(2, Fn.split("/", Aws.STACK_ID))])
        domain = cognito.CfnUserPoolDomain(self, "HostedUi", domain=prefix,
                                          user_pool_id=pool.ref, managed_login_version=1)
        issuer = Fn.join("", ["https://cognito-idp.", Aws.REGION, ".", Aws.URL_SUFFIX, "/", pool.ref])
        public_config = {"issuer": issuer, "client_id": client.ref,
                         "auth_domain": Fn.join("", ["https://", prefix, ".auth.", Aws.REGION,
                                                       ".amazoncognito.com"]),
                         "callback_url": endpoint + "/auth/callback", "logout_url": endpoint + "/",
                         "registration_enabled": registration_enabled}
        log_group = logs.CfnLogGroup(self, "QueryLogs", log_group_name="/aws/lambda/kg-r72-query",
                                     retention_in_days=7)
        role = iam.CfnRole(
            self, "QueryRole", role_name="kg-r72-query-runtime",
            assume_role_policy_document={"Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }]},
            policies=[iam.CfnRole.PolicyProperty(policy_name="kg-r72-tracer-only", policy_document={
                "Version": "2012-10-17", "Statement": [
                    {"Effect": "Allow", "Action": ["dynamodb:GetItem"],
                     "Resource": Fn.sub("arn:${AWS::Partition}:dynamodb:${AWS::Region}:${AWS::AccountId}:table/kg-r72-metadata"),
                     "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["USER#*"]}}},
                    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                     "Resource": Fn.sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/kg-r72-query:log-stream:*")},
                ],
            })],
        )
        if runtime is not None:
            query_table = Fn.sub("arn:${AWS::Partition}:dynamodb:${AWS::Region}:${AWS::AccountId}:table/kg-r72-query-state")
            metadata_table = Fn.sub("arn:${AWS::Partition}:dynamodb:${AWS::Region}:${AWS::AccountId}:table/kg-r72-metadata")
            keys = ["TENANT#demo#USER#*", "QUOTA#kg-r72#*", "RUN#external-pilot10-r72-v1",
                    "FUNDING#kg-r72#external-r72-v1", "BUDGET#kg-r72#*", "OP#op_*"]
            role.policies = [iam.CfnRole.PolicyProperty(policy_name="kg-r72-query-serving", policy_document={
                "Version": "2012-10-17", "Statement": [
                    {"Effect": "Allow", "Action": ["dynamodb:GetItem"], "Resource": metadata_table,
                     "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["USER#*", "TENANT#demo"]}}},
                    {"Effect": "Allow", "Action": ["dynamodb:GetItem"], "Resource": query_table,
                     "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": keys}}},
                    {"Effect": "Allow", "Action": ["dynamodb:Query"], "Resource": query_table,
                     "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["TENANT#demo#USER#*"]}}},
                    {"Effect": "Allow", "Action": ["dynamodb:PutItem"], "Resource": query_table,
                     "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": keys},
                                   "ForAnyValue:StringEquals": {"dynamodb:EnclosingOperation": ["TransactWriteItems"]}}},
                    {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": [
                        Fn.join("", ["arn:", Aws.PARTITION, ":s3:::", runtime["assets"]["runtime_config"]["bucket"], "/", r["key"]])
                        for r in runtime["assets"]["objects"] if r["kind"] in {"MODEL", "SNAPSHOT"}]},
                    {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": runtime["secret_arn"]},
                    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                     "Resource": Fn.sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/kg-r72-query:log-stream:*")},
                ]})]
            if registration_enabled:
                scoped_document = role.policies[0].policy_document
                scoped_document["Statement"].extend([
                    {"Effect": "Allow", "Action": ["dynamodb:PutItem"], "Resource": metadata_table,
                     "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["USER#*"]},
                                   "ForAnyValue:StringEquals": {"dynamodb:EnclosingOperation": ["TransactWriteItems"]}}},
                    {"Effect": "Allow", "Action": ["cognito-idp:AdminGetUser"], "Resource": pool.attr_arn},
                ])
                # CDK returns copies of nested property structs: assign the amended document back.
                role.policies = [iam.CfnRole.PolicyProperty(
                    policy_name="kg-r72-query-serving", policy_document=scoped_document)]
        function = lambda_.CfnFunction(
            self, "QueryFunction", function_name="kg-r72-query", runtime="python3.12",
            architectures=["x86_64"], memory_size=1024, timeout=28,
            ephemeral_storage=lambda_.CfnFunction.EphemeralStorageProperty(size=512),
            handler="index.handler", role=role.attr_arn,
            code=lambda_.CfnFunction.CodeProperty(zip_file=handler_source()),
            environment=lambda_.CfnFunction.EnvironmentProperty(variables={
                "PUBLIC_CONFIG_JSON": self.to_json_string(public_config),
                "DEMO_MODE": "FIXTURE", "MODEL_ENABLED": "false",
            }),
        )
        function.add_resource_dependency(log_group)
        if runtime is not None:
            function.handler = "app.query_handler.handler"
            function.code = lambda_.CfnFunction.CodeProperty(
                s3_bucket=runtime["assets"]["runtime_config"]["bucket"], s3_key=runtime["assets"]["package_key"])
            function.environment = lambda_.CfnFunction.EnvironmentProperty(variables={
                "PUBLIC_CONFIG_JSON": self.to_json_string(public_config), "MODEL_ENABLED": "true",
                "DEMO_MODE": "LIVE", "KG_DEEPSEEK_SECRET_ARN": runtime["secret_arn"]})
        integration = api.CfnIntegration(self, "LambdaIntegration", api_id=http.ref,
                                         integration_type="AWS_PROXY", integration_uri=function.attr_arn,
                                         payload_format_version="2.0", timeout_in_millis=29000)
        target = Fn.join("", ["integrations/", integration.ref])
        authorizer = api.CfnAuthorizer(
            self, "JwtAuthorizer", api_id=http.ref, name="kg-r72-reviewer-jwt", authorizer_type="JWT",
            identity_source=["$request.header.Authorization"],
            jwt_configuration=api.CfnAuthorizer.JWTConfigurationProperty(
                audience=[client.ref], issuer=issuer),
        )
        routes = []
        for index, route in enumerate(("GET /", "GET /assets/{proxy+}", "GET /auth/callback", "GET /config.json")):
            routes.append(api.CfnRoute(self, f"PublicRoute{index}", api_id=http.ref, route_key=route,
                                       authorization_type="NONE", target=target))
        routes.append(api.CfnRoute(self, "PrivateStatus", api_id=http.ref, route_key="GET /v1/status",
                                   authorization_type="JWT", authorizer_id=authorizer.ref,
                                   authorization_scopes=["kg/read"], target=target))
        if runtime is not None:
            for logical, route_key, scope in (
                ("PrivateQueryCreate", "POST /v1/queries", "kg/query"),
                ("PrivateQueryList", "GET /v1/queries", "kg/read"),
                ("PrivateQueryDetail", "GET /v1/queries/{request_id}", "kg/read"),
                ("PrivateSource", "GET /v1/sources/{chunk_id}", "kg/read"),
            ):
                routes.append(api.CfnRoute(self, logical, api_id=http.ref, route_key=route_key,
                    authorization_type="JWT", authorizer_id=authorizer.ref, authorization_scopes=[scope], target=target))
        stage = api.CfnStage(self, "DefaultStage", api_id=http.ref, stage_name="$default", auto_deploy=True,
                     default_route_settings=api.CfnStage.RouteSettingsProperty(
                         throttling_burst_limit=2, throttling_rate_limit=1),
                     route_settings={route: {"ThrottlingBurstLimit": 10, "ThrottlingRateLimit": 5}
                                     for route in ("GET /", "GET /assets/{proxy+}",
                                                   "GET /auth/callback", "GET /config.json")})
        # RouteSettings names are validated by API Gateway when the stage is created.
        # An ApiId reference alone does not make CFN wait for the routes to exist.
        for route in routes:
            stage.add_resource_dependency(route)
        if runtime is not None:
            stage.route_settings = {route: {"ThrottlingBurstLimit": 10, "ThrottlingRateLimit": 5}
                for route in ("GET /", "GET /assets/{proxy+}", "GET /auth/callback", "GET /config.json",
                              "GET /v1/status", "GET /v1/queries", "GET /v1/queries/{request_id}", "GET /v1/sources/{chunk_id}")}
        lambda_.CfnPermission(self, "ApiInvoke", action="lambda:InvokeFunction",
                              function_name=function.ref, principal="apigateway.amazonaws.com",
                              source_account=Aws.ACCOUNT_ID,
                              source_arn=Fn.sub("arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:${ApiId}/*/*", {"ApiId": http.ref}))
        CfnOutput(self, "PublicUrl", value=endpoint + "/")
        CfnOutput(self, "UserPoolId", value=pool.ref)
        CfnOutput(self, "BrowserClientId", value=client.ref)
        CfnOutput(self, "AuthDomain", value=public_config["auth_domain"])
        # Domain is not needed by the function to serve config; CFN creates all before completion.
        self.hosted_ui = domain


def synthesize_tracer():
    app = App(analytics_reporting=False)
    return TracerStack(app)
