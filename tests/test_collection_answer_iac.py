import pytest

from app.cloud_owned_answers import CollectionAnswerPolicy
from tests.test_owned_runs_iac import synthesize

SECRET = "arn:aws:secretsmanager:us-east-1:357433444130:secret:kg-r72/deepseek-fixture"


def test_answer_optin_has_no_new_resources_and_only_worker_reads_exact_secret(tmp_path):
    before = synthesize(tmp_path / "before")["Resources"]
    after = synthesize(tmp_path / "after", answer_policy=CollectionAnswerPolicy().model_dump(mode="json"),
                       deepseek_secret_arn=SECRET)["Resources"]
    assert set(before) == set(after) and len(after) == 30
    for kind in ("Api", "Worker", "Watchdog"):
        variables = after[kind]["Properties"]["Environment"]["Variables"]
        assert variables["MODEL_ENABLED"] == ("true" if kind == "Worker" else "false")
        statements = after[kind + "Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        secrets = [s for s in statements if "secretsmanager:GetSecretValue" in s["Action"]]
        if kind == "Worker":
            assert len(secrets) == 1 and secrets[0]["Resource"] == SECRET
            assert variables["KG_DEEPSEEK_SECRET_ARN"] == SECRET
        else:
            assert not secrets and "KG_DEEPSEEK_SECRET_ARN" not in variables
        financial = [s for s in statements if "OP#*" in s.get("Condition", {}).get(
            "ForAllValues:StringLike", {}).get("dynamodb:LeadingKeys", [])]
        assert len(financial) == 1 and financial[0]["Action"] == ["dynamodb:PutItem"]
        condition = financial[0]["Condition"]
        assert condition["ForAnyValue:StringEquals"]["dynamodb:EnclosingOperation"] == ["TransactWriteItems"]
        assert "RUN#collection316-serving-v1" in condition["ForAllValues:StringLike"]["dynamodb:LeadingKeys"]
        assert "RUN#*" not in condition["ForAllValues:StringLike"]["dynamodb:LeadingKeys"]


def test_cannot_add_secret_without_answer_gate_or_use_foreign_secret_account(tmp_path):
    with pytest.raises(ValueError, match="CANNOT_READ_SECRETS"):
        synthesize(tmp_path / "closed", deepseek_secret_arn=SECRET)
    with pytest.raises(ValueError, match="EXACT_EXISTING_SECRET"):
        synthesize(tmp_path / "foreign", answer_policy=CollectionAnswerPolicy().model_dump(mode="json"),
                   deepseek_secret_arn=SECRET.replace("357433444130", "000000000000"))
