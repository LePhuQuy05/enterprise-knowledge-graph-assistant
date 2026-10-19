"""Two synthetic paid calls at most, behind a separate initial-smoke funding gate."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.cost import PriceSheet, TokenUsage
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import (
    HAIKU,
    HAIKU_ROUTE,
    TITAN,
    BedrockProvider,
    CallContext,
    ModelGateway,
    ModelRoutes,
)

SYSTEM = "Return one JSON object with no markdown and no additional text. Use only the supplied source."
PROMPT = ('Source [s1]: "Demo uploads are read-only."\nQuestion: Are demo uploads writable?\n'
          'Return keys answer (Yes or No), citation_id, quote. Quote the exact supporting sentence.')


def smoke_forecast(prices: PriceSheet, *, embedding_only=False) -> dict:
    now = datetime.now(UTC)
    generation = prices.rate_for(HAIKU, HAIKU_ROUTE, 'us-east-1', now=now, live=True)
    embedding = prices.rate_for(TITAN, TITAN, 'us-east-1', now=now, live=True)
    maximum = 0 if embedding_only else generation.quote(TokenUsage(input_tokens=6144, output_tokens=64))
    maximum += embedding.quote(TokenUsage(input_tokens=8192, output_tokens=0))
    return {'status': 'FORECAST_ONLY', 'maximum_two_call_reservations_micro_usd': maximum,
            'run_cap_micro_usd': 100_000, 'max_inference_calls': 1 if embedding_only else 2, 'inference_calls': 0,
            'notice': 'Conservative token/rate bounds, not an invoice or proof of credit coverage'}


def run_smoke(*, execute=False, embedding_only=False, output_root=Path('results/model_smoke'),
              funding_path=Path('configs/funding.json')) -> dict:
    prices = PriceSheet.model_validate_json(Path('configs/prices/aws-us-east-1-2026-10-16.json').read_bytes())
    forecast = smoke_forecast(prices, embedding_only=embedding_only)
    if not execute:
        return forecast
    funding = json.loads(funding_path.read_bytes())
    if funding.get('anthropic_credit_coverage') != 'CONFIRMED' and (
        type(funding.get('personal_budget_for_initial_smoke_micro_usd')) is not int
        or funding['personal_budget_for_initial_smoke_micro_usd'] != 100_000
    ):
        raise ValueError('Initial paid smoke funding is unconfirmed; no AWS model client created')
    session = boto3.Session(profile_name='kg-dev', region_name='us-east-1')
    if not session.client('sts').get_caller_identity()['Arn'].endswith(':user/kg-dev'):
        raise ValueError('Known non-root development identity required')
    availability = session.client('bedrock').get_foundation_model_availability(modelId=TITAN if embedding_only else HAIKU)
    if availability['agreementAvailability']['status'] != 'AVAILABLE' or (
        availability['authorizationStatus'] != 'AUTHORIZED'
        or availability['entitlementAvailability'] != 'AVAILABLE'
        or availability['regionAvailability'] != 'AVAILABLE'
    ):
        raise ValueError('Model access changed; no inference permitted')
    run = 'initial-model-smoke-v1'  # All retries share one cumulative 0.10 USD allowance.
    probe_id = 'probe-' + uuid.uuid4().hex
    folder = output_root / probe_id
    folder.mkdir(parents=True, exist_ok=False)
    routes = ModelRoutes.model_validate_json(Path('configs/models/bedrock-us-east-1-v2.json').read_bytes())
    # Explicit funding for this bounded command does not enable production/cloud generation.
    routes = routes.model_copy(update={'live_inference_enabled': True})
    ledger = Ledger(DynamoLedgerStore(session.client('dynamodb', config=Config(
        retries={'total_max_attempts': 2}, connect_timeout=3, read_timeout=5,
    )), 'kg-r72-query-state'))
    ledger.initialize_month('kg-r72', datetime.now(UTC).strftime('%Y-%m'))
    ledger.initialize_run('kg-r72', 'demo', run, cap_micro_usd=100_000)
    gateway = ModelGateway(ledger, BedrockProvider(session), routes, prices)
    result = {'status': 'INCOMPLETE', 'run_id': run, 'probe_id': probe_id, 'synthetic_quality_probe': True,
              'smoke_kind': 'EMBEDDING_ONLY' if embedding_only else 'GENERATION_AND_EMBEDDING',
              'price_sheet_version': prices.version, 'model_config_version': routes.version,
              'inference_calls_completed': 0, 'receipts': [], 'production_model_enabled': False}
    try:
        context = CallContext(tenant_id='demo', run_id=run, request_id=probe_id + '-generation', stage='SMOKE')
        if not embedding_only:
            generated = gateway.generate(context, system=SYSTEM, prompt=PROMPT, max_output_tokens=64)
            result['inference_calls_completed'] += 1
            result['receipts'].append({'operation_id': generated.operation_id,
                                       'usage': generated.usage.model_dump(), 'actual_micro_usd': generated.actual_micro_usd})
            expected = {'answer': 'No', 'citation_id': 's1', 'quote': 'Demo uploads are read-only.'}
            result['json_quote_smoke_passed'] = json.loads(generated.text) == expected
            result['synthetic_response'] = generated.text
        vector = gateway.embed(context.model_copy(update={'request_id': probe_id + '-embedding', 'stage': 'EMBED'}),
                               text='Demo uploads are read-only.')
        result['inference_calls_completed'] += 1
        result['receipts'].append({'operation_id': vector.operation_id,
                                   'usage': vector.usage.model_dump(), 'actual_micro_usd': vector.actual_micro_usd})
        result['embedding_dimension'] = len(vector.embedding)
        result['status'] = 'PASS' if embedding_only or result['json_quote_smoke_passed'] else 'QUALITY_SMOKE_FAILED'
    except Exception as error:
        result['error_category'] = type(error).__name__
        if hasattr(error, 'operation_id'):
            result['failed_operation_id'] = error.operation_id
        cause = error.__cause__ or error
        result['failure_category'] = type(cause).__name__
        if hasattr(cause, 'response'):
            result['provider_error_code'] = cause.response.get('Error', {}).get('Code')
            result['provider_http_status'] = cause.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            result['provider_request_id'] = cause.response.get('ResponseMetadata', {}).get('RequestId')
        # Operation may have dispatched before failure; receipts alone are not the ledger total.
        result['reconciliation_required'] = True
    finally:
        try:
            counter = ledger.store.get(ledger.run_key(run))
            result['run_settled_micro_usd'] = counter['settled_micro_usd']
            result['run_reserved_micro_usd'] = counter['reserved_micro_usd']
            result['run_halted'] = counter['halted']
        except Exception:
            result['ledger_read_unavailable'] = True
            result['reconciliation_required'] = True
        (folder / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
