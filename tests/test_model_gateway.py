import io
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from app.cost import PriceSheet
from app.ledger import AlreadyDispatched, BudgetDenied, DynamoLedgerStore, Ledger, MemoryLedgerStore
from app.model_gateway import (
    HAIKU,
    HAIKU_ROUTE,
    BedrockProvider,
    CallContext,
    FixtureProvider,
    GatewayError,
    ModelGateway,
    ModelRoutes,
    UncertainModelCall,
)

NOW = datetime(2026, 10, 16, 17, tzinfo=UTC)
ROUTES = ModelRoutes.model_validate_json(Path('configs/models/bedrock-us-east-1-v2.json').read_bytes())
PRICES = PriceSheet.model_validate_json(Path('configs/prices/aws-us-east-1-2026-10-16.json').read_bytes())
CONTEXT = CallContext(tenant_id='demo', run_id='gateway-test', request_id='r1', stage='SMOKE')


class FaultProvider(FixtureProvider):
    def __init__(self, *, response=None, count=20, count_error=False, invoke_error=False):
        self.response, self.count = response, count
        self.count_error, self.invoke_error = count_error, invoke_error
        self.calls = 0

    def count_input(self, route, body):
        if self.count_error:
            raise GatewayError('Count unavailable')
        return self.count

    def invoke(self, route, body):
        self.calls += 1
        if self.invoke_error:
            raise TimeoutError('secret provider payload must not be surfaced')
        return self.response if self.response is not None else {
            'usage': {'input_tokens': 20, 'output_tokens': 3},
            'content': [{'type': 'text', 'text': 'fixture text'}], 'stop_reason': 'end_turn',
        }


@pytest.fixture
def ledger():
    value = Ledger(MemoryLedgerStore(), clock=lambda: NOW)
    value.initialize_month('kg-r72', '2026-10')
    value.initialize_run('kg-r72', 'demo', 'gateway-test', cap_micro_usd=100_000)
    return value


def gateway(ledger, provider=None):
    return ModelGateway(ledger, provider or FaultProvider(), ROUTES,
                        PRICES.model_copy(update={'purpose': 'FIXTURE'}))


def generate(g):
    return g.generate(CONTEXT, system='Return a tiny fixture.', prompt='Hello', max_output_tokens=16)


def month(ledger):
    return ledger.store.get(ledger.month_key('kg-r72', '2026-10'))


def test_generation_settles_and_replay_never_dispatches_again(ledger):
    provider = FaultProvider()
    g = gateway(ledger, provider)
    result = generate(g)
    assert result.text == 'fixture text'
    assert result.actual_micro_usd == 39  # ceil(20*1.1 + 3*5.5)
    assert ledger.operation(result.operation_id).status == 'SETTLED'
    assert (month(ledger)['reserved_micro_usd'], month(ledger)['settled_micro_usd']) == (0, 39)
    with pytest.raises(AlreadyDispatched):
        generate(g)
    assert provider.calls == 1


@pytest.mark.parametrize('count', [0, 6145])
def test_input_count_outside_limit_does_not_reserve_or_invoke(ledger, count):
    provider = FaultProvider(count=count)
    with pytest.raises(GatewayError):
        generate(gateway(ledger, provider))
    assert provider.calls == 0 and month(ledger)['reserved_micro_usd'] == 0


def test_unsupported_count_fails_closed(ledger):
    provider = FaultProvider(count_error=True)
    with pytest.raises(GatewayError):
        generate(gateway(ledger, provider))
    assert provider.calls == 0 and month(ledger)['reserved_micro_usd'] == 0


def test_no_allowance_means_no_inference(ledger):
    # A new run with an allowance below one request's bound.
    ledger.initialize_run('kg-r72', 'demo', 'small', cap_micro_usd=1)
    provider = FaultProvider()
    g = gateway(ledger, provider)
    with pytest.raises(BudgetDenied):
        g.generate(CONTEXT.model_copy(update={'run_id': 'small'}), system='S', prompt='P',
                   max_output_tokens=16)
    assert provider.calls == 0


@pytest.mark.parametrize('fault', ['timeout', 'usage_missing', 'cache_usage'])
def test_ambiguous_call_keeps_reservation_and_never_retries(ledger, fault):
    provider = FaultProvider(invoke_error=fault == 'timeout')
    if fault == 'usage_missing':
        provider.response = {'content': []}
    if fault == 'cache_usage':
        provider.response = {'usage': {'input_tokens': 20, 'output_tokens': 3,
                                       'cache_read_input_tokens': 5}}
    g = gateway(ledger, provider)
    with pytest.raises(UncertainModelCall) as error:
        generate(g)
    assert 'secret' not in str(error.value)
    operation = ledger.operation(error.value.operation_id)
    assert operation.status == 'UNKNOWN_SPEND'
    assert month(ledger)['reserved_micro_usd'] == operation.spec.max_cost_micro_usd
    with pytest.raises(AlreadyDispatched):
        generate(g)
    assert provider.calls == 1


def test_invalid_output_is_still_settled(ledger):
    provider = FaultProvider(response={'usage': {'input_tokens': 20, 'output_tokens': 3},
                                       'content': [{'type': 'tool_use'}]})
    with pytest.raises(GatewayError, match='already been settled'):
        generate(gateway(ledger, provider))
    assert month(ledger)['settled_micro_usd'] == 39 and month(ledger)['reserved_micro_usd'] == 0


def test_actual_over_token_bound_halts_even_if_under_money_bound(ledger):
    provider = FaultProvider(response={'usage': {'input_tokens': 21, 'output_tokens': 0}})
    with pytest.raises(GatewayError, match='budgets halted'):
        generate(gateway(ledger, provider))
    assert month(ledger)['halted'] and month(ledger)['settled_micro_usd'] == 24
    assert ledger.store.get(ledger.run_key('gateway-test'))['halted']


def test_embedding_reserves_documented_maximum_and_returns_normalized_vector(ledger):
    result = gateway(ledger, FixtureProvider()).embed(
        CONTEXT.model_copy(update={'stage': 'EMBED'}), text='Synthetic text',
    )
    assert len(result.embedding) == 1024 and result.embedding[0] == 1.0
    operation = ledger.operation(result.operation_id)
    assert operation.spec.max_cost_micro_usd == 164
    assert result.actual_micro_usd == 1


def test_bad_vector_rejected_after_accounting(ledger):
    provider = FaultProvider(response={'inputTextTokenCount': 8, 'embedding': [float('nan')] * 1024})
    with pytest.raises(GatewayError, match='already been settled'):
        gateway(ledger, provider).embed(CONTEXT.model_copy(update={'stage': 'EMBED'}), text='Synthetic')
    assert month(ledger)['settled_micro_usd'] == 1


def test_settlement_failure_never_reissues_provider_call(ledger, monkeypatch):
    provider = FaultProvider()
    def broken(*args):
        raise OSError('DB unavailable')
    monkeypatch.setattr(ledger, 'settle', broken)
    with pytest.raises(UncertainModelCall):
        generate(gateway(ledger, provider))
    assert provider.calls == 1 and month(ledger)['reserved_micro_usd'] > 0


def test_fixtures_cannot_be_mistaken_for_live_accounting(ledger):
    with pytest.raises(GatewayError, match='fixture prices'):
        ModelGateway(ledger, FixtureProvider(), ROUTES, PRICES)


def bedrock_provider():
    session = boto3.Session(aws_access_key_id='testing', aws_secret_access_key='testing',
                            region_name='us-east-1')
    return BedrockProvider(session)


def test_live_provider_disallows_memory_or_local_endpoint_ledger(ledger):
    provider = bedrock_provider()
    assert provider.client.meta.config.retries['total_max_attempts'] == 1
    with pytest.raises(GatewayError, match='disabled'):
        ModelGateway(ledger, provider, ROUTES, PRICES)
    enabled = ROUTES.model_copy(update={'live_inference_enabled': True})
    with pytest.raises(GatewayError, match='managed shared'):
        ModelGateway(ledger, provider, enabled, PRICES)
    local = boto3.client('dynamodb', endpoint_url='http://127.0.0.1:8000', region_name='us-east-1',
                         aws_access_key_id='testing', aws_secret_access_key='testing')
    with pytest.raises(GatewayError, match='managed shared'):
        ModelGateway(Ledger(DynamoLedgerStore(local, 'kg-r72-query-state')), provider, enabled, PRICES)


def test_provider_uses_identical_body_for_counting_and_invocation():
    provider = bedrock_provider()
    body = b'{"messages":[]}'
    payload = {'usage': {'input_tokens': 3, 'output_tokens': 1}}
    raw = json.dumps(payload).encode()
    with Stubber(provider.client) as stub:
        stub.add_response('count_tokens', {'inputTokens': 3},
                          {'modelId': HAIKU, 'input': {'invokeModel': {'body': body}}})
        stub.add_response('invoke_model', {'body': StreamingBody(io.BytesIO(raw), len(raw)),
                                           'contentType': 'application/json'},
                          {'modelId': HAIKU_ROUTE, 'body': body, 'accept': 'application/json',
                           'contentType': 'application/json', 'performanceConfigLatency': 'standard'})
        assert provider.count_input(HAIKU_ROUTE, body) == 3
        assert provider.invoke(HAIKU_ROUTE, body) == payload
        stub.assert_no_pending_responses()


def test_stale_prices_prevent_counting_and_invocation(ledger):
    provider = FaultProvider(count_error=True)
    g = gateway(ledger, provider)
    g.prices = PRICES.model_copy(update={'purpose': 'FIXTURE', 'expires_at': NOW})
    with pytest.raises(ValueError, match='stale'):
        generate(g)
    assert provider.calls == 0
