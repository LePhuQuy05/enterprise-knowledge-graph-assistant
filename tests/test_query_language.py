import json
from types import SimpleNamespace

import pytest

from app.cost import TokenUsage
from app.query_language import RewriteError, rewrite_for_search


@pytest.mark.parametrize("query,valid", [
    ("Cross-account s3:GetObject AccessDenied with kms:Decrypt", True),
    ("Cross-account s3:PutObject AccessDenied with kms:Decrypt", False),
    ("Cross-account s3:GetObject with kms:Decrypt", False),
])
def test_rewrite_identifiers_are_preserved_and_receipt_precedes_validation(query, valid):
    saved = []
    gateway = SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(
        text=json.dumps({"search_query": query}), stop_reason="end_turn", operation_id="synthetic-paid-rewrite",
        usage=TokenUsage(input_tokens=50, output_tokens=20), actual_micro_usd=39))
    args = dict(gateway=gateway, tenant_id="demo", request_id="rewrite-control", run_id="test", receipt_sink=saved.append)
    question = "Do not change s3:GetObject: AccessDenied cross-account kms:Decrypt"
    if valid:
        assert rewrite_for_search(question, **args)[0] == query
    else:
        with pytest.raises(RewriteError):
            rewrite_for_search(question, **args)
    assert len(saved) == 1 and saved[0]["cost_micro_usd"] == 39
    assert saved[0]["operation_id"] == "synthetic-paid-rewrite"
