from types import SimpleNamespace

import pytest

from scripts.audit_semantic_empty_outputs import ROOT, audit
from scripts.run_typed_graph_qualification import ScopedGateway, context_for, source_selection


def unit(index, text, *, group="group", document="doc"):
    return {"unit_id": "legacy" + index, "v7_unit_id": "v7" + index, "text_sha256": index,
            "occurrences": [{"chunk_id": index, "service_group": group, "document_version_id": document}]}


def test_source_selector_prefers_cues_then_control_without_benchmark_fields():
    rich = "Named service supports named protocol. " * 10
    texts = {"a": SimpleNamespace(text="title"), "b": SimpleNamespace(text=rich), "c": SimpleNamespace(text="control")}
    units = [unit("a", "title"), unit("b", rich), unit("c", "control", document="doc-two")]
    selected = source_selection(units, texts, set())
    assert [r["text_sha256"] for r in selected] == ["b", "c"]
    assert source_selection(tuple(reversed(units)), texts, set()) == selected


def test_source_selector_excludes_previous_source_claims_and_rejects_insufficient_group():
    chunks = {"a": SimpleNamespace(text="a"), "b": SimpleNamespace(text="b")}
    with pytest.raises(ValueError, match="DISTINCT_SOURCE"):
        source_selection([unit("a", "a"), unit("b", "b")], chunks, {"a"})


def test_native_scope_claim_prevents_duplicate_dispatch_and_halts_uncertainty():
    row = unit("a", "source")
    context = context_for(row)
    provider = SimpleNamespace(live=False)
    class Gateway:
        routes = object()
        calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("UNKNOWN_FIXTURE")
    base = Gateway()
    base.provider = provider
    scoped = ScopedGateway(base, {"selection": [row]}, {"a": SimpleNamespace(text="source")})
    system, prompt = scoped.allowed[context.model_dump_json()]
    with pytest.raises(ValueError, match="DISPATCH_SCOPE"):
        scoped.generate(context, system=system, prompt="different-source", max_output_tokens=2048)
    assert base.calls == 0
    with pytest.raises(RuntimeError):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
    with pytest.raises(ValueError, match="DISPATCH_SCOPE"):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
    assert base.calls == 1


def test_empty_source_review_has_bound_witness_for_real_v5_miss():
    if not (ROOT / "results/semantic-graph-v7/development-source-audit-v1/sources.jsonl").is_file():
        pytest.skip("Requires the local source export for the recorded semantic-empty-output audit")
    report = audit()
    assert len(report["reviews"]) == 20 and report["categories"]["MISSED_ELIGIBLE_RELATION"] == 1
    missed = next(r for r in report["reviews"] if r["status"] == "MISSED_ELIGIBLE_RELATION")
    assert missed["local_guard_witness_v5"] == missed["local_guard_witness_v7"] == "PASS"
    assert report["human_reviewed"] == 0 and report["recall_estimate"] == "NOT_ESTIMATED"
