import copy
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes, text_hash
from app.corpus import LocalCorpus
from app.external_gateway import (
    ExternalCallUncertain,
    ExternalModelGateway,
    ExternalPriceSheet,
    ExternalRoutes,
)
from app.external_http import ExternalHttpError
from app.extraction import AliasDictionary
from app.ledger import Ledger, MemoryLedgerStore
from app.pilot import pilot_plan, revised_quote_plan, run_selected, select_chunks
from scripts import check_external_smoke

ROOT = Path(__file__).resolve().parents[1]


def fixture_corpus():
    documents, chunks, sources = [], [], []
    text = "Synthetic Alpha uses Synthetic Beta. " * 8
    for service, count in (("Lambda", 3), ("S3", 3), ("IAM", 2), ("KMS", 2)):
        for index in range(count):
            uri = f"https://example.invalid/{service}/{index}"
            doc = DocumentVersion.create(tenant_id="demo", source_uri=uri, title=service + str(index),
                fetched_at=datetime(2026, 10, 17, tzinfo=UTC), raw_sha256=text_hash(text),
                normalizer_version="synthetic-v1", normalized_text=text,
                normalized_text_path="synthetic.txt", source_kind="TEXT")
            chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id,
                chunking_version="synthetic-v1", section_path=("Synthetic",), position=0,
                start_char=0, end_char=len(text), text=text, token_count=100, tokenizer_version="synthetic-v1")
            documents.append(doc)
            chunks.append(chunk)
            sources.append({"service": service, "url": uri})
    return LocalCorpus("synthetic-corpus", "demo", tuple(documents), tuple(chunks), {}), {"sources": sources}


class Stub:
    live = False

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def request(self, provider, method, path, body):
        self.calls += 1
        if self.fail:
            raise TimeoutError("DO_NOT_EXPOSE_FIXTURE_PAYLOAD")
        if provider == "VOYAGE":
            return {"model": "voyage-4-lite", "usage": {"total_tokens": 10},
                    "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1023}]}
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"entities":[],"relationships":[]}'}}]}


def configured(fail=False):
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: prices.checked_at + timedelta(seconds=1))
    ledger.initialize_month("kg-r72", prices.checked_at.strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", "demo", "pilot-test", cap_micro_usd=500_000)
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2_000_000)
    return ExternalModelGateway(ledger, Stub(fail), ExternalRoutes(account_model_metadata_sha256="0" * 64),
                                prices.model_copy(update={"purpose": "FIXTURE"}))


def test_pilot_selects_complete_chunks_round_robin_not_slices():
    corpus, manifest = fixture_corpus()
    selection = select_chunks(corpus, manifest)
    assert [r["service"] for r in selection] == ["Lambda", "S3", "IAM", "KMS"] * 2 + ["Lambda", "S3"]
    assert len({r["document_version_id"] for r in selection}) == 10
    assert all(r["characters"] == len(corpus.chunks[0].text) for r in selection)
    assert select_chunks(corpus, manifest) == selection


def test_insufficient_sources_or_oversized_limit_fail_without_model_calls():
    corpus, manifest = fixture_corpus()
    with pytest.raises(ValueError, match="NOT_ENOUGH"):
        select_chunks(corpus, {"sources": manifest["sources"][:1]})
    with pytest.raises(ValueError, match="LIMIT"):
        select_chunks(corpus, manifest, limit=11)


def test_cache_backed_execution_and_offline_replay_keep_artifacts_and_zero_new_cost(tmp_path):
    corpus, manifest = fixture_corpus()
    gateway = configured()
    aliases = AliasDictionary(version="synthetic-empty-v1", entries=())
    plan = pilot_plan(corpus, select_chunks(corpus, manifest), aliases, gateway.routes)
    progress = []
    kwargs = dict(aliases=aliases, gateway=gateway, run_id="pilot-test", cache_directory=tmp_path / "cache",
                  artifact_directory=tmp_path / "artifacts", checkpoint=lambda r: progress.append(copy.deepcopy(r)))
    first = run_selected(corpus, plan, **kwargs)
    assert first["status"] == "STRUCTURAL_PASS_SEMANTIC_PENDING" and first["semantic_review"] == "PENDING"
    assert len(first["embeddings"]) == len(first["extractions"]) == 10
    # Ten distinct title-prefixed inputs, one identical raw text reused across ten sources.
    assert gateway.provider.calls == 11
    assert sum(r["cache_hit"] for r in first["extractions"]) == 9
    second = run_selected(corpus, plan, **kwargs)
    assert second["new_budget_cost_micro_usd"] == 0 and gateway.provider.calls == 11
    assert first["artifacts"] == second["artifacts"]
    assert len(progress) > 20


@pytest.mark.parametrize("rehash", [False, True])
def test_tampered_plan_or_source_binding_rejected_before_dispatch(tmp_path, rehash):
    corpus, manifest = fixture_corpus()
    gateway = configured()
    aliases = AliasDictionary(version="synthetic-v1", entries=())
    plan = pilot_plan(corpus, select_chunks(corpus, manifest), aliases, gateway.routes)
    plan["selection"][0]["text_sha256"] = "0" * 64
    if rehash:
        plan["plan_sha256"] = sha256_bytes(canonical_json({k: v for k, v in plan.items() if k != "plan_sha256"}))
    with pytest.raises(ValueError, match="CHANGED"):
        run_selected(corpus, plan, aliases=aliases, gateway=gateway, run_id="pilot-test",
                     cache_directory=tmp_path / "cache", artifact_directory=tmp_path / "artifacts", checkpoint=lambda r: None)
    assert gateway.provider.calls == 0


def test_ambiguous_first_call_stops_rest_of_pilot_and_retains_reservation(tmp_path):
    corpus, manifest = fixture_corpus()
    gateway = configured(fail=True)
    aliases = AliasDictionary(version="synthetic-v1", entries=())
    plan = pilot_plan(corpus, select_chunks(corpus, manifest), aliases, gateway.routes)
    with pytest.raises(ExternalCallUncertain):
        run_selected(corpus, plan, aliases=aliases, gateway=gateway, run_id="pilot-test",
                     cache_directory=tmp_path / "cache", artifact_directory=tmp_path / "artifacts", checkpoint=lambda r: None)
    assert gateway.provider.calls == 1
    assert gateway.ledger.store.get(gateway.ledger.external_funding_key("kg-r72"))["reserved_micro_usd"] == 640


def test_smoke_rerun_keeps_pass_receipt_and_creates_no_aws_session(tmp_path, monkeypatch, capsys):
    path = tmp_path / "results/external_model_smoke.json"
    path.parent.mkdir()
    original = json.dumps({"status": "PASS", "run_id": "existing-smoke", "steps": ["retained"]}).encode()
    path.write_bytes(original)
    monkeypatch.setattr(check_external_smoke, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["check_external_smoke", "--execute"])
    def forbidden(*args, **kwargs):
        raise AssertionError("No AWS session permitted on completed smoke replay")
    monkeypatch.setattr(check_external_smoke.boto3, "Session", forbidden)
    assert check_external_smoke.main() == 0
    assert path.read_bytes() == original
    assert json.loads(capsys.readouterr().out)["new_calls"] == 0


def test_independent_extraction_keeps_unknown_embedding_and_same_budget(tmp_path):
    corpus, manifest = fixture_corpus()
    gateway = configured(fail=True)
    aliases = AliasDictionary(version="synthetic-v1", entries=())
    plan = pilot_plan(corpus, select_chunks(corpus, manifest), aliases, gateway.routes)
    kwargs = dict(aliases=aliases, gateway=gateway, run_id="pilot-test", cache_directory=tmp_path / "cache",
                  artifact_directory=tmp_path / "artifacts", checkpoint=lambda r: None)
    with pytest.raises(ExternalCallUncertain):
        run_selected(corpus, plan, **kwargs)
    gateway.provider.fail = False
    result = run_selected(corpus, plan, stages=("EXTRACT",), **kwargs)
    assert result["status"] == "EXTRACTION_STRUCTURAL_PASS_EMBEDDINGS_PENDING"
    assert len(result["extractions"]) == 10 and result["embeddings"] == []
    assert gateway.provider.calls == 2  # One failed embedding, one unique extraction; never retry Voyage.
    assert gateway.ledger.store.get(gateway.ledger.external_funding_key("kg-r72"))["reserved_micro_usd"] == 640
    assert gateway.ledger.store.get(gateway.ledger.run_key("pilot-test"))["cap_micro_usd"] == 500_000


def test_provider_failure_category_survives_cache_tombstone_without_raw_error(tmp_path):
    corpus, manifest = fixture_corpus()
    gateway = configured()
    aliases = AliasDictionary(version="synthetic-v1", entries=())
    plan = pilot_plan(corpus, select_chunks(corpus, manifest), aliases, gateway.routes)

    def fail(*args, **kwargs):
        raise ExternalHttpError("RATE_LIMIT")

    gateway.provider.request = fail
    with pytest.raises(ExternalCallUncertain):
        run_selected(corpus, plan, aliases=aliases, gateway=gateway, run_id="pilot-test",
                     cache_directory=tmp_path / "cache", artifact_directory=tmp_path / "artifacts", checkpoint=lambda r: None)
    with sqlite3.connect(tmp_path / "cache/embeddings.sqlite3") as db:
        stored = db.execute("SELECT state,error_category FROM embeddings").fetchone()
    assert stored == ("FAILED", "ExternalCallUncertain:RATE_LIMIT")


def test_revised_plan_requires_different_unknown_input_and_preserves_allowance():
    corpus, manifest = fixture_corpus()
    gateway = configured()
    aliases = AliasDictionary(version="synthetic-v1", entries=())
    prior = pilot_plan(corpus, select_chunks(corpus, manifest), aliases, gateway.routes)
    with pytest.raises(ValueError, match="NO_ALTERNATIVE"):
        revised_quote_plan(corpus, prior, prior["selection"][3]["chunk_id"], aliases, gateway.routes)
    extra = tuple(Chunk.create(**{k: v for k, v in c.model_dump().items() if k != "chunk_id"}
                    | {"text": c.text[:-35], "end_char": c.end_char - 35, "position": 1}) for c in corpus.chunks)
    revised_corpus = LocalCorpus(corpus.version, corpus.tenant_id, corpus.documents, corpus.chunks + extra, corpus.recipe)
    plan = revised_quote_plan(revised_corpus, prior, prior["selection"][3]["chunk_id"], aliases, gateway.routes)
    assert plan["run_cap_micro_usd"] == prior["run_cap_micro_usd"] == 500_000
    assert plan["selection"][3]["chunk_id"] != prior["selection"][3]["chunk_id"]
    assert plan["request_id_prefix"] == "v2-" and plan["plan_sha256"] != prior["plan_sha256"]
    assert plan["extraction_recipe"]["version"] == "deepseek-quotes-v2"
    assert prior["extraction_recipe"]["version"] == "deepseek-grounded-v1"
