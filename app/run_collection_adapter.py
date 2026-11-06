"""Zero-paid-call adapter from owned child execution to the existing collection runner."""

from app.collection_answering import (
    validate_collection_lexical_binding,
    validate_collection_reranker,
)
from app.contracts import stable_id
from app.owned_runs import OwnedRunReleaseV1
from app.rag_methods import MethodConfigV1, RerankedMethodConfigV1, RunInputV1
from app.rag_runner import _validate_encoder, run_methods


class CollectionRetrievalExecutor:
    def __init__(self, collection, lexical, **runtime):
        if runtime.get("gateway") is not None:
            raise ValueError("OWNED_RETRIEVAL_DISALLOWS_GATEWAY")
        self.collection, self.lexical, self.runtime = collection, lexical, runtime
        validate_collection_lexical_binding(collection, lexical)
        encoder, recipe = runtime.get("encoder"), runtime["recipe"]
        query_recipe = _validate_encoder(collection, encoder) if encoder else None
        configs = {}
        for method in ("bm25", "vector", "hybrid", "hybrid_reranked"):
            if method != "bm25" and query_recipe is None:
                continue
            extra, config_class = {}, MethodConfigV1
            if method == "hybrid_reranked":
                reranker = runtime.get("reranker")
                if reranker is None:
                    continue
                validate_collection_reranker(collection, reranker)
                config_class = RerankedMethodConfigV1
                extra = {"reranker_recipe": reranker.recipe, "reranker_recipe_sha256": reranker.recipe.fingerprint}
            configs[method] = config_class(method_id=method, fusion="RRF60" if method in {"hybrid", "hybrid_reranked"} else "NONE",
                collection_manifest_sha256=collection.manifest_sha256,
                lexical_index_version=lexical.manifest.index_version,
                document_recipe_sha256=collection.manifest.embedding_recipe.fingerprint,
                query_recipe_sha256=query_recipe.fingerprint if method != "bm25" else None,
                generation_recipe_sha256=recipe.fingerprint, **extra).fingerprint
        self.release = OwnedRunReleaseV1(collection_version=collection.manifest.collection_version,
            collection_manifest_sha256=collection.manifest_sha256, method_config_sha256=configs)

    def check_release(self, release):
        pin = OwnedRunReleaseV1.model_validate(release)
        if (pin.collection_version != self.release.collection_version
            or pin.collection_manifest_sha256 != self.release.collection_manifest_sha256
            or any(self.release.method_config_sha256.get(k) != v for k, v in pin.method_config_sha256.items())):
            raise ValueError("OWNED_COLLECTION_RELEASE_CHANGED")

    def __call__(self, auth, parent, method, guard):
        guard()
        self.check_release(parent["release"])
        if auth.tenant_id != self.collection.manifest.tenant_id or parent["request"]["run_kind"] != "retrieval":
            raise ValueError("OWNED_COLLECTION_SCOPE_MISMATCH")
        payload = parent["request"]
        request = RunInputV1(original_question=payload["original_question"], language=payload["language"],
            source_scope="local_collection", method_ids=(method,), run_kind="retrieval",
            idempotency_key=stable_id("job", run=parent["run_id"], method=method))
        report = run_methods(self.collection, self.lexical, request, tenant_id=auth.tenant_id,
            owner_id=auth.user_id, **self.runtime)
        guard()
        if report["status"] != "COMPLETED" or report["new_known_cost_micro_usd"] != 0 or len(report["methods"]) != 1:
            raise ValueError("OWNED_RETRIEVAL_DID_NOT_COMPLETE")
        return report["methods"][0]
