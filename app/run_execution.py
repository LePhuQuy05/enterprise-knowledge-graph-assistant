"""At-least-once publication and one fenced child execution; answers remain gated."""

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.ledger import ConcurrentWrite
from app.owned_runs import RunMessageV1, transition
from app.query_auth import ApiError, AuthContext
from app.rag_methods import MethodResultV1


def validate_result(parent, method, raw):
    result = MethodResultV1.model_validate_json(canonical_json(raw))
    release = parent["release"]
    answering = parent["request"]["run_kind"] == "answer"
    if result.status != ("COMPLETED" if answering else "RETRIEVED") or (
        (result.response is None) == answering or result.effective_method != method
    ) or (
        result.config_sha256 != release["method_config_sha256"][method]
        or result.collection_version != release["collection_version"]
        or result.config.collection_manifest_sha256 != release["collection_manifest_sha256"]
        or (not answering and result.new_cost_micro_usd != 0) or result.planning_trace is not None
        or tuple(source.get("chunk_id") for source in result.context_sources) != result.context_chunk_ids
        or not set(result.context_chunk_ids) <= set(result.ranked_chunk_ids)
        or len(result.ranked_chunk_ids) > 40
        or len(result.ranked_chunk_ids) != len(set(result.ranked_chunk_ids))
        or len(result.context_chunk_ids) != len(set(result.context_chunk_ids))
        or len(result.trace) > 12
    ):
        raise ValueError("RUN_RESULT_BINDING_OR_RETRIEVAL_GATE")
    source_fields = {"source_id", "chunk_id", "document_version_id", "shard_id", "shard_manifest_sha256",
        "source_uri", "title", "section_path", "start_char", "end_char", "text_sha256", "text"}
    for index, source in enumerate(result.context_sources, 1):
        if set(source) != source_fields or source["source_id"] != f"s{index}" or (
            type(source["text"]) is not str or sha256_bytes(source["text"].encode("utf-8")) != source["text_sha256"]
            or type(source["start_char"]) is not int or type(source["end_char"]) is not int
            or not 0 <= source["start_char"] < source["end_char"]
            or source["end_char"] - source["start_char"] != len(source["text"])
        ):
            raise ValueError("RUN_SOURCE_PROVENANCE_MISMATCH")
    if answering:
        response = result.response
        expected_id = stable_id("astage", run=parent["run_id"], owner=parent["owner_id"], slot="GENERATE:" + method)
        if response.request_id != expected_id or not set(c.chunk_id for c in response.citations) <= set(result.context_chunk_ids):
            raise ValueError("OWNED_ANSWER_CITATION_BINDING")
        sources = {source["chunk_id"]: source for source in result.context_sources}
        for citation in response.citations:
            source = sources[citation.chunk_id]
            if (source["text"].count(citation.quote) != 1 or source["document_version_id"] != citation.document_version_id
                or source["source_uri"] != citation.source_uri or source["title"] != citation.title):
                raise ValueError("OWNED_ANSWER_CITATION_SOURCE_MISMATCH")
    return result


class RunOutboxDispatcher:
    def __init__(self, repository, send):
        self.repo, self.send = repository, send

    def publish(self, auth, run_id):
        """Called from a durable outbox event/reconciler; API submit never sends first."""
        sent = 0
        for row in self.repo.outboxes(auth, run_id):
            if row["status"] != "PENDING":
                continue
            message = RunMessageV1.model_validate(row["message"])
            # A crash/timeout after send leaves PENDING; duplicates are safe at the consumer.
            self.send(message.model_dump(mode="json"))
            try:
                self.repo.store.commit([transition(row, status="SENT")])
            except ConcurrentWrite:
                current = self.repo.store.get((row["PK"], row["SK"]))
                if current is None or current["status"] != "SENT":
                    raise
            sent += 1
        return sent


class SqsRunSender:
    def __init__(self, client, queue_url):
        self.client, self.queue_url = client, queue_url

    def __call__(self, message):
        return self.client.send_message(QueueUrl=self.queue_url,
            MessageBody=canonical_json(RunMessageV1.model_validate(message).model_dump(mode="json")).decode("utf-8"))


class OwnedRunWorker:
    def __init__(self, repository, artifacts, executor, *, authorize, release_check):
        self.repo, self.artifacts, self.executor = repository, artifacts, executor
        self.authorize, self.release_check = authorize, release_check

    def process(self, raw_message):
        message = RunMessageV1.model_validate_json(canonical_json(raw_message))
        auth = AuthContext(message.tenant_id, message.owner_id)
        parent = self.repo.get(auth, message.run_id)
        # Bind before authorization/cancellation so a forged nonce cannot change state.
        if parent["admission_token"] != message.admission_token:
            raise ApiError(404, "RUN_NOT_FOUND")
        try:
            self.authorize(auth)
            self.release_check(parent["release"])
        except ApiError as exc:
            if exc.status not in {401, 403}:
                raise
            child = self.repo.child(auth, message.run_id, message.method_id)
            if child["status"] in {"QUEUED", "CLAIMED"}:
                self.repo.finish(auth, message.run_id, message.method_id, "CANCELLED", error="ACCESS_DENIED")
            return "ACK"
        if parent["request"]["run_kind"] == "answer":
            from app.run_budget import RunStageBusy

            if self.repo.child(auth, message.run_id, message.method_id)["status"] in {
                "COMPLETED", "FAILED", "UNKNOWN", "CANCELLED", "EXPIRED"
            }:
                self.repo.cleanup_budget(auth, message.run_id, message.method_id)
                return "ACK"
            try:
                self.executor.prepare(auth, parent)
            except RunStageBusy:
                return "RETRY"
            except Exception:
                safe = self.executor.safe_failure(auth, parent, message.method_id)
                self.repo.finish(auth, message.run_id, message.method_id, "FAILED" if safe else "UNKNOWN",
                    error="RUN_PREPARATION_FAILED" if safe else "RUN_RECONCILIATION_REQUIRED")
                return "ACK"
        outcome, token = self.repo.claim(message)
        if outcome != "CLAIMED":
            return outcome
        started = False
        try:
            self.authorize(auth)
            self.release_check(parent["release"])
            self.repo.start(auth, message.run_id, message.method_id, token)
            started = True
            def guard():
                self.authorize(auth)
                self.release_check(parent["release"])
                self.repo.guard(auth, message.run_id, message.method_id, token)
            guard()
            result = validate_result(parent, message.method_id,
                self.executor(auth, parent, message.method_id, guard))
            # Cancellation does not discard a result already produced; no more compute stages start.
            self.authorize(auth)
            self.release_check(parent["release"])
            self.repo.guard(auth, message.run_id, message.method_id, token, allow_cancelled=True)
            reference = self.artifacts.save(auth, message.run_id, message.method_id, result.model_dump(mode="json"))
            self.authorize(auth)
            self.repo.finish(auth, message.run_id, message.method_id, "COMPLETED", token=token, result_ref=reference)
            return "ACK"
        except Exception:
            # Persisted EXECUTING is never taken over/retried through the adapter.
            # The operation ledger remains responsible for any future paid stage obligations.
            if started:
                safe = parent["request"]["run_kind"] == "answer" and self.executor.safe_failure(auth, parent, message.method_id)
                self.repo.finish(auth, message.run_id, message.method_id, "FAILED" if safe else "UNKNOWN", token=token,
                    error="GENERATION_INVALID" if safe else "RUN_RECONCILIATION_REQUIRED")
                return "ACK"
            return "RETRY"

    def sqs_batch(self, event):
        """Partial batch response; safe errors never include payload/provider exception text."""
        from app.external_http import strict_json

        failed = []
        for record in event.get("Records", []):
            try:
                body = record["body"]
                if type(body) is not str or len(body.encode("utf-8")) > 4096:
                    raise ValueError("RUN_MESSAGE_SIZE_LIMIT")
                if record.get("eventSource") != "aws:sqs" or self.process(strict_json(body.encode("utf-8"))) != "ACK":
                    failed.append({"itemIdentifier": record["messageId"]})
            except Exception:
                failed.append({"itemIdentifier": record["messageId"]})
        return {"batchItemFailures": failed}
