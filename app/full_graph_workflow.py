"""Source-only qualification sampling and bounded dispatch for full graph work.

These gates do not certify semantics. Unknown spending stops the run; a parsed
candidate remains quarantined from serving until a separate source review.
"""

from collections import defaultdict

from app.contracts import sha256_bytes
from app.extraction import extraction_request


def qualification_units(units, *, per_group=2):
    if type(per_group) is not int or not 1 <= per_group <= 4:
        raise ValueError("FULL_GRAPH_QUALIFICATION_BOUND")
    grouped = defaultdict(list)
    for unit in units:
        for group in sorted({o["service_group"] for o in unit["occurrences"]}):
            grouped[group].append(unit)
    selected, seen = [], set()
    for turn in range(per_group):
        for group, rows in sorted(grouped.items()):
            candidates = sorted(rows, key=lambda u: (u["text_sha256"], u["unit_id"]))
            # First prefer a document different from the previous round. Never
            # consult benchmark questions, gold evidence, scores, or graph paths.
            used_docs = {o["document_version_id"] for r in selected if r["selected_group"] == group
                         for o in r["occurrences"]}
            candidates.sort(key=lambda u: bool(used_docs.intersection(
                o["document_version_id"] for o in u["occurrences"])))
            choice = next((u for u in candidates if u["unit_id"] not in seen), None)
            if choice is None:
                raise ValueError("FULL_GRAPH_GROUP_INSUFFICIENT_DISTINCT_UNITS")
            seen.add(choice["unit_id"])
            selected.append(dict(choice, selected_group=group, selection_round=turn))
    return tuple(selected)


class ScopedExtractionGateway:
    def __init__(self, gateway, contexts_and_texts, recipe, *, observed_stop_micro_usd=100000):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {}
        for context, text in contexts_and_texts:
            system, prompt = extraction_request(text, recipe)
            self.allowed[context.model_dump_json()] = (sha256_bytes(system.encode()), sha256_bytes(prompt.encode()))
        self.seen, self.halted, self.cost = set(), False, 0
        self.stop = observed_stop_micro_usd

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or self.cost >= self.stop or key not in self.allowed or key in self.seen
                or max_output_tokens != 2048 or self.allowed[key] !=
                (sha256_bytes(system.encode()), sha256_bytes(prompt.encode()))):
            raise ValueError("FULL_GRAPH_DISPATCH_SCOPE_OR_COST_BOUND")
        self.seen.add(key)
        try:
            result = self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
        except Exception:
            self.halted = True
            raise
        self.cost += result.actual_micro_usd
        return result
