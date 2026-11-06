"""Auditable mechanical normalization; never alter an actor, name or predicate."""

from copy import deepcopy

from app.contracts import canonical_json, sha256_bytes
from app.extraction import ExtractionError
from app.semantic_extraction_v7 import complete_evidence, minimal_trigger


def normalize_record(raw, text, *, kind):
    record, edits = deepcopy(raw), []
    clause = record["clause_quote"]
    complete_evidence(clause, text)
    fields = {"RELATION": ("subject", "object"), "PERMISSION": ("principal", "resource"),
        "ACTION": ("actor", "target", "enabler")}[kind]
    for field in fields:
        endpoint = record.get(field)
        if not endpoint:
            continue
        name = endpoint["name"]
        if clause.count(name) == 1 and endpoint["occurrence"] != 1:
            old = endpoint["occurrence"]
            endpoint["occurrence"] = 1
            edits.append({"path": field + ".occurrence", "before": old, "after": 1,
                "reason": "EXACT_NAME_UNIQUE_IN_EXACT_CLAUSE", "witness": name})
    if kind == "PERMISSION" and record["trigger_quote"].endswith(" to"):
        before, after = record["trigger_quote"], record["trigger_quote"][:-3]
        try:
            minimal_trigger(after, "PERMISSION")
        except ExtractionError:
            pass
        else:
            if clause.count(before) == 1 and clause.count(after) == 1:
                record["trigger_quote"] = after
                edits.append({"path": "trigger_quote", "before": before, "after": after,
                    "reason": "EXACT_PERMISSION_TRIGGER_TRAILING_TO", "witness": before})
    return record, {"version": "source-address-normalization-v10", "raw_record": deepcopy(raw),
        "raw_sha256": sha256_bytes(canonical_json(raw)), "normalized_sha256": sha256_bytes(canonical_json(record)),
        "source_sha256": sha256_bytes(text.encode()), "edits": edits}
