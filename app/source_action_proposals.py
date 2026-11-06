"""Dependency-derived proposals from explicit source clauses, never serving facts.

Only controlled-catalog actors and seven action families; no coreference, no
inferred API actions, no use of a heading as actor, and no LLM/network calls.
"""

from app.contracts import canonical_json
from app.extraction import ExtractionError
from app.semantic_extraction_v7 import catalog_entries
from app.semantic_extraction_v10 import SourceActionV1
from app.source_syntax import ACTION_LEMMAS, validate_syntax_record


def nominal(parsed, root):
    if root["pos"] not in {"NOUN", "PROPN", "PRON"}:
        raise ExtractionError("PROPOSAL_NOMINAL_PARSE_REQUIRED")
    included, pending = {root["i"]}, [root]
    while pending:
        for child in parsed.children(pending.pop(), "compound", "amod", "det", "nmod", "poss", "predet", "nummod", "quantmod"):
            if child["i"] not in included:
                included.add(child["i"])
                pending.append(child)
    tokens = [parsed.tokens[i] for i in sorted(included)]
    start, end = tokens[0]["start"], tokens[-1]["end"]
    # A relative resource restriction must remain part of the proposed name.
    relatives = parsed.children(root, "relcl", "acl")
    if relatives:
        end = max(end, max(t["end"] for child in relatives for t in parsed.descendants(child)))
    return start, end, parsed.text[start:end]


def endpoint(parsed, root, *, require_service=False):
    start, end, name = nominal(parsed, root)
    entries = catalog_entries()
    # Determiners can be outside the product's registered alias; source actor
    # must still resolve to the complete nominal head, not a compound substring.
    known = entries.get(name.casefold())
    if require_service and (not known or known["semantic_type"] != "SERVICE_PRODUCT"):
        raise ExtractionError("PROPOSAL_CATALOG_SERVICE_ACTOR_REQUIRED")
    if known:
        semantic_type = known["semantic_type"]
    else:
        semantic_type = "RESOURCE_TYPE" if root["lemma"] in {"bucket", "queue", "topic", "function", "role", "key", "instance", "cluster", "table", "stream"} else "CONCEPT"
    positions, cursor = [], -1
    while (cursor := parsed.text.find(name, cursor + 1)) >= 0:
        positions.append(cursor)
    return {"name": name, "semantic_type": semantic_type, "occurrence": positions.index(start) + 1,
        "head_quote": name if known else root["text"], "restrictions": [], "scope_qualifiers": []}


def propose_actions(parsed, source_text):
    proposals, rejected = [], []
    for verb in parsed.tokens:
        families = [family for family, lemmas in ACTION_LEMMAS.items() if verb["pos"] == "VERB" and verb["lemma"] in lemmas]
        if not families:
            continue
        family = families[0]
        try:
            construction, enabler = "FINITE_ACTIVE", None
            parent = parsed.tokens[verb["head"]]
            if verb["dep"] == "xcomp" and parent["lemma"] == "enable":
                subjects, actors = parsed.children(parent, "nsubj"), parsed.children(parent, "dobj")
                if len(subjects) != 1 or len(actors) != 1:
                    raise ExtractionError("PROPOSAL_ENABLED_ROLES_AMBIGUOUS")
                actor, chain = actors[0], []
                enabler = endpoint(parsed, subjects[0])
                construction = "ENABLED_ACTION"
            else:
                actor, chain = parsed.subject(verb)
            subject = endpoint(parsed, actor, require_service=True)
            objects = parsed.children(verb, "dobj")
            if len(objects) != 1:
                raise ExtractionError("PROPOSAL_DIRECT_OBJECT_REQUIRED")
            direct = objects[0]
            role, payload = {"INVOKE": "CALLEE", "DELIVER": "DESTINATION", "ASSUME_ROLE": "ROLE"}.get(family, "DATA_RESOURCE"), None
            words = {"from"} if family == "READ" else {"with", "using"} if family in {"ENCRYPT", "DECRYPT"} else {"to", "into"}
            targets = [p for prep in parsed.children(verb, "prep") if prep["text"].casefold() in words for p in parsed.children(prep, "pobj")]
            if targets:
                if len(targets) != 1:
                    raise ExtractionError("PROPOSAL_DESTINATION_AMBIGUOUS")
                target = targets[0]
                payload = nominal(parsed, direct)[2]
                if family in {"ENCRYPT", "DECRYPT"}:
                    role = "KEY"
            else:
                target = direct
            governing = parsed.governing(verb, chain)
            lemmas = {t["lemma"] for t in governing}
            modality = "CAPABILITY" if construction == "ENABLED_ACTION" or lemmas & {"can", "may", "might", "could"} else "REQUIRED" if "must" in lemmas else "RECOMMENDED" if "should" in lemmas else "ASSERTED"
            record = SourceActionV1.model_validate_json(canonical_json({"clause_quote": parsed.text, "actor": subject,
                "action_family": family, "action_quote": verb["text"], "target": endpoint(parsed, target), "target_role": role,
                "payload_quote": payload, "construction": construction, "enabler": enabler,
                "polarity": "NEGATIVE" if any(t["dep"] == "neg" or t["text"].casefold() == "never" for t in governing) else "POSITIVE", "modality": modality}))
            candidate = validate_syntax_record(record, source_text, parsed, kind="ACTION")
            proposals.append(candidate | {"origin": "DEPENDENCY_SOURCE_PROPOSAL_NOT_NATIVE_LLM_EXTRACTION"})
        except ExtractionError as exc:
            rejected.append({"verb_token": verb["i"], "lemma": verb["lemma"], "reason": str(exc)})
    return proposals, rejected
