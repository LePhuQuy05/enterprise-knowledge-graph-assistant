"""Bound source token dependencies; linguistic evidence is not entailment proof.

No spaCy imports in app runtime. A pinned isolated local worker supplies exact
annotations, and every consumed role retains its token/character witnesses.
"""

import re
from copy import deepcopy
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import ExtractionError
from app.semantic_extraction import endpoint_position
from app.semantic_extraction_v7 import complete_evidence
from app.semantic_extraction_v9 import EndpointV9, identity
from app.source_entity_boundary import check_reference


class SyntaxDocument:
    def __init__(self, data):
        self.text, self.tokens = data["text"], deepcopy(data["tokens"])
        if not 1 <= len(self.text) <= 20000 or not 1 <= len(self.tokens) <= 4000:
            raise ExtractionError("SYNTAX_DOCUMENT_BOUND")
        previous = 0
        for i, token in enumerate(self.tokens):
            if (token["i"] != i or type(token["head"]) is not int or not 0 <= token["head"] < len(self.tokens)
                    or not previous <= token["start"] < token["end"] <= len(self.text)
                    or self.text[token["start"]:token["end"]] != token["text"]
                    or self.text[previous:token["start"]].strip()):
                raise ExtractionError("SYNTAX_TOKEN_SOURCE_OR_IDENTITY")
            previous = token["end"]
        if self.text[previous:].strip():
            raise ExtractionError("SYNTAX_TRAILING_TEXT_NOT_ANNOTATED")
        for token in self.tokens:
            visited, node = set(), token
            while node["head"] != node["i"]:
                if node["i"] in visited:
                    raise ExtractionError("SYNTAX_DEPENDENCY_CYCLE")
                visited.add(node["i"])
                node = self.tokens[node["head"]]

    def children(self, token, *deps):
        return [t for t in self.tokens if t["head"] == token["i"] and t["i"] != token["i"] and (not deps or t["dep"] in deps)]

    def span_tokens(self, start, end):
        selected = [t for t in self.tokens if start <= t["start"] and t["end"] <= end]
        if not selected or selected[0]["start"] != start or selected[-1]["end"] != end:
            raise ExtractionError("SYNTAX_ENDPOINT_TOKEN_BOUNDARY")
        return selected

    def root_in_span(self, start, end):
        tokens = self.span_tokens(start, end)
        ids = {t["i"] for t in tokens}
        roots = [t for t in tokens if t["head"] not in ids or t["head"] == t["i"]]
        if len(roots) != 1:
            raise ExtractionError("SYNTAX_SINGLE_ENDPOINT_ROOT_REQUIRED")
        return roots[0]

    def descendants(self, root):
        found, pending = {root["i"]}, [root]
        while pending:
            for child in self.children(pending.pop()):
                if child["i"] not in found:
                    found.add(child["i"])
                    pending.append(child)
        return [t for t in self.tokens if t["i"] in found]

    def subject(self, verb):
        """Follow only explicit subjects, same-subject coordination or continue."""
        node, chain = verb, []
        for _ in range(5):
            direct = self.children(node, "nsubj")
            if len(direct) == 1:
                return direct[0], chain
            if direct or self.children(node, "nsubjpass"):
                break
            parent = self.tokens[node["head"]]
            if node["dep"] == "conj" and parent["pos"] == "VERB":
                chain.append({"kind": "SAME_SUBJECT_COORDINATION", "from": node["i"], "to": parent["i"]})
            elif node["dep"] == "xcomp" and parent["lemma"] == "continue":
                chain.append({"kind": "CONTINUE_ASPECT", "from": node["i"], "to": parent["i"]})
            else:
                break
            node = parent
        raise ExtractionError("SYNTAX_EXPLICIT_ACTIVE_SUBJECT_REQUIRED")

    def governing(self, verb, chain):
        ids = {verb["i"]} | {link["to"] for link in chain}
        for i in tuple(ids):
            ids.update(t["i"] for t in self.children(self.tokens[i], "aux", "auxpass", "neg", "advmod"))
        return [t for t in self.tokens if t["i"] in ids]


def endpoint_witness(endpoint, parsed, text):
    """Derive source annotations without changing the original endpoint/receipt."""
    start, end = endpoint_position(endpoint, parsed.text)
    okay, reason = check_reference(endpoint, parsed.text)
    if not okay:
        raise ExtractionError(reason)
    tokens = parsed.span_tokens(start, end)
    attachment_note = None
    try:
        root = parsed.root_in_span(start, end)
    except ExtractionError as exc:
        if str(exc) != "SYNTAX_SINGLE_ENDPOINT_ROOT_REQUIRED" or not endpoint.scope_qualifiers:
            raise
        spans = [(start + endpoint.name.index(q), start + endpoint.name.index(q) + len(q)) for q in endpoint.scope_qualifiers if q in endpoint.name]
        if len(spans) != len(endpoint.scope_qualifiers):
            raise ExtractionError("SYNTAX_ENDPOINT_SCOPE_NOT_EXACT") from None
        split = min(s for s, _ in spans)
        primary_end = len(parsed.text[:split].rstrip())
        root = parsed.root_in_span(start, primary_end)
        if any(not any(s <= t["start"] and t["end"] <= e for s, e in spans) for t in tokens if t["start"] >= split):
            raise ExtractionError("SYNTAX_UNCOVERED_NONCONSTITUENT_ENDPOINT") from None
        attachment_note = "PARSER_SCOPE_ATTACHMENT_DISAGREEMENT_REQUIRES_SEMANTIC_REVIEW"
    projected = endpoint.model_dump(mode="json", exclude={"scope_qualifiers"})
    annotations = []
    # Quantification remains exact source metadata, never normalized to ALL.
    match = re.match(r"(?i)(?:some(?: or all)?|all|each|every|any|either|neither)(?: of)?(?: the)?\s+", endpoint.name)
    quantifier = None
    if match:
        head_start = endpoint.name.find(endpoint.head_quote, match.end())
        if endpoint.semantic_type == "SERVICE_PRODUCT" or head_start < 0:
            raise ExtractionError("SYNTAX_QUANTIFIED_NOMINAL_HEAD_REQUIRED")
        heads = [t for t in tokens if start + head_start <= t["start"] < start + head_start + len(endpoint.head_quote)]
        if not heads or not any(t["pos"] in {"NOUN", "PROPN"} for t in heads):
            raise ExtractionError("SYNTAX_QUANTIFIED_LEXICAL_NOUN_REQUIRED")
        quantifier = {"start_char": start, "end_char": start + match.end(), "quote": match.group(),
            "interpretation": "SOURCE_QUANTIFIER_NOT_NORMALIZED_OR_UNIVERSAL"}
        residual = endpoint.name[match.end():]
        target = start + match.end()
        occurrences = [m.start() for m in re.finditer(re.escape(residual), parsed.text)]
        projected.update(name=residual, occurrence=occurrences.index(target) + 1)
    restriction = re.search(r"\b(?:using|with|that|which)\b.*", endpoint.name, re.I)
    if endpoint.semantic_type in {"RESOURCE_TYPE", "RESOURCE_INSTANCE"} and restriction and restriction.group() not in projected["restrictions"]:
        projected["restrictions"] = [*projected["restrictions"], restriction.group()]
        annotations.append({"kind": "SOURCE_RESOURCE_RESTRICTION", "quote": restriction.group(),
            "start_char": start + restriction.start(), "end_char": end, "was_missing_in_native_annotation": True})
    for qualifier in endpoint.scope_qualifiers:
        if qualifier not in endpoint.name or qualifier == endpoint.head_quote:
            raise ExtractionError("SYNTAX_ENDPOINT_SCOPE_NOT_EXACT")
    resolved = identity(EndpointV9.model_validate_json(canonical_json(projected)), parsed.text, text)
    resolved.update(start_char=text.index(parsed.text) + start, end_char=text.index(parsed.text) + end,
        scope_qualifiers=endpoint.scope_qualifiers)
    if quantifier or annotations:
        resolved["canonical_key"] = stable_id("syntax-source-local", base=resolved["canonical_key"],
            exact_endpoint=endpoint.model_dump(mode="json"), quantifier=quantifier, annotations=annotations)
    # A catalog service must be the whole nominal actor, not part of 'S3 clients'.
    for token in parsed.tokens:
        if token["i"] not in {t["i"] for t in tokens} and token["pos"] in {"NOUN", "PROPN"}:
            if any(t["head"] == token["i"] and t["dep"] in {"compound", "amod", "poss"} for t in tokens):
                raise ExtractionError("SYNTAX_SERVICE_SUBSTRING_OF_LARGER_NOMINAL")
    return {"identity": resolved, "root_token": root["i"], "span": {"start_char": start, "end_char": end, "quote": endpoint.name},
        "quantifier": quantifier, "source_annotations": annotations, "scope_attachment_note": attachment_note}


def unique_verb(parsed, quote):
    matches = list(re.finditer(re.escape(quote), parsed.text))
    if len(matches) != 1:
        raise ExtractionError("SYNTAX_UNIQUE_TRIGGER_REQUIRED")
    start, end = matches[0].span()
    verbs = [t for t in parsed.span_tokens(start, end) if t["pos"] == "VERB"]
    if len(verbs) != 1:
        raise ExtractionError("SYNTAX_SINGLE_LEXICAL_VERB_REQUIRED")
    return verbs[0]


def same_role(token, endpoint):
    return endpoint["span"]["start_char"] <= token["start"] and token["end"] <= endpoint["span"]["end_char"]


def check_modality(record, parsed, verb, chain):
    governing = parsed.governing(verb, chain)
    words = {t["lemma"].casefold() for t in governing}
    negated = any(t["dep"] == "neg" or t["text"].casefold() == "never" for t in governing)
    if negated != (record.polarity == "NEGATIVE"):
        raise ExtractionError("SYNTAX_POLARITY_SCOPE")
    if words & {"can", "may", "might", "could"} and record.modality not in {"CAPABILITY", "OPTIONAL", "PROHIBITED"}:
        raise ExtractionError("SYNTAX_MODALITY_REQUIRED")
    if "must" in words and record.modality not in {"REQUIRED", "PROHIBITED"}:
        raise ExtractionError("SYNTAX_REQUIRED_MODALITY")
    if "should" in words or record.modality == "RECOMMENDED":
        raise ExtractionError("SYNTAX_RECOMMENDATION_NOT_ACTUAL_ACTION")
    if record.modality == "REQUIRED" and not words & {"must", "require", "need"}:
        raise ExtractionError("SYNTAX_REQUIRED_WITHOUT_WITNESS")
    return governing


def require_finite_action(parsed, verb, chain):
    governing = parsed.tokens[chain[-1]["to"]] if chain else verb
    auxiliaries = parsed.children(governing, "aux", "auxpass")
    finite = governing["tag"] in {"VBD", "VBP", "VBZ"} or any(t["tag"] in {"MD", "VBD", "VBP", "VBZ"} for t in auxiliaries)
    if not finite or any(t["text"].casefold() == "to" for t in auxiliaries):
        raise ExtractionError("SYNTAX_INFINITIVE_OR_NONFINITE_NOT_ASSERTED_ACTION")


RELATION_LEMMAS = {"USES": {"use", "utilize"}, "SUPPORTS": {"support", "provide", "offer", "enable"},
    "DEPENDS_ON": {"require", "need", "depend"}, "REPLACES": {"replace"}, "INTEGRATES_WITH": {"integrate"}}
ACTION_LEMMAS = {"INVOKE": {"invoke", "trigger"}, "DELIVER": {"send", "deliver", "publish"}, "READ": {"read"},
    "WRITE": {"write"}, "ENCRYPT": {"encrypt"}, "DECRYPT": {"decrypt"}, "ASSUME_ROLE": {"assume"}}


def validate_syntax_record(record, text, parsed, *, kind):
    complete_evidence(record.clause_quote, text)
    if parsed.text != record.clause_quote:
        raise ExtractionError("SYNTAX_WRONG_SOURCE_CLAUSE")
    actor = endpoint_witness(record.subject if kind == "RELATION" else record.actor, parsed, text)
    target = endpoint_witness(record.object if kind == "RELATION" else record.target, parsed, text)
    if max(actor["span"]["start_char"], target["span"]["start_char"]) < min(actor["span"]["end_char"], target["span"]["end_char"]):
        raise ExtractionError("SYNTAX_ENDPOINT_OVERLAP")
    verb = unique_verb(parsed, record.trigger_quote if kind == "RELATION" else record.action_quote)
    family = record.predicate if kind == "RELATION" else record.action_family
    if verb["lemma"] not in (RELATION_LEMMAS if kind == "RELATION" else ACTION_LEMMAS).get(family, set()):
        raise ExtractionError("SYNTAX_LEMMA_OUTSIDE_CONTRACT")
    enabler, chain = None, []
    if kind == "ACTION" and record.construction == "ENABLED_ACTION":
        parent = parsed.tokens[verb["head"]]
        enabler = endpoint_witness(record.enabler, parsed, text) if record.enabler else None
        subjects = parsed.children(parent, "nsubj")
        objects = parsed.children(parent, "dobj")
        if (verb["dep"] != "xcomp" or parent["lemma"] != "enable" or not enabler or record.modality != "CAPABILITY"
                or len(subjects) != 1 or not same_role(subjects[0], enabler) or len(objects) != 1 or not same_role(objects[0], actor)):
            raise ExtractionError("SYNTAX_EXPLICIT_ENABLED_ACTION_REQUIRED")
        governing = parsed.governing(parent, []) + [verb]
        if any(t["dep"] == "neg" for t in governing) or record.polarity != "POSITIVE":
            raise ExtractionError("SYNTAX_NEGATED_ENABLE_SCOPE_UNSUPPORTED")
    else:
        subject, chain = parsed.subject(verb)
        if not same_role(subject, actor):
            raise ExtractionError("SYNTAX_ACTOR_NOT_VERB_SUBJECT")
        governing = check_modality(record, parsed, verb, chain)
        if kind == "ACTION":
            require_finite_action(parsed, verb, chain)
    if kind == "RELATION":
        if record.voice != "ACTIVE" or (verb["tag"] in {"VBG", "VBN"} and not parsed.children(verb, "aux")):
            raise ExtractionError("SYNTAX_FINITE_ACTIVE_RELATION_REQUIRED")
        if family == "SUPPORTS" and verb["lemma"] == "enable" and parsed.children(verb, "xcomp"):
            raise ExtractionError("SYNTAX_ENABLES_ACTION_NOT_BINARY")
        if family == "DEPENDS_ON" and re.search(r"\bpermissions?\b", parsed.text, re.I):
            raise ExtractionError("SYNTAX_PERMISSION_REQUIRES_SEPARATE_RECORD")
        if family in {"INTEGRATES_WITH", "DEPENDS_ON"} and verb["lemma"] in {"integrate", "depend"}:
            preposition = "with" if family == "INTEGRATES_WITH" else "on"
            arguments = [p for prep in parsed.children(verb, "prep") if prep["text"].casefold() == preposition for p in parsed.children(prep, "pobj")]
        else:
            arguments = parsed.children(verb, "dobj")
            # A nominal gerund may be tagged xcomp ('uses caching'). Preserve
            # the supplied concept span and flag the parse interpretation.
            if not arguments and family == "USES" and record.object.semantic_type in {"CONCEPT", "FEATURE"}:
                arguments = [t for t in parsed.children(verb, "xcomp") if t["tag"] == "VBG" and same_role(t, target)]
        if len(arguments) != 1 or not same_role(arguments[0], target):
            raise ExtractionError("SYNTAX_TARGET_NOT_DIRECT_ARGUMENT")
        payload_span = None
    else:
        if record.construction == "FINITE_ACTIVE" and record.enabler is not None:
            raise ExtractionError("SYNTAX_UNEXPECTED_ENABLER")
        role = record.target_role
        allowed = {"INVOKE": {"CALLEE"}, "DELIVER": {"DESTINATION"}, "READ": {"DATA_RESOURCE"}, "WRITE": {"DATA_RESOURCE"},
            "ENCRYPT": {"DATA_RESOURCE", "KEY"}, "DECRYPT": {"DATA_RESOURCE", "KEY"}, "ASSUME_ROLE": {"ROLE"}}
        if role not in allowed[family]:
            raise ExtractionError("SYNTAX_ARGUMENT_ROLE_MISMATCH")
        objects = parsed.children(verb, "dobj")
        payload_span = None
        if record.payload_quote is not None:
            positions = list(re.finditer(re.escape(record.payload_quote), parsed.text))
            if len(positions) != 1:
                raise ExtractionError("SYNTAX_UNIQUE_PAYLOAD_REQUIRED")
            ps, pe = positions[0].span()
            payload_span = {"start_char": ps, "end_char": pe, "quote": record.payload_quote}
            if re.match(r"^(?:it|them|this|that|these|those)\b", record.payload_quote, re.I):
                raise ExtractionError("SYNTAX_UNRESOLVED_PAYLOAD_REFERENCE")
            if len(objects) != 1 or not (ps <= objects[0]["start"] < objects[0]["end"] <= pe):
                raise ExtractionError("SYNTAX_PAYLOAD_NOT_DIRECT_OBJECT")
        if family == "DELIVER" and not payload_span:
            raise ExtractionError("SYNTAX_DELIVERY_REQUIRES_PAYLOAD")
        if payload_span:
            prep_words = {"from"} if family == "READ" else {"with", "using"} if role == "KEY" else {"to", "into"}
            arguments = [p for prep in parsed.children(verb, "prep") if prep["text"].casefold() in prep_words for p in parsed.children(prep, "pobj")]
        else:
            arguments = objects
        if len(arguments) != 1 or not same_role(arguments[0], target):
            raise ExtractionError("SYNTAX_TARGET_NOT_ROLE_ARGUMENT")
        if role in {"KEY", "ROLE"} and not re.fullmatch(r"keys?" if role == "KEY" else r"roles?", record.target.head_quote, re.I):
            raise ExtractionError("SYNTAX_KEY_OR_ROLE_HEAD_REQUIRED")
    return {"version": "source-syntax-candidate-v3", "kind": kind, "record": record.model_dump(mode="json"),
        "actor": actor, "target": target, "enabler": enabler, "verb_token": verb, "subject_chain": chain,
        "governing_tokens": governing, "payload_span": payload_span, "clause_sha256": sha256_bytes(parsed.text.encode()),
        "parse_sha256": sha256_bytes(canonical_json({"text": parsed.text, "tokens": parsed.tokens})),
        "evidence": {"start_char": 0, "end_char": len(text), "quote": text}, "source_sha256": sha256_bytes(text.encode()),
        "semantic_review": "PENDING_NOT_SERVING", "native_v10_inference_admitted": False}


def policy_recipe():
    return {"version": "source-syntax-candidate-v3", "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "contract": "HARD_SOURCE_AND_DEPENDENCY_WITNESSES_REVIEW_REQUIRED_NOT_ENTAILMENT"}
