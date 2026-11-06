"""Opt-in V11 coverage/antecedent contract with an explicit four-key protocol.

Generation and validation use the existing frozen V11 implementation. The raw
provider response is never repaired. This is a coverage-contract experiment,
not a format-only comparison: V11 also changes citation/scope/gap instructions.
"""

from app.contracts import canonical_json, sha256_bytes
from app.generation import GenerationError, GenerationRecipeV11

PROTOCOL = (
    '\nFOUR-KEY COVERAGE OUTPUT PROTOCOL. Return exactly reason, answer_status, claims, limitations. '
    'Write reason first and always include limitations. ANSWERED uses reason NONE and limitations []; '
    'PARTIAL/ABSTAINED require an allowed non-NONE reason and 1-3 limitations. '
    'These examples demonstrate structure ONLY, not facts, requested parts or citation IDs to copy: '
    '{"reason":"NONE","answer_status":"ANSWERED","claims":[{"text":"A supplied supported fact.",'
    '"evidence":[{"source_id":"s1","span_id":"s1p1"}]}],"limitations":[]} '
    '{"reason":"INSUFFICIENT_EVIDENCE","answer_status":"PARTIAL","claims":[{"text":"A supplied supported fact.",'
    '"evidence":[{"source_id":"s1","span_id":"s1p1"}]}],"limitations":[{"code":"MISSING_SUPPLIED_EVIDENCE",'
    '"question_part":"The requested part not established by these passages"}]} '
    '{"reason":"INSUFFICIENT_EVIDENCE","answer_status":"ABSTAINED","claims":[],"limitations":'
    '[{"code":"ACCOUNT_STATE_UNAVAILABLE","question_part":"The requested current account configuration"}]} '
    'Choose real claims, status and question_part labels from this question and supplied evidence. '
    'A limitation is a concise label for a missing requested part in the requested language, not an answer, '
    'a new source, proof of absence, or a claim about actual account state. '
    'Do not fill claims with unrelated background. Keep each claim within the exact actor, feature, '
    'antecedents and conditions established by its own selected one or two spans. '
    'Do not invent a conclusion to remove a limitation or force ANSWERED. Return one complete JSON object.'
)
PROTOCOL_SHA256 = sha256_bytes(PROTOCOL.encode())


# V11 deliberately revalidates its exact recipe schema. Do not weaken that
# guard or smuggle extra fields into it. Bind the provider suffix separately in
# the experiment/request identity and audit the actual provider payload.
CoverageProtocolRecipe = GenerationRecipeV11

def coverage_contract_sha256(recipe):
    if type(recipe) is not GenerationRecipeV11 or PROTOCOL_SHA256 != sha256_bytes(PROTOCOL.encode()):
        raise ValueError("COVERAGE_OUTPUT_PROTOCOL_CHANGED")
    GenerationRecipeV11.model_validate_json(recipe.model_dump_json())
    return sha256_bytes(canonical_json({"version": "v11-four-key-coverage-contract-v1",
        "generation_recipe_sha256": recipe.fingerprint, "provider_protocol_sha256": PROTOCOL_SHA256}))


def protocol_system(base_system):
    if not isinstance(base_system, str) or not base_system.strip():
        raise GenerationError("COVERAGE_PROTOCOL_BASE_SYSTEM_REQUIRED")
    return base_system + PROTOCOL


class CoverageProtocolGateway:
    def __init__(self, base):
        self.base, self.provider, self.routes = base, base.provider, base.routes

    def generate(self, context, *, system, prompt, max_output_tokens):
        if context.stage != "GENERATE" or max_output_tokens != 800:
            raise GenerationError("COVERAGE_PROTOCOL_GENERATION_ONLY")
        return self.base.generate(context, system=protocol_system(system), prompt=prompt, max_output_tokens=max_output_tokens)
