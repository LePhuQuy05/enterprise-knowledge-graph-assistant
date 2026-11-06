"""Versioned format-only prompt adapter; never repair a model's response.

Frozen generation.py/AnswerCache keep their replay hashes. A distinct recipe
binds this suffix into cache identity; the transport records the actual composed
system message. User message, sources, citation validation and limits stay fixed.
"""

from typing import Literal

from pydantic import model_validator

from app.contracts import Sha256, sha256_bytes
from app.generation import GenerationError, GenerationRecipeV9

PROTOCOL = (
    '\nOUTPUT JSON PROTOCOL — REQUIRED REASON FIRST. '
    'The top-level JSON object MUST contain all three keys: reason, answer_status, claims. '
    'Write reason FIRST so it is never omitted after a long claims array. '
    'For ANSWERED use reason NONE; for PARTIAL or ABSTAINED use an applicable allowed non-NONE reason. '
    'Never omit reason, even when the question is fully answered. '
    'The following are FORMAT-ONLY examples, never facts, text or citation IDs to copy into a real answer: '
    '{"reason":"NONE","answer_status":"ANSWERED","claims":[{"text":"A fact explicitly supported by the supplied span.",'
    '"evidence":[{"source_id":"s1","span_id":"s1p1"}]}]} '
    '{"reason":"INSUFFICIENT_EVIDENCE","answer_status":"PARTIAL","claims":[{"text":"A supported requested fact.",'
    '"evidence":[{"source_id":"s1","span_id":"s1p1"}]}]} '
    '{"reason":"INSUFFICIENT_EVIDENCE","answer_status":"ABSTAINED","claims":[]} '
    'Choose the real status and claim text only from the actual supplied sources; '
    'retain every relevant condition and use only source/span IDs that actually exist in the user message. '
    'Return one complete JSON object with the mandatory reason key.'
)
PROTOCOL_SHA256 = sha256_bytes(PROTOCOL.encode())


class RequiredReasonRecipe(GenerationRecipeV9):
    version: Literal["deepseek-grounded-v9-required-reason-v1"] = "deepseek-grounded-v9-required-reason-v1"
    prompt_version: Literal["grounded-spans-reason-first-v1"] = "grounded-spans-reason-first-v1"
    output_protocol_sha256: Sha256 = PROTOCOL_SHA256

    @model_validator(mode="after")
    def exact_protocol(self):
        if self.output_protocol_sha256 != PROTOCOL_SHA256:
            raise ValueError("OUTPUT_PROTOCOL_RECIPE_CHANGED")
        return self


def protocol_system(base_system):
    if not isinstance(base_system, str) or not base_system.strip():
        raise GenerationError("OUTPUT_PROTOCOL_BASE_SYSTEM_REQUIRED")
    return base_system + PROTOCOL


class RequiredReasonGateway:
    def __init__(self, base):
        self.base, self.provider, self.routes = base, base.provider, base.routes

    def generate(self, context, *, system, prompt, max_output_tokens):
        if context.stage != "GENERATE" or max_output_tokens != 800:
            raise GenerationError("OUTPUT_PROTOCOL_GENERATION_ONLY")
        # No response field is inserted, deleted, coerced or repaired.
        return self.base.generate(context, system=protocol_system(system), prompt=prompt, max_output_tokens=max_output_tokens)
