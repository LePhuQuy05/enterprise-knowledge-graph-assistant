"""Explicit provider JSON-mode precondition; V1 receipts remain unchanged."""

import re

from app.generation import GenerationError
from app.routed_source_reader import READ_SYSTEM as READ_V1
from app.routed_source_reader import ROUTE_SYSTEM as ROUTE_V1

ROUTE_SYSTEM = ROUTE_V1.replace("Return ONLY ", "Return ONLY JSON ", 1)
READ_SYSTEM = READ_V1.replace("Return ONLY ", "Return ONLY JSON ", 1)


def require_json_mode_instruction(system):
    if type(system) is not str or not re.search(r"\bjson\b", system, re.I):
        raise GenerationError("READER_JSON_MODE_INSTRUCTION_REQUIRED_BEFORE_DISPATCH")
