import pytest

from app.semantic_extraction_v9 import EndpointV9
from app.source_entity_boundary import check_reference


@pytest.mark.parametrize("name,clause,allowed", [
    ("two calls", "These two calls use taskToken.", False),
    ("calls", "These two calls use taskToken.", False),
    ("CloudWatch", "CloudWatch uses Amazon SNS.", True),
    ("two-factor authentication", "two-factor authentication requires credentials.", True),
])
def test_reference_scope_cannot_be_erased_by_endpoint_substring(name, clause, allowed):
    endpoint = EndpointV9(name=name, semantic_type="CONCEPT", head_quote=name, occurrence=1, restrictions=())
    assert check_reference(endpoint, clause)[0] is allowed
