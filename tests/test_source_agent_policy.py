import pytest

from app.contracts import canonical_json
from app.semantic_extraction_v7 import TypedFrameV7
from app.source_agent_policy import check_agent
from tests.test_semantic_extraction_v7 import endpoint, frame


@pytest.mark.parametrize("text", [
    "To build an API with Lambda integrations, you can use Lambda proxy integration.",
    "In either a Lambda proxy or a Lambda non-proxy integration, you can use a Lambda function.",
])
def test_subject_in_introductory_clause_is_not_the_agent(text):
    row = frame(text, endpoint("Lambda", "SERVICE_PRODUCT"), endpoint("Lambda proxy integration" if "To build" in text else "Lambda function", "FEATURE"),
        predicate="USES", trigger="can use")
    row["modality"] = "CAPABILITY"
    assert check_agent(TypedFrameV7.model_validate_json(canonical_json(row))) == (
        False, "AGENT_V1_SUBJECT_NOT_DIRECT_CLAUSE_AGENT")


@pytest.mark.parametrize("text,name,trigger", [
    ("CloudWatch also supports PromQL.", "CloudWatch", "supports"),
    ("Granular Permissions Management: Uses access entries.", "Granular Permissions Management", "Uses"),
    ("A NAT gateway supports TCP.", "NAT gateway", "supports"),
    ("CloudWatch can use TLS.", "CloudWatch", "can use"),
    ("CloudWatch does not use TLS.", "CloudWatch", "use"),
])
def test_direct_agents_keep_modal_negation_adverbs_and_heading(text, name, trigger):
    row = frame(text, endpoint(name, "CONCEPT"), endpoint("TLS" if "TLS" in text else text.split()[-1].rstrip("."), "CONCEPT"),
        predicate="USES", trigger=trigger)
    assert check_agent(TypedFrameV7.model_validate_json(canonical_json(row))) == (True, None)
