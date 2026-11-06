from app.semantic_extraction import SemanticExtractionOutputV5
from app.semantic_extraction_v6 import review_output_v6
from tests.test_semantic_extraction import frame


def output(frames=(), permissions=()):
    import json

    return SemanticExtractionOutputV5.model_validate_json(json.dumps({"relationships": frames, "permission_requirements": permissions}))


def test_overlong_trigger_cannot_hide_wrong_object_but_exact_verb_can_be_recovered():
    text = "Managed secrets provide monitoring capabilities through CloudTrail logs."
    row = frame(text, "Managed secrets", "CloudTrail logs", "SUPPORTS", "provide monitoring capabilities through")
    assert review_output_v6(output([row]), text)[0]["reason"] == "SEMANTIC_V6_DIRECT_ARGUMENT_REQUIRED"
    text = "CloudWatch also supports PromQL for querying metrics."
    row = frame(text, "CloudWatch", "PromQL", "SUPPORTS", "supports PromQL")
    record = review_output_v6(output([row]), text)[0]
    assert record["status"] == "CANDIDATE_SOURCE_GUARDS_PASS" and record["normalized_trigger"] == "supports"
    assert record["original_trigger"] == "supports PromQL" and record["trigger_normalized"]


def test_permission_trigger_can_be_minimized_without_changing_agent_action_or_resource():
    text = "If logs are enabled, the execution role needs permission to write data to CloudWatch Logs."
    permission = {"evidence_quote": text, "principal": {"name": "execution role", "entity_type": "RESOURCE", "occurrence": 1},
        "action_text": "write data", "resource": {"name": "CloudWatch Logs", "entity_type": "AWS_SERVICE", "occurrence": 1},
        "trigger_quote": "needs permission to write data to CloudWatch Logs",
        "qualifiers": [{"kind": "OTHER", "text": "If logs are enabled"}]}
    record = review_output_v6(output(permissions=[permission]), text)[0]
    assert record["status"] == "CANDIDATE_SOURCE_GUARDS_PASS" and record["normalized_trigger"] == "needs permission"
    assert record["normalized_record"]["action_text"] == permission["action_text"]
    permission["qualifiers"] = []
    assert review_output_v6(output(permissions=[permission]), text)[0]["reason"] == "SEMANTIC_V6_PERMISSION_CONDITION_REQUIRED"
