from pathlib import Path

from app.extraction import AliasDictionary, normalize_alias


def test_explicit_identity_resolution_keeps_resource_types_and_service_separate():
    path = Path(__file__).resolve().parents[1] / "configs/extraction/aws-approved-aliases-v2.json"
    aliases = AliasDictionary.load(path)
    lookup = {(e.entity_type, normalize_alias(a)): e.canonical_key for e in aliases.entries for a in (*e.aliases, e.canonical_name)}
    assert lookup[("AWS_SERVICE", "s3")] == lookup[("AWS_SERVICE", "amazon s3")] == "aws/s3"
    identities = {lookup[(kind, name)] for kind, name in (("RESOURCE", "alb"), ("RESOURCE", "nlb"),
        ("AWS_SERVICE", "elastic load balancing"))}
    assert len(identities) == 3
    assert all((kind, name) not in lookup for kind in ("RESOURCE", "CONCEPT") for name in ("object", "key", "role", "function"))
    assert len(aliases.entries) == 26
