import io
import zipfile

import pytest

from scripts.prepare_funding_transition_review import build_patch

OLD = b'import os\ndef gateway(timeout):\n    return 2\ndef handler(event, context):\n    return "preserved"\n'
NEW = b'import os\nfrom app.funding_policy import validate_external_funding\ndef gateway(timeout):\n    return 8\ndef handler(event, context):\n    return "preserved"\n'


def package():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("app/query_handler.py", OLD)
        archive.writestr("frontend/app.js", "unchanged frontend")
        archive.writestr("configs/prices.json", "unchanged prices")
    return buffer.getvalue()


def test_package_changes_only_funding_module_and_gateway_code():
    patched, old, changed = build_patch(package(), handler_bytes=NEW, policy_bytes=b"# read-only policy\n")
    assert old == OLD and changed == ["app/funding_policy.py", "app/query_handler.py"]
    with zipfile.ZipFile(io.BytesIO(patched)) as archive:
        assert archive.read("frontend/app.js") == b"unchanged frontend"
        assert archive.read("configs/prices.json") == b"unchanged prices"


def test_package_rejects_unrelated_handler_change():
    with pytest.raises(ValueError, match="UNRELATED_HANDLER"):
        build_patch(package(), handler_bytes=NEW.replace(b'"preserved"', b'"auth changed"'), policy_bytes=b"pass")
