import json
from zipfile import ZipFile

import pytest

from scripts.build_benchmark_web import STATIC, patch_assets


def test_only_static_entry_changes_and_backend_bytes_stay_identical(tmp_path):
    source, target = tmp_path / "base.zip", tmp_path / "candidate.zip"
    old = {"/": {"body": "old", "content_type": "text/html"}}
    new = old | {"/assets/benchmark-data.json": {"body": "{}", "content_type": "application/json"}}
    with ZipFile(source, "w") as archive:
        archive.writestr(STATIC, json.dumps(old))
        archive.writestr("app/query_handler.py", b"deployed backend bytes")
        archive.writestr("configs/prices.json", b"deployed prices bytes")
    report = patch_assets(source, target, new)
    assert report["changed_zip_entries"] == [STATIC]
    assert report["unchanged_entries"] == 2
    with ZipFile(target) as archive:
        assert archive.read("app/query_handler.py") == b"deployed backend bytes"
        assert archive.read("configs/prices.json") == b"deployed prices bytes"
        assert json.loads(archive.read(STATIC)) == new
    with pytest.raises(ValueError, match="EXISTS"):
        patch_assets(source, target, new)
    with pytest.raises(ValueError, match="REMOVE_EXISTING"):
        patch_assets(source, tmp_path / "removed.zip", {})
