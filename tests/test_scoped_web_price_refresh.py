import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_scoped_web_price_refresh import HANDLER, NEW_PRICE, OLD_PRICE, patch_archive

PRICES = Path(NEW_PRICE).read_bytes()


def source(path, *, count=1, already_added=False):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(HANDLER, ('import funding_policy\n' + repr(Path(OLD_PRICE).name)+'\n')*count)
        z.writestr(OLD_PRICE, b"HISTORICAL_IMMUTABLE_PRICE")
        z.writestr("app/funding_policy.py", b"AUDITED_2_TO_8_USD_V2")
        z.writestr("configs/frontend-static.json", b"CURRENT_REVIEWED_STATIC_ASSETS")
        z.writestr("native/runtime.so", b"NATIVE_DEPENDENCIES")
        if already_added:
            z.writestr(NEW_PRICE, PRICES)


def test_combined_patch_preserves_ui_funding_logic_dependencies_and_historical_prices(tmp_path):
    old, new = tmp_path/"old.zip", tmp_path/"new.zip"
    source(old)
    assert patch_archive(old, new, PRICES)["changed_from_static_zip"] == [HANDLER]
    with zipfile.ZipFile(old) as a, zipfile.ZipFile(new) as b:
        assert set(b.namelist())-set(a.namelist()) == {NEW_PRICE}
        for name in a.namelist():
            expected = a.read(name)
            if name == HANDLER:
                expected = expected.replace(Path(OLD_PRICE).name.encode(), Path(NEW_PRICE).name.encode())
            assert b.read(name) == expected


@pytest.mark.parametrize("count", [0, 2])
def test_ambiguous_or_missing_price_literal_is_rejected_before_build(tmp_path, count):
    old, new = tmp_path/"old.zip", tmp_path/"new.zip"
    source(old, count=count)
    with pytest.raises(ValueError, match="ONE_LITERAL"):
        patch_archive(old, new, PRICES)
    assert not new.exists()


def test_cannot_reapply_to_a_package_with_current_price_asset(tmp_path):
    old = tmp_path/"old.zip"
    source(old, already_added=True)
    with pytest.raises(ValueError, match="UNEXPECTED_ARCHIVE"):
        patch_archive(old, tmp_path/"new.zip", PRICES)


def test_stale_price_sheet_is_rejected_before_any_package_is_created(tmp_path):
    old, new = tmp_path/"old.zip", tmp_path/"new.zip"
    source(old)
    raw = json.loads(PRICES)
    raw.update(checked_at="2020-01-01T00:00:00Z", expires_at="2020-01-02T00:00:00Z")
    with pytest.raises(ValueError):
        patch_archive(old, new, json.dumps(raw).encode())
    assert not new.exists()
