import zipfile
from pathlib import Path

import pytest

from scripts.build_query_price_refresh import HANDLER, NEW_PRICE, OLD_PRICE, patch_archive


def base_zip(path, *, occurrences=1, already_added=False):
    with zipfile.ZipFile(path, "w") as output:
        output.writestr(HANDLER, (Path(OLD_PRICE).name + "\n") * occurrences)
        output.writestr("configs/frontend-static.json", '{"/":"EXISTING_UI"}')
        output.writestr(OLD_PRICE, b"ORIGINAL_PRICE_REMAINS_UNCHANGED")
        output.writestr("numpy/synthetic.so", b"ORIGINAL_NATIVE_DEPENDENCY")
        if already_added:
            output.writestr(NEW_PRICE, b"UNEXPECTED")


def test_price_patch_preserves_every_unrelated_byte_and_original_historical_rate_sheet(tmp_path):
    before, after = tmp_path / "base.zip", tmp_path / "new.zip"
    base_zip(before)
    delta = patch_archive(before, after, b"VERIFIED_NEW_PRICE")
    assert delta["changed"] == [HANDLER] and delta["added"] == [NEW_PRICE] and delta["removed"] == []
    assert delta["unchanged_entries"] == 3
    with zipfile.ZipFile(before) as left, zipfile.ZipFile(after) as right:
        assert right.read(HANDLER) == Path(NEW_PRICE).name.encode() + b"\n"
        assert right.read(NEW_PRICE) == b"VERIFIED_NEW_PRICE"
        assert all(left.read(name) == right.read(name) for name in left.namelist() if name != HANDLER)


@pytest.mark.parametrize("count", [0, 2])
def test_patch_rejects_a_different_or_ambiguous_handler_before_any_publication(tmp_path, count):
    before = tmp_path / "base.zip"
    base_zip(before, occurrences=count)
    with pytest.raises(ValueError, match="ONE_LITERAL_PATH"):
        patch_archive(before, tmp_path / "new.zip", b"CURRENT")


def test_cannot_apply_the_patch_to_an_already_modified_base(tmp_path):
    before = tmp_path / "base.zip"
    base_zip(before, already_added=True)
    with pytest.raises(ValueError, match="UNEXPECTED_BASE"):
        patch_archive(before, tmp_path / "new.zip", b"CURRENT")
