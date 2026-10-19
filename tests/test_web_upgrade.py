import ast
import re
from pathlib import Path

import pytest

from infra.tracer import handler_source
from scripts import preview_web_upgrade

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_stays_small_and_shares_auth_with_the_complete_query_package():
    assert handler_source() == handler_source()  # Asset compression must not create false deployment changes.
    namespace = {}
    exec(compile(handler_source(), "synthetic_inline_handler", "exec"), namespace)
    assets = namespace["STATIC_ASSETS"]
    source = ast.parse((ROOT / "scripts/build_query_package.py").read_text(encoding="utf-8"))
    assignment = next(node for node in ast.walk(source) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "static_paths" for target in node.targets))
    package_map = ast.literal_eval(assignment.value)
    assert set(assets) == {"/", "/assets/app.js", "/assets/style.css", "/assets/auth.mjs"}
    assert set(assets) < set(package_map)
    assert assets["/"]["body"] == (ROOT / "frontend/tracer/bootstrap.html").read_text(encoding="utf-8")
    assert assets["/assets/app.js"]["body"] == (ROOT / "frontend/tracer/bootstrap.mjs").read_text(encoding="utf-8")
    for route, (name, mime) in package_map.items():
        if route not in {"/assets/style.css", "/assets/auth.mjs"}:
            continue  # Workspace modules and corpus data are query-only; bootstrap has its own small entry.
        assert assets[route]["body"] == (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8")
        assert assets[route]["content_type"] == mime
        for imported in re.findall(r'from "\./([^"]+)"', assets[route]["body"]):
            assert "/assets/" + imported in package_map
    assert not any("preview" in path or "fixture" in path for path in assets)


def test_preview_is_closed_public_copy_and_does_not_bootstrap_production(tmp_path, monkeypatch):
    # The dedicated build root remains mandatory even under a temporary test workspace.
    monkeypatch.setattr(preview_web_upgrade, "ROOT", tmp_path)
    frontend = tmp_path / "frontend/tracer"
    frontend.mkdir(parents=True)
    for name in (*preview_web_upgrade.ASSETS, "index.html"):
        (frontend / name).write_bytes((ROOT / "frontend/tracer" / name).read_bytes())
    fixture = tmp_path / "tests/fixtures/web_upgrade"
    fixture.mkdir(parents=True)
    for name in ("records.mjs", "preview.mjs"):
        (fixture / name).write_bytes((ROOT / "tests/fixtures/web_upgrade" / name).read_bytes())
    target = tmp_path / "data/build/preview"
    report = preview_web_upgrade.build_preview(target)
    assert len(report["files"]) == 14
    assert 'src="/assets/app.js"' not in (target / "index.html").read_text(encoding="utf-8")
    assert 'src="/preview.mjs"' in (target / "index.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="DEDICATED_BUILD"):
        preview_web_upgrade.build_preview(tmp_path)
    (target / "private.json").write_text("not for serving", encoding="utf-8")
    with pytest.raises(ValueError, match="UNEXPECTED_CONTENT"):
        preview_web_upgrade.build_preview(target)
