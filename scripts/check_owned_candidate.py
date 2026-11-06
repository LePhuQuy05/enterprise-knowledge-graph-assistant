"""Offline completion checks for the reviewed cloud/Compare candidate; no AWS clients."""
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from app.contracts import canonical_json, sha256_bytes
from scripts.query_cloud import inspect_code_update

root = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path.cwd()
out = root / "results/owned-cloud"
def read(path):
    return json.loads((root / path).read_bytes())

native = read("results/owned-cloud/native-package-v1.json")
web = read("results/owned-cloud/web-package-v1.json")
plan = read("results/owned-cloud/review-v1/plan.json")
template = read("results/owned-cloud/review-v1/template.json")
assert sha256_bytes(canonical_json(template)) == plan["template_sha256"]
assert len(template["Resources"]) == 30
assert plan["public_routes_added"] == 8 and not plan["answering_enabled"]
assert inspect_code_update(read("results/owned-cloud/review-v1/website-template-baseline.json"),
                           read("results/owned-cloud/review-v1/website-static-template.json")) == plan["web_update"]
with ZipFile(native["package_path"]) as archive:
    changed = [name for name in archive.namelist() if name.startswith("app/") and name.endswith(".py")
               and archive.read(name) != (root / name).read_bytes()]
    assert not changed, changed
assert native["owned_workflow"]["status"] == "PASS_NATIVE_OWNED_WORKFLOW"
assert native["owned_workflow"]["source_appearances"] == 24
with ZipFile(web["package_path"]) as archive:
    assets = json.loads(archive.read("configs/frontend-static.json"))
assert len(assets) == 13
for route, item in assets.items():
    assert sha256_bytes(item["body"].encode()) == web["public_assets"][route]["sha256"]
assert web["sha256"] == plan["web_package_sha256"]
assert native["package"]["sha256"] == plan["native_package_sha256"]
for package, digest in ((native["package_path"], plan["native_package_sha256"]),
                        (web["package_path"], plan["web_package_sha256"])):
    with Path(package).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
xml = ET.parse(out / "checks/pytest.xml")
suites = list(xml.getroot().iter("testsuite"))
assert sum(int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in suites) == 0
assert sum(int(s.attrib["tests"]) for s in suites) == 1951
assert sum(int(s.attrib["skipped"]) for s in suites) == 2
assert "pass 79" in (out / "checks/ui-tests-final.txt").read_text(encoding="utf-8")
assert "All checks passed!" in (out / "checks/ruff.txt").read_text(encoding="utf-8")
browser = read("results/owned-cloud/browser/final-functional.json")
assert browser["status"] == "PASS_LOCAL_COMPARE_BROWSER" and len(browser["checks"]) == 19
old = read("results/web-upgrade/2026-10-25T10-52-17-760Z/report.json")
assert old["status"] == "PASS_LOCAL_SYNTHETIC_BROWSER" and len(old["checks"]) == 41
screens = {}
for width, height in ((1440, 1000), (768, 1000), (390, 844)):
    path = out / "browser" / f"compare-{width}.png"
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (width, height)
    screens[str(width)] = {"file": path.relative_to(root).as_posix(), "sha256": sha256_bytes(data)}
docpaths = [
    "docs/owned_cloud_report.md", "docs/guide/LIVE_COMPARE.md",
    "docs/decisions/0038-transactional-work-slots-and-live-retrieval-compare.md",
    "docs/learning-log/12_aws_iac_and_workflows/2026-10-25_174700_owned_cloud_compare.md",
    ".scratch/rag-lab/issues/16-cloud-retrieval-compare.md",
]
for name in docpaths:
    path = root / name
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" not in target and not target.startswith("#"):
            assert (path.parent / target.split("#")[0]).resolve().exists(), (name, target)
journal = (root / docpaths[3]).read_text(encoding="utf-8")
assert [int(n) for n in re.findall(r"^## (\d+)\.", journal, re.M)] == list(range(1, 13))
contract = read("docs/api/owned-runs.openapi.json")
assert len(contract["paths"]) == 7  # Two HTTP methods share /v1/runs: eight routes.
report = {
    "status": "PASS_LOCAL_CANDIDATE_NOT_AWS_RELEASE",
    "checked_at": datetime.now(UTC).isoformat(),
    "python_passed": 1949, "python_skipped": 2, "node_passed": 79,
    "compare_browser_checks": 19, "old_ui_browser_checks": 41,
    "native_current_source_bytes_match": True, "native_source_appearances": 24,
    "template_resources": 30, "jwt_routes": 8, "screenshots": screens,
    "documentation_links_exist": True, "journal_sections": 12,
    "aws_writes": 0, "model_calls": 0,
    "cloud_deployed": False, "managed_queue_workflow_verified": False,
    "positive_jwt_browser_verified": False, "quality_metrics": None,
}
(out / "completion.json").write_bytes(canonical_json(report) + b"\n")
print(json.dumps({key: report[key] for key in ("status", "python_passed", "node_passed", "compare_browser_checks", "old_ui_browser_checks", "cloud_deployed")}))

