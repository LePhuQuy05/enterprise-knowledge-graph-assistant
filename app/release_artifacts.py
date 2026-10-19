"""Fail-closed selection for local release audits; never export private runtime state."""

import json
import re
import subprocess
import tomllib
from pathlib import PurePosixPath

from app.contracts import canonical_json, sha256_bytes

ROOT_FILES = {"README.md", "AGENTS.md", "GLOSSARY.md", "LICENSE", "NOTICE.md", ".gitignore",
              ".python-version", "pyproject.toml", "uv.lock", "package.json", "package-lock.json", "cdk.json"}
SOURCE_ROOTS = {"app", "configs", "docs", "frontend", "infra", "schemas", "scripts", "tests", ".scratch", ".github"}
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".json", ".toml", ".js", ".mjs", ".html", ".css", ".yml", ".yaml"}


def relative_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("UNSAFE_RELEASE_PATH")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in {"..", "."} for p in name.split("/")) or path.as_posix() != name:
        raise ValueError("UNSAFE_RELEASE_PATH")
    return path


def approved_source(name):
    path = relative_path(name)
    denied = {"private", "credentials", "__pycache__", "node_modules", ".aws", ".venv"}
    if any(p.lower() in denied or p.lower().startswith(".env") for p in path.parts):
        return False
    if path.name.lower() in {"credentials.json", "reviewer_credentials.json"}:
        return False
    return name in ROOT_FILES or (len(path.parts) > 1 and path.parts[0] in SOURCE_ROOTS
                                  and path.suffix.lower() in TEXT_SUFFIXES)


def read_source(root, name):
    relative_path(name)
    path = root / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("SYMLINK_RELEASE_SOURCE")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("UNSAFE_RELEASE_SOURCE")
    return path.read_bytes()


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError("RELEASE_GIT_READ_FAILED")
    return result.stdout


def workspace_sources(root):
    paths = sorted(set(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""})
    return {name: read_source(root, name) for name in paths if approved_source(name) and (root / name).exists()}


def index_sources(root):
    files = {}
    for record in git(root, "ls-files", "--stage", "-z").decode().split("\0"):
        if not record:
            continue
        meta, name = record.split("\t", 1)
        mode, oid, stage = meta.split()
        relative_path(name)
        if mode not in {"100644", "100755"} or stage != "0" or not re.fullmatch(r"[0-9a-f]{40,64}", oid):
            raise ValueError("UNSAFE_OR_CONFLICTED_INDEX")
        # Do not read a staged private credential file even by its blob ID.
        if not approved_source(name) and name != "results/environment_preflight.json":
            raise ValueError("UNAPPROVED_INDEX_PATH")
        raw = git(root, "cat-file", "blob", oid)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("INDEX_BLOB_TOO_LARGE")
        files[name] = raw
    return files


def ref_tree_sources(root):
    """Codex snapshot refs may point to trees, not commits; scan their actual blobs too."""
    refs = git(root, "for-each-ref", "--format=%(objectname) %(objecttype)").decode().splitlines()
    files = {}
    roots = sorted({line.split()[0] for line in refs})
    for oid in roots:
        for record in git(root, "ls-tree", "-r", "-z", oid).decode().split("\0"):
            if not record:
                continue
            meta, name = record.split("\t", 1)
            mode, kind, blob = meta.split()
            relative_path(name)
            if mode not in {"100644", "100755"} or kind != "blob" or not re.fullmatch(r"[0-9a-f]{40,64}", blob):
                raise ValueError("UNSAFE_REF_TREE_ENTRY")
            if not approved_source(name) and name != "results/environment_preflight.json":
                raise ValueError("UNAPPROVED_REF_TREE_PATH")
            raw = git(root, "cat-file", "blob", blob)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("REF_BLOB_TOO_LARGE")
            files[oid + "/" + name] = raw
    return files, len(refs), len(roots)


def triage_findings(rows, files, entries, *, ref_tree=False):
    """Exact reviewed line identities only; raw findings remain, unknown findings block."""
    reviewed, pending = [], []
    for row in rows:
        name, number = row["File"], row["StartLine"]
        original = name.split("/", 1)[1] if ref_tree else name
        lines = files.get(name, b"").decode("utf-8", errors="replace").splitlines()
        digest = sha256_bytes(lines[number - 1].encode()) if 0 < number <= len(lines) else None
        match = next((e for e in entries if row.get("EndLine", number) == number
            and e["path"] == original and e["line_number"] == number
            and e["rule_id"] == row["RuleID"] and e["line_sha256"] == digest), None)
        if match:
            reviewed.append({"file": original, "line": number, "classification": match["classification"]})
        else:
            pending.append(row)
    return {"reviewed_false_positives": reviewed, "unreviewed_findings": pending,
            "unreviewed_count": len(pending)}


def materialize(files, directory):
    directory.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name, raw in sorted(files.items()):
        relative_path(name)
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        inventory.append({"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    return {"files": inventory, "content_sha256": sha256_bytes(canonical_json(inventory))}


def locked_packages(root):
    py = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    result = [{"ecosystem": "PyPI", "name": p["name"], "version": p["version"]}
              for p in py["package"] if "registry" in p.get("source", {})]
    npm = json.loads((root / "package-lock.json").read_bytes())
    result.extend({"ecosystem": "npm", "name": name.split("node_modules/")[-1], "version": p["version"]}
                  for name, p in npm["packages"].items() if name and "version" in p)
    if not result or len(result) > 1000:
        raise ValueError("INVALID_DEPENDENCY_INVENTORY")
    for row in result:
        if not all(isinstance(v, str) and 0 < len(v) <= 200 for v in row.values()):
            raise ValueError("INVALID_DEPENDENCY_IDENTITY")
    return sorted(result, key=lambda p: (p["ecosystem"], p["name"], p["version"]))


def osv_summary(packages, response):
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list) or len(results) != len(packages):
        raise ValueError("INCOMPLETE_OSV_RESPONSE")
    findings = []
    for package, result in zip(packages, results, strict=True):
        if not isinstance(result, dict) or result.get("next_page_token") or result.get("error"):
            raise ValueError("INCOMPLETE_OSV_PAGINATION_OR_ERROR")
        vulns = result.get("vulns", [])
        if not isinstance(vulns, list) or any(not isinstance(v, dict) or not isinstance(v.get("id"), str) for v in vulns):
            raise ValueError("INVALID_OSV_FINDINGS")
        if vulns:
            findings.append(package | {"advisory_ids": sorted({v["id"] for v in vulns})})
    return {"status": "KNOWN_ADVISORIES_FOUND" if findings else "NO_KNOWN_ADVISORIES_RETURNED",
            "packages_queried": len(packages), "pypi_packages": sum(p["ecosystem"] == "PyPI" for p in packages),
            "npm_packages": sum(p["ecosystem"] == "npm" for p in packages), "findings": findings}
