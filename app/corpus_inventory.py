"""Offline, read-only inventory of archive content suitable for corpus selection.

An archive DONE state is a crawler assertion. This audit independently verifies
the referenced bytes and metadata, without claiming that the content is useful,
indexed, current, or semantically reviewed. HTML/Markdown URL pairs are only
deduplication candidates until their normalized text hashes agree.
"""

import json
import re
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import ValidationError

from app.archive import (
    NORMALIZED_LIMIT,
    Archive,
    ArchivePolicy,
    archive_url,
    location,
    resource_format,
)
from app.contracts import canonical_json, stable_id, text_hash
from app.ingestion.models import Section

RECIPE = "archive-coverage-v1"
DOCUMENT_KEYS = {
    "schema_version", "archive_document_version_id", "source_uri", "effective_uri", "family",
    "language", "format", "version_hint", "fetched_at", "raw_sha256", "normalized_sha256",
    "normalizer_version", "title", "text", "sections", "warnings", "semantic_review",
}
REFERENCE_KEYS = (
    "archive_document_version_id", "source_uri", "effective_uri", "family", "language", "format",
    "version_hint", "fetched_at", "raw_sha256", "normalized_sha256", "normalizer_version",
    "title", "warnings", "semantic_review",
)
SHA256 = re.compile(r"[a-f0-9]{64}")


class InventoryInvalid(ValueError):
    """A stable diagnostic code without arbitrary document content."""


def guide_key(url: str) -> str:
    """Preserve the version and guide path rather than merging all versions."""
    parts = urlsplit(archive_url(url)).path.strip("/").split("/")
    if parts and re.fullmatch(r"[a-zA-Z]{2}_[a-zA-Z]{2}", parts[0]):
        parts = parts[1:]
    return "/".join(parts[:-1][:3]) or "_root"


def page_stem(url: str) -> str:
    """Only a *candidate* identity, not content equivalence or document identity."""
    url = archive_url(url)
    return re.sub(r"\.(?:html?|md|markdown)$", "", url, flags=re.I)


def _require(condition, code):
    if not condition:
        raise InventoryInvalid(code)


def _validate_document(doc, row):
    _require(type(doc) is dict and set(doc) == DOCUMENT_KEYS, "DOCUMENT_SCHEMA_INVALID")
    _require(type(doc["schema_version"]) is int and doc["schema_version"] == 1,
             "DOCUMENT_SCHEMA_VERSION_INVALID")
    for name in ("source_uri", "effective_uri"):
        _require(type(doc[name]) is str and archive_url(doc[name]) == doc[name],
                 "DOCUMENT_URI_INVALID")
    _require(doc["source_uri"] == row["url"], "DOCUMENT_SOURCE_MISMATCH")
    _require(type(row["effective_url"]) is str and archive_url(row["effective_url"]) == row["effective_url"],
             "RESOURCE_EFFECTIVE_URI_INVALID")
    language, family = location(doc["source_uri"])
    _require((doc["language"], doc["family"], doc["format"]) == (
        language, family, resource_format(doc["source_uri"])), "DOCUMENT_LOCATION_MISMATCH")
    _require((doc["language"], doc["family"], doc["format"]) == (
        row["language"], row["family"], row["format"]), "RESOURCE_METADATA_MISMATCH")
    _require(doc["format"] in {"html", "markdown", "text"}, "DOCUMENT_FORMAT_INVALID")
    for name in ("raw_sha256", "normalized_sha256"):
        _require(type(doc[name]) is str and SHA256.fullmatch(doc[name]), "DOCUMENT_HASH_INVALID")
    _require(doc["raw_sha256"] == row["raw_sha256"], "DOCUMENT_RAW_HASH_MISMATCH")
    _require(type(doc["text"]) is str and doc["text"].strip() and "\r" not in doc["text"]
             and not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", doc["text"]),
             "DOCUMENT_TEXT_INVALID")
    _require(text_hash(doc["text"]) == doc["normalized_sha256"], "DOCUMENT_TEXT_HASH_MISMATCH")
    for name in ("title", "normalizer_version"):
        _require(type(doc[name]) is str and bool(doc[name].strip()), "DOCUMENT_METADATA_INVALID")
    expected = stable_id("archdoc", source_uri=doc["source_uri"], raw_sha256=doc["raw_sha256"],
                         normalizer=doc["normalizer_version"])
    _require(doc["archive_document_version_id"] == expected, "DOCUMENT_IDENTITY_MISMATCH")
    expected_hint = "latest" if "/latest/" in doc["source_uri"] else "versioned_or_unspecified"
    _require(doc["version_hint"] == expected_hint, "DOCUMENT_VERSION_HINT_MISMATCH")
    _require(type(doc["fetched_at"]) is str and type(row["fetched_at"]) is str,
             "DOCUMENT_FETCH_TIME_INVALID")
    try:
        fetched_at = datetime.fromisoformat(doc["fetched_at"])
        resource_fetched_at = datetime.fromisoformat(row["fetched_at"])
        _require(fetched_at.utcoffset() is not None and fetched_at.utcoffset().total_seconds() == 0,
                 "DOCUMENT_FETCH_TIME_INVALID")
        _require(resource_fetched_at.utcoffset() is not None and
                 resource_fetched_at.utcoffset().total_seconds() == 0, "RESOURCE_FETCH_TIME_INVALID")
        # Re-fetching identical bytes reuses an immutable version record. Its
        # original timestamp/redirect URI need not equal the latest fetch.
        _require(fetched_at <= resource_fetched_at, "DOCUMENT_FETCH_TIME_ORDER")
    except (TypeError, ValueError) as exc:
        raise InventoryInvalid("DOCUMENT_FETCH_TIME_INVALID") from exc
    _require(type(doc["warnings"]) is list and all(type(v) is str for v in doc["warnings"]),
             "DOCUMENT_WARNINGS_INVALID")
    # This archive format has no reviewed/approved state. Do not infer one from
    # a changed string in an untrusted normalized record.
    _require(doc["semantic_review"] == "NOT_REVIEWED", "DOCUMENT_REVIEW_STATUS_INVALID")
    _require(type(doc["sections"]) is list, "DOCUMENT_SECTIONS_INVALID")
    for value in doc["sections"]:
        try:
            section = Section.model_validate_json(canonical_json(value))
        except (TypeError, ValueError, ValidationError) as exc:
            raise InventoryInvalid("DOCUMENT_SECTIONS_INVALID") from exc
        _require(section.end_char <= len(doc["text"]), "DOCUMENT_SECTION_OUT_OF_RANGE")


def _outside_archive(path: Path, root: Path):
    _require(not path.resolve().is_relative_to(root.resolve()), "OUTPUT_MUST_BE_OUTSIDE_ARCHIVE")


def audit_archive(root: Path, policy: ArchivePolicy, *, candidate_output: Path | None = None,
                  selected_urls: tuple[str, ...] | None = None) -> dict:
    """Validate current PAGE/DONE records and optionally emit sorted JSONL references.

    The DB is opened with mode=ro and a stable read transaction. Validation is
    streaming: one bounded normalized document/raw blob is loaded at a time.
    Only small reference metadata, counters and digest group members accumulate.
    No crawler, model, AWS client, FTS index, or semantic review is run.
    """
    root = Path(root).resolve()
    if selected_urls is not None:
        _require(type(selected_urls) is tuple and len(selected_urls) <= 10_000,
                 "INVENTORY_SELECTION_INVALID")
        _require(all(type(url) is str and archive_url(url) == url for url in selected_urls),
                 "INVENTORY_SELECTION_URI_INVALID")
        selected_urls = tuple(sorted(set(selected_urls)))
    temporary = None
    manifest = None
    if candidate_output is not None:
        candidate_output = Path(candidate_output)
        _outside_archive(candidate_output, root)
        candidate_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = candidate_output.with_name(candidate_output.name + "." + uuid4().hex + ".tmp")
    counts = Counter()
    reasons = Counter()
    families = defaultdict(Counter)
    guides = Counter()
    formats = Counter()
    languages = Counter()
    versions = Counter()
    exact = defaultdict(list)
    stems = defaultdict(list)
    checked_raw = set()
    invalid_samples = []
    selected_missing = selected_not_page = selected_page_not_done = None
    total_chars = total_bytes = 0
    try:
        if temporary is not None:
            manifest = temporary.open("xb")
        with Archive(root, policy, read_only=True) as archive:
            archive.db.execute("PRAGMA query_only=ON")
            archive.db.execute("BEGIN")
            states = {}
            if selected_urls is None:
                for row in archive.db.execute(
                        "SELECT kind,state,COUNT(*) AS n FROM resources GROUP BY kind,state ORDER BY kind,state"):
                    states.setdefault(row["kind"], {})[row["state"]] = row["n"]
                for row in archive.db.execute(
                        "SELECT family,state,COUNT(*) AS n FROM resources WHERE kind='PAGE' "
                        "GROUP BY family,state ORDER BY family,state"):
                    families[row["family"]]["state_" + row["state"]] = row["n"]
                rows = archive.db.execute(
                    "SELECT * FROM resources WHERE kind='PAGE' AND state='DONE' ORDER BY url")
            else:
                # Primary-key lookups avoid re-scanning a very large frontier.
                # Only bounded metadata is retained, not the source bodies.
                selected_rows = []
                selected_missing = selected_not_page = selected_page_not_done = 0
                for url in selected_urls:
                    row = archive.db.execute("SELECT * FROM resources WHERE url=?", (url,)).fetchone()
                    if row is None:
                        selected_missing += 1
                        continue
                    own_states = states.setdefault(row["kind"], {})
                    own_states[row["state"]] = own_states.get(row["state"], 0) + 1
                    if row["kind"] != "PAGE":
                        selected_not_page += 1
                        continue
                    families[row["family"]]["state_" + row["state"]] += 1
                    if row["state"] != "DONE":
                        selected_page_not_done += 1
                    else:
                        selected_rows.append(row)
                rows = selected_rows
            for row in rows:
                counts["page_done_records"] += 1
                try:
                    _require(archive_url(row["url"]) == row["url"], "RESOURCE_URI_INVALID")
                    _require(type(row["document_sha256"]) is str and
                             SHA256.fullmatch(row["document_sha256"]), "DOCUMENT_BLOB_HASH_INVALID")
                    raw = archive.read_blob(row["document_sha256"], NORMALIZED_LIMIT)
                    try:
                        doc = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise InventoryInvalid("DOCUMENT_JSON_INVALID") from exc
                    _validate_document(doc, row)
                    linked = archive.db.execute(
                        "SELECT 1 FROM versions WHERE version_id=? AND url=? AND document_sha256=?",
                        (doc["archive_document_version_id"], row["url"], row["document_sha256"]),
                    ).fetchone()
                    _require(linked is not None, "DOCUMENT_VERSION_LINK_MISSING")
                    if doc["raw_sha256"] not in checked_raw:
                        archive.read_blob(doc["raw_sha256"], policy.max_page_bytes)
                        checked_raw.add(doc["raw_sha256"])
                    reference = {key: doc[key] for key in REFERENCE_KEYS}
                    reference.update({
                        "schema_version": 1, "inventory_recipe": RECIPE,
                        "document_sha256": row["document_sha256"],
                        "archive_resource_fetched_at": row["fetched_at"],
                        "archive_resource_effective_uri": row["effective_url"],
                        "guide_key": guide_key(doc["source_uri"]),
                        "page_stem": page_stem(doc["source_uri"]),
                        "text_characters": len(doc["text"]),
                        "text_utf8_bytes": len(doc["text"].encode("utf-8")),
                        "structural_validation": "PASS", "raw_blob_verified": True,
                        "serving_index_status": "NOT_ASSESSED",
                    })
                    if manifest is not None:
                        manifest.write(canonical_json(reference) + b"\n")
                    counts["structurally_usable_records"] += 1
                    counts["records_with_warnings"] += bool(doc["warnings"])
                    families[doc["family"]]["structurally_usable_records"] += 1
                    guides[reference["guide_key"]] += 1
                    formats[doc["format"]] += 1
                    languages[doc["language"]] += 1
                    versions[doc["version_hint"]] += 1
                    exact[doc["normalized_sha256"]].append(doc["source_uri"])
                    if doc["format"] in {"html", "markdown"}:
                        stems[reference["page_stem"]].append((doc["format"], doc["normalized_sha256"]))
                    total_chars += reference["text_characters"]
                    total_bytes += reference["text_utf8_bytes"]
                except (OSError, EOFError, ValueError, TypeError, KeyError, RecursionError, zlib.error) as exc:
                    code = str(exc) if isinstance(exc, InventoryInvalid) else "ARCHIVE_RECORD_INVALID"
                    reasons[code] += 1
                    counts["invalid_done_records"] += 1
                    if len(invalid_samples) < 10:
                        invalid_samples.append({"source_uri_sha256": text_hash(str(row["url"])), "reason": code})
            archive.db.rollback()
        if manifest is not None:
            manifest.close()
            manifest = None
            temporary.replace(candidate_output)
        exact_groups = [urls for urls in exact.values() if len(urls) > 1]
        pairs = [items for items in stems.values() if {v[0] for v in items} >= {"html", "markdown"}]
        identical_pairs = sum(len({digest for _, digest in items}) == 1 for items in pairs)
        return {
            "schema_version": 1, "inventory_recipe": RECIPE,
            "audit_scope": "FULL" if selected_urls is None else "FILTERED",
            "resource_state_counts_scope": "FULL" if selected_urls is None else "FILTERED",
            "selected_url_count": len(selected_urls) if selected_urls is not None else None,
            "selected_urls_sha256": text_hash(canonical_json(selected_urls).decode("utf-8")) if selected_urls is not None else None,
            "selected_missing_resource_count": selected_missing,
            "selected_not_page_count": selected_not_page,
            "selected_page_not_done_count": selected_page_not_done,
            "status": "AUDITED_WITH_INVALID_RECORDS" if counts["invalid_done_records"] else "AUDITED",
            "checked_at": datetime.now(UTC).isoformat(), "policy_version": policy.version,
            "policy_fingerprint": policy.fingerprint, "archive_resource_states": states,
            "counts": {key: counts[key] for key in (
                "page_done_records", "structurally_usable_records", "invalid_done_records", "records_with_warnings")},
            "verified_raw_blobs": len(checked_raw), "text_characters": total_chars,
            "text_utf8_bytes": total_bytes,
            "family_counts": {key: dict(sorted(value.items())) for key, value in sorted(families.items())},
            "guide_counts": dict(sorted(guides.items())), "format_counts": dict(sorted(formats.items())),
            "language_counts": dict(sorted(languages.items())), "version_hint_counts": dict(sorted(versions.items())),
            "deduplication": {
                "exact_normalized_text_groups": len(exact_groups),
                "exact_duplicate_records_beyond_first": sum(len(urls) - 1 for urls in exact_groups),
                "unique_normalized_text_hashes": len(exact),
                "html_markdown_same_stem_candidate_groups": len(pairs),
                "html_markdown_same_stem_exact_text_groups": identical_pairs,
                "html_markdown_same_stem_unverified_equivalence_groups": len(pairs) - identical_pairs,
                "notice": "Same stem never authorizes merging different text; retain all source provenance.",
            },
            "invalid_reason_counts": dict(sorted(reasons.items())), "invalid_samples": invalid_samples,
            "candidate_manifest": str(candidate_output) if candidate_output is not None else None,
            "serving_indexed_records": None, "semantic_reviewed_records": 0,
            "semantic_review": "NOT_REVIEWED", "source_http_requests": 0, "inference_calls": 0,
            "absolute_aws_completeness_claimed": False,
            "notice": "URL families are not a count of distinct AWS services. Validated bytes are not proof of semantic quality, freshness, serving-index membership, or sufficient coverage.",
        }
    finally:
        if manifest is not None:
            manifest.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
