"""Local ingestion recipes and source metadata; serving schemas remain in contracts."""

import re
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from app.contracts import Nonempty, Record, canonical_json, canonical_uri, sha256_bytes

SOURCE_PREFIXES = (
    "/lambda/latest/dg/", "/AmazonS3/latest/userguide/",
    "/IAM/latest/UserGuide/", "/kms/latest/developerguide/",
)


def source_uri(value: str) -> str:
    value = canonical_uri(value)
    parsed = urlsplit(value)
    if parsed.hostname != "docs.aws.amazon.com" or parsed.port not in (None, 443):
        raise ValueError("SOURCE_HOST_NOT_ALLOWED")
    if not parsed.path.startswith(SOURCE_PREFIXES) or not parsed.path.endswith(".html"):
        raise ValueError("SOURCE_PATH_NOT_ALLOWED")
    if "%" in parsed.path or "/../" in parsed.path or "/./" in parsed.path:
        raise ValueError("SOURCE_PATH_NOT_ALLOWED")
    return value


class SourcePage(Record):
    source_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
    service: Literal["Lambda", "S3", "IAM", "KMS"]
    url: Nonempty
    expected_title: Nonempty
    evidence_intent: tuple[Literal["E1", "E2", "E3", "GENERAL"], ...]
    selection_reason: Nonempty
    usage_note: Nonempty

    _url = field_validator("url")(source_uri)


class SourceManifest(Record):
    version: Nonempty
    language: Literal["en"]
    sources: Annotated[tuple[SourcePage, ...], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        for attr in ("url", "source_id"):
            if len({getattr(s, attr) for s in self.sources}) != len(self.sources):
                raise ValueError("DUPLICATE_MANIFEST_SOURCE")
        return self


class ChunkRecipe(Record):
    algorithm: Literal["section-contiguous-v1"] = "section-contiguous-v1"
    target_tokens: Annotated[int, Field(ge=16, le=768)] = 512
    max_tokens: Annotated[int, Field(ge=16, le=768)] = 768
    overlap_tokens: Annotated[int, Field(ge=0, le=64)] = 64

    @model_validator(mode="after")
    def sensible_bounds(self) -> Self:
        if not self.overlap_tokens < self.target_tokens <= self.max_tokens:
            raise ValueError("INVALID_CHUNK_BOUNDS")
        return self

    def version(self, tokenizer_version: str) -> str:
        return "chunk-" + sha256_bytes(canonical_json({
            **self.model_dump(), "tokenizer_version": tokenizer_version,
        }))


class Section(Record):
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]
    section_path: tuple[Nonempty, ...]
    anchor: str | None = None

    @model_validator(mode="after")
    def positive_span(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("EMPTY_SECTION")
        if self.anchor is not None and re.search(r"[\x00-\x20]", self.anchor):
            raise ValueError("INVALID_SECTION_ANCHOR")
        return self
