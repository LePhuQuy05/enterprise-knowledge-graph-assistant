"""Closed local UI fixture with recorded public retrieval; no live AWS or private credentials."""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.tracer import HEADERS
from scripts.preview_web_upgrade import ROOT, build_preview


def build(target):
    if target.exists():
        raise ValueError("OWNED_PREVIEW_EXISTS_CHOOSE_FRESH_DIRECTORY")
    report = build_preview(target)
    base = target / "preview.mjs"
    base.rename(target / "base-preview.mjs")
    base.write_bytes((ROOT / "tests/fixtures/owned_runs/preview.mjs").read_bytes())
    receipt = json.loads((ROOT / "results/owned-runs/retrieval-ldh-v1.json").read_bytes())
    assets = json.loads((ROOT / "results/workbench-cloud/assets-build-v1.json").read_bytes())
    directory = Path(assets["bundle_directory"])
    docs = {}
    for path in (directory / "collection").glob("*/documents.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            document = json.loads(line)
            docs[document["document_version_id"]] = document
    sources = {}
    for result in receipt["results"]:
        for source in result["context_sources"]:
            document = docs[source["document_version_id"]]
            text = document["normalized_text"][source["start_char"]:source["end_char"]]
            if sha256_bytes(text.encode()) != source["text_sha256"]:
                raise ValueError("PREVIEW_SOURCE_PIN_MISMATCH")
            sources[result["method_id"] + ":" + source["chunk_id"]] = source | {"text": text}
    data = {"kind": "RECORDED_PUBLIC_RETRIEVAL_WITH_SYNTHETIC_ASYNC_AUTH", "run": receipt["run"],
        "results": receipt["results"], "sources": sources,
        "corpus": {k: assets[k] for k in ("documents", "chunks", "service_groups")}}
    (target / "owned-fixture.json").write_bytes(canonical_json(data) + b"\n")
    report.update(kind=data["kind"], sources=len(sources), aws_calls=0, model_calls=0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8865)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve())), flush=True)
    if args.serve:
        class Handler(SimpleHTTPRequestHandler):
            def end_headers(self):
                for name, value in HEADERS.items():
                    self.send_header(name, value)
                super().end_headers()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(args.output.resolve())))
        print("Owned workflow synthetic UI preview: " + str(server.server_address), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
