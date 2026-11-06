"""Serve only an isolated local fixture preview; never the repo or credentials."""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.tracer import HEADERS
from scripts.preview_web_upgrade import build_preview

ROOT = Path(__file__).resolve().parents[1]


def build(target, fixture):
    if not fixture.is_file():
        raise ValueError("SCOPED_UI_FIXTURE_REQUIRED")
    if target.exists():
        raise ValueError("SCOPED_UI_PREVIEW_NEW_DIRECTORY_REQUIRED")
    build_preview(target)
    (target/"base-preview.mjs").write_bytes((ROOT/"tests/fixtures/web_upgrade/preview.mjs").read_bytes())
    (target/"preview.mjs").write_bytes((ROOT/"tests/fixtures/scoped_runs/preview.mjs").read_bytes())
    (target/"scoped-fixture.json").write_bytes(fixture.read_bytes())
    (target/"auth-start.mjs").write_bytes((ROOT/"tests/fixtures/scoped_runs/auth-start.mjs").read_bytes())
    (target/"auth-start.html").write_text((ROOT/"frontend/tracer/index.html").read_text(encoding="utf-8").replace(
        'src="/assets/app.js"', 'src="/auth-start.mjs"'), encoding="utf-8")
    return {"status": "LOCAL_SCOPED319_PREVIEW", "directory": str(target), "provider_calls": 0, "aws_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/build/scoped319-preview-v1")
    parser.add_argument("--fixture", type=Path, default=ROOT/"results/scoped-ui319-2026-11-05-v1/fixture-build-v4/fixture.json")
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.fixture)), flush=True)
    if args.serve:
        class Handler(SimpleHTTPRequestHandler):
            def end_headers(self):
                for key, value in HEADERS.items():
                    self.send_header(key, value)
                super().end_headers()

            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(args.output.resolve())))
        print(f"Fixture preview http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
