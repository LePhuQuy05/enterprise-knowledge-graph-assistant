"""Build an isolated, synthetic preview. Never copy credentials, raw results, or serve the repo."""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("style.css", "app.js", "auth.mjs", "client.mjs", "ui.mjs", "investigation.mjs", "evidence.mjs", "brief.mjs",
          "dashboard.mjs", "onboarding.mjs", "dashboard-data.json")


def build_preview(target: Path):
    target = target.resolve()
    allowed_root = (ROOT / "data/build").resolve()
    if not target.is_relative_to(allowed_root) or target == allowed_root:
        raise ValueError("PREVIEW_MUST_STAY_IN_DEDICATED_BUILD_DIRECTORY")
    files = {"index.html": (ROOT / "frontend/tracer/index.html").read_text(encoding="utf-8").replace(
        'src="/assets/app.js"', 'src="/preview.mjs"')}
    files.update({"assets/" + name: (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8") for name in ASSETS})
    files.update({name: (ROOT / "tests/fixtures/web_upgrade" / source).read_text(encoding="utf-8")
                  for name, source in (("fixtures.mjs", "records.mjs"), ("preview.mjs", "preview.mjs"))})
    if target.exists():
        for existing in target.rglob("*"):
            if existing.is_symlink() or (existing.is_file() and existing.relative_to(target).as_posix() not in files):
                raise ValueError("PREVIEW_DIRECTORY_HAS_UNEXPECTED_CONTENT")
    for relative, content in files.items():
        path = target / relative
        if not path.resolve().is_relative_to(target):
            raise ValueError("UNSAFE_PREVIEW_PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return {"status": "SYNTHETIC_PREVIEW_BUILT", "directory": target.relative_to(ROOT).as_posix(),
            "files": sorted(files), "aws_calls": 0, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/build/web-upgrade-preview")
    parser.add_argument("--serve", action="store_true", help="Serve only the synthetic build on localhost with production security headers")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(json.dumps(build_preview(args.output), indent=2))
    if args.serve:
        from app.tracer import HEADERS

        class PreviewHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                for name, value in HEADERS.items():
                    self.send_header(name, value)
                super().end_headers()

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(PreviewHandler, directory=str(args.output.resolve())))
        print(f"Synthetic preview: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
