"""Serve an isolated synthetic account UI; never contact Cognito, AWS or model APIs."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from app.tracer import HEADERS
from scripts.preview_web_upgrade import ROOT, build_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    target = ROOT / "data/build/account-preview"
    # Rebuild in a fresh dedicated directory. build_preview deliberately refuses unexpected files.
    if not (target / "account-preview.mjs").exists():
        build_preview(target)
    else:
        allowed = {"index.html", "preview.mjs", "fixtures.mjs", "account-preview.mjs"}
        if target.is_symlink() or any(p.is_symlink() or (
            p.is_file() and p.relative_to(target).parts[0] != "assets" and p.name not in allowed
        ) for p in target.rglob("*")):
            raise ValueError("UNEXPECTED_PREVIEW_CONTENT")
        for file in (ROOT / "frontend/tracer").iterdir():
            if file.name in {"style.css", "app.js", "auth.mjs", "ui.mjs", "client.mjs"}:
                (target / "assets" / file.name).write_bytes(file.read_bytes())
    html = (ROOT / "frontend/tracer/index.html").read_text(encoding="utf-8").replace(
        'src="/assets/app.js"', 'src="/account-preview.mjs"')
    (target / "index.html").write_text(html, encoding="utf-8")
    (target / "account-preview.mjs").write_bytes((ROOT / "tests/fixtures/account-preview.mjs").read_bytes())
    if args.build_only:
        print("Synthetic account preview rebuilt locally; no AWS/model calls")
        return

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            for key, value in HEADERS.items():
                self.send_header(key, value)
            super().end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(target.resolve())))
    print(f"Synthetic account preview: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
