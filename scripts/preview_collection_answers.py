"""Serve only an explicit local UI fixture; no credentials or production data."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.preview_owned_runs import ROOT, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8870)
    args = parser.parse_args()
    target = args.output.resolve()
    build(target)
    (target / "preview.mjs").rename(target / "retrieval-preview.mjs")
    (target / "preview.mjs").write_bytes((ROOT / "tests/fixtures/collection_answers/preview.mjs").read_bytes())
    (target / "answer-fixture.json").write_bytes((ROOT / "tests/fixtures/owned_runs/answer.json").read_bytes())
    from app.tracer import HEADERS

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            for k, v in HEADERS.items():
                self.send_header(k, v)
            super().end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(target)))
    print("Synthetic recorded-answer preview: " + str(server.server_address), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
