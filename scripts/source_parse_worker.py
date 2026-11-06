"""Isolated spaCy process: exact token annotations only, not graph fact extraction."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if len(raw) > 32000000:
        raise ValueError("SOURCE_PARSE_INPUT_BOUND")
    request = json.loads(raw)
    texts = request["texts"]
    if len(texts) > 4000 or any(type(t) is not str or not 1 <= len(t) <= 20000 for t in texts):
        raise ValueError("SOURCE_PARSE_TEXT_BOUND")
    if importlib.metadata.version("spacy") != "3.8.18" or importlib.metadata.version("en_core_web_sm") != "3.8.0":
        raise ValueError("SOURCE_PARSE_PINNED_VERSIONS_REQUIRED")
    import spacy

    started = time.perf_counter()
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    load_seconds = time.perf_counter() - started
    if args.output.exists():
        raise ValueError("SOURCE_PARSE_OUTPUT_ALREADY_EXISTS")
    started, documents, count = time.perf_counter(), [], 0
    handle = args.output.open("x", encoding="utf-8") if args.jsonl else None
    if handle:
        handle.write(json.dumps({"header": {"version": "source-token-annotations-jsonl-v1", "installation": request["installation"],
            "spacy_version": spacy.__version__, "model_version": nlp.meta["version"], "expected_texts": len(texts)}}, sort_keys=True) + "\n")
    for doc in nlp.pipe(texts, batch_size=32):
        data = {"text": doc.text, "tokens": [{"i": t.i, "start": t.idx, "end": t.idx + len(t), "text": t.text,
            "pos": t.pos_, "tag": t.tag_, "lemma": t.lemma_, "dep": t.dep_, "head": t.head.i} for t in doc],
            "sentences": [{"start": s.start_char, "end": s.end_char} for s in doc.sents]}
        count += 1
        if handle:
            handle.write(json.dumps({"document": data}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            documents.append(data)
    result = {"version": "source-token-annotations-v1", "installation": request["installation"], "documents": documents,
        "load_seconds": load_seconds, "parse_seconds": time.perf_counter() - started,
        "spacy_version": spacy.__version__, "model_version": nlp.meta["version"], "semantic_assertions_emitted": 0}
    metrics = {"parsed_texts": count, "load_seconds": result["load_seconds"], "parse_seconds": result["parse_seconds"]}
    if handle:
        handle.write(json.dumps({"complete": metrics}, sort_keys=True) + "\n")
        handle.close()
    else:
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
