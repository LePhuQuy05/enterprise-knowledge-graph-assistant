"""Export V5 authoring schemas, without claiming serving approval."""

from pathlib import Path

from app.contracts import canonical_json
from app.semantic_extraction import PermissionRequirement, SemanticExtractionOutputV5, TriggerFrame


def main():
    directory = Path(__file__).resolve().parents[1] / "schemas/extraction"
    directory.mkdir(exist_ok=True)
    for name, model in (("SemanticExtractionOutput.v5", SemanticExtractionOutputV5),
            ("TriggerFrame.v5", TriggerFrame), ("PermissionRequirement.v5", PermissionRequirement)):
        (directory / (name + ".schema.json")).write_bytes(canonical_json(model.model_json_schema()) + b"\n")
    print("Exported 3 V5 authoring schemas; serving remains gated")


if __name__ == "__main__":
    main()
