"""Export local schemas for the new, explicitly versioned method boundaries."""

from app.collection_graph_v2 import CollectionGraphOverlayV2
from app.contracts import canonical_json
from app.method_execution import MethodExecutionV2
from app.query_planning_v2 import ValidatedQueryPlanV2
from app.query_rewrite_v2 import ValidatedRewriteV2
from scripts.run_collection_benchmark import ROOT


def main():
    directory = ROOT / "schemas/methods"
    directory.mkdir(exist_ok=True)
    for model in (MethodExecutionV2, ValidatedQueryPlanV2, ValidatedRewriteV2, CollectionGraphOverlayV2):
        (directory / (model.__name__ + ".schema.json")).write_bytes(canonical_json(model.model_json_schema()) + b"\n")
    print("4 strict-method schemas exported")


if __name__ == "__main__":
    main()
