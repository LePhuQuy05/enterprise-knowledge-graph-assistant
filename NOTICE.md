# Third-party material and release scope

The repository's MIT license covers original project code and original documentation only. It does not relicense third-party packages, model artifacts, AWS documentation, trademarks or service outputs.

- AWS source pages remain attributed to their original URLs. Raw/normalized documents, full corpus snapshots and downloaded archives are excluded from the recruiter evidence ZIP. No blanket redistribution permission is asserted.
- The dashboard projection `frontend/tracer/dashboard-data.json` contains attributed AWS source excerpts and recorded development annotations. Original project code's MIT license does not relicense those source excerpts. It is a reviewed public UI projection, not a full corpus dump or a final recruiter evidence packet.
- BAAI/bge-small-en-v1.5 is pinned to revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; its [model card](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md) declares MIT. Model weights and tokenizer assets are downloaded separately and are not included in the evidence ZIP; preserve upstream notices when distributing them.
- Python/npm dependencies and bundled native components retain their upstream terms. Lockfiles identify versions; installed metadata inventory is an aid to review, not a complete license-compliance analysis. The npm CDK package declares Apache-2.0.
- The locally downloaded Gitleaks tool is not redistributed in the candidate ZIP. Its upstream license and verified download metadata remain local.

The owner selected MIT for original code on October 18, 2026. Primary references checked that date: [MIT license text](https://opensource.org/license/mit), [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md).
