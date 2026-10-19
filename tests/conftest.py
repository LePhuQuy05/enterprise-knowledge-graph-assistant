import pytest

from app.fixtures import FIXTURE_EMBEDDING_MODEL, FIXTURE_TENANT, build_fixture
from app.snapshot import load_snapshot


@pytest.fixture
def snapshot_dir(tmp_path):
    return build_fixture(tmp_path / "snapshots")


@pytest.fixture
def loaded(snapshot_dir):
    return load_snapshot(
        snapshot_dir, tenant_id=FIXTURE_TENANT,
        embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True,
    )
