from __future__ import annotations

import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local tmp_path replacement for restricted Windows environments."""

    base = Path("outputs") / "test_tmp"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    return path
