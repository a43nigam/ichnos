import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from physicaldb import EgressLimitError, query


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("duckdb") is None,
    reason="duckdb is not installed",
)


def test_query_returns_numpy_tensor(tmp_path: Path) -> None:
    source = tmp_path / "session.parquet"
    catalog = tmp_path / "catalog.parquet"
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "ingest",
            "synthetic-parquet",
            "--out",
            str(source),
            "--row-group-rows",
            "25",
            "--hz",
            "50",
            "--duration-ns",
            "1000000000",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "catalog",
            "build",
            "--input",
            str(source),
            "--out",
            str(catalog),
        ],
        check=True,
    )

    result = query(
        catalog=catalog,
        robot_id="humanoid_01",
        start_ts_ns=0,
        end_ts_ns=480_000_000,
        bbox=(-0.1, 2.0, -1.1, 1.1, -0.1, 0.1),
        min_velocity=2.0,
        channels=("pos_xyz", "rot_wxyz"),
        target_hz=30.0,
        output="numpy",
        source=source,
    )

    assert isinstance(result.tensor, np.ndarray)
    assert result.tensor.shape == (15, 7)
    assert result.timestamps_ns.shape == (15,)
    assert result.selected_bytes > 0
    assert result.row_groups == (0,)


def test_query_blocks_before_materialization_when_egress_limit_is_too_low(tmp_path: Path) -> None:
    source = tmp_path / "session.parquet"
    catalog = tmp_path / "catalog.parquet"
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "ingest",
            "synthetic-parquet",
            "--out",
            str(source),
            "--row-group-rows",
            "25",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "catalog",
            "build",
            "--input",
            str(source),
            "--out",
            str(catalog),
        ],
        check=True,
    )

    with pytest.raises(EgressLimitError):
        query(catalog=catalog, source=source, max_egress_bytes=1)
