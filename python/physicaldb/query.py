from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

CHANNELS = {
    "pos_xyz": (0, 1, 2),
    "rot_wxyz": (3, 4, 5, 6),
    "vel_xyz": (7, 8, 9),
}


class EgressLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryResult:
    tensor: object
    timestamps_ns: np.ndarray
    row_groups: tuple[int, ...]
    file_uri: str
    selected_bytes: int
    output: str


def query(
    *,
    catalog: str | os.PathLike[str],
    robot_id: str | None = None,
    start_ts_ns: int | None = None,
    end_ts_ns: int | None = None,
    bbox: Sequence[float] | None = None,
    min_velocity: float | None = None,
    channels: Sequence[str] = ("pos_xyz", "rot_wxyz", "vel_xyz"),
    target_hz: float = 30.0,
    output: Literal["numpy", "torch"] = "numpy",
    source: str | os.PathLike[str] | None = None,
    max_egress_bytes: int = 1_000_000_000,
    limit: int | None = None,
    robotics_bin: str | os.PathLike[str] | None = None,
) -> QueryResult:
    """Query the hot catalog with DuckDB and return a training-shaped tensor.

    The current bridge materializes selected Parquet row groups through the Rust
    CLI and loads `.npy` files into NumPy/PyTorch. It preserves the product API
    shape while DLPack/zero-copy Arrow interop is still pending.
    """

    if output not in {"numpy", "torch"}:
        raise ValueError("output must be 'numpy' or 'torch'")
    channel_indices = _channel_indices(channels)
    rows = _duckdb_query_catalog(
        Path(catalog),
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        bbox=bbox,
        min_velocity=min_velocity,
        limit=limit,
    )
    if not rows:
        raise LookupError("query matched no catalog row groups")

    selected_bytes = sum(int(row["byte_length"]) for row in rows)
    if selected_bytes > max_egress_bytes:
        raise EgressLimitError(
            f"query selected {selected_bytes} bytes, exceeding max_egress_bytes={max_egress_bytes}"
        )

    file_uris = {str(row["file_uri"]) for row in rows}
    if len(file_uris) != 1 and source is None:
        raise ValueError("query matched multiple file_uri values; pass source=... for this prototype")
    file_uri = str(source) if source is not None else next(iter(file_uris))
    if file_uri.startswith(("s3://", "s3a://")):
        raise ValueError("tensor materialization currently requires a local Parquet source path")

    query_start_ns = (
        int(start_ts_ns) if start_ts_ns is not None else min(int(row["start_ts_ns"]) for row in rows)
    )
    query_end_ns = (
        int(end_ts_ns) if end_ts_ns is not None else max(int(row["end_ts_ns"]) for row in rows)
    )
    row_groups = tuple(sorted({int(row["row_group_id"]) for row in rows}))

    values, timestamps = _materialize_with_rust(
        source=Path(file_uri),
        row_groups=row_groups,
        start_ts_ns=query_start_ns,
        end_ts_ns=query_end_ns,
        target_hz=target_hz,
        robotics_bin=robotics_bin,
    )
    values = values[:, channel_indices]
    tensor: object
    if output == "torch":
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("output='torch' requires the torch package") from exc
        tensor = torch.from_numpy(values)
    else:
        tensor = values

    return QueryResult(
        tensor=tensor,
        timestamps_ns=timestamps,
        row_groups=row_groups,
        file_uri=file_uri,
        selected_bytes=selected_bytes,
        output=output,
    )


def _duckdb_query_catalog(
    catalog: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    bbox: Sequence[float] | None,
    min_velocity: float | None,
    limit: int | None,
) -> list[dict[str, object]]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates = []
    params: list[object] = [str(catalog)]
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    if start_ts_ns is not None:
        predicates.append("end_ts_ns >= ?")
        params.append(int(start_ts_ns))
    if end_ts_ns is not None:
        predicates.append("start_ts_ns <= ?")
        params.append(int(end_ts_ns))
    if bbox is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox(bbox)
        predicates.extend(
            [
                "min_x <= ?",
                "max_x >= ?",
                "min_y <= ?",
                "max_y >= ?",
                "min_z <= ?",
                "max_z >= ?",
            ]
        )
        params.extend([max_x, min_x, max_y, min_y, max_z, min_z])
    if min_velocity is not None:
        predicates.append("max_velocity >= ?")
        params.append(float(min_velocity))

    where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))

    sql = f"""
        SELECT
            robot_id, session_id, file_uri, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length
        FROM read_parquet(?)
        {where}
        ORDER BY start_ts_ns, row_group_id
        {limit_sql}
    """
    with duckdb.connect(":memory:") as con:
        columns = [column[0] for column in con.execute(sql, params).description]
        return [dict(zip(columns, row)) for row in con.fetchall()]


def _materialize_with_rust(
    *,
    source: Path,
    row_groups: Sequence[int],
    start_ts_ns: int,
    end_ts_ns: int,
    target_hz: float,
    robotics_bin: str | os.PathLike[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="physicaldb_") as tmp:
        prefix = Path(tmp) / "tensor"
        cmd = _robotics_command(robotics_bin) + [
            "tensor",
            "parquet-row-groups",
            "--input",
            str(source),
            "--row-groups",
            ",".join(str(row_group) for row_group in row_groups),
            "--start-ts-ns",
            str(start_ts_ns),
            "--end-ts-ns",
            str(end_ts_ns),
            "--hz",
            str(target_hz),
            "--out",
            str(prefix),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        values = np.load(prefix.with_name(f"{prefix.name}.values.npy"))
        timestamps = np.load(prefix.with_name(f"{prefix.name}.timestamps_ns.npy"))
    return values, timestamps


def _robotics_command(robotics_bin: str | os.PathLike[str] | None) -> list[str]:
    if robotics_bin is not None:
        return [str(robotics_bin)]
    root = Path(__file__).resolve().parents[2]
    binary = root / "target" / "debug" / "robotics"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "-p", "robotics-cli", "--"]


def _channel_indices(channels: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for channel in channels:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; expected one of {sorted(CHANNELS)}")
        indices.extend(CHANNELS[channel])
    return indices


def _bbox(values: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ValueError("bbox must contain min_x,max_x,min_y,max_y,min_z,max_z")
    return tuple(float(value) for value in values)  # type: ignore[return-value]
