#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb import EgressLimitError, plan_batch, query_batch

DEFAULT_SEQUENCES = ("V1_01_easy", "V1_02_medium", "V1_03_difficult")
CHANNELS = ("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate EuRoC Vicon Room 1 easy/medium/difficult sessions through plan_batch/query_batch."
    )
    parser.add_argument("--input-root", type=Path, default=Path("vicon_room1"))
    parser.add_argument("--sequence", action="append", default=[], help="Override sequence as NAME=PATH. Repeat three times.")
    parser.add_argument("--output-root", type=Path, default=Path("data/validation/euroc_vicon_room1"))
    parser.add_argument("--robot-id", default="mav0")
    parser.add_argument("--stream-id", default="cam0")
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-egress-bytes", type=int, default=1_000_000_000_000)
    parser.add_argument("--footer-allowance-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="Optional s3://bucket/prefix for validating through S3-compatible object-store URIs.",
    )
    parser.add_argument("--pose-row-group-rows", type=int, default=500)
    parser.add_argument("--imu-row-group-rows", type=int, default=2000)
    parser.add_argument("--camera-row-group-rows", type=int, default=20)
    parser.add_argument("--full-catalog", action="store_true", help="Combine all row groups instead of one representative pose window per session.")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if importlib.util.find_spec("duckdb") is None:
        print("SKIP: duckdb Python package is required for EuRoC Vicon Room 1 validation")
        return 0
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    report = run_validation(args)
    report_path = args.output_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(report_path, report)
    return 0


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sequences = discover_sequences(args.input_root, args.sequence)
    if len(sequences) != 3:
        raise SystemExit(f"expected exactly three sequences, found {len(sequences)}")
    s3_prefix = normalize_s3_prefix(args.s3_prefix) if args.s3_prefix else ""
    if s3_prefix:
        require_s3_env()

    per_sequence: list[dict[str, Any]] = []
    for session_id, input_path in sequences:
        paths = SequencePaths(args.output_root / session_id, args.stream_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        ingest = build_or_reuse_sequence(args, session_id, input_path, paths)
        object_store = upload_sequence_artifacts(paths, s3_prefix, session_id) if s3_prefix else {}
        catalogs = build_or_reuse_catalogs(args, paths, object_store)
        per_sequence.append(
            {
                "session_id": session_id,
                "input": str(input_path),
                "paths": paths_payload(paths),
                "object_store": object_store,
                "ingest": ingest,
                "catalogs": catalogs,
                "monotonic_sources": source_monotonicity(paths),
            }
        )

    combined = CombinedPaths(args.output_root, args.stream_id)
    selected_windows = combine_catalogs(
        combined,
        [SequencePaths(args.output_root / session_id, args.stream_id) for session_id, _ in sequences],
        representative=not args.full_catalog,
    )
    run_robotics(
        "catalog",
        "duckdb-build",
        "--pose-catalog",
        str(combined.pose_catalog),
        "--imu-catalog",
        str(combined.imu_catalog),
        "--media-catalog",
        str(combined.media_catalog),
        "--spatial-index",
        "hilbert",
        "--out",
        str(combined.catalog_db),
    )
    combined_object_store = upload_combined_artifacts(combined, s3_prefix) if s3_prefix else {}

    predicate = choose_batch_predicate(combined.pose_catalog, args.robot_id)
    timings_ms: list[float] = []
    last_plan = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        last_plan = plan_batch(
            catalog_db=combined.catalog_db,
            robot_id=args.robot_id,
            predicate=predicate,
            channels=CHANNELS,
            max_egress_bytes=args.max_egress_bytes,
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    assert last_plan is not None
    validate_plan(last_plan, [session_id for session_id, _ in sequences])

    blocked_media_out = args.output_root / "blocked_media"
    egress_probe = plan_batch(
        catalog_db=combined.catalog_db,
        robot_id=args.robot_id,
        predicate=predicate,
        channels=CHANNELS,
        max_egress_bytes=last_plan.materialized_pose_imu_bytes,
    )
    if not egress_probe.blocked_by_egress or not egress_probe.diagnostics.media_blocked_by_egress:
        raise AssertionError("low egress budget did not block media before materialization")
    try:
        query_batch(
            catalog_db=combined.catalog_db,
            robot_id=args.robot_id,
            predicate=predicate,
            channels=CHANNELS,
            target_hz=args.target_hz,
            max_egress_bytes=last_plan.materialized_pose_imu_bytes,
            materialize_media=True,
            media_out=blocked_media_out,
            robotics_bin=args.output_root / "missing_robotics_binary",
        )
    except EgressLimitError:
        pass
    else:
        raise AssertionError("query_batch did not raise EgressLimitError for low media egress budget")
    if blocked_media_out.exists():
        raise AssertionError("blocked egress probe created media output")

    media_out = args.output_root / "media_materialized"
    manifest_out = args.output_root / "batch_manifest.json"
    result = query_batch(
        catalog_db=combined.catalog_db,
        robot_id=args.robot_id,
        predicate=predicate,
        channels=CHANNELS,
        target_hz=args.target_hz,
        max_egress_bytes=args.max_egress_bytes,
        materialize_media=True,
        media_out=media_out,
        enforce_ranges=True,
        footer_allowance_bytes=args.footer_allowance_bytes,
        manifest_out=manifest_out,
    )
    window_reports = validate_results(result, last_plan, args.target_hz)

    return {
        "version": 1,
        "storage_mode": "s3" if s3_prefix else "local",
        "s3_prefix": s3_prefix,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "robot_id": args.robot_id,
        "stream_id": args.stream_id,
        "representative_combined_catalog": not args.full_catalog,
        "sequences": per_sequence,
        "combined": {
            "paths": {
                "pose_catalog": str(combined.pose_catalog),
                "imu_catalog": str(combined.imu_catalog),
                "media_catalog": str(combined.media_catalog),
                "catalog_db": str(combined.catalog_db),
                "batch_manifest": str(manifest_out),
            },
            "selected_windows": selected_windows,
            "catalogs": combined_catalog_stats(combined),
            "object_store": combined_object_store,
        },
        "query": {
            "predicate": predicate,
            "channels": list(CHANNELS),
            "target_hz": args.target_hz,
            "planning": {
                "iterations": args.iterations,
                "p50_ms": percentile(timings_ms, 0.50),
                "p95_ms": percentile(timings_ms, 0.95),
                "mean_ms": statistics.fmean(timings_ms),
                "samples_ms": timings_ms,
            },
            "plan": batch_plan_payload(last_plan),
            "egress_probe": {
                "max_egress_bytes": egress_probe.egress_limit_bytes,
                "blocked_by_egress": egress_probe.blocked_by_egress,
                "media_blocked_by_egress": egress_probe.diagnostics.media_blocked_by_egress,
                "blocked_media_out_created": blocked_media_out.exists(),
            },
            "materialization": {
                "window_count": len(result.windows),
                "media_out": str(media_out),
                "diagnostics": asdict(result.diagnostics),
                "windows": window_reports,
            },
        },
    }


class SequencePaths:
    def __init__(self, root: Path, stream_id: str) -> None:
        self.root = root
        self.pose = root / "pose.parquet"
        self.imu = root / "imu.parquet"
        self.camera = root / f"{stream_id}.parquet"
        self.pose_catalog = root / "pose_catalog.parquet"
        self.imu_catalog = root / "imu_catalog.parquet"
        self.media_catalog = root / "media_catalog.parquet"


class CombinedPaths:
    def __init__(self, root: Path, stream_id: str) -> None:
        self.pose_catalog = root / "combined_pose_catalog.parquet"
        self.imu_catalog = root / "combined_imu_catalog.parquet"
        self.media_catalog = root / "combined_media_catalog.parquet"
        self.catalog_db = root / "fleet.duckdb"
        self.stream_id = stream_id


def discover_sequences(input_root: Path, sequence_args: Sequence[str]) -> list[tuple[str, Path]]:
    if sequence_args:
        sequences = []
        for item in sequence_args:
            name, sep, raw_path = item.partition("=")
            if not sep or not name or not raw_path:
                raise SystemExit("--sequence must use NAME=PATH")
            sequences.append((name, Path(raw_path)))
        return sequences
    return [
        (name, input_root / name / f"{name}.zip")
        for name in DEFAULT_SEQUENCES
    ]


def build_or_reuse_sequence(
    args: argparse.Namespace,
    session_id: str,
    input_path: Path,
    paths: SequencePaths,
) -> dict[str, Any]:
    if not input_path.exists():
        raise SystemExit(f"missing EuRoC input for {session_id}: {input_path}")
    outputs: dict[str, Any] = {}
    commands = [
        (
            "pose",
            paths.pose,
            (
                "ingest",
                "euroc-groundtruth",
                "--input",
                str(input_path),
                "--out",
                str(paths.pose),
                "--robot-id",
                args.robot_id,
                "--session-id",
                session_id,
                "--row-group-rows",
                str(args.pose_row_group_rows),
            ),
        ),
        (
            "imu",
            paths.imu,
            (
                "ingest",
                "euroc-imu",
                "--input",
                str(input_path),
                "--out",
                str(paths.imu),
                "--robot-id",
                args.robot_id,
                "--session-id",
                session_id,
                "--row-group-rows",
                str(args.imu_row_group_rows),
            ),
        ),
        (
            "camera",
            paths.camera,
            (
                "ingest",
                "euroc-camera",
                "--input",
                str(input_path),
                "--out",
                str(paths.camera),
                "--stream-id",
                args.stream_id,
                "--robot-id",
                args.robot_id,
                "--session-id",
                session_id,
                "--row-group-rows",
                str(args.camera_row_group_rows),
            ),
        ),
    ]
    for label, output, command in commands:
        if args.rebuild or not output.exists():
            completed = run_robotics(*command)
            outputs[label] = parse_cli_metrics(completed.stdout)
        else:
            outputs[label] = parquet_stats(output)
    return outputs


def build_or_reuse_catalogs(args: argparse.Namespace, paths: SequencePaths, object_store: dict[str, str]) -> dict[str, Any]:
    pose_uri = object_store.get("pose_uri", paths.pose.resolve().as_uri())
    imu_uri = object_store.get("imu_uri", paths.imu.resolve().as_uri())
    media_uri = object_store.get("camera_uri", paths.camera.resolve().as_uri())
    commands = [
        (
            "pose",
            paths.pose_catalog,
            (
                "catalog",
                "build",
                "--input",
                str(paths.pose),
                "--out",
                str(paths.pose_catalog),
                "--uri",
                pose_uri,
            ),
        ),
        (
            "imu",
            paths.imu_catalog,
            (
                "catalog",
                "build-imu",
                "--input",
                str(paths.imu),
                "--out",
                str(paths.imu_catalog),
                "--uri",
                imu_uri,
            ),
        ),
        (
            "media",
            paths.media_catalog,
            (
                "catalog",
                "build-media",
                "--input",
                str(paths.camera),
                "--out",
                str(paths.media_catalog),
                "--modality",
                "camera",
                "--stream-id",
                args.stream_id,
                "--uri",
                media_uri,
            ),
        ),
    ]
    outputs: dict[str, Any] = {}
    for label, output, command in commands:
        if args.rebuild or object_store or not output.exists():
            completed = run_robotics(*command)
            outputs[label] = parse_cli_metrics(completed.stdout)
        else:
            outputs[label] = catalog_stats(output)
        outputs[label]["path"] = str(output)
    return outputs


def combine_catalogs(combined: CombinedPaths, paths: Sequence[SequencePaths], *, representative: bool) -> list[dict[str, Any]]:
    import duckdb

    combined.pose_catalog.parent.mkdir(parents=True, exist_ok=True)
    selected = select_representative_windows(paths) if representative else select_full_windows(paths)
    with duckdb.connect(":memory:") as con:
        copy_union(con, combined.pose_catalog, [
            f"SELECT * FROM read_parquet('{quote_sql(row['pose_catalog'])}') "
            f"WHERE session_id = '{quote_sql(row['session_id'])}' AND row_group_id = {int(row['pose_row_group_id'])}"
            if representative
            else f"SELECT * FROM read_parquet('{quote_sql(row['pose_catalog'])}')"
            for row in selected
        ])
        copy_union(con, combined.imu_catalog, [
            f"SELECT * FROM read_parquet('{quote_sql(row['imu_catalog'])}') "
            f"WHERE session_id = '{quote_sql(row['session_id'])}' "
            f"AND end_ts_ns >= {int(row['start_ts_ns'])} AND start_ts_ns <= {int(row['end_ts_ns'])}"
            if representative
            else f"SELECT * FROM read_parquet('{quote_sql(row['imu_catalog'])}')"
            for row in selected
        ])
        copy_union(con, combined.media_catalog, [
            f"SELECT * FROM read_parquet('{quote_sql(row['media_catalog'])}') "
            f"WHERE session_id = '{quote_sql(row['session_id'])}' "
            f"AND end_ts_ns >= {int(row['start_ts_ns'])} AND start_ts_ns <= {int(row['end_ts_ns'])}"
            if representative
            else f"SELECT * FROM read_parquet('{quote_sql(row['media_catalog'])}')"
            for row in selected
        ])
    return selected


def select_representative_windows(paths: Sequence[SequencePaths]) -> list[dict[str, Any]]:
    import duckdb

    selected = []
    with duckdb.connect(":memory:") as con:
        for item in paths:
            row = con.execute(
                """
                WITH common AS (
                    SELECT
                        greatest(
                            (SELECT min(start_ts_ns) FROM read_parquet(?) ),
                            (SELECT min(start_ts_ns) FROM read_parquet(?) ),
                            (SELECT min(start_ts_ns) FROM read_parquet(?) )
                        ) AS start_ts_ns,
                        least(
                            (SELECT max(end_ts_ns) FROM read_parquet(?) ),
                            (SELECT max(end_ts_ns) FROM read_parquet(?) ),
                            (SELECT max(end_ts_ns) FROM read_parquet(?) )
                        ) AS end_ts_ns
                ),
                candidates AS (
                    SELECT p.*, row_number() OVER (ORDER BY p.start_ts_ns, p.row_group_id) AS rn,
                           count(*) OVER () AS n
                    FROM read_parquet(?) p, common c
                    WHERE p.end_ts_ns >= c.start_ts_ns AND p.start_ts_ns <= c.end_ts_ns
                )
                SELECT session_id, row_group_id, start_ts_ns, end_ts_ns, max_velocity
                FROM candidates
                WHERE rn = CAST(ceil(n / 2.0) AS BIGINT)
                """,
                [
                    str(item.pose_catalog),
                    str(item.imu_catalog),
                    str(item.media_catalog),
                    str(item.pose_catalog),
                    str(item.imu_catalog),
                    str(item.media_catalog),
                    str(item.pose_catalog),
                ],
            ).fetchone()
            if row is None:
                raise AssertionError(f"no representative overlap for {item.root.name}")
            selected.append(
                {
                    "session_id": str(row[0]),
                    "pose_row_group_id": int(row[1]),
                    "start_ts_ns": int(row[2]),
                    "end_ts_ns": int(row[3]),
                    "max_velocity": float(row[4]),
                    "pose_catalog": str(item.pose_catalog),
                    "imu_catalog": str(item.imu_catalog),
                    "media_catalog": str(item.media_catalog),
                }
            )
    return selected


def select_full_windows(paths: Sequence[SequencePaths]) -> list[dict[str, Any]]:
    import duckdb

    selected = []
    with duckdb.connect(":memory:") as con:
        for item in paths:
            row = con.execute(
                "SELECT session_id, min(start_ts_ns), max(end_ts_ns), max(max_velocity) FROM read_parquet(?) GROUP BY session_id",
                [str(item.pose_catalog)],
            ).fetchone()
            if row is None:
                raise AssertionError(f"empty pose catalog for {item.root.name}")
            selected.append(
                {
                    "session_id": str(row[0]),
                    "pose_row_group_id": -1,
                    "start_ts_ns": int(row[1]),
                    "end_ts_ns": int(row[2]),
                    "max_velocity": float(row[3]),
                    "pose_catalog": str(item.pose_catalog),
                    "imu_catalog": str(item.imu_catalog),
                    "media_catalog": str(item.media_catalog),
                }
            )
    return selected


def copy_union(con: Any, out: Path, selects: Sequence[str]) -> None:
    if not selects:
        raise AssertionError(f"no selects for {out}")
    sql = " UNION ALL ".join(selects)
    con.execute(f"COPY ({sql}) TO '{quote_sql(str(out))}' (FORMAT PARQUET)")


def choose_batch_predicate(pose_catalog: Path, robot_id: str) -> str:
    import duckdb

    with duckdb.connect(":memory:") as con:
        row = con.execute(
            """
            SELECT min(session_max_velocity)
            FROM (
                SELECT session_id, max(max_velocity) AS session_max_velocity
                FROM read_parquet(?)
                WHERE robot_id = ?
                GROUP BY session_id
            )
            """,
            [str(pose_catalog), robot_id],
        ).fetchone()
    if row is None or row[0] is None:
        raise AssertionError("combined pose catalog has no velocity statistics")
    threshold = max(float(row[0]) * 0.5, 0.0)
    return f"velocity_magnitude >= {threshold:.6f}"


def validate_plan(batch_plan: Any, expected_sessions: Sequence[str]) -> None:
    if len(batch_plan.windows) != len(expected_sessions):
        raise AssertionError(f"plan_batch returned {len(batch_plan.windows)} windows, expected {len(expected_sessions)}")
    sessions = []
    for window in batch_plan.windows:
        window_sessions = {str(row["session_id"]) for row in window._pose_rows}
        if len(window_sessions) != 1:
            raise AssertionError(f"window has non-unique session IDs: {window_sessions}")
        session_id = next(iter(window_sessions))
        sessions.append(session_id)
        if len({span.file_uri for span in window.pose_row_groups}) != 1:
            raise AssertionError(f"{session_id} window has multiple pose file URIs")
        if not window.pose_row_groups or not window.imu_row_groups or not window.media_row_groups:
            raise AssertionError(f"{session_id} window did not select pose, IMU, and media row groups")
    if sorted(sessions) != sorted(expected_sessions):
        raise AssertionError(f"planned sessions {sessions} did not match {list(expected_sessions)}")


def validate_results(result: Any, batch_plan: Any, target_hz: float) -> list[dict[str, Any]]:
    if len(result.windows) != len(batch_plan.windows):
        raise AssertionError("query_batch window count does not match plan_batch")
    diag = result.diagnostics
    if not diag.range_enforced or diag.range_violations != 0:
        raise AssertionError("range enforcement failed")
    if diag.actual_cold_reads <= 0 or diag.footer_bytes <= 0:
        raise AssertionError("cold-read/footer accounting did not record reads")
    if diag.extrapolation_rejected:
        raise AssertionError("selected windows reported extrapolation rejection")

    reports = []
    expected_step_ns = int(round(1_000_000_000 / target_hz))
    for index, (window_result, window_plan) in enumerate(zip(result.windows, batch_plan.windows)):
        tensor = np.asarray(window_result.tensor)
        timestamps = np.asarray(window_result.timestamps_ns)
        session_id = str(window_plan._pose_rows[0]["session_id"])
        if list(tensor.shape)[1] != 9:
            raise AssertionError(f"{session_id} tensor shape {tensor.shape} does not have 9 channels")
        if tensor.shape[0] != timestamps.shape[0]:
            raise AssertionError(f"{session_id} tensor row count and timestamp count differ")
        if timestamps.shape[0] > 1:
            deltas = np.diff(timestamps)
            if not np.all(deltas > 0):
                raise AssertionError(f"{session_id} output timestamps are not monotonic")
            if not np.all(np.abs(deltas - expected_step_ns) <= 1):
                raise AssertionError(f"{session_id} output timestamps are not uniformly spaced at {target_hz}Hz")
        if not np.isfinite(tensor).all():
            raise AssertionError(f"{session_id} tensor contains NaN or infinity")
        pose_start = min(span.start_ts_ns for span in window_plan.pose_row_groups)
        pose_end = max(span.end_ts_ns for span in window_plan.pose_row_groups)
        if window_plan.start_ts_ns < pose_start or window_plan.end_ts_ns > pose_end:
            raise AssertionError(f"{session_id} query window exceeds selected pose row-group bounds")
        frame_count = len(window_result.media_manifest["frames"]) if window_result.media_manifest else 0
        if frame_count <= 0:
            raise AssertionError(f"{session_id} did not materialize camera frames")
        reports.append(
            {
                "index": index,
                "session_id": session_id,
                "tensor_shape": list(tensor.shape),
                "timestamp_count": int(timestamps.shape[0]),
                "timestamp_step_ns": int(np.diff(timestamps)[0]) if timestamps.shape[0] > 1 else 0,
                "pose_row_groups": len(window_plan.pose_row_groups),
                "imu_row_groups": len(window_plan.imu_row_groups),
                "media_row_groups": len(window_plan.media_row_groups),
                "media_frame_count": frame_count,
                "diagnostics": asdict(window_result.diagnostics),
            }
        )
    return reports


def source_monotonicity(paths: SequencePaths) -> dict[str, bool]:
    import duckdb

    checks = {}
    with duckdb.connect(":memory:") as con:
        for label, path in (("pose", paths.pose), ("imu", paths.imu), ("camera", paths.camera)):
            violations = con.execute(
                """
                SELECT count(*)
                FROM (
                    SELECT timestamp_ns, lag(timestamp_ns) OVER (ORDER BY timestamp_ns) AS prev_ts
                    FROM read_parquet(?)
                )
                WHERE prev_ts IS NOT NULL AND timestamp_ns <= prev_ts
                """,
                [str(path)],
            ).fetchone()[0]
            checks[label] = int(violations) == 0
    return checks


def parquet_stats(path: Path) -> dict[str, Any]:
    import duckdb

    with duckdb.connect(":memory:") as con:
        rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    return {"path": str(path), "rows": int(rows)}


def catalog_stats(path: Path) -> dict[str, Any]:
    import duckdb

    with duckdb.connect(":memory:") as con:
        row = con.execute(
            "SELECT count(*), coalesce(sum(byte_length), 0) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    return {"row_groups": int(row[0]), "indexed_bytes": int(row[1])}


def combined_catalog_stats(paths: CombinedPaths) -> dict[str, Any]:
    return {
        "pose": catalog_stats(paths.pose_catalog),
        "imu": catalog_stats(paths.imu_catalog),
        "media": catalog_stats(paths.media_catalog),
    }


def normalize_s3_prefix(prefix: str) -> str:
    value = prefix.rstrip("/")
    if not value.startswith("s3://") or len(value) <= len("s3://"):
        raise SystemExit("--s3-prefix must use s3://bucket/prefix")
    return value


def require_s3_env() -> None:
    missing = [
        name
        for name in ("AWS_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"--s3-prefix requires S3 env vars: {', '.join(missing)}")
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("AWS_ALLOW_HTTP", "true")
    os.environ.setdefault("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "false")
    os.environ.setdefault("AWS_ENDPOINT_URL_S3", os.environ["AWS_ENDPOINT"])


def upload_sequence_artifacts(paths: SequencePaths, s3_prefix: str, session_id: str) -> dict[str, str]:
    uris = {
        "pose_uri": f"{s3_prefix}/{session_id}/pose.parquet",
        "imu_uri": f"{s3_prefix}/{session_id}/imu.parquet",
        "camera_uri": f"{s3_prefix}/{session_id}/{paths.camera.name}",
    }
    upload(paths.pose, uris["pose_uri"])
    upload(paths.imu, uris["imu_uri"])
    upload(paths.camera, uris["camera_uri"])
    return uris


def upload_combined_artifacts(paths: CombinedPaths, s3_prefix: str) -> dict[str, str]:
    artifacts = {
        "combined_pose_catalog_uri": f"{s3_prefix}/combined_pose_catalog.parquet",
        "combined_imu_catalog_uri": f"{s3_prefix}/combined_imu_catalog.parquet",
        "combined_media_catalog_uri": f"{s3_prefix}/combined_media_catalog.parquet",
        "catalog_db_uri": f"{s3_prefix}/fleet.duckdb",
    }
    upload(paths.pose_catalog, artifacts["combined_pose_catalog_uri"])
    upload(paths.imu_catalog, artifacts["combined_imu_catalog_uri"])
    upload(paths.media_catalog, artifacts["combined_media_catalog_uri"])
    upload(paths.catalog_db, artifacts["catalog_db_uri"])
    return artifacts


def upload(input_path: Path, uri: str) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        try:
            run_robotics("object-store", "put", "--input", str(input_path), "--uri", uri)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(float(attempt * 2))
                continue
    print(
        "S3 upload failed. Verify the bucket exists and AWS_* env vars point at the intended S3-compatible endpoint.",
        file=sys.stderr,
    )
    assert last_error is not None
    raise last_error


def paths_payload(paths: SequencePaths) -> dict[str, str]:
    return {
        "pose": str(paths.pose),
        "imu": str(paths.imu),
        "camera": str(paths.camera),
        "pose_catalog": str(paths.pose_catalog),
        "imu_catalog": str(paths.imu_catalog),
        "media_catalog": str(paths.media_catalog),
    }


def batch_plan_payload(batch_plan: Any) -> dict[str, Any]:
    return {
        "window_count": len(batch_plan.windows),
        "authorized_pose_bytes": batch_plan.authorized_pose_bytes,
        "authorized_imu_bytes": batch_plan.authorized_imu_bytes,
        "authorized_media_bytes": batch_plan.authorized_media_bytes,
        "authorized_total_bytes": batch_plan.authorized_total_bytes,
        "materialized_pose_imu_bytes": batch_plan.materialized_pose_imu_bytes,
        "planned_range_reads": batch_plan.planned_range_reads,
        "blocked_by_egress": batch_plan.blocked_by_egress,
        "diagnostics": asdict(batch_plan.diagnostics),
        "windows": [
            {
                "session_id": str(window._pose_rows[0]["session_id"]),
                "pose_file_uri": window.pose_file_uri,
                "start_ts_ns": window.start_ts_ns,
                "end_ts_ns": window.end_ts_ns,
                "pose_row_groups": [span.row_group_id for span in window.pose_row_groups],
                "imu_row_groups": [span.row_group_id for span in window.imu_row_groups],
                "media_row_groups": [span.row_group_id for span in window.media_row_groups],
                "authorized_total_bytes": window.authorized_total_bytes,
            }
            for window in batch_plan.windows
        ],
    }


def parse_cli_metrics(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if value.lstrip("-").isdigit():
            metrics[key.strip()] = int(value)
        else:
            metrics[key.strip()] = value
    return metrics


def run_robotics(*args: str) -> subprocess.CompletedProcess[str]:
    robotics_bin = os.environ.get("ROBOTICS_BIN")
    if robotics_bin:
        cmd = [robotics_bin, *args]
    else:
        target_dir = os.environ.get("CARGO_TARGET_DIR")
        candidate = Path(target_dir) / "debug" / "robotics" if target_dir else None
        if candidate is not None and candidate.exists():
            cmd = [str(candidate), *args]
        else:
            cmd = ["cargo", "run", "-p", "robotics-cli", "--", *args]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stderr.write(f"command failed: {' '.join(cmd)}\n")
        if completed.stdout:
            sys.stderr.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        completed.check_returncode()
    return completed


def quote_sql(value: str) -> str:
    return value.replace("'", "''")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(max(round((len(ordered) - 1) * fraction), 0), len(ordered) - 1)
    return ordered[index]


def print_summary(report_path: Path, report: dict[str, Any]) -> None:
    planning = report["query"]["planning"]
    materialization = report["query"]["materialization"]
    diagnostics = materialization["diagnostics"]
    print(f"report={report_path}")
    print(f"window_count={materialization['window_count']}")
    print(f"planning_p50_ms={planning['p50_ms']:.3f}")
    print(f"planning_p95_ms={planning['p95_ms']:.3f}")
    print(f"authorized_total_bytes={diagnostics['authorized_total_bytes']}")
    print(f"actual_cold_read_bytes={diagnostics['actual_cold_read_bytes']}")
    print(f"footer_bytes={diagnostics['footer_bytes']}")
    print(f"range_violations={diagnostics['range_violations']}")
    print(f"egress_blocked={str(report['query']['egress_probe']['blocked_by_egress']).lower()}")
    for window in materialization["windows"]:
        print(
            f"{window['session_id']}: tensor_shape={window['tensor_shape']} "
            f"media_frame_count={window['media_frame_count']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
