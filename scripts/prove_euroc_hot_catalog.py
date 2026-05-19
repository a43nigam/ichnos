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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb import EgressLimitError, plan, query


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove hot-catalog planning and enforced cold reads on EuRoC pose+IMU+camera data."
    )
    parser.add_argument("--input", type=Path, required=True, help="Extracted EuRoC sequence directory or .zip")
    parser.add_argument("--work-dir", type=Path, required=True, help="Generated Parquet/catalog/proof output directory")
    parser.add_argument("--s3-prefix", help="Optional s3://bucket/prefix for generated Parquet objects")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-authorized-bytes", type=int)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--robot-id", default="mav0")
    parser.add_argument("--session-id", default="euroc_session")
    parser.add_argument("--stream-id", default="cam0")
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument("--pose-row-group-rows", type=int, default=500)
    parser.add_argument("--imu-row-group-rows", type=int, default=2000)
    parser.add_argument("--camera-row-group-rows", type=int, default=20)
    parser.add_argument("--footer-allowance-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if importlib.util.find_spec("duckdb") is None:
        print("SKIP: duckdb Python package is required for the EuRoC hot-catalog proof")
        return 0
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    paths = ProofPaths(args.work_dir, args.stream_id)

    build_or_reuse_assets(args, paths)
    uris = local_uris(paths)
    if args.s3_prefix:
        uris = upload_assets(args.s3_prefix, paths)

    build_or_reuse_catalogs(args, paths, uris)
    proof_query = choose_proof_query(paths.catalog_db, args.robot_id)
    channels = ("pos_xyz", "imu_accel", "imu_gyro", f"camera:{args.stream_id}")

    timings_ms: list[float] = []
    last_plan = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        last_plan = plan(
            catalog_db=paths.catalog_db,
            robot_id=args.robot_id,
            start_ts_ns=proof_query["start_ts_ns"],
            end_ts_ns=proof_query["end_ts_ns"],
            predicate=proof_query["predicate"],
            channels=channels,
            max_egress_bytes=1_000_000_000_000,
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    assert last_plan is not None

    egress_probe = plan(
        catalog_db=paths.catalog_db,
        robot_id=args.robot_id,
        start_ts_ns=proof_query["start_ts_ns"],
        end_ts_ns=proof_query["end_ts_ns"],
        predicate=proof_query["predicate"],
        channels=channels,
        max_egress_bytes=max(last_plan.authorized_pose_bytes + last_plan.authorized_imu_bytes, 0),
    )

    media_out = args.work_dir / "media_materialized"
    seek_manifest = args.work_dir / "query_seek_manifest.json"
    result = query(
        catalog_db=paths.catalog_db,
        robot_id=args.robot_id,
        start_ts_ns=proof_query["start_ts_ns"],
        end_ts_ns=proof_query["end_ts_ns"],
        predicate=proof_query["predicate"],
        channels=channels,
        target_hz=args.target_hz,
        max_egress_bytes=1_000_000_000_000,
        materialize_media=True,
        media_out=media_out,
        enforce_ranges=True,
        footer_allowance_bytes=args.footer_allowance_bytes,
        manifest_out=seek_manifest,
    )

    p50 = percentile(timings_ms, 0.50)
    p95 = percentile(timings_ms, 0.95)
    proof = {
        "version": 1,
        "input": str(args.input),
        "work_dir": str(args.work_dir),
        "s3_prefix": args.s3_prefix,
        "catalog_db": str(paths.catalog_db),
        "pose_uri": uris["pose"],
        "imu_uri": uris["imu"],
        "media_uri": uris["camera"],
        "query": proof_query,
        "planning": {
            "iterations": args.iterations,
            "p50_ms": p50,
            "p95_ms": p95,
            "mean_ms": statistics.fmean(timings_ms),
            "samples_ms": timings_ms,
        },
        "plan": plan_payload(last_plan),
        "egress_probe": {
            "max_egress_bytes": egress_probe.egress_limit_bytes,
            "blocked_by_egress": egress_probe.blocked_by_egress,
            "media_blocked_by_egress": egress_probe.diagnostics.media_blocked_by_egress,
        },
        "materialization": {
            "tensor_shape": list(result.tensor.shape),
            "timestamp_count": int(result.timestamps_ns.shape[0]),
            "media_frame_count": len(result.media_manifest["frames"]) if result.media_manifest else 0,
            "media_out": str(media_out),
            "seek_manifest": str(seek_manifest),
            "diagnostics": asdict(result.diagnostics),
        },
    }

    manifest_out = args.manifest_out or (args.work_dir / "euroc_hot_catalog_proof.json")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")

    failures = threshold_failures(args, p95, last_plan.authorized_total_bytes)
    print_summary(manifest_out, proof)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


class ProofPaths:
    def __init__(self, work_dir: Path, stream_id: str) -> None:
        self.pose = work_dir / "pose.parquet"
        self.imu = work_dir / "imu.parquet"
        self.camera = work_dir / f"{stream_id}.parquet"
        self.pose_catalog = work_dir / "pose_catalog.parquet"
        self.imu_catalog = work_dir / "imu_catalog.parquet"
        self.media_catalog = work_dir / "media_catalog.parquet"
        self.catalog_db = work_dir / "fleet.duckdb"


def build_or_reuse_assets(args: argparse.Namespace, paths: ProofPaths) -> None:
    if args.rebuild or not paths.pose.exists():
        run_robotics(
            "ingest",
            "euroc-groundtruth",
            "--input",
            str(args.input),
            "--out",
            str(paths.pose),
            "--robot-id",
            args.robot_id,
            "--session-id",
            args.session_id,
            "--row-group-rows",
            str(args.pose_row_group_rows),
        )
    if args.rebuild or not paths.imu.exists():
        run_robotics(
            "ingest",
            "euroc-imu",
            "--input",
            str(args.input),
            "--out",
            str(paths.imu),
            "--robot-id",
            args.robot_id,
            "--session-id",
            args.session_id,
            "--row-group-rows",
            str(args.imu_row_group_rows),
        )
    if args.rebuild or not paths.camera.exists():
        run_robotics(
            "ingest",
            "euroc-camera",
            "--input",
            str(args.input),
            "--out",
            str(paths.camera),
            "--stream-id",
            args.stream_id,
            "--robot-id",
            args.robot_id,
            "--session-id",
            args.session_id,
            "--row-group-rows",
            str(args.camera_row_group_rows),
        )


def build_or_reuse_catalogs(args: argparse.Namespace, paths: ProofPaths, uris: dict[str, str]) -> None:
    rebuilt_catalog = False
    if args.rebuild or catalog_needs_rebuild(paths.pose_catalog, uris["pose"]):
        run_robotics("catalog", "build", "--input", str(paths.pose), "--out", str(paths.pose_catalog), "--uri", uris["pose"])
        rebuilt_catalog = True
    if args.rebuild or catalog_needs_rebuild(paths.imu_catalog, uris["imu"]):
        run_robotics("catalog", "build-imu", "--input", str(paths.imu), "--out", str(paths.imu_catalog), "--uri", uris["imu"])
        rebuilt_catalog = True
    if args.rebuild or catalog_needs_rebuild(paths.media_catalog, uris["camera"]):
        run_robotics(
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
            uris["camera"],
        )
        rebuilt_catalog = True
    if args.rebuild or rebuilt_catalog or not paths.catalog_db.exists():
        run_robotics(
            "catalog",
            "duckdb-build",
            "--pose-catalog",
            str(paths.pose_catalog),
            "--imu-catalog",
            str(paths.imu_catalog),
            "--media-catalog",
            str(paths.media_catalog),
            "--spatial-index",
            "hilbert",
            "--out",
            str(paths.catalog_db),
        )


def catalog_needs_rebuild(catalog: Path, expected_uri: str) -> bool:
    if not catalog.exists():
        return True
    import duckdb

    try:
        with duckdb.connect(":memory:") as con:
            uris = {
                str(row[0])
                for row in con.execute(
                    "SELECT DISTINCT file_uri FROM read_parquet(?)",
                    [str(catalog)],
                ).fetchall()
            }
    except Exception:
        return True
    return uris != {expected_uri}


def local_uris(paths: ProofPaths) -> dict[str, str]:
    return {
        "pose": paths.pose.resolve().as_uri(),
        "imu": paths.imu.resolve().as_uri(),
        "camera": paths.camera.resolve().as_uri(),
    }


def upload_assets(s3_prefix: str, paths: ProofPaths) -> dict[str, str]:
    prefix = s3_prefix.rstrip("/")
    if not prefix.startswith("s3://"):
        raise SystemExit("--s3-prefix must start with s3://")
    uris = {
        "pose": f"{prefix}/pose.parquet",
        "imu": f"{prefix}/imu.parquet",
        "camera": f"{prefix}/camera/{paths.camera.name}",
    }
    for key, source in (("pose", paths.pose), ("imu", paths.imu), ("camera", paths.camera)):
        run_robotics("object-store", "put", "--input", str(source), "--uri", uris[key])
    return uris


def choose_proof_query(catalog_db: Path, robot_id: str) -> dict[str, Any]:
    import duckdb

    with duckdb.connect(str(catalog_db), read_only=True) as con:
        common = con.execute(
            """
            SELECT
                greatest(
                    (SELECT min(start_ts_ns) FROM pose_row_groups WHERE robot_id = ?),
                    (SELECT min(start_ts_ns) FROM imu_row_groups WHERE robot_id = ?),
                    (SELECT min(start_ts_ns) FROM media_row_groups WHERE robot_id = ?)
                ) AS start_ts_ns,
                least(
                    (SELECT max(end_ts_ns) FROM pose_row_groups WHERE robot_id = ?),
                    (SELECT max(end_ts_ns) FROM imu_row_groups WHERE robot_id = ?),
                    (SELECT max(end_ts_ns) FROM media_row_groups WHERE robot_id = ?)
                ) AS end_ts_ns
            """,
            [robot_id, robot_id, robot_id, robot_id, robot_id, robot_id],
        ).fetchone()
        if common is None or common[0] is None or common[1] is None or int(common[0]) > int(common[1]):
            raise SystemExit(f"no overlapping pose/IMU/media catalog window for robot_id={robot_id}")

        rows = con.execute(
            """
            SELECT row_group_id, start_ts_ns, end_ts_ns, min_x, max_x, min_y, max_y, min_z, max_z, max_velocity
            FROM pose_row_groups
            WHERE robot_id = ? AND end_ts_ns >= ? AND start_ts_ns <= ?
            ORDER BY start_ts_ns, row_group_id
            """,
            [robot_id, int(common[0]), int(common[1])],
        ).fetchall()
    if not rows:
        raise SystemExit(f"no pose row groups overlap media/IMU for robot_id={robot_id}")

    row = rows[len(rows) // 2]
    start_ts_ns = max(int(row[1]), int(common[0]))
    end_ts_ns = min(int(row[2]), int(common[1]))
    if start_ts_ns >= end_ts_ns:
        start_ts_ns, end_ts_ns = int(common[0]), int(common[1])

    min_x, max_x = sorted((float(row[3]), float(row[4])))
    min_y, max_y = sorted((float(row[5]), float(row[6])))
    min_z, max_z = sorted((float(row[7]), float(row[8])))
    pad_xy = max((max_x - min_x) * 0.25, (max_y - min_y) * 0.25, 0.25)
    pad_z = max((max_z - min_z) * 0.25, 0.25)
    velocity_threshold = max(float(row[9]) * 0.5, 0.0)
    bbox = (
        min_x - pad_xy,
        max_x + pad_xy,
        min_y - pad_xy,
        max_y + pad_xy,
        min_z - pad_z,
        max_z + pad_z,
    )
    predicate = (
        f"time_overlap({start_ts_ns},{end_ts_ns}) AND "
        f"ST_Intersects(position, bbox({bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},"
        f"{bbox[3]:.6f},{bbox[4]:.6f},{bbox[5]:.6f})) AND "
        f"velocity_magnitude >= {velocity_threshold:.6f}"
    )
    return {
        "predicate": predicate,
        "start_ts_ns": start_ts_ns,
        "end_ts_ns": end_ts_ns,
        "bbox": list(bbox),
        "velocity_threshold": velocity_threshold,
        "seed_pose_row_group_id": int(row[0]),
    }


def plan_payload(seek_plan: Any) -> dict[str, Any]:
    diag = seek_plan.diagnostics
    candidate = diag.candidate_row_groups
    matched = diag.matched_row_groups
    return {
        "candidate_row_groups": candidate,
        "matched_row_groups": matched,
        "prune_ratio": 0.0 if candidate == 0 else 1.0 - (matched / candidate),
        "time_pruned_row_groups": diag.time_pruned_row_groups,
        "hilbert_pruned_row_groups": diag.hilbert_pruned_row_groups,
        "exact_spatial_pruned_row_groups": diag.exact_spatial_pruned_row_groups,
        "velocity_pruned_row_groups": diag.velocity_pruned_row_groups,
        "index_strategy": diag.index_strategy,
        "pose_row_groups": [span.row_group_id for span in seek_plan.pose_row_groups],
        "imu_row_groups": [span.row_group_id for span in seek_plan.imu_row_groups],
        "media_row_groups": [span.row_group_id for span in seek_plan.media_row_groups],
        "authorized_pose_bytes": seek_plan.authorized_pose_bytes,
        "authorized_imu_bytes": seek_plan.authorized_imu_bytes,
        "authorized_media_bytes": seek_plan.authorized_media_bytes,
        "authorized_total_bytes": seek_plan.authorized_total_bytes,
        "materialized_pose_imu_bytes": seek_plan.materialized_pose_imu_bytes,
        "planned_range_reads": seek_plan.planned_range_reads,
        "blocked_by_egress": seek_plan.blocked_by_egress,
    }


def threshold_failures(args: argparse.Namespace, p95: float, authorized_total_bytes: int) -> list[str]:
    failures: list[str] = []
    if args.max_p95_ms is not None and p95 > args.max_p95_ms:
        failures.append(f"planning p95 {p95:.3f}ms exceeded {args.max_p95_ms:.3f}ms")
    if args.max_authorized_bytes is not None and authorized_total_bytes > args.max_authorized_bytes:
        failures.append(
            f"authorized bytes {authorized_total_bytes} exceeded {args.max_authorized_bytes}"
        )
    return failures


def print_summary(manifest_out: Path, proof: dict[str, Any]) -> None:
    planning = proof["planning"]
    plan_info = proof["plan"]
    diag = proof["materialization"]["diagnostics"]
    print(f"manifest_out={manifest_out}")
    print(f"predicate={proof['query']['predicate']}")
    print(f"planning_p50_ms={planning['p50_ms']:.3f}")
    print(f"planning_p95_ms={planning['p95_ms']:.3f}")
    print(f"candidate_row_groups={plan_info['candidate_row_groups']}")
    print(f"matched_row_groups={plan_info['matched_row_groups']}")
    print(f"hilbert_pruned_row_groups={plan_info['hilbert_pruned_row_groups']}")
    print(f"exact_spatial_pruned_row_groups={plan_info['exact_spatial_pruned_row_groups']}")
    print(f"velocity_pruned_row_groups={plan_info['velocity_pruned_row_groups']}")
    print(f"time_pruned_row_groups={plan_info['time_pruned_row_groups']}")
    print(f"authorized_pose_bytes={plan_info['authorized_pose_bytes']}")
    print(f"authorized_imu_bytes={plan_info['authorized_imu_bytes']}")
    print(f"authorized_media_bytes={plan_info['authorized_media_bytes']}")
    print(f"authorized_total_bytes={plan_info['authorized_total_bytes']}")
    print(f"actual_cold_read_bytes={diag['actual_cold_read_bytes']}")
    print(f"footer_bytes={diag['footer_bytes']}")
    print(f"blocked_by_egress={str(proof['egress_probe']['blocked_by_egress']).lower()}")
    print(f"tensor_shape={proof['materialization']['tensor_shape']}")
    print(f"media_frame_count={proof['materialization']['media_frame_count']}")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(max(round((len(ordered) - 1) * fraction), 0), len(ordered) - 1)
    return ordered[index]


def run_robotics(*args: str) -> subprocess.CompletedProcess[str]:
    robotics_bin = os.environ.get("ROBOTICS_BIN")
    cmd = [robotics_bin, *args] if robotics_bin else ["cargo", "run", "-p", "robotics-cli", "--", *args]
    return subprocess.run(cmd, check=True, text=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EgressLimitError as exc:
        print(f"FAIL: materialization blocked by egress: {exc}", file=sys.stderr)
        raise SystemExit(1)
