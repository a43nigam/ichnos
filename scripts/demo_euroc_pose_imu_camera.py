#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb import plan, query


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a EuRoC pose+IMU+camera catalog demo.")
    parser.add_argument("--input", type=Path, required=True, help="Extracted EuRoC sequence or .zip")
    parser.add_argument("--workdir", type=Path, default=Path("data/demo/euroc"))
    parser.add_argument("--robot-id", default="mav0")
    parser.add_argument("--session-id", default="euroc_session")
    parser.add_argument("--stream-id", default="cam0")
    parser.add_argument("--start-ts-ns", type=int)
    parser.add_argument("--end-ts-ns", type=int)
    parser.add_argument("--min-velocity", type=float, default=0.1)
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument("--max-egress-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--materialize-media", action="store_true")
    parser.add_argument("--media-out", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")
    args.workdir.mkdir(parents=True, exist_ok=True)
    pose = args.workdir / "pose.parquet"
    imu = args.workdir / "imu.parquet"
    camera = args.workdir / f"{args.stream_id}.parquet"
    pose_catalog = args.workdir / "pose_catalog.parquet"
    imu_catalog = args.workdir / "imu_catalog.parquet"
    media_catalog = args.workdir / "media_catalog.parquet"
    catalog_db = args.workdir / "fleet.duckdb"
    manifest_out = args.manifest_out or (args.workdir / "query_manifest.json")
    media_out = args.media_out or (args.workdir / "media_frames")

    run_robotics(
        "ingest",
        "euroc-groundtruth",
        "--input",
        str(args.input),
        "--out",
        str(pose),
        "--robot-id",
        args.robot_id,
        "--session-id",
        args.session_id,
    )
    run_robotics(
        "ingest",
        "euroc-imu",
        "--input",
        str(args.input),
        "--out",
        str(imu),
        "--robot-id",
        args.robot_id,
        "--session-id",
        args.session_id,
    )
    run_robotics(
        "ingest",
        "euroc-camera",
        "--input",
        str(args.input),
        "--out",
        str(camera),
        "--stream-id",
        args.stream_id,
        "--robot-id",
        args.robot_id,
        "--session-id",
        args.session_id,
    )
    run_robotics("catalog", "build", "--input", str(pose), "--out", str(pose_catalog))
    run_robotics("catalog", "build-imu", "--input", str(imu), "--out", str(imu_catalog))
    run_robotics(
        "catalog",
        "build-media",
        "--input",
        str(camera),
        "--out",
        str(media_catalog),
        "--modality",
        "camera",
        "--stream-id",
        args.stream_id,
        "--uri",
        camera.as_uri(),
    )
    run_robotics(
        "catalog",
        "duckdb-build",
        "--pose-catalog",
        str(pose_catalog),
        "--imu-catalog",
        str(imu_catalog),
        "--media-catalog",
        str(media_catalog),
        "--out",
        str(catalog_db),
    )

    predicate = f"velocity_magnitude > {args.min_velocity}"
    kwargs = {
        "catalog_db": catalog_db,
        "robot_id": args.robot_id,
        "predicate": predicate,
        "channels": ("pos_xyz", "imu_accel", "imu_gyro", f"camera:{args.stream_id}"),
        "target_hz": args.target_hz,
        "max_egress_bytes": args.max_egress_bytes,
        "manifest_out": manifest_out,
    }
    if args.materialize_media:
        kwargs["materialize_media"] = True
        kwargs["media_out"] = media_out
    if args.start_ts_ns is not None:
        kwargs["start_ts_ns"] = args.start_ts_ns
    if args.end_ts_ns is not None:
        kwargs["end_ts_ns"] = args.end_ts_ns

    plan_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"target_hz", "manifest_out", "materialize_media", "media_out"}
    }
    seek_plan = plan(**plan_kwargs)
    result = query(**kwargs)
    print(f"catalog_db={catalog_db}")
    print(f"manifest_out={manifest_out}")
    if args.materialize_media:
        print(f"media_out={media_out}")
    print(f"tensor_shape={result.tensor.shape}")
    print(f"pose_row_groups={len(seek_plan.pose_row_groups)}")
    print(f"imu_row_groups={len(seek_plan.imu_row_groups)}")
    print(f"media_row_groups={len(seek_plan.media_row_groups)}")
    print(f"authorized_total_bytes={seek_plan.authorized_total_bytes}")
    print(f"materialized_pose_imu_bytes={seek_plan.materialized_pose_imu_bytes}")
    if result.media_manifest is not None:
        print(f"materialized_media_frames={len(result.media_manifest['frames'])}")
    return 0


def run_robotics(*args: str) -> subprocess.CompletedProcess[str]:
    robotics_bin = os.environ.get("ROBOTICS_BIN")
    cmd = (
        [robotics_bin, *args]
        if robotics_bin
        else ["cargo", "run", "-p", "robotics-cli", "--", *args]
    )
    return subprocess.run(cmd, check=True, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
