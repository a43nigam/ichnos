#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb import query


START_TS_NS = 1_000_000_000
NS_PER_SECOND = 1_000_000_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate larger pose/IMU Parquet fixtures, upload to S3, and validate enforced range reads."
    )
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--pose-hz", type=float, default=100.0)
    parser.add_argument("--imu-hz", type=float, default=200.0)
    parser.add_argument("--pose-row-group-rows", type=int, default=500)
    parser.add_argument("--imu-row-group-rows", type=int, default=2_000)
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "robotics"))
    parser.add_argument("--prefix", default=f"physicaldb-large-ranges/{uuid.uuid4().hex}")
    parser.add_argument("--footer-allowance-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-violations", type=int, default=0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--include-media", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if importlib.util.find_spec("duckdb") is None:
        print("SKIP: duckdb Python package is required for large S3 range validation")
        return 0

    missing = [
        name
        for name in ("AWS_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        print(f"SKIP: missing required S3 env vars: {', '.join(missing)}")
        return 0

    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("AWS_ALLOW_HTTP", "true")
    os.environ.setdefault("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "false")
    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")

    validate_args(args)

    with tempfile.TemporaryDirectory(prefix="physicaldb_s3_large_ranges_") as tmp:
        root = Path(tmp)
        euroc = root / "euroc"
        pose = root / "pose_large.parquet"
        imu = root / "imu_large.parquet"
        pose_catalog = root / "pose_catalog.parquet"
        imu_catalog = root / "imu_catalog.parquet"
        media_catalog = root / "media_catalog.parquet"
        catalog_db = root / "fleet.duckdb"

        pose_uri = f"s3://{args.bucket}/{args.prefix}/pose_large.parquet"
        imu_uri = f"s3://{args.bucket}/{args.prefix}/imu_large.parquet"
        media_uri = f"s3://{args.bucket}/{args.prefix}/camera/cam0_large.parquet"

        write_euroc_fixture(
            euroc,
            duration_sec=args.duration_sec,
            pose_hz=args.pose_hz,
            imu_hz=args.imu_hz,
        )
        run_robotics(
            "ingest",
            "euroc-groundtruth",
            "--input",
            str(euroc),
            "--out",
            str(pose),
            "--row-group-rows",
            str(args.pose_row_group_rows),
        )
        run_robotics(
            "ingest",
            "euroc-imu",
            "--input",
            str(euroc),
            "--out",
            str(imu),
            "--row-group-rows",
            str(args.imu_row_group_rows),
        )
        upload(pose, pose_uri)
        upload(imu, imu_uri)
        if args.include_media:
            upload(pose, media_uri)

        run_robotics("catalog", "build", "--input", str(pose), "--out", str(pose_catalog), "--uri", pose_uri)
        run_robotics(
            "catalog",
            "build-imu",
            "--input",
            str(imu),
            "--out",
            str(imu_catalog),
            "--uri",
            imu_uri,
        )
        duckdb_args = [
            "catalog",
            "duckdb-build",
            "--pose-catalog",
            str(pose_catalog),
            "--imu-catalog",
            str(imu_catalog),
            "--out",
            str(catalog_db),
        ]
        if args.include_media:
            run_robotics(
                "catalog",
                "build-media",
                "--input",
                str(pose),
                "--out",
                str(media_catalog),
                "--modality",
                "camera",
                "--stream-id",
                "cam0",
                "--uri",
                media_uri,
            )
            duckdb_args.extend(["--media-catalog", str(media_catalog)])
        run_robotics(*duckdb_args)

        windows = query_windows(args.duration_sec, args.pose_hz, args.pose_row_group_rows)
        total_violations = 0
        for label, start_ns, end_ns in windows:
            channels = ("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0") if args.include_media else (
                "pos_xyz",
                "imu_accel",
                "imu_gyro",
            )
            manifest_out = root / f"{label}.manifest.json"
            try:
                result = query(
                    catalog_db=catalog_db,
                    robot_id="mav0",
                    start_ts_ns=start_ns,
                    end_ts_ns=end_ns,
                    predicate="velocity_magnitude > 1.5 AND ST_Intersects(position, bbox(-1,250,-1,1,-1,1))",
                    channels=channels,
                    target_hz=args.target_hz,
                    output="numpy",
                    enforce_ranges=True,
                    footer_allowance_bytes=args.footer_allowance_bytes,
                    manifest_out=manifest_out,
                )
            except RuntimeError as exc:
                print(f"FAIL: window={label} enforced query failed: {exc}", file=sys.stderr)
                return 1
            total_violations += result.diagnostics.range_violations
            print(f"window={label}")
            print(f"window_start_ts_ns={start_ns}")
            print(f"window_end_ts_ns={end_ns}")
            print(f"tensor_shape={result.tensor.shape}")
            print(f"pose_matched_row_groups={result.diagnostics.pose_matched_row_groups}")
            print(f"imu_matched_row_groups={result.diagnostics.imu_matched_row_groups}")
            print(f"media_matched_row_groups={result.diagnostics.media_matched_row_groups}")
            print(f"authorized_total_bytes={result.diagnostics.authorized_total_bytes}")
            print(f"planned_read_bytes={result.diagnostics.planned_read_bytes}")
            print(f"actual_cold_reads={result.diagnostics.actual_cold_reads}")
            print(f"actual_cold_read_bytes={result.diagnostics.actual_cold_read_bytes}")
            print(f"actual_authorized_bytes={result.diagnostics.actual_authorized_bytes}")
            print(f"footer_allowance_bytes={result.diagnostics.footer_allowance_bytes}")
            print(f"footer_bytes={result.diagnostics.footer_bytes}")
            print(f"largest_metadata_read={result.diagnostics.largest_metadata_read}")
            print(f"max_footer_read_offset={result.diagnostics.max_footer_read_offset}")
            print(f"max_footer_read_end={result.diagnostics.max_footer_read_end}")
            print(f"range_violations={result.diagnostics.range_violations}")
            print(f"manifest_out={manifest_out}")

        if total_violations > args.max_violations:
            print(
                f"FAIL: total range violations {total_violations} exceeded max {args.max_violations}",
                file=sys.stderr,
            )
            return 1
        print("large_s3_range_validation=passed")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be positive")
    if args.pose_hz <= 0 or args.imu_hz <= 0 or args.target_hz <= 0:
        raise SystemExit("--pose-hz, --imu-hz, and --target-hz must be positive")
    if args.pose_row_group_rows < 1 or args.imu_row_group_rows < 1:
        raise SystemExit("--pose-row-group-rows and --imu-row-group-rows must be positive")
    if args.footer_allowance_bytes < 1:
        raise SystemExit("--footer-allowance-bytes must be positive")


def write_euroc_fixture(root: Path, *, duration_sec: float, pose_hz: float, imu_hz: float) -> None:
    gt_dir = root / "mav0" / "state_groundtruth_estimate0"
    imu_dir = root / "mav0" / "imu0"
    gt_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)

    pose_count = int(duration_sec * pose_hz) + 1
    pose_step_ns = int(round(NS_PER_SECOND / pose_hz))
    with (gt_dir / "data.csv").open("w", encoding="utf-8") as handle:
        handle.write("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n")
        for index in range(pose_count):
            timestamp_ns = START_TS_NS + index * pose_step_ns
            seconds = index / pose_hz
            x = seconds * 2.0
            handle.write(
                f"{timestamp_ns},{x:.9f},0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
            )

    imu_start_ns = START_TS_NS - 100_000_000
    imu_count = int((duration_sec + 0.2) * imu_hz) + 1
    imu_step_ns = int(round(NS_PER_SECOND / imu_hz))
    with (imu_dir / "data.csv").open("w", encoding="utf-8") as handle:
        handle.write("#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n")
        for index in range(imu_count):
            timestamp_ns = imu_start_ns + index * imu_step_ns
            value = index / imu_hz
            handle.write(
                f"{timestamp_ns},0.1,0.2,0.3,{9.0 + value:.9f},0.0,-1.0\n"
            )


def query_windows(duration_sec: float, pose_hz: float, pose_row_group_rows: int) -> list[tuple[str, int, int]]:
    first_row_group_end = START_TS_NS + int(((pose_row_group_rows - 1) / pose_hz) * NS_PER_SECOND)
    middle_start = START_TS_NS + int(max(duration_sec / 2.0 - 5.0, 0.0) * NS_PER_SECOND)
    middle_end = middle_start + 10 * NS_PER_SECOND
    full_end = START_TS_NS + int(duration_sec * NS_PER_SECOND)
    return [
        ("first_row_group", START_TS_NS, first_row_group_end),
        ("middle_10s", middle_start, min(middle_end, full_end)),
        ("full_file", START_TS_NS, full_end),
    ]


def upload(input_path: Path, uri: str) -> None:
    run_robotics("object-store", "put", "--input", str(input_path), "--uri", uri)


def run_robotics(*args: str) -> subprocess.CompletedProcess[str]:
    robotics_bin = os.environ.get("ROBOTICS_BIN")
    if robotics_bin:
        cmd = [robotics_bin, *args]
    else:
        cmd = ["cargo", "run", "-p", "robotics-cli", "--", *args]
    return subprocess.run(cmd, check=True, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
