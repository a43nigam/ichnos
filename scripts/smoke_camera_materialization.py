#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb import query


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test pose+IMU+camera materialization.")
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="physicaldb_camera_smoke_") as tmp:
            return run_smoke(Path(tmp))
    args.workdir.mkdir(parents=True, exist_ok=True)
    return run_smoke(args.workdir)


def run_smoke(root: Path) -> int:
    euroc = root / "euroc"
    gt_dir = euroc / "mav0" / "state_groundtruth_estimate0"
    imu_dir = euroc / "mav0" / "imu0"
    cam_dir = euroc / "mav0" / "cam0"
    gt_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    (cam_dir / "data").mkdir(parents=True, exist_ok=True)
    (gt_dir / "data.csv").write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
        "1000000000,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0\n"
        "1500000000,1,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0\n"
        "2000000000,2,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    (imu_dir / "data.csv").write_text(
        "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
        "900000000,0.1,0.2,0.3,9,0,-1\n"
        "1250000000,0.2,0.3,0.4,10,1,-2\n"
        "1750000000,0.4,0.5,0.6,12,3,-4\n"
        "2100000000,0.5,0.6,0.7,13,4,-5\n",
        encoding="utf-8",
    )
    (cam_dir / "data" / "1000000000.png").write_bytes(b"frame-one")
    (cam_dir / "data" / "1500000000.png").write_bytes(b"frame-two")
    (cam_dir / "data.csv").write_text(
        "#timestamp [ns],filename\n1000000000,1000000000.png\n1500000000,1500000000.png\n",
        encoding="utf-8",
    )

    pose = root / "pose.parquet"
    imu = root / "imu.parquet"
    camera = root / "cam0.parquet"
    pose_catalog = root / "pose_catalog.parquet"
    imu_catalog = root / "imu_catalog.parquet"
    media_catalog = root / "media_catalog.parquet"
    catalog_db = root / "fleet.duckdb"
    media_out = root / "media"
    manifest_out = root / "query_manifest.json"

    run_robotics("ingest", "euroc-groundtruth", "--input", str(euroc), "--out", str(pose))
    run_robotics("ingest", "euroc-imu", "--input", str(euroc), "--out", str(imu))
    run_robotics(
        "ingest",
        "euroc-camera",
        "--input",
        str(euroc),
        "--out",
        str(camera),
        "--stream-id",
        "cam0",
        "--row-group-rows",
        "1",
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
        "cam0",
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

    result = query(
        catalog_db=catalog_db,
        robot_id="mav0",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        predicate="velocity_magnitude > 1.5",
        channels=("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0"),
        target_hz=2.0,
        materialize_media=True,
        media_out=media_out,
        enforce_ranges=True,
        manifest_out=manifest_out,
    )
    print(f"tensor_shape={result.tensor.shape}")
    print(f"media_frames={len(result.media_manifest['frames']) if result.media_manifest else 0}")
    print(f"media_out={media_out}")
    print(f"manifest_out={manifest_out}")
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
