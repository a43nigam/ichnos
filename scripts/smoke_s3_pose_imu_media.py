#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from physicaldb import EgressLimitError, query


START_TS_NS = 1_000_000_000
END_TS_NS = 2_000_000_000


def main() -> int:
    if importlib.util.find_spec("duckdb") is None:
        print("SKIP: duckdb Python package is required for the S3 smoke")
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
    os.environ.setdefault("AWS_ENDPOINT_URL_S3", os.environ["AWS_ENDPOINT"])
    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")

    bucket = os.environ.get("S3_BUCKET", "robotics")
    prefix = f"physicaldb-smoke/{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory(prefix="physicaldb_s3_smoke_") as tmp:
        root = Path(tmp)
        euroc = root / "euroc"
        pose = root / "pose.parquet"
        imu = root / "imu.parquet"
        catalog = root / "pose_catalog.parquet"
        imu_catalog = root / "imu_catalog.parquet"
        media_catalog = root / "media_catalog.parquet"
        catalog_db = root / "fleet.duckdb"

        pose_uri = f"s3://{bucket}/{prefix}/pose.parquet"
        imu_uri = f"s3://{bucket}/{prefix}/imu.parquet"
        media_uri = f"s3://{bucket}/{prefix}/camera/cam0.parquet"

        write_euroc_fixture(euroc)
        run_robotics(
            "ingest",
            "euroc-groundtruth",
            "--input",
            str(euroc),
            "--out",
            str(pose),
            "--row-group-rows",
            "2",
        )
        run_robotics(
            "ingest",
            "euroc-imu",
            "--input",
            str(euroc),
            "--out",
            str(imu),
            "--row-group-rows",
            "2",
        )

        upload(pose, pose_uri)
        upload(imu, imu_uri)
        upload(pose, media_uri)

        run_robotics(
            "catalog",
            "build",
            "--input",
            str(pose),
            "--out",
            str(catalog),
            "--uri",
            pose_uri,
        )
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
        run_robotics(
            "catalog",
            "duckdb-build",
            "--pose-catalog",
            str(catalog),
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
            start_ts_ns=START_TS_NS,
            end_ts_ns=END_TS_NS,
            predicate=(
                "velocity_magnitude > 1.5 AND "
                "ST_Intersects(position, bbox(-1,3,-1,1,-1,1))"
            ),
            channels=("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0"),
            target_hz=2.0,
            output="numpy",
            enforce_ranges=True,
            manifest_out=root / "enforced_seek_manifest.json",
        )
        assert result.file_uri == pose_uri
        assert result.tensor.shape == (3, 9)
        assert result.diagnostics.pose_matched_row_groups == 2
        assert result.diagnostics.imu_matched_row_groups == 2
        assert result.diagnostics.media_matched_row_groups == 2
        assert result.diagnostics.media_selected_bytes > 0
        assert result.diagnostics.range_audit_passed
        assert result.diagnostics.range_enforced
        assert result.diagnostics.actual_cold_reads > 0
        assert result.diagnostics.footer_bytes > 0
        assert result.diagnostics.range_violations == 0
        assert (
            result.diagnostics.total_selected_bytes
            == result.diagnostics.pose_selected_bytes
            + result.diagnostics.imu_selected_bytes
            + result.diagnostics.media_selected_bytes
        )

        blocked = False
        try:
            query(
                catalog_db=catalog_db,
                robot_id="mav0",
                start_ts_ns=START_TS_NS,
                end_ts_ns=END_TS_NS,
                predicate="velocity_magnitude > 1.5",
                channels=("pos_xyz", "camera:cam0"),
                target_hz=2.0,
                max_egress_bytes=result.diagnostics.pose_selected_bytes,
                robotics_bin=root / "missing_robotics_binary",
            )
        except EgressLimitError as exc:
            blocked = True
            blocked_message = str(exc)
        else:
            blocked_message = "low-budget media query was not blocked"
        assert blocked, blocked_message

        print(f"shape={result.tensor.shape}")
        print(f"file_uri={result.file_uri}")
        print(f"pose_matched_row_groups={result.diagnostics.pose_matched_row_groups}")
        print(f"imu_matched_row_groups={result.diagnostics.imu_matched_row_groups}")
        print(f"media_matched_row_groups={result.diagnostics.media_matched_row_groups}")
        print(f"pose_selected_bytes={result.diagnostics.pose_selected_bytes}")
        print(f"imu_selected_bytes={result.diagnostics.imu_selected_bytes}")
        print(f"media_selected_bytes={result.diagnostics.media_selected_bytes}")
        print(f"total_selected_bytes={result.diagnostics.total_selected_bytes}")
        print(f"planned_read_bytes={result.diagnostics.planned_read_bytes}")
        print(f"actual_cold_reads={result.diagnostics.actual_cold_reads}")
        print(f"actual_cold_read_bytes={result.diagnostics.actual_cold_read_bytes}")
        print(f"footer_bytes={result.diagnostics.footer_bytes}")
        print(f"range_audit_passed={str(result.diagnostics.range_audit_passed).lower()}")
        print(f"range_enforced={str(result.diagnostics.range_enforced).lower()}")
        print(f"manifest_out={root / 'enforced_seek_manifest.json'}")
        print(f"catalog_query_ms={result.diagnostics.catalog_query_ms:.3f}")
        print(f"media_egress_blocked_before_materialization={str(blocked).lower()}")
        print("live_s3_pose_imu_media_smoke=passed")

    return 0


def write_euroc_fixture(root: Path) -> None:
    gt_dir = root / "mav0" / "state_groundtruth_estimate0"
    imu_dir = root / "mav0" / "imu0"
    gt_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    (gt_dir / "data.csv").write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
        "1000000000,0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
        "1500000000,1.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
        "2000000000,2.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    (imu_dir / "data.csv").write_text(
        "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
        "900000000,0.1,0.2,0.3,9.0,0.0,-1.0\n"
        "1250000000,0.2,0.3,0.4,10.0,1.0,-2.0\n"
        "1750000000,0.4,0.5,0.6,12.0,3.0,-4.0\n"
        "2100000000,0.5,0.6,0.7,13.0,4.0,-5.0\n",
        encoding="utf-8",
    )


def upload(input_path: Path, uri: str) -> None:
    try:
        run_robotics("object-store", "put", "--input", str(input_path), "--uri", uri)
    except subprocess.CalledProcessError as exc:
        print(
            "S3 upload failed. Verify the bucket exists and the AWS_* env vars point "
            "at the intended S3-compatible endpoint.",
            file=sys.stderr,
        )
        raise exc


def run_robotics(*args: str) -> subprocess.CompletedProcess[str]:
    robotics_bin = os.environ.get("ROBOTICS_BIN")
    if robotics_bin:
        cmd = [robotics_bin, *args]
    else:
        cmd = ["cargo", "run", "-p", "robotics-cli", "--", *args]
    return subprocess.run(cmd, check=True, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
