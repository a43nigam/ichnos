import importlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from physicaldb import EgressLimitError, TemporalGapError, plan, query


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
        enforce_ranges=True,
        manifest_out=tmp_path / "query.manifest.json",
    )

    assert isinstance(result.tensor, np.ndarray)
    assert result.tensor.shape == (15, 7)
    assert result.timestamps_ns.shape == (15,)
    assert result.selected_bytes > 0
    assert result.row_groups == (0,)
    assert result.diagnostics.range_enforced
    assert result.diagnostics.actual_cold_reads > 0
    assert result.diagnostics.footer_bytes > 0
    assert result.diagnostics.footer_allowance_bytes == 16 * 1024 * 1024
    assert result.diagnostics.largest_metadata_read > 0
    assert result.diagnostics.actual_authorized_bytes > 0
    assert result.diagnostics.range_violations == 0
    assert result.manifest is not None
    manifest = json.loads((tmp_path / "query.manifest.json").read_text())
    assert manifest["enforcement_enabled"]
    assert manifest["footer_allowance_bytes"] == 16 * 1024 * 1024
    assert manifest["authorized_row_group_bytes"] == result.diagnostics.pose_selected_bytes
    assert manifest["actual_cold_reads"] == result.diagnostics.actual_cold_reads

    with pytest.raises(RuntimeError, match="footer_allowance_bytes=1"):
        query(
            catalog=catalog,
            robot_id="humanoid_01",
            start_ts_ns=0,
            end_ts_ns=480_000_000,
            bbox=(-0.1, 2.0, -1.1, 1.1, -0.1, 0.1),
            min_velocity=2.0,
            channels=("pos_xyz",),
            target_hz=30.0,
            output="numpy",
            source=source,
            enforce_ranges=True,
            footer_allowance_bytes=1,
        )


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

    cold_plan = plan(
        catalog=catalog,
        source=tmp_path / "missing_pose_source.parquet",
        max_egress_bytes=1,
    )

    assert cold_plan.blocked_by_egress
    assert cold_plan.pose_file_uri == str(tmp_path / "missing_pose_source.parquet")
    assert cold_plan.authorized_pose_bytes > 0
    assert cold_plan.authorized_total_bytes == cold_plan.authorized_pose_bytes
    assert cold_plan.materialized_pose_imu_bytes == cold_plan.authorized_pose_bytes
    assert cold_plan.planned_range_reads == len(cold_plan.pose_row_groups)
    assert cold_plan.diagnostics.blocked_by_egress

    with pytest.raises(EgressLimitError):
        query(
            catalog=catalog,
            source=source,
            max_egress_bytes=1,
            robotics_bin=tmp_path / "missing_robotics_binary",
        )


def test_query_returns_synchronized_pose_and_imu_tensor(tmp_path: Path) -> None:
    euroc = tmp_path / "euroc"
    gt_dir = euroc / "mav0" / "state_groundtruth_estimate0"
    imu_dir = euroc / "mav0" / "imu0"
    gt_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    (gt_dir / "data.csv").write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
        "1000000000,0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
        "1500000000,1.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
        "2000000000,2.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
    )
    (imu_dir / "data.csv").write_text(
        "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
        "900000000,0.1,0.2,0.3,9.0,0.0,-1.0\n"
        "1250000000,0.2,0.3,0.4,10.0,1.0,-2.0\n"
        "1750000000,0.4,0.5,0.6,12.0,3.0,-4.0\n"
        "2100000000,0.5,0.6,0.7,13.0,4.0,-5.0\n"
    )
    pose = tmp_path / "pose.parquet"
    imu = tmp_path / "imu.parquet"
    catalog = tmp_path / "catalog.parquet"
    imu_catalog = tmp_path / "imu_catalog.parquet"
    media_catalog = tmp_path / "media_catalog.parquet"
    catalog_db = tmp_path / "fleet.duckdb"
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "ingest",
            "euroc-groundtruth",
            "--input",
            str(euroc),
            "--out",
            str(pose),
            "--row-group-rows",
            "2",
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
            "file:///tmp/cam0.parquet",
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
            "ingest",
            "euroc-imu",
            "--input",
            str(euroc),
            "--out",
            str(imu),
            "--row-group-rows",
            "2",
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
            str(pose),
            "--out",
            str(catalog),
            "--uri",
            pose.as_uri(),
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
            "build-imu",
            "--input",
            str(imu),
            "--out",
            str(imu_catalog),
            "--uri",
            imu.as_uri(),
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
            "duckdb-build",
            "--pose-catalog",
            str(catalog),
            "--imu-catalog",
            str(imu_catalog),
            "--media-catalog",
            str(media_catalog),
            "--out",
            str(catalog_db),
        ],
        check=True,
    )

    result = query(
        catalog=catalog,
        robot_id="mav0",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        min_velocity=1.5,
        channels=("pos_xyz", "imu_accel", "imu_gyro"),
        target_hz=2.0,
        source=pose,
        imu_source=imu,
        imu_catalog=imu_catalog,
        enforce_ranges=True,
    )

    assert result.tensor.shape == (3, 9)
    np.testing.assert_allclose(
        result.tensor[1],
        np.array([1.0, 0.0, 0.0, 11.0, 2.0, -3.0, 0.3, 0.4, 0.5]),
    )
    assert result.row_groups == (0, 1)
    assert result.diagnostics.velocity_threshold == 1.5
    assert result.diagnostics.matched_row_groups == 2
    assert result.diagnostics.pose_matched_row_groups == 2
    assert result.diagnostics.imu_matched_row_groups == 2
    assert result.diagnostics.pose_selected_bytes > 0
    assert result.diagnostics.imu_selected_bytes > 0
    assert (
        result.diagnostics.total_selected_bytes
        == result.diagnostics.pose_selected_bytes + result.diagnostics.imu_selected_bytes
    )
    assert result.diagnostics.pose_planned_range_reads == 2
    assert result.diagnostics.imu_planned_range_reads == 2
    assert result.diagnostics.planned_read_bytes == result.diagnostics.total_selected_bytes
    assert result.diagnostics.range_audit_passed
    assert result.diagnostics.range_enforced
    assert result.diagnostics.actual_cold_reads > 0
    assert result.diagnostics.footer_bytes > 0
    assert result.diagnostics.range_violations == 0
    assert result.diagnostics.null_count == 0
    assert result.diagnostics.extrapolation_checked
    assert not result.diagnostics.extrapolation_rejected

    catalog_uri_result = query(
        catalog=catalog,
        robot_id="mav0",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        min_velocity=1.5,
        channels=("pos_xyz", "imu_accel", "imu_gyro"),
        target_hz=2.0,
        imu_catalog=imu_catalog,
    )
    np.testing.assert_allclose(catalog_uri_result.tensor, result.tensor)
    assert catalog_uri_result.file_uri == pose.as_uri()

    catalog_db_result = query(
        catalog_db=catalog_db,
        robot_id="mav0",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        predicate=(
            "velocity_magnitude > 1.5 AND "
            "ST_Intersects(position, bbox(-1,3,-1,1,-1,1))"
        ),
        channels=("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0"),
        target_hz=2.0,
    )
    np.testing.assert_allclose(catalog_db_result.tensor, result.tensor)
    assert (
        catalog_db_result.diagnostics.selected_bytes
        == result.diagnostics.selected_bytes + catalog_db_result.diagnostics.media_selected_bytes
    )
    assert catalog_db_result.diagnostics.range_audit_passed
    assert catalog_db_result.diagnostics.catalog_query_ms >= 0.0
    assert catalog_db_result.diagnostics.candidate_row_groups >= catalog_db_result.diagnostics.matched_row_groups
    assert catalog_db_result.diagnostics.media_matched_row_groups == 2
    assert catalog_db_result.diagnostics.media_selected_bytes > 0
    assert not catalog_db_result.diagnostics.media_blocked_by_egress
    assert catalog_db_result.diagnostics.authorized_total_bytes == catalog_db_result.diagnostics.total_selected_bytes
    assert (
        catalog_db_result.diagnostics.materialized_pose_imu_bytes
        == catalog_db_result.diagnostics.pose_selected_bytes
        + catalog_db_result.diagnostics.imu_selected_bytes
    )
    assert catalog_db_result.diagnostics.planned_range_reads == (
        catalog_db_result.diagnostics.pose_planned_range_reads
        + catalog_db_result.diagnostics.imu_planned_range_reads
    )

    catalog_db_plan = plan(
        catalog_db=catalog_db,
        robot_id="mav0",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        predicate=(
            "velocity_magnitude > 1.5 AND "
            "ST_Intersects(position, bbox(-1,3,-1,1,-1,1))"
        ),
        channels=("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0"),
        max_egress_bytes=10_000_000,
    )

    assert catalog_db_plan.row_groups == catalog_db_result.row_groups
    assert catalog_db_plan.authorized_total_bytes == catalog_db_result.diagnostics.authorized_total_bytes
    assert catalog_db_plan.materialized_pose_imu_bytes == catalog_db_result.diagnostics.materialized_pose_imu_bytes
    assert catalog_db_plan.authorized_media_bytes == catalog_db_result.diagnostics.authorized_media_bytes
    assert not catalog_db_plan.blocked_by_egress

    blocked_media_plan = plan(
        catalog_db=catalog_db,
        robot_id="mav0",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        predicate="velocity_magnitude > 1.5",
        channels=("pos_xyz", "camera:cam0"),
        max_egress_bytes=result.diagnostics.pose_selected_bytes,
    )
    assert blocked_media_plan.blocked_by_egress
    assert blocked_media_plan.diagnostics.media_blocked_by_egress
    assert blocked_media_plan.authorized_media_bytes > 0

    with pytest.raises(EgressLimitError):
        query(
            catalog_db=catalog_db,
            robot_id="mav0",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            predicate="velocity_magnitude > 1.5",
            channels=("pos_xyz", "camera:cam0"),
            target_hz=2.0,
            max_egress_bytes=1,
            robotics_bin=tmp_path / "missing_robotics_binary",
        )

    with pytest.raises(ValueError, match="unsupported predicate"):
        query(
            catalog_db=catalog_db,
            robot_id="mav0",
            predicate="speed > 1.5",
            channels=("pos_xyz",),
        )

    with pytest.raises(ValueError, match="imu_catalog"):
        query(
            catalog=catalog,
            robot_id="mav0",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            min_velocity=1.5,
            channels=("pos_xyz", "imu_accel"),
            target_hz=2.0,
            source=pose,
            imu_source=imu,
        )


def test_query_rejects_temporal_gap_by_default(tmp_path: Path) -> None:
    euroc = tmp_path / "euroc_gap"
    gt_dir = euroc / "mav0" / "state_groundtruth_estimate0"
    gt_dir.mkdir(parents=True)
    (gt_dir / "data.csv").write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
        "0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0,0,0,0,0,0\n"
        "100000000,0.1,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0,0,0,0,0,0\n"
        "200000000,0.2,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0,0,0,0,0,0\n"
        "1000000000,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0,0,0,0,0,0\n"
    )
    pose = tmp_path / "pose_gap.parquet"
    catalog = tmp_path / "catalog_gap.parquet"
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "ingest",
            "euroc-groundtruth",
            "--input",
            str(euroc),
            "--out",
            str(pose),
            "--row-group-rows",
            "4",
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
            str(pose),
            "--out",
            str(catalog),
            "--uri",
            pose.as_uri(),
        ],
        check=True,
    )

    with pytest.raises(TemporalGapError):
        query(
            catalog=catalog,
            robot_id="mav0",
            start_ts_ns=0,
            end_ts_ns=1_000_000_000,
            channels=("pos_xyz",),
            target_hz=2.0,
        )

    result = query(
        catalog=catalog,
        robot_id="mav0",
        start_ts_ns=0,
        end_ts_ns=1_000_000_000,
        channels=("pos_xyz",),
        target_hz=2.0,
        gap_policy="allow",
    )

    assert result.tensor.shape == (3, 3)
    assert result.diagnostics.pose_gap_count == 1
    assert result.diagnostics.pose_max_gap_ns == 800_000_000
    assert result.diagnostics.pose_max_gap_start_ts_ns == 200_000_000
    assert result.diagnostics.pose_max_gap_end_ts_ns == 1_000_000_000


def test_fake_catalog_db_spatial_explain_prunes_row_groups(tmp_path: Path) -> None:
    catalog_db = tmp_path / "fake_fleet.duckdb"
    subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "robotics-cli",
            "--",
            "catalog",
            "fake-duckdb",
            "--sessions",
            "128",
            "--out",
            str(catalog_db),
        ],
        check=True,
    )

    query_module = importlib.import_module("physicaldb.query")
    rows, explain = query_module._query_pose_catalog(
        catalog=None,
        catalog_db=catalog_db,
        robot_id=None,
        start_ts_ns=None,
        end_ts_ns=None,
        bbox=(-55.0, -45.0, -30.0, -20.0, -2.0, 4.0),
        min_velocity=None,
        predicate_filters=query_module._parse_predicate(None),
        limit=None,
    )

    assert rows
    assert explain.candidate_row_groups == 128
    assert explain.spatial_pruned_row_groups > 0
    assert len(rows) < explain.candidate_row_groups
