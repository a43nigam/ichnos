import importlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from physicaldb import EgressLimitError, TemporalGapError, plan, plan_batch, query, query_batch


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
    cam_dir = euroc / "mav0" / "cam0"
    gt_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    (cam_dir / "data").mkdir(parents=True)
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
    (cam_dir / "data" / "1000000000.png").write_bytes(b"fake-camera-frame-1")
    (cam_dir / "data" / "1500000000.png").write_bytes(b"fake-camera-frame-2")
    (cam_dir / "data.csv").write_text(
        "#timestamp [ns],filename\n"
        "1000000000,1000000000.png\n"
        "1500000000,1500000000.png\n"
    )
    pose = tmp_path / "pose.parquet"
    imu = tmp_path / "imu.parquet"
    camera = tmp_path / "cam0.parquet"
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
            str(camera),
            "--out",
            str(media_catalog),
            "--modality",
            "camera",
            "--stream-id",
            "cam0",
            "--uri",
            camera.as_uri(),
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

    media_out = tmp_path / "media_frames"
    media_result = query(
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
    )
    np.testing.assert_allclose(media_result.tensor, result.tensor)
    assert media_result.media_manifest is not None
    assert len(media_result.media_manifest["frames"]) == 2
    assert (media_out / "cam0" / "1000000000.png").read_bytes() == b"fake-camera-frame-1"
    assert (media_out / "cam0" / "1500000000.png").read_bytes() == b"fake-camera-frame-2"
    assert media_result.diagnostics.range_enforced
    assert media_result.diagnostics.actual_authorized_bytes >= (
        media_result.diagnostics.pose_selected_bytes + media_result.diagnostics.imu_selected_bytes
    )
    assert media_result.manifest is not None
    assert media_result.manifest["media_materialization_manifest"]["frames"]

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

    blocked_media_out = tmp_path / "blocked_media"
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
            materialize_media=True,
            media_out=blocked_media_out,
            robotics_bin=tmp_path / "missing_robotics_binary",
        )
    assert not blocked_media_out.exists()

    with pytest.raises(RuntimeError, match="footer_allowance_bytes=1"):
        query(
            catalog_db=catalog_db,
            robot_id="mav0",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            predicate="velocity_magnitude > 1.5",
            channels=("pos_xyz", "camera:cam0"),
            target_hz=2.0,
            materialize_media=True,
            media_out=tmp_path / "media_low_footer",
            enforce_ranges=True,
            footer_allowance_bytes=1,
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


def test_query_batch_materializes_multiple_sessions(tmp_path: Path) -> None:
    import duckdb

    def write_fixture(root: Path, session: str, start_ns: int, frame_prefix: bytes) -> None:
        gt_dir = root / "mav0" / "state_groundtruth_estimate0"
        imu_dir = root / "mav0" / "imu0"
        cam_dir = root / "mav0" / "cam0"
        gt_dir.mkdir(parents=True)
        imu_dir.mkdir(parents=True)
        (cam_dir / "data").mkdir(parents=True)
        gt_dir.joinpath("data.csv").write_text(
            "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
            f"{start_ns},0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
            f"{start_ns + 500_000_000},1.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
            f"{start_ns + 1_000_000_000},2.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n",
            encoding="utf-8",
        )
        imu_dir.joinpath("data.csv").write_text(
            "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
            f"{start_ns - 100_000_000},0.1,0.2,0.3,9.0,0.0,-1.0\n"
            f"{start_ns + 250_000_000},0.2,0.3,0.4,10.0,1.0,-2.0\n"
            f"{start_ns + 750_000_000},0.4,0.5,0.6,12.0,3.0,-4.0\n"
            f"{start_ns + 1_100_000_000},0.5,0.6,0.7,13.0,4.0,-5.0\n",
            encoding="utf-8",
        )
        first_frame = f"{start_ns}.png"
        second_frame = f"{start_ns + 500_000_000}.png"
        (cam_dir / "data" / first_frame).write_bytes(frame_prefix + b"-frame-1")
        (cam_dir / "data" / second_frame).write_bytes(frame_prefix + b"-frame-2")
        cam_dir.joinpath("data.csv").write_text(
            f"#timestamp [ns],filename\n{start_ns},{first_frame}\n{start_ns + 500_000_000},{second_frame}\n",
            encoding="utf-8",
        )
        # Touch the session variable so fixture callers keep the generated paths readable.
        assert session

    def combine_catalogs(out: Path, first: Path, second: Path) -> None:
        def quote(path: Path) -> str:
            return str(path).replace("'", "''")

        with duckdb.connect(":memory:") as con:
            con.execute(
                "COPY (SELECT * FROM read_parquet(['{}', '{}'])) TO '{}' (FORMAT PARQUET)".format(
                    quote(first),
                    quote(second),
                    quote(out),
                )
            )

    session_specs = [
        ("session_a", 1_000_000_000, b"a"),
        ("session_b", 4_000_000_000, b"b"),
    ]
    pose_catalogs = []
    imu_catalogs = []
    media_catalogs = []
    for session, start_ns, prefix in session_specs:
        root = tmp_path / session / "euroc"
        write_fixture(root, session, start_ns, prefix)
        pose = tmp_path / session / "pose.parquet"
        imu = tmp_path / session / "imu.parquet"
        camera = tmp_path / session / "cam0.parquet"
        pose_catalog = tmp_path / session / "pose_catalog.parquet"
        imu_catalog = tmp_path / session / "imu_catalog.parquet"
        media_catalog = tmp_path / session / "media_catalog.parquet"
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
                str(root),
                "--out",
                str(pose),
                "--session-id",
                session,
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
                "ingest",
                "euroc-imu",
                "--input",
                str(root),
                "--out",
                str(imu),
                "--session-id",
                session,
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
                "ingest",
                "euroc-camera",
                "--input",
                str(root),
                "--out",
                str(camera),
                "--stream-id",
                "cam0",
                "--session-id",
                session,
                "--row-group-rows",
                "1",
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
                str(pose_catalog),
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
            ],
            check=True,
        )
        pose_catalogs.append(pose_catalog)
        imu_catalogs.append(imu_catalog)
        media_catalogs.append(media_catalog)

    combined_pose_catalog = tmp_path / "combined_pose_catalog.parquet"
    combined_imu_catalog = tmp_path / "combined_imu_catalog.parquet"
    combined_media_catalog = tmp_path / "combined_media_catalog.parquet"
    catalog_db = tmp_path / "fleet.duckdb"
    combine_catalogs(combined_pose_catalog, pose_catalogs[0], pose_catalogs[1])
    combine_catalogs(combined_imu_catalog, imu_catalogs[0], imu_catalogs[1])
    combine_catalogs(combined_media_catalog, media_catalogs[0], media_catalogs[1])
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
            str(combined_pose_catalog),
            "--imu-catalog",
            str(combined_imu_catalog),
            "--media-catalog",
            str(combined_media_catalog),
            "--out",
            str(catalog_db),
        ],
        check=True,
    )

    with pytest.raises(ValueError, match="query_batch"):
        query(
            catalog_db=catalog_db,
            robot_id="mav0",
            min_velocity=1.5,
            channels=("pos_xyz",),
        )

    batch_plan = plan_batch(
        catalog_db=catalog_db,
        robot_id="mav0",
        min_velocity=1.5,
        channels=("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0"),
        max_egress_bytes=10_000_000,
    )
    assert len(batch_plan.windows) == 2
    assert batch_plan.authorized_total_bytes == sum(
        window.authorized_total_bytes for window in batch_plan.windows
    )
    assert [window.pose_file_uri for window in batch_plan.windows] == [
        pose_catalogs[0].with_name("pose.parquet").as_uri(),
        pose_catalogs[1].with_name("pose.parquet").as_uri(),
    ]
    assert [window.diagnostics.media_matched_row_groups for window in batch_plan.windows] == [2, 2]
    assert not batch_plan.blocked_by_egress

    media_out = tmp_path / "batch_media"
    batch_result = query_batch(
        catalog_db=catalog_db,
        robot_id="mav0",
        min_velocity=1.5,
        channels=("pos_xyz", "imu_accel", "imu_gyro", "camera:cam0"),
        target_hz=2.0,
        materialize_media=True,
        media_out=media_out,
        enforce_ranges=True,
        manifest_out=tmp_path / "batch_manifest.json",
    )

    assert len(batch_result.windows) == 2
    assert batch_result.selected_bytes == batch_plan.authorized_total_bytes
    assert batch_result.diagnostics.pose_matched_row_groups == 4
    assert batch_result.diagnostics.imu_matched_row_groups == 4
    assert batch_result.diagnostics.media_matched_row_groups == 4
    assert batch_result.diagnostics.range_enforced
    assert batch_result.diagnostics.range_violations == 0
    assert batch_result.manifest is not None
    assert batch_result.manifest["window_count"] == 2
    assert json.loads((tmp_path / "batch_manifest.json").read_text())["window_count"] == 2
    for result in batch_result.windows:
        assert result.tensor.shape == (3, 9)
        assert result.media_manifest is not None
        assert len(result.media_manifest["frames"]) == 2
        assert result.diagnostics.range_enforced
        assert result.diagnostics.actual_cold_reads > 0
    assert (media_out / "window_000_session_a" / "cam0" / "1000000000.png").read_bytes() == b"a-frame-1"
    assert (media_out / "window_001_session_b" / "cam0" / "4000000000.png").read_bytes() == b"b-frame-1"

    blocked_out = tmp_path / "blocked_batch_media"
    with pytest.raises(EgressLimitError):
        query_batch(
            catalog_db=catalog_db,
            robot_id="mav0",
            min_velocity=1.5,
            channels=("pos_xyz", "camera:cam0"),
            max_egress_bytes=1,
            materialize_media=True,
            media_out=blocked_out,
            robotics_bin=tmp_path / "missing_robotics_binary",
        )
    assert not blocked_out.exists()


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

    import duckdb

    with duckdb.connect(str(catalog_db), read_only=True) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(pose_row_groups)").fetchall()}
    assert {"hilbert_xy", "hilbert_min_xy", "hilbert_max_xy", "time_bucket_ns"} <= columns

    query_module = importlib.import_module("physicaldb.query")
    rows, explain = query_module._query_pose_catalog(
        catalog=None,
        catalog_db=catalog_db,
        robot_id=None,
        session_id=None,
        start_ts_ns=None,
        end_ts_ns=None,
        bbox=(-55.0, -45.0, -30.0, -20.0, -2.0, 4.0),
        min_velocity=None,
        predicate_filters=query_module._parse_predicate(None),
        limit=None,
    )

    assert rows
    assert explain.candidate_row_groups == 128
    assert explain.index_strategy == "hilbert"
    assert explain.hilbert_pruned_row_groups >= 0
    assert explain.exact_spatial_pruned_row_groups >= 0
    assert explain.spatial_pruned_row_groups > 0
    assert len(rows) < explain.candidate_row_groups


def test_prove_euroc_hot_catalog_script_outputs_manifest(tmp_path: Path) -> None:
    euroc = tmp_path / "euroc"
    gt_dir = euroc / "mav0" / "state_groundtruth_estimate0"
    imu_dir = euroc / "mav0" / "imu0"
    cam_dir = euroc / "mav0" / "cam0"
    gt_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    (cam_dir / "data").mkdir(parents=True)
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

    work_dir = tmp_path / "proof"
    manifest_out = tmp_path / "proof_manifest.json"
    subprocess.run(
        [
            "python3",
            "scripts/prove_euroc_hot_catalog.py",
            "--input",
            str(euroc),
            "--work-dir",
            str(work_dir),
            "--iterations",
            "1",
            "--manifest-out",
            str(manifest_out),
        ],
        check=True,
    )

    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["query"]["predicate"]
    assert manifest["planning"]["iterations"] == 1
    assert manifest["plan"]["index_strategy"] == "hilbert"
    assert manifest["plan"]["authorized_pose_bytes"] > 0
    assert manifest["plan"]["authorized_imu_bytes"] > 0
    assert manifest["plan"]["authorized_media_bytes"] > 0
    assert manifest["egress_probe"]["blocked_by_egress"]
    assert manifest["materialization"]["tensor_shape"][1] == 9
    assert manifest["materialization"]["media_frame_count"] == 2
    assert manifest["materialization"]["diagnostics"]["range_enforced"]
    assert manifest["materialization"]["diagnostics"]["actual_cold_read_bytes"] > 0
    assert manifest["materialization"]["diagnostics"]["footer_bytes"] > 0


def test_validate_euroc_vicon_room1_script_on_synthetic_fixtures(tmp_path: Path) -> None:
    def write_fixture(root: Path, session: str, start_ns: int, prefix: bytes) -> None:
        gt_dir = root / "mav0" / "state_groundtruth_estimate0"
        imu_dir = root / "mav0" / "imu0"
        cam_dir = root / "mav0" / "cam0"
        gt_dir.mkdir(parents=True)
        imu_dir.mkdir(parents=True)
        (cam_dir / "data").mkdir(parents=True)
        gt_dir.joinpath("data.csv").write_text(
            "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
            f"{start_ns},0.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
            f"{start_ns + 500_000_000},1.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n"
            f"{start_ns + 1_000_000_000},2.0,0.0,0.0,1.0,0.0,0.0,0.0,2.0,0.0,0.0,0,0,0,0,0,0\n",
            encoding="utf-8",
        )
        imu_dir.joinpath("data.csv").write_text(
            "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
            f"{start_ns - 100_000_000},0.1,0.2,0.3,9.0,0.0,-1.0\n"
            f"{start_ns + 250_000_000},0.2,0.3,0.4,10.0,1.0,-2.0\n"
            f"{start_ns + 750_000_000},0.4,0.5,0.6,12.0,3.0,-4.0\n"
            f"{start_ns + 1_100_000_000},0.5,0.6,0.7,13.0,4.0,-5.0\n",
            encoding="utf-8",
        )
        first_frame = f"{start_ns}.png"
        second_frame = f"{start_ns + 500_000_000}.png"
        (cam_dir / "data" / first_frame).write_bytes(prefix + b"-frame-1")
        (cam_dir / "data" / second_frame).write_bytes(prefix + b"-frame-2")
        cam_dir.joinpath("data.csv").write_text(
            f"#timestamp [ns],filename\n{start_ns},{first_frame}\n{start_ns + 500_000_000},{second_frame}\n",
            encoding="utf-8",
        )
        assert session

    sessions = [
        ("V1_01_easy", 1_000_000_000, b"easy"),
        ("V1_02_medium", 4_000_000_000, b"medium"),
        ("V1_03_difficult", 7_000_000_000, b"difficult"),
    ]
    sequence_args = []
    for session, start_ns, prefix in sessions:
        root = tmp_path / "fixtures" / session
        write_fixture(root, session, start_ns, prefix)
        sequence_args.extend(["--sequence", f"{session}={root}"])

    output_root = tmp_path / "validation"
    subprocess.run(
        [
            "python3",
            "scripts/validate_euroc_vicon_room1.py",
            *sequence_args,
            "--output-root",
            str(output_root),
            "--iterations",
            "1",
            "--target-hz",
            "2.0",
            "--pose-row-group-rows",
            "2",
            "--imu-row-group-rows",
            "4",
            "--camera-row-group-rows",
            "1",
        ],
        check=True,
    )

    report = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    assert report["query"]["materialization"]["window_count"] == 3
    assert report["query"]["plan"]["window_count"] == 3
    assert report["query"]["egress_probe"]["blocked_by_egress"]
    assert not report["query"]["egress_probe"]["blocked_media_out_created"]
    assert report["query"]["materialization"]["diagnostics"]["range_enforced"]
    assert report["query"]["materialization"]["diagnostics"]["range_violations"] == 0
    assert report["query"]["materialization"]["diagnostics"]["actual_cold_read_bytes"] > 0
    assert report["query"]["materialization"]["diagnostics"]["footer_bytes"] > 0
    for window in report["query"]["materialization"]["windows"]:
        assert window["tensor_shape"][1] == 9
        assert window["timestamp_step_ns"] == 500_000_000
        assert window["media_frame_count"] >= 1
