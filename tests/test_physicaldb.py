import importlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from physicaldb import (
    AdapterRegistry,
    DatasetAdapter,
    EgressLimitError,
    TemporalGapError,
    check_quaternion_norms,
    check_timestamp_monotonicity,
    ingest_manifest,
    inspect_dataset,
    linear_motion_interpolation_benchmark,
    list_adapters,
    plan,
    plan_batch,
    query,
    query_batch,
    slerp_constant_angular_velocity_benchmark,
    suggest_manifest,
    validate_manifest,
)


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
    assert result.certificate is not None
    assert result.certificate.channels == ("pos_xyz", "rot_wxyz")
    assert result.certificate.shape == result.tensor.shape
    assert result.certificate.range_enforcement["enabled"]
    assert manifest["tensor_certificate"]["shape"] == list(result.tensor.shape)

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


def test_onboarding_inspect_suggest_and_validate_normalized_parquet(tmp_path: Path) -> None:
    source = tmp_path / "session.parquet"
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

    profile = inspect_dataset(source)
    assert profile.dataset_format == "parquet"
    assert profile.files[0].row_count is not None
    assert profile.streams[0].modality == "pose"
    assert "timestamp_ns" in profile.streams[0].timestamp_candidates

    manifest = suggest_manifest(
        profile,
        dataset_id="dataset_001",
        robot_id="humanoid_01",
        session_id="session_001",
    )
    report = validate_manifest(manifest)
    assert report.valid
    assert report.stream_count == 1
    assert report.modalities == ("pose",)
    assert profile.adapter_id == "normalized_parquet"
    assert manifest["adapter_id"] == "normalized_parquet"


def test_dataset_adapter_registry_and_cli_list_adapters() -> None:
    adapter_ids = [adapter.adapter_id for adapter in list_adapters()]
    assert adapter_ids[:3] == ["euroc", "normalized_parquet", "mcap_pose"]
    assert adapter_ids[-2:] == ["generic_dataset", "generic_media_placeholder"]
    assert "generic_media_placeholder" in adapter_ids

    completed = subprocess.run(
        ["python3", "-m", "physicaldb.onboarding_cli", "adapters"],
        env={**os.environ, "PYTHONPATH": "python"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "normalized_parquet" in completed.stdout
    assert "euroc" in completed.stdout


def test_dataset_adapter_unknown_id_error_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown dataset adapter 'missing_adapter'.*known adapters"):
        inspect_dataset(tmp_path, adapter_id="missing_adapter")


def test_dataset_adapter_entry_point_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEntryPoint:
        def load(self) -> type[DatasetAdapter]:
            return FakeAdapter

    class FakeEntryPoints:
        def select(self, *, group: str) -> list[FakeEntryPoint]:
            return [FakeEntryPoint()] if group == "physicaldb.dataset_adapters" else []

    class FakeAdapter(DatasetAdapter):
        adapter_id = "fake_external"

        def can_inspect(self, path_or_uri: str) -> bool:
            return False

        def inspect(self, path_or_uri: str):
            raise NotImplementedError

        def suggest_manifest(self, profile, *, dataset_id: str, robot_id: str, session_id: str, adapter_options=None):
            raise NotImplementedError

        def validate_manifest(self, manifest):
            raise NotImplementedError

        def ingest(self, manifest, *, output_root: str, row_group_rows: int = 500, robotics_bin=None):
            raise NotImplementedError

    import physicaldb.adapters as adapters

    monkeypatch.setattr(adapters.metadata, "entry_points", lambda: FakeEntryPoints())
    registry = AdapterRegistry(load_entry_points=True)
    assert registry.get("fake_external").adapter_id == "fake_external"


def test_dataset_adapter_auto_detects_existing_adapter_fixtures(tmp_path: Path) -> None:
    euroc = tmp_path / "euroc"
    (euroc / "mav0" / "state_groundtruth_estimate0").mkdir(parents=True)
    (euroc / "mav0" / "state_groundtruth_estimate0" / "data.csv").write_text("#timestamp,p_x\n", encoding="utf-8")
    assert inspect_dataset(euroc).adapter_id == "euroc"

    mcap = tmp_path / "session.mcap"
    mcap.write_bytes(b"not-a-real-mcap")
    assert inspect_dataset(mcap).adapter_id == "mcap_pose"

    kitti = tmp_path / "kitti" / "oxts" / "data"
    kitti.mkdir(parents=True)
    (kitti / "0000000000.txt").write_text("0 0 0\n", encoding="utf-8")
    assert inspect_dataset(kitti.parent.parent).adapter_id == "kitti_oxts"

    nuscenes = tmp_path / "nuscenes"
    nuscenes.mkdir()
    (nuscenes / "ego_pose.json").write_text("[]\n", encoding="utf-8")
    assert inspect_dataset(nuscenes).adapter_id == "nuscenes_ego"


def test_generic_dataset_inspects_csv_and_image_folder_mapping_draft(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    images = root / "cam0"
    images.mkdir(parents=True)
    (root / "pose.csv").write_text(
        "timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z\n"
        "100,0,0,0,1,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    (images / "100000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (images / "100000100.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    profile = inspect_dataset(root, adapter_id="generic_dataset")
    assert profile.adapter_id == "generic_dataset"
    assert profile.discovery["file_count"] >= 3
    modalities = {stream.modality for stream in profile.streams}
    assert {"pose", "camera"} <= modalities
    pose = next(stream for stream in profile.streams if stream.modality == "pose")
    assert pose.channels["x"] == "p_x"
    assert pose.confidence is not None and pose.confidence > 0.8

    manifest = suggest_manifest(
        profile,
        dataset_id="dataset_001",
        robot_id="robot_001",
        session_id="session_001",
        adapter_id="generic_dataset",
    )
    assert manifest["mapping_status"] == "draft"
    assert "confidence" in manifest
    report = validate_manifest(manifest, adapter_id="generic_dataset")
    assert report.valid
    assert any("draft" in warning for warning in report.warnings) is False


def test_generic_ambiguous_draft_validates_but_invalid_final_fails(tmp_path: Path) -> None:
    source = tmp_path / "misc.csv"
    source.write_text("foo,bar\n1,2\n", encoding="utf-8")
    profile = inspect_dataset(source, adapter_id="generic_dataset")
    manifest = suggest_manifest(
        profile,
        dataset_id="dataset_001",
        robot_id="robot_001",
        session_id="session_001",
        adapter_id="generic_dataset",
    )
    assert manifest["streams"][0]["mapping_status"] == "draft"
    draft_report = validate_manifest(manifest, adapter_id="generic_dataset")
    assert draft_report.valid
    assert draft_report.warnings

    final_manifest = dict(manifest)
    final_manifest["mapping_status"] = "final"
    final_manifest["streams"] = [dict(manifest["streams"][0], mapping_status="final", timestamp="", channels={})]
    final_report = validate_manifest(final_manifest, adapter_id="generic_dataset")
    assert not final_report.valid
    assert any("missing timestamp field" in error for error in final_report.errors)


def test_generic_final_csv_pose_imu_ingests_managed_outputs(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"
    imu = tmp_path / "imu.csv"
    pose.write_text(
        "timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z\n"
        "1000000000,0,0,0,1,0,0,0,1,0,0\n"
        "1100000000,1,0,0,1,0,0,0,1,0,0\n",
        encoding="utf-8",
    )
    imu.write_text(
        "timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
        "1000000000,0,0,9.8,0.1,0.2,0.3\n"
        "1100000000,0,1,9.8,0.2,0.3,0.4\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "final",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [
            {"source_id": "src_pose", "path": str(pose), "type": "csv"},
            {"source_id": "src_imu", "path": str(imu), "type": "csv"},
        ],
        "streams": [
            {
                "stream_id": "pose",
                "type": "pose",
                "modality": "pose",
                "source_id": "src_pose",
                "timestamp": "timestamp_ns",
                "channels": {
                    "x": "p_x",
                    "y": "p_y",
                    "z": "p_z",
                    "qw": "q_w",
                    "qx": "q_x",
                    "qy": "q_y",
                    "qz": "q_z",
                    "vx": "v_x",
                    "vy": "v_y",
                    "vz": "v_z",
                },
                "units": {},
                "frame_id": "base_link",
                "mapping_status": "final",
            },
            {
                "stream_id": "imu0",
                "type": "imu",
                "modality": "imu",
                "source_id": "src_imu",
                "timestamp": "timestamp_ns",
                "channels": {
                    "ax": "accel_x",
                    "ay": "accel_y",
                    "az": "accel_z",
                    "gx": "gyro_x",
                    "gy": "gyro_y",
                    "gz": "gyro_z",
                },
                "units": {},
                "frame_id": "imu0",
                "mapping_status": "final",
            },
        ],
    }
    report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_dataset")
    assert Path(report.outputs["pose_parquet"]).exists()
    assert Path(report.outputs["pose_catalog"]).exists()
    assert Path(report.outputs["imu0_parquet"]).exists()
    assert Path(report.outputs["imu0_catalog"]).exists()
    assert Path(report.outputs["catalog_db"]).exists()
    assert report.adapter_id == "generic_dataset"

    import duckdb

    with duckdb.connect(":memory:") as con:
        columns = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [report.outputs["pose_parquet"]]
            ).fetchall()
        }
    assert {"timestamp_ns", "robot_id", "session_id", "x", "qw", "velocity"} <= columns


def test_generic_csv_timestamp_unit_and_derived_pose_velocity(tmp_path: Path) -> None:
    pose = tmp_path / "pose_seconds.csv"
    pose.write_text(
        "time,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
        "0.0,0,0,0,1,0,0,0\n"
        "0.1,1,0,0,1,0,0,0\n"
        "0.2,3,0,0,1,0,0,0\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "final",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_pose", "path": str(pose), "type": "csv"}],
        "streams": [
            {
                "stream_id": "pose",
                "type": "pose",
                "modality": "pose",
                "source_id": "src_pose",
                "timestamp": "time",
                "timestamp_unit": "s",
                "channels": {"x": "p_x", "y": "p_y", "z": "p_z", "qw": "q_w", "qx": "q_x", "qy": "q_y", "qz": "q_z"},
                "units": {},
                "frame_id": "base_link",
                "mapping_status": "final",
            }
        ],
    }
    report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_dataset")

    import duckdb

    with duckdb.connect(":memory:") as con:
        rows = con.execute(
            "SELECT timestamp_ns, vx, vy, vz, velocity FROM read_parquet(?) ORDER BY timestamp_ns",
            [report.outputs["pose_parquet"]],
        ).fetchall()
    assert [row[0] for row in rows] == [0, 100_000_000, 200_000_000]
    assert [row[1] for row in rows] == pytest.approx([10.0, 10.0, 20.0])
    assert [row[2] for row in rows] == pytest.approx([0.0, 0.0, 0.0])
    assert [row[3] for row in rows] == pytest.approx([0.0, 0.0, 0.0])
    assert [row[4] for row in rows] == pytest.approx([10.0, 10.0, 20.0])


def test_generic_parquet_timestamp_scale_normalizes_to_ns(tmp_path: Path) -> None:
    source = tmp_path / "imu.parquet"
    import duckdb

    with duckdb.connect(":memory:") as con:
        con.execute(
            "CREATE TABLE imu AS SELECT * FROM (VALUES "
            "(1000, 0.0, 0.0, 9.8, 0.1, 0.2, 0.3), "
            "(1001, 0.0, 1.0, 9.8, 0.2, 0.3, 0.4)"
            ") AS t(t_ms, ax, ay, az, gx, gy, gz)"
        )
        con.execute(f"COPY imu TO '{str(source).replace(chr(39), chr(39) + chr(39))}' (FORMAT PARQUET)")
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "final",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_imu", "path": str(source), "type": "parquet"}],
        "streams": [
            {
                "stream_id": "imu0",
                "type": "imu",
                "modality": "imu",
                "source_id": "src_imu",
                "timestamp": "t_ms",
                "timestamp_scale": 1_000_000,
                "channels": {"ax": "ax", "ay": "ay", "az": "az", "gx": "gx", "gy": "gy", "gz": "gz"},
                "units": {},
                "frame_id": "imu0",
                "mapping_status": "final",
            }
        ],
    }
    report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_dataset")

    with duckdb.connect(":memory:") as con:
        timestamps = [
            row[0]
            for row in con.execute(
                "SELECT timestamp_ns FROM read_parquet(?) ORDER BY timestamp_ns",
                [report.outputs["imu0_parquet"]],
            ).fetchall()
        ]
    assert timestamps == [1_000_000_000, 1_001_000_000]


def test_generic_final_image_sequence_ingests_media_catalog(tmp_path: Path) -> None:
    images = tmp_path / "cam0"
    images.mkdir()
    (images / "1000000000.png").write_bytes(b"frame-a")
    (images / "1100000000.png").write_bytes(b"frame-b")
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "final",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_cam", "path": str(images), "type": "directory"}],
        "streams": [
            {
                "stream_id": "cam0",
                "type": "camera",
                "modality": "camera",
                "source_id": "src_cam",
                "timestamp": "timestamp_from_filename",
                "channels": {"frame_path": "path"},
                "units": {},
                "frame_id": "cam0",
                "mapping_status": "final",
            }
        ],
    }
    report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_dataset")
    assert Path(report.outputs["cam0_parquet"]).exists()
    assert Path(report.outputs["cam0_catalog"]).exists()


def test_generic_camera_sidecar_discovery_and_filename_timestamp_unit(tmp_path: Path) -> None:
    images = tmp_path / "cam0"
    images.mkdir()
    (images / "1000000.png").write_bytes(b"frame-a")
    (images / "1000001.png").write_bytes(b"frame-b")
    (images / "calibration.json").write_text(
        json.dumps(
            {
                "camera_model": "pinhole",
                "resolution": [640, 480],
                "intrinsics": [100.0, 101.0, 50.0, 51.0],
                "distortion_coefficients": [0.1, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    profile = inspect_dataset(images, adapter_id="generic_dataset")
    camera_stream = next(stream for stream in profile.streams if stream.modality == "camera")
    assert camera_stream.calibration is not None
    assert camera_stream.calibration["intrinsics"] == [100.0, 101.0, 50.0, 51.0]
    assert camera_stream.calibration["resolution"] == [640, 480]

    manifest = suggest_manifest(
        profile,
        dataset_id="dataset_001",
        robot_id="robot_001",
        session_id="session_001",
        adapter_id="generic_dataset",
    )
    manifest["mapping_status"] = "final"
    manifest["streams"][0]["mapping_status"] = "final"
    manifest["streams"][0]["timestamp_unit"] = "ms"
    assert manifest["streams"][0]["calibration"]["camera_model"] == "pinhole"

    report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_dataset")
    assert report.calibrations["cam0"]["intrinsics"] == [100.0, 101.0, 50.0, 51.0]

    import duckdb

    with duckdb.connect(":memory:") as con:
        timestamps = [
            row[0]
            for row in con.execute(
                "SELECT timestamp_ns FROM read_parquet(?) ORDER BY timestamp_ns",
                [report.outputs["cam0_parquet"]],
            ).fetchall()
        ]
    assert timestamps == [1_000_000_000_000, 1_000_001_000_000]


def test_generic_ingest_rejects_draft_and_s3_sources(tmp_path: Path) -> None:
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "draft",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_000", "path": "s3://robotics/pose.csv", "type": "csv"}],
        "streams": [
            {
                "stream_id": "pose",
                "type": "pose",
                "modality": "pose",
                "source_id": "src_000",
                "timestamp": "timestamp_ns",
                "channels": {"x": "x", "y": "y", "z": "z", "qw": "qw", "qx": "qx", "qy": "qy", "qz": "qz"},
                "units": {},
                "frame_id": "base_link",
                "mapping_status": "draft",
            }
        ],
    }
    output_root = tmp_path / "managed"
    with pytest.raises(ValueError, match="mapping_status='final'"):
        ingest_manifest(manifest, output_root=output_root, adapter_id="generic_dataset")
    assert not output_root.exists()

    manifest["mapping_status"] = "final"
    manifest["streams"][0]["mapping_status"] = "final"
    with pytest.raises(ValueError, match="S3 raw ingest is not supported"):
        ingest_manifest(manifest, output_root=output_root, adapter_id="generic_dataset")
    assert not output_root.exists()


def test_generic_ingest_preflight_failures_leave_no_outputs(tmp_path: Path) -> None:
    source = tmp_path / "pose.csv"
    source.write_text(
        "timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
        "not-a-timestamp,0,0,0,1,0,0,0\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "final",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_000", "path": str(source), "type": "csv"}],
        "streams": [
            {
                "stream_id": "pose",
                "type": "pose",
                "modality": "pose",
                "source_id": "src_000",
                "timestamp": "timestamp_ns",
                "channels": {
                    "x": "p_x",
                    "y": "p_y",
                    "z": "p_z",
                    "qw": "q_w",
                    "qx": "q_x",
                    "qy": "q_y",
                    "qz": "q_z",
                    "vx": "missing_velocity",
                },
                "units": {},
                "frame_id": "base_link",
                "mapping_status": "final",
            }
        ],
    }
    output_root = tmp_path / "managed"
    with pytest.raises(ValueError, match="missing_velocity"):
        ingest_manifest(manifest, output_root=output_root, adapter_id="generic_dataset")
    assert not output_root.exists()

    manifest["streams"][0]["channels"].pop("vx")
    with pytest.raises(ValueError, match="cannot be cast to BIGINT"):
        ingest_manifest(manifest, output_root=output_root, adapter_id="generic_dataset")
    assert not output_root.exists()

    output_root.mkdir()
    (output_root / "keep.txt").write_text("user data\n", encoding="utf-8")
    source.write_text(
        "timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
        "1000000000,0,0,0,1,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already exists and is not empty"):
        ingest_manifest(manifest, output_root=output_root, adapter_id="generic_dataset")
    assert (output_root / "keep.txt").read_text(encoding="utf-8") == "user data\n"


def test_generic_ingest_rejects_untimestamped_image_sequence_before_output(tmp_path: Path) -> None:
    images = tmp_path / "cam0"
    images.mkdir()
    (images / "frame.png").write_bytes(b"frame")
    manifest = {
        "version": 1,
        "adapter_id": "generic_dataset",
        "mapping_status": "final",
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_cam", "path": str(images), "type": "directory"}],
        "streams": [
            {
                "stream_id": "cam0",
                "type": "camera",
                "modality": "camera",
                "source_id": "src_cam",
                "timestamp": "timestamp_from_filename",
                "channels": {"frame_path": "path"},
                "units": {},
                "frame_id": "cam0",
                "mapping_status": "final",
            }
        ],
    }
    output_root = tmp_path / "managed"
    with pytest.raises(ValueError, match="parseable timestamps"):
        ingest_manifest(manifest, output_root=output_root, adapter_id="generic_dataset")
    assert not output_root.exists()


def test_generic_inspect_mapping_finalize_validate_ingest_workflow(tmp_path: Path) -> None:
    root = tmp_path / "customer"
    images = root / "cam0"
    images.mkdir(parents=True)
    (root / "pose.csv").write_text(
        "timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z\n"
        "1000000000,0,0,0,1,0,0,0,1,0,0\n"
        "1100000000,1,0,0,1,0,0,0,1,0,0\n",
        encoding="utf-8",
    )
    (images / "1000000000.png").write_bytes(b"frame-a")
    profile = inspect_dataset(root, adapter_id="generic_dataset")
    manifest = suggest_manifest(
        profile,
        dataset_id="dataset_001",
        robot_id="robot_001",
        session_id="session_001",
        adapter_id="generic_dataset",
    )
    manifest["mapping_status"] = "final"
    for stream in manifest["streams"]:
        stream["mapping_status"] = "final"
    report = validate_manifest(manifest, adapter_id="generic_dataset")
    assert report.valid

    ingest_report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_dataset")
    assert Path(ingest_report.outputs["pose_parquet"]).exists()
    assert Path(ingest_report.outputs["cam0_parquet"]).exists()


def test_dataset_cli_init_mapping_alias_for_generic_dataset(tmp_path: Path) -> None:
    source = tmp_path / "misc.csv"
    source.write_text("foo,bar\n1,2\n", encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    manifest_path = tmp_path / "dataset.json"
    validation_path = tmp_path / "validation.json"
    env = {**os.environ, "PYTHONPATH": "python"}

    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "inspect",
            "--adapter",
            "generic_dataset",
            "--input",
            str(source),
            "--out",
            str(profile_path),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "init-mapping",
            "--adapter",
            "generic_dataset",
            "--profile",
            str(profile_path),
            "--out",
            str(manifest_path),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "validate",
            "--manifest",
            str(manifest_path),
            "--out",
            str(validation_path),
        ],
        env=env,
        check=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["adapter_id"] == "generic_dataset"
    assert manifest["mapping_status"] == "draft"


def test_dataset_cli_demo_and_query_generic_customer_workflow(tmp_path: Path) -> None:
    workdir = tmp_path / "demo"
    env = {**os.environ, "PYTHONPATH": "python"}
    robotics_bin = _compatible_robotics_binary()
    if robotics_bin is None:
        pytest.skip("compatible robotics binary is not built")

    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "demo",
            "--workdir",
            str(workdir),
            "--row-group-rows",
            "24",
            "--target-hz",
            "10",
            "--robotics-bin",
            str(robotics_bin),
        ],
        env=env,
        check=True,
    )

    summary_path = workdir / "artifacts" / "demo_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["workflow"] == ["inspect", "init-mapping", "finalize", "validate", "ingest", "query"]
    assert summary["query"]["tensor_shape"][1] == 10
    assert Path(summary["artifacts"]["profile"]).exists()
    assert Path(summary["artifacts"]["draft_mapping"]).exists()
    assert Path(summary["artifacts"]["final_mapping"]).exists()
    assert Path(summary["ingest"]["outputs"]["catalog_db"]).exists()

    query_out = tmp_path / "query_summary.json"
    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "query",
            "--catalog-db",
            summary["ingest"]["outputs"]["catalog_db"],
            "--source",
            summary["ingest"]["outputs"]["pose_parquet"],
            "--robot-id",
            summary["robot_id"],
            "--session-id",
            summary["session_id"],
            "--channels",
            "pos_xyz,rot_wxyz",
            "--target-hz",
            "10",
            "--robotics-bin",
            str(robotics_bin),
            "--out",
            str(query_out),
        ],
        env=env,
        check=True,
    )

    query_summary = json.loads(query_out.read_text(encoding="utf-8"))
    assert query_summary["tensor_shape"][1] == 7
    assert query_summary["timestamp_count"] > 0
    assert query_summary["diagnostics"]["matched_row_groups"] > 0
    assert query_summary["tensor_certificate"]["channels"] == ["pos_xyz", "rot_wxyz"]


def _compatible_robotics_binary() -> Path | None:
    candidates = []
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    if target_dir:
        candidates.append(Path(target_dir) / "debug" / "robotics")
    candidates.extend([Path("/tmp/robotics-target/debug/robotics"), Path("target/debug/robotics")])
    for candidate in candidates:
        if not candidate.exists():
            continue
        completed = subprocess.run([str(candidate), "--help"], text=True, capture_output=True, check=False)
        help_text = completed.stdout + completed.stderr
        if "catalog duckdb-build" in help_text and "tensor imu-parquet-row-groups" in help_text:
            return candidate
    return None


def _write_euroc_onboarding_fixture(root: Path, *, calibrated: bool = True, invalid_cam_yaml: bool = False) -> None:
    gt_dir = root / "mav0" / "state_groundtruth_estimate0"
    imu_dir = root / "mav0" / "imu0"
    cam_dir = root / "mav0" / "cam0"
    gt_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    cam_dir.mkdir(parents=True)
    gt_dir.joinpath("data.csv").write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bgx,bgy,bgz,bax,bay,baz\n"
        "100,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    imu_dir.joinpath("data.csv").write_text(
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        "100,0,0,0,0,0,9.81\n",
        encoding="utf-8",
    )
    cam_dir.joinpath("data.csv").write_text("#timestamp [ns],filename\n100,frame.png\n", encoding="utf-8")
    if not calibrated:
        return
    imu_dir.joinpath("sensor.yaml").write_text(
        "sensor_type: imu\n"
        "comment: synthetic imu\n"
        "T_BS:\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  data: [1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.3, 0, 0, 0, 1]\n"
        "rate_hz: 200\n",
        encoding="utf-8",
    )
    cam_yaml = (
        "sensor_type: camera\n"
        "comment: synthetic cam0\n"
        "T_BS:\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  data: [1, 0, 0, 0.4, 0, 1, 0, 0.5, 0, 0, 1, 0.6, 0, 0, 0, 1]\n"
        "rate_hz: 20\n"
        "resolution: [752, 480]\n"
        "camera_model: pinhole\n"
        "intrinsics: [458.654, 457.296, 367.215, 248.375]\n"
        "distortion_model: radial-tangential\n"
        "distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]\n"
    )
    if invalid_cam_yaml:
        cam_yaml = "sensor_type: camera\nT_BS:\n  data: [1, 2\nresolution: not-a-list\n"
    cam_dir.joinpath("sensor.yaml").write_text(cam_yaml, encoding="utf-8")


def test_dataset_cli_inspect_init_validate_euroc_fixture(tmp_path: Path) -> None:
    root = tmp_path / "euroc"
    _write_euroc_onboarding_fixture(root)
    profile = tmp_path / "profile.json"
    manifest = tmp_path / "dataset.json"
    validation = tmp_path / "validation.json"
    env = {**os.environ, "PYTHONPATH": "python"}

    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "inspect",
            "--adapter",
            "auto",
            "--input",
            str(root),
            "--out",
            str(profile),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "init-manifest",
            "--adapter",
            "auto",
            "--profile",
            str(profile),
            "--out",
            str(manifest),
            "--dataset-id",
            "dataset_001",
            "--robot-id",
            "mav0",
            "--session-id",
            "V1_01_easy",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "-m",
            "physicaldb.onboarding_cli",
            "validate",
            "--manifest",
            str(manifest),
            "--out",
            str(validation),
        ],
        env=env,
        check=True,
    )

    profile_json = json.loads(profile.read_text(encoding="utf-8"))
    manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
    validation_json = json.loads(validation.read_text(encoding="utf-8"))
    assert profile_json["adapter_id"] == "euroc"
    assert manifest_json["adapter_id"] == "euroc"
    assert validation_json["valid"]
    profile_streams = {stream["stream_id"]: stream for stream in profile_json["streams"]}
    manifest_streams = {stream["stream_id"]: stream for stream in manifest_json["streams"]}
    assert profile_streams["imu0"]["calibration"]["rate_hz"] == 200.0
    assert profile_streams["cam0"]["calibration"]["resolution"] == [752, 480]
    assert manifest_streams["cam0"]["calibration"]["sensor_frame_id"] == "cam0"
    assert manifest_streams["cam0"]["calibration"]["body_frame_id"] == "body"
    assert manifest_streams["cam0"]["calibration"]["T_body_sensor"][3] == 0.4


def test_euroc_calibration_missing_warns_but_manifest_validates(tmp_path: Path) -> None:
    root = tmp_path / "euroc"
    _write_euroc_onboarding_fixture(root, calibrated=False)
    profile = inspect_dataset(root, adapter_id="euroc")
    assert any("EuRoC calibration missing for imu0" in warning for warning in profile.warnings)
    manifest = suggest_manifest(profile, dataset_id="dataset_001", robot_id="mav0", session_id="V1_01_easy")
    assert "calibration" not in {stream["stream_id"]: stream for stream in manifest["streams"]}["imu0"]
    assert validate_manifest(manifest).valid


def test_euroc_partial_calibration_warns_and_preserves_parseable_metadata(tmp_path: Path) -> None:
    root = tmp_path / "euroc"
    _write_euroc_onboarding_fixture(root, invalid_cam_yaml=True)
    profile = inspect_dataset(root, adapter_id="euroc")
    cam0 = {stream.stream_id: stream for stream in profile.streams}["cam0"]
    assert cam0.calibration is not None
    assert cam0.calibration["sensor_type"] == "camera"
    assert cam0.calibration["sensor_frame_id"] == "cam0"
    assert "T_body_sensor" not in cam0.calibration
    assert any("could not parse value" in warning or "invalid resolution" in warning for warning in profile.warnings)
    manifest = suggest_manifest(profile, dataset_id="dataset_001", robot_id="mav0", session_id="V1_01_easy")
    assert validate_manifest(manifest).valid


def test_ingest_report_includes_stream_calibration_summary(tmp_path: Path) -> None:
    manifest = {
        "version": 1,
        "dataset_id": "dataset_001",
        "robot_id": "mav0",
        "session_id": "V1_01_easy",
        "sources": [{"source_id": "src_000", "path": str(tmp_path / "raw.bin"), "type": "file"}],
        "streams": [
            {
                "stream_id": "cam0",
                "type": "camera",
                "modality": "camera",
                "source_id": "src_000",
                "timestamp": "#timestamp [ns]",
                "channels": {"frame_path": "filename", "camera_bytes": "data"},
                "units": {},
                "frame_id": "cam0",
                "calibration": {"sensor_frame_id": "cam0", "body_frame_id": "body", "rate_hz": 20.0},
            }
        ],
    }
    report = ingest_manifest(manifest, output_root=tmp_path / "managed", adapter_id="generic_media_placeholder")
    assert report.calibrations == {"cam0": {"sensor_frame_id": "cam0", "body_frame_id": "body", "rate_hz": 20.0}}


def test_onboarding_validation_rejects_bad_manifests() -> None:
    base = {
        "version": 1,
        "dataset_id": "dataset_001",
        "robot_id": "robot_001",
        "session_id": "session_001",
        "sources": [{"source_id": "src_000", "path": "pose.parquet", "type": "parquet"}],
        "streams": [
            {
                "stream_id": "pose",
                "type": "pose",
                "modality": "pose",
                "source_id": "src_000",
                "timestamp": "timestamp_ns",
                "channels": {"x": "x", "y": "y", "z": "z", "qw": "qw", "qx": "qx", "qy": "qy", "qz": "qz"},
                "units": {"x": "m"},
                "frame_id": "world",
            }
        ],
    }
    assert validate_manifest(base).valid

    missing_timestamp = json.loads(json.dumps(base))
    missing_timestamp["streams"][0]["timestamp"] = ""
    assert not validate_manifest(missing_timestamp).valid

    unsupported_modality = json.loads(json.dumps(base))
    unsupported_modality["streams"][0]["modality"] = "lidar"
    assert not validate_manifest(unsupported_modality).valid

    duplicate_stream = json.loads(json.dumps(base))
    duplicate_stream["streams"].append(json.loads(json.dumps(duplicate_stream["streams"][0])))
    assert not validate_manifest(duplicate_stream).valid

    unknown_channel = json.loads(json.dumps(base))
    unknown_channel["streams"][0]["channels"]["unknown"] = "field"
    assert not validate_manifest(unknown_channel).valid


def test_trust_check_helpers_are_deterministic() -> None:
    linear = linear_motion_interpolation_benchmark()
    assert linear["max_error"] == 0.0
    slerp = slerp_constant_angular_velocity_benchmark()
    assert slerp["max_error_rad"] < 1e-7
    assert check_quaternion_norms(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert not check_quaternion_norms(np.array([[2.0, 0.0, 0.0, 0.0]]))
    assert check_timestamp_monotonicity(np.array([1, 1, 2], dtype=np.int64))
    assert not check_timestamp_monotonicity(np.array([1, 0, 2], dtype=np.int64))


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
