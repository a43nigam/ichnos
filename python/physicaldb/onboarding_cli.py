from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .onboarding import (
    DatasetProfile,
    ingest_manifest,
    inspect_dataset,
    suggest_manifest,
    validate_manifest,
)
from .adapters import list_adapters
from .query import _robotics_command, query, query_managed_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotics dataset")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("adapters")

    inspect_parser = subcommands.add_parser("inspect")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--out", required=True)
    inspect_parser.add_argument("--adapter", default="auto")

    for command_name in ("init-manifest", "init-mapping"):
        init_parser = subcommands.add_parser(command_name)
        init_parser.add_argument("--profile", required=True)
        init_parser.add_argument("--out", required=True)
        init_parser.add_argument("--dataset-id", default="dataset_001")
        init_parser.add_argument("--robot-id", default="robot_001")
        init_parser.add_argument("--session-id", default="session_001")
        init_parser.add_argument("--adapter", default="auto")
        init_parser.add_argument("--adapter-option", action="append", default=[])

    validate_parser = subcommands.add_parser("validate")
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--out", required=True)
    validate_parser.add_argument("--adapter", default="auto")

    ingest_parser = subcommands.add_parser("ingest")
    ingest_parser.add_argument("--manifest", required=True)
    ingest_parser.add_argument("--output-root", required=True)
    ingest_parser.add_argument("--row-group-rows", type=int, default=500)
    ingest_parser.add_argument("--out")
    ingest_parser.add_argument("--adapter", default="auto")

    query_parser = subcommands.add_parser("query")
    query_parser.add_argument("--catalog")
    query_parser.add_argument("--catalog-db")
    query_parser.add_argument("--managed-root")
    query_parser.add_argument("--robot-id")
    query_parser.add_argument("--session-id")
    query_parser.add_argument("--start-ts-ns", type=int)
    query_parser.add_argument("--end-ts-ns", type=int)
    query_parser.add_argument("--bbox", nargs=6, type=float)
    query_parser.add_argument("--min-velocity", type=float)
    query_parser.add_argument("--predicate")
    query_parser.add_argument("--channels", default="pos_xyz,rot_wxyz,vel_xyz")
    query_parser.add_argument("--target-hz", type=float, default=30.0)
    query_parser.add_argument("--output", choices=("numpy", "torch"), default="numpy")
    query_parser.add_argument("--source")
    query_parser.add_argument("--imu-source")
    query_parser.add_argument("--imu-catalog")
    query_parser.add_argument("--media-catalog")
    query_parser.add_argument("--max-egress-bytes", type=int, default=1_000_000_000)
    query_parser.add_argument("--limit", type=int)
    query_parser.add_argument("--gap-policy", choices=("reject", "allow"), default="reject")
    query_parser.add_argument("--enforce-ranges", action="store_true")
    query_parser.add_argument("--footer-allowance-bytes", type=int, default=16 * 1024 * 1024)
    query_parser.add_argument("--manifest-out")
    query_parser.add_argument("--materialize-media", action="store_true")
    query_parser.add_argument("--media-out")
    query_parser.add_argument("--robotics-bin")
    query_parser.add_argument("--out", required=True)

    stage_parser = subcommands.add_parser("stage-s3")
    stage_parser.add_argument("--input", required=True)
    stage_parser.add_argument("--out", required=True)
    stage_parser.add_argument("--manifest", required=True)
    stage_parser.add_argument("--limit", type=int, default=500)
    stage_parser.add_argument("--robotics-bin")

    upload_parser = subcommands.add_parser("upload-managed")
    upload_parser.add_argument("--managed-root", required=True)
    upload_parser.add_argument("--uri", required=True)
    upload_parser.add_argument("--manifest", required=True)
    upload_parser.add_argument("--robotics-bin")

    demo_parser = subcommands.add_parser("demo")
    demo_parser.add_argument("--workdir", default="data/demo/generic_customer")
    demo_parser.add_argument("--dataset-id", default="generic_customer_m1")
    demo_parser.add_argument("--robot-id", default="customer_bot_001")
    demo_parser.add_argument("--session-id", default="demo_session_001")
    demo_parser.add_argument("--row-group-rows", type=int, default=8)
    demo_parser.add_argument("--target-hz", type=float, default=10.0)
    demo_parser.add_argument("--robotics-bin")
    demo_parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "adapters":
        for adapter in list_adapters():
            print(f"{adapter.adapter_id}\t{getattr(adapter, 'version', '1')}")
        return 0
    if args.command == "inspect":
        profile = inspect_dataset(args.input, adapter_id=args.adapter)
        _write_json(Path(args.out), profile.to_dict())
        return 0
    if args.command in {"init-manifest", "init-mapping"}:
        profile = DatasetProfile.from_dict(_read_json(Path(args.profile)))
        manifest = suggest_manifest(
            profile,
            dataset_id=args.dataset_id,
            robot_id=args.robot_id,
            session_id=args.session_id,
            adapter_id=args.adapter,
            adapter_options=_parse_adapter_options(args.adapter_option),
        )
        _write_json(Path(args.out), manifest)
        return 0
    if args.command == "validate":
        report = validate_manifest(Path(args.manifest), adapter_id=args.adapter)
        _write_json(Path(args.out), report.to_dict())
        return 0 if report.valid else 1
    if args.command == "ingest":
        report = ingest_manifest(
            Path(args.manifest),
            output_root=args.output_root,
            row_group_rows=args.row_group_rows,
            adapter_id=args.adapter,
        )
        out = Path(args.out) if args.out else Path(args.output_root) / "ingest_report.json"
        _write_json(out, report.to_dict())
        return 0
    if args.command == "query":
        if args.managed_root is not None:
            result = query_managed_dataset(
                args.managed_root,
                robot_id=args.robot_id,
                session_id=args.session_id,
                start_ts_ns=args.start_ts_ns,
                end_ts_ns=args.end_ts_ns,
                bbox=tuple(args.bbox) if args.bbox is not None else None,
                min_velocity=args.min_velocity,
                predicate=args.predicate,
                channels=_parse_channels(args.channels),
                target_hz=args.target_hz,
                output=args.output,
                max_egress_bytes=args.max_egress_bytes,
                limit=args.limit,
                gap_policy=args.gap_policy,
                enforce_ranges=args.enforce_ranges,
                footer_allowance_bytes=args.footer_allowance_bytes,
                manifest_out=args.manifest_out,
                materialize_media=args.materialize_media,
                media_out=args.media_out,
                robotics_bin=args.robotics_bin,
            )
        else:
            if args.catalog is None and args.catalog_db is None:
                raise SystemExit("dataset query requires --managed-root, --catalog, or --catalog-db")
            result = query(
                catalog=args.catalog,
                catalog_db=args.catalog_db,
                robot_id=args.robot_id,
                session_id=args.session_id,
                start_ts_ns=args.start_ts_ns,
                end_ts_ns=args.end_ts_ns,
                bbox=tuple(args.bbox) if args.bbox is not None else None,
                min_velocity=args.min_velocity,
                predicate=args.predicate,
                channels=_parse_channels(args.channels),
                target_hz=args.target_hz,
                output=args.output,
                source=args.source,
                imu_source=args.imu_source,
                imu_catalog=args.imu_catalog,
                media_catalog=args.media_catalog,
                max_egress_bytes=args.max_egress_bytes,
                limit=args.limit,
                gap_policy=args.gap_policy,
                enforce_ranges=args.enforce_ranges,
                footer_allowance_bytes=args.footer_allowance_bytes,
                manifest_out=args.manifest_out,
                materialize_media=args.materialize_media,
                media_out=args.media_out,
                robotics_bin=args.robotics_bin,
            )
        _write_json(Path(args.out), _query_summary(result))
        return 0
    if args.command == "stage-s3":
        manifest = stage_s3_prefix(
            args.input,
            Path(args.out),
            manifest_path=Path(args.manifest),
            limit=args.limit,
            robotics_bin=args.robotics_bin,
        )
        _write_json(Path(args.manifest), manifest)
        return 0
    if args.command == "upload-managed":
        manifest = upload_managed_dataset(
            Path(args.managed_root),
            args.uri,
            manifest_path=Path(args.manifest),
            robotics_bin=args.robotics_bin,
        )
        _write_json(Path(args.manifest), manifest)
        return 0
    if args.command == "demo":
        run_demo_workflow(
            Path(args.workdir),
            dataset_id=args.dataset_id,
            robot_id=args.robot_id,
            session_id=args.session_id,
            row_group_rows=args.row_group_rows,
            target_hz=args.target_hz,
            robotics_bin=args.robotics_bin,
            force=args.force,
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"out={path}")


def _parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(channel.strip() for channel in value.split(",") if channel.strip())
    if not channels:
        raise SystemExit("--channels must include at least one channel group")
    return channels


def _parse_adapter_options(values: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for value in values:
        key, separator, option_value = value.partition("=")
        if not separator or not key:
            raise SystemExit(f"--adapter-option must be KEY=VALUE, got {value!r}")
        options[key] = option_value
    return options


def run_demo_workflow(
    workdir: Path,
    *,
    dataset_id: str = "generic_customer_m1",
    robot_id: str = "customer_bot_001",
    session_id: str = "demo_session_001",
    row_group_rows: int = 8,
    target_hz: float = 10.0,
    robotics_bin: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if force and workdir.exists():
        shutil.rmtree(workdir)
    raw_root = workdir / "raw_customer_drop"
    artifacts = workdir / "artifacts"
    managed = workdir / "managed"
    if managed.exists() and any(managed.iterdir()):
        raise SystemExit(f"{managed} is not empty; pass --force to regenerate the demo")

    fixture = _write_messy_customer_fixture(raw_root)
    profile = inspect_dataset(raw_root, adapter_id="generic_dataset")
    draft = suggest_manifest(
        profile,
        dataset_id=dataset_id,
        robot_id=robot_id,
        session_id=session_id,
        adapter_id="generic_dataset",
    )
    final_manifest = _finalize_demo_mapping(draft, fixture)
    validation = validate_manifest(final_manifest, adapter_id="generic_dataset")
    if not validation.valid:
        raise SystemExit("final demo mapping did not validate: " + "; ".join(validation.errors))

    ingest_report = ingest_manifest(
        final_manifest,
        output_root=managed,
        row_group_rows=row_group_rows,
        adapter_id="generic_dataset",
        robotics_bin=robotics_bin,
    )
    query_manifest = artifacts / "query_manifest.json"
    query_result = query(
        catalog_db=ingest_report.outputs["catalog_db"],
        robot_id=robot_id,
        session_id=session_id,
        start_ts_ns=fixture["start_ts_ns"],
        end_ts_ns=fixture["end_ts_ns"],
        bbox=(-1.0, 20.0, -2.0, 4.0, -1.0, 2.0),
        min_velocity=0.1,
        channels=("pos_xyz", "rot_wxyz", "vel_xyz"),
        target_hz=target_hz,
        output="numpy",
        source=ingest_report.outputs["pose_parquet"],
        manifest_out=query_manifest,
        robotics_bin=robotics_bin,
    )

    paths = {
        "raw_root": raw_root,
        "profile": artifacts / "profile.json",
        "draft_mapping": artifacts / "mapping.draft.json",
        "final_mapping": artifacts / "mapping.final.json",
        "validation": artifacts / "validation.json",
        "ingest_report": artifacts / "ingest_report.json",
        "query_summary": artifacts / "query_summary.json",
        "query_manifest": query_manifest,
        "summary": artifacts / "demo_summary.json",
    }
    _write_json(paths["profile"], profile.to_dict())
    _write_json(paths["draft_mapping"], draft)
    _write_json(paths["final_mapping"], final_manifest)
    _write_json(paths["validation"], validation.to_dict())
    _write_json(paths["ingest_report"], ingest_report.to_dict())
    query_summary = _query_summary(query_result)
    _write_json(paths["query_summary"], query_summary)

    summary = {
        "dataset_id": dataset_id,
        "robot_id": robot_id,
        "session_id": session_id,
        "workflow": ["inspect", "init-mapping", "finalize", "validate", "ingest", "query"],
        "fixture": fixture,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "ingest": ingest_report.to_dict(),
        "query": query_summary,
    }
    _write_json(paths["summary"], summary)
    return summary


def _write_messy_customer_fixture(raw_root: Path) -> dict[str, Any]:
    raw_root.mkdir(parents=True, exist_ok=True)
    pose = raw_root / "customer_export_42.csv"
    imu = raw_root / "imu_vendor_dump.csv"
    images = raw_root / "front_cam_frames"
    images.mkdir(parents=True, exist_ok=True)
    start_ts_ns = 1_000_000_000
    step_ns = 100_000_000
    rows = 24

    pose_lines = [
        "robot_clock_ns,north_m,east_m,up_m,att_w,att_i,att_j,att_k,"
        "speed_n_mps,speed_e_mps,speed_u_mps,battery_v,operator_note"
    ]
    imu_lines = [
        "robot_clock_ns,linacc_right,linacc_forward,linacc_up,"
        "gyro_rollish,gyro_pitchish,gyro_yawish,temp_c"
    ]
    for index in range(rows):
        timestamp = start_ts_ns + index * step_ns
        pose_lines.append(
            f"{timestamp},{index * 0.2:.3f},{(index % 5) * 0.05:.3f},0.100,1.0,0.0,0.0,0.0,"
            "2.0,0.5,0.0,24.1,ok"
        )
        imu_lines.append(
            f"{timestamp},{0.01 * index:.4f},0.0200,9.8100,"
            f"{0.001 * index:.4f},0.0020,0.0030,32.0"
        )
        if index % 4 == 0:
            (images / f"front_{timestamp}_frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    pose.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    imu.write_text("\n".join(imu_lines) + "\n", encoding="utf-8")
    (raw_root / "README_customer.txt").write_text(
        "Customer export with nonstandard telemetry column names.\n",
        encoding="utf-8",
    )
    return {
        "pose_csv": str(pose),
        "imu_csv": str(imu),
        "image_dir": str(images),
        "start_ts_ns": start_ts_ns,
        "end_ts_ns": start_ts_ns + (rows - 1) * step_ns,
        "rows": rows,
    }


def _finalize_demo_mapping(draft: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    final = json.loads(json.dumps(draft))
    final["mapping_status"] = "final"
    final["streams"] = [
        {
            "stream_id": "pose",
            "type": "pose",
            "modality": "pose",
            "source_id": _source_id_for_path(final, fixture["pose_csv"]),
            "timestamp": "robot_clock_ns",
            "channels": {
                "x": "north_m",
                "y": "east_m",
                "z": "up_m",
                "qw": "att_w",
                "qx": "att_i",
                "qy": "att_j",
                "qz": "att_k",
            },
            "units": {"x": "m", "y": "m", "z": "m", "vx": "m/s", "vy": "m/s", "vz": "m/s"},
            "frame_id": "base_link",
            "mapping_status": "final",
        },
        {
            "stream_id": "imu0",
            "type": "imu",
            "modality": "imu",
            "source_id": _source_id_for_path(final, fixture["imu_csv"]),
            "timestamp": "robot_clock_ns",
            "channels": {
                "ax": "linacc_right",
                "ay": "linacc_forward",
                "az": "linacc_up",
                "gx": "gyro_rollish",
                "gy": "gyro_pitchish",
                "gz": "gyro_yawish",
            },
            "units": {
                "ax": "m/s^2",
                "ay": "m/s^2",
                "az": "m/s^2",
                "gx": "rad/s",
                "gy": "rad/s",
                "gz": "rad/s",
            },
            "frame_id": "imu0",
            "mapping_status": "final",
        },
        {
            "stream_id": "front_cam",
            "type": "camera",
            "modality": "camera",
            "source_id": _source_id_for_path(final, fixture["image_dir"]),
            "timestamp": "timestamp_from_filename",
            "channels": {"frame_path": "path"},
            "units": {},
            "frame_id": "front_cam",
            "mapping_status": "final",
        },
    ]
    return final


def _source_id_for_path(final: dict[str, Any], path: str) -> str:
    wanted = str(Path(path))
    for source in final.get("sources", []):
        if str(source.get("path")) == wanted:
            return str(source["source_id"])
    source_id = f"src_{len(final.get('sources', [])):03}"
    final.setdefault("sources", []).append(
        {"source_id": source_id, "path": wanted, "type": _source_type_for_demo(path)}
    )
    return source_id


def _source_type_for_demo(path: str) -> str:
    candidate = Path(path)
    if candidate.is_dir():
        return "directory"
    return candidate.suffix.lower().lstrip(".") or "file"


def stage_s3_prefix(
    input_uri: str,
    output_root: Path,
    *,
    manifest_path: Path,
    limit: int = 500,
    robotics_bin: str | None = None,
) -> dict[str, Any]:
    command = _robotics_command(robotics_bin) + [
        "object-store",
        "sync-prefix",
        "--uri",
        input_uri,
        "--out",
        str(output_root),
        "--limit",
        str(limit),
    ]
    completed = _run_command(command)
    payload = json.loads(completed.stdout)
    objects = payload.get("objects", []) if isinstance(payload, dict) else []
    copied_at = _utc_now()
    staged = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        staged.append(
            {
                "source_uri": item.get("uri"),
                "staged_path": item.get("local_path"),
                "size_bytes": item.get("size_bytes"),
                "last_modified": item.get("last_modified"),
                "copied_at": copied_at,
            }
        )
    return {
        "version": 1,
        "kind": "s3_stage_manifest",
        "input_uri": input_uri,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "object_count": len(staged),
        "objects": staged,
    }


def upload_managed_dataset(
    managed_root: Path,
    uri: str,
    *,
    manifest_path: Path,
    robotics_bin: str | None = None,
) -> dict[str, Any]:
    if not managed_root.is_dir():
        raise SystemExit(f"managed root does not exist or is not a directory: {managed_root}")
    base_uri = uri.rstrip("/")
    uploaded = []
    for path in sorted(item for item in managed_root.rglob("*") if item.is_file()):
        relative = path.relative_to(managed_root).as_posix()
        target_uri = f"{base_uri}/{relative}"
        command = _robotics_command(robotics_bin) + [
            "object-store",
            "put",
            "--input",
            str(path),
            "--uri",
            target_uri,
        ]
        _run_command(command)
        uploaded.append(
            {
                "path": str(path),
                "uri": target_uri,
                "size_bytes": path.stat().st_size,
                "uploaded_at": _utc_now(),
            }
        )
    return {
        "version": 1,
        "kind": "managed_upload_manifest",
        "managed_root": str(managed_root),
        "uri": base_uri,
        "manifest_path": str(manifest_path),
        "object_count": len(uploaded),
        "objects": uploaded,
    }


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise SystemExit(message)
    return completed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_summary(result: Any) -> dict[str, Any]:
    shape = [int(value) for value in getattr(result.tensor, "shape", ())]
    timestamps = result.timestamps_ns
    timestamp_count = int(len(timestamps))
    timestamp_range = [int(timestamps[0]), int(timestamps[-1])] if timestamp_count else None
    return {
        "output": result.output,
        "tensor_shape": shape,
        "timestamp_count": timestamp_count,
        "timestamp_range_ns": timestamp_range,
        "row_groups": [int(row_group) for row_group in result.row_groups],
        "file_uri": result.file_uri,
        "selected_bytes": int(result.selected_bytes),
        "diagnostics": asdict(result.diagnostics),
        "tensor_certificate": asdict(result.certificate) if result.certificate is not None else None,
        "manifest": result.manifest,
        "media_manifest": result.media_manifest,
    }


if __name__ == "__main__":
    raise SystemExit(main())
