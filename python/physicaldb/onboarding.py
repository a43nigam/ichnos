from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import subprocess
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .query import _parse_cli_metrics, _robotics_command


SUPPORTED_MODALITIES = {"pose", "imu", "camera", "media"}
SUPPORTED_STREAM_TYPES = {"pose", "imu", "camera", "generic_media"}
KNOWN_CHANNELS = {
    "pose": {"x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"},
    "imu": {"ax", "ay", "az", "gx", "gy", "gz"},
    "camera": {"frame_path", "camera_bytes"},
    "media": {"uri", "bytes"},
}
TIMESTAMP_CANDIDATES = (
    "timestamp_ns",
    "#timestamp",
    "#timestamp [ns]",
    "timestamp",
    "time_ns",
    "time",
    "t",
    "stamp",
    "log_time_ns",
    "publish_time_ns",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TIMESTAMP_UNIT_TO_NS = {
    "ns": 1.0,
    "nanosecond": 1.0,
    "nanoseconds": 1.0,
    "us": 1_000.0,
    "microsecond": 1_000.0,
    "microseconds": 1_000.0,
    "ms": 1_000_000.0,
    "millisecond": 1_000_000.0,
    "milliseconds": 1_000_000.0,
    "s": 1_000_000_000.0,
    "sec": 1_000_000_000.0,
    "second": 1_000_000_000.0,
    "seconds": 1_000_000_000.0,
}


@dataclass(frozen=True)
class DatasetFile:
    path: str
    kind: str
    size_bytes: int = 0
    row_count: int | None = None
    columns: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    schema_names: tuple[str, ...] = ()
    discovery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStream:
    stream_id: str
    modality: str
    source_path: str
    timestamp_candidates: tuple[str, ...] = ()
    channels: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    frame_id: str = ""
    row_count: int | None = None
    calibration: dict[str, Any] | None = None
    confidence: float | None = None
    warnings: tuple[str, ...] = ()
    discovery: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetProfile:
    version: int
    input_uri: str
    dataset_format: str
    files: tuple[DatasetFile, ...]
    streams: tuple[DatasetStream, ...]
    warnings: tuple[str, ...] = ()
    adapter_id: str = ""
    discovery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetProfile":
        return cls(
            version=int(data.get("version", 1)),
            input_uri=str(data.get("input_uri", "")),
            dataset_format=str(data.get("dataset_format", "unknown")),
            files=tuple(DatasetFile(**item) for item in data.get("files", [])),
            streams=tuple(
                DatasetStream(
                    stream_id=str(item.get("stream_id", "")),
                    modality=str(item.get("modality", "")),
                    source_path=str(item.get("source_path", "")),
                    timestamp_candidates=tuple(item.get("timestamp_candidates", ())),
                    channels=dict(item.get("channels", {})),
                    units=dict(item.get("units", {})),
                    frame_id=str(item.get("frame_id", "")),
                    row_count=item.get("row_count"),
                    calibration=dict(item["calibration"])
                    if isinstance(item.get("calibration"), Mapping)
                    else None,
                    confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                    warnings=tuple(str(warning) for warning in item.get("warnings", [])),
                    discovery=dict(item.get("discovery", {})),
                )
                for item in data.get("streams", [])
            ),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            adapter_id=str(data.get("adapter_id", "")),
            discovery=dict(data.get("discovery", {})),
        )


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    stream_count: int = 0
    modalities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestReport:
    dataset_id: str
    output_root: str
    outputs: dict[str, str]
    row_groups: dict[str, int]
    bytes: dict[str, int]
    warnings: tuple[str, ...] = ()
    unsupported_streams: tuple[str, ...] = ()
    adapter_id: str = ""
    calibrations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_dataset(
    path_or_uri: str | os.PathLike[str],
    *,
    adapter_id: str = "auto",
) -> DatasetProfile:
    from .adapters import adapter_registry

    registry = adapter_registry()
    adapter = registry.select_for_path(path_or_uri) if adapter_id == "auto" else registry.get(adapter_id)
    return adapter.inspect(path_or_uri)


def _inspect_dataset_impl(path_or_uri: str | os.PathLike[str]) -> DatasetProfile:
    path = Path(path_or_uri)
    warnings: list[str] = []
    if not path.exists():
        return DatasetProfile(
            version=1,
            input_uri=str(path_or_uri),
            dataset_format="uri",
            files=(),
            streams=(),
            warnings=(f"{path_or_uri} is not a local path; only URI metadata was recorded",),
        )

    if path.is_dir():
        return _inspect_directory(path)
    if path.suffix.lower() == ".zip":
        return _inspect_zip(path)
    if path.suffix.lower() == ".parquet":
        file, streams = _inspect_parquet(path, warnings)
        return DatasetProfile(1, str(path), "parquet", (file,), streams, tuple(warnings))
    if path.suffix.lower() == ".mcap":
        file, streams, mcap_warnings = _inspect_mcap(path)
        return DatasetProfile(1, str(path), "mcap", (file,), streams, mcap_warnings)
    return DatasetProfile(
        version=1,
        input_uri=str(path),
        dataset_format="unknown",
        files=(DatasetFile(str(path), path.suffix.lower().lstrip(".") or "file", path.stat().st_size),),
        streams=(),
        warnings=(f"unsupported dataset input type: {path.suffix or 'file'}",),
    )


def suggest_manifest(
    profile: DatasetProfile | Mapping[str, Any],
    *,
    dataset_id: str,
    robot_id: str,
    session_id: str,
    adapter_id: str = "auto",
    adapter_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .adapters import adapter_registry

    if not isinstance(profile, DatasetProfile):
        profile = DatasetProfile.from_dict(profile)
    registry = adapter_registry()
    adapter = registry.select_for_profile(profile) if adapter_id == "auto" else registry.get(adapter_id)
    return adapter.suggest_manifest(
        profile,
        dataset_id=dataset_id,
        robot_id=robot_id,
        session_id=session_id,
        adapter_options=adapter_options,
    )


def _suggest_manifest_impl(
    profile: DatasetProfile | Mapping[str, Any],
    *,
    dataset_id: str,
    robot_id: str,
    session_id: str,
) -> dict[str, Any]:
    if not isinstance(profile, DatasetProfile):
        profile = DatasetProfile.from_dict(profile)
    sources: list[dict[str, Any]] = []
    streams: list[dict[str, Any]] = []
    seen_sources: dict[str, str] = {}
    for stream in profile.streams:
        source_id = seen_sources.get(stream.source_path)
        if source_id is None:
            source_id = f"src_{len(seen_sources):03}"
            seen_sources[stream.source_path] = source_id
            sources.append({"source_id": source_id, "path": stream.source_path, "type": _source_type(stream.source_path)})
        manifest_stream = {
            "stream_id": stream.stream_id,
            "type": _stream_type(stream.modality),
            "modality": stream.modality,
            "source_id": source_id,
            "timestamp": _preferred_timestamp(stream.timestamp_candidates),
            "channels": stream.channels,
            "units": stream.units or _default_units(stream.modality),
            "frame_id": stream.frame_id or ("base_link" if stream.modality in {"pose", "imu"} else stream.stream_id),
        }
        if stream.confidence is not None:
            manifest_stream["confidence"] = stream.confidence
        if stream.warnings:
            manifest_stream["warnings"] = list(stream.warnings)
        if stream.discovery:
            manifest_stream["discovery"] = dict(stream.discovery)
            for timestamp_field in ("timestamp_unit", "timestamp_scale"):
                if timestamp_field in stream.discovery:
                    manifest_stream[timestamp_field] = stream.discovery[timestamp_field]
        if stream.calibration:
            manifest_stream["calibration"] = dict(stream.calibration)
        streams.append(manifest_stream)
    manifest: dict[str, Any] = {
        "version": 1,
        "dataset_id": dataset_id,
        "robot_id": robot_id,
        "session_id": session_id,
        "sources": sources,
        "streams": streams,
    }
    if profile.discovery:
        manifest["discovery"] = dict(profile.discovery)
    confidences = [stream.confidence for stream in profile.streams if stream.confidence is not None]
    if confidences:
        manifest["confidence"] = round(sum(confidences) / len(confidences), 3)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    adapter_id: str = "auto",
) -> ValidationReport:
    from .adapters import adapter_registry

    data = _load_manifest(manifest)
    registry = adapter_registry()
    adapter = registry.select_for_manifest(data) if adapter_id == "auto" else registry.get(adapter_id)
    return adapter.validate_manifest(data)


def _validate_manifest_impl(manifest: Mapping[str, Any] | str | os.PathLike[str]) -> ValidationReport:
    data = _load_manifest(manifest)
    errors: list[str] = []
    warnings: list[str] = []
    required_top = {"version", "dataset_id", "robot_id", "session_id", "sources", "streams"}
    for field_name in sorted(required_top):
        if field_name not in data:
            errors.append(f"missing top-level field: {field_name}")
    if data.get("version") != 1:
        errors.append("version must be 1")

    sources_raw = data.get("sources", [])
    streams_raw = data.get("streams", [])
    if not isinstance(sources_raw, list):
        errors.append("sources must be a list")
        sources_raw = []
    if not isinstance(streams_raw, list):
        errors.append("streams must be a list")
        streams_raw = []

    source_ids: set[str] = set()
    for index, source in enumerate(sources_raw):
        if not isinstance(source, Mapping):
            errors.append(f"sources[{index}] must be an object")
            continue
        for field_name in ("source_id", "path", "type"):
            if field_name not in source:
                errors.append(f"sources[{index}] missing {field_name}")
        source_id = str(source.get("source_id", ""))
        if source_id in source_ids:
            errors.append(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)

    stream_ids: set[str] = set()
    channel_refs: dict[str, str] = {}
    modalities: list[str] = []
    allowed_required = {"timestamp", "channels", "units", "frame_id", "calibration"}
    for index, stream in enumerate(streams_raw):
        if not isinstance(stream, Mapping):
            errors.append(f"streams[{index}] must be an object")
            continue
        prefix = f"streams[{index}]"
        stream_mapping_status = str(stream.get("mapping_status") or data.get("mapping_status") or "final")
        draft_mapping = stream_mapping_status == "draft"
        for field_name in ("stream_id", "type", "modality", "source_id", "timestamp", "channels", "units", "frame_id"):
            if field_name not in stream:
                if draft_mapping and field_name in {"timestamp", "channels", "units", "frame_id"}:
                    warnings.append(f"{prefix} draft mapping missing {field_name}")
                else:
                    errors.append(f"{prefix} missing {field_name}")
        stream_id = str(stream.get("stream_id", ""))
        if stream_id in stream_ids:
            errors.append(f"duplicate stream_id: {stream_id}")
        stream_ids.add(stream_id)
        modality = str(stream.get("modality", ""))
        modalities.append(modality)
        if modality not in SUPPORTED_MODALITIES:
            errors.append(f"{prefix} unsupported modality: {modality}")
        stream_type = str(stream.get("type", ""))
        if stream_type not in SUPPORTED_STREAM_TYPES:
            errors.append(f"{prefix} unsupported stream type: {stream_type}")
        if stream.get("source_id") not in source_ids:
            errors.append(f"{prefix} references unknown source_id: {stream.get('source_id')}")
        if not stream.get("timestamp"):
            if draft_mapping:
                warnings.append(f"{prefix} draft mapping has no timestamp field")
            else:
                errors.append(f"{prefix} missing timestamp field")
        required = stream.get("required", [])
        if required:
            if not isinstance(required, list):
                errors.append(f"{prefix} required must be a list")
            else:
                for required_field in required:
                    if str(required_field) not in allowed_required:
                        errors.append(f"{prefix} unknown required field: {required_field}")
        channels = stream.get("channels", {})
        if not isinstance(channels, Mapping) or not channels:
            if draft_mapping:
                warnings.append(f"{prefix} draft mapping has no channel mappings")
            else:
                errors.append(f"{prefix} channels must be a non-empty object")
        else:
            known = KNOWN_CHANNELS.get(modality, set())
            for logical_name, physical_name in channels.items():
                logical = str(logical_name)
                if known and logical not in known:
                    errors.append(f"{prefix} unknown channel reference: {logical}")
                if logical in channel_refs:
                    errors.append(
                        f"ambiguous channel mapping: {logical} appears in {channel_refs[logical]} and {stream_id}"
                    )
                channel_refs[logical] = stream_id
                if not str(physical_name):
                    errors.append(f"{prefix} channel {logical} maps to an empty field")
        if "calibration" in stream and stream["calibration"] is not None and not isinstance(stream["calibration"], Mapping):
            errors.append(f"{prefix} calibration must be an object when present")

    if not streams_raw:
        warnings.append("manifest has no streams")
    return ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        stream_count=len(streams_raw),
        modalities=tuple(sorted(set(modalities))),
    )


def ingest_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str],
    row_group_rows: int = 500,
    robotics_bin: str | os.PathLike[str] | None = None,
    adapter_id: str = "auto",
) -> IngestReport:
    from .adapters import adapter_registry

    data = _load_manifest(manifest)
    registry = adapter_registry()
    adapter = registry.select_for_manifest(data) if adapter_id == "auto" else registry.get(adapter_id)
    return adapter.ingest(
        data,
        output_root=output_root,
        row_group_rows=row_group_rows,
        robotics_bin=robotics_bin,
    )


def _ingest_manifest_impl(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str],
    row_group_rows: int = 500,
    robotics_bin: str | os.PathLike[str] | None = None,
    adapter_id: str = "",
) -> IngestReport:
    data = _load_manifest(manifest)
    report = _validate_manifest_impl(data)
    if not report.valid:
        raise ValueError("invalid dataset manifest: " + "; ".join(report.errors))
    root = Path(output_root)
    parquet_root = root / "parquet"
    catalog_root = root / "catalog"
    parquet_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)
    sources = {str(source["source_id"]): source for source in data["sources"]}
    outputs: dict[str, str] = {}
    row_groups: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    warnings: list[str] = []
    unsupported: list[str] = []
    pose_catalog: str | None = None
    imu_catalog: str | None = None
    media_catalogs: list[str] = []
    calibrations: dict[str, dict[str, Any]] = {}

    for stream in data["streams"]:
        stream_id = str(stream["stream_id"])
        modality = str(stream["modality"])
        source = sources[str(stream["source_id"])]
        source_path = str(source["path"])
        source_type = str(source.get("type", _source_type(source_path)))
        out_parquet = parquet_root / f"{stream_id}.parquet"
        command = None
        if isinstance(stream.get("calibration"), Mapping):
            calibrations[stream_id] = dict(stream["calibration"])
        if source_type == "euroc":
            command = _euroc_ingest_command(
                modality=modality,
                source_path=source_path,
                out_parquet=out_parquet,
                stream_id=stream_id,
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
                row_group_rows=row_group_rows,
                robotics_bin=robotics_bin,
            )
        elif source_type == "mcap":
            command = _mcap_ingest_command(
                source_path=source_path,
                out_parquet=out_parquet,
                topic=str(data.get("adapter_options", {}).get("topic", "/pose"))
                if isinstance(data.get("adapter_options", {}), Mapping)
                else "/pose",
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
                row_group_rows=row_group_rows,
                robotics_bin=robotics_bin,
            )
        elif source_type == "kitti_oxts":
            command = _single_pose_ingest_command(
                subcommand="kitti-oxts",
                source_path=source_path,
                out_parquet=out_parquet,
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
                row_group_rows=row_group_rows,
                robotics_bin=robotics_bin,
            )
        elif source_type == "nuscenes_ego":
            command = _single_pose_ingest_command(
                subcommand="nuscenes-ego",
                source_path=source_path,
                out_parquet=out_parquet,
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
                row_group_rows=row_group_rows,
                robotics_bin=robotics_bin,
            )
        if command is not None:
            metrics = _run_robotics(command)
            row_groups[stream_id] = int(metrics.get("row_groups", 0))
        elif source_type == "parquet":
            source_file = Path(source_path)
            if source_file.resolve() != out_parquet.resolve():
                shutil.copyfile(source_file, out_parquet)
            row_groups[stream_id] = 0
        else:
            unsupported.append(stream_id)
            warnings.append(f"stream {stream_id} source type {source_type} is not ingestable in v1")
            continue
        outputs[f"{stream_id}_parquet"] = str(out_parquet)
        byte_counts[f"{stream_id}_parquet"] = out_parquet.stat().st_size if out_parquet.exists() else 0

        catalog_path = catalog_root / f"{stream_id}_catalog.parquet"
        catalog_command = _catalog_command(
            modality=modality,
            parquet_path=out_parquet,
            catalog_path=catalog_path,
            stream_id=stream_id,
            robotics_bin=robotics_bin,
        )
        if catalog_command is None:
            warnings.append(f"stream {stream_id} catalog build skipped for modality {modality}")
            continue
        catalog_metrics = _run_robotics(catalog_command)
        outputs[f"{stream_id}_catalog"] = str(catalog_path)
        byte_counts[f"{stream_id}_catalog"] = catalog_path.stat().st_size if catalog_path.exists() else 0
        row_groups[f"{stream_id}_catalog"] = int(catalog_metrics.get("indexed_row_groups", 0))
        if modality == "pose":
            pose_catalog = str(catalog_path)
        elif modality == "imu":
            imu_catalog = str(catalog_path)
        elif modality == "camera":
            media_catalogs.append(str(catalog_path))

    if pose_catalog is not None:
        catalog_db = catalog_root / "dataset.duckdb"
        cmd = _robotics_command(robotics_bin) + [
            "catalog",
            "duckdb-build",
            "--pose-catalog",
            pose_catalog,
            "--out",
            str(catalog_db),
            "--spatial-index",
            "hilbert",
        ]
        if imu_catalog is not None:
            cmd.extend(["--imu-catalog", imu_catalog])
        if media_catalogs:
            cmd.extend(["--media-catalog", media_catalogs[0]])
            if len(media_catalogs) > 1:
                warnings.append("only the first camera media catalog is included in the v1 DuckDB catalog")
        _run_robotics(cmd)
        outputs["catalog_db"] = str(catalog_db)
        byte_counts["catalog_db"] = catalog_db.stat().st_size if catalog_db.exists() else 0

    return IngestReport(
        dataset_id=str(data["dataset_id"]),
        output_root=str(root),
        outputs=outputs,
        row_groups=row_groups,
        bytes=byte_counts,
        warnings=tuple(warnings),
        unsupported_streams=tuple(unsupported),
        adapter_id=adapter_id or str(data.get("adapter_id", "")),
        calibrations=calibrations,
    )


def _ingest_generic_dataset_impl(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str],
    row_group_rows: int = 500,
    robotics_bin: str | os.PathLike[str] | None = None,
    adapter_id: str = "generic_dataset",
) -> IngestReport:
    data = _load_manifest(manifest)
    report = _validate_manifest_impl(data)
    if not report.valid:
        raise ValueError("invalid dataset manifest: " + "; ".join(report.errors))
    root = Path(output_root)
    _preflight_generic_dataset(data, root)
    root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = root.with_name(f".{root.name}.tmp-{uuid.uuid4().hex}")
    try:
        report = _ingest_generic_dataset_into_root(
            data,
            output_root=temp_root,
            public_root=root,
            row_group_rows=row_group_rows,
            robotics_bin=robotics_bin,
            adapter_id=adapter_id,
        )
        if root.exists():
            root.rmdir()
        temp_root.rename(root)
        return IngestReport(
            dataset_id=report.dataset_id,
            output_root=str(root),
            outputs={key: _replace_path_prefix(value, temp_root, root) for key, value in report.outputs.items()},
            row_groups=report.row_groups,
            bytes=report.bytes,
            warnings=report.warnings,
            unsupported_streams=report.unsupported_streams,
            adapter_id=report.adapter_id,
            calibrations=report.calibrations,
        )
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def _ingest_generic_dataset_into_root(
    data: Mapping[str, Any],
    *,
    output_root: Path,
    public_root: Path,
    row_group_rows: int,
    robotics_bin: str | os.PathLike[str] | None,
    adapter_id: str,
) -> IngestReport:
    root = Path(output_root)
    parquet_root = root / "parquet"
    catalog_root = root / "catalog"
    parquet_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)

    sources = {str(source["source_id"]): source for source in data["sources"]}
    outputs: dict[str, str] = {}
    row_groups: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    warnings: list[str] = []
    unsupported: list[str] = []
    pose_catalog: str | None = None
    imu_catalog: str | None = None
    media_catalogs: list[str] = []
    calibrations: dict[str, dict[str, Any]] = {}

    for stream in data["streams"]:
        stream_id = str(stream["stream_id"])
        modality = str(stream["modality"])
        source = sources[str(stream["source_id"])]
        source_path = str(source["path"])
        source_type = str(source.get("type") or _source_type(source_path))
        out_parquet = parquet_root / f"{stream_id}.parquet"

        if isinstance(stream.get("calibration"), Mapping):
            calibrations[stream_id] = dict(stream["calibration"])

        if source_type == "mcap":
            topic = _generic_mcap_topic(data, stream)
            command = _mcap_ingest_command(
                source_path=source_path,
                out_parquet=out_parquet,
                topic=topic,
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
                row_group_rows=row_group_rows,
                robotics_bin=robotics_bin,
            )
            metrics = _run_robotics(command)
            row_groups[stream_id] = int(metrics.get("row_groups", 0))
        elif _generic_tabular_source_type(source_type, source_path) in {"csv", "parquet"} and modality in {"pose", "imu"}:
            _write_generic_tabular_parquet(
                source_path=Path(source_path),
                source_type=_generic_tabular_source_type(source_type, source_path),
                stream=stream,
                out_parquet=out_parquet,
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
            )
            row_groups[stream_id] = 0
        elif modality == "camera" and _generic_camera_sequence_source(source_type, source_path):
            _write_generic_camera_parquet(
                source_path=Path(source_path),
                stream=stream,
                out_parquet=out_parquet,
                robot_id=str(data["robot_id"]),
                session_id=str(data["session_id"]),
            )
            row_groups[stream_id] = 0
        else:
            unsupported.append(stream_id)
            raise ValueError(
                f"generic_dataset stream {stream_id} source type {source_type} with modality {modality} "
                "is not ingestable in v1"
            )

        outputs[f"{stream_id}_parquet"] = str(out_parquet)
        byte_counts[f"{stream_id}_parquet"] = out_parquet.stat().st_size if out_parquet.exists() else 0

        catalog_path = catalog_root / f"{stream_id}_catalog.parquet"
        public_parquet = public_root / "parquet" / f"{stream_id}.parquet"
        catalog_command = _catalog_command(
            modality=modality,
            parquet_path=out_parquet,
            catalog_path=catalog_path,
            stream_id=stream_id,
            robotics_bin=robotics_bin,
            uri_path=public_parquet,
        )
        if catalog_command is None:
            warnings.append(f"stream {stream_id} catalog build skipped for modality {modality}")
            continue
        catalog_metrics = _run_robotics(catalog_command)
        outputs[f"{stream_id}_catalog"] = str(catalog_path)
        byte_counts[f"{stream_id}_catalog"] = catalog_path.stat().st_size if catalog_path.exists() else 0
        row_groups[f"{stream_id}_catalog"] = int(catalog_metrics.get("indexed_row_groups", 0))
        if modality == "pose":
            pose_catalog = str(catalog_path)
        elif modality == "imu":
            imu_catalog = str(catalog_path)
        elif modality == "camera":
            media_catalogs.append(str(catalog_path))

    if pose_catalog is not None:
        catalog_db = catalog_root / "dataset.duckdb"
        cmd = _robotics_command(robotics_bin) + [
            "catalog",
            "duckdb-build",
            "--pose-catalog",
            pose_catalog,
            "--out",
            str(catalog_db),
            "--spatial-index",
            "hilbert",
        ]
        if imu_catalog is not None:
            cmd.extend(["--imu-catalog", imu_catalog])
        if media_catalogs:
            cmd.extend(["--media-catalog", media_catalogs[0]])
            if len(media_catalogs) > 1:
                warnings.append("only the first camera media catalog is included in the v1 DuckDB catalog")
        _run_robotics(cmd)
        outputs["catalog_db"] = str(catalog_db)
        byte_counts["catalog_db"] = catalog_db.stat().st_size if catalog_db.exists() else 0

    return IngestReport(
        dataset_id=str(data["dataset_id"]),
        output_root=str(root),
        outputs=outputs,
        row_groups=row_groups,
        bytes=byte_counts,
        warnings=tuple(warnings),
        unsupported_streams=tuple(unsupported),
        adapter_id=adapter_id,
        calibrations=calibrations,
    )


def _inspect_directory(path: Path) -> DatasetProfile:
    warnings: list[str] = []
    files: list[DatasetFile] = []
    streams: list[DatasetStream] = []
    euroc_root = _find_euroc_root(path)
    dataset_format = "euroc" if euroc_root is not None else "directory"
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        kind = child.suffix.lower().lstrip(".") or "file"
        files.append(DatasetFile(str(child), kind, child.stat().st_size))
    if euroc_root is not None:
        streams.extend(_euroc_streams(euroc_root, warnings))
    for parquet in sorted(path.rglob("*.parquet")):
        file, parquet_streams = _inspect_parquet(parquet, warnings)
        files.append(file)
        streams.extend(parquet_streams)
    return DatasetProfile(1, str(path), dataset_format, tuple(files), tuple(_dedupe_streams(streams)), tuple(warnings))


def _inspect_generic_dataset(path_or_uri: str | os.PathLike[str]) -> DatasetProfile:
    uri = str(path_or_uri)
    if _is_s3_uri(uri):
        return _inspect_s3_prefix(uri)
    path = Path(path_or_uri)
    warnings: list[str] = []
    if not path.exists():
        return DatasetProfile(
            version=1,
            input_uri=uri,
            dataset_format="uri",
            files=(),
            streams=(),
            warnings=(f"{uri} is not a local path; no local inspection was performed",),
            discovery={"status": "unresolved_uri"},
        )
    if path.is_dir():
        return _inspect_generic_directory(path)
    if path.suffix.lower() == ".parquet":
        file, streams = _inspect_parquet(path, warnings)
        streams = tuple(_with_generic_stream_metadata(stream, file.columns) for stream in streams)
        return DatasetProfile(
            1,
            str(path),
            "parquet",
            (file,),
            streams,
            tuple(warnings),
            discovery=_profile_discovery((file,), streams),
        )
    if path.suffix.lower() == ".mcap":
        file, streams, mcap_warnings = _inspect_mcap_generic(path)
        return DatasetProfile(
            1,
            str(path),
            "mcap",
            (file,),
            streams,
            mcap_warnings,
            discovery=_profile_discovery((file,), streams),
        )
    if path.suffix.lower() == ".csv":
        file, stream = _inspect_csv_table(path, warnings)
        streams = (stream,) if stream is not None else ()
        return DatasetProfile(1, str(path), "csv", (file,), streams, tuple(warnings), discovery=_profile_discovery((file,), streams))
    file = DatasetFile(str(path), path.suffix.lower().lstrip(".") or "file", path.stat().st_size)
    return DatasetProfile(1, str(path), "unknown", (file,), (), (f"no generic inspector for {path.suffix or 'file'}",))


def _inspect_generic_directory(path: Path) -> DatasetProfile:
    warnings: list[str] = []
    files: list[DatasetFile] = []
    streams: list[DatasetStream] = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        suffix = child.suffix.lower()
        if suffix == ".parquet":
            file, parquet_streams = _inspect_parquet(child, warnings)
            files.append(file)
            streams.extend(_with_generic_stream_metadata(stream, file.columns) for stream in parquet_streams)
        elif suffix == ".mcap":
            file, mcap_streams, mcap_warnings = _inspect_mcap_generic(child)
            warnings.extend(mcap_warnings)
            files.append(file)
            streams.extend(mcap_streams)
        elif suffix == ".csv":
            file, stream = _inspect_csv_table(child, warnings)
            files.append(file)
            if stream is not None:
                streams.append(stream)
        else:
            files.append(DatasetFile(str(child), suffix.lstrip(".") or "file", child.stat().st_size))
    image_streams = _inspect_image_sequences(path, warnings)
    streams.extend(image_streams)
    if not streams:
        warnings.append("no candidate streams found")
    return DatasetProfile(
        1,
        str(path),
        "directory",
        tuple(files),
        tuple(_dedupe_streams(streams)),
        tuple(_dedupe_strings(warnings)),
        discovery=_profile_discovery(tuple(files), tuple(streams)),
    )


def _inspect_s3_prefix(uri: str) -> DatasetProfile:
    warnings: list[str] = []
    files: list[DatasetFile] = []
    discovery: dict[str, Any] = {"prefix": uri, "source": "object_store"}
    try:
        command = _robotics_command(None) + ["object-store", "list", "--uri", uri, "--limit", "500"]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        listed = json.loads(completed.stdout)
    except Exception as exc:
        warnings.append(f"failed to list S3-compatible prefix with robotics object-store helper: {exc}")
        listed = {"objects": []}
    objects = listed.get("objects", []) if isinstance(listed, Mapping) else []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        key_uri = str(obj.get("uri") or obj.get("path") or "")
        files.append(
            DatasetFile(
                key_uri,
                Path(key_uri).suffix.lower().lstrip(".") or "object",
                int(obj.get("size_bytes") or 0),
                discovery={"last_modified": obj.get("last_modified")},
            )
        )
    discovery["object_count"] = len(files)
    discovery["size_bytes"] = sum(file.size_bytes for file in files)
    streams = _streams_from_object_listing(uri, files, warnings)
    if not files:
        warnings.append("S3-compatible prefix listing returned no objects")
    return DatasetProfile(
        1,
        uri,
        "s3_prefix",
        tuple(files),
        streams,
        tuple(_dedupe_strings(warnings)),
        discovery=discovery | _profile_discovery(tuple(files), streams),
    )


def _inspect_csv_table(path: Path, warnings: list[str]) -> tuple[DatasetFile, DatasetStream | None]:
    columns: tuple[str, ...] = ()
    row_count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
            reader = csv.reader(handle, dialect)
            header = next(reader, [])
            columns = tuple(column.strip() for column in header if column.strip())
            for row_count, _row in enumerate(reader, start=1):
                pass
    except Exception as exc:
        warnings.append(f"failed to inspect CSV {path}: {exc}")
        return DatasetFile(str(path), "csv", path.stat().st_size), None
    modality = _infer_tabular_modality(columns)
    channels = _channels_for_columns(modality, columns)
    timestamps = _timestamp_candidates_for_columns(columns)
    stream_warnings: list[str] = []
    if not timestamps:
        stream_warnings.append("no timestamp column candidate found")
    if not channels:
        stream_warnings.append("no known channel mapping candidates found")
    confidence = _mapping_confidence(modality, columns, timestamps, channels)
    timestamp_unit = _infer_timestamp_unit(_preferred_timestamp(timestamps))
    discovery: dict[str, Any] = {
        "columns": list(columns),
        "row_count": row_count,
        "sample_rate_hz": None,
        "source_kind": "csv",
    }
    if timestamp_unit:
        discovery["timestamp_unit"] = timestamp_unit
    stream = DatasetStream(
        stream_id=_stream_id_from_path(path, modality),
        modality=modality,
        source_path=str(path),
        timestamp_candidates=timestamps,
        channels=channels,
        units=_default_units(modality),
        frame_id="base_link" if modality in {"pose", "imu"} else path.stem,
        row_count=row_count,
        confidence=confidence,
        warnings=tuple(stream_warnings),
        discovery=discovery,
    )
    return DatasetFile(str(path), "csv", path.stat().st_size, row_count, columns), stream


def _inspect_image_sequences(path: Path, warnings: list[str]) -> tuple[DatasetStream, ...]:
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        return ()
    by_parent: dict[Path, list[Path]] = {}
    for image in images:
        by_parent.setdefault(image.parent, []).append(image)
    streams: list[DatasetStream] = []
    for index, (parent, parent_images) in enumerate(sorted(by_parent.items())):
        timestamps = _image_timestamp_candidates(parent_images)
        calibration_paths = [
            candidate
            for candidate in sorted(parent.iterdir())
            if candidate.is_file() and candidate.name.lower() in {"sensor.yaml", "calib.txt", "calibration.json", "camera.yaml"}
        ]
        calibration_files = [str(candidate) for candidate in calibration_paths]
        stream_warnings = []
        if not timestamps:
            stream_warnings.append("image filenames do not contain a clear numeric timestamp pattern")
        if not calibration_files:
            stream_warnings.append("no nearby camera calibration file found")
        stream_id = parent.name if parent.name else f"camera_{index}"
        calibration = _generic_camera_calibration_from_sidecars(calibration_paths, stream_id, stream_warnings)
        discovery: dict[str, Any] = {
            "source_kind": "image_sequence",
            "image_count": len(parent_images),
            "extensions": sorted({image.suffix.lower() for image in parent_images}),
            "calibration_files": calibration_files,
        }
        streams.append(
            DatasetStream(
                stream_id=stream_id,
                modality="camera",
                source_path=str(parent),
                timestamp_candidates=timestamps,
                channels={"frame_path": "path"},
                units={},
                frame_id=stream_id,
                row_count=len(parent_images),
                calibration=calibration,
                confidence=0.75 if timestamps else 0.45,
                warnings=tuple(stream_warnings),
                discovery=discovery,
            )
        )
    return tuple(streams)


def _inspect_mcap_generic(path: Path) -> tuple[DatasetFile, tuple[DatasetStream, ...], tuple[str, ...]]:
    file, legacy_streams, warnings = _inspect_mcap(path)
    streams: list[DatasetStream] = []
    topic_infos = _mcap_topic_infos(path)
    if not topic_infos:
        return file, tuple(_with_mcap_generic_metadata(stream, file.topics, file.schema_names) for stream in legacy_streams), warnings
    for topic, schema_name in topic_infos:
        modality = _infer_mcap_modality(topic, schema_name)
        channels = _pose_channels() if modality == "pose" else _imu_channels() if modality == "imu" else _camera_channels() if modality == "camera" else {}
        stream_warnings: list[str] = []
        if modality == "media":
            stream_warnings.append("topic semantics are ambiguous; review before ingest")
        streams.append(
            DatasetStream(
                stream_id=topic.strip("/").replace("/", "_") or "mcap",
                modality=modality,
                source_path=str(path),
                timestamp_candidates=("header.stamp", "log_time_ns", "publish_time_ns"),
                channels=channels,
                units=_default_units(modality),
                frame_id="world" if modality == "pose" else (topic.strip("/").split("/")[-1] or "base_link"),
                confidence=0.85 if modality in {"pose", "imu", "camera"} else 0.35,
                warnings=tuple(stream_warnings),
                discovery={"topic": topic, "schema_name": schema_name, "source_kind": "mcap_topic"},
            )
        )
    return (
        DatasetFile(str(path), "mcap", path.stat().st_size, topics=tuple(topic for topic, _ in topic_infos), schema_names=tuple(schema for _, schema in topic_infos)),
        tuple(streams),
        warnings,
    )


def _mcap_topic_infos(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        from mcap.reader import make_reader  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return ()
    try:
        with path.open("rb") as handle:
            summary = make_reader(handle).get_summary()
        channels = getattr(summary, "channels", {}) or {}
        schemas = getattr(summary, "schemas", {}) or {}
        topic_infos: list[tuple[str, str]] = []
        for channel in channels.values():
            schema = schemas.get(getattr(channel, "schema_id", None))
            topic_infos.append((str(channel.topic), str(getattr(schema, "name", "") or "")))
        return tuple(sorted(set(topic_infos)))
    except Exception:
        return ()


def _with_generic_stream_metadata(stream: DatasetStream, columns: Sequence[str]) -> DatasetStream:
    timestamps = stream.timestamp_candidates or _timestamp_candidates_for_columns(columns)
    channels = stream.channels or _channels_for_columns(stream.modality, columns)
    stream_warnings: list[str] = list(stream.warnings)
    if not timestamps:
        stream_warnings.append("no timestamp column candidate found")
    if not channels:
        stream_warnings.append("no known channel mapping candidates found")
    discovery = {"columns": list(columns), "source_kind": "parquet", **stream.discovery}
    timestamp_unit = _infer_timestamp_unit(_preferred_timestamp(timestamps))
    if timestamp_unit and "timestamp_unit" not in discovery:
        discovery["timestamp_unit"] = timestamp_unit
    return DatasetStream(
        stream.stream_id,
        stream.modality,
        stream.source_path,
        timestamps,
        channels,
        stream.units,
        stream.frame_id,
        stream.row_count,
        stream.calibration,
        _mapping_confidence(stream.modality, columns, timestamps, channels),
        tuple(_dedupe_strings(stream_warnings)),
        discovery,
    )


def _with_mcap_generic_metadata(
    stream: DatasetStream, topics: Sequence[str], schema_names: Sequence[str]
) -> DatasetStream:
    warnings = list(stream.warnings)
    if not topics:
        warnings.append("MCAP topics were not available; pose mapping is a low-confidence placeholder")
    return DatasetStream(
        stream.stream_id,
        stream.modality,
        stream.source_path,
        stream.timestamp_candidates,
        stream.channels,
        stream.units,
        stream.frame_id,
        stream.row_count,
        stream.calibration,
        0.55 if topics else 0.25,
        tuple(_dedupe_strings(warnings)),
        {"topics": list(topics), "schema_names": list(schema_names), "source_kind": "mcap"},
    )


def _streams_from_object_listing(
    prefix_uri: str, files: Sequence[DatasetFile], warnings: list[str]
) -> tuple[DatasetStream, ...]:
    parquet_files = [file for file in files if file.path.lower().endswith(".parquet")]
    mcap_files = [file for file in files if file.path.lower().endswith(".mcap")]
    image_files = [file for file in files if Path(file.path).suffix.lower() in IMAGE_SUFFIXES]
    streams: list[DatasetStream] = []
    for file in parquet_files[:20]:
        modality = _infer_name_modality(file.path)
        streams.append(
            DatasetStream(
                _stream_id_from_name(file.path, modality),
                modality,
                file.path,
                ("timestamp_ns", "timestamp"),
                _default_channels_for_modality(modality),
                _default_units(modality),
                "base_link" if modality in {"pose", "imu"} else Path(file.path).stem,
                confidence=0.45,
                warnings=("S3 Parquet schema was not read during prefix inspection; review columns before ingest",),
                discovery={"source_kind": "s3_object", "prefix": prefix_uri},
            )
        )
    for file in mcap_files[:20]:
        modality = _infer_name_modality(file.path)
        streams.append(
            DatasetStream(
                _stream_id_from_name(file.path, modality),
                modality,
                file.path,
                ("header.stamp", "log_time_ns", "publish_time_ns"),
                _default_channels_for_modality(modality),
                _default_units(modality),
                "world" if modality == "pose" else Path(file.path).stem,
                confidence=0.4,
                warnings=("S3 MCAP topics were not read during prefix inspection; review topic mappings before ingest",),
                discovery={"source_kind": "s3_object", "prefix": prefix_uri},
            )
        )
    if image_files:
        streams.append(
            DatasetStream(
                "camera_images",
                "camera",
                prefix_uri,
                ("timestamp_from_filename",),
                {"frame_path": "uri"},
                {},
                "camera_images",
                row_count=len(image_files),
                confidence=0.4,
                warnings=("S3 image sequence timestamps and calibration require review",),
                discovery={"source_kind": "s3_image_sequence", "image_count": len(image_files)},
            )
        )
    if files and not streams:
        warnings.append("listed objects did not include recognizable Parquet, MCAP, or image candidates")
    return tuple(streams)


def _inspect_zip(path: Path) -> DatasetProfile:
    warnings: list[str] = []
    files: list[DatasetFile] = []
    streams: list[DatasetStream] = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for name in names:
            info = archive.getinfo(name)
            files.append(DatasetFile(name, Path(name).suffix.lower().lstrip(".") or "file", info.file_size))
        if any(name.endswith("mav0/state_groundtruth_estimate0/data.csv") for name in names):
            warnings.append("EuRoC-style zip detected; extract before ingesting the v1 manifest")
            prefix = _euroc_zip_prefix(names)
            streams.append(
                DatasetStream("pose", "pose", str(path), ("#timestamp",), _pose_channels(), _default_units("pose"), "world")
            )
            for stream_id in _euroc_zip_sensor_stream_ids(names, prefix):
                sensor_yaml = _read_euroc_zip_sensor_yaml(archive, prefix, stream_id)
                calibration = _euroc_calibration_from_yaml(sensor_yaml, stream_id, warnings)
                if stream_id.startswith("imu"):
                    streams.append(
                        DatasetStream(
                            stream_id,
                            "imu",
                            str(path),
                            ("#timestamp",),
                            _imu_channels(),
                            _default_units("imu"),
                            stream_id,
                            calibration=calibration,
                        )
                    )
                elif stream_id.startswith("cam"):
                    streams.append(
                        DatasetStream(
                            stream_id,
                            "camera",
                            str(path),
                            ("#timestamp [ns]",),
                            _camera_channels(),
                            {},
                            stream_id,
                            calibration=calibration,
                        )
                    )
    return DatasetProfile(1, str(path), "zip", tuple(files), tuple(streams), tuple(warnings))


def _zip_looks_like_euroc(path: Path) -> bool:
    if not path.exists() or path.suffix.lower() != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("mav0/state_groundtruth_estimate0/data.csv") for name in archive.namelist())
    except zipfile.BadZipFile:
        return False


def _inspect_parquet(path: Path, warnings: list[str]) -> tuple[DatasetFile, tuple[DatasetStream, ...]]:
    columns: tuple[str, ...] = ()
    row_count: int | None = None
    try:
        import duckdb
    except ModuleNotFoundError:
        warnings.append("duckdb is not installed; parquet schema and row counts were not inspected")
        return DatasetFile(str(path), "parquet", path.stat().st_size), ()
    try:
        with duckdb.connect(":memory:") as con:
            result = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
            columns = tuple(column[0] for column in result.description)
            row_count = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0])
    except Exception as exc:  # pragma: no cover - duckdb error text is version-specific
        warnings.append(f"failed to inspect parquet {path}: {exc}")
    modality = _infer_parquet_modality(columns)
    stream_id = path.stem if modality != "pose" else "pose"
    channels = _channels_for_columns(modality, columns)
    timestamps = tuple(column for column in TIMESTAMP_CANDIDATES if column in columns)
    stream = DatasetStream(
        stream_id=stream_id,
        modality=modality,
        source_path=str(path),
        timestamp_candidates=timestamps,
        channels=channels,
        units=_default_units(modality),
        frame_id="base_link" if modality in {"pose", "imu"} else stream_id,
        row_count=row_count,
    )
    return DatasetFile(str(path), "parquet", path.stat().st_size, row_count, columns), (stream,)


def _inspect_mcap(path: Path) -> tuple[DatasetFile, tuple[DatasetStream, ...], tuple[str, ...]]:
    warnings: list[str] = []
    topics: tuple[str, ...] = ()
    schema_names: tuple[str, ...] = ()
    try:
        from mcap.reader import make_reader  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        warnings.append("python mcap package is not installed; MCAP topics and schemas were not inspected")
    else:
        try:
            with path.open("rb") as handle:
                summary = make_reader(handle).get_summary()
            channels = getattr(summary, "channels", {}) or {}
            schemas = getattr(summary, "schemas", {}) or {}
            topics = tuple(sorted({str(channel.topic) for channel in channels.values()}))
            schema_names = tuple(sorted({str(schema.name) for schema in schemas.values()}))
        except Exception as exc:  # pragma: no cover - depends on optional mcap reader behavior
            warnings.append(f"failed to inspect MCAP summary: {exc}")
    stream_id = topics[0].strip("/").replace("/", "_") if topics else "mcap"
    return (
        DatasetFile(str(path), "mcap", path.stat().st_size, topics=topics, schema_names=schema_names),
        (
            DatasetStream(
                stream_id=stream_id or "mcap",
                modality="pose",
                source_path=str(path),
                timestamp_candidates=("log_time_ns", "publish_time_ns"),
                channels=_pose_channels(),
                units=_default_units("pose"),
                frame_id="world",
            ),
        ),
        tuple(warnings),
    )


def _inspect_kitti_oxts(path: Path) -> DatasetProfile:
    files = _dataset_files(path)
    source = _kitti_source_root(path)
    warnings = () if source is not None else ("KITTI OXTS path shape was not recognized exactly",)
    return DatasetProfile(
        1,
        str(path),
        "kitti_oxts",
        files,
        (
            DatasetStream(
                "pose",
                "pose",
                str(source or path),
                ("timestamp_ns", "timestamp"),
                _pose_channels(),
                _default_units("pose"),
                "world",
            ),
        ),
        warnings,
    )


def _inspect_nuscenes_ego(path: Path) -> DatasetProfile:
    files = _dataset_files(path)
    return DatasetProfile(
        1,
        str(path),
        "nuscenes_ego",
        files,
        (
            DatasetStream(
                "pose",
                "pose",
                str(path),
                ("timestamp_ns", "timestamp"),
                _pose_channels(),
                _default_units("pose"),
                "world",
            ),
        ),
    )


def _dataset_files(path: Path) -> tuple[DatasetFile, ...]:
    if path.is_file():
        return (DatasetFile(str(path), path.suffix.lower().lstrip(".") or "file", path.stat().st_size),)
    if path.is_dir():
        return tuple(
            DatasetFile(str(item), item.suffix.lower().lstrip(".") or "file", item.stat().st_size)
            for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        )
    return ()


def _find_euroc_root(path: Path) -> Path | None:
    candidates = [path, *(item for item in path.rglob("*") if item.is_dir())]
    for candidate in candidates:
        if (candidate / "mav0" / "state_groundtruth_estimate0" / "data.csv").exists():
            return candidate
    return None


def _euroc_streams(root: Path, warnings: list[str] | None = None) -> tuple[DatasetStream, ...]:
    warnings = warnings if warnings is not None else []
    streams = []
    if (root / "mav0" / "state_groundtruth_estimate0" / "data.csv").exists():
        streams.append(DatasetStream("pose", "pose", str(root), ("#timestamp",), _pose_channels(), _default_units("pose"), "world"))
    if (root / "mav0" / "imu0" / "data.csv").exists():
        streams.append(
            DatasetStream(
                "imu0",
                "imu",
                str(root),
                ("#timestamp",),
                _imu_channels(),
                _default_units("imu"),
                "imu0",
                calibration=_euroc_calibration_for_stream(root, "imu0", warnings),
            )
        )
    for cam_dir in sorted((root / "mav0").glob("cam*")):
        if (cam_dir / "data.csv").exists():
            streams.append(
                DatasetStream(
                    cam_dir.name,
                    "camera",
                    str(root),
                    ("#timestamp [ns]",),
                    _camera_channels(),
                    {},
                    cam_dir.name,
                    calibration=_euroc_calibration_for_stream(root, cam_dir.name, warnings),
                )
            )
    return tuple(streams)


def _euroc_calibration_for_stream(root: Path, stream_id: str, warnings: list[str]) -> dict[str, Any] | None:
    sensor_path = root / "mav0" / stream_id / "sensor.yaml"
    if not sensor_path.exists():
        warnings.append(f"EuRoC calibration missing for {stream_id}: {sensor_path}")
        return None
    try:
        text = sensor_path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"EuRoC calibration could not be read for {stream_id}: {exc}")
        return None
    return _euroc_calibration_from_yaml(text, stream_id, warnings)


def _euroc_zip_prefix(names: Sequence[str]) -> str:
    marker = "mav0/state_groundtruth_estimate0/data.csv"
    for name in names:
        if name.endswith(marker):
            return name[: -len(marker)]
    return ""


def _euroc_zip_sensor_stream_ids(names: Sequence[str], prefix: str) -> tuple[str, ...]:
    stream_ids: set[str] = set()
    marker_prefix = f"{prefix}mav0/"
    for name in names:
        if not name.startswith(marker_prefix) or not name.endswith("/data.csv"):
            continue
        parts = name[len(marker_prefix) :].split("/")
        if len(parts) == 2 and (parts[0].startswith("imu") or parts[0].startswith("cam")):
            stream_ids.add(parts[0])
    return tuple(sorted(stream_ids))


def _read_euroc_zip_sensor_yaml(archive: zipfile.ZipFile, prefix: str, stream_id: str) -> str | None:
    member = f"{prefix}mav0/{stream_id}/sensor.yaml"
    try:
        return archive.read(member).decode("utf-8")
    except KeyError:
        return None
    except UnicodeDecodeError:
        return ""


def _euroc_calibration_from_yaml(
    text: str | None, stream_id: str, warnings: list[str]
) -> dict[str, Any] | None:
    if text is None:
        warnings.append(f"EuRoC calibration missing for {stream_id}: mav0/{stream_id}/sensor.yaml")
        return None

    parsed = _parse_euroc_sensor_yaml(text, stream_id, warnings)
    calibration: dict[str, Any] = {
        "frame_id": stream_id,
        "sensor_frame_id": stream_id,
        "body_frame_id": "body",
    }

    t_bs = parsed.get("T_BS")
    if isinstance(t_bs, Mapping):
        data = t_bs.get("data")
        if isinstance(data, list) and len(data) == 16 and all(isinstance(value, (int, float)) for value in data):
            calibration["T_body_sensor"] = [float(value) for value in data]
        elif data is not None:
            warnings.append(f"EuRoC calibration for {stream_id} has invalid T_BS.data; expected 16 numbers")

    for key in ("sensor_type", "comment", "rostopic", "camera_model"):
        if key in parsed and parsed[key] not in (None, ""):
            calibration[key] = str(parsed[key])
    if "rate_hz" in parsed:
        rate = _coerce_float(parsed["rate_hz"])
        if rate is None:
            warnings.append(f"EuRoC calibration for {stream_id} has invalid rate_hz")
        else:
            calibration["rate_hz"] = rate
    if "resolution" in parsed:
        resolution = _coerce_int_list(parsed["resolution"])
        if resolution is None or len(resolution) != 2:
            warnings.append(f"EuRoC calibration for {stream_id} has invalid resolution; expected [width, height]")
        else:
            calibration["resolution"] = resolution
    if "intrinsics" in parsed:
        intrinsics = _coerce_float_list(parsed["intrinsics"])
        if intrinsics is None:
            warnings.append(f"EuRoC calibration for {stream_id} has invalid intrinsics")
        else:
            calibration["intrinsics"] = intrinsics
    if "distortion_model" in parsed and parsed["distortion_model"] not in (None, ""):
        calibration["distortion_model"] = str(parsed["distortion_model"])
    if "distortion_coefficients" in parsed:
        coefficients = _coerce_float_list(parsed["distortion_coefficients"])
        if coefficients is None:
            warnings.append(f"EuRoC calibration for {stream_id} has invalid distortion_coefficients")
        else:
            calibration["distortion_coefficients"] = coefficients
    return calibration


def _generic_camera_calibration_from_sidecars(
    paths: Sequence[Path], stream_id: str, warnings: list[str]
) -> dict[str, Any] | None:
    for path in paths:
        try:
            parsed = _read_generic_camera_calibration_sidecar(path)
        except Exception as exc:
            warnings.append(f"camera calibration sidecar {path.name} could not be read: {exc}")
            continue
        calibration = _generic_camera_calibration_from_mapping(parsed, stream_id, path, warnings)
        if calibration is not None:
            return calibration
    return None


def _read_generic_camera_calibration_sidecar(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise ValueError("JSON calibration must be an object")
        return dict(parsed)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _parse_euroc_sensor_yaml(text, path.stem, [])
    parsed: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        separator = ":" if ":" in line else "=" if "=" in line else None
        if separator is None:
            continue
        key, value = line.split(separator, 1)
        parsed[key.strip()] = _parse_simple_calibration_value(value.strip())
    if not parsed:
        raise ValueError("no key/value calibration fields found")
    return parsed


def _parse_simple_calibration_value(value: str) -> Any:
    if not value:
        return None
    if value.startswith("["):
        return _parse_yaml_scalar(value)
    parts = [part for part in value.replace(",", " ").split() if part]
    if len(parts) > 1:
        parsed_parts = [_parse_simple_calibration_value(part) for part in parts]
        if all(isinstance(item, (int, float)) for item in parsed_parts):
            return parsed_parts
    return _parse_yaml_scalar(value)


def _generic_camera_calibration_from_mapping(
    parsed: Mapping[str, Any], stream_id: str, path: Path, warnings: list[str]
) -> dict[str, Any] | None:
    calibration: dict[str, Any] = {
        "frame_id": stream_id,
        "sensor_frame_id": stream_id,
        "body_frame_id": "body",
        "calibration_file": str(path),
    }
    for key in ("sensor_type", "comment", "rostopic", "camera_model", "distortion_model"):
        if key in parsed and parsed[key] not in (None, ""):
            calibration[key] = str(parsed[key])
    if "rate_hz" in parsed:
        rate = _coerce_float(parsed["rate_hz"])
        if rate is not None:
            calibration["rate_hz"] = rate
    resolution = _coerce_int_list(parsed.get("resolution"))
    if resolution is None and (
        {"image_width", "image_height"} <= set(parsed) or {"width", "height"} <= set(parsed)
    ):
        width = _coerce_int(parsed.get("image_width", parsed.get("width")))
        height = _coerce_int(parsed.get("image_height", parsed.get("height")))
        if width is not None and height is not None:
            resolution = [width, height]
    if resolution is not None and len(resolution) == 2:
        calibration["resolution"] = resolution
    intrinsics = _coerce_float_list(parsed.get("intrinsics"))
    if intrinsics is None:
        intrinsics = _opencv_matrix_data(parsed.get("camera_matrix"))
    if intrinsics is None:
        intrinsics = _coerce_float_list(parsed.get("K"))
    if intrinsics is None:
        intrinsics = _coerce_float_list(parsed.get("P"))
    if intrinsics is None:
        fx = _coerce_float(parsed.get("fx"))
        fy = _coerce_float(parsed.get("fy"))
        cx = _coerce_float(parsed.get("cx"))
        cy = _coerce_float(parsed.get("cy"))
        if None not in (fx, fy, cx, cy):
            intrinsics = [float(fx), float(fy), float(cx), float(cy)]
    if intrinsics is not None:
        calibration["intrinsics"] = intrinsics
    coefficients = _coerce_float_list(parsed.get("distortion_coefficients"))
    if coefficients is None:
        coefficients = _coerce_float_list(parsed.get("D"))
    if coefficients is not None:
        calibration["distortion_coefficients"] = coefficients
    t_bs = parsed.get("T_BS")
    if isinstance(t_bs, Mapping):
        t_body_sensor = _coerce_float_list(t_bs.get("data"))
        if t_body_sensor is not None and len(t_body_sensor) == 16:
            calibration["T_body_sensor"] = t_body_sensor
    if len(calibration) == 4:
        warnings.append(f"camera calibration sidecar {path.name} did not contain recognized calibration fields")
        return None
    return calibration


def _opencv_matrix_data(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        return _coerce_float_list(value.get("data"))
    return _coerce_float_list(value)


def _parse_euroc_sensor_yaml(text: str, stream_id: str, warnings: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw_line = _strip_yaml_comment(lines[index])
        if not raw_line.strip():
            index += 1
            continue
        if raw_line[:1].isspace() or ":" not in raw_line:
            warnings.append(f"EuRoC calibration for {stream_id} skipped malformed line {index + 1}: {lines[index].strip()}")
            index += 1
            continue
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            value, index = _parse_yaml_value(raw_value, lines, index, stream_id, warnings)
            parsed[key] = value
            index += 1
            continue
        block: dict[str, Any] = {}
        index += 1
        while index < len(lines):
            nested_line = _strip_yaml_comment(lines[index])
            if not nested_line.strip():
                index += 1
                continue
            if not nested_line[:1].isspace():
                break
            if ":" not in nested_line:
                warnings.append(
                    f"EuRoC calibration for {stream_id} skipped malformed nested line {index + 1}: {lines[index].strip()}"
                )
                index += 1
                continue
            nested_key, nested_raw_value = nested_line.split(":", 1)
            nested_key = nested_key.strip()
            nested_raw_value = nested_raw_value.strip()
            value, index = _parse_yaml_value(nested_raw_value, lines, index, stream_id, warnings)
            block[nested_key] = value
            index += 1
        parsed[key] = block
    if not parsed:
        warnings.append(f"EuRoC calibration for {stream_id} did not contain parseable YAML fields")
    return parsed


def _strip_yaml_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _parse_yaml_value(
    raw_value: str, lines: Sequence[str], index: int, stream_id: str, warnings: list[str]
) -> tuple[Any, int]:
    value_text = raw_value
    if value_text.startswith("[") and "]" not in value_text:
        while index + 1 < len(lines):
            index += 1
            value_text += " " + _strip_yaml_comment(lines[index]).strip()
            if "]" in value_text:
                break
    try:
        return _parse_yaml_scalar(value_text), index
    except ValueError as exc:
        warnings.append(f"EuRoC calibration for {stream_id} could not parse value {value_text!r}: {exc}")
        return None, index


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(parsed, list):
            raise ValueError("expected a list")
        return parsed
    if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        return None
    return [float(item) for item in value]


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        return None
    return list(value)


def _dedupe_streams(streams: Sequence[DatasetStream]) -> tuple[DatasetStream, ...]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[DatasetStream] = []
    for stream in streams:
        key = (stream.stream_id, stream.modality, stream.source_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(stream)
    return tuple(deduped)


def _infer_parquet_modality(columns: Sequence[str]) -> str:
    return _infer_tabular_modality(columns)


def _infer_tabular_modality(columns: Sequence[str]) -> str:
    column_set = set(columns)
    if {"x", "y", "z", "qw", "qx", "qy", "qz"} <= column_set:
        return "pose"
    if {"p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"} <= column_set:
        return "pose"
    if {"ax", "ay", "az", "gx", "gy", "gz"} <= column_set:
        return "imu"
    if {"a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"} <= column_set:
        return "imu"
    if {"stream_id", "frame_path", "camera_bytes"} <= column_set:
        return "camera"
    if "filename" in column_set or "image" in column_set or "frame_path" in column_set:
        return "camera"
    return "media"


def _channels_for_columns(modality: str, columns: Sequence[str]) -> dict[str, str]:
    known = KNOWN_CHANNELS.get(modality, set())
    direct = {column: column for column in columns if column in known}
    if direct:
        return direct
    aliases = _channel_aliases(modality)
    return {logical: physical for logical, names in aliases.items() for physical in columns if physical in names}


def _channel_aliases(modality: str) -> dict[str, tuple[str, ...]]:
    if modality == "pose":
        return {
            "x": ("p_x", "pos_x", "position_x"),
            "y": ("p_y", "pos_y", "position_y"),
            "z": ("p_z", "pos_z", "position_z"),
            "qw": ("q_w", "quat_w", "orientation_w"),
            "qx": ("q_x", "quat_x", "orientation_x"),
            "qy": ("q_y", "quat_y", "orientation_y"),
            "qz": ("q_z", "quat_z", "orientation_z"),
            "vx": ("v_x", "vel_x", "velocity_x"),
            "vy": ("v_y", "vel_y", "velocity_y"),
            "vz": ("v_z", "vel_z", "velocity_z"),
        }
    if modality == "imu":
        return {
            "ax": ("accel_x", "linear_acceleration_x", "a_RS_S_x [m s^-2]"),
            "ay": ("accel_y", "linear_acceleration_y", "a_RS_S_y [m s^-2]"),
            "az": ("accel_z", "linear_acceleration_z", "a_RS_S_z [m s^-2]"),
            "gx": ("gyro_x", "angular_velocity_x", "w_RS_S_x [rad s^-1]"),
            "gy": ("gyro_y", "angular_velocity_y", "w_RS_S_y [rad s^-1]"),
            "gz": ("gyro_z", "angular_velocity_z", "w_RS_S_z [rad s^-1]"),
        }
    if modality == "camera":
        return {"frame_path": ("filename", "image", "path", "frame_path"), "camera_bytes": ("camera_bytes", "data")}
    return {}


def _pose_channels() -> dict[str, str]:
    return {name: name for name in ("x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz")}


def _imu_channels() -> dict[str, str]:
    return {name: name for name in ("ax", "ay", "az", "gx", "gy", "gz")}


def _camera_channels() -> dict[str, str]:
    return {"frame_path": "filename", "camera_bytes": "data"}


def _default_units(modality: str) -> dict[str, str]:
    if modality == "pose":
        return {"x": "m", "y": "m", "z": "m", "qw": "1", "qx": "1", "qy": "1", "qz": "1", "vx": "m/s", "vy": "m/s", "vz": "m/s"}
    if modality == "imu":
        return {"ax": "m/s^2", "ay": "m/s^2", "az": "m/s^2", "gx": "rad/s", "gy": "rad/s", "gz": "rad/s"}
    return {}


def _preferred_timestamp(candidates: Sequence[str]) -> str:
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in candidates:
            return candidate
    return candidates[0] if candidates else ""


def _timestamp_candidates_for_columns(columns: Sequence[str]) -> tuple[str, ...]:
    lowered = {column.lower(): column for column in columns}
    candidates = []
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in columns:
            candidates.append(candidate)
        elif candidate.lower() in lowered:
            candidates.append(lowered[candidate.lower()])
    for column in columns:
        name = column.lower()
        if column not in candidates and ("timestamp" in name or name.endswith("_time") or name in {"time", "stamp"}):
            candidates.append(column)
    return tuple(candidates)


def _infer_timestamp_unit(timestamp_name: str) -> str | None:
    name = timestamp_name.lower()
    if not name:
        return None
    if "timestamp_from_filename" == name:
        return None
    if "ns" in name or "nanosecond" in name:
        return "ns"
    if "us" in name or "microsecond" in name:
        return "us"
    if "ms" in name or "millisecond" in name:
        return "ms"
    if name.endswith("_s") or "seconds" in name:
        return "s"
    return None


def _mapping_confidence(
    modality: str, columns: Sequence[str], timestamps: Sequence[str], channels: Mapping[str, str]
) -> float:
    if modality == "media":
        return 0.3 if timestamps or channels else 0.15
    expected = KNOWN_CHANNELS.get(modality, set())
    required = {"x", "y", "z", "qw", "qx", "qy", "qz"} if modality == "pose" else expected
    coverage = len(set(channels) & required) / max(len(required), 1)
    score = 0.25 + 0.6 * coverage + (0.15 if timestamps else 0.0)
    return round(min(score, 0.95), 3)


def _infer_mcap_modality(topic: str, schema_name: str) -> str:
    text = f"{topic} {schema_name}".lower()
    if any(token in text for token in ("pose", "odometry", "tf", "transform", "nav_msgs/msg/odometry")):
        return "pose"
    if "imu" in text or "inertial" in text:
        return "imu"
    if any(token in text for token in ("image", "camera", "compressedimage")):
        return "camera"
    return "media"


def _infer_name_modality(name: str) -> str:
    lowered = name.lower()
    if "imu" in lowered:
        return "imu"
    if any(token in lowered for token in ("cam", "image", "rgb", "depth")):
        return "camera"
    if any(token in lowered for token in ("pose", "odom", "oxts", "ego")):
        return "pose"
    return "media"


def _default_channels_for_modality(modality: str) -> dict[str, str]:
    if modality == "pose":
        return _pose_channels()
    if modality == "imu":
        return _imu_channels()
    if modality == "camera":
        return {"frame_path": "uri"}
    return {}


def _stream_id_from_path(path: Path, modality: str) -> str:
    if modality == "pose":
        return "pose" if path.stem in {"data", "session"} else path.stem
    if modality == "imu" and "imu" not in path.stem:
        return f"{path.stem}_imu"
    return path.stem or modality


def _stream_id_from_name(name: str, modality: str) -> str:
    stem = Path(name).stem.replace(".", "_").replace("-", "_")
    return stem or modality


def _image_timestamp_candidates(images: Sequence[Path]) -> tuple[str, ...]:
    if not images:
        return ()
    numeric = 0
    for image in images[:50]:
        stem = image.stem
        digits = "".join(char for char in stem if char.isdigit())
        if len(digits) >= 6:
            numeric += 1
    return ("timestamp_from_filename",) if numeric >= max(1, min(len(images), 50) // 2) else ()


def _profile_discovery(files: Sequence[DatasetFile], streams: Sequence[DatasetStream]) -> dict[str, Any]:
    return {
        "file_count": len(files),
        "size_bytes": sum(file.size_bytes for file in files),
        "candidate_streams": [
            {
                "stream_id": stream.stream_id,
                "modality": stream.modality,
                "source_path": stream.source_path,
                "confidence": stream.confidence,
                "warnings": list(stream.warnings),
            }
            for stream in streams
        ],
    }


def _dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://") or uri.startswith("s3a://")


def _source_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".mcap":
        return "mcap"
    if suffix == ".zip":
        return "zip"
    path_obj = Path(path)
    if path_obj.is_dir() and _find_euroc_root(path_obj) is not None:
        return "euroc"
    if _looks_like_kitti_oxts(path_obj):
        return "kitti_oxts"
    if _looks_like_nuscenes_ego(path_obj):
        return "nuscenes_ego"
    return "directory" if path_obj.is_dir() else "file"


def _looks_like_kitti_oxts(path: Path) -> bool:
    if not path.exists():
        return False
    return _kitti_source_root(path) is not None


def _kitti_source_root(path: Path) -> Path | None:
    candidates = [path]
    if path.is_dir():
        candidates.extend(item for item in path.rglob("*") if item.is_dir())
    for candidate in candidates:
        if (candidate / "oxts" / "data").is_dir() or (candidate / "data").is_dir() and candidate.name == "oxts":
            return candidate if (candidate / "oxts" / "data").is_dir() else candidate.parent
        if candidate.name == "data" and candidate.parent.name == "oxts":
            return candidate.parent.parent
    if path.is_file() and path.parent.name == "data" and path.parent.parent.name == "oxts":
        return path.parent.parent.parent
    return None


def _looks_like_nuscenes_ego(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.name == "ego_pose.json"
    if (path / "ego_pose.json").exists():
        return True
    return any(candidate.name == "ego_pose.json" for candidate in path.glob("v*/*"))


def _profile_with_adapter(profile: DatasetProfile, adapter_id: str) -> DatasetProfile:
    return DatasetProfile(
        version=profile.version,
        input_uri=profile.input_uri,
        dataset_format=profile.dataset_format,
        files=profile.files,
        streams=profile.streams,
        warnings=profile.warnings,
        adapter_id=adapter_id,
        discovery=profile.discovery,
    )


def _stream_type(modality: str) -> str:
    if modality in {"pose", "imu", "camera"}:
        return modality
    return "generic_media"


def _load_manifest(manifest: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    return json.loads(Path(manifest).read_text(encoding="utf-8"))


def _euroc_ingest_command(
    *,
    modality: str,
    source_path: str,
    out_parquet: Path,
    stream_id: str,
    robot_id: str,
    session_id: str,
    row_group_rows: int,
    robotics_bin: str | os.PathLike[str] | None,
) -> list[str] | None:
    base = _robotics_command(robotics_bin) + ["ingest"]
    if modality == "pose":
        return base + ["euroc-groundtruth", "--input", source_path, "--out", str(out_parquet), "--robot-id", robot_id, "--session-id", session_id, "--row-group-rows", str(row_group_rows)]
    if modality == "imu":
        return base + ["euroc-imu", "--input", source_path, "--out", str(out_parquet), "--robot-id", robot_id, "--session-id", session_id, "--row-group-rows", str(row_group_rows)]
    if modality == "camera":
        return base + ["euroc-camera", "--input", source_path, "--out", str(out_parquet), "--stream-id", stream_id, "--robot-id", robot_id, "--session-id", session_id, "--row-group-rows", str(row_group_rows)]
    return None


def _mcap_ingest_command(
    *,
    source_path: str,
    out_parquet: Path,
    topic: str,
    robot_id: str,
    session_id: str,
    row_group_rows: int,
    robotics_bin: str | os.PathLike[str] | None,
) -> list[str]:
    return _robotics_command(robotics_bin) + [
        "ingest",
        "mcap-pose",
        "--input",
        source_path,
        "--out",
        str(out_parquet),
        "--topic",
        topic,
        "--robot-id",
        robot_id,
        "--session-id",
        session_id,
        "--row-group-rows",
        str(row_group_rows),
    ]


def _single_pose_ingest_command(
    *,
    subcommand: str,
    source_path: str,
    out_parquet: Path,
    robot_id: str,
    session_id: str,
    robotics_bin: str | os.PathLike[str] | None,
    row_group_rows: int = 500,
) -> list[str]:
    return _robotics_command(robotics_bin) + [
        "ingest",
        subcommand,
        "--input",
        source_path,
        "--out",
        str(out_parquet),
        "--robot-id",
        robot_id,
        "--session-id",
        session_id,
        "--row-group-rows",
        str(row_group_rows),
    ]


def _catalog_command(
    *,
    modality: str,
    parquet_path: Path,
    catalog_path: Path,
    stream_id: str,
    robotics_bin: str | os.PathLike[str] | None,
    uri_path: Path | str | None = None,
) -> list[str] | None:
    base = _robotics_command(robotics_bin) + ["catalog"]
    uri = str(uri_path) if uri_path is not None else str(parquet_path)
    if modality == "pose":
        return base + ["build", "--input", str(parquet_path), "--out", str(catalog_path), "--uri", uri]
    if modality == "imu":
        return base + ["build-imu", "--input", str(parquet_path), "--out", str(catalog_path), "--uri", uri]
    if modality == "camera":
        return base + ["build-media", "--input", str(parquet_path), "--out", str(catalog_path), "--uri", uri, "--modality", "camera", "--stream-id", stream_id]
    return None


def _run_robotics(command: Sequence[str]) -> dict[str, int]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return _parse_cli_metrics(completed.stdout)


def _preflight_generic_dataset(data: Mapping[str, Any], output_root: Path) -> None:
    _validate_generic_final_mapping(data)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"generic_dataset output_root already exists and is not empty: {output_root}")
    sources = {str(source["source_id"]): source for source in data.get("sources", []) if isinstance(source, Mapping)}
    errors: list[str] = []
    for stream in data.get("streams", []):
        if not isinstance(stream, Mapping):
            continue
        stream_id = str(stream.get("stream_id", ""))
        modality = str(stream.get("modality", ""))
        source = sources.get(str(stream.get("source_id", "")), {})
        source_path = str(source.get("path", ""))
        source_type = str(source.get("type") or _source_type(source_path))
        if not source_path or _is_s3_uri(source_path):
            continue
        if modality in {"pose", "imu"} and source_type != "mcap":
            errors.extend(_preflight_generic_tabular_stream(source_path, source_type, stream))
        elif modality == "camera":
            errors.extend(_preflight_generic_camera_stream(source_path, stream))
    if errors:
        raise ValueError("invalid generic_dataset final mapping: " + "; ".join(errors))


def _preflight_generic_tabular_stream(
    source_path: str, source_type: str, stream: Mapping[str, Any]
) -> list[str]:
    stream_id = str(stream.get("stream_id", ""))
    tabular_type = _generic_tabular_source_type(source_type, source_path)
    columns = _generic_tabular_columns(Path(source_path), tabular_type)
    column_set = set(columns)
    errors: list[str] = []
    timestamp = str(stream.get("timestamp") or "")
    if timestamp not in column_set:
        errors.append(f"stream {stream_id} timestamp column {timestamp!r} not found in {source_path}")
    channels = stream.get("channels", {})
    if isinstance(channels, Mapping):
        for logical, physical in channels.items():
            if not physical:
                continue
            if str(physical) not in column_set:
                errors.append(
                    f"stream {stream_id} channel {logical} maps to missing column {str(physical)!r} in {source_path}"
                )
    if timestamp in column_set:
        cast_error = _generic_timestamp_cast_error(Path(source_path), tabular_type, timestamp, stream)
        if cast_error:
            errors.append(
                f"stream {stream_id} timestamp column {timestamp!r} cannot be cast to BIGINT timestamp_ns: {cast_error}"
            )
    return errors


def _preflight_generic_camera_stream(source_path: str, stream: Mapping[str, Any]) -> list[str]:
    stream_id = str(stream.get("stream_id", ""))
    path = Path(source_path)
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        return [f"stream {stream_id} camera source has no image files: {source_path}"]
    try:
        _timestamp_scale_to_ns(stream, timestamp=str(stream.get("timestamp") or "timestamp_from_filename"))
    except ValueError as exc:
        return [f"stream {stream_id} camera timestamp metadata is invalid: {exc}"]
    if not any(_timestamp_from_filename(image) is not None for image in images):
        return [f"stream {stream_id} camera image filenames do not contain parseable timestamps: {source_path}"]
    return []


def _validate_generic_final_mapping(data: Mapping[str, Any]) -> None:
    if str(data.get("mapping_status") or "") != "final":
        raise ValueError("generic_dataset ingest requires top-level mapping_status='final'")
    sources = {str(source["source_id"]): source for source in data.get("sources", []) if isinstance(source, Mapping)}
    errors: list[str] = []
    for index, stream in enumerate(data.get("streams", [])):
        if not isinstance(stream, Mapping):
            continue
        prefix = f"streams[{index}]"
        stream_id = str(stream.get("stream_id", index))
        if str(stream.get("mapping_status") or data.get("mapping_status") or "") != "final":
            errors.append(f"{prefix} ({stream_id}) requires mapping_status='final'")
        modality = str(stream.get("modality", ""))
        source = sources.get(str(stream.get("source_id", "")), {})
        source_path = str(source.get("path", ""))
        source_type = str(source.get("type") or _source_type(source_path))
        if _is_s3_uri(source_path):
            errors.append(f"{prefix} ({stream_id}) S3 raw ingest is not supported in generic_dataset v1")
        if modality in {"pose", "imu"}:
            required = {"x", "y", "z", "qw", "qx", "qy", "qz"} if modality == "pose" else {"ax", "ay", "az", "gx", "gy", "gz"}
            channels = stream.get("channels", {})
            missing = sorted(required - set(channels if isinstance(channels, Mapping) else ()))
            if missing:
                errors.append(f"{prefix} ({stream_id}) missing required {modality} channels: {', '.join(missing)}")
            if _generic_tabular_source_type(source_type, source_path) not in {"csv", "parquet"} and source_type != "mcap":
                errors.append(f"{prefix} ({stream_id}) {modality} source must be local CSV, Parquet, or MCAP")
            if source_type == "mcap" and modality != "pose":
                errors.append(f"{prefix} ({stream_id}) generic MCAP ingest only supports pose streams in v1")
            if source_type == "mcap" and not _generic_mcap_topic(data, stream, default=""):
                errors.append(f"{prefix} ({stream_id}) generic MCAP pose ingest requires an explicit topic")
        elif modality == "camera":
            if not _generic_camera_sequence_source(source_type, source_path):
                errors.append(f"{prefix} ({stream_id}) camera source must be a local image sequence directory")
            if str(stream.get("timestamp") or "") != "timestamp_from_filename":
                errors.append(f"{prefix} ({stream_id}) camera image sequence requires timestamp='timestamp_from_filename'")
        else:
            errors.append(f"{prefix} ({stream_id}) modality {modality} is not supported by generic ingest")
    if errors:
        raise ValueError("invalid generic_dataset final mapping: " + "; ".join(errors))


def _generic_tabular_source_type(source_type: str, source_path: str) -> str:
    suffix = Path(source_path).suffix.lower()
    if source_type in {"csv", "parquet"}:
        return source_type
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    return source_type


def _generic_relation_sql(source_path: Path, source_type: str) -> str:
    if source_type == "csv":
        return f"read_csv_auto({_duckdb_literal(str(source_path))}, header=true)"
    if source_type == "parquet":
        return f"read_parquet({_duckdb_literal(str(source_path))})"
    raise ValueError(f"unsupported generic tabular source type {source_type!r} for {source_path}")


def _generic_tabular_columns(source_path: Path, source_type: str) -> tuple[str, ...]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - tests skip when duckdb is absent
        raise RuntimeError("generic_dataset ingest requires duckdb") from exc
    if not source_path.exists():
        raise ValueError(f"generic_dataset source does not exist: {source_path}")
    relation = _generic_relation_sql(source_path, source_type)
    try:
        with duckdb.connect(":memory:") as con:
            return tuple(str(row[0]) for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall())
    except Exception as exc:
        raise ValueError(f"failed to inspect generic_dataset source {source_path}: {exc}") from exc


def _generic_timestamp_cast_error(
    source_path: Path, source_type: str, timestamp: str, stream: Mapping[str, Any]
) -> str:
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - tests skip when duckdb is absent
        raise RuntimeError("generic_dataset ingest requires duckdb") from exc
    relation = _generic_relation_sql(source_path, source_type)
    try:
        timestamp_expr = _generic_timestamp_expr_sql(timestamp, stream)
    except ValueError as exc:
        return str(exc)
    try:
        with duckdb.connect(":memory:") as con:
            bad_count = con.execute(
                f"SELECT count(*) FROM {relation} WHERE TRY_CAST({_identifier(timestamp)} AS DOUBLE) IS NULL"
            ).fetchone()[0]
            if not int(bad_count):
                con.execute(f"SELECT {timestamp_expr} FROM {relation} LIMIT 1").fetchone()
        return f"{bad_count} row(s) failed timestamp conversion" if int(bad_count) else ""
    except Exception as exc:
        return str(exc)


def _generic_camera_sequence_source(source_type: str, source_path: str) -> bool:
    path = Path(source_path)
    if source_type not in {"directory", "file"} and not path.is_dir():
        return False
    return path.is_dir() and any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.rglob("*"))


def _generic_mcap_topic(
    data: Mapping[str, Any], stream: Mapping[str, Any], *, default: str | None = "/pose"
) -> str:
    discovery = stream.get("discovery", {})
    if isinstance(discovery, Mapping) and discovery.get("topic"):
        return str(discovery["topic"])
    options = data.get("adapter_options", {})
    if isinstance(options, Mapping) and options.get("topic"):
        return str(options["topic"])
    return "" if default is None else str(default)


def _write_generic_tabular_parquet(
    *,
    source_path: Path,
    source_type: str,
    stream: Mapping[str, Any],
    out_parquet: Path,
    robot_id: str,
    session_id: str,
) -> None:
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - tests skip when duckdb is absent
        raise RuntimeError("generic_dataset ingest requires duckdb") from exc
    if not source_path.exists():
        raise ValueError(f"generic_dataset source does not exist: {source_path}")
    relation = _generic_relation_sql(source_path, source_type)
    modality = str(stream["modality"])
    timestamp = str(stream.get("timestamp") or "")
    channels = stream.get("channels", {})
    if not isinstance(channels, Mapping):
        raise ValueError(f"stream {stream.get('stream_id')} channels must be an object")
    timestamp_expr = _generic_timestamp_expr_sql(timestamp, stream)
    if modality == "pose":
        select_sql = _generic_pose_select_sql(relation, timestamp_expr, channels, robot_id, session_id)
    elif modality == "imu":
        select_sql = _generic_imu_select_sql(relation, timestamp_expr, channels, robot_id, session_id)
    else:
        raise ValueError(f"generic tabular ingest does not support modality {modality}")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        con.execute(f"COPY ({select_sql}) TO {_duckdb_literal(str(out_parquet))} (FORMAT PARQUET)")


def _generic_pose_select_sql(
    relation: str, timestamp_expr: str, channels: Mapping[str, Any], robot_id: str, session_id: str
) -> str:
    vx = _pose_velocity_sql(channels, "vx", "x")
    vy = _pose_velocity_sql(channels, "vy", "y")
    vz = _pose_velocity_sql(channels, "vz", "z")
    return "\n".join(
        [
            "WITH normalized AS (",
            "SELECT",
            f"  {timestamp_expr} AS timestamp_ns,",
            f"  {_duckdb_literal(robot_id)} AS robot_id,",
            f"  {_duckdb_literal(session_id)} AS session_id,",
            f"  {_numeric_channel_sql(channels, 'x')} AS x,",
            f"  {_numeric_channel_sql(channels, 'y')} AS y,",
            f"  {_numeric_channel_sql(channels, 'z')} AS z,",
            f"  {_numeric_channel_sql(channels, 'qw')} AS qw,",
            f"  {_numeric_channel_sql(channels, 'qx')} AS qx,",
            f"  {_numeric_channel_sql(channels, 'qy')} AS qy,",
            f"  {_numeric_channel_sql(channels, 'qz')} AS qz,",
            f"  {_optional_numeric_channel_sql(channels, 'vx')} AS vx_input,",
            f"  {_optional_numeric_channel_sql(channels, 'vy')} AS vy_input,",
            f"  {_optional_numeric_channel_sql(channels, 'vz')} AS vz_input",
            f"FROM {relation}",
            "), windowed AS (",
            "SELECT",
            "  *,",
            "  lag(timestamp_ns) OVER (ORDER BY timestamp_ns) AS prev_timestamp_ns,",
            "  lead(timestamp_ns) OVER (ORDER BY timestamp_ns) AS next_timestamp_ns,",
            "  lag(x) OVER (ORDER BY timestamp_ns) AS prev_x,",
            "  lead(x) OVER (ORDER BY timestamp_ns) AS next_x,",
            "  lag(y) OVER (ORDER BY timestamp_ns) AS prev_y,",
            "  lead(y) OVER (ORDER BY timestamp_ns) AS next_y,",
            "  lag(z) OVER (ORDER BY timestamp_ns) AS prev_z,",
            "  lead(z) OVER (ORDER BY timestamp_ns) AS next_z",
            "FROM normalized",
            ")",
            "SELECT",
            "  timestamp_ns,",
            "  robot_id,",
            "  session_id,",
            "  x,",
            "  y,",
            "  z,",
            "  qw,",
            "  qx,",
            "  qy,",
            "  qz,",
            f"  {vx} AS vx,",
            f"  {vy} AS vy,",
            f"  {vz} AS vz,",
            f"  sqrt(({vx}) * ({vx}) + ({vy}) * ({vy}) + ({vz}) * ({vz})) AS velocity",
            "FROM windowed",
            "ORDER BY timestamp_ns",
        ]
    )


def _generic_imu_select_sql(
    relation: str, timestamp_expr: str, channels: Mapping[str, Any], robot_id: str, session_id: str
) -> str:
    return "\n".join(
        [
            "SELECT",
            f"  {timestamp_expr} AS timestamp_ns,",
            f"  {_duckdb_literal(robot_id)} AS robot_id,",
            f"  {_duckdb_literal(session_id)} AS session_id,",
            f"  {_numeric_channel_sql(channels, 'ax')} AS ax,",
            f"  {_numeric_channel_sql(channels, 'ay')} AS ay,",
            f"  {_numeric_channel_sql(channels, 'az')} AS az,",
            f"  {_numeric_channel_sql(channels, 'gx')} AS gx,",
            f"  {_numeric_channel_sql(channels, 'gy')} AS gy,",
            f"  {_numeric_channel_sql(channels, 'gz')} AS gz",
            f"FROM {relation}",
            "ORDER BY timestamp_ns",
        ]
    )


def _generic_timestamp_expr_sql(timestamp: str, stream: Mapping[str, Any]) -> str:
    scale = _timestamp_scale_to_ns(stream, timestamp=timestamp)
    return f"CAST(round(CAST({_identifier(timestamp)} AS DOUBLE) * {scale:.17g}) AS BIGINT)"


def _timestamp_scale_to_ns(stream: Mapping[str, Any], *, timestamp: str = "") -> float:
    raw_scale = stream.get("timestamp_scale")
    if raw_scale is not None:
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timestamp_scale must be numeric, got {raw_scale!r}") from exc
        if scale <= 0:
            raise ValueError(f"timestamp_scale must be positive, got {raw_scale!r}")
        return scale
    raw_unit = stream.get("timestamp_unit")
    if raw_unit is None:
        discovery = stream.get("discovery", {})
        if isinstance(discovery, Mapping):
            raw_unit = discovery.get("timestamp_unit")
    unit = _normalize_timestamp_unit(str(raw_unit)) if raw_unit is not None else _infer_timestamp_unit(timestamp)
    return TIMESTAMP_UNIT_TO_NS.get(unit or "ns", 1.0)


def _normalize_timestamp_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized not in TIMESTAMP_UNIT_TO_NS:
        raise ValueError(f"unsupported timestamp_unit {unit!r}")
    return normalized


def _pose_velocity_sql(channels: Mapping[str, Any], velocity_name: str, position_name: str) -> str:
    return (
        f"COALESCE({velocity_name}_input, CASE "
        f"WHEN prev_timestamp_ns IS NOT NULL AND timestamp_ns != prev_timestamp_ns "
        f"THEN ({position_name} - prev_{position_name}) / ((timestamp_ns - prev_timestamp_ns) / 1000000000.0) "
        f"WHEN next_timestamp_ns IS NOT NULL AND next_timestamp_ns != timestamp_ns "
        f"THEN (next_{position_name} - {position_name}) / ((next_timestamp_ns - timestamp_ns) / 1000000000.0) "
        "ELSE 0.0 END)"
    )


def _optional_numeric_channel_sql(channels: Mapping[str, Any], logical_name: str) -> str:
    physical = channels.get(logical_name)
    if not physical:
        return "CAST(NULL AS DOUBLE)"
    return f"CAST({_identifier(str(physical))} AS DOUBLE)"


def _numeric_channel_sql(channels: Mapping[str, Any], logical_name: str, *, default: str | None = None) -> str:
    physical = channels.get(logical_name)
    if not physical:
        if default is not None:
            return f"CAST({default} AS DOUBLE)"
        raise ValueError(f"missing channel mapping for {logical_name}")
    return f"CAST({_identifier(str(physical))} AS DOUBLE)"


def _write_generic_camera_parquet(
    *,
    source_path: Path,
    stream: Mapping[str, Any],
    out_parquet: Path,
    robot_id: str,
    session_id: str,
) -> None:
    try:
        import duckdb
    except ModuleNotFoundError as exc:  # pragma: no cover - tests skip when duckdb is absent
        raise RuntimeError("generic_dataset ingest requires duckdb") from exc
    images = sorted(item for item in source_path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    timestamp_scale = _timestamp_scale_to_ns(stream, timestamp=str(stream.get("timestamp") or "timestamp_from_filename"))
    rows = [
        (
            _scale_filename_timestamp(_timestamp_from_filename(image), timestamp_scale),
            robot_id,
            session_id,
            str(stream["stream_id"]),
            str(image),
            image.read_bytes(),
        )
        for image in images
    ]
    rows = [row for row in rows if row[0] is not None]
    if not rows:
        raise ValueError(f"camera image sequence {source_path} has no timestamped image filenames")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        con.execute(
            "CREATE TABLE frames(timestamp_ns BIGINT, robot_id VARCHAR, session_id VARCHAR, "
            "stream_id VARCHAR, frame_path VARCHAR, camera_bytes BLOB)"
        )
        con.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", rows)
        con.execute(
            f"COPY (SELECT * FROM frames ORDER BY timestamp_ns) TO {_duckdb_literal(str(out_parquet))} "
            "(FORMAT PARQUET)"
        )


def _timestamp_from_filename(path: Path) -> int | None:
    digits = "".join(char for char in path.stem if char.isdigit())
    if len(digits) < 6:
        return None
    return int(digits)


def _scale_filename_timestamp(timestamp: int | None, scale: float) -> int | None:
    if timestamp is None:
        return None
    return int(round(float(timestamp) * scale))


def _identifier(name: str) -> str:
    if not name:
        raise ValueError("empty column identifier")
    return '"' + name.replace('"', '""') + '"'


def _duckdb_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _replace_path_prefix(value: str, old_root: Path, new_root: Path) -> str:
    old = str(old_root)
    if value == old:
        return str(new_root)
    prefix = old + os.sep
    if value.startswith(prefix):
        return str(new_root / value[len(prefix) :])
    return value
