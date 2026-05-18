from __future__ import annotations

import os
import json
import math
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

CHANNELS = {
    "pos_xyz": (0, 1, 2),
    "rot_wxyz": (3, 4, 5, 6),
    "vel_xyz": (7, 8, 9),
    "imu_accel": (10, 11, 12),
    "imu_gyro": (13, 14, 15),
}

DEFAULT_FOOTER_ALLOWANCE_BYTES = 16 * 1024 * 1024


class EgressLimitError(RuntimeError):
    pass


class TemporalGapError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorrectnessReport:
    pose_gap_count: int
    imu_gap_count: int
    pose_max_gap_ns: int
    imu_max_gap_ns: int
    null_count: int
    quaternion_inversions_applied: int
    extrapolation_checked: bool
    extrapolation_rejected: bool
    velocity_threshold: float | None
    matched_row_groups: int
    selected_bytes: int
    pose_matched_row_groups: int = 0
    pose_selected_bytes: int = 0
    imu_matched_row_groups: int = 0
    imu_selected_bytes: int = 0
    total_selected_bytes: int = 0
    pose_max_gap_start_ts_ns: int = 0
    pose_max_gap_end_ts_ns: int = 0
    imu_max_gap_start_ts_ns: int = 0
    imu_max_gap_end_ts_ns: int = 0
    pose_planned_range_reads: int = 0
    imu_planned_range_reads: int = 0
    planned_read_bytes: int = 0
    range_audit_passed: bool = False
    catalog_query_ms: float = 0.0
    candidate_row_groups: int = 0
    time_pruned_row_groups: int = 0
    spatial_pruned_row_groups: int = 0
    velocity_pruned_row_groups: int = 0
    gap_rejected_row_groups: int = 0
    media_matched_row_groups: int = 0
    media_selected_bytes: int = 0
    media_blocked_by_egress: bool = False
    authorized_pose_bytes: int = 0
    authorized_imu_bytes: int = 0
    authorized_media_bytes: int = 0
    authorized_total_bytes: int = 0
    materialized_pose_imu_bytes: int = 0
    planned_range_reads: int = 0
    blocked_by_egress: bool = False
    actual_cold_reads: int = 0
    actual_cold_read_bytes: int = 0
    actual_authorized_bytes: int = 0
    footer_allowance_bytes: int = DEFAULT_FOOTER_ALLOWANCE_BYTES
    footer_bytes: int = 0
    largest_metadata_read: int = 0
    max_footer_read_offset: int = 0
    max_footer_read_end: int = 0
    range_enforced: bool = False
    range_violations: int = 0

    def log_lines(self) -> tuple[str, ...]:
        return (
            f"pose_gaps_detected={self.pose_gap_count}",
            f"imu_gaps_detected={self.imu_gap_count}",
            f"pose_max_gap_ns={self.pose_max_gap_ns}",
            f"imu_max_gap_ns={self.imu_max_gap_ns}",
            f"pose_max_gap_start_ts_ns={self.pose_max_gap_start_ts_ns}",
            f"pose_max_gap_end_ts_ns={self.pose_max_gap_end_ts_ns}",
            f"imu_max_gap_start_ts_ns={self.imu_max_gap_start_ts_ns}",
            f"imu_max_gap_end_ts_ns={self.imu_max_gap_end_ts_ns}",
            f"nulls_handled={self.null_count}",
            f"quaternion_inversions_applied={self.quaternion_inversions_applied}",
            f"extrapolation_checked={str(self.extrapolation_checked).lower()}",
            f"extrapolation_rejected={str(self.extrapolation_rejected).lower()}",
            f"velocity_threshold={self.velocity_threshold}",
            f"matched_row_groups={self.matched_row_groups}",
            f"selected_bytes={self.selected_bytes}",
            f"pose_matched_row_groups={self.pose_matched_row_groups}",
            f"pose_selected_bytes={self.pose_selected_bytes}",
            f"imu_matched_row_groups={self.imu_matched_row_groups}",
            f"imu_selected_bytes={self.imu_selected_bytes}",
            f"total_selected_bytes={self.total_selected_bytes}",
            f"pose_planned_range_reads={self.pose_planned_range_reads}",
            f"imu_planned_range_reads={self.imu_planned_range_reads}",
            f"planned_read_bytes={self.planned_read_bytes}",
            f"range_audit_passed={str(self.range_audit_passed).lower()}",
            f"catalog_query_ms={self.catalog_query_ms:.3f}",
            f"candidate_row_groups={self.candidate_row_groups}",
            f"time_pruned_row_groups={self.time_pruned_row_groups}",
            f"spatial_pruned_row_groups={self.spatial_pruned_row_groups}",
            f"velocity_pruned_row_groups={self.velocity_pruned_row_groups}",
            f"gap_rejected_row_groups={self.gap_rejected_row_groups}",
            f"media_matched_row_groups={self.media_matched_row_groups}",
            f"media_selected_bytes={self.media_selected_bytes}",
            f"media_blocked_by_egress={str(self.media_blocked_by_egress).lower()}",
            f"authorized_pose_bytes={self.authorized_pose_bytes}",
            f"authorized_imu_bytes={self.authorized_imu_bytes}",
            f"authorized_media_bytes={self.authorized_media_bytes}",
            f"authorized_total_bytes={self.authorized_total_bytes}",
            f"materialized_pose_imu_bytes={self.materialized_pose_imu_bytes}",
            f"planned_range_reads={self.planned_range_reads}",
            f"blocked_by_egress={str(self.blocked_by_egress).lower()}",
            f"actual_cold_reads={self.actual_cold_reads}",
            f"actual_cold_read_bytes={self.actual_cold_read_bytes}",
            f"actual_authorized_bytes={self.actual_authorized_bytes}",
            f"footer_allowance_bytes={self.footer_allowance_bytes}",
            f"footer_bytes={self.footer_bytes}",
            f"largest_metadata_read={self.largest_metadata_read}",
            f"max_footer_read_offset={self.max_footer_read_offset}",
            f"max_footer_read_end={self.max_footer_read_end}",
            f"range_enforced={str(self.range_enforced).lower()}",
            f"range_violations={self.range_violations}",
        )


@dataclass(frozen=True)
class RowGroupSpan:
    modality: str
    file_uri: str
    row_group_id: int
    start_ts_ns: int
    end_ts_ns: int
    byte_offset: int
    byte_length: int
    stream_id: str = ""


@dataclass(frozen=True)
class SeekPlan:
    start_ts_ns: int
    end_ts_ns: int
    pose_file_uri: str
    pose_row_groups: tuple[RowGroupSpan, ...]
    imu_row_groups: tuple[RowGroupSpan, ...]
    media_row_groups: tuple[RowGroupSpan, ...]
    authorized_pose_bytes: int
    authorized_imu_bytes: int
    authorized_media_bytes: int
    authorized_total_bytes: int
    materialized_pose_imu_bytes: int
    planned_range_reads: int
    blocked_by_egress: bool
    egress_limit_bytes: int
    diagnostics: CorrectnessReport
    _pose_rows: tuple[dict[str, object], ...] = field(default=(), repr=False, compare=False)
    _imu_rows: tuple[dict[str, object], ...] = field(default=(), repr=False, compare=False)
    _media_rows: tuple[dict[str, object], ...] = field(default=(), repr=False, compare=False)

    @property
    def row_groups(self) -> tuple[int, ...]:
        return tuple(span.row_group_id for span in self.pose_row_groups)

    def to_manifest(self) -> dict[str, object]:
        return {
            "version": 1,
            "start_ts_ns": self.start_ts_ns,
            "end_ts_ns": self.end_ts_ns,
            "selected_row_groups": [asdict(span) for span in self.pose_row_groups],
            "authorized_spans": [asdict(span) for span in (*self.pose_row_groups, *self.imu_row_groups)],
            "media_planned_spans": [asdict(span) for span in self.media_row_groups],
            "authorized_pose_bytes": self.authorized_pose_bytes,
            "authorized_imu_bytes": self.authorized_imu_bytes,
            "authorized_media_bytes": self.authorized_media_bytes,
            "authorized_total_bytes": self.authorized_total_bytes,
            "materialized_pose_imu_bytes": self.materialized_pose_imu_bytes,
            "planned_range_reads": self.planned_range_reads,
            "blocked_by_egress": self.blocked_by_egress,
            "diagnostics": asdict(self.diagnostics),
        }


@dataclass(frozen=True)
class QueryResult:
    tensor: object
    timestamps_ns: np.ndarray
    row_groups: tuple[int, ...]
    file_uri: str
    selected_bytes: int
    output: str
    diagnostics: CorrectnessReport
    manifest: dict[str, object] | None = None


@dataclass(frozen=True)
class _PredicateFilters:
    sql: tuple[str, ...] = ()
    params: tuple[object, ...] = ()
    velocity_threshold: float | None = None
    velocity_conditions: tuple[tuple[str, float], ...] = ()
    bboxes: tuple[tuple[float, float, float, float, float, float], ...] = ()
    time_windows: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _CatalogExplain:
    catalog_query_ms: float = 0.0
    candidate_row_groups: int = 0
    time_pruned_row_groups: int = 0
    spatial_pruned_row_groups: int = 0
    velocity_pruned_row_groups: int = 0


def query(
    *,
    catalog: str | os.PathLike[str] | None = None,
    catalog_db: str | os.PathLike[str] | None = None,
    robot_id: str | None = None,
    start_ts_ns: int | None = None,
    end_ts_ns: int | None = None,
    bbox: Sequence[float] | None = None,
    min_velocity: float | None = None,
    predicate: str | None = None,
    channels: Sequence[str] = ("pos_xyz", "rot_wxyz", "vel_xyz"),
    target_hz: float = 30.0,
    output: Literal["numpy", "torch"] = "numpy",
    source: str | os.PathLike[str] | None = None,
    imu_source: str | os.PathLike[str] | None = None,
    imu_catalog: str | os.PathLike[str] | None = None,
    media_catalog: str | os.PathLike[str] | None = None,
    max_egress_bytes: int = 1_000_000_000,
    limit: int | None = None,
    gap_policy: Literal["reject", "allow"] = "reject",
    enforce_ranges: bool = False,
    footer_allowance_bytes: int = DEFAULT_FOOTER_ALLOWANCE_BYTES,
    manifest_out: str | os.PathLike[str] | None = None,
    robotics_bin: str | os.PathLike[str] | None = None,
) -> QueryResult:
    """Query the hot catalog with DuckDB and return a training-shaped tensor.

    The current bridge materializes selected Parquet row groups through the Rust
    CLI and loads `.npy` files into NumPy/PyTorch. It preserves the product API
    shape while DLPack/zero-copy Arrow interop is still pending.
    """

    if output not in {"numpy", "torch"}:
        raise ValueError("output must be 'numpy' or 'torch'")
    scalar_channels, _media_channels = _split_channels(channels)
    needs_imu = any(channel.startswith("imu_") for channel in scalar_channels)
    channel_indices = _channel_indices(scalar_channels)
    seek_plan = plan(
        catalog=catalog,
        catalog_db=catalog_db,
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        bbox=bbox,
        min_velocity=min_velocity,
        predicate=predicate,
        channels=channels,
        source=source,
        imu_source=imu_source,
        imu_catalog=imu_catalog,
        media_catalog=media_catalog,
        max_egress_bytes=max_egress_bytes,
        limit=limit,
        gap_policy=gap_policy,
    )
    if seek_plan.blocked_by_egress:
        raise EgressLimitError(
            "query selected "
            f"{seek_plan.authorized_total_bytes} bytes across pose+IMU+media, "
            f"including {seek_plan.authorized_media_bytes} media bytes, "
            f"exceeding max_egress_bytes={seek_plan.egress_limit_bytes}"
        )

    pose_values, timestamps, pose_metrics, pose_manifest = _materialize_with_rust(
        source=seek_plan.pose_file_uri,
        row_groups=seek_plan.row_groups,
        start_ts_ns=seek_plan.start_ts_ns,
        end_ts_ns=seek_plan.end_ts_ns,
        target_hz=target_hz,
        audit_ranges=_catalog_audit_ranges(seek_plan._pose_rows),
        enforce_ranges=enforce_ranges,
        footer_allowance_bytes=footer_allowance_bytes,
        robotics_bin=robotics_bin,
    )
    if needs_imu:
        imu_row_groups = tuple(span.row_group_id for span in seek_plan.imu_row_groups)
        imu_file_uri = str(imu_source) if imu_source is not None else _single_file_uri(seek_plan._imu_rows, "IMU")
        imu_values, imu_null_count, imu_gap_count, imu_max_gap_ns, imu_metrics, imu_manifest = _materialize_imu_with_rust(
            source=imu_file_uri,
            row_groups=imu_row_groups,
            timestamps_ns=timestamps,
            audit_ranges=_catalog_audit_ranges(seek_plan._imu_rows),
            enforce_ranges=enforce_ranges,
            footer_allowance_bytes=footer_allowance_bytes,
            robotics_bin=robotics_bin,
        )
    else:
        imu_values = np.empty((pose_values.shape[0], 0), dtype=np.float64)
        imu_null_count = 0
        imu_gap_count = 0
        imu_max_gap_ns = 0
        imu_metrics = {}
        imu_manifest = None
    combined_values = np.concatenate([pose_values, imu_values], axis=1)
    values = combined_values[:, channel_indices]
    pose_gap_summary = _catalog_gap_summary(seek_plan._pose_rows, seek_plan.start_ts_ns, seek_plan.end_ts_ns)
    imu_gap_summary = _catalog_gap_summary(seek_plan._imu_rows, seek_plan.start_ts_ns, seek_plan.end_ts_ns)
    pose_null_count = int(pose_metrics.get("pose_null_count", 0))
    pose_gap_count = pose_gap_summary["gap_count"] or int(pose_metrics.get("pose_gap_count", 0))
    pose_max_gap_ns = pose_gap_summary["max_gap_ns"] or int(pose_metrics.get("pose_max_gap_ns", 0))
    quaternion_inversions = int(pose_metrics.get("quaternion_inversions_applied", 0))
    pose_planned_range_reads = int(pose_metrics.get("planned_range_reads", 0))
    imu_planned_range_reads = int(imu_metrics.get("planned_range_reads", 0))
    planned_read_bytes = int(pose_metrics.get("planned_read_bytes", 0)) + int(
        imu_metrics.get("planned_read_bytes", 0)
    )
    range_audit_passed = bool(int(pose_metrics.get("range_audit_passed", 0))) and (
        not needs_imu or bool(int(imu_metrics.get("range_audit_passed", 0)))
    )
    actual_cold_reads = int(pose_metrics.get("actual_cold_reads", 0)) + int(
        imu_metrics.get("actual_cold_reads", 0)
    )
    actual_cold_read_bytes = int(pose_metrics.get("actual_cold_read_bytes", 0)) + int(
        imu_metrics.get("actual_cold_read_bytes", 0)
    )
    footer_bytes = int(pose_metrics.get("footer_bytes", 0)) + int(imu_metrics.get("footer_bytes", 0))
    actual_authorized_bytes = int(pose_metrics.get("actual_authorized_bytes", 0)) + int(
        imu_metrics.get("actual_authorized_bytes", 0)
    )
    largest_metadata_read = max(
        int(pose_metrics.get("largest_metadata_read", 0)),
        int(imu_metrics.get("largest_metadata_read", 0)),
    )
    max_footer_read_offset = max(
        int(pose_metrics.get("max_footer_read_offset", 0)),
        int(imu_metrics.get("max_footer_read_offset", 0)),
    )
    max_footer_read_end = max(
        int(pose_metrics.get("max_footer_read_end", 0)),
        int(imu_metrics.get("max_footer_read_end", 0)),
    )
    range_violations = int(pose_metrics.get("range_violations", 0)) + int(
        imu_metrics.get("range_violations", 0)
    )
    diagnostics = CorrectnessReport(
        pose_gap_count=pose_gap_count,
        imu_gap_count=imu_gap_summary["gap_count"] or imu_gap_count,
        pose_max_gap_ns=pose_max_gap_ns,
        imu_max_gap_ns=imu_gap_summary["max_gap_ns"] or imu_max_gap_ns,
        null_count=pose_null_count + imu_null_count,
        quaternion_inversions_applied=quaternion_inversions,
        extrapolation_checked=True,
        extrapolation_rejected=False,
        velocity_threshold=seek_plan.diagnostics.velocity_threshold,
        matched_row_groups=len(seek_plan.row_groups),
        selected_bytes=seek_plan.authorized_total_bytes,
        pose_matched_row_groups=len(seek_plan.row_groups),
        pose_selected_bytes=seek_plan.authorized_pose_bytes,
        imu_matched_row_groups=len(seek_plan.imu_row_groups),
        imu_selected_bytes=seek_plan.authorized_imu_bytes,
        total_selected_bytes=seek_plan.authorized_total_bytes,
        pose_max_gap_start_ts_ns=pose_gap_summary["max_gap_start_ts_ns"],
        pose_max_gap_end_ts_ns=pose_gap_summary["max_gap_end_ts_ns"],
        imu_max_gap_start_ts_ns=imu_gap_summary["max_gap_start_ts_ns"],
        imu_max_gap_end_ts_ns=imu_gap_summary["max_gap_end_ts_ns"],
        pose_planned_range_reads=pose_planned_range_reads,
        imu_planned_range_reads=imu_planned_range_reads,
        planned_read_bytes=planned_read_bytes,
        range_audit_passed=range_audit_passed,
        catalog_query_ms=seek_plan.diagnostics.catalog_query_ms,
        candidate_row_groups=seek_plan.diagnostics.candidate_row_groups,
        time_pruned_row_groups=seek_plan.diagnostics.time_pruned_row_groups,
        spatial_pruned_row_groups=seek_plan.diagnostics.spatial_pruned_row_groups,
        velocity_pruned_row_groups=seek_plan.diagnostics.velocity_pruned_row_groups,
        gap_rejected_row_groups=pose_gap_summary["gap_row_groups"] + imu_gap_summary["gap_row_groups"],
        media_matched_row_groups=len(seek_plan.media_row_groups),
        media_selected_bytes=seek_plan.authorized_media_bytes,
        media_blocked_by_egress=False,
        authorized_pose_bytes=seek_plan.authorized_pose_bytes,
        authorized_imu_bytes=seek_plan.authorized_imu_bytes,
        authorized_media_bytes=seek_plan.authorized_media_bytes,
        authorized_total_bytes=seek_plan.authorized_total_bytes,
        materialized_pose_imu_bytes=seek_plan.materialized_pose_imu_bytes,
        planned_range_reads=pose_planned_range_reads + imu_planned_range_reads,
        blocked_by_egress=False,
        actual_cold_reads=actual_cold_reads,
        actual_cold_read_bytes=actual_cold_read_bytes,
        actual_authorized_bytes=actual_authorized_bytes,
        footer_allowance_bytes=footer_allowance_bytes,
        footer_bytes=footer_bytes,
        largest_metadata_read=largest_metadata_read,
        max_footer_read_offset=max_footer_read_offset,
        max_footer_read_end=max_footer_read_end,
        range_enforced=enforce_ranges,
        range_violations=range_violations,
    )
    tensor: object
    if output == "torch":
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("output='torch' requires the torch package") from exc
        tensor = torch.from_numpy(values)
    else:
        tensor = values

    manifest = _query_manifest(
        seek_plan=seek_plan,
        diagnostics=diagnostics,
        pose_manifest=pose_manifest,
        imu_manifest=imu_manifest,
        predicate=predicate,
        channels=channels,
        target_hz=target_hz,
        enforce_ranges=enforce_ranges,
        footer_allowance_bytes=footer_allowance_bytes,
    )
    if manifest_out is not None:
        manifest_path = Path(manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return QueryResult(
        tensor=tensor,
        timestamps_ns=timestamps,
        row_groups=seek_plan.row_groups,
        file_uri=seek_plan.pose_file_uri,
        selected_bytes=seek_plan.authorized_total_bytes,
        output=output,
        diagnostics=diagnostics,
        manifest=manifest,
    )


def plan(
    *,
    catalog: str | os.PathLike[str] | None = None,
    catalog_db: str | os.PathLike[str] | None = None,
    robot_id: str | None = None,
    start_ts_ns: int | None = None,
    end_ts_ns: int | None = None,
    bbox: Sequence[float] | None = None,
    min_velocity: float | None = None,
    predicate: str | None = None,
    channels: Sequence[str] = ("pos_xyz", "rot_wxyz", "vel_xyz"),
    source: str | os.PathLike[str] | None = None,
    imu_source: str | os.PathLike[str] | None = None,
    imu_catalog: str | os.PathLike[str] | None = None,
    media_catalog: str | os.PathLike[str] | None = None,
    max_egress_bytes: int = 1_000_000_000,
    limit: int | None = None,
    gap_policy: Literal["reject", "allow"] = "reject",
) -> SeekPlan:
    """Plan a behavioral cold seek without materializing source Parquet bytes."""

    if gap_policy not in {"reject", "allow"}:
        raise ValueError("gap_policy must be 'reject' or 'allow'")
    if (catalog is None) == (catalog_db is None):
        raise ValueError("pass exactly one of catalog=... or catalog_db=...")
    scalar_channels, media_channels = _split_channels(channels)
    needs_imu = any(channel.startswith("imu_") for channel in scalar_channels)
    needs_media = bool(media_channels)
    if needs_imu and imu_catalog is None and catalog_db is None:
        raise ValueError(
            "IMU channels require imu_catalog=... or catalog_db=... for selected row-group materialization"
        )
    if needs_media and media_catalog is None and catalog_db is None:
        raise ValueError("media channels require media_catalog=... or catalog_db=... for egress planning")

    predicate_filters = _parse_predicate(predicate)
    rows, catalog_explain = _query_pose_catalog(
        catalog=Path(catalog) if catalog is not None else None,
        catalog_db=Path(catalog_db) if catalog_db is not None else None,
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        bbox=bbox,
        min_velocity=min_velocity,
        predicate_filters=predicate_filters,
        limit=limit,
    )
    if not rows:
        raise LookupError("query matched no catalog row groups")

    file_uris = {str(row["file_uri"]) for row in rows}
    if len(file_uris) != 1 and source is None:
        raise ValueError("query matched multiple file_uri values; pass source=... for this prototype")
    pose_file_uri = str(source) if source is not None else next(iter(file_uris))

    query_start_ns = (
        int(start_ts_ns) if start_ts_ns is not None else min(int(row["start_ts_ns"]) for row in rows)
    )
    query_end_ns = (
        int(end_ts_ns) if end_ts_ns is not None else max(int(row["end_ts_ns"]) for row in rows)
    )
    pose_selected_bytes = sum(int(row["byte_length"]) for row in rows)

    imu_rows: list[dict[str, object]] = []
    if imu_catalog is not None or catalog_db is not None:
        imu_rows = _query_imu_catalog(
            catalog=Path(imu_catalog) if imu_catalog is not None else None,
            catalog_db=Path(catalog_db) if catalog_db is not None else None,
            robot_id=robot_id,
            start_ts_ns=query_start_ns,
            end_ts_ns=query_end_ns,
        )
        if needs_imu and not imu_rows:
            raise LookupError("query matched no IMU catalog row groups")
    imu_selected_bytes = sum(int(row["byte_length"]) for row in imu_rows)

    media_rows: list[dict[str, object]] = []
    if needs_media:
        media_rows = _query_media_catalog(
            catalog=Path(media_catalog) if media_catalog is not None else None,
            catalog_db=Path(catalog_db) if catalog_db is not None else None,
            robot_id=robot_id,
            start_ts_ns=query_start_ns,
            end_ts_ns=query_end_ns,
            media_channels=media_channels,
        )
        if not media_rows:
            raise LookupError("query matched no media catalog row groups")
    media_selected_bytes = sum(int(row["byte_length"]) for row in media_rows)
    total_selected_bytes = pose_selected_bytes + imu_selected_bytes + media_selected_bytes
    materialized_pose_imu_bytes = pose_selected_bytes + imu_selected_bytes
    blocked_by_egress = total_selected_bytes > max_egress_bytes

    pose_gap_summary = _catalog_gap_summary(rows, query_start_ns, query_end_ns)
    imu_gap_summary = _catalog_gap_summary(imu_rows, query_start_ns, query_end_ns)
    if gap_policy == "reject":
        _raise_if_temporal_gap(pose_gap_summary, "pose")
        _raise_if_temporal_gap(imu_gap_summary, "IMU")

    pose_row_groups = _row_group_spans("pose", rows, file_uri_override=pose_file_uri)
    imu_row_groups = _row_group_spans(
        "imu",
        imu_rows,
        file_uri_override=str(imu_source) if imu_source is not None else None,
    )
    media_row_groups = _row_group_spans("media", media_rows)
    diagnostics = CorrectnessReport(
        pose_gap_count=pose_gap_summary["gap_count"],
        imu_gap_count=imu_gap_summary["gap_count"],
        pose_max_gap_ns=pose_gap_summary["max_gap_ns"],
        imu_max_gap_ns=imu_gap_summary["max_gap_ns"],
        null_count=0,
        quaternion_inversions_applied=0,
        extrapolation_checked=False,
        extrapolation_rejected=False,
        velocity_threshold=min_velocity if min_velocity is not None else predicate_filters.velocity_threshold,
        matched_row_groups=len(pose_row_groups),
        selected_bytes=total_selected_bytes,
        pose_matched_row_groups=len(pose_row_groups),
        pose_selected_bytes=pose_selected_bytes,
        imu_matched_row_groups=len(imu_row_groups),
        imu_selected_bytes=imu_selected_bytes,
        total_selected_bytes=total_selected_bytes,
        pose_max_gap_start_ts_ns=pose_gap_summary["max_gap_start_ts_ns"],
        pose_max_gap_end_ts_ns=pose_gap_summary["max_gap_end_ts_ns"],
        imu_max_gap_start_ts_ns=imu_gap_summary["max_gap_start_ts_ns"],
        imu_max_gap_end_ts_ns=imu_gap_summary["max_gap_end_ts_ns"],
        pose_planned_range_reads=len(pose_row_groups),
        imu_planned_range_reads=len(imu_row_groups),
        planned_read_bytes=materialized_pose_imu_bytes,
        range_audit_passed=False,
        catalog_query_ms=catalog_explain.catalog_query_ms,
        candidate_row_groups=catalog_explain.candidate_row_groups,
        time_pruned_row_groups=catalog_explain.time_pruned_row_groups,
        spatial_pruned_row_groups=catalog_explain.spatial_pruned_row_groups,
        velocity_pruned_row_groups=catalog_explain.velocity_pruned_row_groups,
        gap_rejected_row_groups=pose_gap_summary["gap_row_groups"] + imu_gap_summary["gap_row_groups"],
        media_matched_row_groups=len(media_row_groups),
        media_selected_bytes=media_selected_bytes,
        media_blocked_by_egress=blocked_by_egress and media_selected_bytes > 0,
        authorized_pose_bytes=pose_selected_bytes,
        authorized_imu_bytes=imu_selected_bytes,
        authorized_media_bytes=media_selected_bytes,
        authorized_total_bytes=total_selected_bytes,
        materialized_pose_imu_bytes=materialized_pose_imu_bytes,
        planned_range_reads=len(pose_row_groups) + len(imu_row_groups),
        blocked_by_egress=blocked_by_egress,
    )
    return SeekPlan(
        start_ts_ns=query_start_ns,
        end_ts_ns=query_end_ns,
        pose_file_uri=pose_file_uri,
        pose_row_groups=pose_row_groups,
        imu_row_groups=imu_row_groups,
        media_row_groups=media_row_groups,
        authorized_pose_bytes=pose_selected_bytes,
        authorized_imu_bytes=imu_selected_bytes,
        authorized_media_bytes=media_selected_bytes,
        authorized_total_bytes=total_selected_bytes,
        materialized_pose_imu_bytes=materialized_pose_imu_bytes,
        planned_range_reads=len(pose_row_groups) + len(imu_row_groups),
        blocked_by_egress=blocked_by_egress,
        egress_limit_bytes=max_egress_bytes,
        diagnostics=diagnostics,
        _pose_rows=tuple(rows),
        _imu_rows=tuple(imu_rows),
        _media_rows=tuple(media_rows),
    )


def _query_pose_catalog(
    *,
    catalog: Path | None,
    catalog_db: Path | None,
    robot_id: str | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    bbox: Sequence[float] | None,
    min_velocity: float | None,
    predicate_filters: _PredicateFilters,
    limit: int | None,
) -> tuple[list[dict[str, object]], _CatalogExplain]:
    if catalog_db is not None:
        return _duckdb_query_catalog_db(
            catalog_db,
            robot_id=robot_id,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            bbox=bbox,
            min_velocity=min_velocity,
            predicate_filters=predicate_filters,
            limit=limit,
        )
    if catalog is None:
        raise ValueError("catalog is required when catalog_db is not provided")
    started = time.perf_counter()
    rows = _duckdb_query_catalog(
        catalog,
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        bbox=bbox,
        min_velocity=min_velocity,
        predicate_filters=predicate_filters,
        limit=limit,
    )
    return rows, _CatalogExplain(
        catalog_query_ms=(time.perf_counter() - started) * 1000.0,
        candidate_row_groups=len(rows),
    )


def _query_imu_catalog(
    *,
    catalog: Path | None,
    catalog_db: Path | None,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
) -> list[dict[str, object]]:
    if catalog_db is not None:
        return _duckdb_query_imu_catalog_db(
            catalog_db,
            robot_id=robot_id,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
        )
    if catalog is None:
        return []
    return _duckdb_query_imu_catalog(
        catalog,
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
    )


def _query_media_catalog(
    *,
    catalog: Path | None,
    catalog_db: Path | None,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
    media_channels: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    if catalog_db is not None:
        return _duckdb_query_media_catalog_db(
            catalog_db,
            robot_id=robot_id,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            media_channels=media_channels,
        )
    if catalog is None:
        return []
    return _duckdb_query_media_catalog(
        catalog,
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        media_channels=media_channels,
    )


def _duckdb_query_catalog(
    catalog: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    bbox: Sequence[float] | None,
    min_velocity: float | None,
    predicate_filters: _PredicateFilters,
    limit: int | None,
) -> list[dict[str, object]]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates = []
    params: list[object] = [str(catalog)]
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    if start_ts_ns is not None:
        predicates.append("end_ts_ns >= ?")
        params.append(int(start_ts_ns))
    if end_ts_ns is not None:
        predicates.append("start_ts_ns <= ?")
        params.append(int(end_ts_ns))
    if bbox is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox(bbox)
        predicates.extend(
            [
                "min_x <= ?",
                "max_x >= ?",
                "min_y <= ?",
                "max_y >= ?",
                "min_z <= ?",
                "max_z >= ?",
            ]
        )
        params.extend([max_x, min_x, max_y, min_y, max_z, min_z])
    if min_velocity is not None:
        predicates.append("max_velocity >= ?")
        params.append(float(min_velocity))
    predicates.extend(predicate_filters.sql)
    params.extend(predicate_filters.params)

    where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))

    sql = f"""
        SELECT
            robot_id, session_id, file_uri, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length, max_velocity,
            gap_count, max_gap_ns, max_gap_start_ts_ns, max_gap_end_ts_ns, nominal_dt_ns
        FROM read_parquet(?)
        {where}
        ORDER BY start_ts_ns, row_group_id
        {limit_sql}
    """
    with duckdb.connect(":memory:") as con:
        columns = [column[0] for column in con.execute(sql, params).description]
        return [dict(zip(columns, row)) for row in con.fetchall()]


def _duckdb_query_catalog_db(
    catalog_db: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    bbox: Sequence[float] | None,
    min_velocity: float | None,
    predicate_filters: _PredicateFilters,
    limit: int | None,
) -> tuple[list[dict[str, object]], _CatalogExplain]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates = []
    params: list[object] = []
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    if start_ts_ns is not None:
        predicates.append("end_ts_ns >= ?")
        params.append(int(start_ts_ns))
    if end_ts_ns is not None:
        predicates.append("start_ts_ns <= ?")
        params.append(int(end_ts_ns))
    if bbox is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = _bbox(bbox)
        predicates.extend(_tile_overlap_predicates())
        params.extend(_tile_overlap_params((min_x, max_x, min_y, max_y, min_z, max_z)))
        predicates.extend(
            [
                "min_x <= ?",
                "max_x >= ?",
                "min_y <= ?",
                "max_y >= ?",
                "min_z <= ?",
                "max_z >= ?",
            ]
        )
        params.extend([max_x, min_x, max_y, min_y, max_z, min_z])
    if min_velocity is not None:
        predicates.append("max_velocity >= ?")
        params.append(float(min_velocity))
    for predicate_bbox in predicate_filters.bboxes:
        predicates.extend(_tile_overlap_predicates())
        params.extend(_tile_overlap_params(predicate_bbox))
    predicates.extend(predicate_filters.sql)
    params.extend(predicate_filters.params)

    where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))

    sql = f"""
        SELECT
            robot_id, session_id, file_uri, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length, max_velocity,
            gap_count, max_gap_ns, max_gap_start_ts_ns, max_gap_end_ts_ns, nominal_dt_ns
        FROM pose_row_groups
        {where}
        ORDER BY start_ts_ns, row_group_id
        {limit_sql}
    """
    started = time.perf_counter()
    with duckdb.connect(str(catalog_db), read_only=True) as con:
        explain = _duckdb_catalog_explain(
            con,
            robot_id=robot_id,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            bbox=bbox,
            min_velocity=min_velocity,
            predicate_filters=predicate_filters,
        )
        columns = [column[0] for column in con.execute(sql, params).description]
        rows = [dict(zip(columns, row)) for row in con.fetchall()]
    explain = _CatalogExplain(
        catalog_query_ms=(time.perf_counter() - started) * 1000.0,
        candidate_row_groups=explain.candidate_row_groups,
        time_pruned_row_groups=explain.time_pruned_row_groups,
        spatial_pruned_row_groups=explain.spatial_pruned_row_groups,
        velocity_pruned_row_groups=explain.velocity_pruned_row_groups,
    )
    return rows, explain


def _duckdb_query_imu_catalog(
    catalog: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
) -> list[dict[str, object]]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates = ["end_ts_ns >= ?", "start_ts_ns <= ?"]
    params: list[object] = [str(catalog), int(start_ts_ns), int(end_ts_ns)]
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    where = f"WHERE {' AND '.join(predicates)}"
    sql = f"""
        SELECT
            robot_id, session_id, file_uri, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length,
            gap_count, max_gap_ns, max_gap_start_ts_ns, max_gap_end_ts_ns, nominal_dt_ns
        FROM read_parquet(?)
        {where}
        ORDER BY start_ts_ns, row_group_id
    """
    with duckdb.connect(":memory:") as con:
        columns = [column[0] for column in con.execute(sql, params).description]
        return [dict(zip(columns, row)) for row in con.fetchall()]


def _duckdb_query_imu_catalog_db(
    catalog_db: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
) -> list[dict[str, object]]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates = ["end_ts_ns >= ?", "start_ts_ns <= ?"]
    params: list[object] = [int(start_ts_ns), int(end_ts_ns)]
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    where = f"WHERE {' AND '.join(predicates)}"
    sql = f"""
        SELECT
            robot_id, session_id, file_uri, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length,
            gap_count, max_gap_ns, max_gap_start_ts_ns, max_gap_end_ts_ns, nominal_dt_ns
        FROM imu_row_groups
        {where}
        ORDER BY start_ts_ns, row_group_id
    """
    with duckdb.connect(str(catalog_db), read_only=True) as con:
        columns = [column[0] for column in con.execute(sql, params).description]
        return [dict(zip(columns, row)) for row in con.fetchall()]


def _duckdb_query_media_catalog(
    catalog: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
    media_channels: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates, params = _media_predicates(
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        media_channels=media_channels,
    )
    params = [str(catalog), *params]
    sql = f"""
        SELECT
            robot_id, session_id, file_uri, modality, stream_id, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length, row_count
        FROM read_parquet(?)
        WHERE {' AND '.join(predicates)}
        ORDER BY modality, stream_id, start_ts_ns, row_group_id
    """
    with duckdb.connect(":memory:") as con:
        columns = [column[0] for column in con.execute(sql, params).description]
        return [dict(zip(columns, row)) for row in con.fetchall()]


def _duckdb_query_media_catalog_db(
    catalog_db: Path,
    *,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
    media_channels: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    predicates, params = _media_predicates(
        robot_id=robot_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        media_channels=media_channels,
    )
    sql = f"""
        SELECT
            robot_id, session_id, file_uri, modality, stream_id, row_group_id,
            start_ts_ns, end_ts_ns, byte_offset, byte_length, row_count
        FROM media_row_groups
        WHERE {' AND '.join(predicates)}
        ORDER BY modality, stream_id, start_ts_ns, row_group_id
    """
    with duckdb.connect(str(catalog_db), read_only=True) as con:
        columns = [column[0] for column in con.execute(sql, params).description]
        return [dict(zip(columns, row)) for row in con.fetchall()]


def _media_predicates(
    *,
    robot_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
    media_channels: Sequence[tuple[str, str]],
) -> tuple[list[str], list[object]]:
    predicates = ["end_ts_ns >= ?", "start_ts_ns <= ?"]
    params: list[object] = [int(start_ts_ns), int(end_ts_ns)]
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    channel_predicates = []
    for modality, stream_id in media_channels:
        channel_predicates.append("(modality = ? AND stream_id = ?)")
        params.extend([modality, stream_id])
    predicates.append(f"({' OR '.join(channel_predicates)})")
    return predicates, params


def _duckdb_catalog_explain(
    con: object,
    *,
    robot_id: str | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    bbox: Sequence[float] | None,
    min_velocity: float | None,
    predicate_filters: _PredicateFilters,
) -> _CatalogExplain:
    candidate_predicates, candidate_params = _robot_predicates(robot_id)
    candidate = _count_pose_row_groups(con, candidate_predicates, candidate_params)

    time_predicates = list(candidate_predicates)
    time_params = list(candidate_params)
    if start_ts_ns is not None:
        time_predicates.append("end_ts_ns >= ?")
        time_params.append(int(start_ts_ns))
    if end_ts_ns is not None:
        time_predicates.append("start_ts_ns <= ?")
        time_params.append(int(end_ts_ns))
    for time_start_ns, time_end_ns in predicate_filters.time_windows:
        time_predicates.extend(["end_ts_ns >= ?", "start_ts_ns <= ?"])
        time_params.extend([time_start_ns, time_end_ns])
    after_time = _count_pose_row_groups(con, time_predicates, time_params)

    spatial_predicates = list(time_predicates)
    spatial_params = list(time_params)
    bboxes = []
    if bbox is not None:
        bboxes.append(_bbox(bbox))
    bboxes.extend(predicate_filters.bboxes)
    for spatial_bbox in bboxes:
        spatial_predicates.extend(_tile_overlap_predicates())
        spatial_params.extend(_tile_overlap_params(spatial_bbox))
        spatial_predicates.extend(_exact_bbox_predicates())
        spatial_params.extend(_exact_bbox_params(spatial_bbox))
    after_spatial = _count_pose_row_groups(con, spatial_predicates, spatial_params)

    velocity_predicates = list(spatial_predicates)
    velocity_params = list(spatial_params)
    if min_velocity is not None:
        velocity_predicates.append("max_velocity >= ?")
        velocity_params.append(float(min_velocity))
    for op, value in predicate_filters.velocity_conditions:
        velocity_predicates.append(f"max_velocity {op} ?")
        velocity_params.append(value)
    after_velocity = _count_pose_row_groups(con, velocity_predicates, velocity_params)

    return _CatalogExplain(
        candidate_row_groups=candidate,
        time_pruned_row_groups=max(candidate - after_time, 0),
        spatial_pruned_row_groups=max(after_time - after_spatial, 0),
        velocity_pruned_row_groups=max(after_spatial - after_velocity, 0),
    )


def _robot_predicates(robot_id: str | None) -> tuple[list[str], list[object]]:
    if robot_id is None:
        return [], []
    return ["robot_id = ?"], [robot_id]


def _count_pose_row_groups(con: object, predicates: Sequence[str], params: Sequence[object]) -> int:
    where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    return int(con.execute(f"SELECT count(*) FROM pose_row_groups {where}", list(params)).fetchone()[0])


def _tile_overlap_predicates() -> list[str]:
    return ["tile_min_x <= ?", "tile_max_x >= ?", "tile_min_y <= ?", "tile_max_y >= ?"]


def _tile_overlap_params(
    bbox: tuple[float, float, float, float, float, float],
) -> list[int]:
    min_x, max_x, min_y, max_y, _min_z, _max_z = bbox
    return [
        math.floor(max_x),
        math.floor(min_x),
        math.floor(max_y),
        math.floor(min_y),
    ]


def _exact_bbox_predicates() -> list[str]:
    return [
        "min_x <= ?",
        "max_x >= ?",
        "min_y <= ?",
        "max_y >= ?",
        "min_z <= ?",
        "max_z >= ?",
    ]


def _exact_bbox_params(
    bbox: tuple[float, float, float, float, float, float],
) -> list[float]:
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    return [max_x, min_x, max_y, min_y, max_z, min_z]


def _parse_predicate(predicate: str | None) -> _PredicateFilters:
    if predicate is None or not predicate.strip():
        return _PredicateFilters()

    sql: list[str] = []
    params: list[object] = []
    velocity_threshold: float | None = None
    velocity_conditions: list[tuple[str, float]] = []
    bboxes: list[tuple[float, float, float, float, float, float]] = []
    time_windows: list[tuple[int, int]] = []
    parts = re.split(r"\s+AND\s+", predicate.strip(), flags=re.IGNORECASE)
    for raw_part in parts:
        part = raw_part.strip()
        velocity = re.fullmatch(
            r"velocity_magnitude\s*(>=|>)\s*(-?\d+(?:\.\d+)?)",
            part,
            flags=re.IGNORECASE,
        )
        if velocity is not None:
            op = velocity.group(1)
            value = float(velocity.group(2))
            sql.append(f"max_velocity {op} ?")
            params.append(value)
            velocity_conditions.append((op, value))
            velocity_threshold = value if velocity_threshold is None else max(velocity_threshold, value)
            continue

        spatial = re.fullmatch(
            r"ST_Intersects\s*\(\s*position\s*,\s*bbox\s*\(\s*"
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*"
            r"\)\s*\)",
            part,
            flags=re.IGNORECASE,
        )
        if spatial is not None:
            spatial_bbox = _bbox([float(v) for v in spatial.groups()])
            sql.extend(_exact_bbox_predicates())
            params.extend(_exact_bbox_params(spatial_bbox))
            bboxes.append(spatial_bbox)
            continue

        time_overlap = re.fullmatch(
            r"time_overlap\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
            part,
            flags=re.IGNORECASE,
        )
        if time_overlap is not None:
            start_ns = int(time_overlap.group(1))
            end_ns = int(time_overlap.group(2))
            if start_ns > end_ns:
                raise ValueError("time_overlap start must be <= end")
            sql.extend(["end_ts_ns >= ?", "start_ts_ns <= ?"])
            params.extend([start_ns, end_ns])
            time_windows.append((start_ns, end_ns))
            continue

        raise ValueError(
            "unsupported predicate expression; supported forms are "
            "'velocity_magnitude > N', "
            "'ST_Intersects(position, bbox(min_x,max_x,min_y,max_y,min_z,max_z))', "
            "and 'time_overlap(start_ns,end_ns)' joined with AND"
        )

    return _PredicateFilters(
        sql=tuple(sql),
        params=tuple(params),
        velocity_threshold=velocity_threshold,
        velocity_conditions=tuple(velocity_conditions),
        bboxes=tuple(bboxes),
        time_windows=tuple(time_windows),
    )


def _single_file_uri(rows: Sequence[dict[str, object]], label: str) -> str:
    file_uris = {str(row["file_uri"]) for row in rows}
    if len(file_uris) != 1:
        raise ValueError(f"query matched multiple {label} file_uri values; pass an explicit source")
    return next(iter(file_uris))


def _row_group_spans(
    modality: str,
    rows: Sequence[dict[str, object]],
    *,
    file_uri_override: str | None = None,
) -> tuple[RowGroupSpan, ...]:
    spans = []
    for row in sorted(rows, key=lambda row: (str(row["file_uri"]), int(row["row_group_id"]))):
        spans.append(
            RowGroupSpan(
                modality=str(row.get("modality", modality)),
                file_uri=file_uri_override or str(row["file_uri"]),
                row_group_id=int(row["row_group_id"]),
                start_ts_ns=int(row["start_ts_ns"]),
                end_ts_ns=int(row["end_ts_ns"]),
                byte_offset=int(row["byte_offset"]),
                byte_length=int(row["byte_length"]),
                stream_id=str(row.get("stream_id", "")),
            )
        )
    return tuple(spans)


def _catalog_gap_summary(
    rows: Sequence[dict[str, object]],
    start_ts_ns: int,
    end_ts_ns: int,
) -> dict[str, int]:
    summary = {
        "gap_count": 0,
        "gap_row_groups": 0,
        "max_gap_ns": 0,
        "max_gap_start_ts_ns": 0,
        "max_gap_end_ts_ns": 0,
    }
    for row in rows:
        gap_count = int(row.get("gap_count", 0))
        if gap_count <= 0:
            continue
        gap_start = int(row.get("max_gap_start_ts_ns", 0))
        gap_end = int(row.get("max_gap_end_ts_ns", 0))
        if gap_start > end_ts_ns or gap_end < start_ts_ns:
            continue
        summary["gap_row_groups"] += 1
        summary["gap_count"] += gap_count
        gap_ns = int(row.get("max_gap_ns", 0))
        if gap_ns > summary["max_gap_ns"]:
            summary["max_gap_ns"] = gap_ns
            summary["max_gap_start_ts_ns"] = gap_start
            summary["max_gap_end_ts_ns"] = gap_end
    return summary


def _raise_if_temporal_gap(summary: dict[str, int], label: str) -> None:
    if summary["gap_count"] > 0:
        raise TemporalGapError(
            f"{label} query window crosses {summary['gap_count']} telemetry gap(s); "
            f"max_gap_ns={summary['max_gap_ns']} "
            f"[{summary['max_gap_start_ts_ns']}, {summary['max_gap_end_ts_ns']}]. "
            "Pass gap_policy='allow' to materialize anyway."
        )


def _materialize_with_rust(
    *,
    source: str,
    row_groups: Sequence[int],
    start_ts_ns: int,
    end_ts_ns: int,
    target_hz: float,
    audit_ranges: str,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
    robotics_bin: str | os.PathLike[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, object] | None]:
    with tempfile.TemporaryDirectory(prefix="physicaldb_") as tmp:
        prefix = Path(tmp) / "tensor"
        manifest_path = Path(tmp) / "pose.manifest.json"
        cmd = _robotics_command(robotics_bin) + [
            "tensor",
            "parquet-row-groups",
            "--input",
            source,
            "--row-groups",
            ",".join(str(row_group) for row_group in row_groups),
            "--start-ts-ns",
            str(start_ts_ns),
            "--end-ts-ns",
            str(end_ts_ns),
            "--hz",
            str(target_hz),
            "--out",
            str(prefix),
            "--audit-ranges",
            audit_ranges,
        ]
        if enforce_ranges:
            cmd.extend(
                [
                    "--enforce-ranges",
                    "--footer-allowance-bytes",
                    str(footer_allowance_bytes),
                    "--manifest-out",
                    str(manifest_path),
                ]
            )
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        values = np.load(prefix.with_name(f"{prefix.name}.values.npy"))
        timestamps = np.load(prefix.with_name(f"{prefix.name}.timestamps_ns.npy"))
        metrics = _parse_cli_metrics(completed.stdout)
        manifest = _load_json_manifest(manifest_path) if enforce_ranges else None
    return values, timestamps, metrics, manifest


def _materialize_imu_with_rust(
    *,
    source: str,
    row_groups: Sequence[int],
    timestamps_ns: np.ndarray,
    audit_ranges: str,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
    robotics_bin: str | os.PathLike[str] | None,
) -> tuple[np.ndarray, int, int, int, dict[str, int], dict[str, object] | None]:
    if not row_groups:
        raise LookupError("query matched no IMU row groups")
    with tempfile.TemporaryDirectory(prefix="physicaldb_imu_") as tmp:
        tmp_path = Path(tmp)
        timestamps_path = tmp_path / "pose.timestamps_ns.npy"
        prefix = tmp_path / "imu"
        manifest_path = tmp_path / "imu.manifest.json"
        np.save(timestamps_path, timestamps_ns.astype(np.int64, copy=False))
        cmd = _robotics_command(robotics_bin) + [
            "tensor",
            "imu-parquet-row-groups",
            "--input",
            source,
            "--row-groups",
            ",".join(str(row_group) for row_group in row_groups),
            "--timestamps-npy",
            str(timestamps_path),
            "--out",
            str(prefix),
            "--audit-ranges",
            audit_ranges,
        ]
        if enforce_ranges:
            cmd.extend(
                [
                    "--enforce-ranges",
                    "--footer-allowance-bytes",
                    str(footer_allowance_bytes),
                    "--manifest-out",
                    str(manifest_path),
                ]
            )
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        values = np.load(prefix.with_name(f"{prefix.name}.values.npy"))
        metrics = _parse_cli_metrics(completed.stdout)
        manifest = _load_json_manifest(manifest_path) if enforce_ranges else None
    if values.shape != (timestamps_ns.shape[0], 6):
        raise RuntimeError(
            f"IMU materializer returned shape {values.shape}, expected {(timestamps_ns.shape[0], 6)}"
        )
    return (
        values,
        int(metrics.get("imu_null_count", 0)),
        int(metrics.get("imu_gap_count", 0)),
        int(metrics.get("imu_max_gap_ns", 0)),
        metrics,
        manifest,
    )


def _load_json_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_manifest(
    *,
    seek_plan: SeekPlan,
    diagnostics: CorrectnessReport,
    pose_manifest: dict[str, object] | None,
    imu_manifest: dict[str, object] | None,
    predicate: str | None,
    channels: Sequence[str],
    target_hz: float,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
) -> dict[str, object]:
    actual_reads: list[object] = []
    violations: list[object] = []
    footer_bytes = 0
    materialized_bytes = 0
    actual_authorized_bytes = 0
    largest_metadata_read = 0
    max_footer_read_offset = 0
    max_footer_read_end = 0
    for materialized_manifest in (pose_manifest, imu_manifest):
        if not materialized_manifest:
            continue
        actual_reads.extend(materialized_manifest.get("actual_reads", []))  # type: ignore[arg-type]
        violations.extend(materialized_manifest.get("violations", []))  # type: ignore[arg-type]
        footer_bytes += int(materialized_manifest.get("footer_bytes", 0))
        materialized_bytes += int(materialized_manifest.get("materialized_bytes", 0))
        actual_authorized_bytes += int(materialized_manifest.get("actual_authorized_bytes", 0))
        largest_metadata_read = max(
            largest_metadata_read,
            int(materialized_manifest.get("largest_metadata_read", 0)),
        )
        max_footer_read_offset = max(
            max_footer_read_offset,
            int(materialized_manifest.get("max_footer_read_offset", 0)),
        )
        max_footer_read_end = max(
            max_footer_read_end,
            int(materialized_manifest.get("max_footer_read_end", 0)),
        )

    manifest = seek_plan.to_manifest()
    manifest.update(
        {
            "plan_inputs": {
                "predicate": predicate,
                "channels": list(channels),
                "target_hz": target_hz,
                "enforce_ranges": enforce_ranges,
                "footer_allowance_bytes": footer_allowance_bytes,
            },
            "actual_reads": actual_reads,
            "actual_cold_reads": len(actual_reads),
            "actual_cold_read_bytes": sum(
                int(read.get("length", 0)) for read in actual_reads if isinstance(read, dict)
            ),
            "actual_authorized_bytes": actual_authorized_bytes,
            "authorized_row_group_bytes": seek_plan.materialized_pose_imu_bytes,
            "footer_allowance_bytes": footer_allowance_bytes,
            "footer_bytes": footer_bytes,
            "largest_metadata_read": largest_metadata_read,
            "max_footer_read_offset": max_footer_read_offset,
            "max_footer_read_end": max_footer_read_end,
            "materialized_bytes": materialized_bytes,
            "media_planned_bytes": seek_plan.authorized_media_bytes,
            "violations": violations,
            "enforcement_enabled": enforce_ranges,
            "diagnostics": asdict(diagnostics),
            "pose_materialization_manifest": pose_manifest,
            "imu_materialization_manifest": imu_manifest,
        }
    )
    return manifest


def _parse_cli_metrics(stdout: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        clean_value = value.strip().lower()
        if sep and clean_value in {"true", "false"}:
            metrics[key.strip()] = 1 if clean_value == "true" else 0
        elif sep and value.strip().lstrip("-").isdigit():
            metrics[key.strip()] = int(value.strip())
    return metrics


def _catalog_audit_ranges(rows: Sequence[dict[str, object]]) -> str:
    return ",".join(
        f"{int(row['row_group_id'])}:{int(row['byte_offset'])}:{int(row['byte_length'])}"
        for row in sorted(rows, key=lambda row: int(row["row_group_id"]))
    )


def _robotics_command(robotics_bin: str | os.PathLike[str] | None) -> list[str]:
    if robotics_bin is not None:
        return [str(robotics_bin)]
    root = Path(__file__).resolve().parents[2]
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    if target_dir is not None:
        binary = Path(target_dir) / "debug" / "robotics"
        if binary.exists():
            return [str(binary)]
    binary = root / "target" / "debug" / "robotics"
    if binary.exists():
        return [str(binary)]
    return ["cargo", "run", "-p", "robotics-cli", "--"]


def _split_channels(channels: Sequence[str]) -> tuple[list[str], list[tuple[str, str]]]:
    scalar_channels: list[str] = []
    media_channels: list[tuple[str, str]] = []
    for channel in channels:
        if channel in CHANNELS:
            scalar_channels.append(channel)
            continue
        if ":" in channel:
            modality, stream_id = channel.split(":", 1)
            if modality in {"camera", "lidar"} and stream_id:
                media_channels.append((modality, stream_id))
                continue
        expected = sorted(CHANNELS) + ["camera:<stream_id>", "lidar:<stream_id>"]
        raise ValueError(f"unknown channel {channel!r}; expected one of {expected}")
    return scalar_channels, media_channels


def _channel_indices(channels: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for channel in channels:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; expected one of {sorted(CHANNELS)}")
        indices.extend(CHANNELS[channel])
    return indices


def _bbox(values: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ValueError("bbox must contain min_x,max_x,min_y,max_y,min_z,max_z")
    return tuple(float(value) for value in values)  # type: ignore[return-value]
