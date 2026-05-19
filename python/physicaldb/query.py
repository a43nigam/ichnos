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
_DUCKDB_CATALOG_CONNECTIONS: dict[tuple[str, int], object] = {}


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
    hilbert_pruned_row_groups: int = 0
    exact_spatial_pruned_row_groups: int = 0
    velocity_pruned_row_groups: int = 0
    index_strategy: str = "none"
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
            f"hilbert_pruned_row_groups={self.hilbert_pruned_row_groups}",
            f"exact_spatial_pruned_row_groups={self.exact_spatial_pruned_row_groups}",
            f"velocity_pruned_row_groups={self.velocity_pruned_row_groups}",
            f"index_strategy={self.index_strategy}",
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
class BatchSeekPlan:
    windows: tuple[SeekPlan, ...]
    authorized_pose_bytes: int
    authorized_imu_bytes: int
    authorized_media_bytes: int
    authorized_total_bytes: int
    materialized_pose_imu_bytes: int
    planned_range_reads: int
    blocked_by_egress: bool
    egress_limit_bytes: int
    diagnostics: CorrectnessReport

    def to_manifest(self) -> dict[str, object]:
        return {
            "version": 1,
            "windows": [window.to_manifest() for window in self.windows],
            "authorized_pose_bytes": self.authorized_pose_bytes,
            "authorized_imu_bytes": self.authorized_imu_bytes,
            "authorized_media_bytes": self.authorized_media_bytes,
            "authorized_total_bytes": self.authorized_total_bytes,
            "materialized_pose_imu_bytes": self.materialized_pose_imu_bytes,
            "planned_range_reads": self.planned_range_reads,
            "blocked_by_egress": self.blocked_by_egress,
            "egress_limit_bytes": self.egress_limit_bytes,
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
    media_manifest: dict[str, object] | None = None


@dataclass(frozen=True)
class BatchQueryResult:
    windows: tuple[QueryResult, ...]
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
    hilbert_pruned_row_groups: int = 0
    exact_spatial_pruned_row_groups: int = 0
    velocity_pruned_row_groups: int = 0
    index_strategy: str = "none"


def query(
    *,
    catalog: str | os.PathLike[str] | None = None,
    catalog_db: str | os.PathLike[str] | None = None,
    robot_id: str | None = None,
    session_id: str | None = None,
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
    materialize_media: bool = False,
    media_out: str | os.PathLike[str] | None = None,
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
        session_id=session_id,
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
    return _query_from_seek_plan(
        seek_plan=seek_plan,
        channels=channels,
        target_hz=target_hz,
        output=output,
        materialize_media=materialize_media,
        media_out=Path(media_out) if media_out is not None else None,
        enforce_ranges=enforce_ranges,
        footer_allowance_bytes=footer_allowance_bytes,
        manifest_out=Path(manifest_out) if manifest_out is not None else None,
        predicate=predicate,
        robotics_bin=robotics_bin,
    )


def query_batch(
    *,
    catalog: str | os.PathLike[str] | None = None,
    catalog_db: str | os.PathLike[str] | None = None,
    robot_id: str | None = None,
    session_id: str | None = None,
    start_ts_ns: int | None = None,
    end_ts_ns: int | None = None,
    bbox: Sequence[float] | None = None,
    min_velocity: float | None = None,
    predicate: str | None = None,
    channels: Sequence[str] = ("pos_xyz", "rot_wxyz", "vel_xyz"),
    target_hz: float = 30.0,
    output: Literal["numpy", "torch"] = "numpy",
    imu_catalog: str | os.PathLike[str] | None = None,
    media_catalog: str | os.PathLike[str] | None = None,
    max_egress_bytes: int = 1_000_000_000,
    limit: int | None = None,
    gap_policy: Literal["reject", "allow"] = "reject",
    enforce_ranges: bool = False,
    footer_allowance_bytes: int = DEFAULT_FOOTER_ALLOWANCE_BYTES,
    manifest_out: str | os.PathLike[str] | None = None,
    materialize_media: bool = False,
    media_out: str | os.PathLike[str] | None = None,
    robotics_bin: str | os.PathLike[str] | None = None,
) -> BatchQueryResult:
    """Run a behavioral query as independent per-session/per-file windows."""

    if output not in {"numpy", "torch"}:
        raise ValueError("output must be 'numpy' or 'torch'")
    batch_plan = plan_batch(
        catalog=catalog,
        catalog_db=catalog_db,
        robot_id=robot_id,
        session_id=session_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        bbox=bbox,
        min_velocity=min_velocity,
        predicate=predicate,
        channels=channels,
        imu_catalog=imu_catalog,
        media_catalog=media_catalog,
        max_egress_bytes=max_egress_bytes,
        limit=limit,
        gap_policy=gap_policy,
    )
    if batch_plan.blocked_by_egress:
        raise EgressLimitError(
            "query selected "
            f"{batch_plan.authorized_total_bytes} bytes across {len(batch_plan.windows)} windows, "
            f"including {batch_plan.authorized_media_bytes} media bytes, "
            f"exceeding max_egress_bytes={batch_plan.egress_limit_bytes}"
        )
    if materialize_media and media_out is None:
        raise ValueError("materialize_media=True requires media_out=...")

    results: list[QueryResult] = []
    for index, window in enumerate(batch_plan.windows):
        window_media_out = None
        if materialize_media:
            assert media_out is not None
            session = _safe_path_component(_plan_session_id(window))
            window_media_out = Path(media_out) / f"window_{index:03}_{session}"
        results.append(
            _query_from_seek_plan(
                seek_plan=window,
                channels=channels,
                target_hz=target_hz,
                output=output,
                materialize_media=materialize_media,
                media_out=window_media_out,
                enforce_ranges=enforce_ranges,
                footer_allowance_bytes=footer_allowance_bytes,
                manifest_out=None,
                predicate=predicate,
                robotics_bin=robotics_bin,
            )
        )

    diagnostics = _aggregate_query_diagnostics(
        tuple(result.diagnostics for result in results),
        batch_plan.diagnostics,
        enforce_ranges=enforce_ranges,
        footer_allowance_bytes=footer_allowance_bytes,
    )
    manifest = _batch_query_manifest(
        batch_plan=batch_plan,
        results=tuple(results),
        diagnostics=diagnostics,
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

    return BatchQueryResult(
        windows=tuple(results),
        selected_bytes=batch_plan.authorized_total_bytes,
        output=output,
        diagnostics=diagnostics,
        manifest=manifest,
    )


def _query_from_seek_plan(
    *,
    seek_plan: SeekPlan,
    channels: Sequence[str],
    target_hz: float,
    output: Literal["numpy", "torch"],
    materialize_media: bool,
    media_out: Path | None,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
    manifest_out: Path | None,
    predicate: str | None,
    robotics_bin: str | os.PathLike[str] | None,
) -> QueryResult:
    scalar_channels, _media_channels = _split_channels(channels)
    needs_imu = any(channel.startswith("imu_") for channel in scalar_channels)
    channel_indices = _channel_indices(scalar_channels)
    if materialize_media and not seek_plan.media_row_groups:
        raise ValueError("materialize_media=True requires at least one media channel")
    if materialize_media and media_out is None:
        raise ValueError("materialize_media=True requires media_out=...")

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
        imu_file_uri = _single_file_uri(seek_plan._imu_rows, "IMU")
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
    media_manifest = None
    media_metrics: dict[str, int] = {}
    if materialize_media:
        assert media_out is not None
        media_metrics, media_manifest = _materialize_media_with_rust(
            row_groups=seek_plan.media_row_groups,
            media_rows=seek_plan._media_rows,
            output_dir=Path(media_out),
            enforce_ranges=enforce_ranges,
            footer_allowance_bytes=footer_allowance_bytes,
            robotics_bin=robotics_bin,
        )
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
    ) + int(media_metrics.get("actual_cold_reads", 0))
    actual_cold_read_bytes = int(pose_metrics.get("actual_cold_read_bytes", 0)) + int(
        imu_metrics.get("actual_cold_read_bytes", 0)
    ) + int(media_metrics.get("actual_cold_read_bytes", 0))
    footer_bytes = (
        int(pose_metrics.get("footer_bytes", 0))
        + int(imu_metrics.get("footer_bytes", 0))
        + int(media_metrics.get("footer_bytes", 0))
    )
    actual_authorized_bytes = int(pose_metrics.get("actual_authorized_bytes", 0)) + int(
        imu_metrics.get("actual_authorized_bytes", 0)
    ) + int(media_metrics.get("actual_authorized_bytes", 0))
    largest_metadata_read = max(
        int(pose_metrics.get("largest_metadata_read", 0)),
        int(imu_metrics.get("largest_metadata_read", 0)),
        int(media_metrics.get("largest_metadata_read", 0)),
    )
    max_footer_read_offset = max(
        int(pose_metrics.get("max_footer_read_offset", 0)),
        int(imu_metrics.get("max_footer_read_offset", 0)),
        int(media_metrics.get("max_footer_read_offset", 0)),
    )
    max_footer_read_end = max(
        int(pose_metrics.get("max_footer_read_end", 0)),
        int(imu_metrics.get("max_footer_read_end", 0)),
        int(media_metrics.get("max_footer_read_end", 0)),
    )
    range_violations = int(pose_metrics.get("range_violations", 0)) + int(
        imu_metrics.get("range_violations", 0)
    ) + int(media_metrics.get("range_violations", 0))
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
        hilbert_pruned_row_groups=seek_plan.diagnostics.hilbert_pruned_row_groups,
        exact_spatial_pruned_row_groups=seek_plan.diagnostics.exact_spatial_pruned_row_groups,
        velocity_pruned_row_groups=seek_plan.diagnostics.velocity_pruned_row_groups,
        index_strategy=seek_plan.diagnostics.index_strategy,
        gap_rejected_row_groups=pose_gap_summary["gap_row_groups"] + imu_gap_summary["gap_row_groups"],
        media_matched_row_groups=len(seek_plan.media_row_groups),
        media_selected_bytes=seek_plan.authorized_media_bytes,
        media_blocked_by_egress=False,
        authorized_pose_bytes=seek_plan.authorized_pose_bytes,
        authorized_imu_bytes=seek_plan.authorized_imu_bytes,
        authorized_media_bytes=seek_plan.authorized_media_bytes,
        authorized_total_bytes=seek_plan.authorized_total_bytes,
        materialized_pose_imu_bytes=seek_plan.materialized_pose_imu_bytes,
        planned_range_reads=pose_planned_range_reads
        + imu_planned_range_reads
        + int(media_metrics.get("planned_range_reads", 0)),
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
        media_manifest=media_manifest,
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
        media_manifest=media_manifest,
    )


def plan(
    *,
    catalog: str | os.PathLike[str] | None = None,
    catalog_db: str | os.PathLike[str] | None = None,
    robot_id: str | None = None,
    session_id: str | None = None,
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
        session_id=session_id,
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
        raise ValueError("query matched multiple file_uri values; use query_batch()/plan_batch()")
    pose_file_uri = str(source) if source is not None else next(iter(file_uris))
    row_session_id = _unique_row_value(rows, "session_id")
    related_session_id = session_id if session_id is not None else row_session_id

    query_start_ns, query_end_ns = _query_window_bounds(
        rows,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        predicate_filters=predicate_filters,
    )
    pose_selected_bytes = sum(int(row["byte_length"]) for row in rows)

    imu_rows: list[dict[str, object]] = []
    if imu_catalog is not None or catalog_db is not None:
        imu_rows = _query_imu_catalog(
            catalog=Path(imu_catalog) if imu_catalog is not None else None,
            catalog_db=Path(catalog_db) if catalog_db is not None else None,
            robot_id=robot_id,
            session_id=related_session_id,
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
            session_id=related_session_id,
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
        hilbert_pruned_row_groups=catalog_explain.hilbert_pruned_row_groups,
        exact_spatial_pruned_row_groups=catalog_explain.exact_spatial_pruned_row_groups,
        velocity_pruned_row_groups=catalog_explain.velocity_pruned_row_groups,
        index_strategy=catalog_explain.index_strategy,
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


def plan_batch(
    *,
    catalog: str | os.PathLike[str] | None = None,
    catalog_db: str | os.PathLike[str] | None = None,
    robot_id: str | None = None,
    session_id: str | None = None,
    start_ts_ns: int | None = None,
    end_ts_ns: int | None = None,
    bbox: Sequence[float] | None = None,
    min_velocity: float | None = None,
    predicate: str | None = None,
    channels: Sequence[str] = ("pos_xyz", "rot_wxyz", "vel_xyz"),
    imu_catalog: str | os.PathLike[str] | None = None,
    media_catalog: str | os.PathLike[str] | None = None,
    max_egress_bytes: int = 1_000_000_000,
    limit: int | None = None,
    gap_policy: Literal["reject", "allow"] = "reject",
) -> BatchSeekPlan:
    """Plan a behavioral cold seek as independent per-session/per-file windows."""

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
        session_id=session_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        bbox=bbox,
        min_velocity=min_velocity,
        predicate_filters=predicate_filters,
        limit=limit,
    )
    if not rows:
        raise LookupError("query matched no catalog row groups")

    windows: list[SeekPlan] = []
    for group_rows in _group_pose_rows_by_window(rows):
        group_robot_id = _unique_row_value(group_rows, "robot_id") or robot_id
        group_session_id = _unique_row_value(group_rows, "session_id") or session_id
        windows.append(
            _build_window_plan(
                rows=group_rows,
                catalog_explain=catalog_explain,
                predicate_filters=predicate_filters,
                robot_id=group_robot_id,
                session_id=group_session_id,
                min_velocity=min_velocity,
                start_ts_ns=start_ts_ns,
                end_ts_ns=end_ts_ns,
                imu_catalog=Path(imu_catalog) if imu_catalog is not None else None,
                media_catalog=Path(media_catalog) if media_catalog is not None else None,
                catalog_db=Path(catalog_db) if catalog_db is not None else None,
                media_channels=media_channels,
                needs_imu=needs_imu,
                needs_media=needs_media,
                max_egress_bytes=max_egress_bytes,
                gap_policy=gap_policy,
            )
        )

    authorized_pose_bytes = sum(window.authorized_pose_bytes for window in windows)
    authorized_imu_bytes = sum(window.authorized_imu_bytes for window in windows)
    authorized_media_bytes = sum(window.authorized_media_bytes for window in windows)
    authorized_total_bytes = authorized_pose_bytes + authorized_imu_bytes + authorized_media_bytes
    materialized_pose_imu_bytes = sum(window.materialized_pose_imu_bytes for window in windows)
    planned_range_reads = sum(window.planned_range_reads for window in windows)
    blocked_by_egress = authorized_total_bytes > max_egress_bytes
    diagnostics = _aggregate_plan_diagnostics(
        tuple(windows),
        catalog_explain,
        authorized_pose_bytes=authorized_pose_bytes,
        authorized_imu_bytes=authorized_imu_bytes,
        authorized_media_bytes=authorized_media_bytes,
        authorized_total_bytes=authorized_total_bytes,
        materialized_pose_imu_bytes=materialized_pose_imu_bytes,
        planned_range_reads=planned_range_reads,
        blocked_by_egress=blocked_by_egress,
        max_egress_bytes=max_egress_bytes,
    )
    return BatchSeekPlan(
        windows=tuple(windows),
        authorized_pose_bytes=authorized_pose_bytes,
        authorized_imu_bytes=authorized_imu_bytes,
        authorized_media_bytes=authorized_media_bytes,
        authorized_total_bytes=authorized_total_bytes,
        materialized_pose_imu_bytes=materialized_pose_imu_bytes,
        planned_range_reads=planned_range_reads,
        blocked_by_egress=blocked_by_egress,
        egress_limit_bytes=max_egress_bytes,
        diagnostics=diagnostics,
    )


def _build_window_plan(
    *,
    rows: Sequence[dict[str, object]],
    catalog_explain: _CatalogExplain,
    predicate_filters: _PredicateFilters,
    robot_id: str | None,
    session_id: str | None,
    min_velocity: float | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    imu_catalog: Path | None,
    media_catalog: Path | None,
    catalog_db: Path | None,
    media_channels: Sequence[tuple[str, str]],
    needs_imu: bool,
    needs_media: bool,
    max_egress_bytes: int,
    gap_policy: Literal["reject", "allow"],
) -> SeekPlan:
    pose_file_uri = _single_file_uri(rows, "pose")
    query_start_ns, query_end_ns = _query_window_bounds(
        rows,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        predicate_filters=predicate_filters,
    )
    pose_selected_bytes = sum(int(row["byte_length"]) for row in rows)

    imu_rows: list[dict[str, object]] = []
    if imu_catalog is not None or catalog_db is not None:
        imu_rows = _query_imu_catalog(
            catalog=imu_catalog,
            catalog_db=catalog_db,
            robot_id=robot_id,
            session_id=session_id,
            start_ts_ns=query_start_ns,
            end_ts_ns=query_end_ns,
        )
        if needs_imu and not imu_rows:
            raise LookupError("query matched no IMU catalog row groups")
    imu_selected_bytes = sum(int(row["byte_length"]) for row in imu_rows)

    media_rows: list[dict[str, object]] = []
    if needs_media:
        media_rows = _query_media_catalog(
            catalog=media_catalog,
            catalog_db=catalog_db,
            robot_id=robot_id,
            session_id=session_id,
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

    pose_row_groups = _row_group_spans("pose", rows)
    imu_row_groups = _row_group_spans("imu", imu_rows)
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
        hilbert_pruned_row_groups=catalog_explain.hilbert_pruned_row_groups,
        exact_spatial_pruned_row_groups=catalog_explain.exact_spatial_pruned_row_groups,
        velocity_pruned_row_groups=catalog_explain.velocity_pruned_row_groups,
        index_strategy=catalog_explain.index_strategy,
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
    session_id: str | None,
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
            session_id=session_id,
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
        session_id=session_id,
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
        index_strategy="parquet",
    )


def _query_imu_catalog(
    *,
    catalog: Path | None,
    catalog_db: Path | None,
    robot_id: str | None,
    session_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
) -> list[dict[str, object]]:
    if catalog_db is not None:
        return _duckdb_query_imu_catalog_db(
            catalog_db,
            robot_id=robot_id,
            session_id=session_id,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
        )
    if catalog is None:
        return []
    return _duckdb_query_imu_catalog(
        catalog,
        robot_id=robot_id,
        session_id=session_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
    )


def _query_media_catalog(
    *,
    catalog: Path | None,
    catalog_db: Path | None,
    robot_id: str | None,
    session_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
    media_channels: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    if catalog_db is not None:
        return _duckdb_query_media_catalog_db(
            catalog_db,
            robot_id=robot_id,
            session_id=session_id,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            media_channels=media_channels,
        )
    if catalog is None:
        return []
    return _duckdb_query_media_catalog(
        catalog,
        robot_id=robot_id,
        session_id=session_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        media_channels=media_channels,
    )


def _duckdb_query_catalog(
    catalog: Path,
    *,
    robot_id: str | None,
    session_id: str | None,
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
    if session_id is not None:
        predicates.append("session_id = ?")
        params.append(session_id)
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
    session_id: str | None,
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

    con = _duckdb_catalog_connection(catalog_db)
    columns = _duckdb_table_columns(con, "pose_row_groups")
    index_strategy = (
        "hilbert"
        if {"hilbert_xy", "hilbert_min_xy", "hilbert_max_xy", "time_bucket_ns"} <= columns
        else "tile"
    )

    candidate_predicates, candidate_params = _identity_predicates(robot_id, session_id)
    time_expr, time_params = _time_stage_expression(
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        predicate_filters=predicate_filters,
    )
    coarse_expr, coarse_params = _coarse_spatial_stage_expression(
        bbox=bbox,
        predicate_filters=predicate_filters,
        index_strategy=index_strategy,
    )
    exact_expr, exact_params = _exact_spatial_stage_expression(
        bbox=bbox,
        predicate_filters=predicate_filters,
    )
    velocity_expr, velocity_params = _velocity_stage_expression(
        min_velocity=min_velocity,
        predicate_filters=predicate_filters,
    )
    where = f"WHERE {' AND '.join(candidate_predicates)}" if candidate_predicates else ""
    limit_sql = ""
    params = [
        *candidate_params,
        *time_params,
        *coarse_params,
        *exact_params,
        *velocity_params,
    ]
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))

    sql = f"""
        WITH candidate AS (
            SELECT *
            FROM pose_row_groups
            {where}
        ),
        marked AS (
            SELECT
                *,
                ({time_expr}) AS pass_time,
                ({coarse_expr}) AS pass_coarse_spatial,
                ({exact_expr}) AS pass_exact_spatial,
                ({velocity_expr}) AS pass_velocity
            FROM candidate
        ),
        stats AS (
            SELECT
                count(*) AS candidate_row_groups,
                sum(CASE WHEN pass_time THEN 1 ELSE 0 END) AS after_time_row_groups,
                sum(CASE WHEN pass_time AND pass_coarse_spatial THEN 1 ELSE 0 END)
                    AS after_coarse_spatial_row_groups,
                sum(CASE WHEN pass_time AND pass_coarse_spatial AND pass_exact_spatial THEN 1 ELSE 0 END)
                    AS after_exact_spatial_row_groups,
                sum(CASE
                        WHEN pass_time AND pass_coarse_spatial AND pass_exact_spatial AND pass_velocity
                        THEN 1 ELSE 0
                    END) AS after_velocity_row_groups
            FROM marked
        ),
        selected AS (
            SELECT
                robot_id, session_id, file_uri, row_group_id,
                start_ts_ns, end_ts_ns, byte_offset, byte_length, max_velocity,
                gap_count, max_gap_ns, max_gap_start_ts_ns, max_gap_end_ts_ns, nominal_dt_ns
            FROM marked
            WHERE pass_time AND pass_coarse_spatial AND pass_exact_spatial AND pass_velocity
            ORDER BY start_ts_ns, row_group_id
            {limit_sql}
        )
        SELECT
            selected.*,
            stats.candidate_row_groups,
            stats.after_time_row_groups,
            stats.after_coarse_spatial_row_groups,
            stats.after_exact_spatial_row_groups,
            stats.after_velocity_row_groups
        FROM selected
        CROSS JOIN stats
    """
    started = time.perf_counter()
    result = con.execute(sql, params)
    result_columns = [column[0] for column in result.description]
    result_rows = [dict(zip(result_columns, row)) for row in result.fetchall()]
    catalog_query_ms = (time.perf_counter() - started) * 1000.0

    stat_columns = {
        "candidate_row_groups",
        "after_time_row_groups",
        "after_coarse_spatial_row_groups",
        "after_exact_spatial_row_groups",
        "after_velocity_row_groups",
    }
    rows = [{key: value for key, value in row.items() if key not in stat_columns} for row in result_rows]
    if result_rows:
        stats = result_rows[0]
        candidate = int(stats["candidate_row_groups"] or 0)
        after_time = int(stats["after_time_row_groups"] or 0)
        after_spatial = int(stats["after_coarse_spatial_row_groups"] or 0)
        after_exact_spatial = int(stats["after_exact_spatial_row_groups"] or 0)
        after_velocity = int(stats["after_velocity_row_groups"] or 0)
    else:
        candidate = after_time = after_spatial = after_exact_spatial = after_velocity = 0
    explain = _CatalogExplain(
        catalog_query_ms=catalog_query_ms,
        candidate_row_groups=candidate,
        time_pruned_row_groups=max(candidate - after_time, 0),
        spatial_pruned_row_groups=max(after_time - after_exact_spatial, 0),
        hilbert_pruned_row_groups=max(after_time - after_spatial, 0)
        if index_strategy == "hilbert"
        else 0,
        exact_spatial_pruned_row_groups=max(after_spatial - after_exact_spatial, 0),
        velocity_pruned_row_groups=max(after_exact_spatial - after_velocity, 0),
        index_strategy=index_strategy,
    )
    return rows, explain

def _duckdb_query_imu_catalog(
    catalog: Path,
    *,
    robot_id: str | None,
    session_id: str | None,
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
    if session_id is not None:
        predicates.append("session_id = ?")
        params.append(session_id)
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
    session_id: str | None,
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
    if session_id is not None:
        predicates.append("session_id = ?")
        params.append(session_id)
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
    session_id: str | None,
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
        session_id=session_id,
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
    session_id: str | None,
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
        session_id=session_id,
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
    session_id: str | None,
    start_ts_ns: int,
    end_ts_ns: int,
    media_channels: Sequence[tuple[str, str]],
) -> tuple[list[str], list[object]]:
    predicates = ["end_ts_ns >= ?", "start_ts_ns <= ?"]
    params: list[object] = [int(start_ts_ns), int(end_ts_ns)]
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    if session_id is not None:
        predicates.append("session_id = ?")
        params.append(session_id)
    channel_predicates = []
    for modality, stream_id in media_channels:
        channel_predicates.append("(modality = ? AND stream_id = ?)")
        params.extend([modality, stream_id])
    predicates.append(f"({' OR '.join(channel_predicates)})")
    return predicates, params


def _stage_sql(predicates: Sequence[str]) -> str:
    return " AND ".join(predicates) if predicates else "TRUE"


def _time_stage_expression(
    *,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    predicate_filters: _PredicateFilters,
) -> tuple[str, list[object]]:
    predicates: list[str] = []
    params: list[object] = []
    if start_ts_ns is not None:
        predicates.append("end_ts_ns >= ?")
        params.append(int(start_ts_ns))
    if end_ts_ns is not None:
        predicates.append("start_ts_ns <= ?")
        params.append(int(end_ts_ns))
    for time_start_ns, time_end_ns in predicate_filters.time_windows:
        predicates.extend(["end_ts_ns >= ?", "start_ts_ns <= ?"])
        params.extend([time_start_ns, time_end_ns])
    return _stage_sql(predicates), params


def _query_bboxes(
    bbox: Sequence[float] | None,
    predicate_filters: _PredicateFilters,
) -> list[tuple[float, float, float, float, float, float]]:
    bboxes = []
    if bbox is not None:
        bboxes.append(_bbox(bbox))
    bboxes.extend(predicate_filters.bboxes)
    return bboxes


def _coarse_spatial_stage_expression(
    *,
    bbox: Sequence[float] | None,
    predicate_filters: _PredicateFilters,
    index_strategy: str,
) -> tuple[str, list[object]]:
    predicates: list[str] = []
    params: list[object] = []
    for spatial_bbox in _query_bboxes(bbox, predicate_filters):
        predicates.extend(_spatial_coarse_predicates(spatial_bbox, index_strategy))
        params.extend(_spatial_coarse_params(spatial_bbox, index_strategy))
    return _stage_sql(predicates), params


def _exact_spatial_stage_expression(
    *,
    bbox: Sequence[float] | None,
    predicate_filters: _PredicateFilters,
) -> tuple[str, list[object]]:
    predicates: list[str] = []
    params: list[object] = []
    for spatial_bbox in _query_bboxes(bbox, predicate_filters):
        predicates.extend(_exact_bbox_predicates())
        params.extend(_exact_bbox_params(spatial_bbox))
    return _stage_sql(predicates), params


def _velocity_stage_expression(
    *,
    min_velocity: float | None,
    predicate_filters: _PredicateFilters,
) -> tuple[str, list[object]]:
    predicates: list[str] = []
    params: list[object] = []
    if min_velocity is not None:
        predicates.append("max_velocity >= ?")
        params.append(float(min_velocity))
    for op, value in predicate_filters.velocity_conditions:
        predicates.append(f"max_velocity {op} ?")
        params.append(value)
    return _stage_sql(predicates), params


def _duckdb_catalog_explain(
    con: object,
    *,
    robot_id: str | None,
    session_id: str | None,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    bbox: Sequence[float] | None,
    min_velocity: float | None,
    predicate_filters: _PredicateFilters,
    index_strategy: str,
) -> _CatalogExplain:
    candidate_predicates, candidate_params = _identity_predicates(robot_id, session_id)
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
    exact_spatial_predicates = list(time_predicates)
    exact_spatial_params = list(time_params)
    bboxes = []
    if bbox is not None:
        bboxes.append(_bbox(bbox))
    bboxes.extend(predicate_filters.bboxes)
    for spatial_bbox in bboxes:
        spatial_predicates.extend(_spatial_coarse_predicates(spatial_bbox, index_strategy))
        spatial_params.extend(_spatial_coarse_params(spatial_bbox, index_strategy))
        exact_spatial_predicates.extend(_spatial_coarse_predicates(spatial_bbox, index_strategy))
        exact_spatial_params.extend(_spatial_coarse_params(spatial_bbox, index_strategy))
        exact_spatial_predicates.extend(_exact_bbox_predicates())
        exact_spatial_params.extend(_exact_bbox_params(spatial_bbox))
    after_spatial = _count_pose_row_groups(con, spatial_predicates, spatial_params)
    after_exact_spatial = _count_pose_row_groups(con, exact_spatial_predicates, exact_spatial_params)

    velocity_predicates = list(exact_spatial_predicates)
    velocity_params = list(exact_spatial_params)
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
        spatial_pruned_row_groups=max(after_time - after_exact_spatial, 0),
        hilbert_pruned_row_groups=max(after_time - after_spatial, 0)
        if index_strategy == "hilbert"
        else 0,
        exact_spatial_pruned_row_groups=max(after_spatial - after_exact_spatial, 0),
        velocity_pruned_row_groups=max(after_exact_spatial - after_velocity, 0),
        index_strategy=index_strategy,
    )


def _robot_predicates(robot_id: str | None) -> tuple[list[str], list[object]]:
    if robot_id is None:
        return [], []
    return ["robot_id = ?"], [robot_id]


def _identity_predicates(
    robot_id: str | None,
    session_id: str | None,
) -> tuple[list[str], list[object]]:
    predicates: list[str] = []
    params: list[object] = []
    if robot_id is not None:
        predicates.append("robot_id = ?")
        params.append(robot_id)
    if session_id is not None:
        predicates.append("session_id = ?")
        params.append(session_id)
    return predicates, params


def _count_pose_row_groups(con: object, predicates: Sequence[str], params: Sequence[object]) -> int:
    where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    return int(con.execute(f"SELECT count(*) FROM pose_row_groups {where}", list(params)).fetchone()[0])


def _duckdb_catalog_connection(catalog_db: Path) -> object:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("physicaldb.query requires duckdb; install the project dependencies") from exc

    resolved = str(catalog_db.resolve())
    mtime_ns = catalog_db.stat().st_mtime_ns
    key = (resolved, mtime_ns)
    connection = _DUCKDB_CATALOG_CONNECTIONS.get(key)
    if connection is None:
        connection = duckdb.connect(resolved, read_only=True)
        _DUCKDB_CATALOG_CONNECTIONS[key] = connection
    return connection


def _duckdb_table_columns(con: object, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


def _spatial_coarse_predicates(
    bbox: tuple[float, float, float, float, float, float],
    index_strategy: str,
) -> list[str]:
    if index_strategy == "hilbert":
        return ["hilbert_min_xy <= ?", "hilbert_max_xy >= ?"]
    return _tile_overlap_predicates()


def _spatial_coarse_params(
    bbox: tuple[float, float, float, float, float, float],
    index_strategy: str,
) -> list[int]:
    if index_strategy == "hilbert":
        start, end = _hilbert_range_for_bbox(bbox)
        return [end, start]
    return _tile_overlap_params(bbox)


def _hilbert_range_for_bbox(
    bbox: tuple[float, float, float, float, float, float],
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y, _min_z, _max_z = bbox
    x0 = _hilbert_quantize(math.floor(min_x))
    x1 = _hilbert_quantize(math.floor(max_x))
    y0 = _hilbert_quantize(math.floor(min_y))
    y1 = _hilbert_quantize(math.floor(max_y))
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    cell_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if cell_count > 1_024:
        return 0, (1 << 32) - 1
    values = [
        _hilbert_xy_key(x, y)
        for x in range(x_min, x_max + 1)
        for y in range(y_min, y_max + 1)
    ]
    return min(values), max(values)


def _hilbert_quantize(value: float) -> int:
    shifted = int(math.floor(value)) + 32_768
    return min(max(shifted, 0), 65_535)


def _hilbert_xy_key(x: int, y: int) -> int:
    d = 0
    s = 1 << 15
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = 65_535 - x
                y = 65_535 - y
            x, y = y, x
        s //= 2
    return d


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


def _unique_row_value(rows: Sequence[dict[str, object]], key: str) -> str | None:
    values = {str(row[key]) for row in rows if row.get(key) is not None}
    if len(values) == 1:
        return next(iter(values))
    return None


def _group_pose_rows_by_window(
    rows: Sequence[dict[str, object]],
) -> tuple[tuple[dict[str, object], ...], ...]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["robot_id"]), str(row["session_id"]), str(row["file_uri"]))
        groups.setdefault(key, []).append(row)
    return tuple(
        tuple(sorted(group_rows, key=lambda row: (int(row["start_ts_ns"]), int(row["row_group_id"]))))
        for _key, group_rows in sorted(
            groups.items(),
            key=lambda item: (
                min(int(row["start_ts_ns"]) for row in item[1]),
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        )
    )


def _query_window_bounds(
    rows: Sequence[dict[str, object]],
    *,
    start_ts_ns: int | None,
    end_ts_ns: int | None,
    predicate_filters: _PredicateFilters,
) -> tuple[int, int]:
    lower_bounds = [min(int(row["start_ts_ns"]) for row in rows)]
    upper_bounds = [max(int(row["end_ts_ns"]) for row in rows)]
    if start_ts_ns is not None:
        lower_bounds.append(int(start_ts_ns))
    if end_ts_ns is not None:
        upper_bounds.append(int(end_ts_ns))
    for time_start_ns, time_end_ns in predicate_filters.time_windows:
        lower_bounds.append(time_start_ns)
        upper_bounds.append(time_end_ns)
    query_start_ns = max(lower_bounds)
    query_end_ns = min(upper_bounds)
    if query_start_ns > query_end_ns:
        raise LookupError("query matched no valid time window after clipping")
    return query_start_ns, query_end_ns


def _plan_session_id(seek_plan: SeekPlan) -> str:
    return _unique_row_value(seek_plan._pose_rows, "session_id") or "session"


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "session"


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


def _materialize_media_with_rust(
    *,
    row_groups: Sequence[RowGroupSpan],
    media_rows: Sequence[dict[str, object]],
    output_dir: Path,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
    robotics_bin: str | os.PathLike[str] | None,
) -> tuple[dict[str, int], dict[str, object]]:
    rows_by_key: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in media_rows:
        key = (str(row["file_uri"]), str(row["modality"]), str(row["stream_id"]))
        rows_by_key.setdefault(key, []).append(row)
    if not rows_by_key:
        raise LookupError("query matched no media row groups")

    materializations: list[dict[str, object]] = []
    aggregate_metrics = {
        "planned_range_reads": 0,
        "planned_read_bytes": 0,
        "actual_cold_reads": 0,
        "actual_cold_read_bytes": 0,
        "actual_authorized_bytes": 0,
        "footer_bytes": 0,
        "largest_metadata_read": 0,
        "max_footer_read_offset": 0,
        "max_footer_read_end": 0,
        "range_violations": 0,
    }
    actual_reads: list[object] = []
    violations: list[object] = []
    frames: list[object] = []

    output_dir = Path(output_dir)
    for index, ((file_uri, modality, stream_id), rows) in enumerate(sorted(rows_by_key.items())):
        if modality != "camera":
            raise ValueError(f"media materialization only supports camera channels, got {modality!r}")
        group_row_groups = tuple(int(row["row_group_id"]) for row in rows)
        manifest_path = output_dir / f".{stream_id}_{index}.manifest.json"
        cmd = _robotics_command(robotics_bin) + [
            "media",
            "camera-row-groups",
            "--input",
            file_uri,
            "--row-groups",
            ",".join(str(row_group) for row_group in group_row_groups),
            "--out",
            str(output_dir),
            "--audit-ranges",
            _catalog_audit_ranges(rows),
            "--manifest-out",
            str(manifest_path),
        ]
        if enforce_ranges:
            cmd.extend(
                [
                    "--enforce-ranges",
                    "--footer-allowance-bytes",
                    str(footer_allowance_bytes),
                ]
            )
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        metrics = _parse_cli_metrics(completed.stdout)
        manifest = _load_json_manifest(manifest_path)
        seek = manifest.get("seek", {})
        if isinstance(seek, dict):
            actual_reads.extend(seek.get("actual_reads", []))  # type: ignore[arg-type]
            violations.extend(seek.get("violations", []))  # type: ignore[arg-type]
            aggregate_metrics["actual_authorized_bytes"] += int(
                seek.get("actual_authorized_bytes", 0)
            )
            aggregate_metrics["footer_bytes"] += int(seek.get("footer_bytes", 0))
            aggregate_metrics["largest_metadata_read"] = max(
                aggregate_metrics["largest_metadata_read"],
                int(seek.get("largest_metadata_read", 0)),
            )
            aggregate_metrics["max_footer_read_offset"] = max(
                aggregate_metrics["max_footer_read_offset"],
                int(seek.get("max_footer_read_offset", 0)),
            )
            aggregate_metrics["max_footer_read_end"] = max(
                aggregate_metrics["max_footer_read_end"],
                int(seek.get("max_footer_read_end", 0)),
            )
            aggregate_metrics["planned_read_bytes"] += int(seek.get("planned_read_bytes", 0))
        aggregate_metrics["planned_range_reads"] += int(metrics.get("planned_range_reads", 0))
        aggregate_metrics["range_violations"] += int(metrics.get("range_violations", 0))
        manifest_frames = manifest.get("frames", [])
        if isinstance(manifest_frames, list):
            frames.extend(manifest_frames)
        materializations.append(manifest)

    aggregate_metrics["actual_cold_reads"] = len(actual_reads)
    aggregate_metrics["actual_cold_read_bytes"] = sum(
        int(read.get("length", 0)) for read in actual_reads if isinstance(read, dict)
    )
    media_manifest = {
        "version": 1,
        "output_dir": str(output_dir),
        "selected_row_groups": [asdict(span) for span in row_groups],
        "frames": frames,
        "materializations": materializations,
        "actual_reads": actual_reads,
        "actual_cold_reads": aggregate_metrics["actual_cold_reads"],
        "actual_cold_read_bytes": aggregate_metrics["actual_cold_read_bytes"],
        "actual_authorized_bytes": aggregate_metrics["actual_authorized_bytes"],
        "footer_bytes": aggregate_metrics["footer_bytes"],
        "largest_metadata_read": aggregate_metrics["largest_metadata_read"],
        "max_footer_read_offset": aggregate_metrics["max_footer_read_offset"],
        "max_footer_read_end": aggregate_metrics["max_footer_read_end"],
        "materialized_bytes": aggregate_metrics["actual_authorized_bytes"],
        "planned_read_bytes": aggregate_metrics["planned_read_bytes"],
        "violations": violations,
        "enforcement_enabled": enforce_ranges,
    }
    return aggregate_metrics, media_manifest


def _load_json_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_manifest(
    *,
    seek_plan: SeekPlan,
    diagnostics: CorrectnessReport,
    pose_manifest: dict[str, object] | None,
    imu_manifest: dict[str, object] | None,
    media_manifest: dict[str, object] | None,
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
    for materialized_manifest in (pose_manifest, imu_manifest, media_manifest):
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
            "media_materialization_manifest": media_manifest,
        }
    )
    return manifest


def _aggregate_plan_diagnostics(
    windows: Sequence[SeekPlan],
    catalog_explain: _CatalogExplain,
    *,
    authorized_pose_bytes: int,
    authorized_imu_bytes: int,
    authorized_media_bytes: int,
    authorized_total_bytes: int,
    materialized_pose_imu_bytes: int,
    planned_range_reads: int,
    blocked_by_egress: bool,
    max_egress_bytes: int,
) -> CorrectnessReport:
    del max_egress_bytes
    return CorrectnessReport(
        pose_gap_count=sum(window.diagnostics.pose_gap_count for window in windows),
        imu_gap_count=sum(window.diagnostics.imu_gap_count for window in windows),
        pose_max_gap_ns=max((window.diagnostics.pose_max_gap_ns for window in windows), default=0),
        imu_max_gap_ns=max((window.diagnostics.imu_max_gap_ns for window in windows), default=0),
        null_count=0,
        quaternion_inversions_applied=0,
        extrapolation_checked=False,
        extrapolation_rejected=False,
        velocity_threshold=next(
            (window.diagnostics.velocity_threshold for window in windows if window.diagnostics.velocity_threshold is not None),
            None,
        ),
        matched_row_groups=sum(len(window.pose_row_groups) for window in windows),
        selected_bytes=authorized_total_bytes,
        pose_matched_row_groups=sum(len(window.pose_row_groups) for window in windows),
        pose_selected_bytes=authorized_pose_bytes,
        imu_matched_row_groups=sum(len(window.imu_row_groups) for window in windows),
        imu_selected_bytes=authorized_imu_bytes,
        total_selected_bytes=authorized_total_bytes,
        pose_planned_range_reads=sum(len(window.pose_row_groups) for window in windows),
        imu_planned_range_reads=sum(len(window.imu_row_groups) for window in windows),
        planned_read_bytes=materialized_pose_imu_bytes,
        range_audit_passed=False,
        catalog_query_ms=catalog_explain.catalog_query_ms,
        candidate_row_groups=catalog_explain.candidate_row_groups,
        time_pruned_row_groups=catalog_explain.time_pruned_row_groups,
        spatial_pruned_row_groups=catalog_explain.spatial_pruned_row_groups,
        hilbert_pruned_row_groups=catalog_explain.hilbert_pruned_row_groups,
        exact_spatial_pruned_row_groups=catalog_explain.exact_spatial_pruned_row_groups,
        velocity_pruned_row_groups=catalog_explain.velocity_pruned_row_groups,
        index_strategy=catalog_explain.index_strategy,
        gap_rejected_row_groups=sum(window.diagnostics.gap_rejected_row_groups for window in windows),
        media_matched_row_groups=sum(len(window.media_row_groups) for window in windows),
        media_selected_bytes=authorized_media_bytes,
        media_blocked_by_egress=blocked_by_egress and authorized_media_bytes > 0,
        authorized_pose_bytes=authorized_pose_bytes,
        authorized_imu_bytes=authorized_imu_bytes,
        authorized_media_bytes=authorized_media_bytes,
        authorized_total_bytes=authorized_total_bytes,
        materialized_pose_imu_bytes=materialized_pose_imu_bytes,
        planned_range_reads=planned_range_reads,
        blocked_by_egress=blocked_by_egress,
    )


def _aggregate_query_diagnostics(
    diagnostics: Sequence[CorrectnessReport],
    plan_diagnostics: CorrectnessReport,
    *,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
) -> CorrectnessReport:
    return CorrectnessReport(
        pose_gap_count=sum(report.pose_gap_count for report in diagnostics),
        imu_gap_count=sum(report.imu_gap_count for report in diagnostics),
        pose_max_gap_ns=max((report.pose_max_gap_ns for report in diagnostics), default=0),
        imu_max_gap_ns=max((report.imu_max_gap_ns for report in diagnostics), default=0),
        null_count=sum(report.null_count for report in diagnostics),
        quaternion_inversions_applied=sum(report.quaternion_inversions_applied for report in diagnostics),
        extrapolation_checked=all(report.extrapolation_checked for report in diagnostics) if diagnostics else False,
        extrapolation_rejected=any(report.extrapolation_rejected for report in diagnostics),
        velocity_threshold=plan_diagnostics.velocity_threshold,
        matched_row_groups=sum(report.matched_row_groups for report in diagnostics),
        selected_bytes=plan_diagnostics.authorized_total_bytes,
        pose_matched_row_groups=sum(report.pose_matched_row_groups for report in diagnostics),
        pose_selected_bytes=plan_diagnostics.authorized_pose_bytes,
        imu_matched_row_groups=sum(report.imu_matched_row_groups for report in diagnostics),
        imu_selected_bytes=plan_diagnostics.authorized_imu_bytes,
        total_selected_bytes=plan_diagnostics.authorized_total_bytes,
        pose_planned_range_reads=sum(report.pose_planned_range_reads for report in diagnostics),
        imu_planned_range_reads=sum(report.imu_planned_range_reads for report in diagnostics),
        planned_read_bytes=sum(report.planned_read_bytes for report in diagnostics),
        range_audit_passed=all(report.range_audit_passed for report in diagnostics) if diagnostics else False,
        catalog_query_ms=plan_diagnostics.catalog_query_ms,
        candidate_row_groups=plan_diagnostics.candidate_row_groups,
        time_pruned_row_groups=plan_diagnostics.time_pruned_row_groups,
        spatial_pruned_row_groups=plan_diagnostics.spatial_pruned_row_groups,
        hilbert_pruned_row_groups=plan_diagnostics.hilbert_pruned_row_groups,
        exact_spatial_pruned_row_groups=plan_diagnostics.exact_spatial_pruned_row_groups,
        velocity_pruned_row_groups=plan_diagnostics.velocity_pruned_row_groups,
        index_strategy=plan_diagnostics.index_strategy,
        gap_rejected_row_groups=sum(report.gap_rejected_row_groups for report in diagnostics),
        media_matched_row_groups=sum(report.media_matched_row_groups for report in diagnostics),
        media_selected_bytes=plan_diagnostics.authorized_media_bytes,
        media_blocked_by_egress=False,
        authorized_pose_bytes=plan_diagnostics.authorized_pose_bytes,
        authorized_imu_bytes=plan_diagnostics.authorized_imu_bytes,
        authorized_media_bytes=plan_diagnostics.authorized_media_bytes,
        authorized_total_bytes=plan_diagnostics.authorized_total_bytes,
        materialized_pose_imu_bytes=plan_diagnostics.materialized_pose_imu_bytes,
        planned_range_reads=sum(report.planned_range_reads for report in diagnostics),
        blocked_by_egress=False,
        actual_cold_reads=sum(report.actual_cold_reads for report in diagnostics),
        actual_cold_read_bytes=sum(report.actual_cold_read_bytes for report in diagnostics),
        actual_authorized_bytes=sum(report.actual_authorized_bytes for report in diagnostics),
        footer_allowance_bytes=footer_allowance_bytes,
        footer_bytes=sum(report.footer_bytes for report in diagnostics),
        largest_metadata_read=max((report.largest_metadata_read for report in diagnostics), default=0),
        max_footer_read_offset=max((report.max_footer_read_offset for report in diagnostics), default=0),
        max_footer_read_end=max((report.max_footer_read_end for report in diagnostics), default=0),
        range_enforced=enforce_ranges,
        range_violations=sum(report.range_violations for report in diagnostics),
    )


def _batch_query_manifest(
    *,
    batch_plan: BatchSeekPlan,
    results: Sequence[QueryResult],
    diagnostics: CorrectnessReport,
    predicate: str | None,
    channels: Sequence[str],
    target_hz: float,
    enforce_ranges: bool,
    footer_allowance_bytes: int,
) -> dict[str, object]:
    actual_reads: list[object] = []
    violations: list[object] = []
    for result in results:
        if result.manifest is None:
            continue
        actual_reads.extend(result.manifest.get("actual_reads", []))  # type: ignore[arg-type]
        violations.extend(result.manifest.get("violations", []))  # type: ignore[arg-type]
    return {
        "version": 1,
        "plan_inputs": {
            "predicate": predicate,
            "channels": list(channels),
            "target_hz": target_hz,
            "enforce_ranges": enforce_ranges,
            "footer_allowance_bytes": footer_allowance_bytes,
        },
        "windows": [result.manifest for result in results],
        "window_count": len(results),
        "authorized_pose_bytes": batch_plan.authorized_pose_bytes,
        "authorized_imu_bytes": batch_plan.authorized_imu_bytes,
        "authorized_media_bytes": batch_plan.authorized_media_bytes,
        "authorized_total_bytes": batch_plan.authorized_total_bytes,
        "materialized_pose_imu_bytes": batch_plan.materialized_pose_imu_bytes,
        "planned_range_reads": diagnostics.planned_range_reads,
        "blocked_by_egress": batch_plan.blocked_by_egress,
        "egress_limit_bytes": batch_plan.egress_limit_bytes,
        "actual_reads": actual_reads,
        "actual_cold_reads": len(actual_reads),
        "actual_cold_read_bytes": diagnostics.actual_cold_read_bytes,
        "actual_authorized_bytes": diagnostics.actual_authorized_bytes,
        "footer_allowance_bytes": footer_allowance_bytes,
        "footer_bytes": diagnostics.footer_bytes,
        "largest_metadata_read": diagnostics.largest_metadata_read,
        "max_footer_read_offset": diagnostics.max_footer_read_offset,
        "max_footer_read_end": diagnostics.max_footer_read_end,
        "violations": violations,
        "enforcement_enabled": enforce_ranges,
        "diagnostics": asdict(diagnostics),
    }


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
