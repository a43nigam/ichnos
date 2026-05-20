from __future__ import annotations

import os
from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping


ENTRY_POINT_GROUP = "physicaldb.dataset_adapters"


class DatasetAdapter(ABC):
    adapter_id: str
    version: str = "1"

    @abstractmethod
    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def inspect(self, path_or_uri: str | os.PathLike[str]):
        raise NotImplementedError

    @abstractmethod
    def suggest_manifest(
        self,
        profile,
        *,
        dataset_id: str,
        robot_id: str,
        session_id: str,
        adapter_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_manifest(self, manifest: Mapping[str, Any] | str | os.PathLike[str]):
        raise NotImplementedError

    @abstractmethod
    def ingest(
        self,
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        *,
        output_root: str | os.PathLike[str],
        row_group_rows: int = 500,
        robotics_bin: str | os.PathLike[str] | None = None,
    ):
        raise NotImplementedError


class AdapterRegistry:
    def __init__(self, *, load_entry_points: bool = True) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        for adapter in _builtin_adapters():
            self.register(adapter)
        if load_entry_points:
            self.load_entry_points()

    def register(self, adapter: DatasetAdapter) -> None:
        adapter_id = getattr(adapter, "adapter_id", "")
        if not adapter_id:
            raise ValueError("dataset adapter must define adapter_id")
        self._adapters[str(adapter_id)] = adapter

    def get(self, adapter_id: str) -> DatasetAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._adapters))
            raise KeyError(f"unknown dataset adapter {adapter_id!r}; known adapters: {known}") from exc

    def list(self) -> tuple[DatasetAdapter, ...]:
        return tuple(self._adapters.values())

    def select_for_path(self, path_or_uri: str | os.PathLike[str]) -> DatasetAdapter:
        for adapter in self.list():
            if adapter.can_inspect(path_or_uri):
                return adapter
        return self.get("generic_media_placeholder")

    def select_for_profile(self, profile) -> DatasetAdapter:
        adapter_id = getattr(profile, "adapter_id", "")
        if adapter_id:
            return self.get(str(adapter_id))
        return self.select_for_path(getattr(profile, "input_uri", ""))

    def select_for_manifest(self, manifest: Mapping[str, Any]) -> DatasetAdapter:
        adapter_id = str(manifest.get("adapter_id") or "")
        if adapter_id:
            return self.get(adapter_id)
        sources = manifest.get("sources", [])
        if isinstance(sources, list) and sources:
            first = sources[0]
            if isinstance(first, Mapping):
                source_path = first.get("path")
                if source_path is not None:
                    return self.select_for_path(str(source_path))
        return self.get("generic_media_placeholder")

    def load_entry_points(self) -> None:
        try:
            entry_points = metadata.entry_points()
        except Exception:
            return
        if hasattr(entry_points, "select"):
            selected = entry_points.select(group=ENTRY_POINT_GROUP)
        else:  # pragma: no cover - old importlib.metadata compatibility
            selected = entry_points.get(ENTRY_POINT_GROUP, [])
        for entry_point in selected:
            loaded = entry_point.load()
            adapter = loaded() if isinstance(loaded, type) else loaded
            self.register(adapter)


_REGISTRY: AdapterRegistry | None = None


def adapter_registry() -> AdapterRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = AdapterRegistry()
    return _REGISTRY


def register_adapter(adapter: DatasetAdapter) -> None:
    adapter_registry().register(adapter)


def get_adapter(adapter_id: str) -> DatasetAdapter:
    return adapter_registry().get(adapter_id)


def list_adapters() -> tuple[DatasetAdapter, ...]:
    return adapter_registry().list()


class _BaseAdapter(DatasetAdapter):
    adapter_id = "base"

    def inspect(self, path_or_uri: str | os.PathLike[str]):
        from . import onboarding

        profile = onboarding._inspect_dataset_impl(path_or_uri)
        return onboarding._profile_with_adapter(profile, self.adapter_id)

    def suggest_manifest(
        self,
        profile,
        *,
        dataset_id: str,
        robot_id: str,
        session_id: str,
        adapter_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from . import onboarding

        manifest = onboarding._suggest_manifest_impl(
            profile,
            dataset_id=dataset_id,
            robot_id=robot_id,
            session_id=session_id,
        )
        manifest["adapter_id"] = self.adapter_id
        if adapter_options:
            manifest["adapter_options"] = dict(adapter_options)
        return manifest

    def validate_manifest(self, manifest: Mapping[str, Any] | str | os.PathLike[str]):
        from . import onboarding

        return onboarding._validate_manifest_impl(manifest)

    def ingest(
        self,
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        *,
        output_root: str | os.PathLike[str],
        row_group_rows: int = 500,
        robotics_bin: str | os.PathLike[str] | None = None,
    ):
        from . import onboarding

        return onboarding._ingest_manifest_impl(
            manifest,
            output_root=output_root,
            row_group_rows=row_group_rows,
            robotics_bin=robotics_bin,
            adapter_id=self.adapter_id,
        )


class NormalizedParquetAdapter(_BaseAdapter):
    adapter_id = "normalized_parquet"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        path = Path(path_or_uri)
        return path.suffix.lower() == ".parquet" or (path.is_dir() and any(path.rglob("*.parquet")))


class EurocAdapter(_BaseAdapter):
    adapter_id = "euroc"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        from . import onboarding

        path = Path(path_or_uri)
        return (path.is_dir() and onboarding._find_euroc_root(path) is not None) or (
            path.suffix.lower() == ".zip" and onboarding._zip_looks_like_euroc(path)
        )


class McapPoseAdapter(_BaseAdapter):
    adapter_id = "mcap_pose"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        return Path(path_or_uri).suffix.lower() == ".mcap"

    def suggest_manifest(
        self,
        profile,
        *,
        dataset_id: str,
        robot_id: str,
        session_id: str,
        adapter_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(adapter_options or {})
        topics = tuple(getattr(profile.files[0], "topics", ()) if getattr(profile, "files", ()) else ())
        if topics and "topic" not in options:
            options["topic"] = topics[0]
        return super().suggest_manifest(
            profile,
            dataset_id=dataset_id,
            robot_id=robot_id,
            session_id=session_id,
            adapter_options=options,
        )


class KittiOxtsAdapter(_BaseAdapter):
    adapter_id = "kitti_oxts"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        from . import onboarding

        return onboarding._looks_like_kitti_oxts(Path(path_or_uri))

    def inspect(self, path_or_uri: str | os.PathLike[str]):
        from . import onboarding

        return onboarding._profile_with_adapter(onboarding._inspect_kitti_oxts(Path(path_or_uri)), self.adapter_id)


class NuscenesEgoAdapter(_BaseAdapter):
    adapter_id = "nuscenes_ego"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        from . import onboarding

        return onboarding._looks_like_nuscenes_ego(Path(path_or_uri))

    def inspect(self, path_or_uri: str | os.PathLike[str]):
        from . import onboarding

        return onboarding._profile_with_adapter(onboarding._inspect_nuscenes_ego(Path(path_or_uri)), self.adapter_id)


class GenericDatasetAdapter(_BaseAdapter):
    adapter_id = "generic_dataset"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        return True

    def inspect(self, path_or_uri: str | os.PathLike[str]):
        from . import onboarding

        return onboarding._profile_with_adapter(onboarding._inspect_generic_dataset(path_or_uri), self.adapter_id)

    def suggest_manifest(
        self,
        profile,
        *,
        dataset_id: str,
        robot_id: str,
        session_id: str,
        adapter_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = super().suggest_manifest(
            profile,
            dataset_id=dataset_id,
            robot_id=robot_id,
            session_id=session_id,
            adapter_options=adapter_options,
        )
        manifest["mapping_status"] = "draft"
        for stream in manifest.get("streams", []):
            if isinstance(stream, dict):
                stream.setdefault("mapping_status", "draft")
                warnings = list(stream.get("warnings", []))
                if stream.get("confidence", 1.0) < 0.75:
                    warnings.append("low-confidence inferred mapping; review before ingest")
                if not stream.get("timestamp"):
                    warnings.append("timestamp mapping is unresolved")
                if not stream.get("channels"):
                    warnings.append("channel mapping is unresolved")
                if warnings:
                    stream["warnings"] = sorted(set(str(warning) for warning in warnings))
        return manifest

    def ingest(
        self,
        manifest: Mapping[str, Any] | str | os.PathLike[str],
        *,
        output_root: str | os.PathLike[str],
        row_group_rows: int = 500,
        robotics_bin: str | os.PathLike[str] | None = None,
    ):
        from . import onboarding

        return onboarding._ingest_generic_dataset_impl(
            manifest,
            output_root=output_root,
            row_group_rows=row_group_rows,
            robotics_bin=robotics_bin,
            adapter_id=self.adapter_id,
        )


class GenericMediaPlaceholderAdapter(_BaseAdapter):
    adapter_id = "generic_media_placeholder"

    def can_inspect(self, path_or_uri: str | os.PathLike[str]) -> bool:
        return True


def _builtin_adapters() -> tuple[DatasetAdapter, ...]:
    return (
        EurocAdapter(),
        NormalizedParquetAdapter(),
        McapPoseAdapter(),
        KittiOxtsAdapter(),
        NuscenesEgoAdapter(),
        GenericDatasetAdapter(),
        GenericMediaPlaceholderAdapter(),
    )
