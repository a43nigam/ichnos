# Dataset Adapter SDK

The dataset adapter SDK is the Python extension point for self-serve dataset onboarding.
Adapters inspect an input dataset, suggest a JSON manifest, validate the manifest, and dispatch ingestion through the existing Rust CLI paths.

## Public API

```python
from physicaldb import DatasetAdapter, list_adapters, get_adapter, register_adapter
```

`DatasetAdapter` implementations provide:

- `adapter_id`: stable string used by manifests and CLI `--adapter`.
- `can_inspect(path_or_uri) -> bool`: returns whether the adapter can handle an input.
- `inspect(path_or_uri) -> DatasetProfile`: returns files, streams, timestamp candidates, channels, and warnings.
- `suggest_manifest(profile, dataset_id=..., robot_id=..., session_id=..., adapter_options=None) -> dict`: returns a v1 JSON manifest.
- `validate_manifest(manifest) -> ValidationReport`: validates the manifest.
- `ingest(manifest, output_root=..., row_group_rows=..., robotics_bin=None) -> IngestReport`: writes managed Parquet/catalog outputs.

Built-in adapters:

- `euroc`
- `normalized_parquet`
- `mcap_pose`
- `kitti_oxts`
- `nuscenes_ego`
- `generic_dataset`
- `generic_media_placeholder`

Adapter ordering is intentional: dataset-specific adapters are selected before
`generic_dataset`, and the placeholder remains the final fallback.

## Generic Dataset Drafts

`generic_dataset` is a deterministic profiler for messy customer datasets. It
does not claim ingest-ready semantics by default; it emits reviewable mapping
drafts with confidence scores and warnings.

Supported first-slice inputs:

- MCAP files: topics and schema names when the optional Python MCAP reader is available.
- Parquet files: columns, row counts, timestamp candidates, and likely pose/IMU/camera/media streams.
- CSV and image folders: table columns, image sequence counts, timestamp-in-filename candidates, and nearby calibration file hints.
- S3-compatible prefixes: object listing through the Rust `object_store` stack, with schema/topic decoding deferred.

Profile and manifest JSON may include additive optional metadata:
`discovery`, `confidence`, `mapping_status`, and per-stream `warnings`.
Existing v1 manifests without these fields remain valid.

Draft mappings validate with warnings when inferred fields are unresolved.
Explicit final mappings still fail validation when required fields are missing
or invalid.

Reviewed generic mappings can be ingested after setting the top-level and
per-stream `mapping_status` fields to `final`. Generic ingest currently
normalizes local CSV/Parquet pose and IMU streams plus local timestamped camera
image sequences into the existing managed Parquet/catalog layout. Generic MCAP
pose ingest requires an explicit reviewed topic. S3 prefixes are inspect-only in
this slice; raw S3 generic ingest fails clearly rather than producing outputs.
Final generic ingest preflights mapped columns, timestamp casts, image filename
timestamps, and output-root safety before writing; failed ingests clean up their
temporary work directory and do not leave partial managed outputs.

## EuRoC Calibration Metadata

The EuRoC adapter inspects `mav0/<stream_id>/sensor.yaml` for sensor streams such as
`imu0`, `cam0`, and `cam1`. When present, the generated `DatasetProfile` stream and
manifest stream include an optional `calibration` object with sensor/body frame IDs,
`T_body_sensor` from EuRoC `T_BS`, and available camera/IMU metadata such as
`resolution`, `intrinsics`, `distortion_model`, `distortion_coefficients`, and
`rate_hz`.

Missing or partially parsed calibration files are warnings, not validation failures.
Manifest v1 remains backward-compatible: old manifests without `calibration` still
validate. The metadata is audit-only in this milestone; transforms are not applied
during Parquet ingestion, tensor materialization, or query planning.

## CLI Workflow

```bash
robotics dataset adapters
robotics dataset inspect --adapter auto --input path/to/dataset --out profile.json
robotics dataset init-manifest --adapter auto --profile profile.json --out dataset.json \
  --dataset-id dataset_001 --robot-id robot_001 --session-id session_001
robotics dataset validate --manifest dataset.json --out validation.json
robotics dataset ingest --manifest dataset.json --output-root data/managed/dataset_001
```

For generic datasets, use `init-mapping` to make the draft status explicit:

```bash
robotics dataset inspect --adapter generic_dataset --input path/to/customer_dataset --out profile.json
robotics dataset init-mapping --adapter generic_dataset --profile profile.json --out dataset.json
robotics dataset validate --manifest dataset.json --out validation.json
# Review dataset.json, set mapping_status fields to "final", then:
robotics dataset ingest --adapter generic_dataset --manifest dataset.json --output-root data/managed/dataset_001
```

The M1 generic customer demo generates a deliberately messy local fixture,
runs inspect, init-mapping, finalization with a known mapping, validation,
ingest, and a catalog query, then writes each artifact under the workdir:

```bash
robotics dataset demo --workdir data/demo/generic_customer
# Equivalent script entry point:
python3 scripts/demo_generic_customer_workflow.py --workdir data/demo/generic_customer
```

Managed catalogs can be queried through the dataset CLI. The query command
wraps `physicaldb.query()` and writes a JSON summary with tensor shape,
timestamps, diagnostics, and the tensor certificate:

```bash
robotics dataset query --managed-root data/demo/generic_customer/managed \
  --robot-id customer_bot_001 --session-id demo_session_001 \
  --channels pos_xyz,rot_wxyz,vel_xyz --out query_summary.json
```

For S3-compatible inspection, the dataset profiler shells through the Rust
object-store helper rather than adding Python S3 dependencies:

```bash
robotics object-store list --uri s3://bucket/prefix --limit 500
robotics dataset inspect --adapter generic_dataset --input s3://bucket/prefix --out profile.json
```

For customer S3 data, the supported v1 path is staging-first: copy a bounded
prefix locally, inspect the staged directory, then ingest locally. This keeps
raw S3 ingest explicit rather than silently downloading arbitrary buckets.

```bash
robotics object-store sync-prefix --uri s3://bucket/raw-session --out data/staged/raw-session --limit 500
robotics dataset stage-s3 --input s3://bucket/raw-session \
  --out data/staged/raw-session --manifest staged_objects.json
robotics dataset inspect --adapter generic_dataset --input data/staged/raw-session --out profile.json
```

Managed outputs can be uploaded after ingest:

```bash
robotics dataset upload-managed --managed-root data/managed/dataset_001 \
  --uri s3://bucket/managed/dataset_001 --manifest upload_manifest.json
```

Use `--adapter <adapter_id>` to force an adapter. Use `--adapter-option KEY=VALUE` during `init-manifest` for adapter-specific hints, for example an MCAP topic:

```bash
robotics dataset init-manifest --adapter mcap_pose --profile profile.json --out dataset.json \
  --adapter-option topic=/pose
```

## External Adapter Packages

External packages register adapters through the Python entry-point group `physicaldb.dataset_adapters`.

Example `pyproject.toml`:

```toml
[project.entry-points."physicaldb.dataset_adapters"]
my_dataset = "my_package.adapters:MyDatasetAdapter"
```

Example adapter:

```python
from physicaldb import DatasetAdapter


class MyDatasetAdapter(DatasetAdapter):
    adapter_id = "my_dataset"
    version = "1"

    def can_inspect(self, path_or_uri):
        return str(path_or_uri).endswith(".mylog")

    def inspect(self, path_or_uri):
        ...

    def suggest_manifest(self, profile, *, dataset_id, robot_id, session_id, adapter_options=None):
        ...

    def validate_manifest(self, manifest):
        ...

    def ingest(self, manifest, *, output_root, row_group_rows=500, robotics_bin=None):
        ...
```

The registry loads entry points lazily when `physicaldb.list_adapters()` or adapter-backed onboarding functions are first used.

## Compatibility

Manifest v1 remains JSON-only. `adapter_id` and `adapter_options` are optional additive fields.
Old manifests without `adapter_id` continue to validate and ingest through source-shape auto-detection.
The optional per-stream `calibration`, `confidence`, `discovery`,
`mapping_status`, and `warnings` fields are additive and can be omitted.
