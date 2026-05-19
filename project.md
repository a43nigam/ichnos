# Robotics Fleet Query Engine

## Goal
Build a Rust-first robotics data query engine that indexes robotics sessions, prunes candidate row groups, performs byte-accounted reads, and returns uniformly sampled tensor-shaped windows for model training and evaluation.

Here is the original tech spec:

# Product Definition

You are building a **behavioral query engine for physical AI data**. The core value proposition is simple: given petabytes of raw robotics logs sitting in cold cloud storage, an engineer can describe a physical event and receive a training-ready tensor back in seconds, without downloading or preprocessing anything.

Phase 2 is the product. Phase 3 is the interface that makes it feel real. Phase 1 is the necessary plumbing underneath.

The customer is an ML engineer at a Series A-C robotics or physical AI startup that is generating serious log volume but doesn't have a dedicated data platform team. They are currently either writing brittle custom parsers, downloading entire files to find one event, or both.

---

# Full Engineering Specification

## Module 1: Ingestion & Columnar Storage

### Objective
Convert raw multi-modal log files into a queryable, optimized Parquet layout on S3. This runs once per log file, either at upload time or on a scheduled batch.

### Inputs
- MCAP, ROS2 bag, Protobuf, raw Parquet
- Delivered via local disk, S3 upload, or network socket

### Data Schema

| Entity | Column Name | Physical Type | Encoding |
|---|---|---|---|
| Timestamp | `ts_ns` | Int64 | Nanoseconds since Unix epoch, UTC |
| Position | `pos_xyz` | FixedSizeList(Float64, 3) | Contiguous x,y,z array |
| Orientation | `rot_wxyz` | FixedSizeList(Float64, 4) | Normalized quaternion |
| Rigid Transform | `transform_4x4` | FixedSizeList(Float64, 16) | Row-major SE(3) matrix |
| LiDAR | `lidar_geometry` | GeoArrow.point / WKB | Native Parquet geometry type |
| Camera | `camera_bytes` | LargeBinary | Sharded, inline-compressed chunks |
| IMU | `imu_accel`, `imu_gyro` | FixedSizeList(Float64, 3) | Scalar high-frequency columns |

### Architecture
- **Ingestion Daemon:** Multithreaded Rust parser. Stream-reads chunks from disk or socket. Never loads full file into memory.
- **Schema Evolution Handler:** Detects mid-session sensor driver renames and merges via Arrow schema evolution without breaking write path.
- **Null Frame Handler:** Dropped or malformed frames written as explicit Arrow bitmask nulls. Never silently skipped.
- **Columnar Writer:** Partitions row groups by modality. High-frequency scalar streams (IMU, joints) and low-frequency high-bandwidth streams (LiDAR, camera) written to decoupled column blocks to maximize downstream read-skipping.

### Output
Partitioned Parquet files on S3, organized as:
```
s3://bucket/fleet/{robot_id}/{date}/{session_id}/
    scalars.parquet        # IMU, joints, kinematics
    geometry.parquet       # LiDAR point clouds
    video.parquet          # Camera byte buffers
    metadata.parquet       # Session-level index entry
```

### Definition of Done
- Ingest a 10GB MCAP file containing unaligned video, LiDAR, and 400Hz IMU
- Output valid, compressed, GeoArrow-compliant Parquet files
- Zero full-file memory loads during ingestion

---

## Module 2: Metadata Index & Seek Engine

### Objective
Build and maintain a lightweight behavioral index over all ingested sessions. Execute spatial, temporal, and kinematic queries directly against this index to identify matching byte offsets in S3, then stream only those bytes.

### Index Structure
Each ingested session produces one metadata record containing:
- `robot_id`, `session_id`, `start_ts`, `end_ts`
- Bounding box of full trajectory (min/max x, y, z)
- Per-row-group: byte offset, start/end timestamp, spatial bounding box, min/max velocity magnitude
- Explicit temporal gap markers where telemetry was lost

This catalog is stored as a small, always-hot Parquet file (or DuckDB database file) — never exceeding a few GB even across thousands of hours of logs.

### Query Engine Architecture
```
[ Python Query API ]
        │
        ▼
[ DuckDB Query Optimizer ]
        │
  (MobilityDuck Extension)
        │
        ▼
[ Spatial-Temporal Metadata Catalog ]
    (R-Tree / Hilbert Curve Index)
        │
   ┌────┴────┐
   ▼         ▼
[Matching   [Irrelevant
Row Groups] Row Groups
            PRUNED]
        │
        ▼
[ HTTP Range-GET → S3 ]
  (Matching bytes only)
```

### Query Predicate Support
```sql
SELECT file_path, row_group_id, byte_offset, start_time, end_time
FROM fleet_metadata_catalog
WHERE robot_id = 'humanoid_04'
  AND duration(joint_trajectory, 'seconds') > 5
  AND ST_Within(base_link_position, ST_MakeEnvelope(0, 0, 10, 10))
  AND velocity_magnitude > 5.0
```

Supported predicate types:
- Robot / session / date filters
- Spatial: `ST_Within`, `ST_Intersects`, bounding box
- Temporal: duration, time range, gap exclusion
- Kinematic: velocity magnitude, acceleration thresholds, joint angle ranges

### Critical Constraints
- **Egress safety gate:** Scalar metadata columns are always evaluated before any byte-range requests are authorized against camera or LiDAR columns. A query that would pull >N GB of imagery must warn or block before execution.
- **Temporal gap honesty:** Gaps in telemetry are explicit index entries. Queries never interpolate across a gap silently.
- **Cold file, zero download:** The engine must be able to identify and return a 3-second window from a 500GB S3 file using only HTTP range requests against known byte offsets.

### Definition of Done
- Sub-millisecond query execution against an index representing 1,000+ hours of sessions
- Successful extraction of a 3-second window from a cold S3 file without downloading surrounding data
- Egress gate correctly blocks an overly broad imagery query before any bytes are streamed

---

## Module 3: Kinematic Interpolation & Tensor Bridge

### Objective
Take the raw columnar byte ranges returned by Module 2 and materialize them as a perfectly time-aligned, resampled, multi-channel PyTorch tensor. Zero memory copy. No preprocessing by the caller.

### Interpolation Logic

**Scalar / Translation (Linear)**
Standard linear interpolation between the two bounding raw samples at the requested timestamp.

**Orientation (Slerp)**
Quaternions are interpolated spherically across the 4D unit hypersphere. Before each Slerp step, the kernel evaluates `q1 · q2`. If the dot product is negative, the target quaternion is negated to guarantee shortest-path interpolation and prevent rotational jitter.

**Resampling Controller**
Accepts a target frequency from the caller (e.g., 30Hz). Aligns all input channels — regardless of their native sample rates — onto a uniform output timeline. A 400Hz IMU and a 12Hz positional tracker both land on the same 30Hz frame matrix.

### Tensor Bridge
- Output is a `torch.Tensor` delivered via DLPack or direct pointer memory mapping
- The Arrow table produced by interpolation is mapped directly into PyTorch memory space
- Zero RAM duplication: verified by memory profiling that no copy occurs during the Arrow → Tensor transition

### API Surface (Python)
```python
from physicaldb import query

tensor = query(
    robot_id="humanoid_04",
    predicate="velocity_magnitude > 5.0 AND ST_Within(position, zone_A)",
    channels=["pos_xyz", "rot_wxyz", "imu_accel"],
    target_hz=30,
    output="torch"
)
# Returns: torch.Tensor, shape [T, C]
```

### Critical Constraints
- **Extrapolation rejection:** If the requested window extends beyond the bounds of a recording, the kernel throws an explicit extrapolation error. It never guesses.
- **Quaternion inversion:** Enforced on every Slerp step, not optional.
- **No caller preprocessing:** The tensor returned is immediately usable in a training loop. Shape, dtype, and alignment are guaranteed.

### Definition of Done
- Single Python call returns a synchronized multi-channel tensor from unaligned raw logs
- Memory profiler confirms zero-copy transition from Arrow to PyTorch
- Extrapolation boundary correctly raises an error rather than returning fabricated data

---

## Build Order Recommendation

**Start with Module 2.** Build a minimal DuckDB-backed metadata catalog that can index pre-processed Parquet files and execute byte-range queries against S3. This is your core defensible technology and your demo.

**Then Module 3.** Wire the byte ranges from Module 2 into a Python interpolation layer that outputs a tensor. This completes the demo loop — one query call, one tensor back.

**Then Module 1.** Once you have paying customers or serious pilots, build the ingestion daemon properly so customers can onboard their own raw logs. Initially you can accept pre-converted Parquet manually or write a quick-and-dirty Python converter just to unblock pilots.

This order gets you to a convincing, defensible demo fastest, without spending months on ingestion infrastructure before you've validated that anyone will pay for the query layer.

## Current Status
| Area | Status | Notes |
| --- | --- | --- |
| Rust workspace | Done for M0 | Core crates compile and pass tests/clippy. |
| Catalog generation/query | Done for M1 | Fake catalog, predicate filtering, pose/IMU/media Parquet metadata indexing, and explicit row-group temporal gap metadata implemented in Rust. |
| Row-group byte accounting | Done for M2 | Local and object-store range readers implemented with planned/transferred byte accounting, object-store upload support for live smoke setup, and enforced row-group range auditing around actual Parquet materialization reads. |
| Tensorization | Done for M4 | Quaternion slerp, pose/IMU resampling, tensor-shaped buffers, extrapolation checks, selected local/object-store Parquet row-group loading, and NumPy `.npy` export implemented. |
| Dataset ingestion | In progress | Synthetic pose, synthetic Parquet, JSON-pose MCAP, ROS pose MCAP, KITTI raw OXTS, nuScenes ego-pose, and EuRoC groundtruth/IMU/camera ingestion implemented; richer public-dataset calibration adapters remain. |
| Arrow/Parquet/S3 | Done for M2 | Arrow/Parquet, object-store upload/range-read plumbing, S3 URI parsing, and live MinIO validation are implemented. |
| DuckDB/Python facade | Prototype complete for enforced cold pose+IMU+camera planning/materialization | Rust can write hot pose, IMU, and media metadata catalog Parquet files, build a persistent DuckDB catalog DB with derived spatial tile keys, and generate large fake catalog DBs; Python `physicaldb.plan()`/`query()` handle one matched file window, while `physicaldb.plan_batch()`/`query_batch()` split multi-session or multi-file matches into independent windows with aggregate egress gates, enforced cold range reads, pose+IMU tensors, and optional camera frame materialization. |
| Demo loop | Done for synthetic M4 | `robotics demo` now runs synthetic ingest -> Parquet index -> velocity/bbox catalog query -> byte-accounted row-group range read -> tensorization; `physicaldb.query()` provides the product-shaped Python path. |

## Roadmap
1. **M0: Compiling Rust core**
   - Workspace, CLI, core schemas, fake catalog, query filtering, synthetic generator, tensorizer.
   - Unit tests for predicate filtering, quaternion math, resampling, and byte accounting.
   - Status: complete.
2. **M1: Real Parquet storage**
   - Add `arrow-rs`/`parquet`.
   - Write synthetic sessions to Parquet with controlled row-group sizes.
   - Extract row-group statistics and byte ranges into the catalog.
   - Status: complete for local files.
3. **M2: S3-compatible range reads**
   - Add `object_store` or AWS SDK support.
   - Verify local and S3-compatible range reads transfer only selected row groups.
   - Record bytes transferred versus file size in benchmarks.
   - Status: complete for local object-store backend and live MinIO validation, with selected row-group audit spans enforced around actual Parquet materialization reads, bounded footer/metadata reads counted separately, JSON seek manifests, and an opt-in S3-compatible pose+IMU+media smoke.
4. **M3: Public dataset ingestion**
   - MCAP ingestion first.
   - KITTI raw conversion next.
   - nuScenes mini ingestion after core path is stable.
   - Status: JSON-pose MCAP, common ROS pose MCAP schemas, KITTI raw OXTS, nuScenes ego-pose, and EuRoC groundtruth/IMU/camera paths complete; CARLA/Hilti dataset-specific calibration conventions remain.
5. **M4: Demo loop**
   - Ingest one public/synthetic session.
   - Query velocity/bounding-box behavior.
   - Return `[T, C]` output and report row groups matched, bytes read, and wall-clock time.
   - Status: complete for synthetic Parquet sessions and EuRoC pose+IMU plus media planning, including tensor loading from selected pose and IMU row groups through local/object-store catalog URIs, hot pose/IMU/media catalog Parquet output, persistent DuckDB catalog DB construction, coarse spatial tile pruning, Python/DuckDB velocity/spatial predicate pruning, plan-only cold-seek contracts, default gap rejection, enforced cold-read auditing, media egress gates, optional NumPy/PyTorch loading, JSON seek manifests, and caller-visible correctness/explain diagnostics.

## Canonical Data Model
Sensor rows use nanosecond timestamps, `robot_id`, `session_id`, pose position, quaternion orientation, and linear velocity. Catalog rows summarize row-group/session windows with min/max timestamps, bounding boxes, velocity range, URI, row-group id, byte offset, byte length, nominal cadence, and explicit temporal gap markers.

## Benchmark Targets
- Fake 100k-entry DuckDB catalog hot planning: sub-10 ms p95 on a laptop for tight spatial, broad spatial, time-bounded, velocity-only, and mixed spatial/time/velocity predicates.
- Demo query: selected bytes should be close to matched row-group sizes, not full dataset size.
- Tensor output: uniformly spaced timestamps at target Hz, shape `[T, C]`, and deterministic extrapolation rejection.

## Cold-Read Enforcement Contract
- `physicaldb.plan(...)` is hot-catalog only for one matched file window: it returns selected pose/IMU/media row groups, authorized byte spans, egress totals, pruning diagnostics, and gap diagnostics without opening cold source data. `physicaldb.plan_batch(...)` applies the same contract across multiple matched `(robot_id, session_id, file_uri)` windows and blocks on aggregate egress before materialization.
- `physicaldb.query(..., enforce_ranges=True, footer_allowance_bytes=..., manifest_out=...)` materializes pose/IMU tensors through Rust and wraps the Parquet object-store reader with `RangeAuditor`; any post-footer data read outside selected row-group spans fails before data is returned. Passing `materialize_media=True, media_out=...` also materializes selected camera frames with the same enforcement model. `physicaldb.query_batch(...)` returns one `QueryResult` per window instead of concatenating sessions into a false continuous tensor.
- Footer/metadata reads are allowed only during Parquet metadata loading, bounded by the object size and a configurable footer allowance defaulting to 16 MiB, and are reported separately from materialized row-group reads.
- JSON seek manifests include plan inputs, selected row groups, authorized spans, actual cold reads, actual authorized bytes, footer allowance, footer bytes, largest metadata read, max footer offsets, materialized bytes, media planned/materialized bytes, enforcement state, and violations. Batch manifests include per-window manifests plus aggregate authorized bytes, actual cold reads, footer bytes, range violations, and egress state.
- CLI equivalents are `robotics tensor parquet-row-groups --enforce-ranges --footer-allowance-bytes ... --manifest-out ...`, `robotics tensor imu-parquet-row-groups --enforce-ranges --footer-allowance-bytes ... --manifest-out ...`, `robotics media camera-row-groups --enforce-ranges --footer-allowance-bytes ... --manifest-out ...`, and `robotics catalog explain --catalog-db ... --predicate ...`.

## Validation Scripts
- `scripts/bench_catalog_scale.py` is the deterministic hot-catalog gate; it builds or reuses a fake DuckDB catalog, benchmarks one or more predicates, prints p50/p95/prune/authorized-byte metrics, and can fail on threshold flags for CI smoke.
- `scripts/prove_euroc_hot_catalog.py` is the repeatable real-data proof path; it builds or reuses EuRoC pose/IMU/camera Parquet, hot catalogs, and a Hilbert DuckDB catalog, optionally uploads generated Parquet to S3-compatible storage, runs representative behavior predicates through `physicaldb.plan()` and enforced `physicaldb.query(..., materialize_media=True)`, and writes a compact JSON proof manifest.
- `scripts/validate_euroc_vicon_room1.py` is the opt-in real multi-session EuRoC Vicon Room 1 validation; it ingests `V1_01_easy`, `V1_02_medium`, and `V1_03_difficult`, builds per-session pose/IMU/cam0 catalogs plus combined representative catalogs, runs `physicaldb.plan_batch()`/`query_batch()` with enforced cold reads and camera materialization, verifies tensor/interpolation/range/egress invariants, and writes `data/validation/euroc_vicon_room1/report.json`.
- `scripts/smoke_camera_materialization.py` is the no-download camera materialization smoke; it builds a tiny EuRoC-style pose/IMU/camera fixture, creates catalogs and DuckDB, then verifies `physicaldb.query(..., materialize_media=True, enforce_ranges=True)` writes selected camera frames and a manifest.
- `scripts/smoke_s3_pose_imu_media.py` is the compact live S3/MinIO smoke; it generates small pose/IMU fixtures, uploads pose/IMU/media objects, verifies enforced pose+IMU materialization, and verifies media egress blocks before materialization.
- `scripts/validate_s3_large_ranges.py` is the larger live S3/MinIO auditor check; it generates configurable EuRoC-style pose/IMU fixtures, validates first/middle/full-file enforced queries, and supports `--footer-allowance-bytes 1` as an expected failure probe for debuggable range-audit errors.

## Validation Log
- `cargo fmt --check`: passing.
- `cargo test --target-dir /tmp/robotics-target`: passing, 54 Rust tests, including object-store upload coverage, enforced range-audited Parquet materialization, footer allowance rejection, manifest round-trip coverage, camera media materialization, media catalog indexing, and `catalog duckdb-build`/`catalog fake-duckdb` CLI coverage when Python DuckDB is installed.
- `cargo test -p robotics-cli --target-dir /tmp/robotics-target`: passing, 7 CLI tests, including catalog build/tensor commands, DuckDB catalog creation, fake DuckDB catalog generation, demo tensor/byte planning, and selected camera frame materialization.
- `cargo test -p robotics-ingest --target-dir /tmp/robotics-target`: passing, 13 ingestion tests, including EuRoC groundtruth/IMU/camera coverage and selected IMU row-group loading.
- `cargo clippy --all-targets --target-dir /tmp/robotics-target -- -D warnings`: passing.
- `env CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 -m pytest tests/test_physicaldb.py`: passing, 7 Python tests with DuckDB 1.5.2, including synchronized EuRoC-style pose+IMU tensor output from catalog URI-driven Rust row-group materialization, persistent DuckDB catalog DB query parity, constrained behavioral predicates, coarse spatial pruning explain diagnostics, enforced pose+IMU/camera cold-read accounting, proof and batch manifests, camera row-group egress accounting/materialization, multi-session `plan_batch()`/`query_batch()` windows, pre-materialization egress blocking, and default temporal gap rejection.
- `env CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/bench_catalog_scale.py --sessions 100000 --iterations 100 --warmup 3 --max-p95-ms 10`: passed against a 100k-row-group fake DuckDB catalog with Hilbert defaults and staged hot planning; p95 by predicate class was tight spatial `3.148 ms`, broad spatial `4.809 ms`, mixed spatial/velocity `5.659 ms`, velocity-only `5.640 ms`, time-bounded `3.075 ms`, and mixed spatial/time/velocity `4.936 ms`.
- `env CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/prove_euroc_hot_catalog.py --input vicon_room1/V1_01_easy/V1_01_easy.zip --work-dir data/proofs/euroc_v1_01_easy --iterations 20 --max-p95-ms 50 --manifest-out data/proofs/euroc_v1_01_easy/proof.json --rebuild`: passed on the real local EuRoC `V1_01_easy` zip with Hilbert catalog planning p50 `2.788 ms`, p95 `3.335 ms`, 58 candidate and 1 matched pose row group, pose/IMU/media matched row groups `1/1/4`, Hilbert/exact-spatial/velocity/time pruning `0/0/0/57`, authorized bytes pose/IMU/media `55,287/61,945/28,975,054`, authorized total bytes `29,092,286`, actual cold-read bytes `29,325,181`, footer bytes `232,895`, egress probe blocked before media materialization, tensor shape `[75, 9]`, and 80 materialized camera frames.
- `env CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/validate_euroc_vicon_room1.py --input-root vicon_room1 --output-root data/validation/euroc_vicon_room1 --iterations 5 --rebuild`: passed on the three local EuRoC Vicon Room 1 zips using `plan_batch()`/`query_batch()` with representative combined catalogs. Planning p50 was `4.235 ms` and p95 was `14.360 ms`; `plan_batch()` returned 3 windows, one each for `V1_01_easy`, `V1_02_medium`, and `V1_03_difficult`, with pose/IMU/media matched row groups `3/3/10`. Per-session tensor shapes were `[75, 9]`, `[75, 9]`, and `[75, 9]`; authorized bytes pose/IMU/media were `165,683/206,243/72,437,661`, authorized total bytes were `72,809,587`, actual cold-read bytes were `73,350,797`, footer bytes were `541,210`, range violations were `0`, and the low-budget egress probe blocked before media materialization without creating output. Camera frames materialized per session were `60`, `60`, and `80`.
- `env AWS_ACCESS_KEY_ID=robotics AWS_SECRET_ACCESS_KEY=robotics123 AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ENDPOINT_URL_S3=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_REGION=us-east-1 AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/prove_euroc_hot_catalog.py --input vicon_room1/V1_01_easy/V1_01_easy.zip --work-dir data/proofs/euroc_v1_01_easy_s3 --s3-prefix s3://robotics/euroc-proof/V1_01_easy --iterations 20 --max-p95-ms 50 --manifest-out data/proofs/euroc_v1_01_easy_s3/proof.json`: passed against live MinIO/S3-compatible URIs with Hilbert catalog planning p50 `2.842 ms`, p95 `5.145 ms`, 58 candidate and 1 matched pose row group, pose/IMU/media matched row groups `1/1/4`, Hilbert/exact-spatial/velocity/time pruning `0/0/0/57`, authorized bytes pose/IMU/media `55,287/61,945/28,975,054`, authorized total bytes `29,092,286`, actual cold-read bytes `29,325,181`, footer bytes `232,895`, egress probe blocked before media materialization, tensor shape `[75, 9]`, and 80 materialized camera frames.
- `PYTHONPATH=python python3 scripts/prove_euroc_hot_catalog.py --input /tmp/euroc_proof_fixture/euroc --work-dir /tmp/euroc_hot_proof --iterations 5 --manifest-out /tmp/euroc_hot_proof/proof_reuse.json`: passed on the tiny EuRoC-style fixture with Hilbert catalog planning p50 `2.472 ms`, p95 `11.703 ms`, 1 candidate and matched pose row group, authorized bytes pose/IMU/media `619/560/312`, actual cold-read bytes `7,477`, footer bytes `5,986`, egress probe blocked before media materialization, tensor shape `[16, 9]`, and 2 materialized camera frames.
- `cargo run -p robotics-cli -- demo fake`: matched 1 synthetic window, planned 1 range read, selected 65,536 bytes, returned tensor shape `[31, 10]`.
- `cargo run -p robotics-cli -- demo`: wrote 3 synthetic Parquet row groups, indexed 3 row groups, applied default velocity/bbox predicates, matched 1 window `[0, 480000000]` ns, executed 1 range read, transferred 1,551 bytes from an 11,253 byte Parquet file, loaded 25 tensor source rows from the selected row group, returned tensor shape `[15, 10]`.
- `cargo run -p robotics-cli -- demo --tensor-out data/tensor/demo`: wrote `data/tensor/demo.values.npy` with shape `[15, 10]` and `data/tensor/demo.timestamps_ns.npy` with shape `[15]` for NumPy/PyTorch interop.
- `cargo run -p robotics-cli -- ingest synthetic-parquet --out data/parquet/synthetic/session.parquet --row-group-rows 25 --hz 50 --duration-ns 1000000000`: wrote 3 row groups.
- `cargo run -p robotics-cli -- index parquet --input data/parquet/synthetic/session.parquet`: indexed 3 row groups, 3,696 selected bytes, first window `[0, 480000000]` ns.
- `cargo run -p robotics-cli -- catalog build --input data/parquet/synthetic/session.parquet --out data/catalog/fleet_metadata.parquet`: writes a hot catalog Parquet file with row-group byte offsets, time ranges, trajectory bounds, velocity statistics, nominal cadence, and temporal gap metadata.
- `cargo run -p robotics-cli -- catalog build-media --input data/parquet/camera/cam0.parquet --out data/catalog/cam0_media.parquet --modality camera --stream-id cam0`: writes a hot media catalog Parquet file with time ranges, stream identity, row-group byte offsets, and optional spatial bounds.
- `cargo run -p robotics-cli -- catalog duckdb-build --pose-catalog data/catalog/fleet_metadata.parquet --imu-catalog data/catalog/euroc_v1_01_easy_imu.parquet --media-catalog data/catalog/cam0_media.parquet --out data/catalog/fleet.duckdb`: builds a persistent DuckDB catalog with `pose_row_groups`, `imu_row_groups`, and `media_row_groups` tables plus robot/time/spatial-tile/velocity/media stream indexes for DB-backed behavioral pruning and egress planning.
- `cargo run -p robotics-cli -- catalog fake-duckdb --sessions 100000 --out data/catalog/fake_fleet.duckdb`: builds a synthetic persistent DuckDB catalog for hot-index latency and pruning demos without requiring source Parquet logs.
- `cargo run -p robotics-cli -- range-read parquet --input data/parquet/synthetic/session.parquet --limit 1`: indexed 3 row groups, executed 1 range read, transferred 1,550 bytes from an 11,238 byte Parquet file.
- `cargo run -p robotics-cli -- tensor parquet-row-groups --input data/parquet/synthetic/session.parquet --row-groups 0 --start-ts-ns 0 --end-ts-ns 480000000 --audit-ranges 0:byte_offset:byte_length --enforce-ranges --manifest-out data/tensor/query.manifest.json --out data/tensor/query`: materializes selected row groups through the audited object-store Parquet reader into tensor `.npy` files, rejects unauthorized cold reads, counts footer bytes separately, writes a JSON manifest, and reports pose gap/null/quaternion diagnostics.
- `cargo run -p robotics-cli -- ingest synthetic-mcap --out data/mcap/synthetic/session.mcap --topic /pose --hz 20 --duration-ns 1000000000`: wrote 21 JSON pose MCAP messages.
- `cargo run -p robotics-cli -- ingest mcap-json --input data/mcap/synthetic/session.mcap --out data/parquet/mcap/session.parquet --topic /pose --row-group-rows 10`: converted 21 MCAP samples into 3 Parquet row groups.
- `cargo run -p robotics-cli -- ingest mcap-pose --input path/to/poses.mcap --out data/parquet/mcap/pose.parquet --topic /pose`: supports JSON pose payloads plus ROS1/ROS2 `geometry_msgs/PoseStamped`, `geometry_msgs/TransformStamped`, and `nav_msgs/Odometry` pose messages.
- `cargo run -p robotics-cli -- ingest euroc-camera --input vicon_room1/V1_01_easy --out data/parquet/euroc/V1_01_easy/cam0.parquet --stream-id cam0 --session-id V1_01_easy`: converts EuRoC camera `data.csv` plus image files into camera Parquet with raw `camera_bytes`.
- `cargo run -p robotics-cli -- media camera-row-groups --input data/parquet/euroc/V1_01_easy/cam0.parquet --row-groups 0 --out data/media/euroc_cam0 --audit-ranges 0:byte_offset:byte_length --enforce-ranges --manifest-out data/media/euroc_cam0.manifest.json`: materializes selected camera frames through the audited object-store Parquet reader and writes a JSON frame/seek manifest.
- `cargo run -p robotics-cli -- range-read parquet --input data/parquet/mcap/session.parquet --limit 1`: transferred 962 bytes from a 10,064 byte converted MCAP Parquet file.
- `cargo run -p robotics-cli -- ingest kitti-oxts --input /tmp/robotics_kitti_smoke --out data/parquet/kitti/oxts.parquet --row-group-rows 1 --robot-id kitti_car --session-id smoke_drive`: converted 2 KITTI-style OXTS packets into 2 Parquet row groups.
- `cargo run -p robotics-cli -- range-read parquet --input data/parquet/kitti/oxts.parquet --limit 1`: transferred 586 bytes from a 6,596 byte KITTI OXTS Parquet file.
- `cargo run -p robotics-cli -- ingest nuscenes-ego --input path/to/v1.0-mini --out data/parquet/nuscenes/ego_pose.parquet`: converts nuScenes `ego_pose.json` into pose Parquet with finite-difference velocity.
- `cargo run -p robotics-cli --target-dir /tmp/robotics-target -- ingest euroc-groundtruth --input vicon_room1/V1_01_easy --out data/parquet/euroc/V1_01_easy/pose.parquet --session-id V1_01_easy --row-group-rows 500`: converted 28,712 EuRoC groundtruth pose rows into 58 Parquet row groups.
- `cargo run -p robotics-cli --target-dir /tmp/robotics-target -- ingest euroc-imu --input vicon_room1/V1_01_easy --out data/parquet/euroc/V1_01_easy/imu.parquet --session-id V1_01_easy --row-group-rows 2000`: converted 29,120 EuRoC IMU rows into 15 Parquet row groups.
- `cargo run -p robotics-cli --target-dir /tmp/robotics-target -- catalog build-imu --input data/parquet/euroc/V1_01_easy/imu.parquet --out data/catalog/euroc_v1_01_easy_imu.parquet`: indexed 15 EuRoC IMU row groups and 905,059 row-group bytes.
- `cargo run -p robotics-cli --target-dir /tmp/robotics-target -- range-read parquet --input data/parquet/euroc/V1_01_easy/pose.parquet --limit 1`: indexed 58 EuRoC pose row groups and transferred 46,748 bytes from a 3,283,443 byte Parquet file.
- `cargo run -p robotics-cli --target-dir /tmp/robotics-target -- tensor parquet-row-groups --input data/parquet/euroc/V1_01_easy/pose.parquet --row-groups 0 --start-ts-ns 1403715274302142976 --end-ts-ns 1403715276797143040 --hz 30 --out data/tensor/euroc_v1_01_easy_pose`: materialized 500 EuRoC source rows into tensor shape `[75, 10]`.
- `PYTHONPATH=python python3 -c "from physicaldb import query; r=query(catalog='data/catalog/euroc_v1_01_easy.parquet', robot_id='mav0', min_velocity=0.1, channels=('pos_xyz','rot_wxyz','vel_xyz','imu_accel','imu_gyro'), target_hz=30.0, imu_catalog='data/catalog/euroc_v1_01_easy_imu.parquet', limit=1, enforce_ranges=True, manifest_out='data/tensor/euroc_manifest.json'); print(r.tensor.shape); print(r.diagnostics.log_lines())"`: selects velocity-matching EuRoC pose row groups and time-overlapping IMU row groups from hot catalog URIs, materializes pose+IMU through Rust enforced selected-row-group paths, returns synchronized pose+IMU tensors, writes a manifest, and surfaces gap/null/quaternion/extrapolation diagnostics with planned and actual cold-read accounting.
- `PYTHONPATH=python python3 scripts/bench_catalog_scale.py`: builds or reuses a fake persistent DuckDB catalog representing 1,000+ hours, runs repeated `physicaldb.plan()` behavioral predicates without materialization, and reports p50/p95 planning latency, candidate/matched row groups, prune counts, prune ratio, authorized bytes, manifest-ready row-group plans, and optional threshold failures.
- `env AWS_ACCESS_KEY_ID=robotics AWS_SECRET_ACCESS_KEY=robotics123 AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_REGION=us-east-1 AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/validate_s3_large_ranges.py --duration-sec 30 --pose-row-group-rows 250 --imu-row-group-rows 1000`: generated larger pose/IMU Parquet fixtures, uploaded them to MinIO, built S3 URI catalogs, and passed enforced range validation over first-row-group, middle-window, and full-file queries with zero violations; the full-file query reported 13 pose row groups, 7 IMU row groups, 13 media row groups, 238,360 authorized total bytes, 176,397 planned materialized bytes, 249 actual cold reads, 205,455 actual cold-read bytes, 176,397 actual authorized bytes, 29,058 footer bytes, and 21,161 largest metadata read.
- `env AWS_ACCESS_KEY_ID=robotics AWS_SECRET_ACCESS_KEY=robotics123 AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_REGION=us-east-1 AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/validate_s3_large_ranges.py --duration-sec 2 --pose-row-group-rows 100 --imu-row-group-rows 400 --footer-allowance-bytes 1`: expected failure path verified with a concise `FAIL:` message naming the URI, requested footer range, `footer_allowance_bytes=1`, and authorized row-group span.
- `env AWS_ACCESS_KEY_ID=robotics AWS_SECRET_ACCESS_KEY=robotics123 AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ENDPOINT_URL_S3=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_REGION=us-east-1 AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false ./target/debug/robotics validate s3-parquet --input data/parquet/synthetic/session.parquet --uri s3://robotics/session.parquet --limit 1`: live MinIO validation passed, indexed 3 row groups, executed 1 S3 range read, transferred 1,550 bytes from an 11,238 byte Parquet file.
- `env AWS_ACCESS_KEY_ID=robotics AWS_SECRET_ACCESS_KEY=robotics123 AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_REGION=us-east-1 AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/smoke_s3_pose_imu_media.py`: opt-in live S3/MinIO smoke passed with tensor shape `[3, 9]`, pose/IMU/media matched row groups `2/2/2`, selected bytes `1184/882/1184`, total selected bytes `3250`, planned materialized read bytes `2066`, actual cold reads `50`, actual cold-read bytes `9072`, footer bytes `7006`, range audit/enforcement passed, manifest output written, and low-budget media egress blocked before materialization.

## External Sources To Integrate
- nuScenes: https://arxiv.org/abs/1903.11027
- KITTI raw data: https://www.cvlibs.net/datasets/kitti/raw_data.php?type=calibration
- Hilti SLAM: https://www.hilti-challenge.com/dataset-2021
- CARLA: https://arxiv.org/abs/1711.03938
- MCAP: https://mcap.dev/
- Apache Parquet row groups: https://parquet.apache.org/docs/concepts/
- DuckDB/DataFusion/S3 partial reads remain candidate implementation options for catalog and object-store layers.

## Open Risks
- Parquet row-group byte ranges are currently derived from min/max column chunk byte ranges; this is validated against local files and live MinIO, but should still be tested against production S3 objects.
- Enforced materialization now records actual Parquet object-store byte ranges and supports configurable footer allowance; the default 16 MiB allowance passed generated large MinIO fixtures but should still be validated against real production S3 objects.
- DuckDB integration now supports persistent hot catalogs with tile fallback and Hilbert/time columns for staged coarse spatial pruning; the remaining risk is production-data selectivity tuning rather than absence of a Hilbert path.
- Temporal gap metadata is currently row-group-level with the largest gap interval surfaced; multiple interval-level gap rows can be added later if needed for finer exclusion.
- Pose and IMU materialization now decode selected Parquet row groups through Rust, including object-store URI support; live MinIO pose+IMU+media smoke coverage exists, while production S3 validation for large pose+IMU objects is still pending.
- Camera media row groups are now planned, egress-gated, and materialized through enforced range-audited reads; LiDAR media extraction remains deferred.
- MCAP pose ingestion supports JSON pose, common ROS pose schemas, ROS1 binary, and basic ROS2 CDR; CARLA/Hilti bag variants may still need dataset-specific topic/schema mapping.
- KITTI OXTS ingestion currently covers pose/GPS/velocity packets only; camera/LiDAR calibration and sensor frame transforms are not yet modeled.
- nuScenes ingestion currently covers `ego_pose.json`; sample, sensor calibration, camera/LiDAR annotations, and scene joins are not yet modeled.
- EuRoC ingestion currently covers groundtruth pose, IMU CSV tables, and camera byte extraction; calibration joins are not yet modeled.
- Zero-copy DLPack/PyTorch interop is deferred; current Python interop uses NumPy `.npy` files loaded by `physicaldb.query()` and optionally wrapped with `torch.from_numpy`.
- Public datasets vary substantially in schema and calibration conventions; initial demo should use synthetic or MCAP/CARLA data with controlled ground truth.
