# Robotics Fleet Query Engine

## Goal
Build a Rust-first robotics data query engine that indexes robotics sessions, prunes candidate row groups, performs byte-accounted reads, and returns uniformly sampled tensor-shaped windows for model training and evaluation.

## Current Status
| Area | Status | Notes |
| --- | --- | --- |
| Rust workspace | Done for M0 | Core crates compile and pass tests/clippy. |
| Catalog generation/query | Done for M1 | Fake catalog, predicate filtering, pose/IMU/media Parquet metadata indexing, and explicit row-group temporal gap metadata implemented in Rust. |
| Row-group byte accounting | Done for M2 | Local and object-store range readers implemented with planned/transferred byte accounting, object-store upload support for live smoke setup, and enforced row-group range auditing around actual Parquet materialization reads. |
| Tensorization | Done for M4 | Quaternion slerp, pose/IMU resampling, tensor-shaped buffers, extrapolation checks, selected local/object-store Parquet row-group loading, and NumPy `.npy` export implemented. |
| Dataset ingestion | In progress | Synthetic pose, synthetic Parquet, JSON-pose MCAP, ROS pose MCAP, KITTI raw OXTS, nuScenes ego-pose, and EuRoC groundtruth/IMU CSV ingestion implemented; richer public-dataset calibration adapters remain. |
| Arrow/Parquet/S3 | Done for M2 | Arrow/Parquet, object-store upload/range-read plumbing, S3 URI parsing, and live MinIO validation are implemented. |
| DuckDB/Python facade | Prototype complete for enforced cold pose+IMU+media planning | Rust can write hot pose, IMU, and media metadata catalog Parquet files, build a persistent DuckDB catalog DB with derived spatial tile keys, and generate large fake catalog DBs; Python `physicaldb.plan()` returns a plan-only cold-seek contract with authorized bytes and pruning diagnostics, while `physicaldb.query(..., enforce_ranges=True)` uses that same plan to enforce egress/gap gates and validate actual Parquet cold range reads before returning selected pose+IMU tensors. |
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
   - Status: JSON-pose MCAP, common ROS pose MCAP schemas, KITTI raw OXTS, nuScenes ego-pose, and EuRoC groundtruth/IMU CSV paths complete; CARLA/Hilti dataset-specific calibration conventions remain.
5. **M4: Demo loop**
   - Ingest one public/synthetic session.
   - Query velocity/bounding-box behavior.
   - Return `[T, C]` output and report row groups matched, bytes read, and wall-clock time.
   - Status: complete for synthetic Parquet sessions and EuRoC pose+IMU plus media planning, including tensor loading from selected pose and IMU row groups through local/object-store catalog URIs, hot pose/IMU/media catalog Parquet output, persistent DuckDB catalog DB construction, coarse spatial tile pruning, Python/DuckDB velocity/spatial predicate pruning, plan-only cold-seek contracts, default gap rejection, enforced cold-read auditing, media egress gates, optional NumPy/PyTorch loading, JSON seek manifests, and caller-visible correctness/explain diagnostics.

## Canonical Data Model
Sensor rows use nanosecond timestamps, `robot_id`, `session_id`, pose position, quaternion orientation, and linear velocity. Catalog rows summarize row-group/session windows with min/max timestamps, bounding boxes, velocity range, URI, row-group id, byte offset, byte length, nominal cadence, and explicit temporal gap markers.

## Benchmark Targets
- Fake 50k-entry catalog predicate query: sub-10 ms p95 on a laptop.
- Demo query: selected bytes should be close to matched row-group sizes, not full dataset size.
- Tensor output: uniformly spaced timestamps at target Hz, shape `[T, C]`, and deterministic extrapolation rejection.

## Cold-Read Enforcement Contract
- `physicaldb.plan(...)` is hot-catalog only: it returns selected pose/IMU/media row groups, authorized byte spans, egress totals, pruning diagnostics, and gap diagnostics without opening cold source data.
- `physicaldb.query(..., enforce_ranges=True, footer_allowance_bytes=..., manifest_out=...)` materializes pose/IMU tensors through Rust and wraps the Parquet object-store reader with `RangeAuditor`; any post-footer data read outside selected row-group spans fails before data is returned.
- Footer/metadata reads are allowed only during Parquet metadata loading, bounded by the object size and a configurable footer allowance defaulting to 16 MiB, and are reported separately from materialized row-group reads.
- JSON seek manifests include plan inputs, selected row groups, authorized spans, actual cold reads, actual authorized bytes, footer allowance, footer bytes, largest metadata read, max footer offsets, materialized bytes, media planned bytes, enforcement state, and violations; media remains planning/egress-gated only.
- CLI equivalents are `robotics tensor parquet-row-groups --enforce-ranges --footer-allowance-bytes ... --manifest-out ...`, `robotics tensor imu-parquet-row-groups --enforce-ranges --footer-allowance-bytes ... --manifest-out ...`, and `robotics catalog explain --catalog-db ... --predicate ...`.

## Validation Scripts
- `scripts/bench_catalog_scale.py` is the deterministic hot-catalog gate; it builds or reuses a fake DuckDB catalog, benchmarks one or more predicates, prints p50/p95/prune/authorized-byte metrics, and can fail on threshold flags for CI smoke.
- `scripts/smoke_s3_pose_imu_media.py` is the compact live S3/MinIO smoke; it generates small pose/IMU fixtures, uploads pose/IMU/media objects, verifies enforced pose+IMU materialization, and verifies media egress blocks before materialization.
- `scripts/validate_s3_large_ranges.py` is the larger live S3/MinIO auditor check; it generates configurable EuRoC-style pose/IMU fixtures, validates first/middle/full-file enforced queries, and supports `--footer-allowance-bytes 1` as an expected failure probe for debuggable range-audit errors.

## Validation Log
- `cargo fmt --check`: passing.
- `cargo test --target-dir /tmp/robotics-target`: passing, 50 Rust tests, including object-store upload coverage, enforced range-audited Parquet materialization, footer allowance rejection, manifest round-trip coverage, media catalog indexing, and `catalog duckdb-build`/`catalog fake-duckdb` CLI coverage when Python DuckDB is installed.
- `cargo test -p robotics-ingest --target-dir /tmp/robotics-target`: passing, 12 ingestion tests, including EuRoC groundtruth/IMU CSV coverage and selected IMU row-group loading.
- `cargo clippy --all-targets --target-dir /tmp/robotics-target -- -D warnings`: passing.
- `env CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 -m pytest tests/test_physicaldb.py`: passing, 5 Python tests with DuckDB 1.5.2, including synchronized EuRoC-style pose+IMU tensor output from catalog URI-driven Rust row-group materialization, persistent DuckDB catalog DB query parity, constrained behavioral predicates, coarse spatial pruning explain diagnostics, enforced pose+IMU cold-read accounting, manifest output, media row-group egress accounting, pre-materialization egress blocking, and default temporal gap rejection.
- `env CARGO_TARGET_DIR=/tmp/robotics-target PYTHONPATH=python python3 scripts/bench_catalog_scale.py`: built a 2,000-hour fake DuckDB catalog and measured plan-only hot-catalog behavior across 50 iterations and 3 predicates; predicate 1 p50/p95 `14.833/19.657 ms`, prune ratio `0.650000`, authorized bytes `2,293,760`; predicate 2 p50/p95 `12.748/18.683 ms`, prune ratio `0.780000`, authorized bytes `1,441,792`; predicate 3 p50/p95 `14.685/20.024 ms`, prune ratio `0.600000`, authorized bytes `2,621,440`.
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
- DuckDB integration now supports a persistent catalog DB with coarse fixed-grid spatial tile keys, but there is no true R-tree/Hilbert index or SQL extension yet.
- Temporal gap metadata is currently row-group-level with the largest gap interval surfaced; multiple interval-level gap rows can be added later if needed for finer exclusion.
- Pose and IMU materialization now decode selected Parquet row groups through Rust, including object-store URI support; live MinIO pose+IMU+media smoke coverage exists, while production S3 validation for large pose+IMU objects is still pending.
- Camera/LiDAR media row groups are currently planned and egress-gated only; media byte materialization is intentionally deferred.
- MCAP pose ingestion supports JSON pose, common ROS pose schemas, ROS1 binary, and basic ROS2 CDR; CARLA/Hilti bag variants may still need dataset-specific topic/schema mapping.
- KITTI OXTS ingestion currently covers pose/GPS/velocity packets only; camera/LiDAR calibration and sensor frame transforms are not yet modeled.
- nuScenes ingestion currently covers `ego_pose.json`; sample, sensor calibration, camera/LiDAR annotations, and scene joins are not yet modeled.
- EuRoC ingestion currently covers groundtruth pose and IMU CSV tables; camera metadata/image indexing and calibration joins are not yet modeled.
- Zero-copy DLPack/PyTorch interop is deferred; current Python interop uses NumPy `.npy` files loaded by `physicaldb.query()` and optionally wrapped with `torch.from_numpy`.
- Public datasets vary substantially in schema and calibration conventions; initial demo should use synthetic or MCAP/CARLA data with controlled ground truth.
