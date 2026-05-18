# Robotics Fleet Query Engine

## Goal
Build a Rust-first robotics data query engine that indexes robotics sessions, prunes candidate row groups, performs byte-accounted reads, and returns uniformly sampled tensor-shaped windows for model training and evaluation.

## Current Status
| Area | Status | Notes |
| --- | --- | --- |
| Rust workspace | Done for M0 | Core crates compile and pass tests/clippy. |
| Catalog generation/query | Done for M1 | Fake catalog, predicate filtering, and Parquet metadata indexing implemented in Rust. |
| Row-group byte accounting | Done for M2 | Local and object-store range readers implemented with planned/transferred byte accounting. |
| Tensorization | Done for M4 | Quaternion slerp, resampling, tensor-shaped buffers, extrapolation checks, selected Parquet row-group loading, and NumPy `.npy` export implemented. |
| Dataset ingestion | In progress | Synthetic pose, synthetic Parquet, JSON-pose MCAP, ROS pose MCAP, KITTI raw OXTS, and nuScenes ego-pose ingestion implemented; richer public-dataset calibration adapters remain. |
| Arrow/Parquet/S3 | Done for M2 | Arrow/Parquet, object-store range-read plumbing, S3 URI parsing, and live MinIO validation are implemented. |
| DuckDB/Python facade | Prototype complete for pose | Rust can write a hot metadata catalog Parquet file; Python `physicaldb.query()` uses DuckDB to prune row groups, enforces an egress byte gate, and returns NumPy/PyTorch-shaped tensors through the Rust row-group tensor path. |
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
   - Status: complete for local object-store backend and live MinIO validation.
4. **M3: Public dataset ingestion**
   - MCAP ingestion first.
   - KITTI raw conversion next.
   - nuScenes mini ingestion after core path is stable.
   - Status: JSON-pose MCAP, common ROS pose MCAP schemas, KITTI raw OXTS, and nuScenes ego-pose table paths complete; CARLA/Hilti dataset-specific calibration conventions remain.
5. **M4: Demo loop**
   - Ingest one public/synthetic session.
   - Query velocity/bounding-box behavior.
   - Return `[T, C]` output and report row groups matched, bytes read, and wall-clock time.
   - Status: complete for synthetic Parquet sessions, including tensor loading from selected row groups, hot catalog Parquet output, and Python/DuckDB query facade with optional NumPy/PyTorch loading.

## Canonical Data Model
Sensor rows use nanosecond timestamps, `robot_id`, `session_id`, pose position, quaternion orientation, and linear velocity. Catalog rows summarize row-group/session windows with min/max timestamps, bounding boxes, velocity range, URI, row-group id, byte offset, and byte length.

## Benchmark Targets
- Fake 50k-entry catalog predicate query: sub-10 ms p95 on a laptop.
- Demo query: selected bytes should be close to matched row-group sizes, not full dataset size.
- Tensor output: uniformly spaced timestamps at target Hz, shape `[T, C]`, and deterministic extrapolation rejection.

## Validation Log
- `cargo fmt --check`: passing.
- `cargo test`: passing, 26 Rust tests.
- `cargo clippy --all-targets -- -D warnings`: passing.
- `PYTHONPATH=python python3 -m pytest tests/test_physicaldb.py`: passing, 2 Python tests with DuckDB 1.5.2.
- `cargo run -p robotics-cli -- demo fake`: matched 1 synthetic window, planned 1 range read, selected 65,536 bytes, returned tensor shape `[31, 10]`.
- `cargo run -p robotics-cli -- demo`: wrote 3 synthetic Parquet row groups, indexed 3 row groups, applied default velocity/bbox predicates, matched 1 window `[0, 480000000]` ns, executed 1 range read, transferred 1,551 bytes from an 11,253 byte Parquet file, loaded 25 tensor source rows from the selected row group, returned tensor shape `[15, 10]`.
- `cargo run -p robotics-cli -- demo --tensor-out data/tensor/demo`: wrote `data/tensor/demo.values.npy` with shape `[15, 10]` and `data/tensor/demo.timestamps_ns.npy` with shape `[15]` for NumPy/PyTorch interop.
- `cargo run -p robotics-cli -- ingest synthetic-parquet --out data/parquet/synthetic/session.parquet --row-group-rows 25 --hz 50 --duration-ns 1000000000`: wrote 3 row groups.
- `cargo run -p robotics-cli -- index parquet --input data/parquet/synthetic/session.parquet`: indexed 3 row groups, 3,696 selected bytes, first window `[0, 480000000]` ns.
- `cargo run -p robotics-cli -- catalog build --input data/parquet/synthetic/session.parquet --out data/catalog/fleet_metadata.parquet`: writes a hot catalog Parquet file with row-group byte offsets, time ranges, trajectory bounds, and velocity statistics.
- `cargo run -p robotics-cli -- range-read parquet --input data/parquet/synthetic/session.parquet --limit 1`: indexed 3 row groups, executed 1 range read, transferred 1,550 bytes from an 11,238 byte Parquet file.
- `cargo run -p robotics-cli -- tensor parquet-row-groups --input data/parquet/synthetic/session.parquet --row-groups 0 --start-ts-ns 0 --end-ts-ns 480000000 --out data/tensor/query`: materializes selected row groups into tensor `.npy` files without loading unrelated row groups.
- `cargo run -p robotics-cli -- ingest synthetic-mcap --out data/mcap/synthetic/session.mcap --topic /pose --hz 20 --duration-ns 1000000000`: wrote 21 JSON pose MCAP messages.
- `cargo run -p robotics-cli -- ingest mcap-json --input data/mcap/synthetic/session.mcap --out data/parquet/mcap/session.parquet --topic /pose --row-group-rows 10`: converted 21 MCAP samples into 3 Parquet row groups.
- `cargo run -p robotics-cli -- ingest mcap-pose --input path/to/poses.mcap --out data/parquet/mcap/pose.parquet --topic /pose`: supports JSON pose payloads plus ROS1/ROS2 `geometry_msgs/PoseStamped`, `geometry_msgs/TransformStamped`, and `nav_msgs/Odometry` pose messages.
- `cargo run -p robotics-cli -- range-read parquet --input data/parquet/mcap/session.parquet --limit 1`: transferred 962 bytes from a 10,064 byte converted MCAP Parquet file.
- `cargo run -p robotics-cli -- ingest kitti-oxts --input /tmp/robotics_kitti_smoke --out data/parquet/kitti/oxts.parquet --row-group-rows 1 --robot-id kitti_car --session-id smoke_drive`: converted 2 KITTI-style OXTS packets into 2 Parquet row groups.
- `cargo run -p robotics-cli -- range-read parquet --input data/parquet/kitti/oxts.parquet --limit 1`: transferred 586 bytes from a 6,596 byte KITTI OXTS Parquet file.
- `cargo run -p robotics-cli -- ingest nuscenes-ego --input path/to/v1.0-mini --out data/parquet/nuscenes/ego_pose.parquet`: converts nuScenes `ego_pose.json` into pose Parquet with finite-difference velocity.
- `env AWS_ACCESS_KEY_ID=robotics AWS_SECRET_ACCESS_KEY=robotics123 AWS_ENDPOINT=http://127.0.0.1:9000 AWS_ENDPOINT_URL_S3=http://127.0.0.1:9000 AWS_ALLOW_HTTP=true AWS_REGION=us-east-1 AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false ./target/debug/robotics validate s3-parquet --input data/parquet/synthetic/session.parquet --uri s3://robotics/session.parquet --limit 1`: live MinIO validation passed, indexed 3 row groups, executed 1 S3 range read, transferred 1,550 bytes from an 11,238 byte Parquet file.

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
- DuckDB integration currently lives in the Python facade over a hot catalog Parquet file; there is no persisted DuckDB database, R-tree/Hilbert spatial index, or SQL extension yet.
- MCAP pose ingestion supports JSON pose, common ROS pose schemas, ROS1 binary, and basic ROS2 CDR; CARLA/Hilti bag variants may still need dataset-specific topic/schema mapping.
- KITTI OXTS ingestion currently covers pose/GPS/velocity packets only; camera/LiDAR calibration and sensor frame transforms are not yet modeled.
- nuScenes ingestion currently covers `ego_pose.json`; sample, sensor calibration, camera/LiDAR annotations, and scene joins are not yet modeled.
- Zero-copy DLPack/PyTorch interop is deferred; current Python interop uses NumPy `.npy` files loaded by `physicaldb.query()` and optionally wrapped with `torch.from_numpy`.
- Public datasets vary substantially in schema and calibration conventions; initial demo should use synthetic or MCAP/CARLA data with controlled ground truth.
