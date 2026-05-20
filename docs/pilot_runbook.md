# Pilot Runbook

This runbook is the demo path for a customer-style robotics dataset. It shows
the inspect -> mapping draft -> validation -> managed ingest -> query loop, with
optional S3-compatible staging.

## Prerequisites

- Rust toolchain with `cargo`.
- Python environment with the project dependencies installed.
- Run commands from the repository root.
- Export `PYTHONPATH=python`.
- Use a stable cargo target directory for repeat runs:

```bash
export CARGO_TARGET_DIR=/tmp/robotics-target
export PYTHONPATH=python
cargo build -p robotics-cli
export ROBOTICS_BIN=/tmp/robotics-target/debug/robotics
```

For LocalStack/S3 validation, Docker must be running. S3-compatible endpoints
also need the usual `AWS_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION`, `AWS_ALLOW_HTTP`, and `AWS_VIRTUAL_HOSTED_STYLE_REQUEST` values.

## One-Command Pilot Smoke

Run this before any customer demo:

```bash
python3 scripts/pilot_smoke.py --workdir /tmp/robotics_pilot_smoke
```

The smoke writes `/tmp/robotics_pilot_smoke/pilot_smoke_summary.json` and fails
nonzero if a demo-critical command fails.

To include LocalStack:

```bash
python3 scripts/pilot_smoke.py --workdir /tmp/robotics_pilot_smoke_s3 --skip-localstack false
```

## Customer Demo Flow

Generate the messy customer fixture and run the full managed workflow:

```bash
$ROBOTICS_BIN dataset demo --workdir /tmp/robotics_demo --force
```

Important outputs:

- `/tmp/robotics_demo/raw_customer_drop/`: generated customer-style source data.
- `/tmp/robotics_demo/artifacts/profile.json`: inspected sources, columns, streams, confidence, and warnings.
- `/tmp/robotics_demo/artifacts/mapping.draft.json`: reviewable draft mapping.
- `/tmp/robotics_demo/artifacts/mapping.final.json`: reviewed final mapping used for ingest.
- `/tmp/robotics_demo/artifacts/validation.json`: validation report.
- `/tmp/robotics_demo/managed/`: managed Parquet/catalog output.
- `/tmp/robotics_demo/artifacts/query_summary.json`: query/tensor summary.

Run a managed query directly:

```bash
$ROBOTICS_BIN dataset query \
  --managed-root /tmp/robotics_demo/managed \
  --robot-id customer_bot_001 \
  --session-id demo_session_001 \
  --channels pos_xyz,rot_wxyz,vel_xyz \
  --out /tmp/robotics_demo/query_again.json
```

The query summary should include a non-empty `tensor_shape`, timestamp range,
selected row groups, selected bytes, diagnostics, and a tensor certificate.

## Manual Inspect/Mapping Loop

Use this flow when testing a real local customer sample:

```bash
$ROBOTICS_BIN dataset inspect \
  --adapter generic_dataset \
  --input /path/to/customer_drop \
  --out /tmp/customer_profile.json

$ROBOTICS_BIN dataset init-mapping \
  --adapter generic_dataset \
  --profile /tmp/customer_profile.json \
  --out /tmp/customer_mapping.json \
  --dataset-id customer_dataset_001 \
  --robot-id customer_robot_001 \
  --session-id customer_session_001

$ROBOTICS_BIN dataset validate \
  --adapter generic_dataset \
  --manifest /tmp/customer_mapping.json \
  --out /tmp/customer_validation.json
```

Review `customer_mapping.json`. Draft mappings may validate with warnings; do
not ingest until the top-level and per-stream `mapping_status` values are
reviewed and set to `final`.

Then ingest and query:

```bash
$ROBOTICS_BIN dataset ingest \
  --adapter generic_dataset \
  --manifest /tmp/customer_mapping.json \
  --output-root /tmp/customer_managed \
  --out /tmp/customer_ingest_report.json

$ROBOTICS_BIN dataset query \
  --managed-root /tmp/customer_managed \
  --robot-id customer_robot_001 \
  --session-id customer_session_001 \
  --channels pos_xyz,rot_wxyz,vel_xyz \
  --out /tmp/customer_query.json
```

## S3-Compatible Staging

Raw generic S3 ingest is intentionally staging-first. Stage a bounded prefix,
inspect the local staged directory, then ingest locally:

```bash
$ROBOTICS_BIN dataset stage-s3 \
  --input s3://bucket/raw-session \
  --out /tmp/customer_staged \
  --manifest /tmp/customer_staged_manifest.json \
  --limit 500

$ROBOTICS_BIN dataset inspect \
  --adapter generic_dataset \
  --input /tmp/customer_staged \
  --out /tmp/customer_profile.json
```

After local ingest, upload managed outputs:

```bash
$ROBOTICS_BIN dataset upload-managed \
  --managed-root /tmp/customer_managed \
  --uri s3://bucket/managed/customer_dataset_001 \
  --manifest /tmp/customer_upload_manifest.json
```

## Demo Talk Track

- "We start by profiling the data instead of assuming semantics."
- "The mapping is a draft with confidence and warnings, so uncertainty is visible."
- "Validation fails only for invalid explicit mappings; ambiguous inferred fields stay reviewable."
- "Once reviewed, ingest writes managed Parquet and catalogs."
- "The query path returns tensor-shaped data plus diagnostics, byte accounting, and a certificate."
- "For S3, v1 stages raw data first and uploads managed outputs after validation."
