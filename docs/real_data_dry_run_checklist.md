# Real Data Dry-Run Checklist

Use this checklist before asking a customer to watch a live run on their data.

## Intake

- Confirm input shape: local folder, CSV tables, Parquet files, image folders, MCAP files, or S3-compatible prefix.
- Ask for one bounded sample first: minutes of data, not a full bucket.
- Record `dataset_id`, `robot_id`, `session_id`, and expected sensor streams.
- Confirm timestamp columns and units: ns, us, ms, or seconds.
- Confirm coordinate frames and whether pose is world-to-body or body-to-world.
- Ask whether camera/IMU calibration files are present and where they live.
- Ask whether image filenames contain timestamps.
- Ask whether data can be copied locally, or must stay behind an S3-compatible endpoint.
- Confirm sensitive-data constraints before materializing camera frames.

## Operator Flow

1. Run `python3 scripts/pilot_smoke.py --workdir /tmp/robotics_pilot_smoke`.
2. Stage customer data if the source is S3-compatible.
3. Run `robotics dataset inspect --adapter generic_dataset`.
4. Read profile warnings and confidence scores before drafting the mapping.
5. Run `robotics dataset init-mapping`.
6. Review and edit the mapping:
   - Set top-level `mapping_status` to `final`.
   - Set every ingested stream `mapping_status` to `final`.
   - Keep uncertain streams as draft or remove them from the ingest path.
7. Run `robotics dataset validate`.
8. If validation has warnings only, decide whether they are acceptable for the demo.
9. Run `robotics dataset ingest`.
10. Run `robotics dataset query --managed-root`.
11. Save profile, mapping, validation, ingest report, query summary, and smoke summary.

## Failure Triage

- Ambiguous timestamp: ask the customer for timestamp units and clock origin, then set the stream timestamp explicitly.
- Missing pose channels: keep the mapping as draft; do not claim ingest readiness.
- Missing velocity: let generic ingest derive velocity only when pose timestamps and positions are valid.
- Invalid explicit mapping: fix the manifest; validation should fail until corrected.
- Image sequence has no timestamped filenames: ask for a CSV frame index or timestamped export.
- Missing calibration: proceed only if calibration is not required for the demo claim; keep the warning visible.
- S3 prefix unreadable: verify endpoint, bucket, credentials, region, and `AWS_ALLOW_HTTP`/virtual-host settings.
- Oversized sample: reduce the prefix/file set before staging.
- Demo query returns no rows: check `robot_id`, `session_id`, timestamp range, bbox, and velocity filters.

## Demo Exit Criteria

- Profile contains the expected sources and streams.
- Draft mapping makes uncertainty visible.
- Final mapping validates.
- Managed ingest writes Parquet outputs and a catalog DB.
- Query summary has non-empty tensor shape and timestamps.
- Byte accounting and diagnostics are present.
- All commands and artifacts are recorded so the run can be repeated.
