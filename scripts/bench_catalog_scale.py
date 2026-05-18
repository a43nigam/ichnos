#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb import plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark hot catalog planning latency.")
    parser.add_argument("--sessions", type=int, default=2_000)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--catalog-db", type=Path)
    parser.add_argument("--robot-id", default="humanoid_04")
    parser.add_argument(
        "--predicate",
        action="append",
        help="Predicate to benchmark. May be passed multiple times.",
    )
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--min-prune-ratio", type=float)
    parser.add_argument("--max-authorized-bytes", type=int)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if importlib.util.find_spec("duckdb") is None:
        print("SKIP: duckdb Python package is required for catalog scale benchmark")
        return 0
    if args.sessions < 1:
        raise SystemExit("--sessions must be positive")
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    os.environ.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")
    with tempfile.TemporaryDirectory(prefix="physicaldb_catalog_bench_") as tmp:
        catalog_db = args.catalog_db or (Path(tmp) / "fake_fleet.duckdb")
        if args.rebuild or not catalog_db.exists():
            run_robotics(
                "catalog",
                "fake-duckdb",
                "--sessions",
                str(args.sessions),
                "--out",
                str(catalog_db),
            )

        print(f"catalog_db={catalog_db}")
        print(f"sessions={args.sessions}")
        print(f"catalog_hours={args.sessions}")
        print(f"warmup={args.warmup}")
        print(f"iterations={args.iterations}")
        predicates = args.predicate or [
            "velocity_magnitude > 5.0 AND ST_Intersects(position, bbox(-55,-35,-30,-10,-2,4))",
            "velocity_magnitude > 8.0",
            "ST_Intersects(position, bbox(-50,-40,-25,-15,-2,4))",
        ]
        failures: list[str] = []
        for index, predicate in enumerate(predicates, start=1):
            last_plan = None
            for _ in range(args.warmup):
                last_plan = benchmark_plan(catalog_db, args.robot_id, predicate)

            timings_ms: list[float] = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                last_plan = benchmark_plan(catalog_db, args.robot_id, predicate)
                timings_ms.append((time.perf_counter() - started) * 1000.0)

            assert last_plan is not None
            candidate = last_plan.diagnostics.candidate_row_groups
            matched = last_plan.diagnostics.matched_row_groups
            prune_ratio = 0.0 if candidate == 0 else 1.0 - (matched / candidate)
            p50 = percentile(timings_ms, 0.50)
            p95 = percentile(timings_ms, 0.95)

            print(f"predicate_{index}={predicate}")
            print(f"predicate_{index}_p50_catalog_plan_ms={p50:.3f}")
            print(f"predicate_{index}_p95_catalog_plan_ms={p95:.3f}")
            print(f"predicate_{index}_mean_catalog_plan_ms={statistics.fmean(timings_ms):.3f}")
            print(f"predicate_{index}_candidate_row_groups={candidate}")
            print(f"predicate_{index}_matched_row_groups={matched}")
            print(f"predicate_{index}_time_pruned_row_groups={last_plan.diagnostics.time_pruned_row_groups}")
            print(f"predicate_{index}_spatial_pruned_row_groups={last_plan.diagnostics.spatial_pruned_row_groups}")
            print(f"predicate_{index}_velocity_pruned_row_groups={last_plan.diagnostics.velocity_pruned_row_groups}")
            print(f"predicate_{index}_prune_ratio={prune_ratio:.6f}")
            print(f"predicate_{index}_authorized_total_bytes={last_plan.authorized_total_bytes}")
            print(
                f"predicate_{index}_manifest_ready_row_groups="
                + ",".join(str(row_group) for row_group in last_plan.row_groups)
            )
            print(f"predicate_{index}_blocked_by_egress={str(last_plan.blocked_by_egress).lower()}")

            if args.max_p95_ms is not None and p95 > args.max_p95_ms:
                failures.append(f"predicate {index} p95 {p95:.3f}ms exceeded {args.max_p95_ms:.3f}ms")
            if args.min_prune_ratio is not None and prune_ratio < args.min_prune_ratio:
                failures.append(
                    f"predicate {index} prune_ratio {prune_ratio:.6f} below {args.min_prune_ratio:.6f}"
                )
            if (
                args.max_authorized_bytes is not None
                and last_plan.authorized_total_bytes > args.max_authorized_bytes
            ):
                failures.append(
                    f"predicate {index} authorized bytes {last_plan.authorized_total_bytes} "
                    f"exceeded {args.max_authorized_bytes}"
                )
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
    return 0


def benchmark_plan(catalog_db: Path, robot_id: str, predicate: str):
    return plan(
        catalog_db=catalog_db,
        robot_id=robot_id,
        predicate=predicate,
        channels=("pos_xyz",),
        max_egress_bytes=1_000_000_000_000,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(max(round((len(ordered) - 1) * fraction), 0), len(ordered) - 1)
    return ordered[index]


def run_robotics(*args: str) -> subprocess.CompletedProcess[str]:
    robotics_bin = os.environ.get("ROBOTICS_BIN")
    if robotics_bin:
        cmd = [robotics_bin, *args]
    else:
        cmd = ["cargo", "run", "-p", "robotics-cli", "--", *args]
    return subprocess.run(cmd, check=True, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
