#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from physicaldb.onboarding_cli import run_demo_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the generic customer dataset onboarding demo.")
    parser.add_argument("--workdir", type=Path, default=Path("data/demo/generic_customer"))
    parser.add_argument("--dataset-id", default="generic_customer_m1")
    parser.add_argument("--robot-id", default="customer_bot_001")
    parser.add_argument("--session-id", default="demo_session_001")
    parser.add_argument("--row-group-rows", type=int, default=8)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--robotics-bin")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = run_demo_workflow(
        args.workdir,
        dataset_id=args.dataset_id,
        robot_id=args.robot_id,
        session_id=args.session_id,
        row_group_rows=args.row_group_rows,
        target_hz=args.target_hz,
        robotics_bin=args.robotics_bin,
        force=args.force,
    )
    print(f"out={summary['artifacts']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
