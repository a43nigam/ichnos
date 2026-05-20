#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pilot-readiness smoke workflow.")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/robotics_pilot_smoke"))
    parser.add_argument("--robotics-bin")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--skip-localstack", choices=("true", "false"), default="true")
    args = parser.parse_args()

    workdir = args.workdir
    if workdir.exists() and not args.keep:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env.setdefault("CARGO_TARGET_DIR", "/tmp/robotics-target")
    robotics_bin = resolve_robotics_bin(args.robotics_bin, env)

    summary: dict[str, Any] = {
        "version": 1,
        "workdir": str(workdir),
        "robotics_bin": robotics_bin,
        "commands": [],
        "artifacts": {},
        "localstack": {"skipped": args.skip_localstack == "true"},
    }
    summary_path = workdir / "pilot_smoke_summary.json"

    try:
        demo_dir = workdir / "generic_customer_demo"
        run_step(
            summary,
            [
                robotics_bin,
                "dataset",
                "demo",
                "--workdir",
                str(demo_dir),
                "--force",
                "--robotics-bin",
                robotics_bin,
            ],
            env,
        )
        demo_summary = read_json(demo_dir / "artifacts" / "demo_summary.json")
        summary["artifacts"]["demo_summary"] = str(demo_dir / "artifacts" / "demo_summary.json")

        query_out = workdir / "managed_query.json"
        run_step(
            summary,
            [
                robotics_bin,
                "dataset",
                "query",
                "--managed-root",
                str(demo_dir / "managed"),
                "--robot-id",
                "customer_bot_001",
                "--session-id",
                "demo_session_001",
                "--channels",
                "pos_xyz,rot_wxyz,vel_xyz",
                "--out",
                str(query_out),
                "--robotics-bin",
                robotics_bin,
            ],
            env,
        )
        query_summary = read_json(query_out)
        require_non_empty_tensor(query_summary, query_out)
        summary["artifacts"]["managed_query"] = str(query_out)
        summary["query_tensor_shape"] = query_summary.get("tensor_shape")

        source = workdir / "local_object_source"
        (source / "nested").mkdir(parents=True, exist_ok=True)
        (source / "a.txt").write_text("alpha", encoding="utf-8")
        (source / "nested" / "b.txt").write_text("beta", encoding="utf-8")
        stage_manifest = workdir / "stage_manifest.json"
        run_step(
            summary,
            [
                robotics_bin,
                "dataset",
                "stage-s3",
                "--input",
                str(source),
                "--out",
                str(workdir / "staged"),
                "--manifest",
                str(stage_manifest),
                "--limit",
                "10",
                "--robotics-bin",
                robotics_bin,
            ],
            env,
        )
        stage_payload = read_json(stage_manifest)
        if int(stage_payload.get("object_count", 0)) != 2:
            raise RuntimeError(f"expected 2 staged objects, got {stage_payload.get('object_count')}")
        summary["artifacts"]["stage_manifest"] = str(stage_manifest)

        upload_manifest = workdir / "upload_manifest.json"
        run_step(
            summary,
            [
                robotics_bin,
                "dataset",
                "upload-managed",
                "--managed-root",
                str(demo_dir / "managed"),
                "--uri",
                str(workdir / "uploaded_managed"),
                "--manifest",
                str(upload_manifest),
                "--robotics-bin",
                robotics_bin,
            ],
            env,
        )
        upload_payload = read_json(upload_manifest)
        if int(upload_payload.get("object_count", 0)) <= 0:
            raise RuntimeError("managed upload did not report any uploaded objects")
        summary["artifacts"]["upload_manifest"] = str(upload_manifest)

        if args.skip_localstack == "false":
            run_step(
                summary,
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_localstack_s3_smoke.py"),
                    "--timeout-sec",
                    "90",
                ],
                env,
            )
            summary["localstack"]["skipped"] = False

        summary["status"] = "passed"
        write_json(summary_path, summary)
        print(f"out={summary_path}")
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_json(summary_path, summary)
        print(f"out={summary_path}", file=sys.stderr)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


def resolve_robotics_bin(value: str | None, env: dict[str, str]) -> str:
    if value:
        return str(Path(value))
    candidate = Path(env["CARGO_TARGET_DIR"]) / "debug" / "robotics"
    if not candidate.exists():
        run_command(["cargo", "build", "-p", "robotics-cli"], env)
    if not candidate.exists():
        raise RuntimeError(f"robotics binary was not built at {candidate}")
    return str(candidate)


def run_step(summary: dict[str, Any], command: list[str], env: dict[str, str]) -> None:
    completed = run_command(command, env)
    summary["commands"].append(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    )


def run_command(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return completed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def require_non_empty_tensor(payload: dict[str, Any], path: Path) -> None:
    shape = payload.get("tensor_shape")
    if not isinstance(shape, list) or len(shape) < 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise RuntimeError(f"query summary has empty tensor shape in {path}: {shape!r}")


if __name__ == "__main__":
    raise SystemExit(main())
