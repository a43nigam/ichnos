#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start/reuse LocalStack S3 and run the physicaldb S3 smoke against it."
    )
    parser.add_argument("--container-name", default="robotics-localstack-s3")
    parser.add_argument("--image", default="localstack/localstack:4.14.0")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "robotics"))
    parser.add_argument("--endpoint", default=os.environ.get("AWS_ENDPOINT", "http://127.0.0.1:4566"))
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--skip-start", action="store_true", help="Use an already-running LocalStack endpoint.")
    parser.add_argument("--large-range-smoke", action="store_true", help="Also run the larger range validator.")
    parser.add_argument(
        "--vicon-room1",
        action="store_true",
        help="Also run the full three-sequence EuRoC Vicon Room 1 validator through LocalStack S3.",
    )
    parser.add_argument("--vicon-input-root", default="vicon_room1")
    parser.add_argument("--vicon-output-root", default="data/validation/euroc_vicon_room1_s3")
    parser.add_argument("--vicon-s3-prefix", default="")
    parser.add_argument("--vicon-iterations", type=int, default=5)
    parser.add_argument("--vicon-rebuild", action="store_true")
    args = parser.parse_args()

    if not args.skip_start:
        ensure_localstack(args)
    wait_for_localstack(args.endpoint, args.timeout_sec)
    ensure_bucket(args.container_name, args.bucket)

    env = localstack_env(args)
    print(f"localstack_endpoint={args.endpoint}")
    print(f"s3_bucket={args.bucket}")
    run([sys.executable, "scripts/smoke_s3_pose_imu_media.py"], env=env)
    if args.large_range_smoke:
        run(
            [
                sys.executable,
                "scripts/validate_s3_large_ranges.py",
                "--duration-sec",
                "10",
                "--pose-row-group-rows",
                "250",
                "--imu-row-group-rows",
                "1000",
                "--bucket",
                args.bucket,
            ],
            env=env,
        )
    if args.vicon_room1:
        s3_prefix = args.vicon_s3_prefix or f"s3://{args.bucket}/euroc-vicon-room1"
        command = [
            sys.executable,
            "scripts/validate_euroc_vicon_room1.py",
            "--input-root",
            args.vicon_input_root,
            "--output-root",
            args.vicon_output_root,
            "--s3-prefix",
            s3_prefix,
            "--iterations",
            str(args.vicon_iterations),
        ]
        if args.vicon_rebuild:
            command.append("--rebuild")
        run(command, env=env)
    print("localstack_s3_smoke=passed")
    return 0


def ensure_localstack(args: argparse.Namespace) -> None:
    running_names = run(
        [
            "docker",
            "ps",
            "--filter",
            f"name={args.container_name}",
            "--format",
            "{{.Names}}",
        ],
        capture=True,
    ).stdout.splitlines()
    if args.container_name in running_names:
        return

    all_names = run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={args.container_name}",
            "--format",
            "{{.Names}}",
        ],
        capture=True,
    ).stdout.splitlines()
    if args.container_name in all_names:
        run(["docker", "start", args.container_name])
        return

    run_args = [
        "docker",
        "run",
        "-d",
        "--name",
        args.container_name,
        "-p",
        "4566:4566",
        "-e",
        "SERVICES=s3",
        "-e",
        "DEBUG=0",
    ]
    auth_token = os.environ.get("LOCALSTACK_AUTH_TOKEN")
    if auth_token:
        run_args.extend(["-e", "LOCALSTACK_AUTH_TOKEN"])
    run_args.append(args.image)
    run(run_args)


def wait_for_localstack(endpoint: str, timeout_sec: float) -> None:
    health_url = endpoint.rstrip("/") + "/_localstack/health"
    deadline = time.monotonic() + timeout_sec
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise SystemExit(f"LocalStack did not become healthy at {health_url}: {last_error}")


def ensure_bucket(container_name: str, bucket: str) -> None:
    create = subprocess.run(
        ["docker", "exec", container_name, "awslocal", "s3api", "create-bucket", "--bucket", bucket],
        text=True,
        capture_output=True,
    )
    if create.returncode == 0:
        return
    head = subprocess.run(
        ["docker", "exec", container_name, "awslocal", "s3api", "head-bucket", "--bucket", bucket],
        text=True,
        capture_output=True,
    )
    if head.returncode != 0:
        sys.stderr.write(create.stderr)
        sys.stderr.write(head.stderr)
        raise SystemExit(f"failed to create or verify s3://{bucket} in LocalStack")


def localstack_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "AWS_ENDPOINT": args.endpoint,
            "AWS_ENDPOINT_URL_S3": args.endpoint,
            "AWS_ACCESS_KEY_ID": env.get("AWS_ACCESS_KEY_ID", "test"),
            "AWS_SECRET_ACCESS_KEY": env.get("AWS_SECRET_ACCESS_KEY", "test"),
            "AWS_REGION": env.get("AWS_REGION", "us-east-1"),
            "AWS_ALLOW_HTTP": "true",
            "AWS_VIRTUAL_HOSTED_STYLE_REQUEST": "false",
            "S3_BUCKET": args.bucket,
            "CARGO_TARGET_DIR": env.get("CARGO_TARGET_DIR", "/tmp/robotics-target"),
            "PYTHONPATH": env.get("PYTHONPATH", "python"),
        }
    )
    return env


def run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
