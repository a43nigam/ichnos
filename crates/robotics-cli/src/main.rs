use std::env;
use std::process::Command;
use std::process::ExitCode;
use std::time::Instant;

use robotics_catalog::{
    generate_fake_catalog, index_imu_parquet_file_with_uri_and_gap_threshold,
    index_media_parquet_file_with_uri, index_parquet_file, index_parquet_file_with_uri,
    index_parquet_file_with_uri_and_gap_threshold, query_catalog, total_selected_bytes,
    write_catalog_parquet, write_imu_catalog_parquet, write_media_catalog_parquet,
    FakeCatalogConfig,
};
use robotics_core::{BoundingBox, PoseSample, QuerySpec};
use robotics_ingest::{
    generate_synthetic_pose, read_pose_parquet_row_groups, write_euroc_camera_to_parquet,
    write_euroc_groundtruth_to_parquet, write_euroc_imu_to_parquet, write_json_pose_mcap,
    write_json_pose_mcap_to_parquet, write_kitti_oxts_to_parquet,
    write_nuscenes_ego_pose_to_parquet, write_pose_mcap_to_parquet, write_pose_parquet,
    write_synthetic_parquet, EurocConfig, KittiOxtsConfig, McapJsonPoseConfig, McapPoseConfig,
    NuscenesEgoConfig, SyntheticConfig,
};
use robotics_query::{
    account_reads, audit_row_group_range_reads, execute_object_store_range_reads, plan_range_reads,
    put_object_store_file, read_camera_parquet_row_groups_from_uri,
    read_camera_parquet_row_groups_from_uri_enforced, read_imu_parquet_row_groups_from_uri,
    read_imu_parquet_row_groups_from_uri_enforced, read_pose_parquet_row_groups_from_uri,
    read_pose_parquet_row_groups_from_uri_enforced, CameraFrame, RangeAudit, RangeAuditReport,
    RowGroupRange, SeekManifest,
};
use robotics_tensor::{gap_stats, read_i64_npy, tensorize, tensorize_imu, write_tensor_npy};

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("catalog") if args.get(2).map(String::as_str) == Some("fake") => {
            catalog_fake(&args[3..])
        }
        Some("catalog") if args.get(2).map(String::as_str) == Some("build") => {
            catalog_build(&args[3..])
        }
        Some("catalog") if args.get(2).map(String::as_str) == Some("build-imu") => {
            catalog_build_imu(&args[3..])
        }
        Some("catalog") if args.get(2).map(String::as_str) == Some("build-media") => {
            catalog_build_media(&args[3..])
        }
        Some("catalog") if args.get(2).map(String::as_str) == Some("duckdb-build") => {
            catalog_duckdb_build(&args[3..])
        }
        Some("catalog") if args.get(2).map(String::as_str) == Some("fake-duckdb") => {
            catalog_fake_duckdb(&args[3..])
        }
        Some("catalog") if args.get(2).map(String::as_str) == Some("explain") => {
            catalog_explain(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("synthetic-parquet") => {
            ingest_synthetic_parquet(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("synthetic-mcap") => {
            ingest_synthetic_mcap(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("mcap-json") => {
            ingest_mcap_json(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("mcap-pose") => {
            ingest_mcap_pose(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("kitti-oxts") => {
            ingest_kitti_oxts(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("nuscenes-ego") => {
            ingest_nuscenes_ego(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("euroc-groundtruth") => {
            ingest_euroc_groundtruth(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("euroc-imu") => {
            ingest_euroc_imu(&args[3..])
        }
        Some("ingest") if args.get(2).map(String::as_str) == Some("euroc-camera") => {
            ingest_euroc_camera(&args[3..])
        }
        Some("index") if args.get(2).map(String::as_str) == Some("parquet") => {
            index_parquet(&args[3..])
        }
        Some("range-read") if args.get(2).map(String::as_str) == Some("parquet") => {
            range_read_parquet(&args[3..]).await
        }
        Some("validate") if args.get(2).map(String::as_str) == Some("s3-parquet") => {
            validate_s3_parquet(&args[3..]).await
        }
        Some("object-store") if args.get(2).map(String::as_str) == Some("put") => {
            object_store_put(&args[3..]).await
        }
        Some("tensor") if args.get(2).map(String::as_str) == Some("parquet-row-groups") => {
            tensor_parquet_row_groups(&args[3..]).await
        }
        Some("tensor") if args.get(2).map(String::as_str) == Some("imu-parquet-row-groups") => {
            tensor_imu_parquet_row_groups(&args[3..]).await
        }
        Some("media") if args.get(2).map(String::as_str) == Some("camera-row-groups") => {
            media_camera_row_groups(&args[3..]).await
        }
        Some("demo") if args.get(2).map(String::as_str) == Some("fake") => demo_fake(),
        Some("demo") => demo_parquet(&args[2..]).await,
        Some("help") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("unknown command: {command}");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn ingest_synthetic_mcap(args: &[String]) -> ExitCode {
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/mcap/synthetic/session.mcap".to_string());
    let topic = parse_string_arg(args, "--topic").unwrap_or_else(|| "/pose".to_string());
    let hz = parse_f64_arg(args, "--hz").unwrap_or(100.0);
    let duration_ns = parse_i64_arg(args, "--duration-ns").unwrap_or(1_000_000_000);
    let robot_id =
        parse_string_arg(args, "--robot-id").unwrap_or_else(|| "humanoid_01".to_string());
    let session_id =
        parse_string_arg(args, "--session-id").unwrap_or_else(|| "session_001".to_string());
    let samples = generate_synthetic_pose(
        &robot_id,
        &session_id,
        SyntheticConfig {
            hz,
            duration_ns,
            start_ts_ns: 0,
        },
    );

    match write_json_pose_mcap(&out, &samples, &topic) {
        Ok(messages) => {
            println!("out={out}");
            println!("topic={topic}");
            println!("messages={messages}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("synthetic MCAP ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_mcap_json(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/mcap/session.parquet".to_string());
    let topic = parse_string_arg(args, "--topic").unwrap_or_else(|| "/pose".to_string());
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(100);
    let config = McapJsonPoseConfig {
        topic,
        default_robot_id: parse_string_arg(args, "--robot-id")
            .unwrap_or_else(|| "robot_01".to_string()),
        default_session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "session_001".to_string()),
    };

    match write_json_pose_mcap_to_parquet(&input, &out, &config, row_group_rows) {
        Ok((samples, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("topic={}", config.topic);
            println!("samples={samples}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("MCAP JSON ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_mcap_pose(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/mcap/pose.parquet".to_string());
    let topic = parse_string_arg(args, "--topic").unwrap_or_else(|| "/pose".to_string());
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(100);
    let config = McapPoseConfig {
        topic,
        default_robot_id: parse_string_arg(args, "--robot-id")
            .unwrap_or_else(|| "robot_01".to_string()),
        default_session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "session_001".to_string()),
    };

    match write_pose_mcap_to_parquet(&input, &out, &config, row_group_rows) {
        Ok((samples, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("topic={}", config.topic);
            println!("samples={samples}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("MCAP pose ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_kitti_oxts(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/kitti/oxts.parquet".to_string());
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(100);
    let config = KittiOxtsConfig {
        robot_id: parse_string_arg(args, "--robot-id")
            .unwrap_or_else(|| "kitti_vehicle".to_string()),
        session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "kitti_session".to_string()),
    };

    match write_kitti_oxts_to_parquet(&input, &out, &config, row_group_rows) {
        Ok((samples, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("robot_id={}", config.robot_id);
            println!("session_id={}", config.session_id);
            println!("samples={samples}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("KITTI OXTS ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_nuscenes_ego(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/nuscenes/ego_pose.parquet".to_string());
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(100);
    let config = NuscenesEgoConfig {
        robot_id: parse_string_arg(args, "--robot-id")
            .unwrap_or_else(|| "nuscenes_ego".to_string()),
        session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "nuscenes_scene".to_string()),
    };

    match write_nuscenes_ego_pose_to_parquet(&input, &out, &config, row_group_rows) {
        Ok((samples, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("robot_id={}", config.robot_id);
            println!("session_id={}", config.session_id);
            println!("samples={samples}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("nuScenes ego-pose ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_euroc_groundtruth(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/euroc/pose.parquet".to_string());
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(500);
    let config = EurocConfig {
        robot_id: parse_string_arg(args, "--robot-id").unwrap_or_else(|| "mav0".to_string()),
        session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "euroc_session".to_string()),
    };

    match write_euroc_groundtruth_to_parquet(&input, &out, &config, row_group_rows) {
        Ok((samples, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("robot_id={}", config.robot_id);
            println!("session_id={}", config.session_id);
            println!("samples={samples}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("EuRoC groundtruth ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_euroc_imu(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/euroc/imu.parquet".to_string());
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(2_000);
    let config = EurocConfig {
        robot_id: parse_string_arg(args, "--robot-id").unwrap_or_else(|| "mav0".to_string()),
        session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "euroc_session".to_string()),
    };

    match write_euroc_imu_to_parquet(&input, &out, &config, row_group_rows) {
        Ok((samples, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("robot_id={}", config.robot_id);
            println!("session_id={}", config.session_id);
            println!("samples={samples}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("EuRoC IMU ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_euroc_camera(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let stream_id = parse_string_arg(args, "--stream-id").unwrap_or_else(|| "cam0".to_string());
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| format!("data/parquet/euroc/{stream_id}.parquet"));
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(20);
    let config = EurocConfig {
        robot_id: parse_string_arg(args, "--robot-id").unwrap_or_else(|| "mav0".to_string()),
        session_id: parse_string_arg(args, "--session-id")
            .unwrap_or_else(|| "euroc_session".to_string()),
    };

    match write_euroc_camera_to_parquet(&input, &out, &config, &stream_id, row_group_rows) {
        Ok((frames, row_groups)) => {
            println!("input={input}");
            println!("out={out}");
            println!("robot_id={}", config.robot_id);
            println!("session_id={}", config.session_id);
            println!("stream_id={stream_id}");
            println!("frames={frames}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("EuRoC camera ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn ingest_synthetic_parquet(args: &[String]) -> ExitCode {
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/synthetic/session.parquet".to_string());
    let hz = parse_f64_arg(args, "--hz").unwrap_or(100.0);
    let duration_ns = parse_i64_arg(args, "--duration-ns").unwrap_or(1_000_000_000);
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(100);
    let robot_id =
        parse_string_arg(args, "--robot-id").unwrap_or_else(|| "humanoid_01".to_string());
    let session_id =
        parse_string_arg(args, "--session-id").unwrap_or_else(|| "session_001".to_string());

    match write_synthetic_parquet(
        &out,
        &robot_id,
        &session_id,
        SyntheticConfig {
            hz,
            duration_ns,
            start_ts_ns: 0,
        },
        row_group_rows,
    ) {
        Ok(row_groups) => {
            println!("out={out}");
            println!("row_groups={row_groups}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("synthetic parquet ingest failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn index_parquet(args: &[String]) -> ExitCode {
    let Some(path) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };

    match index_parquet_file(&path) {
        Ok(entries) => {
            let bytes: u64 = entries.iter().map(|entry| entry.byte_length).sum();
            println!("indexed_row_groups={}", entries.len());
            println!("indexed_bytes={bytes}");
            if let Some(first) = entries.first() {
                println!("first_robot_id={}", first.robot_id);
                println!("first_session_id={}", first.session_id);
                println!("first_start_ts_ns={}", first.start_ts_ns);
                println!("first_end_ts_ns={}", first.end_ts_ns);
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("parquet index failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn catalog_build(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(out) = parse_string_arg(args, "--out") else {
        eprintln!("missing required --out");
        return ExitCode::from(2);
    };
    let file_uri = parse_string_arg(args, "--uri").unwrap_or_else(|| input.clone());
    let gap_threshold_ns = parse_i64_arg(args, "--gap-threshold-ns");

    let entries =
        match index_parquet_file_with_uri_and_gap_threshold(&input, file_uri, gap_threshold_ns) {
            Ok(entries) => entries,
            Err(err) => {
                eprintln!("catalog index failed: {err}");
                return ExitCode::from(1);
            }
        };
    let indexed_bytes: u64 = entries.iter().map(|entry| entry.byte_length).sum();
    let gap_row_groups = entries.iter().filter(|entry| entry.gap_count > 0).count();
    let max_gap_ns = entries
        .iter()
        .map(|entry| entry.max_gap_ns)
        .max()
        .unwrap_or(0);
    match write_catalog_parquet(&out, &entries) {
        Ok(()) => {
            println!("out={out}");
            println!("indexed_row_groups={}", entries.len());
            println!("indexed_bytes={indexed_bytes}");
            println!("gap_row_groups={gap_row_groups}");
            println!("max_gap_ns={max_gap_ns}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("catalog write failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn catalog_build_imu(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(out) = parse_string_arg(args, "--out") else {
        eprintln!("missing required --out");
        return ExitCode::from(2);
    };
    let file_uri = parse_string_arg(args, "--uri").unwrap_or_else(|| input.clone());
    let gap_threshold_ns = parse_i64_arg(args, "--gap-threshold-ns");

    let entries =
        match index_imu_parquet_file_with_uri_and_gap_threshold(&input, file_uri, gap_threshold_ns)
        {
            Ok(entries) => entries,
            Err(err) => {
                eprintln!("IMU catalog index failed: {err}");
                return ExitCode::from(1);
            }
        };
    let indexed_bytes: u64 = entries.iter().map(|entry| entry.byte_length).sum();
    let gap_row_groups = entries.iter().filter(|entry| entry.gap_count > 0).count();
    let max_gap_ns = entries
        .iter()
        .map(|entry| entry.max_gap_ns)
        .max()
        .unwrap_or(0);
    match write_imu_catalog_parquet(&out, &entries) {
        Ok(()) => {
            println!("out={out}");
            println!("indexed_row_groups={}", entries.len());
            println!("indexed_bytes={indexed_bytes}");
            println!("gap_row_groups={gap_row_groups}");
            println!("max_gap_ns={max_gap_ns}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("IMU catalog write failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn catalog_build_media(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(out) = parse_string_arg(args, "--out") else {
        eprintln!("missing required --out");
        return ExitCode::from(2);
    };
    let Some(modality) = parse_string_arg(args, "--modality") else {
        eprintln!("missing required --modality camera|lidar");
        return ExitCode::from(2);
    };
    if !matches!(modality.as_str(), "camera" | "lidar") {
        eprintln!("--modality must be camera or lidar");
        return ExitCode::from(2);
    }
    let Some(stream_id) = parse_string_arg(args, "--stream-id") else {
        eprintln!("missing required --stream-id");
        return ExitCode::from(2);
    };
    let file_uri = parse_string_arg(args, "--uri").unwrap_or_else(|| input.clone());

    let entries = match index_media_parquet_file_with_uri(&input, file_uri, &modality, &stream_id) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("media catalog index failed: {err}");
            return ExitCode::from(1);
        }
    };
    let indexed_bytes: u64 = entries.iter().map(|entry| entry.byte_length).sum();
    match write_media_catalog_parquet(&out, &entries) {
        Ok(()) => {
            println!("out={out}");
            println!("modality={modality}");
            println!("stream_id={stream_id}");
            println!("indexed_row_groups={}", entries.len());
            println!("indexed_bytes={indexed_bytes}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("media catalog write failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn catalog_duckdb_build(args: &[String]) -> ExitCode {
    let Some(pose_catalog) = parse_string_arg(args, "--pose-catalog") else {
        eprintln!("missing required --pose-catalog");
        return ExitCode::from(2);
    };
    let Some(out) = parse_string_arg(args, "--out") else {
        eprintln!("missing required --out");
        return ExitCode::from(2);
    };
    let imu_catalog = parse_string_arg(args, "--imu-catalog");
    let media_catalog = parse_string_arg(args, "--media-catalog");
    let spatial_index =
        parse_string_arg(args, "--spatial-index").unwrap_or_else(|| "tile".to_string());

    match run_catalog_duckdb_build(
        &out,
        &pose_catalog,
        imu_catalog.as_deref(),
        media_catalog.as_deref(),
        &spatial_index,
    ) {
        Ok(stdout) => {
            print!("{stdout}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprint!("{err}");
            ExitCode::from(1)
        }
    }
}

fn catalog_fake_duckdb(args: &[String]) -> ExitCode {
    let sessions = parse_usize_arg(args, "--sessions").unwrap_or(100_000);
    let spatial_index =
        parse_string_arg(args, "--spatial-index").unwrap_or_else(|| "hilbert".to_string());
    let Some(out) = parse_string_arg(args, "--out") else {
        eprintln!("missing required --out");
        return ExitCode::from(2);
    };
    let temp_catalog = std::env::temp_dir().join(format!(
        "robotics_fake_catalog_{}_{}.parquet",
        std::process::id(),
        sessions
    ));
    let catalog = generate_fake_catalog(FakeCatalogConfig {
        sessions,
        robot_count: (sessions / 400).max(20),
        ..Default::default()
    });
    if let Err(err) = write_catalog_parquet(&temp_catalog, &catalog) {
        eprintln!("fake catalog parquet write failed: {err}");
        return ExitCode::from(1);
    }

    let result = run_catalog_duckdb_build(
        &out,
        temp_catalog
            .to_str()
            .expect("temporary catalog path should be valid UTF-8"),
        None,
        None,
        &spatial_index,
    );
    std::fs::remove_file(&temp_catalog).ok();

    match result {
        Ok(stdout) => {
            print!("{stdout}");
            println!("fake_sessions={sessions}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprint!("{err}");
            ExitCode::from(1)
        }
    }
}

fn catalog_explain(args: &[String]) -> ExitCode {
    let Some(catalog_db) = parse_string_arg(args, "--catalog-db") else {
        eprintln!("missing required --catalog-db");
        return ExitCode::from(2);
    };
    let predicate = parse_string_arg(args, "--predicate").unwrap_or_default();
    let robot_id = parse_string_arg(args, "--robot-id").unwrap_or_default();
    let start_ts_ns = parse_string_arg(args, "--start-ts-ns").unwrap_or_default();
    let end_ts_ns = parse_string_arg(args, "--end-ts-ns").unwrap_or_default();
    let json = has_flag(args, "--json");

    let script = r#"
import json
import sys
from pathlib import Path

root = Path.cwd() / "python"
if root.exists():
    sys.path.insert(0, str(root))

from physicaldb import plan

catalog_db, predicate, robot_id, start_ts_ns, end_ts_ns, emit_json = sys.argv[1:7]
kwargs = {
    "catalog_db": catalog_db,
    "channels": ("pos_xyz",),
    "max_egress_bytes": 1_000_000_000_000,
}
if predicate:
    kwargs["predicate"] = predicate
if robot_id:
    kwargs["robot_id"] = robot_id
if start_ts_ns:
    kwargs["start_ts_ns"] = int(start_ts_ns)
if end_ts_ns:
    kwargs["end_ts_ns"] = int(end_ts_ns)

seek_plan = plan(**kwargs)
diag = seek_plan.diagnostics
payload = {
    "catalog_db": catalog_db,
    "predicate": predicate,
    "candidate_row_groups": diag.candidate_row_groups,
    "matched_row_groups": diag.matched_row_groups,
    "time_pruned_row_groups": diag.time_pruned_row_groups,
    "spatial_pruned_row_groups": diag.spatial_pruned_row_groups,
    "hilbert_pruned_row_groups": diag.hilbert_pruned_row_groups,
    "exact_spatial_pruned_row_groups": diag.exact_spatial_pruned_row_groups,
    "velocity_pruned_row_groups": diag.velocity_pruned_row_groups,
    "catalog_query_ms": diag.catalog_query_ms,
    "pose_selected_bytes": diag.pose_selected_bytes,
    "imu_selected_bytes": diag.imu_selected_bytes,
    "media_selected_bytes": diag.media_selected_bytes,
    "authorized_total_bytes": seek_plan.authorized_total_bytes,
    "blocked_by_egress": seek_plan.blocked_by_egress,
    "index_strategy": diag.index_strategy,
    "row_groups": list(seek_plan.row_groups),
}
if emit_json == "true":
    print(json.dumps(payload, sort_keys=True))
else:
    for key, value in payload.items():
        if key == "row_groups":
            print("row_groups=" + ",".join(str(row_group) for row_group in value))
        elif isinstance(value, bool):
            print(f"{key}={str(value).lower()}")
        elif isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")
"#;

    let output = Command::new("python3")
        .args([
            "-c",
            script,
            &catalog_db,
            &predicate,
            &robot_id,
            &start_ts_ns,
            &end_ts_ns,
            if json { "true" } else { "false" },
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            print!("{}", String::from_utf8_lossy(&output.stdout));
            ExitCode::SUCCESS
        }
        Ok(output) => {
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            ExitCode::from(1)
        }
        Err(err) => {
            eprintln!("catalog explain failed to start python3: {err}");
            ExitCode::from(1)
        }
    }
}

fn run_catalog_duckdb_build(
    out: &str,
    pose_catalog: &str,
    imu_catalog: Option<&str>,
    media_catalog: Option<&str>,
    spatial_index: &str,
) -> std::result::Result<String, String> {
    if !matches!(spatial_index, "tile" | "hilbert") {
        return Err(
            "catalog DuckDB build failed: --spatial-index must be 'tile' or 'hilbert'\n"
                .to_string(),
        );
    }
    let script = r#"
import sys
from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError as exc:
    raise SystemExit("duckdb Python package is required for catalog duckdb-build") from exc

out, pose_catalog, imu_catalog, media_catalog, spatial_index = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
Path(out).parent.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(out)
def hilbert_xy_key(x, y):
    x = max(0, min(65535, int(x)))
    y = max(0, min(65535, int(y)))
    d = 0
    s = 1 << 15
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = 65535 - x
                y = 65535 - y
            x, y = y, x
        s //= 2
    return d

def quantize_xy(value):
    return max(0, min(65535, int(value // 1) + 32768))

def hilbert_center(min_x, max_x, min_y, max_y):
    return hilbert_xy_key(quantize_xy((min_x + max_x) / 2.0), quantize_xy((min_y + max_y) / 2.0))

def hilbert_minmax(min_x, max_x, min_y, max_y, which):
    x0, x1 = sorted((quantize_xy(min_x), quantize_xy(max_x)))
    y0, y1 = sorted((quantize_xy(min_y), quantize_xy(max_y)))
    cell_count = (x1 - x0 + 1) * (y1 - y0 + 1)
    if cell_count > 1024:
        return 0 if which == "min" else (1 << 32) - 1
    values = [hilbert_xy_key(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    return min(values) if which == "min" else max(values)

con.create_function("hilbert_center", hilbert_center, return_type="UBIGINT")
con.create_function("hilbert_minmax", hilbert_minmax, return_type="UBIGINT")
con.execute("""
    CREATE OR REPLACE TABLE pose_row_groups AS
    SELECT
        *,
        ((min_x + max_x) / 2.0) AS center_x,
        ((min_y + max_y) / 2.0) AS center_y,
        CAST(floor(min_x) AS BIGINT) AS tile_min_x,
        CAST(floor(max_x) AS BIGINT) AS tile_max_x,
        CAST(floor(min_y) AS BIGINT) AS tile_min_y,
        CAST(floor(max_y) AS BIGINT) AS tile_max_y,
        CAST(floor(start_ts_ns / 3600000000000) * 3600000000000 AS BIGINT) AS time_bucket_ns,
        hilbert_center(min_x, max_x, min_y, max_y) AS hilbert_xy,
        hilbert_minmax(min_x, max_x, min_y, max_y, 'min') AS hilbert_min_xy,
        hilbert_minmax(min_x, max_x, min_y, max_y, 'max') AS hilbert_max_xy
    FROM read_parquet(?)
""", [pose_catalog])
pose_count = con.execute("SELECT count(*) FROM pose_row_groups").fetchone()[0]
if pose_count == 0:
    raise SystemExit("pose catalog contains no row groups")
con.execute("CREATE INDEX IF NOT EXISTS pose_robot_session_time_idx ON pose_row_groups(robot_id, session_id, start_ts_ns, end_ts_ns)")
con.execute("CREATE INDEX IF NOT EXISTS pose_time_idx ON pose_row_groups(start_ts_ns, end_ts_ns)")
con.execute("CREATE INDEX IF NOT EXISTS pose_bbox_idx ON pose_row_groups(min_x, max_x, min_y, max_y, min_z, max_z)")
con.execute("CREATE INDEX IF NOT EXISTS pose_tile_idx ON pose_row_groups(tile_min_x, tile_max_x, tile_min_y, tile_max_y)")
con.execute("CREATE INDEX IF NOT EXISTS pose_center_idx ON pose_row_groups(center_x, center_y)")
if spatial_index == "hilbert":
    con.execute("CREATE INDEX IF NOT EXISTS pose_hilbert_idx ON pose_row_groups(robot_id, time_bucket_ns, hilbert_xy)")
    con.execute("CREATE INDEX IF NOT EXISTS pose_hilbert_range_idx ON pose_row_groups(hilbert_min_xy, hilbert_max_xy)")
con.execute("CREATE INDEX IF NOT EXISTS pose_velocity_idx ON pose_row_groups(max_velocity)")

if imu_catalog:
    con.execute("CREATE OR REPLACE TABLE imu_row_groups AS SELECT * FROM read_parquet(?)", [imu_catalog])
else:
    con.execute("""
        CREATE OR REPLACE TABLE imu_row_groups (
            robot_id VARCHAR,
            session_id VARCHAR,
            file_uri VARCHAR,
            row_group_id UINTEGER,
            start_ts_ns BIGINT,
            end_ts_ns BIGINT,
            byte_offset UBIGINT,
            byte_length UBIGINT,
            gap_count UBIGINT,
            max_gap_ns BIGINT,
            max_gap_start_ts_ns BIGINT,
            max_gap_end_ts_ns BIGINT,
            nominal_dt_ns BIGINT
        )
    """)
imu_count = con.execute("SELECT count(*) FROM imu_row_groups").fetchone()[0]
con.execute("CREATE INDEX IF NOT EXISTS imu_robot_session_time_idx ON imu_row_groups(robot_id, session_id, start_ts_ns, end_ts_ns)")
con.execute("CREATE INDEX IF NOT EXISTS imu_time_idx ON imu_row_groups(start_ts_ns, end_ts_ns)")

if media_catalog:
    con.execute("CREATE OR REPLACE TABLE media_row_groups AS SELECT * FROM read_parquet(?)", [media_catalog])
else:
    con.execute("""
        CREATE OR REPLACE TABLE media_row_groups (
            robot_id VARCHAR,
            session_id VARCHAR,
            file_uri VARCHAR,
            modality VARCHAR,
            stream_id VARCHAR,
            row_group_id UINTEGER,
            start_ts_ns BIGINT,
            end_ts_ns BIGINT,
            byte_offset UBIGINT,
            byte_length UBIGINT,
            row_count UBIGINT,
            min_x DOUBLE,
            max_x DOUBLE,
            min_y DOUBLE,
            max_y DOUBLE,
            min_z DOUBLE,
            max_z DOUBLE
        )
    """)
media_count = con.execute("SELECT count(*) FROM media_row_groups").fetchone()[0]
con.execute("CREATE INDEX IF NOT EXISTS media_robot_session_time_idx ON media_row_groups(robot_id, session_id, start_ts_ns, end_ts_ns)")
con.execute("CREATE INDEX IF NOT EXISTS media_modality_stream_idx ON media_row_groups(modality, stream_id)")
con.execute("CREATE INDEX IF NOT EXISTS media_time_idx ON media_row_groups(start_ts_ns, end_ts_ns)")
con.close()
print(f"out={out}")
print(f"pose_row_groups={pose_count}")
print(f"imu_row_groups={imu_count}")
print(f"media_row_groups={media_count}")
print(f"spatial_index={spatial_index}")
"#;

    let output = Command::new("python3")
        .args([
            "-c",
            script,
            out,
            pose_catalog,
            imu_catalog.unwrap_or(""),
            media_catalog.unwrap_or(""),
            spatial_index,
        ])
        .output()
        .map_err(|err| format!("catalog DuckDB build failed to start python3: {err}"))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

async fn range_read_parquet(args: &[String]) -> ExitCode {
    let Some(path) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let file_uri = parse_string_arg(args, "--uri").unwrap_or_else(|| path.clone());

    let entries = match index_parquet_file_with_uri(&path, file_uri) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("parquet index failed: {err}");
            return ExitCode::from(1);
        }
    };
    if entries.is_empty() {
        eprintln!("parquet index produced no row groups");
        return ExitCode::from(1);
    }

    let default_start = entries
        .iter()
        .map(|entry| entry.start_ts_ns)
        .min()
        .unwrap_or(0);
    let default_end = entries
        .iter()
        .map(|entry| entry.end_ts_ns)
        .max()
        .unwrap_or(0);
    let spec = QuerySpec {
        robot_id: parse_string_arg(args, "--robot-id"),
        start_ts_ns: parse_i64_arg(args, "--start-ts-ns").unwrap_or(default_start),
        end_ts_ns: parse_i64_arg(args, "--end-ts-ns").unwrap_or(default_end),
        bbox: None,
        min_velocity: parse_f64_arg(args, "--min-velocity"),
        target_hz: 30.0,
    };

    let windows = query_catalog(&entries, &spec);
    let mut reads = plan_range_reads(&windows);
    if let Some(limit) = parse_usize_arg(args, "--limit") {
        reads.truncate(limit);
    }

    let planned = account_reads(&reads);
    let full_file_bytes = std::fs::metadata(&path).map(|meta| meta.len()).ok();
    match execute_object_store_range_reads(&reads).await {
        Ok((completed, actual)) => {
            println!("indexed_row_groups={}", entries.len());
            println!("matched_windows={}", windows.len());
            println!("executed_range_reads={}", completed.len());
            println!("planned_read_bytes={}", planned.requested_bytes);
            println!("transferred_bytes={}", actual.transferred_bytes);
            if let Some(full_file_bytes) = full_file_bytes {
                println!("full_file_bytes={full_file_bytes}");
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("range read failed: {err}");
            ExitCode::from(1)
        }
    }
}

async fn validate_s3_parquet(args: &[String]) -> ExitCode {
    if parse_string_arg(args, "--uri").is_none() {
        eprintln!("missing required --uri s3://bucket/key or compatible object-store URI");
        return ExitCode::from(2);
    }
    let status = range_read_parquet(args).await;
    if status == ExitCode::SUCCESS {
        println!("live_object_store_validation=passed");
    }
    status
}

async fn object_store_put(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(uri) = parse_string_arg(args, "--uri") else {
        eprintln!("missing required --uri s3://bucket/key or compatible object-store URI");
        return ExitCode::from(2);
    };

    match put_object_store_file(&input, &uri).await {
        Ok(uploaded_bytes) => {
            println!("input={input}");
            println!("uri={uri}");
            println!("uploaded_bytes={uploaded_bytes}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("object-store put failed: {err}");
            ExitCode::from(1)
        }
    }
}

async fn tensor_parquet_row_groups(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(row_groups_raw) = parse_string_arg(args, "--row-groups") else {
        eprintln!("missing required --row-groups");
        return ExitCode::from(2);
    };
    let Some(start_ts_ns) = parse_i64_arg(args, "--start-ts-ns") else {
        eprintln!("missing required --start-ts-ns");
        return ExitCode::from(2);
    };
    let Some(end_ts_ns) = parse_i64_arg(args, "--end-ts-ns") else {
        eprintln!("missing required --end-ts-ns");
        return ExitCode::from(2);
    };
    let hz = parse_f64_arg(args, "--hz").unwrap_or(30.0);
    let out = parse_string_arg(args, "--out");
    let enforce_ranges = has_flag(args, "--enforce-ranges");
    let manifest_out = parse_string_arg(args, "--manifest-out");
    let footer_allowance_bytes =
        parse_u64_arg(args, "--footer-allowance-bytes").unwrap_or(16 * 1024 * 1024);
    let row_groups = match parse_u32_csv(&row_groups_raw, "--row-groups") {
        Ok(row_groups) => row_groups,
        Err(err) => {
            eprintln!("tensor argument error: {err}");
            return ExitCode::from(2);
        }
    };
    let range_audit = match parse_string_arg(args, "--audit-ranges") {
        Some(raw) => match parse_row_group_ranges(&raw, "--audit-ranges").and_then(|ranges| {
            audit_row_group_range_reads(&input, &row_groups, &ranges).map_err(|err| err.to_string())
        }) {
            Ok(audit) => Some(audit),
            Err(err) => {
                eprintln!("range audit failed: {err}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    if enforce_ranges && range_audit.is_none() {
        eprintln!("--enforce-ranges requires --audit-ranges");
        return ExitCode::from(2);
    }

    let (samples, audit_report) = if enforce_ranges {
        match read_pose_parquet_row_groups_from_uri_enforced(
            &input,
            &row_groups,
            range_audit.as_ref().expect("checked audit exists"),
            footer_allowance_bytes,
        )
        .await
        {
            Ok((samples, report)) => (samples, Some(report)),
            Err(err) => {
                eprintln!("tensor row-group load failed: {err}");
                return ExitCode::from(1);
            }
        }
    } else {
        match read_pose_parquet_row_groups_from_uri(&input, &row_groups).await {
            Ok(samples) => (samples, None),
            Err(err) => {
                eprintln!("tensor row-group load failed: {err}");
                return ExitCode::from(1);
            }
        }
    };
    let (pose_gap_count, pose_max_gap_ns) = {
        let sample_timestamps = samples
            .iter()
            .map(|sample| sample.timestamp_ns)
            .collect::<Vec<_>>();
        gap_stats(&sample_timestamps)
    };
    let quaternion_inversions = pose_quaternion_inversions(&samples);
    let batch = match tensorize(&samples, start_ts_ns, end_ts_ns, hz) {
        Ok(batch) => batch,
        Err(err) => {
            eprintln!("tensorization failed: {err}");
            return ExitCode::from(1);
        }
    };
    let tensor_npy_files = match out {
        Some(prefix) => match write_tensor_npy(&prefix, &batch) {
            Ok(files) => Some(files),
            Err(err) => {
                eprintln!("tensor export failed: {err}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    println!("input={input}");
    println!("row_groups={row_groups_raw}");
    println!("source_rows={}", samples.len());
    println!("tensor_shape=[{}, {}]", batch.rows, batch.channels);
    println!(
        "uniform_start_ns={}",
        batch.timestamps_ns.first().unwrap_or(&0)
    );
    println!(
        "uniform_end_ns={}",
        batch.timestamps_ns.last().unwrap_or(&0)
    );
    println!("pose_gap_count={pose_gap_count}");
    println!("pose_max_gap_ns={pose_max_gap_ns}");
    println!("pose_null_count=0");
    println!("quaternion_inversions_applied={quaternion_inversions}");
    print_range_audit(&range_audit);
    print_audit_report(&audit_report);
    if let Some(path) = manifest_out {
        let manifest = SeekManifest::new(
            input.clone(),
            "pose",
            row_groups.clone(),
            range_audit.as_ref(),
            audit_report.as_ref(),
        );
        if let Err(err) = write_manifest(&path, &manifest) {
            eprintln!("manifest write failed: {err}");
            return ExitCode::from(1);
        }
        println!("manifest_out={path}");
    }
    if let Some(files) = tensor_npy_files {
        println!("tensor_values_npy={}", files.values_path.display());
        println!("tensor_timestamps_npy={}", files.timestamps_path.display());
    }
    ExitCode::SUCCESS
}

async fn tensor_imu_parquet_row_groups(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(row_groups_raw) = parse_string_arg(args, "--row-groups") else {
        eprintln!("missing required --row-groups");
        return ExitCode::from(2);
    };
    let Some(timestamps_npy) = parse_string_arg(args, "--timestamps-npy") else {
        eprintln!("missing required --timestamps-npy");
        return ExitCode::from(2);
    };
    let out = parse_string_arg(args, "--out");
    let enforce_ranges = has_flag(args, "--enforce-ranges");
    let manifest_out = parse_string_arg(args, "--manifest-out");
    let footer_allowance_bytes =
        parse_u64_arg(args, "--footer-allowance-bytes").unwrap_or(16 * 1024 * 1024);
    let row_groups = match parse_u32_csv(&row_groups_raw, "--row-groups") {
        Ok(row_groups) => row_groups,
        Err(err) => {
            eprintln!("IMU tensor argument error: {err}");
            return ExitCode::from(2);
        }
    };
    let range_audit = match parse_string_arg(args, "--audit-ranges") {
        Some(raw) => match parse_row_group_ranges(&raw, "--audit-ranges").and_then(|ranges| {
            audit_row_group_range_reads(&input, &row_groups, &ranges).map_err(|err| err.to_string())
        }) {
            Ok(audit) => Some(audit),
            Err(err) => {
                eprintln!("IMU range audit failed: {err}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    if enforce_ranges && range_audit.is_none() {
        eprintln!("--enforce-ranges requires --audit-ranges");
        return ExitCode::from(2);
    }
    let timestamps_ns = match read_i64_npy(&timestamps_npy) {
        Ok(timestamps_ns) => timestamps_ns,
        Err(err) => {
            eprintln!("IMU timestamp load failed: {err}");
            return ExitCode::from(1);
        }
    };
    let (samples, audit_report) = if enforce_ranges {
        match read_imu_parquet_row_groups_from_uri_enforced(
            &input,
            &row_groups,
            range_audit.as_ref().expect("checked audit exists"),
            footer_allowance_bytes,
        )
        .await
        {
            Ok((samples, report)) => (samples, Some(report)),
            Err(err) => {
                eprintln!("IMU row-group load failed: {err}");
                return ExitCode::from(1);
            }
        }
    } else {
        match read_imu_parquet_row_groups_from_uri(&input, &row_groups).await {
            Ok(samples) => (samples, None),
            Err(err) => {
                eprintln!("IMU row-group load failed: {err}");
                return ExitCode::from(1);
            }
        }
    };
    let (imu_gap_count, imu_max_gap_ns) = {
        let sample_timestamps = samples
            .iter()
            .map(|sample| sample.timestamp_ns)
            .collect::<Vec<_>>();
        gap_stats(&sample_timestamps)
    };
    let batch = match tensorize_imu(&samples, &timestamps_ns) {
        Ok(batch) => batch,
        Err(err) => {
            eprintln!("IMU tensorization failed: {err}");
            return ExitCode::from(1);
        }
    };
    let tensor_npy_files = match out {
        Some(prefix) => match write_tensor_npy(&prefix, &batch) {
            Ok(files) => Some(files),
            Err(err) => {
                eprintln!("IMU tensor export failed: {err}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    println!("input={input}");
    println!("row_groups={row_groups_raw}");
    println!("source_rows={}", samples.len());
    println!("tensor_shape=[{}, {}]", batch.rows, batch.channels);
    println!(
        "uniform_start_ns={}",
        batch.timestamps_ns.first().unwrap_or(&0)
    );
    println!(
        "uniform_end_ns={}",
        batch.timestamps_ns.last().unwrap_or(&0)
    );
    println!("imu_gap_count={imu_gap_count}");
    println!("imu_max_gap_ns={imu_max_gap_ns}");
    println!("imu_null_count=0");
    print_range_audit(&range_audit);
    print_audit_report(&audit_report);
    if let Some(path) = manifest_out {
        let manifest = SeekManifest::new(
            input.clone(),
            "imu",
            row_groups.clone(),
            range_audit.as_ref(),
            audit_report.as_ref(),
        );
        if let Err(err) = write_manifest(&path, &manifest) {
            eprintln!("manifest write failed: {err}");
            return ExitCode::from(1);
        }
        println!("manifest_out={path}");
    }
    if let Some(files) = tensor_npy_files {
        println!("tensor_values_npy={}", files.values_path.display());
        println!("tensor_timestamps_npy={}", files.timestamps_path.display());
    }
    ExitCode::SUCCESS
}

async fn media_camera_row_groups(args: &[String]) -> ExitCode {
    let Some(input) = parse_string_arg(args, "--input") else {
        eprintln!("missing required --input");
        return ExitCode::from(2);
    };
    let Some(row_groups_raw) = parse_string_arg(args, "--row-groups") else {
        eprintln!("missing required --row-groups");
        return ExitCode::from(2);
    };
    let Some(out) = parse_string_arg(args, "--out") else {
        eprintln!("missing required --out");
        return ExitCode::from(2);
    };
    let enforce_ranges = has_flag(args, "--enforce-ranges");
    let manifest_out = parse_string_arg(args, "--manifest-out");
    let footer_allowance_bytes =
        parse_u64_arg(args, "--footer-allowance-bytes").unwrap_or(16 * 1024 * 1024);
    let row_groups = match parse_u32_csv(&row_groups_raw, "--row-groups") {
        Ok(row_groups) => row_groups,
        Err(err) => {
            eprintln!("camera media argument error: {err}");
            return ExitCode::from(2);
        }
    };
    let range_audit = match parse_string_arg(args, "--audit-ranges") {
        Some(raw) => match parse_row_group_ranges(&raw, "--audit-ranges").and_then(|ranges| {
            audit_row_group_range_reads(&input, &row_groups, &ranges).map_err(|err| err.to_string())
        }) {
            Ok(audit) => Some(audit),
            Err(err) => {
                eprintln!("camera media range audit failed: {err}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };
    if enforce_ranges && range_audit.is_none() {
        eprintln!("--enforce-ranges requires --audit-ranges");
        return ExitCode::from(2);
    }

    let (frames, audit_report) = if enforce_ranges {
        match read_camera_parquet_row_groups_from_uri_enforced(
            &input,
            &row_groups,
            range_audit.as_ref().expect("checked audit exists"),
            footer_allowance_bytes,
        )
        .await
        {
            Ok((frames, report)) => (frames, Some(report)),
            Err(err) => {
                eprintln!("camera media row-group load failed: {err}");
                return ExitCode::from(1);
            }
        }
    } else {
        match read_camera_parquet_row_groups_from_uri(&input, &row_groups).await {
            Ok(frames) => (frames, None),
            Err(err) => {
                eprintln!("camera media row-group load failed: {err}");
                return ExitCode::from(1);
            }
        }
    };
    let written_frames = match write_camera_frames(&out, &frames) {
        Ok(written_frames) => written_frames,
        Err(err) => {
            eprintln!("camera media export failed: {err}");
            return ExitCode::from(1);
        }
    };

    println!("input={input}");
    println!("row_groups={row_groups_raw}");
    println!("source_rows={}", frames.len());
    println!("media_frames={}", written_frames.len());
    println!(
        "media_bytes={}",
        frames
            .iter()
            .map(|frame| frame.camera_bytes.len() as u64)
            .sum::<u64>()
    );
    print_range_audit(&range_audit);
    print_audit_report(&audit_report);
    if let Some(path) = manifest_out {
        let seek_manifest = SeekManifest::new(
            input.clone(),
            "camera",
            row_groups.clone(),
            range_audit.as_ref(),
            audit_report.as_ref(),
        );
        if let Err(err) = write_camera_manifest(&path, &seek_manifest, &written_frames) {
            eprintln!("manifest write failed: {err}");
            return ExitCode::from(1);
        }
        println!("manifest_out={path}");
    }
    ExitCode::SUCCESS
}

fn catalog_fake(args: &[String]) -> ExitCode {
    let sessions = parse_usize_arg(args, "--sessions").unwrap_or(50_000);
    let started = Instant::now();
    let catalog = generate_fake_catalog(FakeCatalogConfig {
        sessions,
        ..Default::default()
    });
    let elapsed = started.elapsed();

    println!("generated_entries={}", catalog.len());
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
    if let Some(first) = catalog.first() {
        println!("first_robot_id={}", first.robot_id);
        println!("first_session_id={}", first.session_id);
    }

    ExitCode::SUCCESS
}

async fn demo_parquet(args: &[String]) -> ExitCode {
    let started = Instant::now();
    let out = parse_string_arg(args, "--out")
        .unwrap_or_else(|| "data/parquet/demo/session.parquet".to_string());
    let hz = parse_f64_arg(args, "--hz").unwrap_or(50.0);
    let tensor_hz = parse_f64_arg(args, "--tensor-hz").unwrap_or(30.0);
    let duration_ns = parse_i64_arg(args, "--duration-ns").unwrap_or(1_000_000_000);
    let row_group_rows = parse_usize_arg(args, "--row-group-rows").unwrap_or(25);
    let robot_id =
        parse_string_arg(args, "--robot-id").unwrap_or_else(|| "humanoid_01".to_string());
    let session_id =
        parse_string_arg(args, "--session-id").unwrap_or_else(|| "demo_session".to_string());
    let min_velocity = parse_f64_arg(args, "--min-velocity").or(Some(2.0));
    let bbox = match parse_string_arg(args, "--bbox") {
        Some(raw) if raw == "none" => None,
        Some(raw) => match parse_bbox_value(&raw, "--bbox") {
            Ok(bbox) => Some(bbox),
            Err(err) => {
                eprintln!("demo argument error: {err}");
                return ExitCode::from(2);
            }
        },
        None => Some(default_demo_bbox()),
    };
    let config = SyntheticConfig {
        hz,
        duration_ns,
        start_ts_ns: parse_i64_arg(args, "--source-start-ts-ns").unwrap_or(0),
    };
    let samples = generate_synthetic_pose(&robot_id, &session_id, config);

    let row_groups = match write_pose_parquet(&out, &samples, row_group_rows) {
        Ok(row_groups) => row_groups,
        Err(err) => {
            eprintln!("demo ingest failed: {err}");
            return ExitCode::from(1);
        }
    };
    let entries = match index_parquet_file(&out) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("demo index failed: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(first) = entries.first() else {
        eprintln!("demo index produced no row groups");
        return ExitCode::from(1);
    };
    let query_start_ns = parse_i64_arg(args, "--start-ts-ns").unwrap_or(first.start_ts_ns);
    let query_end_ns = parse_i64_arg(args, "--end-ts-ns").unwrap_or(first.end_ts_ns);
    let spec = QuerySpec {
        robot_id: Some(robot_id.clone()),
        start_ts_ns: query_start_ns,
        end_ts_ns: query_end_ns,
        bbox,
        min_velocity,
        target_hz: tensor_hz,
    };
    let windows = query_catalog(&entries, &spec);
    if windows.is_empty() {
        eprintln!("demo query matched no row groups");
        return ExitCode::from(1);
    }

    let reads = plan_range_reads(&windows);
    let planned = account_reads(&reads);
    let full_file_bytes = std::fs::metadata(&out).map(|meta| meta.len()).ok();
    let range_accounting = match execute_object_store_range_reads(&reads).await {
        Ok((_completed, accounting)) => accounting,
        Err(err) => {
            eprintln!("demo range read failed: {err}");
            return ExitCode::from(1);
        }
    };
    let row_group_ids = windows
        .iter()
        .map(|window| window.entry.row_group_id)
        .collect::<Vec<_>>();
    let tensor_samples = match read_pose_parquet_row_groups(&out, &row_group_ids) {
        Ok(samples) => samples,
        Err(err) => {
            eprintln!("demo row-group tensor load failed: {err}");
            return ExitCode::from(1);
        }
    };
    let batch = match tensorize(&tensor_samples, query_start_ns, query_end_ns, tensor_hz) {
        Ok(batch) => batch,
        Err(err) => {
            eprintln!("demo tensorization failed: {err}");
            return ExitCode::from(1);
        }
    };
    let tensor_npy_files = match parse_string_arg(args, "--tensor-out") {
        Some(prefix) => match write_tensor_npy(&prefix, &batch) {
            Ok(files) => Some(files),
            Err(err) => {
                eprintln!("demo tensor export failed: {err}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };
    let elapsed = started.elapsed();

    println!("out={out}");
    println!("samples={}", samples.len());
    println!("written_row_groups={row_groups}");
    println!("indexed_row_groups={}", entries.len());
    println!("query_start_ts_ns={query_start_ns}");
    println!("query_end_ts_ns={query_end_ns}");
    if let Some(min_velocity) = min_velocity {
        println!("min_velocity={min_velocity:.3}");
    }
    if let Some(bbox) = bbox {
        println!(
            "bbox=[{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}]",
            bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, bbox.min_z, bbox.max_z
        );
    }
    println!("matched_windows={}", windows.len());
    println!("selected_bytes={}", total_selected_bytes(&windows));
    println!("planned_range_reads={}", planned.completed_reads);
    println!("planned_read_bytes={}", planned.requested_bytes);
    println!("transferred_bytes={}", range_accounting.transferred_bytes);
    if let Some(full_file_bytes) = full_file_bytes {
        println!("full_file_bytes={full_file_bytes}");
    }
    println!("tensor_source_rows={}", tensor_samples.len());
    println!("tensor_shape=[{}, {}]", batch.rows, batch.channels);
    println!(
        "uniform_start_ns={}",
        batch.timestamps_ns.first().unwrap_or(&0)
    );
    println!(
        "uniform_end_ns={}",
        batch.timestamps_ns.last().unwrap_or(&0)
    );
    if let Some(files) = tensor_npy_files {
        println!("tensor_values_npy={}", files.values_path.display());
        println!("tensor_timestamps_npy={}", files.timestamps_path.display());
    }
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
    ExitCode::SUCCESS
}

fn demo_fake() -> ExitCode {
    let catalog = generate_fake_catalog(FakeCatalogConfig {
        sessions: 128,
        ..Default::default()
    });
    let seed = catalog
        .iter()
        .find(|entry| entry.max_velocity >= 3.0)
        .expect("fake catalog should include moving windows");
    let spec = QuerySpec {
        robot_id: Some(seed.robot_id.clone()),
        start_ts_ns: seed.start_ts_ns,
        end_ts_ns: seed.end_ts_ns,
        bbox: None,
        min_velocity: Some(3.0),
        target_hz: 30.0,
    };
    let windows = query_catalog(&catalog, &spec);
    let reads = plan_range_reads(&windows);
    let accounting = account_reads(&reads);
    let samples = generate_synthetic_pose(
        "humanoid_01",
        "demo",
        SyntheticConfig {
            hz: 100.0,
            duration_ns: 1_000_000_000,
            start_ts_ns: 0,
        },
    );

    match tensorize(&samples, 0, 1_000_000_000, spec.target_hz) {
        Ok(batch) => {
            println!("matched_windows={}", windows.len());
            println!("selected_bytes={}", total_selected_bytes(&windows));
            println!("planned_range_reads={}", accounting.completed_reads);
            println!("planned_read_bytes={}", accounting.requested_bytes);
            println!("tensor_shape=[{}, {}]", batch.rows, batch.channels);
            println!(
                "uniform_start_ns={}",
                batch.timestamps_ns.first().unwrap_or(&0)
            );
            println!(
                "uniform_end_ns={}",
                batch.timestamps_ns.last().unwrap_or(&0)
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("fake demo failed: {err}");
            ExitCode::from(1)
        }
    }
}

fn parse_usize_arg(args: &[String], name: &str) -> Option<usize> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<usize>().ok())
}

fn parse_u64_arg(args: &[String], name: &str) -> Option<u64> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<u64>().ok())
}

fn parse_i64_arg(args: &[String], name: &str) -> Option<i64> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<i64>().ok())
}

fn parse_f64_arg(args: &[String], name: &str) -> Option<f64> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<f64>().ok())
}

fn parse_string_arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_u32_csv(raw: &str, name: &str) -> std::result::Result<Vec<u32>, String> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .map_err(|err| format!("{name} contains invalid u32 {part:?}: {err}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("{name} must contain at least one row group id"));
    }
    Ok(values)
}

fn parse_row_group_ranges(
    raw: &str,
    name: &str,
) -> std::result::Result<Vec<RowGroupRange>, String> {
    let ranges = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            let fields = part.split(':').collect::<Vec<_>>();
            if fields.len() != 3 {
                return Err(format!(
                    "{name} entries must be row_group_id:byte_offset:byte_length"
                ));
            }
            let row_group_id = fields[0].parse::<u32>().map_err(|err| {
                format!("{name} contains invalid row group {:?}: {err}", fields[0])
            })?;
            let offset = fields[1]
                .parse::<u64>()
                .map_err(|err| format!("{name} contains invalid offset {:?}: {err}", fields[1]))?;
            let length = fields[2]
                .parse::<u64>()
                .map_err(|err| format!("{name} contains invalid length {:?}: {err}", fields[2]))?;
            Ok(RowGroupRange {
                row_group_id,
                offset,
                length,
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if ranges.is_empty() {
        return Err(format!("{name} must contain at least one range"));
    }
    Ok(ranges)
}

#[derive(Debug, Clone)]
struct WrittenCameraFrame {
    timestamp_ns: i64,
    stream_id: String,
    frame_path: String,
    output_path: String,
    bytes: u64,
}

fn write_camera_frames(
    out: &str,
    frames: &[CameraFrame],
) -> robotics_core::Result<Vec<WrittenCameraFrame>> {
    let out = std::path::Path::new(out);
    std::fs::create_dir_all(out)
        .map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))?;
    let mut written = Vec::with_capacity(frames.len());
    for frame in frames {
        let extension = camera_frame_extension(&frame.frame_path);
        let stream_dir = out.join(sanitize_path_component(&frame.stream_id));
        std::fs::create_dir_all(&stream_dir)
            .map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))?;
        let output_path = stream_dir.join(format!("{}.{extension}", frame.timestamp_ns));
        std::fs::write(&output_path, &frame.camera_bytes)
            .map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))?;
        written.push(WrittenCameraFrame {
            timestamp_ns: frame.timestamp_ns,
            stream_id: frame.stream_id.clone(),
            frame_path: frame.frame_path.clone(),
            output_path: output_path.display().to_string(),
            bytes: frame.camera_bytes.len() as u64,
        });
    }
    Ok(written)
}

fn camera_frame_extension(frame_path: &str) -> String {
    std::path::Path::new(frame_path)
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| {
            !extension.is_empty()
                && extension
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
        })
        .unwrap_or("bin")
        .to_string()
}

fn sanitize_path_component(raw: &str) -> String {
    let sanitized = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "stream".to_string()
    } else {
        sanitized
    }
}

fn write_camera_manifest(
    path: &str,
    seek_manifest: &SeekManifest,
    frames: &[WrittenCameraFrame],
) -> robotics_core::Result<()> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))?;
    }
    let json = serde_json::json!({
        "seek": seek_manifest,
        "frames": frames.iter().map(|frame| {
            serde_json::json!({
                "timestamp_ns": frame.timestamp_ns,
                "stream_id": &frame.stream_id,
                "frame_path": &frame.frame_path,
                "output_path": &frame.output_path,
                "bytes": frame.bytes,
            })
        }).collect::<Vec<_>>(),
    });
    let raw = serde_json::to_string_pretty(&json)
        .map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))?;
    std::fs::write(path, raw).map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))
}

fn print_range_audit(range_audit: &Option<RangeAudit>) {
    if let Some(range_audit) = range_audit {
        println!("planned_range_reads={}", range_audit.planned_range_reads());
        println!("planned_read_bytes={}", range_audit.planned_read_bytes);
        println!("range_audit_passed=true");
    }
}

fn print_audit_report(audit_report: &Option<RangeAuditReport>) {
    if let Some(report) = audit_report {
        println!("range_enforced={}", report.enforcement_enabled);
        println!("footer_allowance_bytes={}", report.footer_allowance_bytes);
        println!("actual_cold_reads={}", report.actual_read_count());
        println!("actual_cold_read_bytes={}", report.actual_read_bytes());
        println!("actual_authorized_bytes={}", report.actual_authorized_bytes);
        println!("materialized_bytes={}", report.materialized_bytes);
        println!("footer_bytes={}", report.footer_bytes);
        println!("largest_metadata_read={}", report.largest_metadata_read);
        println!("max_footer_read_offset={}", report.max_footer_read_offset);
        println!("max_footer_read_end={}", report.max_footer_read_end);
        println!("range_violations={}", report.violations.len());
    } else {
        println!("range_enforced=false");
    }
}

fn write_manifest(path: &str, manifest: &SeekManifest) -> robotics_core::Result<()> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))?;
    }
    let json = manifest.to_json_pretty()?;
    std::fs::write(path, json).map_err(|err| robotics_core::RoboticsError::Io(err.to_string()))
}

fn parse_bbox_value(raw: &str, name: &str) -> std::result::Result<BoundingBox, String> {
    let values = raw
        .split(',')
        .map(|part| {
            part.parse::<f64>()
                .map_err(|err| format!("{name} contains invalid float {part:?}: {err}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.len() != 6 {
        return Err(format!(
            "{name} must contain 6 comma-separated values: min_x,max_x,min_y,max_y,min_z,max_z"
        ));
    }

    Ok(BoundingBox {
        min_x: values[0],
        max_x: values[1],
        min_y: values[2],
        max_y: values[3],
        min_z: values[4],
        max_z: values[5],
    })
}

fn default_demo_bbox() -> BoundingBox {
    BoundingBox {
        min_x: -0.1,
        max_x: 2.0,
        min_y: -1.1,
        max_y: 1.1,
        min_z: -0.1,
        max_z: 0.1,
    }
}

fn pose_quaternion_inversions(samples: &[PoseSample]) -> usize {
    let mut sorted = samples.to_vec();
    sorted.sort_by_key(|sample| sample.timestamp_ns);
    sorted
        .windows(2)
        .filter(|pair| {
            pair[0].qw * pair[1].qw
                + pair[0].qx * pair[1].qx
                + pair[0].qy * pair[1].qy
                + pair[0].qz * pair[1].qz
                < 0.0
        })
        .count()
}

fn print_help() {
    println!("robotics commands:");
    println!("  robotics catalog fake --sessions 50000");
    println!("  robotics catalog build --input data/parquet/synthetic/session.parquet --out data/catalog/fleet_metadata.parquet --gap-threshold-ns 500000000");
    println!("  robotics catalog build-imu --input data/parquet/euroc/V1_01_easy/imu.parquet --out data/catalog/euroc_v1_01_easy_imu.parquet --gap-threshold-ns 50000000");
    println!("  robotics catalog build-media --input data/parquet/camera/cam0.parquet --out data/catalog/cam0_media.parquet --modality camera --stream-id cam0");
    println!("  robotics catalog duckdb-build --pose-catalog data/catalog/fleet_metadata.parquet --imu-catalog data/catalog/euroc_v1_01_easy_imu.parquet --out data/catalog/fleet.duckdb --spatial-index hilbert");
    println!(
        "  robotics catalog fake-duckdb --sessions 100000 --spatial-index hilbert --out data/catalog/fake_fleet.duckdb"
    );
    println!("  robotics catalog explain --catalog-db data/catalog/fake_fleet.duckdb --predicate 'velocity_magnitude > 5.0' --json");
    println!(
        "  robotics ingest synthetic-mcap --out data/mcap/synthetic/session.mcap --topic /pose"
    );
    println!("  robotics ingest mcap-json --input data/mcap/synthetic/session.mcap --out data/parquet/mcap/session.parquet --topic /pose");
    println!("  robotics ingest mcap-pose --input path/to/poses.mcap --out data/parquet/mcap/pose.parquet --topic /pose");
    println!("  robotics ingest kitti-oxts --input path/to/drive_or_oxts --out data/parquet/kitti/oxts.parquet");
    println!("  robotics ingest nuscenes-ego --input path/to/v1.0-mini --out data/parquet/nuscenes/ego_pose.parquet");
    println!("  robotics ingest euroc-groundtruth --input vicon_room1/V1_01_easy --out data/parquet/euroc/V1_01_easy/pose.parquet --session-id V1_01_easy");
    println!("  robotics ingest euroc-imu --input vicon_room1/V1_01_easy --out data/parquet/euroc/V1_01_easy/imu.parquet --session-id V1_01_easy");
    println!("  robotics ingest euroc-camera --input vicon_room1/V1_01_easy --out data/parquet/euroc/V1_01_easy/cam0.parquet --stream-id cam0 --session-id V1_01_easy");
    println!("  robotics ingest synthetic-parquet --out data/parquet/synthetic/session.parquet --row-group-rows 100");
    println!("  robotics index parquet --input data/parquet/synthetic/session.parquet");
    println!(
        "  robotics range-read parquet --input data/parquet/synthetic/session.parquet --limit 1"
    );
    println!("  robotics validate s3-parquet --input data/parquet/synthetic/session.parquet --uri s3://bucket/session.parquet --limit 1");
    println!("  robotics object-store put --input data/parquet/synthetic/session.parquet --uri s3://bucket/session.parquet");
    println!("  robotics tensor parquet-row-groups --input data/parquet/synthetic/session.parquet --row-groups 0 --start-ts-ns 0 --end-ts-ns 480000000 --audit-ranges 0:4096:65536 --enforce-ranges --footer-allowance-bytes 16777216 --manifest-out data/tensor/query.manifest.json --out data/tensor/query");
    println!("  robotics tensor imu-parquet-row-groups --input data/parquet/euroc/V1_01_easy/imu.parquet --row-groups 0 --timestamps-npy data/tensor/query.timestamps_ns.npy --audit-ranges 0:4096:65536 --enforce-ranges --footer-allowance-bytes 16777216 --manifest-out data/tensor/query_imu.manifest.json --out data/tensor/query_imu");
    println!("  robotics media camera-row-groups --input data/parquet/euroc/V1_01_easy/cam0.parquet --row-groups 0 --out data/media/query --audit-ranges 0:4096:65536 --enforce-ranges --manifest-out data/media/query.manifest.json");
    println!("  robotics demo --out data/parquet/demo/session.parquet --row-group-rows 25 --bbox -0.1,2.0,-1.1,1.1,-0.1,0.1 --tensor-out data/tensor/demo");
    println!("  robotics demo fake");
}
