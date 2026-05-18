use std::env;
use std::process::ExitCode;
use std::time::Instant;

use robotics_catalog::{
    generate_fake_catalog, index_parquet_file, index_parquet_file_with_uri, query_catalog,
    total_selected_bytes, write_catalog_parquet, FakeCatalogConfig,
};
use robotics_core::{BoundingBox, QuerySpec};
use robotics_ingest::{
    generate_synthetic_pose, read_pose_parquet_row_groups, write_json_pose_mcap,
    write_json_pose_mcap_to_parquet, write_kitti_oxts_to_parquet,
    write_nuscenes_ego_pose_to_parquet, write_pose_mcap_to_parquet, write_pose_parquet,
    write_synthetic_parquet, KittiOxtsConfig, McapJsonPoseConfig, McapPoseConfig,
    NuscenesEgoConfig, SyntheticConfig,
};
use robotics_query::{account_reads, execute_object_store_range_reads, plan_range_reads};
use robotics_tensor::{tensorize, write_tensor_npy};

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
        Some("index") if args.get(2).map(String::as_str) == Some("parquet") => {
            index_parquet(&args[3..])
        }
        Some("range-read") if args.get(2).map(String::as_str) == Some("parquet") => {
            range_read_parquet(&args[3..]).await
        }
        Some("validate") if args.get(2).map(String::as_str) == Some("s3-parquet") => {
            validate_s3_parquet(&args[3..]).await
        }
        Some("tensor") if args.get(2).map(String::as_str) == Some("parquet-row-groups") => {
            tensor_parquet_row_groups(&args[3..])
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

    let entries = match index_parquet_file_with_uri(&input, file_uri) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("catalog index failed: {err}");
            return ExitCode::from(1);
        }
    };
    let indexed_bytes: u64 = entries.iter().map(|entry| entry.byte_length).sum();
    match write_catalog_parquet(&out, &entries) {
        Ok(()) => {
            println!("out={out}");
            println!("indexed_row_groups={}", entries.len());
            println!("indexed_bytes={indexed_bytes}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("catalog write failed: {err}");
            ExitCode::from(1)
        }
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

fn tensor_parquet_row_groups(args: &[String]) -> ExitCode {
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
    let row_groups = match parse_u32_csv(&row_groups_raw, "--row-groups") {
        Ok(row_groups) => row_groups,
        Err(err) => {
            eprintln!("tensor argument error: {err}");
            return ExitCode::from(2);
        }
    };

    let samples = match read_pose_parquet_row_groups(&input, &row_groups) {
        Ok(samples) => samples,
        Err(err) => {
            eprintln!("tensor row-group load failed: {err}");
            return ExitCode::from(1);
        }
    };
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
    if let Some(files) = tensor_npy_files {
        println!("tensor_values_npy={}", files.values_path.display());
        println!("tensor_timestamps_npy={}", files.timestamps_path.display());
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

fn print_help() {
    println!("robotics commands:");
    println!("  robotics catalog fake --sessions 50000");
    println!("  robotics catalog build --input data/parquet/synthetic/session.parquet --out data/catalog/fleet_metadata.parquet");
    println!(
        "  robotics ingest synthetic-mcap --out data/mcap/synthetic/session.mcap --topic /pose"
    );
    println!("  robotics ingest mcap-json --input data/mcap/synthetic/session.mcap --out data/parquet/mcap/session.parquet --topic /pose");
    println!("  robotics ingest mcap-pose --input path/to/poses.mcap --out data/parquet/mcap/pose.parquet --topic /pose");
    println!("  robotics ingest kitti-oxts --input path/to/drive_or_oxts --out data/parquet/kitti/oxts.parquet");
    println!("  robotics ingest nuscenes-ego --input path/to/v1.0-mini --out data/parquet/nuscenes/ego_pose.parquet");
    println!("  robotics ingest synthetic-parquet --out data/parquet/synthetic/session.parquet --row-group-rows 100");
    println!("  robotics index parquet --input data/parquet/synthetic/session.parquet");
    println!(
        "  robotics range-read parquet --input data/parquet/synthetic/session.parquet --limit 1"
    );
    println!("  robotics validate s3-parquet --input data/parquet/synthetic/session.parquet --uri s3://bucket/session.parquet --limit 1");
    println!("  robotics tensor parquet-row-groups --input data/parquet/synthetic/session.parquet --row-groups 0 --start-ts-ns 0 --end-ts-ns 480000000 --out data/tensor/query");
    println!("  robotics demo --out data/parquet/demo/session.parquet --row-group-rows 25 --bbox -0.1,2.0,-1.1,1.1,-0.1,0.1 --tensor-out data/tensor/demo");
    println!("  robotics demo fake");
}
