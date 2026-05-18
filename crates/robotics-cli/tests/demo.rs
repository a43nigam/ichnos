use robotics_catalog::{
    generate_fake_catalog, index_parquet_file, query_catalog, FakeCatalogConfig,
};
use robotics_core::QuerySpec;
use robotics_ingest::{generate_synthetic_pose, write_synthetic_parquet, SyntheticConfig};
use robotics_query::{account_reads, plan_range_reads};
use robotics_tensor::tensorize;
use std::process::Command;

#[test]
fn demo_loop_returns_tensor_and_byte_plan() {
    let catalog = generate_fake_catalog(FakeCatalogConfig {
        sessions: 16,
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
    let samples = generate_synthetic_pose("humanoid_01", "demo", SyntheticConfig::default());
    let batch = tensorize(&samples, 0, 1_000_000_000, 30.0).unwrap();

    assert!(!windows.is_empty());
    assert_eq!(accounting.requested_bytes, windows.len() as u64 * 65_536);
    assert_eq!(batch.channels, 10);
    assert_eq!(batch.timestamps_ns[0], 0);
}

#[test]
fn synthetic_parquet_indexes_into_catalog_entries() {
    let path = std::env::temp_dir().join(format!(
        "robotics_cli_{}_{}.parquet",
        std::process::id(),
        "roundtrip"
    ));

    let written_row_groups = write_synthetic_parquet(
        &path,
        "humanoid_02",
        "session_cli",
        SyntheticConfig {
            hz: 20.0,
            duration_ns: 950_000_000,
            start_ts_ns: 42,
        },
        10,
    )
    .unwrap();
    let indexed = index_parquet_file(&path).unwrap();

    assert_eq!(written_row_groups, 2);
    assert_eq!(indexed.len(), 2);
    assert_eq!(indexed[0].robot_id, "humanoid_02");
    assert_eq!(indexed[0].session_id, "session_cli");
    assert_eq!(indexed[0].start_ts_ns, 42);
    assert!(indexed.iter().all(|entry| entry.byte_length > 0));

    std::fs::remove_file(path).ok();
}

#[test]
fn demo_command_runs_parquet_e2e() {
    let path = std::env::temp_dir().join(format!(
        "robotics_cli_{}_{}.parquet",
        std::process::id(),
        "demo_e2e"
    ));
    let tensor_prefix =
        std::env::temp_dir().join(format!("robotics_cli_{}_tensor", std::process::id()));

    let output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "demo",
            "--out",
            path.to_str().expect("temp path should be valid UTF-8"),
            "--hz",
            "50",
            "--duration-ns",
            "1000000000",
            "--row-group-rows",
            "25",
            "--tensor-out",
            tensor_prefix
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("demo command should run");

    assert!(
        output.status.success(),
        "demo failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("written_row_groups=3"));
    assert!(stdout.contains("indexed_row_groups=3"));
    assert!(stdout.contains("bbox=[-0.100,2.000,-1.100,1.100,-0.100,0.100]"));
    assert!(stdout.contains("matched_windows=1"));
    assert!(stdout.contains("planned_range_reads=1"));
    assert!(stdout.contains("tensor_source_rows=25"));
    assert!(stdout.contains("tensor_shape=[15, 10]"));
    assert!(stdout.contains("tensor_values_npy="));
    assert!(stdout.contains("tensor_timestamps_npy="));

    let values_path = tensor_prefix.with_file_name(format!(
        "{}.values.npy",
        tensor_prefix.file_name().unwrap().to_string_lossy()
    ));
    let timestamps_path = tensor_prefix.with_file_name(format!(
        "{}.timestamps_ns.npy",
        tensor_prefix.file_name().unwrap().to_string_lossy()
    ));
    assert!(std::fs::read(&values_path)
        .unwrap()
        .starts_with(b"\x93NUMPY"));
    assert!(std::fs::read(&timestamps_path)
        .unwrap()
        .starts_with(b"\x93NUMPY"));

    std::fs::remove_file(path).ok();
    std::fs::remove_file(values_path).ok();
    std::fs::remove_file(timestamps_path).ok();
}

#[test]
fn catalog_build_and_tensor_command_run() {
    let source = std::env::temp_dir().join(format!(
        "robotics_cli_{}_source.parquet",
        std::process::id()
    ));
    let catalog = std::env::temp_dir().join(format!(
        "robotics_cli_{}_catalog.parquet",
        std::process::id()
    ));
    let tensor_prefix =
        std::env::temp_dir().join(format!("robotics_cli_{}_query_tensor", std::process::id()));

    write_synthetic_parquet(
        &source,
        "humanoid_01",
        "session_cli",
        SyntheticConfig {
            hz: 50.0,
            duration_ns: 1_000_000_000,
            start_ts_ns: 0,
        },
        25,
    )
    .unwrap();
    let catalog_output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "catalog",
            "build",
            "--input",
            source.to_str().expect("temp path should be valid UTF-8"),
            "--out",
            catalog.to_str().expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("catalog build should run");

    assert!(
        catalog_output.status.success(),
        "catalog build failed: {}",
        String::from_utf8_lossy(&catalog_output.stderr)
    );
    assert!(std::fs::metadata(&catalog).unwrap().len() > 0);

    let tensor_output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "tensor",
            "parquet-row-groups",
            "--input",
            source.to_str().expect("temp path should be valid UTF-8"),
            "--row-groups",
            "0",
            "--start-ts-ns",
            "0",
            "--end-ts-ns",
            "480000000",
            "--hz",
            "30",
            "--out",
            tensor_prefix
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("tensor command should run");

    assert!(
        tensor_output.status.success(),
        "tensor command failed: {}",
        String::from_utf8_lossy(&tensor_output.stderr)
    );
    let stdout = String::from_utf8_lossy(&tensor_output.stdout);
    assert!(stdout.contains("tensor_shape=[15, 10]"));

    let values_path = tensor_prefix.with_file_name(format!(
        "{}.values.npy",
        tensor_prefix.file_name().unwrap().to_string_lossy()
    ));
    let timestamps_path = tensor_prefix.with_file_name(format!(
        "{}.timestamps_ns.npy",
        tensor_prefix.file_name().unwrap().to_string_lossy()
    ));
    assert!(std::fs::read(&values_path)
        .unwrap()
        .starts_with(b"\x93NUMPY"));
    assert!(std::fs::read(&timestamps_path)
        .unwrap()
        .starts_with(b"\x93NUMPY"));

    std::fs::remove_file(source).ok();
    std::fs::remove_file(catalog).ok();
    std::fs::remove_file(values_path).ok();
    std::fs::remove_file(timestamps_path).ok();
}
