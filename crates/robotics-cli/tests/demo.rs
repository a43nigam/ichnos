use robotics_catalog::{
    generate_fake_catalog, index_parquet_file, index_parquet_file_with_uri, query_catalog,
    write_catalog_parquet, FakeCatalogConfig,
};
use robotics_core::QuerySpec;
use robotics_ingest::{
    generate_synthetic_pose, write_camera_parquet, write_synthetic_parquet, CameraFrame,
    SyntheticConfig,
};
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
    let enforced_tensor_prefix = std::env::temp_dir().join(format!(
        "robotics_cli_{}_query_tensor_enforced",
        std::process::id()
    ));
    let manifest_path = std::env::temp_dir().join(format!(
        "robotics_cli_{}_query_manifest.json",
        std::process::id()
    ));

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

    let indexed_entries = index_parquet_file(&source).unwrap();
    let first_entry = indexed_entries.first().unwrap();
    let audit_ranges = format!(
        "{}:{}:{}",
        first_entry.row_group_id, first_entry.byte_offset, first_entry.byte_length
    );
    let enforced_output = Command::new(env!("CARGO_BIN_EXE_robotics"))
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
            enforced_tensor_prefix
                .to_str()
                .expect("temp path should be valid UTF-8"),
            "--audit-ranges",
            &audit_ranges,
            "--enforce-ranges",
            "--footer-allowance-bytes",
            "16777216",
            "--manifest-out",
            manifest_path
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("enforced tensor command should run");

    assert!(
        enforced_output.status.success(),
        "enforced tensor command failed: {}",
        String::from_utf8_lossy(&enforced_output.stderr)
    );
    let enforced_stdout = String::from_utf8_lossy(&enforced_output.stdout);
    assert!(enforced_stdout.contains("range_enforced=true"));
    assert!(enforced_stdout.contains("footer_allowance_bytes=16777216"));
    assert!(std::fs::read_to_string(&manifest_path)
        .unwrap()
        .contains("\"footer_allowance_bytes\": 16777216"));
    let low_allowance_output = Command::new(env!("CARGO_BIN_EXE_robotics"))
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
            "--audit-ranges",
            &audit_ranges,
            "--enforce-ranges",
            "--footer-allowance-bytes",
            "1",
        ])
        .output()
        .expect("low allowance tensor command should run");

    assert!(!low_allowance_output.status.success());
    let low_allowance_stderr = String::from_utf8_lossy(&low_allowance_output.stderr);
    assert!(low_allowance_stderr.contains("footer_allowance_bytes=1"));
    assert!(low_allowance_stderr.contains("authorized_ranges="));

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
    std::fs::remove_file(manifest_path).ok();
    std::fs::remove_file(enforced_tensor_prefix.with_file_name(format!(
        "{}.values.npy",
        enforced_tensor_prefix
            .file_name()
            .unwrap()
            .to_string_lossy()
    )))
    .ok();
    std::fs::remove_file(enforced_tensor_prefix.with_file_name(format!(
        "{}.timestamps_ns.npy",
        enforced_tensor_prefix
            .file_name()
            .unwrap()
            .to_string_lossy()
    )))
    .ok();
}

#[test]
fn camera_media_row_groups_command_writes_selected_frames() {
    let source = std::env::temp_dir().join(format!(
        "robotics_cli_{}_camera_source.parquet",
        std::process::id()
    ));
    let out_dir =
        std::env::temp_dir().join(format!("robotics_cli_{}_camera_frames", std::process::id()));
    let manifest_path = std::env::temp_dir().join(format!(
        "robotics_cli_{}_camera_manifest.json",
        std::process::id()
    ));
    let frames = vec![
        CameraFrame {
            timestamp_ns: 100,
            robot_id: "mav0".to_string(),
            session_id: "room1".to_string(),
            stream_id: "cam0".to_string(),
            frame_path: "100.png".to_string(),
            camera_bytes: b"frame100".to_vec(),
        },
        CameraFrame {
            timestamp_ns: 200,
            robot_id: "mav0".to_string(),
            session_id: "room1".to_string(),
            stream_id: "cam0".to_string(),
            frame_path: "200.png".to_string(),
            camera_bytes: b"frame200".to_vec(),
        },
    ];
    write_camera_parquet(&source, &frames, 1).unwrap();
    let entries = robotics_catalog::index_media_parquet_file_with_uri(
        &source,
        format!("file://{}", source.display()),
        "camera",
        "cam0",
    )
    .unwrap();
    let second = &entries[1];
    let audit_ranges = format!(
        "{}:{}:{}",
        second.row_group_id, second.byte_offset, second.byte_length
    );

    let output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "media",
            "camera-row-groups",
            "--input",
            source.to_str().expect("temp path should be valid UTF-8"),
            "--row-groups",
            "1",
            "--out",
            out_dir.to_str().expect("temp path should be valid UTF-8"),
            "--audit-ranges",
            &audit_ranges,
            "--enforce-ranges",
            "--footer-allowance-bytes",
            "16777216",
            "--manifest-out",
            manifest_path
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("camera media command should run");

    assert!(
        output.status.success(),
        "camera media command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("media_frames=1"));
    assert!(stdout.contains("range_enforced=true"));
    assert_eq!(
        std::fs::read(out_dir.join("cam0").join("200.png")).unwrap(),
        b"frame200"
    );
    assert!(!out_dir.join("cam0").join("100.png").exists());
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest.contains("\"timestamp_ns\": 200"));
    assert!(manifest.contains("\"footer_allowance_bytes\": 16777216"));

    std::fs::remove_file(source).ok();
    std::fs::remove_file(manifest_path).ok();
    std::fs::remove_dir_all(out_dir).ok();
}

#[test]
fn catalog_duckdb_build_command_creates_persistent_tables() {
    if !python_duckdb_available() {
        eprintln!("skipping DuckDB catalog build test because python duckdb is not installed");
        return;
    }

    let source = std::env::temp_dir().join(format!(
        "robotics_cli_{}_duckdb_source.parquet",
        std::process::id()
    ));
    let catalog = std::env::temp_dir().join(format!(
        "robotics_cli_{}_duckdb_catalog.parquet",
        std::process::id()
    ));
    let media_catalog = std::env::temp_dir().join(format!(
        "robotics_cli_{}_media_catalog.parquet",
        std::process::id()
    ));
    let catalog_db =
        std::env::temp_dir().join(format!("robotics_cli_{}_fleet.duckdb", std::process::id()));

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
    let entries =
        index_parquet_file_with_uri(&source, format!("file://{}", source.display())).unwrap();
    write_catalog_parquet(&catalog, &entries).unwrap();
    let media_output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "catalog",
            "build-media",
            "--input",
            source.to_str().expect("temp path should be valid UTF-8"),
            "--out",
            media_catalog
                .to_str()
                .expect("temp path should be valid UTF-8"),
            "--modality",
            "camera",
            "--stream-id",
            "cam0",
            "--uri",
            "file:///tmp/cam0.parquet",
        ])
        .output()
        .expect("catalog build-media should run");
    assert!(
        media_output.status.success(),
        "catalog build-media failed: {}",
        String::from_utf8_lossy(&media_output.stderr)
    );
    assert!(String::from_utf8_lossy(&media_output.stdout).contains("indexed_row_groups=3"));

    let output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "catalog",
            "duckdb-build",
            "--pose-catalog",
            catalog.to_str().expect("temp path should be valid UTF-8"),
            "--media-catalog",
            media_catalog
                .to_str()
                .expect("temp path should be valid UTF-8"),
            "--out",
            catalog_db
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("catalog duckdb-build should run");

    assert!(
        output.status.success(),
        "catalog duckdb-build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("pose_row_groups=3"));
    assert!(stdout.contains("imu_row_groups=0"));
    assert!(stdout.contains("media_row_groups=3"));
    assert!(stdout.contains("spatial_index=tile"));
    assert!(std::fs::metadata(&catalog_db).unwrap().len() > 0);
    let schema_output = Command::new("python3")
        .args([
            "-c",
            "import duckdb, sys; con=duckdb.connect(sys.argv[1]); print(con.execute(\"SELECT count(center_x), count(tile_min_x), count(hilbert_xy), count(time_bucket_ns) FROM pose_row_groups\").fetchone())",
            catalog_db
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("python duckdb schema check should run");
    assert!(
        schema_output.status.success(),
        "DuckDB schema check failed: {}",
        String::from_utf8_lossy(&schema_output.stderr)
    );
    assert!(String::from_utf8_lossy(&schema_output.stdout).contains("(3, 3, 3, 3)"));

    std::fs::remove_file(source).ok();
    std::fs::remove_file(catalog).ok();
    std::fs::remove_file(media_catalog).ok();
    std::fs::remove_file(&catalog_db).ok();
    std::fs::remove_file(catalog_db.with_extension("duckdb.wal")).ok();
}

#[test]
fn catalog_fake_duckdb_command_creates_large_demo_catalog() {
    if !python_duckdb_available() {
        eprintln!("skipping fake DuckDB catalog test because python duckdb is not installed");
        return;
    }

    let catalog_db = std::env::temp_dir().join(format!(
        "robotics_cli_{}_fake_fleet.duckdb",
        std::process::id()
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_robotics"))
        .args([
            "catalog",
            "fake-duckdb",
            "--sessions",
            "128",
            "--out",
            catalog_db
                .to_str()
                .expect("temp path should be valid UTF-8"),
        ])
        .output()
        .expect("catalog fake-duckdb should run");

    assert!(
        output.status.success(),
        "catalog fake-duckdb failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("pose_row_groups=128"));
    assert!(stdout.contains("fake_sessions=128"));
    assert!(stdout.contains("spatial_index=hilbert"));
    assert!(std::fs::metadata(&catalog_db).unwrap().len() > 0);

    std::fs::remove_file(&catalog_db).ok();
    std::fs::remove_file(catalog_db.with_extension("duckdb.wal")).ok();
}

fn python_duckdb_available() -> bool {
    Command::new("python3")
        .args(["-c", "import duckdb"])
        .status()
        .is_ok_and(|status| status.success())
}
