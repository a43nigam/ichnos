use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use robotics_core::{CatalogEntry, QuerySpec, WindowRef};
use robotics_core::{Result, RoboticsError};

const HOUR_NS: i64 = 3_600_000_000_000;

#[derive(Debug, Clone, Copy)]
pub struct FakeCatalogConfig {
    pub sessions: usize,
    pub robot_count: usize,
    pub start_ts_ns: i64,
    pub session_duration_ns: i64,
}

impl Default for FakeCatalogConfig {
    fn default() -> Self {
        Self {
            sessions: 50_000,
            robot_count: 20,
            start_ts_ns: 1_700_000_000_000_000_000,
            session_duration_ns: HOUR_NS,
        }
    }
}

pub fn generate_fake_catalog(config: FakeCatalogConfig) -> Vec<CatalogEntry> {
    let robot_count = config.robot_count.max(1);
    let mut entries = Vec::with_capacity(config.sessions);

    for i in 0..config.sessions {
        let robot_index = i % robot_count;
        let start_ts_ns = config.start_ts_ns + (i as i64 * config.session_duration_ns);
        let end_ts_ns = start_ts_ns + config.session_duration_ns;
        let phase = (i % 100) as f64;
        let min_x = -50.0 + phase * 0.25;
        let max_x = min_x + 30.0 + (robot_index as f64 % 5.0);
        let min_y = -25.0 + (robot_index as f64);
        let max_y = min_y + 20.0;
        let max_velocity = 2.0 + (i % 9) as f64;

        entries.push(CatalogEntry {
            robot_id: format!("humanoid_{:02}", robot_index + 1),
            session_id: format!("session_{i:06}"),
            file_uri: "data/parquet/synthetic/session.parquet".to_string(),
            row_group_id: (i % 128) as u32,
            start_ts_ns,
            end_ts_ns,
            min_x,
            max_x,
            min_y,
            max_y,
            min_z: -1.0,
            max_z: 3.0,
            min_velocity: (i % 3) as f64 * 0.25,
            max_velocity,
            byte_offset: 4096 + i as u64 * 65_536,
            byte_length: 65_536,
            row_count: 3_600,
        });
    }

    entries
}

pub fn query_catalog(entries: &[CatalogEntry], spec: &QuerySpec) -> Vec<WindowRef> {
    entries
        .iter()
        .filter(|entry| {
            spec.robot_id
                .as_ref()
                .is_none_or(|robot_id| &entry.robot_id == robot_id)
                && entry.overlaps_time(spec.start_ts_ns, spec.end_ts_ns)
                && spec
                    .bbox
                    .as_ref()
                    .is_none_or(|bbox| entry.overlaps_bbox(bbox))
                && spec
                    .min_velocity
                    .is_none_or(|min_velocity| entry.max_velocity >= min_velocity)
        })
        .map(|entry| WindowRef {
            entry: entry.clone(),
            clipped_start_ns: entry.start_ts_ns.max(spec.start_ts_ns),
            clipped_end_ns: entry.end_ts_ns.min(spec.end_ts_ns),
        })
        .collect()
}

pub fn total_selected_bytes(windows: &[WindowRef]) -> u64 {
    windows.iter().map(|window| window.entry.byte_length).sum()
}

pub fn catalog_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("robot_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("file_uri", DataType::Utf8, false),
        Field::new("row_group_id", DataType::UInt32, false),
        Field::new("start_ts_ns", DataType::Int64, false),
        Field::new("end_ts_ns", DataType::Int64, false),
        Field::new("min_x", DataType::Float64, false),
        Field::new("max_x", DataType::Float64, false),
        Field::new("min_y", DataType::Float64, false),
        Field::new("max_y", DataType::Float64, false),
        Field::new("min_z", DataType::Float64, false),
        Field::new("max_z", DataType::Float64, false),
        Field::new("min_velocity", DataType::Float64, false),
        Field::new("max_velocity", DataType::Float64, false),
        Field::new("byte_offset", DataType::UInt64, false),
        Field::new("byte_length", DataType::UInt64, false),
        Field::new("row_count", DataType::UInt64, false),
    ]))
}

pub fn write_catalog_parquet(path: impl AsRef<Path>, entries: &[CatalogEntry]) -> Result<()> {
    if entries.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|err| RoboticsError::Io(err.to_string()))?;
    }

    let schema = catalog_schema();
    let batch = catalog_batch(schema.clone(), entries)?;
    let file = File::create(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    writer
        .write(&batch)
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    writer
        .close()
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    Ok(())
}

fn catalog_batch(schema: SchemaRef, entries: &[CatalogEntry]) -> Result<RecordBatch> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            entries.iter().map(|entry| entry.robot_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            entries.iter().map(|entry| entry.session_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            entries.iter().map(|entry| entry.file_uri.as_str()),
        )),
        Arc::new(UInt32Array::from_iter_values(
            entries.iter().map(|entry| entry.row_group_id),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.start_ts_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.end_ts_ns),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_x),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_x),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_y),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_y),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_z),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_z),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_velocity),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_velocity),
        )),
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.byte_offset),
        )),
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.byte_length),
        )),
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.row_count),
        )),
    ];
    RecordBatch::try_new(schema, arrays).map_err(|err| RoboticsError::Io(err.to_string()))
}

pub fn index_parquet_file(path: impl AsRef<Path>) -> Result<Vec<CatalogEntry>> {
    let path = path.as_ref();
    let file_uri = path.to_string_lossy().to_string();
    index_parquet_file_with_uri(path, file_uri)
}

pub fn index_parquet_file_with_uri(
    path: impl AsRef<Path>,
    file_uri: impl Into<String>,
) -> Result<Vec<CatalogEntry>> {
    let reader = SerializedFileReader::try_from(path.as_ref())
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let metadata = reader.metadata();
    let file_uri = file_uri.into();
    let mut entries = Vec::with_capacity(metadata.num_row_groups());

    for row_group_id in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(row_group_id);
        let columns = columns_by_name(row_group);
        let (byte_offset, byte_length) = row_group_byte_range(row_group)?;
        let robot_id = string_min(&columns, "robot_id")?;
        let session_id = string_min(&columns, "session_id")?;

        entries.push(CatalogEntry {
            robot_id,
            session_id,
            file_uri: file_uri.clone(),
            row_group_id: row_group_id as u32,
            start_ts_ns: int64_min(&columns, "timestamp_ns")?,
            end_ts_ns: int64_max(&columns, "timestamp_ns")?,
            min_x: double_min(&columns, "x")?,
            max_x: double_max(&columns, "x")?,
            min_y: double_min(&columns, "y")?,
            max_y: double_max(&columns, "y")?,
            min_z: double_min(&columns, "z")?,
            max_z: double_max(&columns, "z")?,
            min_velocity: double_min(&columns, "velocity")?,
            max_velocity: double_max(&columns, "velocity")?,
            byte_offset,
            byte_length,
            row_count: row_group.num_rows() as u64,
        });
    }

    Ok(entries)
}

fn columns_by_name(row_group: &RowGroupMetaData) -> HashMap<String, &ColumnChunkMetaData> {
    row_group
        .columns()
        .iter()
        .map(|column| (column.column_path().string(), column))
        .collect()
}

fn row_group_byte_range(row_group: &RowGroupMetaData) -> Result<(u64, u64)> {
    if row_group.num_columns() == 0 {
        return Err(RoboticsError::InvalidArgument(
            "row group has no columns".to_string(),
        ));
    }

    let mut start = u64::MAX;
    let mut end = 0;
    for column in row_group.columns() {
        let (offset, length) = column.byte_range();
        start = start.min(offset);
        end = end.max(offset.saturating_add(length));
    }

    if start == u64::MAX || end <= start {
        return Err(RoboticsError::InvalidArgument(
            "row group has invalid byte ranges".to_string(),
        ));
    }

    Ok((start, end - start))
}

fn column_stats<'a>(
    columns: &'a HashMap<String, &ColumnChunkMetaData>,
    name: &str,
) -> Result<&'a Statistics> {
    columns
        .get(name)
        .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing parquet column {name}")))?
        .statistics()
        .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing statistics for {name}")))
}

fn int64_min(columns: &HashMap<String, &ColumnChunkMetaData>, name: &str) -> Result<i64> {
    match column_stats(columns, name)? {
        Statistics::Int64(stats) => stats
            .min_opt()
            .copied()
            .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing min for {name}"))),
        _ => Err(RoboticsError::InvalidArgument(format!(
            "column {name} is not int64"
        ))),
    }
}

fn int64_max(columns: &HashMap<String, &ColumnChunkMetaData>, name: &str) -> Result<i64> {
    match column_stats(columns, name)? {
        Statistics::Int64(stats) => stats
            .max_opt()
            .copied()
            .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing max for {name}"))),
        _ => Err(RoboticsError::InvalidArgument(format!(
            "column {name} is not int64"
        ))),
    }
}

fn double_min(columns: &HashMap<String, &ColumnChunkMetaData>, name: &str) -> Result<f64> {
    match column_stats(columns, name)? {
        Statistics::Double(stats) => stats
            .min_opt()
            .copied()
            .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing min for {name}"))),
        _ => Err(RoboticsError::InvalidArgument(format!(
            "column {name} is not double"
        ))),
    }
}

fn double_max(columns: &HashMap<String, &ColumnChunkMetaData>, name: &str) -> Result<f64> {
    match column_stats(columns, name)? {
        Statistics::Double(stats) => stats
            .max_opt()
            .copied()
            .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing max for {name}"))),
        _ => Err(RoboticsError::InvalidArgument(format!(
            "column {name} is not double"
        ))),
    }
}

fn string_min(columns: &HashMap<String, &ColumnChunkMetaData>, name: &str) -> Result<String> {
    match column_stats(columns, name)? {
        Statistics::ByteArray(stats) => stats
            .min_opt()
            .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing min for {name}")))?
            .as_utf8()
            .map(str::to_string)
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string())),
        _ => Err(RoboticsError::InvalidArgument(format!(
            "column {name} is not utf8"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use robotics_core::{BoundingBox, QuerySpec};

    use super::*;

    #[test]
    fn fake_catalog_queries_robot_time_bbox_and_velocity() {
        let catalog = generate_fake_catalog(FakeCatalogConfig {
            sessions: 100,
            ..Default::default()
        });
        let first = &catalog[0];
        let spec = QuerySpec {
            robot_id: Some(first.robot_id.clone()),
            start_ts_ns: first.start_ts_ns + 10,
            end_ts_ns: first.end_ts_ns - 10,
            bbox: Some(BoundingBox {
                min_x: first.min_x - 1.0,
                max_x: first.max_x + 1.0,
                min_y: first.min_y - 1.0,
                max_y: first.max_y + 1.0,
                min_z: first.min_z - 1.0,
                max_z: first.max_z + 1.0,
            }),
            min_velocity: Some(first.max_velocity - 0.1),
            target_hz: 30.0,
        };

        let matches = query_catalog(&catalog, &spec);

        assert!(!matches.is_empty());
        assert!(matches
            .iter()
            .all(|window| window.entry.robot_id == first.robot_id));
        assert!(matches
            .iter()
            .all(|window| window.entry.max_velocity >= first.max_velocity - 0.1));
    }

    #[test]
    fn selected_bytes_sum_matched_row_groups() {
        let catalog = generate_fake_catalog(FakeCatalogConfig {
            sessions: 3,
            ..Default::default()
        });
        let spec = QuerySpec {
            robot_id: None,
            start_ts_ns: catalog[0].start_ts_ns,
            end_ts_ns: catalog[2].end_ts_ns,
            bbox: None,
            min_velocity: None,
            target_hz: 30.0,
        };
        let matches = query_catalog(&catalog, &spec);

        assert_eq!(total_selected_bytes(&matches), 3 * 65_536);
    }

    #[test]
    fn indexes_parquet_row_groups_from_metadata() {
        let path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_{}.parquet",
            std::process::id(),
            "index"
        ));
        robotics_ingest::write_synthetic_parquet(
            &path,
            "humanoid_01",
            "session_001",
            robotics_ingest::SyntheticConfig {
                hz: 10.0,
                duration_ns: 900_000_000,
                start_ts_ns: 1_000,
            },
            5,
        )
        .unwrap();

        let entries = index_parquet_file(&path).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].robot_id, "humanoid_01");
        assert_eq!(entries[0].session_id, "session_001");
        assert_eq!(entries[0].start_ts_ns, 1_000);
        assert_eq!(entries[0].row_count, 5);
        assert!(entries[0].byte_length > 0);
        assert!(entries[1].byte_offset > entries[0].byte_offset);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn writes_hot_catalog_as_parquet() {
        let source_path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_source.parquet",
            std::process::id()
        ));
        let catalog_path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_catalog.parquet",
            std::process::id()
        ));
        robotics_ingest::write_synthetic_parquet(
            &source_path,
            "humanoid_01",
            "session_001",
            robotics_ingest::SyntheticConfig {
                hz: 10.0,
                duration_ns: 900_000_000,
                start_ts_ns: 1_000,
            },
            5,
        )
        .unwrap();
        let entries = index_parquet_file(&source_path).unwrap();

        write_catalog_parquet(&catalog_path, &entries).unwrap();
        let reader = SerializedFileReader::try_from(catalog_path.as_path()).unwrap();

        assert_eq!(
            reader.metadata().file_metadata().num_rows(),
            entries.len() as i64
        );
        std::fs::remove_file(source_path).ok();
        std::fs::remove_file(catalog_path).ok();
    }
}
