use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
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

#[derive(Debug, Clone, PartialEq)]
pub struct ImuCatalogEntry {
    pub robot_id: String,
    pub session_id: String,
    pub file_uri: String,
    pub row_group_id: u32,
    pub start_ts_ns: i64,
    pub end_ts_ns: i64,
    pub min_ax: f64,
    pub max_ax: f64,
    pub min_ay: f64,
    pub max_ay: f64,
    pub min_az: f64,
    pub max_az: f64,
    pub min_gx: f64,
    pub max_gx: f64,
    pub min_gy: f64,
    pub max_gy: f64,
    pub min_gz: f64,
    pub max_gz: f64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub row_count: u64,
    pub gap_count: u64,
    pub max_gap_ns: i64,
    pub max_gap_start_ts_ns: i64,
    pub max_gap_end_ts_ns: i64,
    pub nominal_dt_ns: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaCatalogEntry {
    pub robot_id: String,
    pub session_id: String,
    pub file_uri: String,
    pub modality: String,
    pub stream_id: String,
    pub row_group_id: u32,
    pub start_ts_ns: i64,
    pub end_ts_ns: i64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub row_count: u64,
    pub min_x: Option<f64>,
    pub max_x: Option<f64>,
    pub min_y: Option<f64>,
    pub max_y: Option<f64>,
    pub min_z: Option<f64>,
    pub max_z: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GapMetadata {
    pub gap_count: u64,
    pub max_gap_ns: i64,
    pub max_gap_start_ts_ns: i64,
    pub max_gap_end_ts_ns: i64,
    pub nominal_dt_ns: i64,
}

impl GapMetadata {
    fn none() -> Self {
        Self {
            gap_count: 0,
            max_gap_ns: 0,
            max_gap_start_ts_ns: 0,
            max_gap_end_ts_ns: 0,
            nominal_dt_ns: 0,
        }
    }
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
            gap_count: 0,
            max_gap_ns: 0,
            max_gap_start_ts_ns: 0,
            max_gap_end_ts_ns: 0,
            nominal_dt_ns: 1_000_000_000,
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
        Field::new("gap_count", DataType::UInt64, false),
        Field::new("max_gap_ns", DataType::Int64, false),
        Field::new("max_gap_start_ts_ns", DataType::Int64, false),
        Field::new("max_gap_end_ts_ns", DataType::Int64, false),
        Field::new("nominal_dt_ns", DataType::Int64, false),
    ]))
}

pub fn imu_catalog_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("robot_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("file_uri", DataType::Utf8, false),
        Field::new("row_group_id", DataType::UInt32, false),
        Field::new("start_ts_ns", DataType::Int64, false),
        Field::new("end_ts_ns", DataType::Int64, false),
        Field::new("min_ax", DataType::Float64, false),
        Field::new("max_ax", DataType::Float64, false),
        Field::new("min_ay", DataType::Float64, false),
        Field::new("max_ay", DataType::Float64, false),
        Field::new("min_az", DataType::Float64, false),
        Field::new("max_az", DataType::Float64, false),
        Field::new("min_gx", DataType::Float64, false),
        Field::new("max_gx", DataType::Float64, false),
        Field::new("min_gy", DataType::Float64, false),
        Field::new("max_gy", DataType::Float64, false),
        Field::new("min_gz", DataType::Float64, false),
        Field::new("max_gz", DataType::Float64, false),
        Field::new("byte_offset", DataType::UInt64, false),
        Field::new("byte_length", DataType::UInt64, false),
        Field::new("row_count", DataType::UInt64, false),
        Field::new("gap_count", DataType::UInt64, false),
        Field::new("max_gap_ns", DataType::Int64, false),
        Field::new("max_gap_start_ts_ns", DataType::Int64, false),
        Field::new("max_gap_end_ts_ns", DataType::Int64, false),
        Field::new("nominal_dt_ns", DataType::Int64, false),
    ]))
}

pub fn media_catalog_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("robot_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("file_uri", DataType::Utf8, false),
        Field::new("modality", DataType::Utf8, false),
        Field::new("stream_id", DataType::Utf8, false),
        Field::new("row_group_id", DataType::UInt32, false),
        Field::new("start_ts_ns", DataType::Int64, false),
        Field::new("end_ts_ns", DataType::Int64, false),
        Field::new("byte_offset", DataType::UInt64, false),
        Field::new("byte_length", DataType::UInt64, false),
        Field::new("row_count", DataType::UInt64, false),
        Field::new("min_x", DataType::Float64, true),
        Field::new("max_x", DataType::Float64, true),
        Field::new("min_y", DataType::Float64, true),
        Field::new("max_y", DataType::Float64, true),
        Field::new("min_z", DataType::Float64, true),
        Field::new("max_z", DataType::Float64, true),
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

pub fn write_imu_catalog_parquet(
    path: impl AsRef<Path>,
    entries: &[ImuCatalogEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|err| RoboticsError::Io(err.to_string()))?;
    }

    let schema = imu_catalog_schema();
    let batch = imu_catalog_batch(schema.clone(), entries)?;
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

pub fn write_media_catalog_parquet(
    path: impl AsRef<Path>,
    entries: &[MediaCatalogEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|err| RoboticsError::Io(err.to_string()))?;
    }

    let schema = media_catalog_schema();
    let batch = media_catalog_batch(schema.clone(), entries)?;
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
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.gap_count),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gap_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gap_start_ts_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gap_end_ts_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.nominal_dt_ns),
        )),
    ];
    RecordBatch::try_new(schema, arrays).map_err(|err| RoboticsError::Io(err.to_string()))
}

fn imu_catalog_batch(schema: SchemaRef, entries: &[ImuCatalogEntry]) -> Result<RecordBatch> {
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
            entries.iter().map(|entry| entry.min_ax),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_ax),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_ay),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_ay),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_az),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_az),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_gx),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gx),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_gy),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gy),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.min_gz),
        )),
        Arc::new(Float64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gz),
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
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.gap_count),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gap_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gap_start_ts_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.max_gap_end_ts_ns),
        )),
        Arc::new(Int64Array::from_iter_values(
            entries.iter().map(|entry| entry.nominal_dt_ns),
        )),
    ];
    RecordBatch::try_new(schema, arrays).map_err(|err| RoboticsError::Io(err.to_string()))
}

fn media_catalog_batch(schema: SchemaRef, entries: &[MediaCatalogEntry]) -> Result<RecordBatch> {
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
        Arc::new(StringArray::from_iter_values(
            entries.iter().map(|entry| entry.modality.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            entries.iter().map(|entry| entry.stream_id.as_str()),
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
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.byte_offset),
        )),
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.byte_length),
        )),
        Arc::new(UInt64Array::from_iter_values(
            entries.iter().map(|entry| entry.row_count),
        )),
        Arc::new(Float64Array::from_iter(
            entries.iter().map(|entry| entry.min_x),
        )),
        Arc::new(Float64Array::from_iter(
            entries.iter().map(|entry| entry.max_x),
        )),
        Arc::new(Float64Array::from_iter(
            entries.iter().map(|entry| entry.min_y),
        )),
        Arc::new(Float64Array::from_iter(
            entries.iter().map(|entry| entry.max_y),
        )),
        Arc::new(Float64Array::from_iter(
            entries.iter().map(|entry| entry.min_z),
        )),
        Arc::new(Float64Array::from_iter(
            entries.iter().map(|entry| entry.max_z),
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
    index_parquet_file_with_uri_and_gap_threshold(path, file_uri, None)
}

pub fn index_parquet_file_with_uri_and_gap_threshold(
    path: impl AsRef<Path>,
    file_uri: impl Into<String>,
    gap_threshold_ns: Option<i64>,
) -> Result<Vec<CatalogEntry>> {
    let path = path.as_ref();
    let reader =
        SerializedFileReader::try_from(path).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let metadata = reader.metadata();
    let file_uri = file_uri.into();
    let mut entries = Vec::with_capacity(metadata.num_row_groups());

    for row_group_id in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(row_group_id);
        let columns = columns_by_name(row_group);
        let (byte_offset, byte_length) = row_group_byte_range(row_group)?;
        let robot_id = string_min(&columns, "robot_id")?;
        let session_id = string_min(&columns, "session_id")?;
        let gap = row_group_gap_metadata(path, row_group_id, gap_threshold_ns)?;

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
            gap_count: gap.gap_count,
            max_gap_ns: gap.max_gap_ns,
            max_gap_start_ts_ns: gap.max_gap_start_ts_ns,
            max_gap_end_ts_ns: gap.max_gap_end_ts_ns,
            nominal_dt_ns: gap.nominal_dt_ns,
        });
    }

    Ok(entries)
}

pub fn index_imu_parquet_file(path: impl AsRef<Path>) -> Result<Vec<ImuCatalogEntry>> {
    let path = path.as_ref();
    let file_uri = path.to_string_lossy().to_string();
    index_imu_parquet_file_with_uri(path, file_uri)
}

pub fn index_imu_parquet_file_with_uri(
    path: impl AsRef<Path>,
    file_uri: impl Into<String>,
) -> Result<Vec<ImuCatalogEntry>> {
    index_imu_parquet_file_with_uri_and_gap_threshold(path, file_uri, None)
}

pub fn index_imu_parquet_file_with_uri_and_gap_threshold(
    path: impl AsRef<Path>,
    file_uri: impl Into<String>,
    gap_threshold_ns: Option<i64>,
) -> Result<Vec<ImuCatalogEntry>> {
    let path = path.as_ref();
    let reader =
        SerializedFileReader::try_from(path).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let metadata = reader.metadata();
    let file_uri = file_uri.into();
    let mut entries = Vec::with_capacity(metadata.num_row_groups());

    for row_group_id in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(row_group_id);
        let columns = columns_by_name(row_group);
        let (byte_offset, byte_length) = row_group_byte_range(row_group)?;
        let robot_id = string_min(&columns, "robot_id")?;
        let session_id = string_min(&columns, "session_id")?;
        let gap = row_group_gap_metadata(path, row_group_id, gap_threshold_ns)?;

        entries.push(ImuCatalogEntry {
            robot_id,
            session_id,
            file_uri: file_uri.clone(),
            row_group_id: row_group_id as u32,
            start_ts_ns: int64_min(&columns, "timestamp_ns")?,
            end_ts_ns: int64_max(&columns, "timestamp_ns")?,
            min_ax: double_min(&columns, "ax")?,
            max_ax: double_max(&columns, "ax")?,
            min_ay: double_min(&columns, "ay")?,
            max_ay: double_max(&columns, "ay")?,
            min_az: double_min(&columns, "az")?,
            max_az: double_max(&columns, "az")?,
            min_gx: double_min(&columns, "gx")?,
            max_gx: double_max(&columns, "gx")?,
            min_gy: double_min(&columns, "gy")?,
            max_gy: double_max(&columns, "gy")?,
            min_gz: double_min(&columns, "gz")?,
            max_gz: double_max(&columns, "gz")?,
            byte_offset,
            byte_length,
            row_count: row_group.num_rows() as u64,
            gap_count: gap.gap_count,
            max_gap_ns: gap.max_gap_ns,
            max_gap_start_ts_ns: gap.max_gap_start_ts_ns,
            max_gap_end_ts_ns: gap.max_gap_end_ts_ns,
            nominal_dt_ns: gap.nominal_dt_ns,
        });
    }

    Ok(entries)
}

pub fn index_media_parquet_file_with_uri(
    path: impl AsRef<Path>,
    file_uri: impl Into<String>,
    modality: impl Into<String>,
    stream_id: impl Into<String>,
) -> Result<Vec<MediaCatalogEntry>> {
    let path = path.as_ref();
    let reader =
        SerializedFileReader::try_from(path).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let metadata = reader.metadata();
    let file_uri = file_uri.into();
    let modality = modality.into();
    let stream_id = stream_id.into();
    let mut entries = Vec::with_capacity(metadata.num_row_groups());

    for row_group_id in 0..metadata.num_row_groups() {
        let row_group = metadata.row_group(row_group_id);
        let columns = columns_by_name(row_group);
        let (byte_offset, byte_length) = row_group_byte_range(row_group)?;
        entries.push(MediaCatalogEntry {
            robot_id: string_min(&columns, "robot_id")?,
            session_id: string_min(&columns, "session_id")?,
            file_uri: file_uri.clone(),
            modality: modality.clone(),
            stream_id: stream_id.clone(),
            row_group_id: row_group_id as u32,
            start_ts_ns: int64_min(&columns, "timestamp_ns")?,
            end_ts_ns: int64_max(&columns, "timestamp_ns")?,
            byte_offset,
            byte_length,
            row_count: row_group.num_rows() as u64,
            min_x: optional_double_min(&columns, "x")?,
            max_x: optional_double_max(&columns, "x")?,
            min_y: optional_double_min(&columns, "y")?,
            max_y: optional_double_max(&columns, "y")?,
            min_z: optional_double_min(&columns, "z")?,
            max_z: optional_double_max(&columns, "z")?,
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

fn row_group_gap_metadata(
    path: &Path,
    row_group_id: usize,
    gap_threshold_ns: Option<i64>,
) -> Result<GapMetadata> {
    let file = File::open(path).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut reader = builder
        .with_row_groups(vec![row_group_id])
        .build()
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut timestamps = Vec::new();
    for batch in &mut reader {
        let batch = batch.map_err(|err| RoboticsError::Io(err.to_string()))?;
        let timestamp_ns = typed_batch_column::<Int64Array>(&batch, "timestamp_ns")?;
        timestamps.extend((0..timestamp_ns.len()).map(|row| timestamp_ns.value(row)));
    }
    Ok(gap_metadata_from_timestamps(&timestamps, gap_threshold_ns))
}

pub fn gap_metadata_from_timestamps(
    timestamps_ns: &[i64],
    gap_threshold_ns: Option<i64>,
) -> GapMetadata {
    if timestamps_ns.len() < 2 {
        return GapMetadata::none();
    }

    let mut sorted = timestamps_ns.to_vec();
    sorted.sort_unstable();
    let positive_diffs = sorted
        .windows(2)
        .filter_map(|pair| pair[1].checked_sub(pair[0]))
        .filter(|diff| *diff > 0)
        .collect::<Vec<_>>();
    if positive_diffs.is_empty() {
        return GapMetadata::none();
    }

    let nominal_dt_ns = median_i64(&positive_diffs);
    let threshold = gap_threshold_ns
        .unwrap_or_else(|| nominal_dt_ns.saturating_mul(5))
        .max(1);
    let mut gap_count = 0;
    let mut max_gap_ns = 0;
    let mut max_gap_start_ts_ns = 0;
    let mut max_gap_end_ts_ns = 0;

    for pair in sorted.windows(2) {
        let Some(diff) = pair[1].checked_sub(pair[0]) else {
            continue;
        };
        if diff > threshold {
            gap_count += 1;
            if diff > max_gap_ns {
                max_gap_ns = diff;
                max_gap_start_ts_ns = pair[0];
                max_gap_end_ts_ns = pair[1];
            }
        }
    }

    GapMetadata {
        gap_count,
        max_gap_ns,
        max_gap_start_ts_ns,
        max_gap_end_ts_ns,
        nominal_dt_ns,
    }
}

fn median_i64(values: &[i64]) -> i64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    if sorted.len().is_multiple_of(2) {
        let upper = sorted.len() / 2;
        ((sorted[upper - 1] as f64 + sorted[upper] as f64) / 2.0).round() as i64
    } else {
        sorted[sorted.len() / 2]
    }
}

fn typed_batch_column<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
    let column = batch.column(index);
    if column.null_count() > 0 {
        return Err(RoboticsError::InvalidArgument(format!(
            "parquet column {name} contains nulls"
        )));
    }
    column.as_any().downcast_ref::<T>().ok_or_else(|| {
        RoboticsError::InvalidArgument(format!("parquet column {name} has unexpected type"))
    })
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

fn optional_double_min(
    columns: &HashMap<String, &ColumnChunkMetaData>,
    name: &str,
) -> Result<Option<f64>> {
    if columns.contains_key(name) {
        double_min(columns, name).map(Some)
    } else {
        Ok(None)
    }
}

fn optional_double_max(
    columns: &HashMap<String, &ColumnChunkMetaData>,
    name: &str,
) -> Result<Option<f64>> {
    if columns.contains_key(name) {
        double_max(columns, name).map(Some)
    } else {
        Ok(None)
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
    use robotics_core::{BoundingBox, PoseSample, QuerySpec};

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
        assert_eq!(entries[0].gap_count, 0);
        assert_eq!(entries[0].max_gap_ns, 0);
        assert_eq!(entries[0].nominal_dt_ns, 100_000_000);
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
        assert!(reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .any(|column| column.name() == "gap_count"));
        std::fs::remove_file(source_path).ok();
        std::fs::remove_file(catalog_path).ok();
    }

    #[test]
    fn indexes_pose_temporal_gaps() {
        let path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_pose_gap.parquet",
            std::process::id()
        ));
        let samples = vec![
            pose_sample_at(0, 0.0),
            pose_sample_at(100, 1.0),
            pose_sample_at(200, 2.0),
            pose_sample_at(1_000, 3.0),
        ];
        robotics_ingest::write_pose_parquet(&path, &samples, 4).unwrap();

        let entries = index_parquet_file(&path).unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].gap_count, 1);
        assert_eq!(entries[0].nominal_dt_ns, 100);
        assert_eq!(entries[0].max_gap_ns, 800);
        assert_eq!(entries[0].max_gap_start_ts_ns, 200);
        assert_eq!(entries[0].max_gap_end_ts_ns, 1_000);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn indexes_imu_parquet_row_groups_from_metadata() {
        let path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_imu.parquet",
            std::process::id()
        ));
        let samples = vec![
            robotics_ingest::ImuSample {
                timestamp_ns: 100,
                robot_id: "mav0".to_string(),
                session_id: "room1".to_string(),
                ax: 1.0,
                ay: 2.0,
                az: 3.0,
                gx: 0.1,
                gy: 0.2,
                gz: 0.3,
            },
            robotics_ingest::ImuSample {
                timestamp_ns: 200,
                robot_id: "mav0".to_string(),
                session_id: "room1".to_string(),
                ax: 4.0,
                ay: 5.0,
                az: 6.0,
                gx: 0.4,
                gy: 0.5,
                gz: 0.6,
            },
            robotics_ingest::ImuSample {
                timestamp_ns: 300,
                robot_id: "mav0".to_string(),
                session_id: "room1".to_string(),
                ax: 7.0,
                ay: 8.0,
                az: 9.0,
                gx: 0.7,
                gy: 0.8,
                gz: 0.9,
            },
        ];
        robotics_ingest::write_imu_parquet(&path, &samples, 2).unwrap();

        let entries = index_imu_parquet_file(&path).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].robot_id, "mav0");
        assert_eq!(entries[0].session_id, "room1");
        assert_eq!(entries[0].start_ts_ns, 100);
        assert_eq!(entries[0].end_ts_ns, 200);
        assert_eq!(entries[0].min_ax, 1.0);
        assert_eq!(entries[0].max_gz, 0.6);
        assert_eq!(entries[0].row_count, 2);
        assert_eq!(entries[0].gap_count, 0);
        assert_eq!(entries[0].nominal_dt_ns, 100);
        assert!(entries[0].byte_length > 0);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn indexes_imu_temporal_gaps() {
        let path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_imu_gap.parquet",
            std::process::id()
        ));
        let samples = vec![
            imu_sample_at(0, 0.0),
            imu_sample_at(100, 1.0),
            imu_sample_at(200, 2.0),
            imu_sample_at(1_000, 3.0),
        ];
        robotics_ingest::write_imu_parquet(&path, &samples, 4).unwrap();

        let entries = index_imu_parquet_file(&path).unwrap();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].gap_count, 1);
        assert_eq!(entries[0].nominal_dt_ns, 100);
        assert_eq!(entries[0].max_gap_ns, 800);
        assert_eq!(entries[0].max_gap_start_ts_ns, 200);
        assert_eq!(entries[0].max_gap_end_ts_ns, 1_000);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn writes_imu_catalog_as_parquet() {
        let source_path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_imu_source.parquet",
            std::process::id()
        ));
        let catalog_path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_imu_catalog.parquet",
            std::process::id()
        ));
        let samples = vec![
            robotics_ingest::ImuSample {
                timestamp_ns: 100,
                robot_id: "mav0".to_string(),
                session_id: "room1".to_string(),
                ax: 1.0,
                ay: 2.0,
                az: 3.0,
                gx: 0.1,
                gy: 0.2,
                gz: 0.3,
            },
            robotics_ingest::ImuSample {
                timestamp_ns: 200,
                robot_id: "mav0".to_string(),
                session_id: "room1".to_string(),
                ax: 4.0,
                ay: 5.0,
                az: 6.0,
                gx: 0.4,
                gy: 0.5,
                gz: 0.6,
            },
        ];
        robotics_ingest::write_imu_parquet(&source_path, &samples, 1).unwrap();
        let entries = index_imu_parquet_file(&source_path).unwrap();

        write_imu_catalog_parquet(&catalog_path, &entries).unwrap();
        let reader = SerializedFileReader::try_from(catalog_path.as_path()).unwrap();

        assert_eq!(
            reader.metadata().file_metadata().num_rows(),
            entries.len() as i64
        );
        std::fs::remove_file(source_path).ok();
        std::fs::remove_file(catalog_path).ok();
    }

    #[test]
    fn indexes_and_writes_media_catalog() {
        let source_path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_media_source.parquet",
            std::process::id()
        ));
        let catalog_path = std::env::temp_dir().join(format!(
            "robotics_catalog_{}_media_catalog.parquet",
            std::process::id()
        ));
        robotics_ingest::write_synthetic_parquet(
            &source_path,
            "robot_cam",
            "media_session",
            robotics_ingest::SyntheticConfig {
                hz: 10.0,
                duration_ns: 900_000_000,
                start_ts_ns: 5_000,
            },
            5,
        )
        .unwrap();

        let entries = index_media_parquet_file_with_uri(
            &source_path,
            "file:///tmp/cam0.parquet",
            "camera",
            "cam0",
        )
        .unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].robot_id, "robot_cam");
        assert_eq!(entries[0].modality, "camera");
        assert_eq!(entries[0].stream_id, "cam0");
        assert_eq!(entries[0].start_ts_ns, 5_000);
        assert_eq!(entries[0].row_count, 5);
        assert!(entries[0].byte_length > 0);
        assert!(entries[0].min_x.is_some());

        write_media_catalog_parquet(&catalog_path, &entries).unwrap();
        let reader = SerializedFileReader::try_from(catalog_path.as_path()).unwrap();
        assert_eq!(
            reader.metadata().file_metadata().num_rows(),
            entries.len() as i64
        );
        assert!(reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .any(|column| column.name() == "stream_id"));

        std::fs::remove_file(source_path).ok();
        std::fs::remove_file(catalog_path).ok();
    }

    fn pose_sample_at(timestamp_ns: i64, value: f64) -> PoseSample {
        PoseSample {
            timestamp_ns,
            robot_id: "robot".to_string(),
            session_id: "session".to_string(),
            x: value,
            y: 0.0,
            z: 0.0,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            vx: 1.0,
            vy: 0.0,
            vz: 0.0,
        }
    }

    fn imu_sample_at(timestamp_ns: i64, value: f64) -> robotics_ingest::ImuSample {
        robotics_ingest::ImuSample {
            timestamp_ns,
            robot_id: "mav0".to_string(),
            session_id: "room1".to_string(),
            ax: value,
            ay: value + 1.0,
            az: value + 2.0,
            gx: value + 3.0,
            gy: value + 4.0,
            gz: value + 5.0,
        }
    }
}
