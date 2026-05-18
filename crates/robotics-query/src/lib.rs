use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::TryStreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{
    parse_url_opts, CopyOptions, GetOptions, GetRange, GetResult, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload,
    PutResult, RenameOptions,
};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use robotics_core::{ImuSample, PoseSample, Result, RoboticsError, WindowRef};
use serde::{Deserialize, Serialize};
use url::Url;

const DEFAULT_FOOTER_ALLOWANCE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeRead {
    pub uri: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteAccounting {
    pub requested_bytes: u64,
    pub transferred_bytes: u64,
    pub completed_reads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedRangeRead {
    pub read: RangeRead,
    pub bytes_read: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowGroupRange {
    pub row_group_id: u32,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditedRangeRead {
    pub row_group_id: u32,
    pub read: RangeRead,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeAudit {
    pub reads: Vec<AuditedRangeRead>,
    pub planned_read_bytes: u64,
}

impl RangeAudit {
    pub fn planned_range_reads(&self) -> usize {
        self.reads.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActualRangeRead {
    pub uri: String,
    pub offset: u64,
    pub length: u64,
    pub category: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeAuditViolation {
    pub uri: String,
    pub offset: u64,
    pub length: u64,
    pub reason: String,
    pub authorized_ranges: Vec<RangeRead>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeAuditReport {
    pub enforcement_enabled: bool,
    pub footer_allowance_bytes: u64,
    pub actual_reads: Vec<ActualRangeRead>,
    pub actual_authorized_bytes: u64,
    pub materialized_bytes: u64,
    pub footer_bytes: u64,
    pub largest_metadata_read: u64,
    pub max_footer_read_offset: u64,
    pub max_footer_read_end: u64,
    pub violations: Vec<RangeAuditViolation>,
}

impl RangeAuditReport {
    pub fn actual_read_count(&self) -> usize {
        self.actual_reads.len()
    }

    pub fn actual_read_bytes(&self) -> u64 {
        self.actual_reads.iter().map(|read| read.length).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeekManifest {
    pub version: u32,
    pub input_uri: String,
    pub modality: String,
    pub row_groups: Vec<u32>,
    pub authorized_spans: Vec<RangeRead>,
    pub actual_reads: Vec<ActualRangeRead>,
    pub planned_read_bytes: u64,
    pub actual_authorized_bytes: u64,
    pub materialized_bytes: u64,
    pub footer_allowance_bytes: u64,
    pub footer_bytes: u64,
    pub largest_metadata_read: u64,
    pub max_footer_read_offset: u64,
    pub max_footer_read_end: u64,
    pub media_planned_bytes: u64,
    pub enforcement_enabled: bool,
    pub violations: Vec<RangeAuditViolation>,
}

impl SeekManifest {
    pub fn new(
        input_uri: impl Into<String>,
        modality: impl Into<String>,
        row_groups: Vec<u32>,
        audit: Option<&RangeAudit>,
        report: Option<&RangeAuditReport>,
    ) -> Self {
        let authorized_spans = audit
            .map(|audit| {
                audit
                    .reads
                    .iter()
                    .map(|read| read.read.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let planned_read_bytes = audit.map(|audit| audit.planned_read_bytes).unwrap_or(0);
        let actual_reads = report
            .map(|report| report.actual_reads.clone())
            .unwrap_or_default();
        let actual_authorized_bytes = report
            .map(|report| report.actual_authorized_bytes)
            .unwrap_or(0);
        let materialized_bytes = report.map(|report| report.materialized_bytes).unwrap_or(0);
        let footer_allowance_bytes = report
            .map(|report| report.footer_allowance_bytes)
            .unwrap_or(DEFAULT_FOOTER_ALLOWANCE_BYTES);
        let footer_bytes = report.map(|report| report.footer_bytes).unwrap_or(0);
        let largest_metadata_read = report
            .map(|report| report.largest_metadata_read)
            .unwrap_or(0);
        let max_footer_read_offset = report
            .map(|report| report.max_footer_read_offset)
            .unwrap_or(0);
        let max_footer_read_end = report.map(|report| report.max_footer_read_end).unwrap_or(0);
        let enforcement_enabled = report
            .map(|report| report.enforcement_enabled)
            .unwrap_or(false);
        let violations = report
            .map(|report| report.violations.clone())
            .unwrap_or_default();

        Self {
            version: 1,
            input_uri: input_uri.into(),
            modality: modality.into(),
            row_groups,
            authorized_spans,
            actual_reads,
            planned_read_bytes,
            actual_authorized_bytes,
            materialized_bytes,
            footer_allowance_bytes,
            footer_bytes,
            largest_metadata_read,
            max_footer_read_offset,
            max_footer_read_end,
            media_planned_bytes: 0,
            enforcement_enabled,
            violations,
        }
    }

    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|err| RoboticsError::Io(err.to_string()))
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        serde_json::from_str(raw).map_err(|err| RoboticsError::InvalidArgument(err.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct RangeAuditor {
    uri: String,
    authorized_ranges: Vec<RangeRead>,
    footer_allowance_bytes: u64,
    state: Arc<Mutex<RangeAuditorState>>,
}

#[derive(Debug)]
struct RangeAuditorState {
    file_size: Option<u64>,
    allow_footer_reads: bool,
    actual_reads: Vec<ActualRangeRead>,
    materialized_bytes: u64,
    footer_bytes: u64,
    largest_metadata_read: u64,
    max_footer_read_offset: u64,
    max_footer_read_end: u64,
    violations: Vec<RangeAuditViolation>,
}

impl Default for RangeAuditorState {
    fn default() -> Self {
        Self {
            file_size: None,
            allow_footer_reads: true,
            actual_reads: Vec::new(),
            materialized_bytes: 0,
            footer_bytes: 0,
            largest_metadata_read: 0,
            max_footer_read_offset: 0,
            max_footer_read_end: 0,
            violations: Vec::new(),
        }
    }
}

impl RangeAuditor {
    pub fn new(
        uri: impl Into<String>,
        authorized_ranges: Vec<RangeRead>,
        footer_allowance_bytes: u64,
    ) -> Self {
        Self {
            uri: uri.into(),
            authorized_ranges,
            footer_allowance_bytes,
            state: Arc::new(Mutex::new(RangeAuditorState::default())),
        }
    }

    pub fn for_audit(audit: &RangeAudit) -> Self {
        Self::for_audit_with_footer_allowance(audit, DEFAULT_FOOTER_ALLOWANCE_BYTES)
    }

    pub fn for_audit_with_footer_allowance(
        audit: &RangeAudit,
        footer_allowance_bytes: u64,
    ) -> Self {
        let uri = audit
            .reads
            .first()
            .map(|read| read.read.uri.clone())
            .unwrap_or_default();
        let authorized_ranges = audit.reads.iter().map(|read| read.read.clone()).collect();
        Self::new(uri, authorized_ranges, footer_allowance_bytes)
    }

    pub fn set_file_size(&self, file_size: u64) {
        if let Ok(mut state) = self.state.lock() {
            state.file_size = Some(file_size);
        }
    }

    pub fn start_materialization(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.allow_footer_reads = false;
        }
    }

    pub fn record_bounded_read(&self, offset: u64, length: u64) -> Result<()> {
        bounded_range(offset, length)?;
        let category = if self.is_authorized(offset, length) {
            "authorized"
        } else if self.is_footer_read(offset, length) {
            "footer"
        } else {
            let violation = RangeAuditViolation {
                uri: self.uri.clone(),
                offset,
                length,
                reason: format!(
                    "requested range is outside authorized row-group spans and bounded footer/metadata allowance (footer_allowance_bytes={})",
                    self.footer_allowance_bytes
                ),
                authorized_ranges: self.authorized_ranges.clone(),
            };
            if let Ok(mut state) = self.state.lock() {
                state.violations.push(violation.clone());
            }
            return Err(RoboticsError::InvalidArgument(format!(
                "unauthorized cold read: uri={} requested=[{}, {}) authorized_ranges={} reason={}",
                violation.uri,
                offset,
                offset.saturating_add(length),
                format_authorized_ranges(&violation.authorized_ranges),
                violation.reason
            )));
        };

        let mut state = self
            .state
            .lock()
            .map_err(|err| RoboticsError::Io(err.to_string()))?;
        state.actual_reads.push(ActualRangeRead {
            uri: self.uri.clone(),
            offset,
            length,
            category: category.to_string(),
        });
        if category == "authorized" {
            state.materialized_bytes = state.materialized_bytes.saturating_add(length);
        } else {
            state.footer_bytes = state.footer_bytes.saturating_add(length);
            state.largest_metadata_read = state.largest_metadata_read.max(length);
            state.max_footer_read_offset = state.max_footer_read_offset.max(offset);
            state.max_footer_read_end =
                state.max_footer_read_end.max(offset.saturating_add(length));
        }
        Ok(())
    }

    pub fn report(&self) -> RangeAuditReport {
        let state = self.state.lock().expect("range auditor mutex poisoned");
        RangeAuditReport {
            enforcement_enabled: true,
            footer_allowance_bytes: self.footer_allowance_bytes,
            actual_reads: state.actual_reads.clone(),
            actual_authorized_bytes: state.materialized_bytes,
            materialized_bytes: state.materialized_bytes,
            footer_bytes: state.footer_bytes,
            largest_metadata_read: state.largest_metadata_read,
            max_footer_read_offset: state.max_footer_read_offset,
            max_footer_read_end: state.max_footer_read_end,
            violations: state.violations.clone(),
        }
    }

    fn is_authorized(&self, offset: u64, length: u64) -> bool {
        let Some(end) = offset.checked_add(length) else {
            return false;
        };
        self.authorized_ranges.iter().any(|range| {
            let range_end = range.offset.saturating_add(range.length);
            offset >= range.offset && end <= range_end
        })
    }

    fn is_footer_read(&self, offset: u64, length: u64) -> bool {
        let Ok(state) = self.state.lock() else {
            return false;
        };
        if !state.allow_footer_reads {
            return false;
        }
        let Some(file_size) = state.file_size else {
            return false;
        };
        let Some(end) = offset.checked_add(length) else {
            return false;
        };
        let footer_start = file_size.saturating_sub(self.footer_allowance_bytes);
        end <= file_size && offset >= footer_start
    }
}

pub fn plan_range_reads(windows: &[WindowRef]) -> Vec<RangeRead> {
    windows
        .iter()
        .map(|window| RangeRead {
            uri: window.entry.file_uri.clone(),
            offset: window.entry.byte_offset,
            length: window.entry.byte_length,
        })
        .collect()
}

pub fn audit_row_group_range_reads(
    uri: &str,
    row_group_ids: &[u32],
    ranges: &[RowGroupRange],
) -> Result<RangeAudit> {
    if row_group_ids.is_empty() || ranges.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let mut expected = row_group_ids.to_vec();
    expected.sort_unstable();
    expected.dedup();

    let mut actual = ranges
        .iter()
        .map(|range| range.row_group_id)
        .collect::<Vec<_>>();
    actual.sort_unstable();
    actual.dedup();

    if actual.len() != ranges.len() {
        return Err(RoboticsError::InvalidArgument(
            "audit ranges contain duplicate row_group_id values".to_string(),
        ));
    }
    if actual != expected {
        return Err(RoboticsError::InvalidArgument(format!(
            "audit range row groups {:?} do not match requested row groups {:?}",
            actual, expected
        )));
    }

    let mut reads = ranges
        .iter()
        .map(|range| {
            bounded_range(range.offset, range.length)?;
            Ok(AuditedRangeRead {
                row_group_id: range.row_group_id,
                read: RangeRead {
                    uri: uri.to_string(),
                    offset: range.offset,
                    length: range.length,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    reads.sort_by_key(|read| read.row_group_id);
    let planned_read_bytes = reads.iter().map(|read| read.read.length).sum();

    Ok(RangeAudit {
        reads,
        planned_read_bytes,
    })
}

pub fn account_reads(reads: &[RangeRead]) -> ByteAccounting {
    ByteAccounting {
        requested_bytes: reads.iter().map(|read| read.length).sum(),
        transferred_bytes: 0,
        completed_reads: reads.len(),
    }
}

pub fn read_local_range(path: impl AsRef<Path>, offset: u64, length: u64) -> Result<Vec<u8>> {
    let mut file = File::open(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut buffer = vec![0; length as usize];
    let bytes_read = file
        .read(&mut buffer)
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    buffer.truncate(bytes_read);
    Ok(buffer)
}

pub async fn read_object_store_range(read: &RangeRead) -> Result<Vec<u8>> {
    let (store, path) = object_store_for_uri(&read.uri)?;
    read_object_store_range_from(&*store, &path, read.offset, read.length).await
}

pub async fn execute_object_store_range_reads(
    reads: &[RangeRead],
) -> Result<(Vec<CompletedRangeRead>, ByteAccounting)> {
    let mut completed = Vec::with_capacity(reads.len());
    let mut transferred_bytes = 0;

    for read in reads {
        let bytes = read_object_store_range(read).await?;
        transferred_bytes += bytes.len() as u64;
        completed.push(CompletedRangeRead {
            read: read.clone(),
            bytes_read: bytes.len() as u64,
        });
    }

    Ok((
        completed,
        ByteAccounting {
            requested_bytes: reads.iter().map(|read| read.length).sum(),
            transferred_bytes,
            completed_reads: reads.len(),
        },
    ))
}

pub async fn put_object_store_file(input: impl AsRef<Path>, uri: &str) -> Result<u64> {
    let bytes = std::fs::read(input.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let byte_len = bytes.len() as u64;
    put_object_store_bytes(uri, bytes).await?;
    Ok(byte_len)
}

pub async fn put_object_store_bytes(uri: &str, bytes: Vec<u8>) -> Result<()> {
    let (store, path) = object_store_for_uri(uri)?;
    store
        .put(&path, PutPayload::from(bytes))
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    Ok(())
}

pub async fn read_imu_parquet_row_groups_from_uri(
    uri: &str,
    row_group_ids: &[u32],
) -> Result<Vec<ImuSample>> {
    if row_group_ids.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let (store, path) = object_store_for_uri(uri)?;
    let store: Arc<dyn ObjectStore> = store.into();
    let object_reader = ParquetObjectReader::new(store, path);
    let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let row_groups = normalize_row_group_ids(row_group_ids, builder.metadata().num_row_groups())?;
    let batches = builder
        .with_row_groups(row_groups)
        .build()
        .map_err(|err| RoboticsError::Io(err.to_string()))?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut samples = Vec::new();
    for batch in batches {
        samples.extend(imu_samples_from_batch(&batch)?);
    }
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    samples.sort_by_key(|sample| sample.timestamp_ns);
    Ok(samples)
}

pub async fn read_imu_parquet_row_groups_from_uri_enforced(
    uri: &str,
    row_group_ids: &[u32],
    audit: &RangeAudit,
    footer_allowance_bytes: u64,
) -> Result<(Vec<ImuSample>, RangeAuditReport)> {
    if row_group_ids.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let auditor = RangeAuditor::for_audit_with_footer_allowance(audit, footer_allowance_bytes);
    let (store, path) = object_store_for_uri(uri)?;
    let store: Arc<dyn ObjectStore> = store.into();
    let meta = store
        .head(&path)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    auditor.set_file_size(meta.size);
    let audited_store: Arc<dyn ObjectStore> =
        Arc::new(AuditedObjectStore::new(store, auditor.clone()));
    let object_reader = ParquetObjectReader::new(audited_store, path).with_file_size(meta.size);
    let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let row_groups = normalize_row_group_ids(row_group_ids, builder.metadata().num_row_groups())?;
    auditor.start_materialization();
    let batches = builder
        .with_row_groups(row_groups)
        .build()
        .map_err(|err| RoboticsError::Io(err.to_string()))?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut samples = Vec::new();
    for batch in batches {
        samples.extend(imu_samples_from_batch(&batch)?);
    }
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    samples.sort_by_key(|sample| sample.timestamp_ns);
    Ok((samples, auditor.report()))
}

pub async fn read_pose_parquet_row_groups_from_uri(
    uri: &str,
    row_group_ids: &[u32],
) -> Result<Vec<PoseSample>> {
    if row_group_ids.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let (store, path) = object_store_for_uri(uri)?;
    let store: Arc<dyn ObjectStore> = store.into();
    let object_reader = ParquetObjectReader::new(store, path);
    let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let row_groups = normalize_row_group_ids(row_group_ids, builder.metadata().num_row_groups())?;
    let batches = builder
        .with_row_groups(row_groups)
        .build()
        .map_err(|err| RoboticsError::Io(err.to_string()))?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut samples = Vec::new();
    for batch in batches {
        samples.extend(pose_samples_from_batch(&batch)?);
    }
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    samples.sort_by_key(|sample| sample.timestamp_ns);
    Ok(samples)
}

pub async fn read_pose_parquet_row_groups_from_uri_enforced(
    uri: &str,
    row_group_ids: &[u32],
    audit: &RangeAudit,
    footer_allowance_bytes: u64,
) -> Result<(Vec<PoseSample>, RangeAuditReport)> {
    if row_group_ids.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let auditor = RangeAuditor::for_audit_with_footer_allowance(audit, footer_allowance_bytes);
    let (store, path) = object_store_for_uri(uri)?;
    let store: Arc<dyn ObjectStore> = store.into();
    let meta = store
        .head(&path)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    auditor.set_file_size(meta.size);
    let audited_store: Arc<dyn ObjectStore> =
        Arc::new(AuditedObjectStore::new(store, auditor.clone()));
    let object_reader = ParquetObjectReader::new(audited_store, path).with_file_size(meta.size);
    let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let row_groups = normalize_row_group_ids(row_group_ids, builder.metadata().num_row_groups())?;
    auditor.start_materialization();
    let batches = builder
        .with_row_groups(row_groups)
        .build()
        .map_err(|err| RoboticsError::Io(err.to_string()))?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut samples = Vec::new();
    for batch in batches {
        samples.extend(pose_samples_from_batch(&batch)?);
    }
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    samples.sort_by_key(|sample| sample.timestamp_ns);
    Ok((samples, auditor.report()))
}

pub async fn read_object_store_range_from(
    store: &dyn ObjectStore,
    path: &ObjectPath,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    let range = bounded_range(offset, length)?;
    let bytes = store
        .get_range(path, range)
        .await
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    Ok(bytes.to_vec())
}

#[derive(Debug)]
struct AuditedObjectStore {
    inner: Arc<dyn ObjectStore>,
    auditor: RangeAuditor,
}

impl AuditedObjectStore {
    fn new(inner: Arc<dyn ObjectStore>, auditor: RangeAuditor) -> Self {
        Self { inner, auditor }
    }
}

impl fmt::Display for AuditedObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AuditedObjectStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for AuditedObjectStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if let Some(range) = &options.range {
            match range {
                GetRange::Bounded(range) => self.record_object_range(range.clone())?,
                GetRange::Offset(offset) => {
                    return Err(range_auditor_error(format!(
                        "unbounded offset cold read is not allowed for uri={} requested_offset={offset}",
                        self.auditor.uri
                    )));
                }
                GetRange::Suffix(suffix) => {
                    let file_size = self
                        .inner
                        .head(location)
                        .await
                        .map_err(|err| range_auditor_error(err.to_string()))?
                        .size;
                    self.auditor.set_file_size(file_size);
                    let offset = file_size.saturating_sub(*suffix);
                    self.record_object_range(offset..file_size)?;
                }
            }
        } else if !options.head {
            return Err(range_auditor_error(format!(
                "full-object cold read is not allowed for uri={}",
                self.auditor.uri
            )));
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &ObjectPath,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        for range in ranges {
            self.record_object_range(range.clone())?;
        }
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&ObjectPath>,
        offset: &ObjectPath,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

impl AuditedObjectStore {
    fn record_object_range(&self, range: Range<u64>) -> object_store::Result<()> {
        if range.end <= range.start {
            return Err(range_auditor_error(format!(
                "invalid cold read range for uri={} requested=[{}, {})",
                self.auditor.uri, range.start, range.end
            )));
        }
        self.auditor
            .record_bounded_read(range.start, range.end - range.start)
            .map_err(|err| range_auditor_error(err.to_string()))
    }
}

pub fn object_store_for_uri(uri: &str) -> Result<(Box<dyn ObjectStore>, ObjectPath)> {
    if has_uri_scheme(uri) {
        if let Some((scheme, rest)) = uri
            .split_once("://")
            .filter(|(scheme, _)| matches!(*scheme, "s3" | "s3a"))
        {
            let (bucket, object_path) = rest.split_once('/').ok_or_else(|| {
                RoboticsError::InvalidArgument(format!("missing object key in S3 URI {uri}"))
            })?;
            if bucket.is_empty() || object_path.is_empty() {
                return Err(RoboticsError::InvalidArgument(format!(
                    "S3 URI must be {scheme}://bucket/key: {uri}"
                )));
            }
            let store = AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()
                .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
            return Ok((
                Box::new(store),
                ObjectPath::parse(object_path).map_err(path_error)?,
            ));
        }

        let url = Url::parse(uri).map_err(|err| {
            RoboticsError::InvalidArgument(format!("invalid object-store URI {uri}: {err}"))
        })?;
        return parse_url_opts(&url, std::env::vars())
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string()));
    }

    let path = Path::new(uri);
    let (prefix, object_path) = split_local_path(path)?;
    let store = LocalFileSystem::new_with_prefix(prefix)
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    Ok((
        Box::new(store),
        ObjectPath::parse(object_path).map_err(path_error)?,
    ))
}

fn split_local_path(path: &Path) -> Result<(PathBuf, String)> {
    if path.is_absolute() {
        let root = PathBuf::from("/");
        let object_path = path
            .strip_prefix(&root)
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?
            .to_string_lossy()
            .to_string();
        Ok((root, object_path))
    } else {
        let cwd = std::env::current_dir().map_err(|err| RoboticsError::Io(err.to_string()))?;
        Ok((
            cwd,
            path.to_string_lossy().trim_start_matches("./").to_string(),
        ))
    }
}

fn bounded_range(offset: u64, length: u64) -> Result<Range<u64>> {
    if length == 0 {
        return Err(RoboticsError::InvalidArgument(
            "range length must be positive".to_string(),
        ));
    }
    let end = offset.checked_add(length).ok_or_else(|| {
        RoboticsError::InvalidArgument("range offset + length overflowed u64".to_string())
    })?;
    Ok(offset..end)
}

fn format_authorized_ranges(ranges: &[RangeRead]) -> String {
    if ranges.is_empty() {
        return "[]".to_string();
    }
    let parts = ranges
        .iter()
        .map(|range| {
            format!(
                "{}:[{}, {})",
                range.uri,
                range.offset,
                range.offset.saturating_add(range.length)
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(", "))
}

fn range_auditor_error(message: String) -> object_store::Error {
    object_store::Error::Generic {
        store: "RangeAuditor",
        source: Box::new(RoboticsError::InvalidArgument(message)),
    }
}

fn has_uri_scheme(uri: &str) -> bool {
    uri.split_once(':').is_some_and(|(scheme, _)| {
        let mut chars = scheme.chars();
        chars.next().is_some_and(|c| c.is_ascii_alphabetic())
            && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    })
}

fn path_error(err: object_store::path::Error) -> RoboticsError {
    RoboticsError::InvalidArgument(err.to_string())
}

fn pose_samples_from_batch(batch: &RecordBatch) -> Result<Vec<PoseSample>> {
    let timestamp_ns = int64_batch_column(batch, "timestamp_ns")?;
    let robot_id = string_batch_column(batch, "robot_id")?;
    let session_id = string_batch_column(batch, "session_id")?;
    let x = float64_batch_column(batch, "x")?;
    let y = float64_batch_column(batch, "y")?;
    let z = float64_batch_column(batch, "z")?;
    let qw = float64_batch_column(batch, "qw")?;
    let qx = float64_batch_column(batch, "qx")?;
    let qy = float64_batch_column(batch, "qy")?;
    let qz = float64_batch_column(batch, "qz")?;
    let vx = float64_batch_column(batch, "vx")?;
    let vy = float64_batch_column(batch, "vy")?;
    let vz = float64_batch_column(batch, "vz")?;
    let mut samples = Vec::with_capacity(batch.num_rows());

    for row in 0..batch.num_rows() {
        samples.push(PoseSample {
            timestamp_ns: timestamp_ns.value(row),
            robot_id: robot_id.value(row).to_string(),
            session_id: session_id.value(row).to_string(),
            x: x.value(row),
            y: y.value(row),
            z: z.value(row),
            qw: qw.value(row),
            qx: qx.value(row),
            qy: qy.value(row),
            qz: qz.value(row),
            vx: vx.value(row),
            vy: vy.value(row),
            vz: vz.value(row),
        });
    }

    Ok(samples)
}

fn imu_samples_from_batch(batch: &RecordBatch) -> Result<Vec<ImuSample>> {
    let timestamp_ns = int64_batch_column(batch, "timestamp_ns")?;
    let robot_id = string_batch_column(batch, "robot_id")?;
    let session_id = string_batch_column(batch, "session_id")?;
    let ax = float64_batch_column(batch, "ax")?;
    let ay = float64_batch_column(batch, "ay")?;
    let az = float64_batch_column(batch, "az")?;
    let gx = float64_batch_column(batch, "gx")?;
    let gy = float64_batch_column(batch, "gy")?;
    let gz = float64_batch_column(batch, "gz")?;
    let mut samples = Vec::with_capacity(batch.num_rows());

    for row in 0..batch.num_rows() {
        samples.push(ImuSample {
            timestamp_ns: timestamp_ns.value(row),
            robot_id: robot_id.value(row).to_string(),
            session_id: session_id.value(row).to_string(),
            ax: ax.value(row),
            ay: ay.value(row),
            az: az.value(row),
            gx: gx.value(row),
            gy: gy.value(row),
            gz: gz.value(row),
        });
    }

    Ok(samples)
}

fn normalize_row_group_ids(row_group_ids: &[u32], row_group_count: usize) -> Result<Vec<usize>> {
    let mut row_groups = row_group_ids
        .iter()
        .map(|row_group_id| {
            usize::try_from(*row_group_id).map_err(|_| {
                RoboticsError::InvalidArgument(format!("row group id {row_group_id} is too large"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    row_groups.sort_unstable();
    row_groups.dedup();
    if let Some(row_group_id) = row_groups
        .iter()
        .copied()
        .find(|row_group_id| *row_group_id >= row_group_count)
    {
        return Err(RoboticsError::InvalidArgument(format!(
            "row group id {row_group_id} is outside 0..{row_group_count}"
        )));
    }
    Ok(row_groups)
}

fn int64_batch_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    typed_batch_column(batch, name)
}

fn float64_batch_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    typed_batch_column(batch, name)
}

fn string_batch_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    typed_batch_column(batch, name)
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

#[cfg(test)]
mod tests {
    use robotics_catalog::{
        generate_fake_catalog, index_parquet_file, query_catalog, FakeCatalogConfig,
    };
    use robotics_core::ImuSample;
    use robotics_core::QuerySpec;

    use super::*;

    #[test]
    fn range_plan_uses_catalog_byte_spans() {
        let catalog = generate_fake_catalog(FakeCatalogConfig {
            sessions: 2,
            ..Default::default()
        });
        let spec = QuerySpec {
            robot_id: None,
            start_ts_ns: catalog[0].start_ts_ns,
            end_ts_ns: catalog[1].end_ts_ns,
            bbox: None,
            min_velocity: None,
            target_hz: 30.0,
        };
        let windows = query_catalog(&catalog, &spec);
        let reads = plan_range_reads(&windows);
        let accounting = account_reads(&reads);

        assert_eq!(reads.len(), 2);
        assert_eq!(accounting.requested_bytes, 2 * 65_536);
        assert_eq!(accounting.transferred_bytes, 0);
    }

    #[test]
    fn range_audit_accepts_exact_row_group_byte_spans() {
        let ranges = vec![
            RowGroupRange {
                row_group_id: 2,
                offset: 20,
                length: 5,
            },
            RowGroupRange {
                row_group_id: 1,
                offset: 10,
                length: 7,
            },
        ];

        let audit = audit_row_group_range_reads("file:///tmp/session.parquet", &[1, 2], &ranges)
            .expect("audit should pass");

        assert_eq!(audit.planned_range_reads(), 2);
        assert_eq!(audit.planned_read_bytes, 12);
        assert_eq!(audit.reads[0].row_group_id, 1);
        assert_eq!(audit.reads[0].read.offset, 10);
        assert_eq!(audit.reads[1].row_group_id, 2);
    }

    #[test]
    fn range_audit_rejects_mismatched_row_groups() {
        let ranges = vec![RowGroupRange {
            row_group_id: 0,
            offset: 10,
            length: 7,
        }];

        let err = audit_row_group_range_reads("file:///tmp/session.parquet", &[1], &ranges)
            .expect_err("audit should reject mismatched row groups");

        assert!(err
            .to_string()
            .contains("do not match requested row groups"));
    }

    #[test]
    fn local_range_reader_reads_only_requested_span() {
        let path =
            std::env::temp_dir().join(format!("robotics_query_{}_range.bin", std::process::id()));
        std::fs::write(&path, b"abcdefghijklmnopqrstuvwxyz").unwrap();

        let bytes = read_local_range(&path, 4, 6).unwrap();

        assert_eq!(bytes, b"efghij");
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn object_store_range_reader_reads_only_requested_span() {
        let path = std::env::temp_dir().join(format!(
            "robotics_query_{}_object_range.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"abcdefghijklmnopqrstuvwxyz").unwrap();
        let read = RangeRead {
            uri: path.to_string_lossy().to_string(),
            offset: 10,
            length: 5,
        };

        let bytes = read_object_store_range(&read).await.unwrap();
        let (completed, accounting) = execute_object_store_range_reads(&[read]).await.unwrap();

        assert_eq!(bytes, b"klmno");
        assert_eq!(completed[0].bytes_read, 5);
        assert_eq!(accounting.requested_bytes, 5);
        assert_eq!(accounting.transferred_bytes, 5);
        assert_eq!(accounting.completed_reads, 1);
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn object_store_put_writes_local_uri() {
        let path = std::env::temp_dir().join(format!(
            "robotics_query_{}_object_put.bin",
            std::process::id()
        ));
        std::fs::remove_file(&path).ok();

        let uploaded =
            put_object_store_file(&path.with_extension("missing"), &path.to_string_lossy())
                .await
                .expect_err("missing upload source should error");
        assert!(
            uploaded.to_string().contains("No such file")
                || uploaded.to_string().contains("os error")
        );

        let source = path.with_extension("src");
        std::fs::write(&source, b"cold object bytes").unwrap();
        let bytes = put_object_store_file(&source, &path.to_string_lossy())
            .await
            .unwrap();

        assert_eq!(bytes, 17);
        assert_eq!(std::fs::read(&path).unwrap(), b"cold object bytes");
        std::fs::remove_file(path).ok();
        std::fs::remove_file(source).ok();
    }

    #[tokio::test]
    async fn object_store_imu_reader_reads_only_requested_row_groups() {
        let path =
            std::env::temp_dir().join(format!("robotics_query_{}_imu.parquet", std::process::id()));
        let samples = vec![
            imu_at(0, 0.0),
            imu_at(1, 1.0),
            imu_at(2, 10.0),
            imu_at(3, 11.0),
        ];
        robotics_ingest::write_imu_parquet(&path, &samples, 2).unwrap();

        let selected = read_imu_parquet_row_groups_from_uri(&path.to_string_lossy(), &[1])
            .await
            .unwrap();

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].timestamp_ns, 2);
        assert_eq!(selected[0].ax, 10.0);
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn object_store_pose_reader_reads_only_requested_row_groups() {
        let path = std::env::temp_dir().join(format!(
            "robotics_query_{}_pose.parquet",
            std::process::id()
        ));
        let samples = vec![
            pose_at(0, 0.0),
            pose_at(1, 1.0),
            pose_at(2, 10.0),
            pose_at(3, 11.0),
        ];
        robotics_ingest::write_pose_parquet(&path, &samples, 2).unwrap();

        let uri = format!("file://{}", path.display());
        let selected = read_pose_parquet_row_groups_from_uri(&uri, &[1])
            .await
            .unwrap();

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].timestamp_ns, 2);
        assert_eq!(selected[0].x, 10.0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn range_auditor_records_authorized_and_footer_reads() {
        let auditor = RangeAuditor::new(
            "file:///tmp/session.parquet",
            vec![RangeRead {
                uri: "file:///tmp/session.parquet".to_string(),
                offset: 100,
                length: 50,
            }],
            25,
        );
        auditor.set_file_size(200);

        auditor.record_bounded_read(110, 10).unwrap();
        auditor.record_bounded_read(180, 8).unwrap();
        let report = auditor.report();

        assert_eq!(report.actual_read_count(), 2);
        assert_eq!(report.footer_allowance_bytes, 25);
        assert_eq!(report.actual_authorized_bytes, 10);
        assert_eq!(report.materialized_bytes, 10);
        assert_eq!(report.footer_bytes, 8);
        assert_eq!(report.largest_metadata_read, 8);
        assert_eq!(report.max_footer_read_offset, 180);
        assert_eq!(report.max_footer_read_end, 188);
        assert_eq!(report.actual_reads[0].category, "authorized");
        assert_eq!(report.actual_reads[1].category, "footer");
    }

    #[test]
    fn range_auditor_rejects_footer_read_when_allowance_is_too_small() {
        let auditor = RangeAuditor::new("file:///tmp/session.parquet", Vec::new(), 4);
        auditor.set_file_size(200);

        let err = auditor
            .record_bounded_read(190, 8)
            .expect_err("footer read outside allowance should fail");

        assert!(err.to_string().contains("footer_allowance_bytes=4"));
        assert_eq!(auditor.report().violations.len(), 1);
    }

    #[test]
    fn range_auditor_rejects_unauthorized_reads_with_debuggable_error() {
        let auditor = RangeAuditor::new(
            "file:///tmp/session.parquet",
            vec![RangeRead {
                uri: "file:///tmp/session.parquet".to_string(),
                offset: 100,
                length: 50,
            }],
            25,
        );
        auditor.set_file_size(200);

        let err = auditor
            .record_bounded_read(40, 10)
            .expect_err("unauthorized read should fail");
        let message = err.to_string();

        assert!(message.contains("file:///tmp/session.parquet"));
        assert!(message.contains("requested=[40, 50)"));
        assert!(message.contains("authorized_ranges"));
        assert_eq!(auditor.report().violations.len(), 1);
    }

    #[tokio::test]
    async fn enforced_pose_reader_allows_selected_row_group_and_footer_only() {
        let path = std::env::temp_dir().join(format!(
            "robotics_query_{}_pose_enforced.parquet",
            std::process::id()
        ));
        let samples = vec![
            pose_at(0, 0.0),
            pose_at(1, 1.0),
            pose_at(2, 10.0),
            pose_at(3, 11.0),
        ];
        robotics_ingest::write_pose_parquet(&path, &samples, 2).unwrap();
        let entries = index_parquet_file(&path).unwrap();
        let row_group = &entries[1];
        let audit = audit_row_group_range_reads(
            &path.to_string_lossy(),
            &[1],
            &[RowGroupRange {
                row_group_id: 1,
                offset: row_group.byte_offset,
                length: row_group.byte_length,
            }],
        )
        .unwrap();

        let (selected, report) = read_pose_parquet_row_groups_from_uri_enforced(
            &path.to_string_lossy(),
            &[1],
            &audit,
            DEFAULT_FOOTER_ALLOWANCE_BYTES,
        )
        .await
        .unwrap();

        assert_eq!(selected.len(), 2);
        assert!(report.materialized_bytes > 0);
        assert!(report.footer_bytes > 0);
        assert!(report.violations.is_empty());
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn enforced_pose_reader_rejects_tampered_allowlist() {
        let path = std::env::temp_dir().join(format!(
            "robotics_query_{}_pose_tampered.parquet",
            std::process::id()
        ));
        let samples = vec![
            pose_at(0, 0.0),
            pose_at(1, 1.0),
            pose_at(2, 10.0),
            pose_at(3, 11.0),
        ];
        robotics_ingest::write_pose_parquet(&path, &samples, 2).unwrap();
        let entries = index_parquet_file(&path).unwrap();
        let row_group = &entries[1];
        let audit = audit_row_group_range_reads(
            &path.to_string_lossy(),
            &[1],
            &[RowGroupRange {
                row_group_id: 1,
                offset: row_group.byte_offset.saturating_add(1),
                length: row_group.byte_length.saturating_sub(1).max(1),
            }],
        )
        .unwrap();

        let err = read_pose_parquet_row_groups_from_uri_enforced(
            &path.to_string_lossy(),
            &[1],
            &audit,
            DEFAULT_FOOTER_ALLOWANCE_BYTES,
        )
        .await
        .expect_err("tampered allowlist should fail");

        assert!(err.to_string().contains("unauthorized cold read"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn seek_manifest_json_round_trips() {
        let audit = audit_row_group_range_reads(
            "file:///tmp/session.parquet",
            &[0],
            &[RowGroupRange {
                row_group_id: 0,
                offset: 10,
                length: 20,
            }],
        )
        .unwrap();
        let auditor = RangeAuditor::for_audit(&audit);
        auditor.set_file_size(100);
        auditor.record_bounded_read(10, 20).unwrap();
        auditor.record_bounded_read(96, 4).unwrap();
        let report = auditor.report();
        let manifest = SeekManifest::new(
            "file:///tmp/session.parquet",
            "pose",
            vec![0],
            Some(&audit),
            Some(&report),
        );

        let json = manifest.to_json_pretty().unwrap();
        let round_trip = SeekManifest::from_json(&json).unwrap();

        assert_eq!(round_trip, manifest);
        assert_eq!(
            round_trip.footer_allowance_bytes,
            DEFAULT_FOOTER_ALLOWANCE_BYTES
        );
        assert_eq!(round_trip.footer_bytes, 4);
        assert_eq!(round_trip.largest_metadata_read, 4);
        assert!(round_trip.enforcement_enabled);
    }

    #[test]
    fn s3_uri_parser_splits_bucket_and_object_key() {
        let previous_access_key = std::env::var("AWS_ACCESS_KEY_ID").ok();
        let previous_secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
        let previous_region = std::env::var("AWS_REGION").ok();
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ACCESS_KEY_ID", "test");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test");
        std::env::set_var("AWS_ENDPOINT", "http://127.0.0.1:9000");
        std::env::set_var("AWS_ALLOW_HTTP", "true");
        std::env::set_var("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "false");

        let (_store, path) = object_store_for_uri("s3://robotics/session.parquet").unwrap();

        assert_eq!(path.as_ref(), "session.parquet");
        restore_env("AWS_ACCESS_KEY_ID", previous_access_key);
        restore_env("AWS_SECRET_ACCESS_KEY", previous_secret_key);
        restore_env("AWS_REGION", previous_region);
        std::env::remove_var("AWS_ENDPOINT");
        std::env::remove_var("AWS_ALLOW_HTTP");
        std::env::remove_var("AWS_VIRTUAL_HOSTED_STYLE_REQUEST");
    }

    fn restore_env(name: &str, value: Option<String>) {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }

    fn imu_at(timestamp_ns: i64, value: f64) -> ImuSample {
        ImuSample {
            timestamp_ns,
            robot_id: "robot".to_string(),
            session_id: "session".to_string(),
            ax: value,
            ay: value + 1.0,
            az: value + 2.0,
            gx: value + 3.0,
            gy: value + 4.0,
            gz: value + 5.0,
        }
    }

    fn pose_at(timestamp_ns: i64, value: f64) -> PoseSample {
        PoseSample {
            timestamp_ns,
            robot_id: "robot".to_string(),
            session_id: "session".to_string(),
            x: value,
            y: value + 1.0,
            z: value + 2.0,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            vx: 1.0,
            vy: 0.0,
            vz: 0.0,
        }
    }
}
