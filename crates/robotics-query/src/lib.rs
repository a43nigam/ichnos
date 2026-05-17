use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};

use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{parse_url_opts, ObjectStore, ObjectStoreExt};
use robotics_core::{Result, RoboticsError, WindowRef};
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeRead {
    pub uri: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteAccounting {
    pub requested_bytes: u64,
    pub transferred_bytes: u64,
    pub completed_reads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedRangeRead {
    pub read: RangeRead,
    pub bytes_read: u64,
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

#[cfg(test)]
mod tests {
    use robotics_catalog::{generate_fake_catalog, query_catalog, FakeCatalogConfig};
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
}
