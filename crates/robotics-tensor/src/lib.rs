use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use robotics_core::{ImuSample, PoseSample, Result, RoboticsError, TensorBatch, TimestampNs};

const CHANNELS: usize = 10;
const IMU_CHANNELS: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorNpyFiles {
    pub values_path: PathBuf,
    pub timestamps_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn normalized(self) -> Result<Self> {
        let norm = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm == 0.0 {
            return Err(RoboticsError::InvalidArgument(
                "zero-length quaternion".to_string(),
            ));
        }
        Ok(Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        })
    }

    pub fn dot(self, other: Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn slerp(self, mut other: Self, t: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&t) {
            return Err(RoboticsError::InvalidArgument(
                "slerp t must be in [0, 1]".to_string(),
            ));
        }

        let q1 = self.normalized()?;
        other = other.normalized()?;
        let mut dot = q1.dot(other);

        if dot < 0.0 {
            other = Self {
                w: -other.w,
                x: -other.x,
                y: -other.y,
                z: -other.z,
            };
            dot = -dot;
        }

        if dot > 0.9995 {
            return Self {
                w: q1.w + t * (other.w - q1.w),
                x: q1.x + t * (other.x - q1.x),
                y: q1.y + t * (other.y - q1.y),
                z: q1.z + t * (other.z - q1.z),
            }
            .normalized();
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();
        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Self {
            w: s0 * q1.w + s1 * other.w,
            x: s0 * q1.x + s1 * other.x,
            y: s0 * q1.y + s1 * other.y,
            z: s0 * q1.z + s1 * other.z,
        }
        .normalized()
    }
}

pub fn tensorize(
    samples: &[PoseSample],
    start_ns: TimestampNs,
    end_ns: TimestampNs,
    hz: f64,
) -> Result<TensorBatch> {
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    if hz <= 0.0 {
        return Err(RoboticsError::InvalidArgument(
            "hz must be positive".to_string(),
        ));
    }
    if start_ns > end_ns {
        return Err(RoboticsError::InvalidArgument(
            "start must be <= end".to_string(),
        ));
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by_key(|sample| sample.timestamp_ns);
    let min_ns = sorted.first().expect("checked non-empty").timestamp_ns;
    let max_ns = sorted.last().expect("checked non-empty").timestamp_ns;
    if start_ns < min_ns {
        return Err(RoboticsError::Extrapolation {
            requested_ns: start_ns,
            min_ns,
            max_ns,
        });
    }
    if end_ns > max_ns {
        return Err(RoboticsError::Extrapolation {
            requested_ns: end_ns,
            min_ns,
            max_ns,
        });
    }

    let step_ns = (1_000_000_000.0 / hz).round() as i64;
    if step_ns <= 0 {
        return Err(RoboticsError::InvalidArgument(
            "hz produces a sub-nanosecond step".to_string(),
        ));
    }

    let mut timestamps_ns = Vec::new();
    let mut values = Vec::new();
    let mut ts = start_ns;
    while ts <= end_ns {
        let row = interpolate(&sorted, ts)?;
        timestamps_ns.push(ts);
        values.extend_from_slice(&row);
        ts = match ts.checked_add(step_ns) {
            Some(next) => next,
            None => break,
        };
    }

    Ok(TensorBatch {
        rows: timestamps_ns.len(),
        channels: CHANNELS,
        timestamps_ns,
        values,
    })
}

pub fn tensorize_imu(samples: &[ImuSample], timestamps_ns: &[TimestampNs]) -> Result<TensorBatch> {
    if samples.is_empty() || timestamps_ns.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by_key(|sample| sample.timestamp_ns);
    let min_ns = sorted.first().expect("checked non-empty").timestamp_ns;
    let max_ns = sorted.last().expect("checked non-empty").timestamp_ns;
    let first_requested = *timestamps_ns.first().expect("checked non-empty");
    let last_requested = *timestamps_ns.last().expect("checked non-empty");
    if first_requested < min_ns {
        return Err(RoboticsError::Extrapolation {
            requested_ns: first_requested,
            min_ns,
            max_ns,
        });
    }
    if last_requested > max_ns {
        return Err(RoboticsError::Extrapolation {
            requested_ns: last_requested,
            min_ns,
            max_ns,
        });
    }

    let mut values = Vec::with_capacity(timestamps_ns.len() * IMU_CHANNELS);
    for &ts in timestamps_ns {
        let row = interpolate_imu(&sorted, ts)?;
        values.extend_from_slice(&row);
    }

    Ok(TensorBatch {
        rows: timestamps_ns.len(),
        channels: IMU_CHANNELS,
        timestamps_ns: timestamps_ns.to_vec(),
        values,
    })
}

pub fn gap_stats(timestamps_ns: &[TimestampNs]) -> (usize, TimestampNs) {
    if timestamps_ns.len() < 2 {
        return (0, 0);
    }
    let mut diffs = timestamps_ns
        .windows(2)
        .filter_map(|pair| pair[1].checked_sub(pair[0]))
        .filter(|diff| *diff > 0)
        .collect::<Vec<_>>();
    if diffs.is_empty() {
        return (0, 0);
    }
    diffs.sort_unstable();
    let median = if diffs.len() % 2 == 0 {
        let upper = diffs.len() / 2;
        ((diffs[upper - 1] as f64 + diffs[upper] as f64) / 2.0).round() as i64
    } else {
        diffs[diffs.len() / 2]
    };
    let threshold = (median.saturating_mul(5)).max(1);
    let max_gap = *diffs.last().expect("checked non-empty");
    let gap_count = diffs.iter().filter(|diff| **diff > threshold).count();
    (gap_count, max_gap)
}

pub fn write_tensor_npy(prefix: impl AsRef<Path>, batch: &TensorBatch) -> Result<TensorNpyFiles> {
    if batch.values.len() != batch.rows * batch.channels {
        return Err(RoboticsError::InvalidArgument(format!(
            "tensor values length {} does not match shape [{}, {}]",
            batch.values.len(),
            batch.rows,
            batch.channels
        )));
    }
    if batch.timestamps_ns.len() != batch.rows {
        return Err(RoboticsError::InvalidArgument(format!(
            "timestamp length {} does not match tensor rows {}",
            batch.timestamps_ns.len(),
            batch.rows
        )));
    }

    let prefix = prefix.as_ref();
    let values_path = suffixed_path(prefix, "values.npy");
    let timestamps_path = suffixed_path(prefix, "timestamps_ns.npy");
    write_f64_npy(&values_path, &[batch.rows, batch.channels], &batch.values)?;
    write_i64_npy(&timestamps_path, &[batch.rows], &batch.timestamps_ns)?;
    Ok(TensorNpyFiles {
        values_path,
        timestamps_path,
    })
}

pub fn read_i64_npy(path: impl AsRef<Path>) -> Result<Vec<i64>> {
    let mut file =
        std::fs::File::open(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(RoboticsError::InvalidArgument(
            "input is not an npy file".to_string(),
        ));
    }
    let version = bytes[6];
    let header_len_offset = 8;
    let (header_len, data_offset): (usize, usize) = match version {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, header_len_offset + 2)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(RoboticsError::InvalidArgument(
                    "truncated npy header".to_string(),
                ));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, header_len_offset + 4)
        }
        _ => {
            return Err(RoboticsError::InvalidArgument(format!(
                "unsupported npy version {version}"
            )))
        }
    };
    let data_start = data_offset.checked_add(header_len).ok_or_else(|| {
        RoboticsError::InvalidArgument("npy header length overflowed".to_string())
    })?;
    if bytes.len() < data_start {
        return Err(RoboticsError::InvalidArgument(
            "truncated npy header".to_string(),
        ));
    }
    let header = std::str::from_utf8(&bytes[data_offset..data_start])
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
    if !(header.contains("'descr': '<i8'") || header.contains("\"descr\": \"<i8\"")) {
        return Err(RoboticsError::InvalidArgument(
            "expected little-endian int64 npy data".to_string(),
        ));
    }
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        return Err(RoboticsError::InvalidArgument(
            "fortran-order npy arrays are not supported".to_string(),
        ));
    }
    let data = &bytes[data_start..];
    if data.len() % 8 != 0 {
        return Err(RoboticsError::InvalidArgument(
            "int64 npy payload length is not divisible by 8".to_string(),
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn interpolate(samples: &[PoseSample], ts: TimestampNs) -> Result<[f64; CHANNELS]> {
    match samples.binary_search_by_key(&ts, |sample| sample.timestamp_ns) {
        Ok(index) => Ok(samples[index].channels()),
        Err(index) if index == 0 || index >= samples.len() => {
            let min_ns = samples.first().expect("checked by caller").timestamp_ns;
            let max_ns = samples.last().expect("checked by caller").timestamp_ns;
            Err(RoboticsError::Extrapolation {
                requested_ns: ts,
                min_ns,
                max_ns,
            })
        }
        Err(index) => {
            let before = &samples[index - 1];
            let after = &samples[index];
            let span = (after.timestamp_ns - before.timestamp_ns) as f64;
            let t = (ts - before.timestamp_ns) as f64 / span;
            let lerp = |a: f64, b: f64| a + (b - a) * t;
            let q = Quaternion {
                w: before.qw,
                x: before.qx,
                y: before.qy,
                z: before.qz,
            }
            .slerp(
                Quaternion {
                    w: after.qw,
                    x: after.qx,
                    y: after.qy,
                    z: after.qz,
                },
                t,
            )?;

            Ok([
                lerp(before.x, after.x),
                lerp(before.y, after.y),
                lerp(before.z, after.z),
                q.w,
                q.x,
                q.y,
                q.z,
                lerp(before.vx, after.vx),
                lerp(before.vy, after.vy),
                lerp(before.vz, after.vz),
            ])
        }
    }
}

fn interpolate_imu(samples: &[ImuSample], ts: TimestampNs) -> Result<[f64; IMU_CHANNELS]> {
    match samples.binary_search_by_key(&ts, |sample| sample.timestamp_ns) {
        Ok(index) => Ok(samples[index].channels()),
        Err(index) if index == 0 || index >= samples.len() => {
            let min_ns = samples.first().expect("checked by caller").timestamp_ns;
            let max_ns = samples.last().expect("checked by caller").timestamp_ns;
            Err(RoboticsError::Extrapolation {
                requested_ns: ts,
                min_ns,
                max_ns,
            })
        }
        Err(index) => {
            let before = &samples[index - 1];
            let after = &samples[index];
            let span = after.timestamp_ns - before.timestamp_ns;
            if span <= 0 {
                return Err(RoboticsError::InvalidArgument(
                    "IMU timestamps must be strictly increasing for interpolation".to_string(),
                ));
            }
            let t = (ts - before.timestamp_ns) as f64 / span as f64;
            let lerp = |a: f64, b: f64| a + (b - a) * t;
            Ok([
                lerp(before.ax, after.ax),
                lerp(before.ay, after.ay),
                lerp(before.az, after.az),
                lerp(before.gx, after.gx),
                lerp(before.gy, after.gy),
                lerp(before.gz, after.gz),
            ])
        }
    }
}

fn write_f64_npy(path: &Path, shape: &[usize], values: &[f64]) -> Result<()> {
    let mut file = create_output_file(path)?;
    write_npy_header(&mut file, "<f8", shape)?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|err| RoboticsError::Io(err.to_string()))?;
    }
    Ok(())
}

fn write_i64_npy(path: &Path, shape: &[usize], values: &[i64]) -> Result<()> {
    let mut file = create_output_file(path)?;
    write_npy_header(&mut file, "<i8", shape)?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|err| RoboticsError::Io(err.to_string()))?;
    }
    Ok(())
}

fn create_output_file(path: &Path) -> Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| RoboticsError::Io(err.to_string()))?;
    }
    std::fs::File::create(path).map_err(|err| RoboticsError::Io(err.to_string()))
}

fn write_npy_header(writer: &mut impl Write, descr: &str, shape: &[usize]) -> Result<()> {
    if shape.is_empty() {
        return Err(RoboticsError::InvalidArgument(
            "npy shape must have at least one dimension".to_string(),
        ));
    }
    let shape = format_npy_shape(shape);
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_len = u16::try_from(header.len()).map_err(|_| {
        RoboticsError::InvalidArgument("npy header exceeds v1.0 length limit".to_string())
    })?;

    writer
        .write_all(b"\x93NUMPY")
        .and_then(|()| writer.write_all(&[1, 0]))
        .and_then(|()| writer.write_all(&header_len.to_le_bytes()))
        .and_then(|()| writer.write_all(header.as_bytes()))
        .map_err(|err| RoboticsError::Io(err.to_string()))
}

fn format_npy_shape(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn suffixed_path(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = prefix.to_path_buf();
    let file_name = prefix
        .file_name()
        .map(|name| format!("{}.{}", name.to_string_lossy(), suffix))
        .unwrap_or_else(|| suffix.to_string());
    path.set_file_name(file_name);
    path
}

#[cfg(test)]
mod tests {
    use robotics_core::RoboticsError;

    use super::*;

    #[test]
    fn slerp_halfway_identity_to_180_deg_x() {
        let q1 = Quaternion {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let q2 = Quaternion {
            w: 0.0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };

        let result = q1.slerp(q2, 0.5).unwrap();

        assert!((result.w - 2.0_f64.sqrt() / 2.0).abs() < 1e-6);
        assert!((result.x - 2.0_f64.sqrt() / 2.0).abs() < 1e-6);
        assert!(result.y.abs() < 1e-6);
        assert!(result.z.abs() < 1e-6);
    }

    #[test]
    fn slerp_takes_short_path_for_negative_dot() {
        let q1 = Quaternion {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let q2 = Quaternion {
            w: -1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };

        let result = q1.slerp(q2, 0.5).unwrap();

        assert!((result.w - 1.0).abs() < 1e-6);
        assert!(result.x.abs() < 1e-6);
    }

    #[test]
    fn resamples_linear_motion_to_uniform_shape() {
        let samples = vec![sample_at(0, 0.0), sample_at(1_000_000_000, 10.0)];

        let batch = tensorize(&samples, 0, 1_000_000_000, 2.0).unwrap();

        assert_eq!(batch.rows, 3);
        assert_eq!(batch.channels, 10);
        assert_eq!(batch.timestamps_ns, vec![0, 500_000_000, 1_000_000_000]);
        assert_eq!(batch.row(1).unwrap()[0], 5.0);
    }

    #[test]
    fn rejects_extrapolation() {
        let samples = vec![sample_at(0, 0.0), sample_at(1_000_000_000, 1.0)];

        let err = tensorize(&samples, 0, 2_000_000_000, 30.0).unwrap_err();

        assert!(matches!(
            err,
            RoboticsError::Extrapolation {
                requested_ns: 2_000_000_000,
                ..
            }
        ));
    }

    #[test]
    fn writes_numpy_tensor_files() {
        let path =
            std::env::temp_dir().join(format!("robotics_tensor_{}_interop", std::process::id()));
        let batch = TensorBatch {
            timestamps_ns: vec![0, 500_000_000],
            values: vec![1.0, 2.0, 3.0, 4.0],
            rows: 2,
            channels: 2,
        };

        let files = write_tensor_npy(&path, &batch).unwrap();
        let values = std::fs::read(&files.values_path).unwrap();
        let timestamps = std::fs::read(&files.timestamps_path).unwrap();

        assert!(values.starts_with(b"\x93NUMPY"));
        assert!(timestamps.starts_with(b"\x93NUMPY"));
        assert!(String::from_utf8_lossy(&values).contains("'shape': (2, 2)"));
        assert!(String::from_utf8_lossy(&timestamps).contains("'shape': (2,)"));

        std::fs::remove_file(files.values_path).ok();
        std::fs::remove_file(files.timestamps_path).ok();
    }

    #[test]
    fn resamples_imu_to_existing_timestamps() {
        let samples = vec![imu_at(0, 0.0), imu_at(1_000_000_000, 10.0)];
        let batch = tensorize_imu(&samples, &[0, 500_000_000, 1_000_000_000]).unwrap();

        assert_eq!(batch.rows, 3);
        assert_eq!(batch.channels, 6);
        assert_eq!(batch.row(1).unwrap(), &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn imu_resampling_rejects_extrapolation() {
        let samples = vec![imu_at(0, 0.0), imu_at(1_000_000_000, 10.0)];
        let err = tensorize_imu(&samples, &[0, 2_000_000_000]).unwrap_err();

        assert!(matches!(
            err,
            RoboticsError::Extrapolation {
                requested_ns: 2_000_000_000,
                ..
            }
        ));
    }

    #[test]
    fn reads_generated_i64_npy_file() {
        let path =
            std::env::temp_dir().join(format!("robotics_tensor_{}_timestamps", std::process::id()));
        let batch = TensorBatch {
            timestamps_ns: vec![10, 20, 30],
            values: vec![1.0, 2.0, 3.0],
            rows: 3,
            channels: 1,
        };
        let files = write_tensor_npy(&path, &batch).unwrap();

        let timestamps = read_i64_npy(&files.timestamps_path).unwrap();

        assert_eq!(timestamps, vec![10, 20, 30]);
        std::fs::remove_file(files.values_path).ok();
        std::fs::remove_file(files.timestamps_path).ok();
    }

    fn sample_at(timestamp_ns: i64, x: f64) -> PoseSample {
        PoseSample {
            timestamp_ns,
            robot_id: "test".to_string(),
            session_id: "session".to_string(),
            x,
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

    fn imu_at(timestamp_ns: i64, value: f64) -> ImuSample {
        ImuSample {
            timestamp_ns,
            robot_id: "test".to_string(),
            session_id: "session".to_string(),
            ax: value,
            ay: value + 1.0,
            az: value + 2.0,
            gx: value + 3.0,
            gy: value + 4.0,
            gz: value + 5.0,
        }
    }
}
