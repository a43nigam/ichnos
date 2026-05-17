use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use chrono::NaiveDateTime;
use mcap::records::MessageHeader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use robotics_core::PoseSample;
use robotics_core::{Result, RoboticsError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct SyntheticConfig {
    pub hz: f64,
    pub duration_ns: i64,
    pub start_ts_ns: i64,
}

#[derive(Debug, Clone)]
pub struct McapJsonPoseConfig {
    pub topic: String,
    pub default_robot_id: String,
    pub default_session_id: String,
}

#[derive(Debug, Clone)]
pub struct McapPoseConfig {
    pub topic: String,
    pub default_robot_id: String,
    pub default_session_id: String,
}

impl Default for McapJsonPoseConfig {
    fn default() -> Self {
        Self {
            topic: "/pose".to_string(),
            default_robot_id: "robot_01".to_string(),
            default_session_id: "session_001".to_string(),
        }
    }
}

impl Default for McapPoseConfig {
    fn default() -> Self {
        Self {
            topic: "/pose".to_string(),
            default_robot_id: "robot_01".to_string(),
            default_session_id: "session_001".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPoseMessage {
    pub timestamp_ns: Option<i64>,
    pub robot_id: Option<String>,
    pub session_id: Option<String>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub qw: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

#[derive(Debug, Clone)]
pub struct KittiOxtsConfig {
    pub robot_id: String,
    pub session_id: String,
}

#[derive(Debug, Clone)]
pub struct NuscenesEgoConfig {
    pub robot_id: String,
    pub session_id: String,
}

impl Default for KittiOxtsConfig {
    fn default() -> Self {
        Self {
            robot_id: "kitti_vehicle".to_string(),
            session_id: "kitti_session".to_string(),
        }
    }
}

impl Default for NuscenesEgoConfig {
    fn default() -> Self {
        Self {
            robot_id: "nuscenes_ego".to_string(),
            session_id: "nuscenes_scene".to_string(),
        }
    }
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            hz: 100.0,
            duration_ns: 1_000_000_000,
            start_ts_ns: 0,
        }
    }
}

pub fn generate_synthetic_pose(
    robot_id: &str,
    session_id: &str,
    config: SyntheticConfig,
) -> Vec<PoseSample> {
    let step_ns = (1_000_000_000.0 / config.hz).round() as i64;
    let mut samples = Vec::new();
    let mut timestamp_ns = config.start_ts_ns;
    let end_ns = config.start_ts_ns + config.duration_ns;

    while timestamp_ns <= end_ns {
        let seconds = (timestamp_ns - config.start_ts_ns) as f64 / 1_000_000_000.0;
        let x = 2.0 * seconds;
        let y = (seconds * std::f64::consts::TAU).sin();
        samples.push(PoseSample {
            timestamp_ns,
            robot_id: robot_id.to_string(),
            session_id: session_id.to_string(),
            x,
            y,
            z: 0.0,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
            vx: 2.0,
            vy: (seconds * std::f64::consts::TAU).cos(),
            vz: 0.0,
        });
        timestamp_ns += step_ns;
    }

    samples
}

pub fn pose_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("robot_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("qw", DataType::Float64, false),
        Field::new("qx", DataType::Float64, false),
        Field::new("qy", DataType::Float64, false),
        Field::new("qz", DataType::Float64, false),
        Field::new("vx", DataType::Float64, false),
        Field::new("vy", DataType::Float64, false),
        Field::new("vz", DataType::Float64, false),
        Field::new("velocity", DataType::Float64, false),
    ]))
}

pub fn write_pose_parquet(
    path: impl AsRef<Path>,
    samples: &[PoseSample],
    row_group_rows: usize,
) -> Result<usize> {
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    if row_group_rows == 0 {
        return Err(RoboticsError::InvalidArgument(
            "row_group_rows must be positive".to_string(),
        ));
    }

    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|err| RoboticsError::Io(err.to_string()))?;
    }

    let file = File::create(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let schema = pose_schema();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_rows))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut row_groups = 0;

    for chunk in samples.chunks(row_group_rows) {
        let batch = pose_batch(schema.clone(), chunk)?;
        writer
            .write(&batch)
            .map_err(|err| RoboticsError::Io(err.to_string()))?;
        writer
            .flush()
            .map_err(|err| RoboticsError::Io(err.to_string()))?;
        row_groups += 1;
    }

    writer
        .close()
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    Ok(row_groups)
}

pub fn write_synthetic_parquet(
    path: impl AsRef<Path>,
    robot_id: &str,
    session_id: &str,
    config: SyntheticConfig,
    row_group_rows: usize,
) -> Result<usize> {
    let samples = generate_synthetic_pose(robot_id, session_id, config);
    write_pose_parquet(path, &samples, row_group_rows)
}

pub fn read_pose_parquet_row_groups(
    path: impl AsRef<Path>,
    row_group_ids: &[u32],
) -> Result<Vec<PoseSample>> {
    if row_group_ids.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let file = File::open(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let row_group_count = builder.metadata().num_row_groups();
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

    let mut reader = builder
        .with_row_groups(row_groups)
        .build()
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut samples = Vec::new();
    for batch in &mut reader {
        samples.extend(pose_samples_from_batch(
            &batch.map_err(|err| RoboticsError::Io(err.to_string()))?,
        )?);
    }
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    samples.sort_by_key(|sample| sample.timestamp_ns);
    Ok(samples)
}

pub fn read_json_pose_mcap(
    path: impl AsRef<Path>,
    config: &McapJsonPoseConfig,
) -> Result<Vec<PoseSample>> {
    let bytes = std::fs::read(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut samples = Vec::new();

    for message in mcap::MessageStream::new(&bytes)
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?
    {
        let message = message.map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
        if message.channel.topic != config.topic {
            continue;
        }
        let payload: JsonPoseMessage = serde_json::from_slice(&message.data)
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
        samples.push(payload.into_pose_sample(message.log_time, config)?);
    }

    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    samples.sort_by_key(|sample| sample.timestamp_ns);
    Ok(samples)
}

pub fn read_pose_mcap(path: impl AsRef<Path>, config: &McapPoseConfig) -> Result<Vec<PoseSample>> {
    let bytes = std::fs::read(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut decoded = Vec::new();

    for message in mcap::MessageStream::new(&bytes)
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?
    {
        let message = message.map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
        if message.channel.topic != config.topic {
            continue;
        }
        if let Some(sample) = decode_mcap_pose_message(&message, config)? {
            decoded.push(sample);
        }
    }

    if decoded.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    decoded.sort_by_key(|sample| sample.sample.timestamp_ns);
    if !decoded.iter().any(|sample| sample.has_velocity) {
        derive_velocities(&mut decoded);
    }
    Ok(decoded.into_iter().map(|sample| sample.sample).collect())
}

pub fn write_pose_mcap_to_parquet(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    config: &McapPoseConfig,
    row_group_rows: usize,
) -> Result<(usize, usize)> {
    let samples = read_pose_mcap(input, config)?;
    let row_groups = write_pose_parquet(output, &samples, row_group_rows)?;
    Ok((samples.len(), row_groups))
}

pub fn write_json_pose_mcap(
    path: impl AsRef<Path>,
    samples: &[PoseSample],
    topic: &str,
) -> Result<usize> {
    if samples.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|err| RoboticsError::Io(err.to_string()))?;
    }

    let file = File::create(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut writer = mcap::WriteOptions::default()
        .compression(None)
        .create(BufWriter::new(file))
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let schema_id = writer
        .add_schema(
            "robotics.pose.JsonPoseMessage",
            "jsonschema",
            JSON_POSE_SCHEMA.as_bytes(),
        )
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    let channel_id = writer
        .add_channel(schema_id, topic, "json", &BTreeMap::new())
        .map_err(|err| RoboticsError::Io(err.to_string()))?;

    for (sequence, sample) in samples.iter().enumerate() {
        let payload = JsonPoseMessage::from(sample);
        let data = serde_json::to_vec(&payload)
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
        writer
            .write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence: sequence as u32,
                    log_time: sample.timestamp_ns as u64,
                    publish_time: sample.timestamp_ns as u64,
                },
                &data,
            )
            .map_err(|err| RoboticsError::Io(err.to_string()))?;
    }

    writer
        .finish()
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    Ok(samples.len())
}

pub fn write_json_pose_mcap_to_parquet(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    config: &McapJsonPoseConfig,
    row_group_rows: usize,
) -> Result<(usize, usize)> {
    let samples = read_json_pose_mcap(input, config)?;
    let row_groups = write_pose_parquet(output, &samples, row_group_rows)?;
    Ok((samples.len(), row_groups))
}

pub fn read_kitti_oxts(
    input: impl AsRef<Path>,
    config: &KittiOxtsConfig,
) -> Result<Vec<PoseSample>> {
    let input = input.as_ref();
    let oxts_dir = resolve_oxts_dir(input)?;
    let data_dir = oxts_dir.join("data");
    let mut data_files = std::fs::read_dir(&data_dir)
        .map_err(|err| RoboticsError::Io(err.to_string()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|err| RoboticsError::Io(err.to_string()))?;
    data_files.retain(|path| path.extension().is_some_and(|extension| extension == "txt"));
    data_files.sort();
    if data_files.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }

    let timestamps = read_kitti_timestamps(&oxts_dir.join("timestamps.txt"))?;
    if !timestamps.is_empty() && timestamps.len() != data_files.len() {
        return Err(RoboticsError::InvalidArgument(format!(
            "KITTI timestamp count {} does not match OXTS packet count {}",
            timestamps.len(),
            data_files.len()
        )));
    }

    let packets = data_files
        .iter()
        .map(read_kitti_oxts_packet)
        .collect::<Result<Vec<_>>>()?;
    let scale = packets[0].lat.to_radians().cos();
    let origin = mercator_xy(packets[0].lat, packets[0].lon, scale);
    let mut samples = Vec::with_capacity(packets.len());

    for (index, packet) in packets.iter().enumerate() {
        let timestamp_ns = timestamps.get(index).copied().unwrap_or(index as i64);
        let (mx, my) = mercator_xy(packet.lat, packet.lon, scale);
        let (qw, qx, qy, qz) =
            quaternion_from_roll_pitch_yaw(packet.roll, packet.pitch, packet.yaw);
        samples.push(PoseSample {
            timestamp_ns,
            robot_id: config.robot_id.clone(),
            session_id: config.session_id.clone(),
            x: mx - origin.0,
            y: my - origin.1,
            z: packet.alt - packets[0].alt,
            qw,
            qx,
            qy,
            qz,
            vx: packet.ve,
            vy: packet.vn,
            vz: packet.vu,
        });
    }

    Ok(samples)
}

pub fn write_kitti_oxts_to_parquet(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    config: &KittiOxtsConfig,
    row_group_rows: usize,
) -> Result<(usize, usize)> {
    let samples = read_kitti_oxts(input, config)?;
    let row_groups = write_pose_parquet(output, &samples, row_group_rows)?;
    Ok((samples.len(), row_groups))
}

pub fn read_nuscenes_ego_pose(
    input: impl AsRef<Path>,
    config: &NuscenesEgoConfig,
) -> Result<Vec<PoseSample>> {
    let ego_pose_path = resolve_nuscenes_table(input.as_ref(), "ego_pose.json")?;
    let bytes = std::fs::read(&ego_pose_path).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let mut records: Vec<NuscenesEgoPose> = serde_json::from_slice(&bytes)
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
    if records.is_empty() {
        return Err(RoboticsError::EmptyInput);
    }
    records.sort_by_key(|record| record.timestamp);

    let mut samples = records
        .into_iter()
        .map(|record| {
            if record.translation.len() != 3 || record.rotation.len() != 4 {
                return Err(RoboticsError::InvalidArgument(
                    "nuScenes ego_pose translation must have 3 values and rotation must have 4"
                        .to_string(),
                ));
            }
            let timestamp_ns = record.timestamp.checked_mul(1_000).ok_or_else(|| {
                RoboticsError::InvalidArgument("nuScenes timestamp overflowed ns".to_string())
            })?;
            Ok(PoseSample {
                timestamp_ns,
                robot_id: config.robot_id.clone(),
                session_id: config.session_id.clone(),
                x: record.translation[0],
                y: record.translation[1],
                z: record.translation[2],
                qw: record.rotation[0],
                qx: record.rotation[1],
                qy: record.rotation[2],
                qz: record.rotation[3],
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    derive_pose_sample_velocities(&mut samples);
    Ok(samples)
}

pub fn write_nuscenes_ego_pose_to_parquet(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    config: &NuscenesEgoConfig,
    row_group_rows: usize,
) -> Result<(usize, usize)> {
    let samples = read_nuscenes_ego_pose(input, config)?;
    let row_groups = write_pose_parquet(output, &samples, row_group_rows)?;
    Ok((samples.len(), row_groups))
}

impl JsonPoseMessage {
    fn into_pose_sample(
        self,
        fallback_timestamp_ns: u64,
        config: &McapJsonPoseConfig,
    ) -> Result<PoseSample> {
        let fallback_timestamp_ns = i64::try_from(fallback_timestamp_ns).map_err(|_| {
            RoboticsError::InvalidArgument("MCAP log_time exceeds i64 nanoseconds".to_string())
        })?;
        Ok(PoseSample {
            timestamp_ns: self.timestamp_ns.unwrap_or(fallback_timestamp_ns),
            robot_id: self
                .robot_id
                .unwrap_or_else(|| config.default_robot_id.clone()),
            session_id: self
                .session_id
                .unwrap_or_else(|| config.default_session_id.clone()),
            x: self.x,
            y: self.y,
            z: self.z,
            qw: self.qw,
            qx: self.qx,
            qy: self.qy,
            qz: self.qz,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
        })
    }
}

impl From<&PoseSample> for JsonPoseMessage {
    fn from(sample: &PoseSample) -> Self {
        Self {
            timestamp_ns: Some(sample.timestamp_ns),
            robot_id: Some(sample.robot_id.clone()),
            session_id: Some(sample.session_id.clone()),
            x: sample.x,
            y: sample.y,
            z: sample.z,
            qw: sample.qw,
            qx: sample.qx,
            qy: sample.qy,
            qz: sample.qz,
            vx: sample.vx,
            vy: sample.vy,
            vz: sample.vz,
        }
    }
}

#[derive(Debug, Clone)]
struct DecodedPoseSample {
    sample: PoseSample,
    has_velocity: bool,
}

fn decode_mcap_pose_message(
    message: &mcap::Message<'_>,
    config: &McapPoseConfig,
) -> Result<Option<DecodedPoseSample>> {
    let schema_name = message
        .channel
        .schema
        .as_ref()
        .map(|schema| schema.name.as_str())
        .unwrap_or("");
    let encoding = message.channel.message_encoding.as_str();

    if encoding == "json" {
        return decode_json_schema_pose(message, schema_name, config).map(Some);
    }
    if encoding == "ros1" {
        return decode_ros1_pose(message, schema_name, config).map(Some);
    }
    if encoding == "cdr" {
        return decode_cdr_pose(message, schema_name, config).map(Some);
    }
    Ok(None)
}

fn decode_json_schema_pose(
    message: &mcap::Message<'_>,
    schema_name: &str,
    config: &McapPoseConfig,
) -> Result<DecodedPoseSample> {
    if let Ok(payload) = serde_json::from_slice::<JsonPoseMessage>(&message.data) {
        return Ok(DecodedPoseSample {
            sample: payload.into_pose_sample(message.log_time, &config.clone().into())?,
            has_velocity: true,
        });
    }

    let value: serde_json::Value = serde_json::from_slice(&message.data)
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))?;
    let timestamp_ns =
        json_timestamp_ns(&value).unwrap_or(i64::try_from(message.log_time).map_err(|_| {
            RoboticsError::InvalidArgument("MCAP log_time exceeds i64 nanoseconds".to_string())
        })?);

    match schema_basename(schema_name) {
        "PoseStamped" => {
            let pose = value.get("pose").unwrap_or(&value);
            let (x, y, z, qw, qx, qy, qz) = json_pose_fields(pose)?;
            Ok(decoded_sample(
                timestamp_ns,
                config,
                x,
                y,
                z,
                qw,
                qx,
                qy,
                qz,
                [0.0, 0.0, 0.0],
                false,
            ))
        }
        "TransformStamped" => {
            let transform = value
                .get("transform")
                .ok_or_else(|| RoboticsError::InvalidArgument("missing transform".to_string()))?;
            let translation = transform.get("translation").ok_or_else(|| {
                RoboticsError::InvalidArgument("missing transform.translation".to_string())
            })?;
            let rotation = transform.get("rotation").ok_or_else(|| {
                RoboticsError::InvalidArgument("missing transform.rotation".to_string())
            })?;
            Ok(decoded_sample(
                timestamp_ns,
                config,
                json_f64(translation, "x")?,
                json_f64(translation, "y")?,
                json_f64(translation, "z")?,
                json_f64(rotation, "w")?,
                json_f64(rotation, "x")?,
                json_f64(rotation, "y")?,
                json_f64(rotation, "z")?,
                [0.0, 0.0, 0.0],
                false,
            ))
        }
        "Odometry" => {
            let pose = value
                .get("pose")
                .and_then(|pose| pose.get("pose"))
                .ok_or_else(|| RoboticsError::InvalidArgument("missing pose.pose".to_string()))?;
            let twist = value
                .get("twist")
                .and_then(|twist| twist.get("twist"))
                .and_then(|twist| twist.get("linear"));
            let (x, y, z, qw, qx, qy, qz) = json_pose_fields(pose)?;
            Ok(decoded_sample(
                timestamp_ns,
                config,
                x,
                y,
                z,
                qw,
                qx,
                qy,
                qz,
                [
                    twist.map_or(Ok(0.0), |linear| json_f64(linear, "x"))?,
                    twist.map_or(Ok(0.0), |linear| json_f64(linear, "y"))?,
                    twist.map_or(Ok(0.0), |linear| json_f64(linear, "z"))?,
                ],
                twist.is_some(),
            ))
        }
        _ => Err(RoboticsError::InvalidArgument(format!(
            "unsupported JSON MCAP pose schema {schema_name:?}"
        ))),
    }
}

fn decode_ros1_pose(
    message: &mcap::Message<'_>,
    schema_name: &str,
    config: &McapPoseConfig,
) -> Result<DecodedPoseSample> {
    let mut reader = Ros1Reader::new(&message.data);
    let timestamp_ns = i64::try_from(message.log_time).map_err(|_| {
        RoboticsError::InvalidArgument("MCAP log_time exceeds i64 nanoseconds".to_string())
    })?;

    match schema_basename(schema_name) {
        "PoseStamped" => {
            reader.skip_ros1_header()?;
            let pose = reader.read_pose()?;
            Ok(decoded_sample_from_pose(
                timestamp_ns,
                config,
                pose,
                [0.0; 3],
                false,
            ))
        }
        "TransformStamped" => {
            reader.skip_ros1_header()?;
            reader.read_string()?;
            let translation = reader.read_vec3()?;
            let rotation = reader.read_quat()?;
            Ok(decoded_sample(
                timestamp_ns,
                config,
                translation[0],
                translation[1],
                translation[2],
                rotation[0],
                rotation[1],
                rotation[2],
                rotation[3],
                [0.0; 3],
                false,
            ))
        }
        "Odometry" => {
            reader.skip_ros1_header()?;
            reader.read_string()?;
            let pose = reader.read_pose()?;
            reader.skip_f64s(36)?;
            let velocity = reader.read_vec3()?;
            reader.skip_vec3()?;
            Ok(decoded_sample_from_pose(
                timestamp_ns,
                config,
                pose,
                velocity,
                true,
            ))
        }
        _ => Err(RoboticsError::InvalidArgument(format!(
            "unsupported ROS1 MCAP pose schema {schema_name:?}"
        ))),
    }
}

fn decode_cdr_pose(
    message: &mcap::Message<'_>,
    schema_name: &str,
    config: &McapPoseConfig,
) -> Result<DecodedPoseSample> {
    let mut reader = CdrReader::new(&message.data)?;
    let timestamp_ns = i64::try_from(message.log_time).map_err(|_| {
        RoboticsError::InvalidArgument("MCAP log_time exceeds i64 nanoseconds".to_string())
    })?;

    match schema_basename(schema_name) {
        "PoseStamped" => {
            reader.skip_ros2_header()?;
            let pose = reader.read_pose()?;
            Ok(decoded_sample_from_pose(
                timestamp_ns,
                config,
                pose,
                [0.0; 3],
                false,
            ))
        }
        "TransformStamped" => {
            reader.skip_ros2_header()?;
            reader.read_string()?;
            let translation = reader.read_vec3()?;
            let rotation = reader.read_quat()?;
            Ok(decoded_sample(
                timestamp_ns,
                config,
                translation[0],
                translation[1],
                translation[2],
                rotation[0],
                rotation[1],
                rotation[2],
                rotation[3],
                [0.0; 3],
                false,
            ))
        }
        "Odometry" => {
            reader.skip_ros2_header()?;
            reader.read_string()?;
            let pose = reader.read_pose()?;
            reader.skip_f64s(36)?;
            let velocity = reader.read_vec3()?;
            reader.skip_vec3()?;
            Ok(decoded_sample_from_pose(
                timestamp_ns,
                config,
                pose,
                velocity,
                true,
            ))
        }
        _ => Err(RoboticsError::InvalidArgument(format!(
            "unsupported ROS2 CDR MCAP pose schema {schema_name:?}"
        ))),
    }
}

fn decoded_sample_from_pose(
    timestamp_ns: i64,
    config: &McapPoseConfig,
    pose: [f64; 7],
    velocity: [f64; 3],
    has_velocity: bool,
) -> DecodedPoseSample {
    decoded_sample(
        timestamp_ns,
        config,
        pose[0],
        pose[1],
        pose[2],
        pose[3],
        pose[4],
        pose[5],
        pose[6],
        velocity,
        has_velocity,
    )
}

#[allow(clippy::too_many_arguments)]
fn decoded_sample(
    timestamp_ns: i64,
    config: &McapPoseConfig,
    x: f64,
    y: f64,
    z: f64,
    qw: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    velocity: [f64; 3],
    has_velocity: bool,
) -> DecodedPoseSample {
    DecodedPoseSample {
        sample: PoseSample {
            timestamp_ns,
            robot_id: config.default_robot_id.clone(),
            session_id: config.default_session_id.clone(),
            x,
            y,
            z,
            qw,
            qx,
            qy,
            qz,
            vx: velocity[0],
            vy: velocity[1],
            vz: velocity[2],
        },
        has_velocity,
    }
}

fn derive_velocities(samples: &mut [DecodedPoseSample]) {
    for index in 0..samples.len() {
        let (before, after) = match (index.checked_sub(1), samples.get(index + 1)) {
            (Some(before), Some(_)) => (
                pose_motion_fields(&samples[before].sample),
                pose_motion_fields(&samples[index + 1].sample),
            ),
            (Some(before), None) => (
                pose_motion_fields(&samples[before].sample),
                pose_motion_fields(&samples[index].sample),
            ),
            (None, Some(_)) => (
                pose_motion_fields(&samples[index].sample),
                pose_motion_fields(&samples[index + 1].sample),
            ),
            (None, None) => continue,
        };
        let dt = (after.0 - before.0) as f64 / 1_000_000_000.0;
        if dt <= 0.0 {
            continue;
        }
        samples[index].sample.vx = (after.1 - before.1) / dt;
        samples[index].sample.vy = (after.2 - before.2) / dt;
        samples[index].sample.vz = (after.3 - before.3) / dt;
    }
}

fn derive_pose_sample_velocities(samples: &mut [PoseSample]) {
    for index in 0..samples.len() {
        let (before, after) = match (index.checked_sub(1), samples.get(index + 1)) {
            (Some(before), Some(_)) => (
                pose_motion_fields(&samples[before]),
                pose_motion_fields(&samples[index + 1]),
            ),
            (Some(before), None) => (
                pose_motion_fields(&samples[before]),
                pose_motion_fields(&samples[index]),
            ),
            (None, Some(_)) => (
                pose_motion_fields(&samples[index]),
                pose_motion_fields(&samples[index + 1]),
            ),
            (None, None) => continue,
        };
        let dt = (after.0 - before.0) as f64 / 1_000_000_000.0;
        if dt <= 0.0 {
            continue;
        }
        samples[index].vx = (after.1 - before.1) / dt;
        samples[index].vy = (after.2 - before.2) / dt;
        samples[index].vz = (after.3 - before.3) / dt;
    }
}

fn pose_motion_fields(sample: &PoseSample) -> (i64, f64, f64, f64) {
    (sample.timestamp_ns, sample.x, sample.y, sample.z)
}

impl From<McapPoseConfig> for McapJsonPoseConfig {
    fn from(config: McapPoseConfig) -> Self {
        Self {
            topic: config.topic,
            default_robot_id: config.default_robot_id,
            default_session_id: config.default_session_id,
        }
    }
}

fn schema_basename(schema_name: &str) -> &str {
    schema_name.rsplit('/').next().unwrap_or(schema_name)
}

fn json_pose_fields(value: &serde_json::Value) -> Result<(f64, f64, f64, f64, f64, f64, f64)> {
    let position = value
        .get("position")
        .ok_or_else(|| RoboticsError::InvalidArgument("missing position".to_string()))?;
    let orientation = value
        .get("orientation")
        .ok_or_else(|| RoboticsError::InvalidArgument("missing orientation".to_string()))?;
    Ok((
        json_f64(position, "x")?,
        json_f64(position, "y")?,
        json_f64(position, "z")?,
        json_f64(orientation, "w")?,
        json_f64(orientation, "x")?,
        json_f64(orientation, "y")?,
        json_f64(orientation, "z")?,
    ))
}

fn json_f64(value: &serde_json::Value, name: &str) -> Result<f64> {
    value
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| RoboticsError::InvalidArgument(format!("missing numeric field {name}")))
}

fn json_timestamp_ns(value: &serde_json::Value) -> Option<i64> {
    value
        .get("timestamp_ns")
        .and_then(serde_json::Value::as_i64)
        .or_else(|| {
            let stamp = value.get("header")?.get("stamp")?;
            let sec = stamp.get("sec")?.as_i64()?;
            let nsec = stamp
                .get("nanosec")
                .or_else(|| stamp.get("nsec"))?
                .as_i64()?;
            sec.checked_mul(1_000_000_000)?.checked_add(nsec)
        })
}

struct Ros1Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Ros1Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn skip_ros1_header(&mut self) -> Result<()> {
        self.read_u32()?;
        self.read_u32()?;
        self.read_u32()?;
        self.read_string()?;
        Ok(())
    }

    fn read_pose(&mut self) -> Result<[f64; 7]> {
        let position = self.read_vec3()?;
        let quat = self.read_quat()?;
        Ok([
            position[0],
            position[1],
            position[2],
            quat[0],
            quat[1],
            quat[2],
            quat[3],
        ])
    }

    fn read_vec3(&mut self) -> Result<[f64; 3]> {
        Ok([self.read_f64()?, self.read_f64()?, self.read_f64()?])
    }

    fn skip_vec3(&mut self) -> Result<()> {
        self.skip_f64s(3)
    }

    fn read_quat(&mut self) -> Result<[f64; 4]> {
        let x = self.read_f64()?;
        let y = self.read_f64()?;
        let z = self.read_f64()?;
        let w = self.read_f64()?;
        Ok([w, x, y, z])
    }

    fn skip_f64s(&mut self, count: usize) -> Result<()> {
        self.take(count * 8).map(|_| ())
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let bytes = self.take(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| RoboticsError::InvalidArgument("message cursor overflow".to_string()))?;
        let bytes = self.data.get(self.pos..end).ok_or_else(|| {
            RoboticsError::InvalidArgument("message ended while decoding ROS1 payload".to_string())
        })?;
        self.pos = end;
        Ok(bytes)
    }
}

struct CdrReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CdrReader<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(RoboticsError::InvalidArgument(
                "CDR payload missing encapsulation header".to_string(),
            ));
        }
        let little_endian = matches!(data[1], 0x01 | 0x03);
        if !little_endian {
            return Err(RoboticsError::InvalidArgument(
                "big-endian CDR payloads are not supported".to_string(),
            ));
        }
        Ok(Self { data, pos: 4 })
    }

    fn skip_ros2_header(&mut self) -> Result<()> {
        self.read_i32()?;
        self.read_u32()?;
        self.read_string()?;
        Ok(())
    }

    fn read_pose(&mut self) -> Result<[f64; 7]> {
        let position = self.read_vec3()?;
        let quat = self.read_quat()?;
        Ok([
            position[0],
            position[1],
            position[2],
            quat[0],
            quat[1],
            quat[2],
            quat[3],
        ])
    }

    fn read_vec3(&mut self) -> Result<[f64; 3]> {
        Ok([self.read_f64()?, self.read_f64()?, self.read_f64()?])
    }

    fn skip_vec3(&mut self) -> Result<()> {
        self.skip_f64s(3)
    }

    fn read_quat(&mut self) -> Result<[f64; 4]> {
        let x = self.read_f64()?;
        let y = self.read_f64()?;
        let z = self.read_f64()?;
        let w = self.read_f64()?;
        Ok([w, x, y, z])
    }

    fn skip_f64s(&mut self, count: usize) -> Result<()> {
        self.align(8)?;
        self.take(count * 8).map(|_| ())
    }

    fn read_string(&mut self) -> Result<String> {
        self.align(4)?;
        let len = self.read_u32_unaligned()? as usize;
        let bytes = self.take(len)?;
        let bytes = bytes.strip_suffix(&[0]).unwrap_or(bytes);
        String::from_utf8(bytes.to_vec())
            .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.align(4)?;
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.align(4)?;
        self.read_u32_unaligned()
    }

    fn read_u32_unaligned(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.align(8)?;
        let bytes = self.take(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("fixed length")))
    }

    fn align(&mut self, alignment: usize) -> Result<()> {
        let padding = (alignment - (self.pos % alignment)) % alignment;
        self.take(padding).map(|_| ())
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| RoboticsError::InvalidArgument("message cursor overflow".to_string()))?;
        let bytes = self.data.get(self.pos..end).ok_or_else(|| {
            RoboticsError::InvalidArgument("message ended while decoding CDR payload".to_string())
        })?;
        self.pos = end;
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Copy)]
struct KittiOxtsPacket {
    lat: f64,
    lon: f64,
    alt: f64,
    roll: f64,
    pitch: f64,
    yaw: f64,
    vn: f64,
    ve: f64,
    vu: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct NuscenesEgoPose {
    timestamp: i64,
    translation: Vec<f64>,
    rotation: Vec<f64>,
}

fn resolve_nuscenes_table(input: &Path, table: &str) -> Result<std::path::PathBuf> {
    let candidates = [
        input.join(table),
        input.join("v1.0-mini").join(table),
        input.join("v1.0-trainval").join(table),
    ];
    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| {
            RoboticsError::InvalidArgument(format!(
                "could not find nuScenes table {table} under {}",
                input.display()
            ))
        })
}

fn resolve_oxts_dir(input: &Path) -> Result<std::path::PathBuf> {
    if input.join("data").is_dir() {
        return Ok(input.to_path_buf());
    }
    let nested = input.join("oxts");
    if nested.join("data").is_dir() {
        return Ok(nested);
    }
    Err(RoboticsError::InvalidArgument(format!(
        "expected KITTI oxts directory or drive directory containing oxts/data: {}",
        input.display()
    )))
}

fn read_kitti_oxts_packet(path: impl AsRef<Path>) -> Result<KittiOxtsPacket> {
    let text =
        std::fs::read_to_string(path.as_ref()).map_err(|err| RoboticsError::Io(err.to_string()))?;
    let fields = text
        .split_whitespace()
        .map(|field| {
            field.parse::<f64>().map_err(|err| {
                RoboticsError::InvalidArgument(format!(
                    "invalid KITTI OXTS numeric field in {}: {err}",
                    path.as_ref().display()
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if fields.len() < 30 {
        return Err(RoboticsError::InvalidArgument(format!(
            "KITTI OXTS packet {} has {} fields, expected at least 30",
            path.as_ref().display(),
            fields.len()
        )));
    }
    Ok(KittiOxtsPacket {
        lat: fields[0],
        lon: fields[1],
        alt: fields[2],
        roll: fields[3],
        pitch: fields[4],
        yaw: fields[5],
        vn: fields[6],
        ve: fields[7],
        vu: fields[10],
    })
}

fn read_kitti_timestamps(path: &Path) -> Result<Vec<i64>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path).map_err(|err| RoboticsError::Io(err.to_string()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_kitti_timestamp_ns)
        .collect()
}

fn parse_kitti_timestamp_ns(line: &str) -> Result<i64> {
    let trimmed = line.trim();
    let timestamp =
        NaiveDateTime::parse_from_str(trimmed, "%Y-%m-%d %H:%M:%S%.f").map_err(|err| {
            RoboticsError::InvalidArgument(format!("invalid KITTI timestamp {trimmed}: {err}"))
        })?;
    timestamp.and_utc().timestamp_nanos_opt().ok_or_else(|| {
        RoboticsError::InvalidArgument(format!("KITTI timestamp out of range: {trimmed}"))
    })
}

fn mercator_xy(lat_deg: f64, lon_deg: f64, scale: f64) -> (f64, f64) {
    const EARTH_RADIUS_M: f64 = 6_378_137.0;
    let x = scale * lon_deg.to_radians() * EARTH_RADIUS_M;
    let y = scale
        * EARTH_RADIUS_M
        * ((std::f64::consts::FRAC_PI_4 + lat_deg.to_radians() / 2.0).tan()).ln();
    (x, y)
}

fn quaternion_from_roll_pitch_yaw(roll: f64, pitch: f64, yaw: f64) -> (f64, f64, f64, f64) {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let qw = cr * cp * cy + sr * sp * sy;
    let qx = sr * cp * cy - cr * sp * sy;
    let qy = cr * sp * cy + sr * cp * sy;
    let qz = cr * cp * sy - sr * sp * cy;
    (qw, qx, qy, qz)
}

fn pose_batch(schema: SchemaRef, samples: &[PoseSample]) -> Result<RecordBatch> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            samples
                .iter()
                .map(|sample| sample.timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            samples
                .iter()
                .map(|sample| sample.robot_id.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            samples
                .iter()
                .map(|sample| sample.session_id.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.x).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.y).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.z).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.qw).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.qx).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.qy).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.qz).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.vx).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.vy).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples.iter().map(|sample| sample.vz).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            samples
                .iter()
                .map(PoseSample::velocity_magnitude)
                .collect::<Vec<_>>(),
        )),
    ];

    RecordBatch::try_new(schema, columns)
        .map_err(|err| RoboticsError::InvalidArgument(err.to_string()))
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

const JSON_POSE_SCHEMA: &str = r#"{
  "type": "object",
  "required": ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"],
  "properties": {
    "timestamp_ns": {"type": "integer"},
    "robot_id": {"type": "string"},
    "session_id": {"type": "string"},
    "x": {"type": "number"},
    "y": {"type": "number"},
    "z": {"type": "number"},
    "qw": {"type": "number"},
    "qx": {"type": "number"},
    "qy": {"type": "number"},
    "qz": {"type": "number"},
    "vx": {"type": "number"},
    "vy": {"type": "number"},
    "vz": {"type": "number"}
  }
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_generator_includes_endpoints() {
        let samples = generate_synthetic_pose("r", "s", SyntheticConfig::default());

        assert_eq!(samples.first().unwrap().timestamp_ns, 0);
        assert_eq!(samples.last().unwrap().timestamp_ns, 1_000_000_000);
        assert_eq!(samples.len(), 101);
    }

    #[test]
    fn writes_synthetic_parquet_with_controlled_row_groups() {
        let path = std::env::temp_dir().join(format!(
            "robotics_synthetic_{}_{}.parquet",
            std::process::id(),
            "ingest"
        ));

        let row_groups = write_synthetic_parquet(
            &path,
            "humanoid_01",
            "session_001",
            SyntheticConfig {
                hz: 10.0,
                duration_ns: 900_000_000,
                start_ts_ns: 0,
            },
            5,
        )
        .unwrap();

        assert_eq!(row_groups, 2);
        assert!(path.metadata().unwrap().len() > 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn reads_selected_pose_parquet_row_groups() {
        let path = std::env::temp_dir().join(format!(
            "robotics_synthetic_{}_{}.parquet",
            std::process::id(),
            "row_group_read"
        ));
        write_synthetic_parquet(
            &path,
            "humanoid_01",
            "session_001",
            SyntheticConfig {
                hz: 10.0,
                duration_ns: 900_000_000,
                start_ts_ns: 0,
            },
            5,
        )
        .unwrap();

        let samples = read_pose_parquet_row_groups(&path, &[1]).unwrap();

        assert_eq!(samples.len(), 5);
        assert_eq!(samples.first().unwrap().timestamp_ns, 500_000_000);
        assert_eq!(samples.last().unwrap().timestamp_ns, 900_000_000);
        assert!(samples
            .iter()
            .all(|sample| sample.robot_id == "humanoid_01"));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn reads_json_pose_mcap_messages() {
        let path = std::env::temp_dir().join(format!(
            "robotics_pose_{}_{}.mcap",
            std::process::id(),
            "read"
        ));
        let samples = generate_synthetic_pose(
            "humanoid_01",
            "session_mcap",
            SyntheticConfig {
                hz: 10.0,
                duration_ns: 200_000_000,
                start_ts_ns: 100,
            },
        );

        let written = write_json_pose_mcap(&path, &samples, "/pose").unwrap();
        let read = read_json_pose_mcap(
            &path,
            &McapJsonPoseConfig {
                topic: "/pose".to_string(),
                default_robot_id: "fallback".to_string(),
                default_session_id: "fallback".to_string(),
            },
        )
        .unwrap();

        assert_eq!(written, samples.len());
        assert_eq!(read.len(), samples.len());
        assert_eq!(read[0].robot_id, "humanoid_01");
        assert_eq!(read[0].session_id, "session_mcap");
        assert_eq!(read[0].timestamp_ns, 100);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn reads_ros1_pose_stamped_mcap_messages() {
        let path = std::env::temp_dir().join(format!(
            "robotics_pose_{}_{}.mcap",
            std::process::id(),
            "ros1_pose"
        ));
        let file = File::create(&path).unwrap();
        let mut writer = mcap::WriteOptions::default()
            .compression(None)
            .create(BufWriter::new(file))
            .unwrap();
        let schema_id = writer
            .add_schema(
                "geometry_msgs/PoseStamped",
                "ros1msg",
                b"Header header\nPose pose\n",
            )
            .unwrap();
        let channel_id = writer
            .add_channel(schema_id, "/pose", "ros1", &BTreeMap::new())
            .unwrap();
        let payload = ros1_pose_stamped_payload(1.0, 2.0, 3.0);
        writer
            .write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence: 0,
                    log_time: 123,
                    publish_time: 123,
                },
                &payload,
            )
            .unwrap();
        writer.finish().unwrap();

        let samples = read_pose_mcap(
            &path,
            &McapPoseConfig {
                topic: "/pose".to_string(),
                default_robot_id: "robot_ros".to_string(),
                default_session_id: "session_ros".to_string(),
            },
        )
        .unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].timestamp_ns, 123);
        assert_eq!(samples[0].robot_id, "robot_ros");
        assert_eq!(samples[0].x, 1.0);
        assert_eq!(samples[0].y, 2.0);
        assert_eq!(samples[0].z, 3.0);
        assert_eq!(samples[0].qw, 1.0);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn converts_json_pose_mcap_to_parquet() {
        let mcap_path = std::env::temp_dir().join(format!(
            "robotics_pose_{}_{}.mcap",
            std::process::id(),
            "parquet"
        ));
        let parquet_path = std::env::temp_dir().join(format!(
            "robotics_pose_{}_{}.parquet",
            std::process::id(),
            "mcap"
        ));
        let samples = generate_synthetic_pose("robot", "session", SyntheticConfig::default());
        write_json_pose_mcap(&mcap_path, &samples, "/pose").unwrap();

        let (sample_count, row_groups) = write_json_pose_mcap_to_parquet(
            &mcap_path,
            &parquet_path,
            &McapJsonPoseConfig::default(),
            50,
        )
        .unwrap();

        assert_eq!(sample_count, samples.len());
        assert_eq!(row_groups, 3);
        assert!(parquet_path.metadata().unwrap().len() > 0);
        std::fs::remove_file(mcap_path).ok();
        std::fs::remove_file(parquet_path).ok();
    }

    #[test]
    fn reads_kitti_oxts_directory() {
        let root =
            std::env::temp_dir().join(format!("robotics_kitti_{}_{}", std::process::id(), "read"));
        let data_dir = root.join("oxts").join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        write_oxts_packet(
            data_dir.join("0000000000.txt"),
            49.0,
            8.0,
            100.0,
            1.0,
            2.0,
            3.0,
        );
        write_oxts_packet(
            data_dir.join("0000000001.txt"),
            49.000001,
            8.000001,
            101.0,
            1.5,
            2.5,
            3.5,
        );
        std::fs::write(
            root.join("oxts").join("timestamps.txt"),
            "2011-09-26 13:02:25.000000000\n2011-09-26 13:02:25.100000000\n",
        )
        .unwrap();

        let samples = read_kitti_oxts(
            &root,
            &KittiOxtsConfig {
                robot_id: "kitti_car".to_string(),
                session_id: "drive_0001".to_string(),
            },
        )
        .unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].robot_id, "kitti_car");
        assert_eq!(samples[0].session_id, "drive_0001");
        assert_eq!(samples[0].x, 0.0);
        assert_eq!(samples[0].y, 0.0);
        assert_eq!(samples[0].z, 0.0);
        assert_eq!(samples[0].vx, 2.0);
        assert_eq!(samples[0].vy, 1.0);
        assert_eq!(samples[0].vz, 3.0);
        assert!(samples[1].x > 0.0);
        assert!(samples[1].y > 0.0);
        assert_eq!(samples[1].z, 1.0);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn converts_kitti_oxts_to_parquet() {
        let root = std::env::temp_dir().join(format!(
            "robotics_kitti_{}_{}",
            std::process::id(),
            "parquet"
        ));
        let data_dir = root.join("oxts").join("data");
        let parquet_path = root.join("kitti.parquet");
        std::fs::create_dir_all(&data_dir).unwrap();
        write_oxts_packet(
            data_dir.join("0000000000.txt"),
            49.0,
            8.0,
            100.0,
            1.0,
            2.0,
            3.0,
        );
        write_oxts_packet(
            data_dir.join("0000000001.txt"),
            49.000001,
            8.000001,
            101.0,
            1.5,
            2.5,
            3.5,
        );

        let (samples, row_groups) =
            write_kitti_oxts_to_parquet(&root, &parquet_path, &KittiOxtsConfig::default(), 1)
                .unwrap();

        assert_eq!(samples, 2);
        assert_eq!(row_groups, 2);
        assert!(parquet_path.metadata().unwrap().len() > 0);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn reads_nuscenes_ego_pose_table() {
        let root = std::env::temp_dir().join(format!(
            "robotics_nuscenes_{}_{}",
            std::process::id(),
            "ego"
        ));
        let version_dir = root.join("v1.0-mini");
        std::fs::create_dir_all(&version_dir).unwrap();
        std::fs::write(
            version_dir.join("ego_pose.json"),
            r#"[
              {"token":"a","timestamp":1000000,"translation":[0.0,0.0,0.0],"rotation":[1.0,0.0,0.0,0.0]},
              {"token":"b","timestamp":2000000,"translation":[2.0,0.0,0.0],"rotation":[1.0,0.0,0.0,0.0]}
            ]"#,
        )
        .unwrap();

        let samples = read_nuscenes_ego_pose(
            &root,
            &NuscenesEgoConfig {
                robot_id: "ego".to_string(),
                session_id: "mini".to_string(),
            },
        )
        .unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].timestamp_ns, 1_000_000_000);
        assert_eq!(samples[0].robot_id, "ego");
        assert_eq!(samples[0].vx, 2.0);
        assert_eq!(samples[1].vx, 2.0);

        std::fs::remove_dir_all(root).ok();
    }

    fn write_oxts_packet(
        path: impl AsRef<Path>,
        lat: f64,
        lon: f64,
        alt: f64,
        vn: f64,
        ve: f64,
        vu: f64,
    ) {
        let values = [
            lat, lon, alt, 0.1, 0.2, 0.3, vn, ve, 4.0, 5.0, vu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0, 7.0, 5.0, 5.0, 5.0,
        ];
        let line = values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(path, line).unwrap();
    }

    fn ros1_pose_stamped_payload(x: f64, y: f64, z: f64) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&123_u32.to_le_bytes());
        payload.extend_from_slice(&0_u32.to_le_bytes());
        for value in [x, y, z, 0.0, 0.0, 0.0, 1.0_f64] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        payload
    }
}
