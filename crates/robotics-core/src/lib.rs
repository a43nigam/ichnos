pub type TimestampNs = i64;

#[derive(Debug, Clone, PartialEq)]
pub struct PoseSample {
    pub timestamp_ns: TimestampNs,
    pub robot_id: String,
    pub session_id: String,
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

impl PoseSample {
    pub fn velocity_magnitude(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }

    pub fn channels(&self) -> [f64; 10] {
        [
            self.x, self.y, self.z, self.qw, self.qx, self.qy, self.qz, self.vx, self.vy, self.vz,
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImuSample {
    pub timestamp_ns: TimestampNs,
    pub robot_id: String,
    pub session_id: String,
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
    pub gx: f64,
    pub gy: f64,
    pub gz: f64,
}

impl ImuSample {
    pub fn channels(&self) -> [f64; 6] {
        [self.ax, self.ay, self.az, self.gx, self.gy, self.gz]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogEntry {
    pub robot_id: String,
    pub session_id: String,
    pub file_uri: String,
    pub row_group_id: u32,
    pub start_ts_ns: TimestampNs,
    pub end_ts_ns: TimestampNs,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub min_z: f64,
    pub max_z: f64,
    pub min_velocity: f64,
    pub max_velocity: f64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub row_count: u64,
    pub gap_count: u64,
    pub max_gap_ns: TimestampNs,
    pub max_gap_start_ts_ns: TimestampNs,
    pub max_gap_end_ts_ns: TimestampNs,
    pub nominal_dt_ns: TimestampNs,
}

impl CatalogEntry {
    pub fn overlaps_time(&self, start: TimestampNs, end: TimestampNs) -> bool {
        self.start_ts_ns <= end && self.end_ts_ns >= start
    }

    pub fn overlaps_bbox(&self, bbox: &BoundingBox) -> bool {
        self.min_x <= bbox.max_x
            && self.max_x >= bbox.min_x
            && self.min_y <= bbox.max_y
            && self.max_y >= bbox.min_y
            && self.min_z <= bbox.max_z
            && self.max_z >= bbox.min_z
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub min_z: f64,
    pub max_z: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuerySpec {
    pub robot_id: Option<String>,
    pub start_ts_ns: TimestampNs,
    pub end_ts_ns: TimestampNs,
    pub bbox: Option<BoundingBox>,
    pub min_velocity: Option<f64>,
    pub target_hz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowRef {
    pub entry: CatalogEntry,
    pub clipped_start_ns: TimestampNs,
    pub clipped_end_ns: TimestampNs,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorBatch {
    pub timestamps_ns: Vec<TimestampNs>,
    pub values: Vec<f64>,
    pub rows: usize,
    pub channels: usize,
}

impl TensorBatch {
    pub fn row(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.channels)?;
        let end = start.checked_add(self.channels)?;
        self.values.get(start..end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoboticsError {
    EmptyInput,
    InvalidArgument(String),
    Extrapolation {
        requested_ns: TimestampNs,
        min_ns: TimestampNs,
        max_ns: TimestampNs,
    },
    Io(String),
}

impl std::fmt::Display for RoboticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input is empty"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::Extrapolation {
                requested_ns,
                min_ns,
                max_ns,
            } => write!(
                f,
                "requested timestamp {requested_ns} is outside [{min_ns}, {max_ns}]"
            ),
            Self::Io(message) => write!(f, "io error: {message}"),
        }
    }
}

impl std::error::Error for RoboticsError {}

pub type Result<T> = std::result::Result<T, RoboticsError>;
