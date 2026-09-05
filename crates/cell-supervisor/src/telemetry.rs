use cell_core::TraceContext;

pub const MAGIC: [u8; 4] = [0xce, 0x11, 0x54, 0x4d];
pub const VERSION: u8 = 1;
pub const HEADER_SIZE: usize = 12;
pub const MAX_FRAME_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TelemetryType {
    Heartbeat = 0x01,
    PmmSnapshot = 0x02,
    QueueMetrics = 0x03,
    TensorExecution = 0x04,
    FaultIncident = 0x05,
}

impl TelemetryType {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Heartbeat),
            0x02 => Some(Self::PmmSnapshot),
            0x03 => Some(Self::QueueMetrics),
            0x04 => Some(Self::TensorExecution),
            0x05 => Some(Self::FaultIncident),
            _ => None,
        }
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct TelemetryHeader {
    pub magic: [u8; 4],
    pub version: u8,
    pub event_type: TelemetryType,
    pub payload_len: u16,
    pub sequence: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryError {
    BufferTooSmall,
    InvalidMagic,
    UnsupportedVersion,
    UnknownType,
    InvalidPayloadLength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heartbeat {
    pub timestamp: u64,
    pub node_id: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PmmSnapshot {
    pub usable_frames: u64,
    pub free_frames: u64,
    pub largest_free_run: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueMetrics {
    pub queue_id: u16,
    pub capacity: u16,
    pub pushed: u32,
    pub popped: u32,
    pub dropped: u32,
    pub watermark_state: u8,
    pub _padding: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorExecution {
    pub context: TraceContext,
    pub elements: u32,
    pub frame_count: u16,
    pub dtype: u8,
    pub simd_level: u8,
    pub sum_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultIncident {
    pub context: TraceContext,
    pub node_id: u16,
    pub reason_code: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryEvent {
    Heartbeat(Heartbeat),
    PmmSnapshot(PmmSnapshot),
    QueueMetrics(QueueMetrics),
    TensorExecution(TensorExecution),
    FaultIncident(FaultIncident),
}

#[derive(Clone, Copy)]
pub struct EncodedFrame {
    bytes: [u8; MAX_FRAME_SIZE],
    len: usize,
}

impl EncodedFrame {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    pub const fn len(&self) -> usize {
        self.len
    }
}

pub struct TelemetryEncoder {
    sequence: u32,
}

impl TelemetryEncoder {
    pub const fn new() -> Self {
        Self { sequence: 0 }
    }

    pub fn encode(&mut self, event: TelemetryEvent) -> EncodedFrame {
        let mut frame = EncodedFrame {
            bytes: [0; MAX_FRAME_SIZE],
            len: HEADER_SIZE,
        };
        frame.bytes[..4].copy_from_slice(&MAGIC);
        frame.bytes[4] = VERSION;
        frame.bytes[6..10].copy_from_slice(&self.sequence.to_le_bytes());
        self.sequence = self.sequence.wrapping_add(1);

        let (event_type, payload_len) = match event {
            TelemetryEvent::Heartbeat(value) => {
                put_u64(&mut frame.bytes, HEADER_SIZE, value.timestamp);
                put_u16(&mut frame.bytes, HEADER_SIZE + 8, value.node_id);
                (TelemetryType::Heartbeat, 10)
            }
            TelemetryEvent::PmmSnapshot(value) => {
                put_u64(&mut frame.bytes, HEADER_SIZE, value.usable_frames);
                put_u64(&mut frame.bytes, HEADER_SIZE + 8, value.free_frames);
                put_u64(&mut frame.bytes, HEADER_SIZE + 16, value.largest_free_run);
                (TelemetryType::PmmSnapshot, 24)
            }
            TelemetryEvent::QueueMetrics(value) => {
                put_u16(&mut frame.bytes, HEADER_SIZE, value.queue_id);
                put_u16(&mut frame.bytes, HEADER_SIZE + 2, value.capacity);
                put_u32(&mut frame.bytes, HEADER_SIZE + 4, value.pushed);
                put_u32(&mut frame.bytes, HEADER_SIZE + 8, value.popped);
                put_u32(&mut frame.bytes, HEADER_SIZE + 12, value.dropped);
                frame.bytes[HEADER_SIZE + 16] = value.watermark_state;
                (TelemetryType::QueueMetrics, 20)
            }
            TelemetryEvent::TensorExecution(value) => {
                put_context(&mut frame.bytes, HEADER_SIZE, value.context);
                put_u32(&mut frame.bytes, HEADER_SIZE + 18, value.elements);
                put_u16(&mut frame.bytes, HEADER_SIZE + 22, value.frame_count);
                frame.bytes[HEADER_SIZE + 24] = value.dtype;
                frame.bytes[HEADER_SIZE + 25] = value.simd_level;
                put_u32(&mut frame.bytes, HEADER_SIZE + 26, value.sum_bits);
                (TelemetryType::TensorExecution, 30)
            }
            TelemetryEvent::FaultIncident(value) => {
                put_context(&mut frame.bytes, HEADER_SIZE, value.context);
                put_u16(&mut frame.bytes, HEADER_SIZE + 18, value.node_id);
                put_u16(&mut frame.bytes, HEADER_SIZE + 20, value.reason_code);
                (TelemetryType::FaultIncident, 22)
            }
        };
        frame.bytes[5] = event_type as u8;
        frame.bytes[10..12].copy_from_slice(&(payload_len as u16).to_le_bytes());
        frame.len += payload_len;
        frame
    }
}

impl Default for TelemetryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedTelemetry {
    Heartbeat(Heartbeat),
    PmmSnapshot(PmmSnapshot),
    QueueMetrics(QueueMetrics),
    TensorExecution(TensorExecution),
    FaultIncident(FaultIncident),
}

pub fn decode(bytes: &[u8]) -> Result<(u32, DecodedTelemetry), TelemetryError> {
    if bytes.len() < HEADER_SIZE {
        return Err(TelemetryError::BufferTooSmall);
    }
    if bytes[..4] != MAGIC {
        return Err(TelemetryError::InvalidMagic);
    }
    if bytes[4] != VERSION {
        return Err(TelemetryError::UnsupportedVersion);
    }
    let event_type = TelemetryType::from_byte(bytes[5]).ok_or(TelemetryError::UnknownType)?;
    let payload_len = read_u16(bytes, 10) as usize;
    if bytes.len() != HEADER_SIZE + payload_len {
        return Err(TelemetryError::InvalidPayloadLength);
    }
    let sequence = read_u32(bytes, 6);
    let payload = &bytes[HEADER_SIZE..];
    let decoded = match event_type {
        TelemetryType::Heartbeat if payload_len == 10 => DecodedTelemetry::Heartbeat(Heartbeat {
            timestamp: read_u64(payload, 0),
            node_id: read_u16(payload, 8),
        }),
        TelemetryType::PmmSnapshot if payload_len == 24 => {
            DecodedTelemetry::PmmSnapshot(PmmSnapshot {
                usable_frames: read_u64(payload, 0),
                free_frames: read_u64(payload, 8),
                largest_free_run: read_u64(payload, 16),
            })
        }
        TelemetryType::QueueMetrics if payload_len == 20 => {
            DecodedTelemetry::QueueMetrics(QueueMetrics {
                queue_id: read_u16(payload, 0),
                capacity: read_u16(payload, 2),
                pushed: read_u32(payload, 4),
                popped: read_u32(payload, 8),
                dropped: read_u32(payload, 12),
                watermark_state: payload[16],
                _padding: [0; 3],
            })
        }
        TelemetryType::TensorExecution if payload_len == 30 => {
            DecodedTelemetry::TensorExecution(TensorExecution {
                context: read_context(payload, 0),
                elements: read_u32(payload, 18),
                frame_count: read_u16(payload, 22),
                dtype: payload[24],
                simd_level: payload[25],
                sum_bits: read_u32(payload, 26),
            })
        }
        TelemetryType::FaultIncident if payload_len == 22 => {
            DecodedTelemetry::FaultIncident(FaultIncident {
                context: read_context(payload, 0),
                node_id: read_u16(payload, 18),
                reason_code: read_u16(payload, 20),
            })
        }
        _ => return Err(TelemetryError::InvalidPayloadLength),
    };
    Ok((sequence, decoded))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_context(bytes: &mut [u8], offset: usize, context: TraceContext) {
    put_u64(bytes, offset, context.trace_id);
    put_u64(bytes, offset + 8, context.timestamp);
    put_u16(bytes, offset + 16, context.origin_node);
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_context(bytes: &[u8], offset: usize) -> TraceContext {
    TraceContext::new(
        read_u64(bytes, offset),
        read_u64(bytes, offset + 8),
        read_u16(bytes, offset + 16),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_tensor_event_and_sequence() {
        let context = TraceContext::new(7, 99, 2);
        let mut encoder = TelemetryEncoder::new();
        let frame = encoder.encode(TelemetryEvent::TensorExecution(TensorExecution {
            context,
            elements: 4096,
            frame_count: 4,
            dtype: 0,
            simd_level: 2,
            sum_bits: 4096.0_f32.to_bits(),
        }));
        assert_eq!(&frame.as_bytes()[..4], &MAGIC);
        assert_eq!(frame.len(), HEADER_SIZE + 30);
        let (sequence, decoded) = decode(frame.as_bytes()).unwrap();
        assert_eq!(sequence, 0);
        assert_eq!(
            decoded,
            DecodedTelemetry::TensorExecution(TensorExecution {
                context,
                elements: 4096,
                frame_count: 4,
                dtype: 0,
                simd_level: 2,
                sum_bits: 4096.0_f32.to_bits(),
            })
        );
    }

    #[test]
    fn rejects_invalid_frame_header() {
        let mut encoder = TelemetryEncoder::new();
        let mut frame = encoder.encode(TelemetryEvent::Heartbeat(Heartbeat {
            timestamp: 1,
            node_id: 0,
        }));
        frame.bytes[0] = 0;
        assert_eq!(decode(frame.as_bytes()), Err(TelemetryError::InvalidMagic));
    }
}
