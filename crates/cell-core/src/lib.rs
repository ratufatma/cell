#![no_std]

pub mod tensor;

use core::fmt;

pub use tensor::{DType, MutableState, Ready, TensorChunk, TensorShape};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceContext {
    pub trace_id: u64,
    pub timestamp: u64,
    pub origin_node: u16,
}

impl TraceContext {
    pub const fn new(trace_id: u64, timestamp: u64, origin_node: u16) -> Self {
        Self {
            trace_id,
            timestamp,
            origin_node,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawPayload {
    context: TraceContext,
    data: [u8; 512],
    len: usize,
}

impl RawPayload {
    pub fn new(context: TraceContext, input: &[u8]) -> Result<Self, WorkResult> {
        if input.len() > 512 {
            return Err(WorkResult::Failed {
                context,
                node_id: 0,
                reason: "payload exceeds 512 bytes",
            });
        }

        let mut data = [0; 512];
        data[..input.len()].copy_from_slice(input);
        Ok(Self {
            context,
            data,
            len: input.len(),
        })
    }

    pub const fn context(&self) -> TraceContext {
        self.context
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn validate(self, verifier: &ValidatorCapability) -> ValidatedPayload {
        let Self { context, data, len } = self;
        ValidatedPayload {
            context,
            data,
            len,
            token: verifier.token,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatedPayload {
    context: TraceContext,
    data: [u8; 512],
    len: usize,
    token: u32,
}

impl ValidatedPayload {
    pub const fn context(&self) -> TraceContext {
        self.context
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn token(&self) -> u32 {
        self.token
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ValidatorCapability {
    token: u32,
}

impl ValidatorCapability {
    pub const fn new(token: u32) -> Self {
        Self { token }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkResult {
    Completed(TraceContext),
    Failed {
        context: TraceContext,
        node_id: u16,
        reason: &'static str,
    },
}

impl WorkResult {
    pub const fn context(&self) -> TraceContext {
        match *self {
            Self::Completed(context) => context,
            Self::Failed { context, .. } => context,
        }
    }
}

impl fmt::Display for WorkResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed(context) => write!(formatter, "trace {} completed", context.trace_id),
            Self::Failed {
                context,
                node_id,
                reason,
            } => write!(
                formatter,
                "trace {} failed at node {}: {}",
                context.trace_id, node_id, reason
            ),
        }
    }
}

pub trait KernelModule {
    type Input;
    type Output;

    fn process(&mut self, item: Self::Input) -> Result<Self::Output, WorkResult>;
    fn reset_state(&mut self);
}
