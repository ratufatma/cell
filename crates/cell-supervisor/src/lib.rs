#![no_std]

pub mod telemetry;

use cell_core::{KernelModule, TraceContext, WorkResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FailureRecord {
    pub context: TraceContext,
    pub node_id: u16,
    pub reason: &'static str,
}

pub struct Supervisor<const CAP: usize> {
    records: [Option<FailureRecord>; CAP],
    len: usize,
}

impl<const CAP: usize> Supervisor<CAP> {
    pub const fn new() -> Self {
        Self {
            records: [const { None }; CAP],
            len: 0,
        }
    }

    pub fn observe(&mut self, result: WorkResult) -> Option<FailureRecord> {
        let WorkResult::Failed {
            context,
            node_id,
            reason,
        } = result
        else {
            return None;
        };

        let record = FailureRecord {
            context,
            node_id,
            reason,
        };
        if self.len < CAP {
            self.records[self.len] = Some(record);
            self.len += 1;
        }
        Some(record)
    }

    pub fn failure_count(&self) -> usize {
        self.len
    }

    pub fn find_trace(&self, trace_id: u64) -> Option<FailureRecord> {
        self.records[..self.len]
            .iter()
            .flatten()
            .find(|record| record.context.trace_id == trace_id)
            .copied()
    }

    pub fn reset_failed_node<M>(&self, node: &mut M, result: WorkResult) -> bool
    where
        M: KernelModule,
    {
        match result {
            WorkResult::Failed { .. } => {
                node.reset_state();
                true
            }
            WorkResult::Completed(_) => false,
        }
    }
}

impl<const CAP: usize> Default for Supervisor<CAP> {
    fn default() -> Self {
        Self::new()
    }
}
