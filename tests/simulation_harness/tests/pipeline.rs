use cell_core::{
    KernelModule, RawPayload, TraceContext, ValidatedPayload, ValidatorCapability, WorkResult,
};
use cell_queue::SpscQueue;
use cell_supervisor::Supervisor;

const INGRESS: u16 = 1;
const WORKER: u16 = 3;

struct Worker {
    processed: usize,
    reset_count: usize,
}

impl KernelModule for Worker {
    type Input = ValidatedPayload;
    type Output = WorkResult;

    fn process(&mut self, item: Self::Input) -> Result<Self::Output, WorkResult> {
        if item.as_bytes().first() == Some(&0xff) {
            return Err(WorkResult::Failed {
                context: item.context(),
                node_id: WORKER,
                reason: "fault-injected payload",
            });
        }
        self.processed += 1;
        Ok(WorkResult::Completed(item.context()))
    }

    fn reset_state(&mut self) {
        self.reset_count += 1;
    }
}

#[test]
fn pipeline_survives_one_fault_in_ten_packets() {
    let ingress_to_validator = SpscQueue::<RawPayload, 10>::new();
    let validator_to_worker = SpscQueue::<ValidatedPayload, 10>::new();
    let verifier = ValidatorCapability::new(0xCE11_2021);
    let mut supervisor = Supervisor::<10>::new();
    let mut worker = Worker {
        processed: 0,
        reset_count: 0,
    };

    for trace_id in 0..10 {
        let byte = if trace_id == 4 { 0xff } else { trace_id as u8 };
        let context = TraceContext::new(trace_id, 1000 + trace_id, INGRESS);
        let raw = RawPayload::new(context, &[byte]).expect("test payload fits");
        ingress_to_validator
            .push(raw)
            .expect("ingress queue has capacity");
    }

    while let Some(raw) = ingress_to_validator.pop() {
        assert_eq!(raw.context().origin_node, INGRESS);
        let validated = raw.validate(&verifier);
        validator_to_worker
            .push(validated)
            .expect("worker queue has capacity");
    }

    let mut failures = 0;
    while let Some(validated) = validator_to_worker.pop() {
        match worker.process(validated) {
            Ok(WorkResult::Completed(_)) => {}
            Ok(WorkResult::Failed { .. }) => panic!("worker must return failures as Err"),
            Err(failure) => {
                failures += 1;
                assert!(supervisor.observe(failure).is_some());
                assert!(supervisor.reset_failed_node(&mut worker, failure));
            }
        }
    }

    assert_eq!(failures, 1);
    assert_eq!(worker.processed, 9);
    assert_eq!(worker.reset_count, 1);
    let record = supervisor.find_trace(4).expect("fault trace is recorded");
    assert_eq!(record.node_id, WORKER);
    assert_eq!(record.context.origin_node, INGRESS);
    assert_eq!(record.reason, "fault-injected payload");
}

#[test]
fn queue_is_bounded_and_non_blocking() {
    let queue = SpscQueue::<u32, 2>::new();
    assert!(queue.push(10).is_ok());
    assert!(queue.push(20).is_ok());
    assert!(queue.push(30).is_err());
    assert_eq!(queue.pop(), Some(10));
    assert!(queue.push(30).is_ok());
    assert_eq!(queue.pop(), Some(20));
    assert_eq!(queue.pop(), Some(30));
    assert_eq!(queue.pop(), None);
}
