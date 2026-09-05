use cell_queue::SpscQueue;
use std::sync::Arc;
use std::thread;

const MESSAGE_COUNT: u64 = 1_000_000;
const CAPACITY: usize = 1024;

#[test]
fn concurrency_stress_test() {
    let queue = Arc::new(SpscQueue::<u64, CAPACITY>::new());
    let (producer, consumer) = queue.split();
    thread::scope(|scope| {
        let producer_thread = scope.spawn(move || {
            for value in 0..MESSAGE_COUNT {
                loop {
                    if producer.push(value).is_ok() {
                        break;
                    }
                    thread::yield_now();
                }
            }
        });
        let consumer_thread = scope.spawn(move || {
            let mut received = 0;
            while received < MESSAGE_COUNT {
                match consumer.pop() {
                    Some(value) => {
                        assert_eq!(value, received, "SPSC queue violated FIFO ordering");
                        received += 1;
                    }
                    None => thread::yield_now(),
                }
            }
            received
        });

        producer_thread.join().expect("producer must not panic");
        assert_eq!(
            consumer_thread.join().expect("consumer must not panic"),
            MESSAGE_COUNT
        );
    });
}
