mod system;

use crate::Task;
use crossbeam::queue::SegQueue;
use std::sync::{Arc, RwLock};
use system::{System, SystemState};

const DISCOUNT_FACTOR: f64 = 0.9;

pub(crate) trait Scheduler {}

struct Thread {
    q: Arc<SegQueue<Task>>,
    handle: std::thread::JoinHandle<()>,
}

impl Thread {
    /// spawns a thread with a task queue
    fn new() -> Thread {
        let q: Arc<SegQueue<Task>> = Arc::new(SegQueue::<Task>::new());
        let qclone = q.clone();
        let handle = std::thread::spawn(move || loop {
            match qclone.pop() {
                Ok(_task) => {
                    todo!();
                }
                _ => {}
            }
        });

        Thread { q, handle }
    }
}

pub(crate) struct MarkovScheduler {
    sys: RwLock<System>,
    q: Arc<SegQueue<Task>>,
    im: Arc<SegQueue<Task>>,
    threads: Vec<Thread>,
}

impl MarkovScheduler {
    pub fn new(num_threads: usize) -> MarkovScheduler {
        let threads: Vec<Thread> = (0..num_threads).map(|_x| Thread::new()).collect();

        MarkovScheduler {
            sys: RwLock::new(System::new([0.0, 1.0, 1.0]).init(DISCOUNT_FACTOR)),
            q: Arc::new(SegQueue::new()),
            im: Arc::new(SegQueue::new()),
            threads,
        }
    }

    pub fn launch(&mut self) {
        let sys_ref: &mut RwLock<System> = &mut (self.sys);
        self.q.push(Task::construct(Box::pin(async move {
            let new_val = {
                let inner = sys_ref.read().unwrap();
                (*inner).calculate_values(DISCOUNT_FACTOR)
            };
            
            {
                let inner_sys = sys_ref.write().unwrap();
                (*inner_sys).values = new_val;
            }
        })));
        loop {
            
        }
    }
}
