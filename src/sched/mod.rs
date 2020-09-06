mod system;

use crate::Task;
use crossbeam::queue::SegQueue;
use std::sync::{Arc, RwLock};
use system::System;

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
    sys: Arc<RwLock<System>>,
    q: Arc<SegQueue<Task>>,
    im: Arc<SegQueue<Task>>,
    threads: Vec<Thread>,
}

impl MarkovScheduler {
    pub fn new(num_threads: usize) -> MarkovScheduler {
        let threads: Vec<Thread> = (0..num_threads).map(|_x| Thread::new()).collect();

        MarkovScheduler {
            sys: Arc::new(RwLock::new(System::new([0.0, 1.0, 1.0]).init(DISCOUNT_FACTOR))),
            q: Arc::new(SegQueue::new()),
            im: Arc::new(SegQueue::new()),
            threads,
        }
    }

    pub fn launch(&mut self) {
        let sys_ref = self.sys.clone();
        let q_ref = self.q.clone();
        self.q.push(Task::construct(Box::pin( async move {
            update_loop(sys_ref, q_ref)
        })));

        // launch main loop
        loop {
            // capture ExtState
            let ext_state= ExtState(self.q.len(), self.im.len());
            let action = {
                let inner = self.sys.read().unwrap();
                inner.decide(ext_state)
            };

            let task_res = match action.0 {
                1 => self.q.pop(),
                2 => self.im.pop(),
                _ => continue
            };

            let task = match task_res {
                Ok(task) => {
                    // move to next state
                    let mut inner = self.sys.write().unwrap();
                    (*inner).state = action.0;
                    task
                },
                Err(_) => continue,
            };

            let (thr, _n) = self.threads
                .iter()
                .map(|x| (x, x.q.len()))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap();

            thr.q.push(task);
        }
    }
}

fn update_loop(sys_ref: Arc<RwLock<System>>, q_ref: Arc<SegQueue<Task>>) -> () {
    let (new_val, diff) = {
        let inner = sys_ref.read().unwrap();
        let n_val = (*inner).calculate_values(DISCOUNT_FACTOR);
        let diff: f64 = n_val
            .iter()
            .zip((*inner).values.iter())
            .map(|(x, y)| {
                let diff = x - y;
                if diff < 0.0 {
                    diff * -1.0
                } else {
                    diff
                }
            })
            .sum();

        (n_val, diff)
    };

    {
        let mut inner_sys = sys_ref.write().unwrap();
        (*inner_sys).values = new_val;
    }

    // launch next value iteration
    if diff > 0.1 {
        q_ref.clone().push(Task::construct(Box::pin(async move {
            update_loop(sys_ref, q_ref)
        })));
    }
}

pub struct ExtState(usize, usize);