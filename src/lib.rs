#![allow(dead_code, unused_imports)]
mod pool;
mod sched;

use futures::future::BoxFuture;
use std::sync::Mutex;

struct Runtime {
    sched: Box<dyn sched::Scheduler>,
}

struct Task {
    fut: Mutex<Option<BoxFuture<'static, ()>>>,
}

impl Task {
    fn construct(fut: BoxFuture<'static, ()>) -> Task {
        Task {
            fut: Mutex::new(Some(fut)),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
