pub struct System {
    pub state: SystemState,
    values: [f64; 3],
    rewards: [f64; 3],
}

impl System {
    pub fn new(rewards: [f64; 3]) -> System {
        // calculate initial values 

        System {
            state: SystemState::Idle,
            values: [0.0; 3],
            rewards,
        }
    }
}

pub enum SystemState {
    Idle = 0,
    DoQueue = 1,
    DoImmediate = 2,
}

struct Action(SystemState);