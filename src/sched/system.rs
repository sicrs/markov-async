const P_DISTRIB: [[f64; 3]; 3] = [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]];

pub struct System {
    pub state: SystemState,
    pub values: [f64; 3],
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

    pub fn init(mut self, discount_factor: f64) -> System {
        let new_vals = self.calculate_values(discount_factor);
        self.values = new_vals;

        self
    }

    /// NOTE: Don't forget to acquire a WriteLock on System
    /// Implements Value Iteration
    pub fn calculate_values(&self, discount_factor: f64) -> [f64; 3] {
        let mut buf: [f64; 3] = [0.0; 3];

        // iterate for each of the values
        (0..self.values.len()).for_each(|_index| {
            let val = &P_DISTRIB[_index]
                .iter()
                .zip(0..3) // destination node index
                .map(|(p, idx)| p * (self.rewards[idx] + discount_factor * self.values[idx]))
                .max_by(|a, b| a.partial_cmp(b).unwrap());

            if let Some(valu) = val {
                buf[_index] = *valu;
            } else {
                panic!("Zero length output");
            }
        });

        buf
    }
}

pub enum SystemState {
    Idle = 0,
    DoQueue = 1,
    DoImmediate = 2,
}

struct Action(SystemState);
