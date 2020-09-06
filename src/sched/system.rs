use crate::sched::ExtState;

const P_DISTRIB: [[f64; 3]; 3] = [[0.1, 0.5, 0.4], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]];
const ACTION_SPACE: [usize; 3] = [0, 1, 2];

pub struct System {
    pub state: usize,
    pub values: [f64; 3],
    rewards: [f64; 3],
}

impl System {
    pub fn new(rewards: [f64; 3]) -> System {
        // calculate initial values

        System {
            state: 0,
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
        (0..self.values.len()).for_each(|index| {
            let val = &P_DISTRIB[index]
                .iter()
                .zip(0..3) // destination node index
                .map(|(p, idx)| p * (self.rewards[idx] + discount_factor * self.values[idx]))
                .max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

            buf[index] = *val;
        });

        buf
    }

    // TODO(Sebastian): account for external state
    /// calculates policy
    pub fn decide(&self, ext: ExtState) -> Action {
        let act_space = &ACTION_SPACE;
        let cur_state = &self.state;
        let (action, _value) = act_space
            .iter()
            .filter(|x| *x != cur_state)
            .map(|x| (*x, (P_DISTRIB[*cur_state][*x] * self.values[*x])))
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        
        Action(action)
    }
}

pub struct Action(usize);
