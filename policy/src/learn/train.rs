use peroxide::*;
use peroxide::prelude::*;

pub trait Train {
    fn backprop(&self, input: Matrix, output: Matrix) -> (Vec<Matrix>, Vec<Matrix>);
}

pub trait DataAggregator: Iterator<Item = (Matrix, Matrix)> {

}

pub trait TrainingAlgo {
    fn train<D: DataAggregator>(&mut self, aggregator: D);
}

pub struct GradientDescentCommon {
    eta: f64,
    subsample_size: usize
}

struct Accumulator {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    count: usize,
}

impl Accumulator {
    fn new() -> Accumulator {
        Accumulator {
            weights: Vec::new(),
            biases: Vec::new(),
            count: 0,
        }
    }

    fn from_tuple(a: (Vec<Matrix>, Vec<Matrix>)) -> Accumulator {
        Accumulator {
            biases: a.0,
            weights: a.1,
            count: 1,
        }
    }

    fn apply_bias_weight_set(&mut self, a: (Vec<Matrix>, Vec<Matrix>))  {
        if self.biases.len() == 0 {
            assert!(self.biases.len() == self.weights.len());
            self.count = 1;
            self.biases = a.0;
            self.weights = a.1;
        } else {
            let to_merge = Accumulator::from_tuple(a);
            self.merge(to_merge);
        }
    }

    fn merge(&mut self, other: Self) {
        let other_biases = other.biases;
        let other_weights = other.weights;
        let other_count = other.count;
        let new_count = (self.count + other.count) as f64;

        let new_biases = self.biases.iter().zip(other_biases.iter()).map(|(x, y)| {
            let x_scaled = x.mul_scalar(self.count as f64);
            let y_scaled = y.mul_scalar(other_count as f64);

            let res = x_scaled + y_scaled;

            res / new_count
        }).collect::<Vec<Matrix>>();

        let new_weights = self.weights.iter().zip(other_weights.iter()).map(|(x, y)| {
            let x_scaled = x.mul_scalar(self.count as f64);
            let y_scaled = y.mul_scalar(other_count as f64);

            let res = x_scaled + y_scaled;

            res / new_count
        }).collect::<Vec<Matrix>>();

        self.biases = new_biases;
        self.weights = new_weights;
        self.count = new_count as usize;
    }
}

impl TrainingAlgo for GradientDescentCommon {
    fn train<D: DataAggregator>(&mut self, mut aggregator: D) {
        loop {
            match aggregator.next() {
                Some(tuple) => {
                    let mut accumulator = Accumulator::new();
                    todo!()
                },
                None => break,
            }
        }
    }
}