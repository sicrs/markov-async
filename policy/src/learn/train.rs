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

    fn apply_bias_weight_set(&mut self, a: (Vec<Matrix>, Vec<Matrix>))  {
        let new_count = self.count + 1;
        let (b, w) = a;

        let new_biases = self.biases.iter().zip(b.iter()).map(|(x, y)| {
            let scaled = x.mul_scalar(self.count as f64);
            let res = &scaled + y;
            drop(scaled);

            res / (new_count as f64)
        }).collect::<Vec<Matrix>>();
        drop(b);

        let new_weights = self.weights.iter().zip(w.iter()).map(|(x, y)| {
            let scaled = x.mul_scalar(self.count as f64);
            let res: Matrix = &scaled + y;
            drop(scaled);

            res / (new_count as f64)
        }).collect::<Vec<Matrix>>();
        drop(w);

        self.biases = new_biases;
        self.weights = new_weights;
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