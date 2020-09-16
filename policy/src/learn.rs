#![allow(dead_code)]
use logistic::Logistic;
use peroxide::fuga::rbind;
use peroxide::prelude::*;
use peroxide::*;
use train::Train;

pub mod logistic {
    use peroxide::prelude::*;
    pub trait Logistic {
        fn func(&self, input: &Matrix) -> Matrix;
        fn deriv(&self, input: &Matrix) -> Matrix;
    }

    // helper function
    fn apply_elementwise<F: Fn(&f64) -> f64>(input: &Matrix, f: F) -> Matrix {
        let (n_row, n_col) = (&input.row, &input.col);
        let mut out: Matrix = zeros(*n_row, *n_col);
        for i in 0..*n_row {
            for j in 0..*n_col {
                out[(i, j)] = f(&out[(i, j)]);
            }
        }
        out
    }

    pub struct Sigmoid();

    impl Logistic for Sigmoid {
        fn func(&self, input: &Matrix) -> Matrix {
            apply_elementwise(input, |x| 1f64 / (1f64 + std::f64::consts::E.powf(-*x)))
        }
        fn deriv(&self, input: &Matrix) -> Matrix {
            let sgm = self.func(&input);
            let ones = matrix(vec![1f64; sgm.row * sgm.col], sgm.row, sgm.col, Row);
            let sub = &ones - &sgm;
            drop(ones);

            sgm.hadamard(&sub)
        }
    }
}

pub mod train {
    use super::Matrix;
    pub trait Train {
        fn backprop(&self, input: Matrix, output: Matrix) -> (Vec<Matrix>, Vec<Matrix>);
    }
}

pub struct Network<L: Logistic> {
    logistic: Box<L>,
    weights: [Matrix; 3],
    biases: [Matrix; 3],
}

impl<L: Logistic> Network<L> {
    pub fn new(log: L) -> Network<L> {
        let weights: [Matrix; 3] = [
            matrix(vec![1.0; 15], 3, 5, Row),
            matrix(vec![1.0; 6], 2, 3, Row),
            matrix(vec![1.0; 6], 3, 2, Row),
        ];

        let biases: [Matrix; 3] = [
            matrix(vec![0.0; 3], 3, 1, Row),
            matrix(vec![0.0; 2], 2, 1, Row),
            matrix(vec![0.0; 3], 3, 1, Row),
        ];

        Network {
            logistic: Box::new(log),
            weights,
            biases,
        }
    }

    /// runs the input through the network
    pub fn passthrough(&self, input: Matrix) -> Matrix {
        let mut accumulator = input;

        for index in 0..3 {
            let weighted_in: Matrix = &self.weights[index] * &accumulator;
            let result: Matrix = &weighted_in + &self.biases[index];
            drop(weighted_in);

            let res = self.logistic.func(&result);
            drop(result);

            accumulator = res;
        }

        accumulator
    }

    pub fn construct_input_matrix(weighted_values: Matrix, n_im: usize, n_q: usize) -> Matrix {
        // construct the input vector
        // weighted_values is a horizontal 1x3 matrix
        let w_v_vec: Vec<f64> = weighted_values.row(0);
        drop(weighted_values);

        let construct = matrix(c!(vec![n_im as f64, n_q as f64]; w_v_vec), 5, 1, Row);

        construct
    }

    pub fn feed(&self, input: &Matrix, layer: usize) -> (Matrix, Matrix) {
        let weighted_in: Matrix = &self.weights[layer] + input;
        let result: Matrix = &weighted_in + &self.biases[layer];

        let activation = self.logistic.func(&result);

        (result, activation)
    }

    // returns a tuple of accumulated inputs and activations (in that order)
    pub fn accumulate_zs_activ(&self, input: Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut zs: Vec<Matrix> = Vec::new();
        let mut activations: Vec<Matrix> = Vec::new();

        activations.push(input);
        (0..3).for_each(|index| {
            let (out, actv) = self.feed(&activations[activations.len() - 1], index);
            //let (out, actv) = self.feed(&inputs[inputs.len() - 1], index);
            zs.push(out);
            activations.push(actv);
        });

        (zs, activations)
    }
}

impl<L: Logistic> Train for Network<L> {
    fn backprop(&self, input: Matrix, output: Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        // accumulate zs and activations
        let (zs, activations) = self.accumulate_zs_activ(input);

        // traverse backwards
        let mut rev_nabla_bias: Vec<Matrix> = Vec::new();
        let mut rev_nabla_weights: Vec<Matrix> = Vec::new();

        rev_nabla_bias.push({
            let deriv: Matrix = self.logistic.deriv(&zs[zs.len() - 1]);
            let diff: Matrix = &activations[activations.len() - 1] - &output;
            assert!(deriv.row == diff.row);
            assert!(deriv.col == diff.col);
            diff.hadamard(&deriv)
        });

        rev_nabla_weights.push({
            let delta = rev_nabla_bias[0].as_slice();
            let actv = activations[activations.len() - 2].transpose();
            let mut accumulator: Option<Matrix> = None;

            delta.iter().for_each(|multiplier| {
                let clone = actv.mul_scalar(*multiplier);
                if let Some(inner) = accumulator.take() {
                    accumulator = Some(rbind(inner, clone));
                } else {
                    accumulator = Some(clone);
                }
            });

            accumulator.take().unwrap()
        });

        for layer in 1..(self.weights.len() + 1) {
            rev_nabla_bias.push({
                // latest bias
                let prev_delta_bias = &rev_nabla_bias[rev_nabla_bias.len() - 1];
                let weights = &self.weights[self.weights.len() - layer];
                let log_prime = self.logistic.deriv(&zs[zs.len() - (layer + 1)]);

                let dot: Matrix = weights * prev_delta_bias;

                // TODO: remove checks
                assert!(log_prime.row == dot.row);
                assert!(log_prime.col == dot.col);

                dot.hadamard(&log_prime)
            });

            rev_nabla_weights.push({
                let delta = &rev_nabla_bias[rev_nabla_bias.len() - 1];
                let actv = activations[activations.len() - (layer + 1)].transpose();
                let mut accumulator: Option<Matrix> = None;

                delta.as_slice().iter().for_each(|multiplier| {
                    let clone = actv.mul_scalar(*multiplier);
                    if let Some(inner) = accumulator.take() {
                        accumulator = Some(rbind(inner, clone));
                    } else {
                        accumulator = Some(clone);
                    }
                });

                accumulator.take().unwrap()
            });
        }

        (
            rev_nabla_bias.into_iter().rev().collect::<Vec<Matrix>>(),
            rev_nabla_weights.into_iter().rev().collect::<Vec<Matrix>>(),
        )
    }
}
