#![allow(dead_code)]
use peroxide::prelude::*;
use peroxide::*;

pub trait Logistic {
    fn func(&self, input: &Matrix) -> Matrix;
    fn deriv(&self, input: &Matrix) -> Matrix;
}

// helper function
fn apply_elementwise<F: Fn(&f64) -> f64>(input: &Matrix, f: F) -> Matrix {
    let n_row = &input.row;
    let n_col = &input.col;
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

pub mod train {
    pub trait Train {}
}

pub struct Network<Log: Logistic> {
    logistic: Box<Log>,
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
    pub fn accumulate_activ_input(&self, input: Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut inputs: Vec<Matrix> = Vec::new();
        let mut activations: Vec<Matrix> = Vec::new();

        inputs.push(input);
        (0..3).for_each(|index| {
            let (out, actv) = self.feed(&inputs[inputs.len() - 1], index);
            inputs.push(out);
            activations.push(actv);
        });

        (inputs, activations)
    }
}
