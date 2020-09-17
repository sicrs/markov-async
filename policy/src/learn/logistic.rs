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
        let sub = 1f64 - &sgm;

        sgm.hadamard(&sub)
    }
}
