use argh::FromArgs;
use peroxide::prelude::*;
use peroxide::*;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(FromArgs)]
/// Calculate values for the Markov network
struct Calculate {
    #[argh(switch, short = 'v')]
    /// verbosity
    verbose: bool,
    #[argh(option, short = 'o')]
    /// where to save CSV encoded values
    output: Option<String>,
    #[argh(switch, short = 'V')]
    /// version
    version: bool,
    #[argh(option, short = 'i', default = "1")]
    /// number of iterations
    iter: usize,
}

fn main() {
    let c: Calculate = argh::from_env();

    if c.version {
        println!("{}", env!("GIT_HASH"));
    } else {
        let probability_matrix: Matrix =
            matrix(c!(0.1, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4), 3, 3, Row);
        let pm_arc = Arc::new(probability_matrix);

        let mut value_m: Option<Arc<Mutex<Matrix>>> = None;
        let mut durations: Vec<Duration> = Vec::new();
        let start = Instant::now();
        (0..c.iter).for_each(|_x| {
            let (val, dur) = run_iter(pm_arc.clone(), c.verbose);
            durations.push(dur);
            if c.iter == 1 {
                value_m = Some(val);
            }
        });

        let elapsed = start.elapsed();
        let total: Duration = durations.iter().sum();
        let avg_duration = total / (durations.len() as u32);
        println!("--- Summary ---");
        println!("Average duration: {:?}", avg_duration);
        println!("Time to complete {} iterations: {:?}\n", c.iter, elapsed);

        if let Some(path) = c.output {
            if c.iter == 1 {
                let inner = Arc::try_unwrap(value_m.unwrap()).unwrap();
                inner.into_inner().unwrap().write(&path).unwrap();
            }
        }
    }
}

fn run_iter(p_m: Arc<Matrix>, verbose: bool) -> (Arc<Mutex<Matrix>>, std::time::Duration) {
    let value_arc = Arc::new(Mutex::new(matrix(c!(0.0, 0.0, 0.0), 1, 3, Row)));
    let time_start = Instant::now();
    let handle = value_iter_thread(value_arc.clone(), p_m, 0.9, verbose);

    handle.join().unwrap();
    let elapsed = time_start.elapsed();
    if verbose {
        println!("Elapsed time: {:?}", elapsed);
    }

    (value_arc, elapsed)
}

fn value_iter_thread(
    value_matrix: Arc<Mutex<Matrix>>,
    probability_matrix: Arc<Matrix>,
    discount_factor: f64,
    verbose: bool,
) -> JoinHandle<()> {
    let handle = std::thread::spawn(move || {
        let v_m = value_matrix;
        let p_m = probability_matrix;
        let d_f = discount_factor;
        let rew = Arc::new(matrix(c!(5.0, 2.5, 2.5), 1, 3, Row));

        let mut i: usize = 0;
        loop {
            if verbose {
                println!("--- Iteration {} ---", i);
            }
            let mut diff = value_iter(v_m.clone(), p_m.clone(), d_f, rew.clone(), verbose);
            if verbose {
                println!("Mean diff: {}", diff);
            }

            diff = if diff < 0.0 { diff * -1.0 } else { diff };
            if diff < 0.000001 {
                break;
            }

            i += 1;
        }

        println!("Finished converging at {} iterations", i);
    });

    handle
}

fn value_iter(
    value_matrix: Arc<Mutex<Matrix>>,
    p_distrib: Arc<Matrix>,
    disc_fac: f64,
    reward: Arc<Matrix>,
    verbose: bool,
) -> f64 {
    let mut inner = value_matrix.lock().unwrap();
    let mut calc_result: Vec<f64> = Vec::new();
    (0..3).for_each(|index| {
        let rowvec = p_distrib.row(index);
        let p_row_matrix = matrix(rowvec, 1, 3, Row);
        let scaled = inner.mul_scalar(disc_fac);
        let add = reward.as_ref() + &scaled;
        let prod = p_row_matrix.hadamard(&add);
        calc_result.push(max(prod.row(0)));
    });

    let new = matrix(calc_result, 1, 3, Row);
    let diff: Matrix = &(*inner) - &new;
    let diff = {
        let inner = diff.row(0);
        let amt: f64 = inner.iter().sum();
        amt / 3.0
    };

    *inner = new;
    if verbose {
        inner.print();
    }

    diff
}
