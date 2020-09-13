use argh::FromArgs;
use peroxide::prelude::*;
use peroxide::*;
use std::sync::Arc;
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
    #[argh(
        option,
        short = 'i',
        from_str_fn(parse_rewards),
        default = "vec![5.0,2.5,2.5]"
    )]
    /// comma separated rewards values
    rewards: Vec<f64>,
    #[argh(option, short = 'd', from_str_fn(parse_values), default = "vec![0.9]")]
    /// comma separated discount factors, produces a values for each factor given
    discount: Vec<f64>,
    #[argh(
        option,
        short = 'p',
        from_str_fn(parse_probability),
        default = "vec![0.1, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4]"
    )]
    /// probability distribution in the format p00,p01,p02;p10...p12;p20...p22
    probability: Vec<f64>,
}

fn parse_values(_value: &str) -> Result<Vec<f64>, String> {
    let collect: Vec<f64> = _value
        .split(",")
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    Ok(collect)
}

fn parse_rewards(_value: &str) -> Result<Vec<f64>, String> {
    let res = parse_rewards(_value).unwrap();
    if res.len() != 3 {
        Err(format!(
            "{} rewards values are given, instead of 3",
            res.len()
        ))
    } else {
        Ok(res)
    }
}

fn parse_probability(_value: &str) -> Result<Vec<f64>, String> {
    let mut collect: Vec<f64> = Vec::new();
    _value.split(";").for_each(|x| {
        let row = parse_values(x).unwrap();
        if row.len() != 3 {
            panic!("{} probability values are given, instead of 3", row.len());
        }
        row.iter().for_each(|x| collect.push(*x));
    });

    if collect.len() != 9 {
        Err(format!(
            "{} probability values given, instead of 9",
            collect.len()
        ))
    } else {
        Ok(collect)
    }
}

fn main() {
    let c: Calculate = argh::from_env();

    if c.version {
        println!(
            "markov-model v{}: {}",
            env!("CARGO_PKG_VERSION"),
            env!("GIT_HASH")
        );
    } else {
        let pm_arc = Arc::new(matrix(c.probability.clone(), 3, 3, Row));
        let init_v: Arc<Matrix> = Arc::new(zeros(1, 3));
        let rewards: Arc<Matrix> = Arc::new(matrix(c.rewards.clone(), 1, 3, Row));

        let mut disc_val_collect: Vec<(f64, Matrix)> = Vec::new();
        for d in c.discount.iter() {
            let mut values: Vec<Matrix> = Vec::new();

            let mut durations: Vec<Duration> = Vec::new();
            let start: Instant = Instant::now();
            (0..c.iter).for_each(|_x| {
                let (val, dur) = run_itr(
                    init_v.clone(),
                    pm_arc.clone(),
                    *d,
                    rewards.clone(),
                    c.verbose,
                );
                durations.push(dur);
                values.push(val);
            });

            let elapsed = start.elapsed();
            let total: Duration = durations.iter().sum();
            let avg_duration = total / (durations.len() as u32);

            println!("--- Summary ---");
            println!("Average duration: {:?}", avg_duration);
            println!("Time to complete {} iterations: {:?}\n", c.iter, elapsed);

            if let &Some(_) = &c.output {
                let mut sum: Matrix = zeros(1, 3);
                for matrix in values.into_iter() {
                    sum = sum + matrix;
                }

                let avg = sum.mul_scalar(1.0 / (c.iter as f64));
                drop(sum);
                disc_val_collect.push((*d, avg));
            }
        }

        if let Some(path) = c.output {
            let mut collect_matrix: Vec<f64> = Vec::new();
            let c_m_len = disc_val_collect.len();
            for (disc, matrix) in disc_val_collect.into_iter() {
                collect_matrix = c!(collect_matrix; vec![disc]; matrix.row(0));
            }

            let matr: Matrix = matrix(collect_matrix, c_m_len, 4, Row);
            //matr.write_with_header(&path, vec!["disc factor", "v0", "v1", "v2"]).unwrap();
            matr.write(&path).unwrap();
        }
    }
}

/// Wrapper around value_iter_thread to measure time elapsed
fn run_itr(
    initial_value: Arc<Matrix>,
    probability_distrib: Arc<Matrix>,
    disc_fac: f64,
    rewards: Arc<Matrix>,
    verbose: bool,
) -> (Matrix, Duration) {
    let time_start = Instant::now();
    let handle = value_iter_thread(
        initial_value,
        probability_distrib,
        disc_fac,
        rewards,
        verbose,
    );

    let out = handle.join().unwrap();
    let elapsed = time_start.elapsed();
    if verbose {
        println!("Time elapsed: {:?}", elapsed);
    }

    (out, elapsed)
}

/// Runs value iteration on a thread
fn value_iter_thread(
    initial_v_matrix: Arc<Matrix>,
    probability_matrix: Arc<Matrix>,
    discount_factor: f64,
    rewards: Arc<Matrix>,
    verbose: bool,
) -> JoinHandle<Matrix> {
    let handle: JoinHandle<Matrix> = std::thread::spawn(move || {
        let mut v_m = initial_v_matrix.as_ref().clone();
        let p_m = probability_matrix.as_ref();
        let rew = rewards.as_ref();

        let mut i: usize = 0;
        loop {
            if verbose {
                println!("--- Iteration {} ---", i);
            }

            let (m, mut diff) = value_iter(&v_m, p_m, discount_factor, rew, verbose);
            v_m = m;

            if verbose {
                println!("Mean diff: {}", diff);
            }

            diff = if diff < 0.0 { diff * -1.0 } else { diff };
            if diff < 0.000001 {
                break;
            }
            i += 1;
        }

        if verbose {
            println!("Finished converging at {} iterations", i + 1);
        }

        v_m
    });

    handle
}

/// Mathematical logic
fn value_iter(
    v_matrix: &Matrix,
    p_distrib: &Matrix,
    disc_fac: f64,
    reward: &Matrix,
    verbose: bool,
) -> (Matrix, f64) {
    let mut calc_result: Vec<f64> = Vec::new();

    (0..3).for_each(|index| {
        let p_row_matrix: Matrix = matrix(p_distrib.row(index), 1, 3, Row);
        let scaled: Matrix = v_matrix.mul_scalar(disc_fac);
        let add: Matrix = reward + &scaled;
        let hdm_prod = p_row_matrix.hadamard(&add);
        calc_result.push(max(hdm_prod.row(0)));
    });

    let new: Matrix = matrix(calc_result, 1, 3, Row);
    let diff: Matrix = v_matrix - &new;
    let diff: f64 = {
        let inner = diff.row(0);
        let amt: f64 = inner.iter().sum();
        amt / 3.0
    };

    if verbose {
        new.print();
    }

    (new, diff)
}
