use peroxide::*;
use peroxide::prelude::*;
use std::sync::{ Arc, Mutex };
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

fn main() {
    let args = std::env::args();
    let arg: Vec<String> = args.collect();
    let num_iter = if arg.len() == 1 {
        1 as usize
    } else {
        arg[1].parse::<usize>().unwrap()
    };
    
    let probability_matrix: Matrix = matrix(c!(0.1, 0.5, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4), 3, 3, Row);
    let pm_arc = Arc::new(probability_matrix);

    let mut value_m: Option<Arc<Mutex<Matrix>>> = None;
    let mut durations: Vec<Duration> = Vec::new();
    let start = Instant::now();
    (0..num_iter).for_each(|_x| {
        let (val, dur) = run_iter(pm_arc.clone());
        durations.push(dur);
        if num_iter == 1 {
            value_m = Some(val);
        }
    });
    let elapsed = start.elapsed();
    let total: Duration = durations.iter().sum();

    let avg = total / (durations.len() as u32);

    println!("--- Summary ---");
    println!("Average duration: {:?}", avg);
    println!("Time to complete {} iterations: {:?}\n", num_iter, elapsed);
    if num_iter == 1 {
        let inner = Arc::try_unwrap(value_m.unwrap()).unwrap();
        inner.into_inner().unwrap().write("values.csv").unwrap();
    }
}

fn run_iter(p_m: Arc<Matrix>) -> (Arc<Mutex<Matrix>>, std::time::Duration) {
    let value_arc = Arc::new(Mutex::new(matrix(c!(0.0, 0.0, 0.0), 1, 3, Row)));
    let time_start = Instant::now();
    let handle = value_iter_thread(value_arc.clone(), p_m, 0.9);

    handle.join().unwrap();
    let elapsed = time_start.elapsed();
    println!("Elapsed time: {:?}", elapsed);
    
    (value_arc, elapsed)
}

fn value_iter_thread(value_matrix: Arc<Mutex<Matrix>>, probability_matrix: Arc<Matrix>, discount_factor: f64) -> JoinHandle<()> {
    let handle = std::thread::spawn(move || {
        let v_m = value_matrix;
        let p_m = probability_matrix;
        let d_f = discount_factor;
        let rew = Arc::new(matrix(c!(5.0, 2.5, 2.5), 1, 3, Row));

        let mut i: usize = 0;
        loop {
            if let Ok(_) = std::env::var("DBG") {
                println!("--- Iteration {} ---", i);    
            }
            let mut diff = value_iter(v_m.clone(), p_m.clone(), d_f, rew.clone());
            if let Ok(_) = std::env::var("DBG") {
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

fn value_iter(value_matrix: Arc<Mutex<Matrix>>, p_distrib: Arc<Matrix>, disc_fac: f64, reward: Arc<Matrix>) -> f64 {
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

    let new = matrix(calc_result,1, 3, Row);
    let diff: Matrix = &(*inner) - &new;
    let diff = {
        let inner = diff.row(0);
        let amt: f64 = inner.iter().sum();
        amt / 3.0
    };

    *inner = new;
    if let Ok(_) = std::env::var("DBG") {
        inner.print();
    }

    diff
}
