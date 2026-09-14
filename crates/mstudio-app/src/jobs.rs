//! Background work (plan rule R4): filtering, interpolation, pattern-based
//! gap filling and report generation run on a thread; the UI keeps drawing
//! and applies the result when it arrives.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Instant;

use mstudio_core::{DirtyRange, Take};
use mstudio_processing::{apply_filter, interpolate_in_range, pattern_interpolate, Filter, InterpMethod};
use ndarray::{s, Array3};

#[allow(clippy::large_enum_variant)] // moved once into the worker thread
pub enum Job {
    /// `cols` is `[n, 1, 3]`: the marker's three coordinate columns.
    Filter {
        marker: usize,
        cols: Array3<f64>,
        first: usize,
        last: usize,
        filter: Filter,
        fps: f64,
    },
    Interp {
        marker: usize,
        cols: Array3<f64>,
        first: usize,
        last: usize,
        method: InterpMethod,
    },
    /// `sub` is `[n, 1 + refs, 3]`: target first, references after.
    Pattern {
        marker: usize,
        sub: Array3<f64>,
        first: usize,
        last: usize,
    },
    Report {
        take: Take,
        options: mstudio_report::ReportOptions,
        path: PathBuf,
    },
}

pub enum Outcome {
    /// New values for one marker; only `range` differs from the input.
    Columns {
        marker: usize,
        range: DirtyRange,
        cols: [Vec<f64>; 3],
        label: String,
    },
    Report {
        path: PathBuf,
    },
    Error(String),
}

pub struct Worker {
    rx: Receiver<Outcome>,
    pub label: String,
    pub started: Instant,
}

fn columns_of(a: &Array3<f64>, m: usize) -> [Vec<f64>; 3] {
    [a.slice(s![.., m, 0]).to_vec(), a.slice(s![.., m, 1]).to_vec(), a.slice(s![.., m, 2]).to_vec()]
}

impl Worker {
    pub fn spawn(job: Job) -> Worker {
        let (tx, rx) = mpsc::channel();
        let label = match &job {
            Job::Filter { filter, .. } => format!("{} filter", filter.name()),
            Job::Interp { method, .. } => format!("{} interpolation", method.name()),
            Job::Pattern { .. } => "pattern-based interpolation".into(),
            Job::Report { .. } => "report".into(),
        };
        let started = Instant::now();
        std::thread::spawn(move || {
            let outcome = match job {
                Job::Filter { marker, mut cols, first, last, filter, fps } => {
                    match apply_filter(&mut cols, 0, first, last, &filter, fps) {
                        Ok(range) => Outcome::Columns {
                            marker,
                            range,
                            cols: columns_of(&cols, 0),
                            label: format!("{} filter", filter.name()),
                        },
                        Err(e) => Outcome::Error(e.to_string()),
                    }
                }
                Job::Interp { marker, mut cols, first, last, method } => {
                    match interpolate_in_range(&mut cols, 0, first, last, method) {
                        Ok(range) => Outcome::Columns {
                            marker,
                            range,
                            cols: columns_of(&cols, 0),
                            label: format!("{} interpolation", method.name()),
                        },
                        Err(e) => Outcome::Error(e.to_string()),
                    }
                }
                Job::Pattern { marker, mut sub, first, last } => {
                    let refs: Vec<usize> = (1..sub.dim().1).collect();
                    match pattern_interpolate(&mut sub, 0, &refs, first, last) {
                        Ok(range) => Outcome::Columns {
                            marker,
                            range,
                            cols: columns_of(&sub, 0),
                            label: "pattern-based interpolation".into(),
                        },
                        Err(e) => Outcome::Error(e.to_string()),
                    }
                }
                Job::Report { take, options, path } => match mstudio_report::write_report(&take, &options, &path) {
                    Ok(()) => Outcome::Report { path },
                    Err(e) => Outcome::Error(e.to_string()),
                },
            };
            let _ = tx.send(outcome);
        });
        Worker { rx, label, started }
    }

    /// `Some` once the job finished (or its thread died).
    pub fn poll(&self) -> Option<Outcome> {
        match self.rx.try_recv() {
            Ok(o) => Some(o),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Outcome::Error("worker thread ended unexpectedly".into())),
        }
    }
}
