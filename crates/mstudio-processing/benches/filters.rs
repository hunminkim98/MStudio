//! Plan §7 Phase 2 target: full-take Butterworth on 300 markers × 50 000
//! frames in < 200 ms on 8 cores.

use criterion::{criterion_group, criterion_main, Criterion};
use mstudio_processing::{filter_take, Filter};
use ndarray::Array3;

fn synthetic(n_frames: usize, n_markers: usize) -> Array3<f64> {
    Array3::from_shape_fn((n_frames, n_markers, 3), |(f, m, k)| {
        let t = f as f64 / 120.0;
        (t * (1.0 + m as f64 * 0.01) + k as f64).sin() + 0.001 * ((f * 7 + m * 13 + k) % 17) as f64
    })
}

fn bench_filters(c: &mut Criterion) {
    let frames = synthetic(50_000, 300);
    let markers: Vec<usize> = (0..300).collect();
    let mut g = c.benchmark_group("filter_take 300x50000");
    g.sample_size(10);
    for (name, filter) in [
        ("butterworth", Filter::Butterworth { order: 4, cutoff_hz: 6.0 }),
        ("gaussian", Filter::Gaussian { sigma_kernel: 3.0 }),
        ("median", Filter::Median { kernel_size: 5.0 }),
        ("loess", Filter::Loess { nb_values_used: 10.0 }),
    ] {
        g.bench_function(name, |b| {
            b.iter_batched(
                || frames.clone(),
                |mut f| filter_take(&mut f, &markers, &filter, 120.0).unwrap(),
                criterion::BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench_filters);
criterion_main!(benches);
