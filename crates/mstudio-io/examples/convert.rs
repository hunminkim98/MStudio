//! Convert between the formats MStudio reads and writes.
//!
//!     cargo run -p mstudio-io --example convert -- <in.trc|in.c3d|json_dir> <out.trc|out.c3d>

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: convert <in.trc|in.c3d|json_dir> <out.trc|out.c3d>");
        std::process::exit(2);
    }
    let take = match mstudio_io::load(&args[0]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read {}: {e}", args[0]);
            std::process::exit(1);
        }
    };
    if let Err(e) = mstudio_io::save(&args[1], &take) {
        eprintln!("write {}: {e}", args[1]);
        std::process::exit(1);
    }
    eprintln!("{} markers × {} frames @ {} Hz → {}", take.n_markers(), take.n_frames(), take.fps, args[1]);
}
