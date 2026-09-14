//! MStudio desktop application.
//!
//! ```text
//! mstudio [file.trc | file.c3d | json_folder]
//!         [--play] [--demo] [--selftest] [--screenshot out.png] [--exit-after SECONDS]   # automated visual checks
//! ```

use mstudio_app::LaunchOptions;

fn parse_args() -> LaunchOptions {
    let mut opts = LaunchOptions::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--screenshot" => {
                opts.screenshot = args.get(i + 1).map(Into::into);
                i += 2;
            }
            "--selftest" => {
                opts.selftest = true;
                i += 1;
            }
            "--demo" => {
                opts.demo = true;
                i += 1;
            }
            "--play" => {
                opts.play = true;
                i += 1;
            }
            "--exit-after" => {
                opts.exit_after = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            other => {
                opts.path = Some(other.into());
                i += 1;
            }
        }
    }
    opts
}

fn main() -> eframe::Result {
    mstudio_app::run(parse_args())
}
