//! MStudio desktop application.
//!
//! ```text
//! mstudio [file.trc | file.c3d | json_folder]
//!         [--play] [--demo] [--selftest] [--screenshot out.png] [--exit-after SECONDS]   # automated visual checks
//! ```

#![allow(deprecated)] // glam 0.33 camera helpers used via mstudio-render

mod app;
mod jobs;
mod marker_plot;
mod panels;
mod theme;
mod timeline;
mod viewport;

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

pub struct LaunchOptions {
    pub path: Option<std::path::PathBuf>,
    pub screenshot: Option<std::path::PathBuf>,
    pub exit_after: Option<f32>,
    pub play: bool,
    /// Select a marker, show names + trajectory and a frame range (for screenshots).
    pub demo: bool,
    /// Run a filter job, an edit + undo and a report without dialogs, then exit.
    pub selftest: bool,
}

fn parse_args() -> LaunchOptions {
    let mut opts =
        LaunchOptions { path: None, screenshot: None, exit_after: None, play: false, demo: false, selftest: false };
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

fn icon() -> Option<Arc<egui::IconData>> {
    let bytes = include_bytes!("../../../Content/icon.ico");
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Ico).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(Arc::new(egui::IconData { rgba: img.into_raw(), width, height }))
}

fn main() -> eframe::Result {
    let opts = parse_args();

    // Large takes need more than wgpu's default 128 MiB storage binding.
    let mut wgpu_setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| wgpu::DeviceDescriptor {
        label: Some("mstudio"),
        required_limits: adapter.limits(),
        ..Default::default()
    });

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1500.0, 950.0])
        .with_min_inner_size([900.0, 600.0])
        .with_title("MStudio")
        .with_app_id("mstudio");
    if let Some(icon) = icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 24,
        multisampling: app::MSAA as u16,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            surface: egui_wgpu::SurfaceConfig {
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: Some(2),
            },
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native("mstudio", options, Box::new(move |cc| Ok(Box::new(app::App::new(cc, opts)))))
}
