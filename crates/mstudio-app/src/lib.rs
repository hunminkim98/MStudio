//! MStudio desktop application (eframe + egui_dock + wgpu viewport).
//!
//! The binary in `main.rs` parses the command line and calls [`run`]; the
//! Python package (`mstudio.run()`) calls the same function in-process.

#![allow(deprecated)] // glam 0.33 camera helpers used via mstudio-render

mod app;
mod jobs;
mod marker_plot;
mod panels;
mod theme;
mod timeline;
mod viewport;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

/// What to open and which automated behaviours to enable.
#[derive(Debug, Clone, Default)]
pub struct LaunchOptions {
    /// `.trc`, `.c3d` or a Pose2Sim / Sports2D JSON folder.
    pub path: Option<PathBuf>,
    /// Write a PNG of the window after the first frames were drawn.
    pub screenshot: Option<PathBuf>,
    /// Close the window after this many seconds.
    pub exit_after: Option<f32>,
    /// Start playback immediately.
    pub play: bool,
    /// Select a marker, show names + trajectory and a frame range (for screenshots).
    pub demo: bool,
    /// Run a filter job, an edit + undo and a report without dialogs, then exit.
    pub selftest: bool,
}

fn icon() -> Option<Arc<egui::IconData>> {
    let bytes = include_bytes!("../../../Content/icon.ico");
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Ico).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(Arc::new(egui::IconData { rgba: img.into_raw(), width, height }))
}

/// Open the main window and run the event loop until it is closed.
///
/// Must be called on the process's main thread (a winit requirement on macOS)
/// and at most once per process.
pub fn run(opts: LaunchOptions) -> eframe::Result {
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
