//! Headless rendering: draw a synthetic take into a texture and check pixels.
//! Skips (with a message) when the machine has no usable GPU adapter.

use glam::Vec3;
use mstudio_core::{CoordinateSystem, DirtyRange, Take, VisualSettings};
use mstudio_render::{grid_segments, project, Camera, FrameParams, MarkerState, RenderConfig, Renderer};
use ndarray::Array3;

const W: u32 = 256;
const H: u32 = 256;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    color: wgpu::Texture,
    depth: wgpu::Texture,
    readback: wgpu::Buffer,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    eprintln!("offscreen test on {:?}", adapter.get_info());
    let tex = |label, format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = tex("color", FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let depth = tex("depth", DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    Some(Gpu { device, queue, color, depth, readback })
}

/// Render one frame and return RGBA pixels, row-major from the top-left.
fn render(gpu: &Gpu, renderer: &mut Renderer, params: &FrameParams<'_>) -> Vec<[u8; 4]> {
    renderer.prepare(&gpu.device, &gpu.queue, params);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let color_view = gpu.color.create_view(&Default::default());
        let depth_view = gpu.depth.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.draw(&mut pass);
    }
    encoder.copy_texture_to_buffer(
        gpu.color.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &gpu.readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    let slice = gpu.readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let pixels: Vec<[u8; 4]> = slice.get_mapped_range().unwrap().as_chunks::<4>().0.to_vec();
    gpu.readback.unmap();
    pixels
}

fn synthetic_take() -> Take {
    // 3 markers on a line that translates along +x over 10 frames
    let mut f = Array3::<f64>::zeros((10, 3, 3));
    for i in 0..10 {
        for m in 0..3 {
            f[[i, m, 0]] = m as f64 * 0.4 - 0.4 + i as f64 * 0.05;
            f[[i, m, 1]] = 1.0;
        }
    }
    Take::new(vec!["a".into(), "b".into(), "c".into()], 60.0, f)
}

fn lit(pixels: &[[u8; 4]], x: f32, y: f32, radius: i32) -> usize {
    let (cx, cy) = (x.round() as i32, y.round() as i32);
    let mut n = 0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (px, py) = (cx + dx, cy + dy);
            if px >= 0 && py >= 0 && px < W as i32 && py < H as i32 {
                let p = pixels[(py as u32 * W + px as u32) as usize];
                if p[0] > 40 || p[1] > 40 || p[2] > 40 {
                    n += 1;
                }
            }
        }
    }
    n
}

#[test]
fn markers_skeleton_and_edits_show_up_where_expected() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter: skipping offscreen render test");
        return;
    };
    let mut renderer =
        Renderer::new(&gpu.device, RenderConfig { color_format: FORMAT, depth_format: Some(DEPTH), msaa_samples: 1 });
    let mut take = synthetic_take();
    renderer.load_take(&gpu.device, &take);
    renderer.set_skeleton_pairs(&gpu.device, &[(0, 1), (1, 2)]);
    renderer.set_marker_states(&gpu.queue, &[MarkerState::Normal, MarkerState::Selected, MarkerState::Normal]);
    renderer.set_grid(&gpu.device, &gpu.queue, &grid_segments(0.0, 2.0, 0.5, 0.3));

    let mut camera = Camera { target: Vec3::new(0.0, 1.0, 0.0), yaw: 0.0, pitch: 0.0, dist: 2.5, ..Camera::default() };
    let visual = VisualSettings::default();
    let viewport = [W as f32, H as f32];
    let vp = camera.view_proj(1.0, CoordinateSystem::YUp);
    let params = |frame| FrameParams {
        view_proj: vp,
        viewport,
        frame,
        selected: None,
        visual: &visual,
        show_skeleton: true,
        show_grid: true,
    };

    // frame 0: every marker lights up at its projected position
    let px0 = render(&gpu, &mut renderer, &params(0));
    for m in 0..3 {
        let p = take.position(0, m).unwrap();
        let (s, _) = project(vp, viewport, [p[0] as f32, p[1] as f32, p[2] as f32]).unwrap();
        assert!(lit(&px0, s[0], s[1], 3) >= 10, "marker {m} not visible at {s:?}");
    }
    // skeleton: the midpoint between markers 0 and 1 is on a line
    let (m0, m1) = (take.position(0, 0).unwrap(), take.position(0, 1).unwrap());
    let mid = [((m0[0] + m1[0]) / 2.0) as f32, 1.0, 0.0];
    let (sm, _) = project(vp, viewport, mid).unwrap();
    assert!(lit(&px0, sm[0], sm[1], 2) >= 3, "skeleton segment missing");
    // the background stays dark far from everything
    assert_eq!(lit(&px0, 10.0, 10.0, 2), 0);

    // frame 9: markers moved right by 0.45 m → frame advance is a uniform write, pixels differ
    let px9 = render(&gpu, &mut renderer, &params(9));
    assert_ne!(px0, px9);
    let p = take.position(9, 2).unwrap();
    let (s9, _) = project(vp, viewport, [p[0] as f32, p[1] as f32, p[2] as f32]).unwrap();
    assert!(lit(&px9, s9[0], s9[1], 3) >= 10);

    // edit: delete marker 2 in frame 9, patch only that frame (R2) → it disappears
    take.clear_range(2, 9, 9);
    renderer.update_frames(&gpu.queue, &take, DirtyRange::single(9));
    let px9b = render(&gpu, &mut renderer, &params(9));
    assert!(lit(&px9b, s9[0], s9[1], 1) == 0, "deleted marker still drawn");
    // frame 0 untouched
    assert_eq!(render(&gpu, &mut renderer, &params(0)), px0);

    // Z-up: the same data seen with the coordinate model applied still renders markers
    camera.fit_bounds([-0.5, 0.5, -0.5], [0.9, 1.5, 0.5]);
    let vp_z = camera.view_proj(1.0, CoordinateSystem::ZUp);
    let pz = render(&gpu, &mut renderer, &FrameParams { view_proj: vp_z, ..params(0) });
    let p = take.position(0, 0).unwrap();
    let (sz, _) = project(vp_z, viewport, [p[0] as f32, p[1] as f32, p[2] as f32]).unwrap();
    assert!(lit(&pz, sz[0], sz[1], 3) >= 10);
}
