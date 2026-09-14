//! MStudio Phase 0 spike: eframe + wgpu marker viewer.
//!
//! Proves the five performance rules from docs/CROSS_PLATFORM_PLAN.md §3 on
//! real hardware:
//!   R1 whole take uploaded once, frame advance = one uniform write
//!   R3 playback driven by the frame clock, not timers
//!   R5 one draw call per layer (markers instanced, skeleton from index buffer)
//!
//! Usage:
//!   mstudio-spike [path.trc]                     # default: tests/test.trc
//!   mstudio-spike --stress 300 50000             # synthetic take
//!   mstudio-spike --bench 10                     # play 10 s, print stats JSON, exit
//!
//! Keys: Space play/pause, Esc stop, ←/→ step, L labels, F fit view.
//! Mouse: LMB orbit, RMB/MMB pan, wheel zoom, click marker to select.
#![allow(deprecated)] // glam camera helpers; fine for the spike

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

const MSAA: u32 = 4;
const CULL_SENTINEL: f32 = 1.0e30;
const TRAJ_WINDOW: usize = 10;

// ============================================================================
// Take: the data model (plan §5, minimal)
// ============================================================================

struct Take {
    markers: Vec<String>,
    fps: f32,
    n_frames: usize,
    /// frame-major, one vec4 per marker; NaN replaced by CULL_SENTINEL
    positions: Vec<[f32; 4]>,
    pairs: Vec<u32>,
    bbox_min: Vec3,
    bbox_max: Vec3,
}

impl Take {
    fn n_markers(&self) -> usize {
        self.markers.len()
    }

    fn position(&self, frame: usize, marker: usize) -> Option<Vec3> {
        let p = self.positions[frame * self.n_markers() + marker];
        (p[0] < CULL_SENTINEL).then(|| Vec3::new(p[0], p[1], p[2]))
    }

    fn load_trc(path: &str) -> anyhow::Result<Take> {
        let text = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = text.lines().collect();
        anyhow::ensure!(lines.len() > 6, "TRC too short");
        let fps: f32 = lines[2].split('\t').next().unwrap_or("30").trim().parse().unwrap_or(30.0);
        let markers: Vec<String> = lines[3]
            .split('\t')
            .skip(2)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        let n_markers = markers.len();
        let mut positions = Vec::new();
        let mut n_frames = 0;
        for line in &lines[6..] {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            for m in 0..n_markers {
                let mut v = [CULL_SENTINEL, CULL_SENTINEL, CULL_SENTINEL, 0.0];
                let base = 2 + m * 3;
                if base + 2 < fields.len() {
                    let xyz: Vec<Option<f32>> = (0..3).map(|k| fields[base + k].trim().parse::<f32>().ok()).collect();
                    if let (Some(x), Some(y), Some(z)) = (xyz[0], xyz[1], xyz[2]) {
                        if x.is_finite() && y.is_finite() && z.is_finite() {
                            v = [x, y, z, 0.0];
                        }
                    }
                }
                positions.push(v);
            }
            n_frames += 1;
        }
        let pairs = resolve_pairs(&markers, HUMAN_PAIRS);
        let (bbox_min, bbox_max) = bbox(&positions);
        Ok(Take { markers, fps, n_frames, positions, pairs, bbox_min, bbox_max })
    }

    /// Deterministic synthetic take: markers on a chain, smooth motion.
    fn synthetic(n_markers: usize, n_frames: usize) -> Take {
        let fps = 120.0;
        let mut positions = Vec::with_capacity(n_markers * n_frames);
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 53) as f32
        };
        let params: Vec<[f32; 6]> = (0..n_markers)
            .map(|_| [next(), next(), next(), next(), next(), next()])
            .collect();
        for f in 0..n_frames {
            let t = f as f32 / fps;
            for (m, p) in params.iter().enumerate() {
                let base = Vec3::new((m % 20) as f32 * 0.15 - 1.5, (m / 20) as f32 * 0.12, 0.0);
                let x = base.x + 0.15 * (t * (0.5 + p[0]) + p[3] * 6.28).sin();
                let y = base.y + 0.10 * (t * (0.5 + p[1]) + p[4] * 6.28).sin();
                let z = base.z + 0.15 * (t * (0.5 + p[2]) + p[5] * 6.28).cos();
                positions.push([x, y, z, 0.0]);
            }
        }
        let pairs: Vec<u32> = (0..n_markers.saturating_sub(1)).flat_map(|i| [i as u32, i as u32 + 1]).collect();
        let (bbox_min, bbox_max) = bbox(&positions);
        Take {
            markers: (0..n_markers).map(|i| format!("M{i}")).collect(),
            fps,
            n_frames,
            positions,
            pairs,
            bbox_min,
            bbox_max,
        }
    }
}

fn bbox(positions: &[[f32; 4]]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for p in positions.iter().filter(|p| p[0] < CULL_SENTINEL) {
        let v = Vec3::new(p[0], p[1], p[2]);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo.x > hi.x {
        (Vec3::splat(-1.0), Vec3::splat(1.0))
    } else {
        (lo, hi)
    }
}

/// Pose2Sim / HALPE / COCO-style names; pairs whose markers are absent are skipped
/// (same rule as TRCViewer.update_skeleton_pairs).
const HUMAN_PAIRS: &[(&str, &str)] = &[
    ("Hip", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RBigToe"), ("RAnkle", "RSmallToe"), ("RAnkle", "RHeel"),
    ("Hip", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LBigToe"), ("LAnkle", "LSmallToe"), ("LAnkle", "LHeel"),
    ("Hip", "Neck"), ("Neck", "Nose"), ("Nose", "REye"), ("Nose", "LEye"), ("Neck", "Head"),
    ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"), ("RWrist", "RThumb"), ("RWrist", "RIndex"), ("RWrist", "RPinky"),
    ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("LWrist", "LThumb"), ("LWrist", "LIndex"), ("LWrist", "LPinky"),
];

fn resolve_pairs(markers: &[String], table: &[(&str, &str)]) -> Vec<u32> {
    let idx = |name: &str| markers.iter().position(|m| m == name);
    table
        .iter()
        .filter_map(|(a, b)| Some([idx(a)? as u32, idx(b)? as u32]))
        .flatten()
        .collect()
}

// ============================================================================
// Camera
// ============================================================================

struct Camera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Camera {
    fn fit(take: &Take) -> Camera {
        let center = (take.bbox_min + take.bbox_max) * 0.5;
        let extent = (take.bbox_max - take.bbox_min).length().max(0.5);
        Camera { target: center, yaw: 0.6, pitch: 0.35, dist: extent * 1.4 }
    }

    fn eye(&self) -> Vec3 {
        self.target + self.dist * Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos())
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.01, 1000.0);
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        proj * view
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let s = self.dist * 0.0015;
        self.target += (-right * dx + up * dy) * s;
    }

    fn zoom(&mut self, scroll: f32) {
        self.dist = (self.dist * (-scroll * 0.002).exp()).clamp(0.05, 500.0);
    }
}

// ============================================================================
// GPU resources + egui paint callback
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    frame: u32,
    n_markers: u32,
    point_size: f32,
    selected: i32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StaticVertex {
    pos: [f32; 3],
    color: [f32; 3],
}

struct GpuState {
    uniform_buf: wgpu::Buffer,
    states_buf: wgpu::Buffer,
    traj_buf: wgpu::Buffer,
    traj_count: u32,
    grid_buf: wgpu::Buffer,
    grid_count: u32,
    bind_group: wgpu::BindGroup,
    pipe_markers: wgpu::RenderPipeline,
    pipe_skeleton: wgpu::RenderPipeline,
    pipe_lines: wgpu::RenderPipeline,
    pipe_strip: wgpu::RenderPipeline,
    n_markers: u32,
    n_pair_verts: u32,
}

impl GpuState {
    fn new(rs: &egui_wgpu::RenderState, take: &Take) -> GpuState {
        let device = &rs.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spike"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // R1: the whole take, once.
        let positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("positions"),
            contents: bytemuck::cast_slice(&take.positions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let states_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("states"),
            size: (take.n_markers().max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pairs: &[u32] = if take.pairs.is_empty() { &[0, 0] } else { &take.pairs };
        let pairs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pairs"),
            contents: bytemuck::cast_slice(pairs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let traj_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trajectory"),
            size: (std::mem::size_of::<StaticVertex>() * (2 * TRAJ_WINDOW + 1)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid = build_grid(take.bbox_min.y);
        let grid_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                storage(3),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: states_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pairs_buf.as_entire_binding() },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let depth = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        };
        let static_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<StaticVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let make = |label: &str, vs: &str, fs: &str, topology, buffers: &[Option<wgpu::VertexBufferLayout>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: Some(depth.clone()),
                multisample: wgpu::MultisampleState { count: MSAA, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: rs.target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        GpuState {
            pipe_markers: make("markers", "vs_marker", "fs_marker", wgpu::PrimitiveTopology::TriangleList, &[]),
            pipe_skeleton: make("skeleton", "vs_skeleton", "fs_line", wgpu::PrimitiveTopology::LineList, &[]),
            pipe_lines: make("lines", "vs_static", "fs_line", wgpu::PrimitiveTopology::LineList, &[Some(static_layout.clone())]),
            pipe_strip: make("strip", "vs_static", "fs_line", wgpu::PrimitiveTopology::LineStrip, &[Some(static_layout)]),
            uniform_buf,
            states_buf,
            traj_buf,
            traj_count: 0,
            grid_buf,
            grid_count: grid.len() as u32,
            bind_group,
            n_markers: take.n_markers() as u32,
            n_pair_verts: take.pairs.len() as u32,
        }
    }
}

fn build_grid(ground_y: f32) -> Vec<StaticVertex> {
    let mut v = Vec::new();
    let grey = [0.22, 0.22, 0.24];
    let n = 10;
    for i in -n..=n {
        let a = i as f32;
        v.push(StaticVertex { pos: [a, ground_y, -n as f32], color: grey });
        v.push(StaticVertex { pos: [a, ground_y, n as f32], color: grey });
        v.push(StaticVertex { pos: [-n as f32, ground_y, a], color: grey });
        v.push(StaticVertex { pos: [n as f32, ground_y, a], color: grey });
    }
    let o = [0.0, ground_y, 0.0];
    for (axis, color) in [([1.0, 0.0, 0.0], [0.9, 0.2, 0.2]), ([0.0, 1.0, 0.0], [0.2, 0.9, 0.2]), ([0.0, 0.0, 1.0], [0.3, 0.4, 1.0])] {
        v.push(StaticVertex { pos: o, color });
        v.push(StaticVertex { pos: [o[0] + axis[0] * 0.5, o[1] + axis[1] * 0.5, o[2] + axis[2] * 0.5], color });
    }
    v
}

/// Everything the render thread needs for one frame. Tiny by design (R1).
struct SceneCallback {
    uniforms: Uniforms,
    states: Vec<u32>,
    trajectory: Vec<StaticVertex>,
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let g: &mut GpuState = resources.get_mut().expect("GpuState");
        queue.write_buffer(&g.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
        queue.write_buffer(&g.states_buf, 0, bytemuck::cast_slice(&self.states));
        g.traj_count = self.trajectory.len() as u32;
        if g.traj_count > 0 {
            queue.write_buffer(&g.traj_buf, 0, bytemuck::cast_slice(&self.trajectory));
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let g: &GpuState = resources.get().expect("GpuState");
        pass.set_bind_group(0, &g.bind_group, &[]);

        pass.set_pipeline(&g.pipe_lines);
        pass.set_vertex_buffer(0, g.grid_buf.slice(..));
        pass.draw(0..g.grid_count, 0..1);

        if g.n_pair_verts > 0 {
            pass.set_pipeline(&g.pipe_skeleton);
            pass.draw(0..g.n_pair_verts, 0..1);
        }
        if g.traj_count > 1 {
            pass.set_pipeline(&g.pipe_strip);
            pass.set_vertex_buffer(0, g.traj_buf.slice(..));
            pass.draw(0..g.traj_count, 0..1);
        }
        pass.set_pipeline(&g.pipe_markers);
        pass.draw(0..6, 0..g.n_markers);
    }
}

// ============================================================================
// Playback (R3) + stats
// ============================================================================

struct Playback {
    playing: bool,
    frame: usize,
    anchor: Option<(Instant, f64)>,
    looping: bool,
}

impl Playback {
    fn toggle(&mut self, now: Instant) {
        self.playing = !self.playing;
        self.anchor = self.playing.then_some((now, self.frame as f64));
    }

    fn stop(&mut self) {
        self.playing = false;
        self.anchor = None;
        self.frame = 0;
    }

    fn step(&mut self, delta: isize, n: usize) {
        self.frame = (self.frame as isize + delta).rem_euclid(n as isize) as usize;
        self.anchor = self.playing.then_some((Instant::now(), self.frame as f64));
    }

    /// The frame that should be on screen *now*. No accumulated timer drift.
    fn tick(&mut self, now: Instant, fps: f32, n: usize) -> usize {
        if let Some((t0, f0)) = self.anchor {
            let f = f0 + now.duration_since(t0).as_secs_f64() * fps as f64;
            if self.looping {
                self.frame = (f as usize) % n.max(1);
            } else if f as usize >= n {
                self.frame = n - 1;
                self.playing = false;
                self.anchor = None;
            } else {
                self.frame = f as usize;
            }
        }
        self.frame
    }
}

#[derive(Default)]
struct Stats {
    cpu_us: VecDeque<f32>,
    scene_us: VecDeque<f32>,
    frame_times: VecDeque<Instant>,
}

impl Stats {
    fn push(&mut self, cpu: Duration, scene: Duration, now: Instant) {
        for (q, v) in [(&mut self.cpu_us, cpu), (&mut self.scene_us, scene)] {
            q.push_back(v.as_secs_f32() * 1e6);
            if q.len() > 600 {
                q.pop_front();
            }
        }
        self.frame_times.push_back(now);
        while self.frame_times.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) {
            self.frame_times.pop_front();
        }
    }

    fn fps(&self) -> f32 {
        self.frame_times.len() as f32
    }

    fn summary(q: &VecDeque<f32>) -> (f32, f32, f32) {
        if q.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut v: Vec<f32> = q.iter().copied().collect();
        v.sort_by(|a, b| a.total_cmp(b));
        let avg = v.iter().sum::<f32>() / v.len() as f32;
        (avg, v[v.len() * 95 / 100], *v.last().unwrap())
    }
}

// ============================================================================
// App
// ============================================================================

struct App {
    take: Take,
    camera: Camera,
    playback: Playback,
    stats: Stats,
    selected: Option<usize>,
    show_labels: bool,
    point_size: f32,
    bench: Option<(Instant, Duration)>,
    adapter: String,
    screenshot: Option<String>,
    shot_requested: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, take: Take, bench_secs: Option<f32>, screenshot: Option<String>) -> App {
        let rs = cc.wgpu_render_state.as_ref().expect("wgpu renderer required");
        let gpu = GpuState::new(rs, &take);
        rs.renderer.write().callback_resources.insert(gpu);
        let info = rs.adapter.get_info();
        let camera = Camera::fit(&take);
        let mut playback = Playback { playing: false, frame: 0, anchor: None, looping: true };
        let bench = bench_secs.map(|s| {
            playback.toggle(Instant::now());
            (Instant::now(), Duration::from_secs_f32(s))
        });
        // In bench mode pre-select a marker so the screenshot shows selection + trajectory.
        let selected = bench.map(|_| 2.min(take.n_markers().saturating_sub(1)));
        App {
            take,
            camera,
            playback,
            stats: Stats::default(),
            selected,
            show_labels: true,
            point_size: 9.0,
            bench,
            adapter: format!("{} ({:?})", info.name, info.backend),
            screenshot,
            shot_requested: false,
        }
    }

    fn project(&self, vp: Mat4, rect: egui::Rect, p: Vec3) -> Option<(egui::Pos2, f32)> {
        let c = vp * Vec4::new(p.x, p.y, p.z, 1.0);
        if c.w <= 0.0 {
            return None;
        }
        let ndc = c.truncate() / c.w;
        if ndc.z < 0.0 || ndc.z > 1.0 {
            return None;
        }
        Some((
            egui::pos2(rect.min.x + (ndc.x + 1.0) * 0.5 * rect.width(), rect.min.y + (1.0 - ndc.y) * 0.5 * rect.height()),
            ndc.z,
        ))
    }

    fn pick(&self, vp: Mat4, rect: egui::Rect, pointer: egui::Pos2, frame: usize) -> Option<usize> {
        let radius = self.point_size * 0.5 + 4.0;
        (0..self.take.n_markers())
            .filter_map(|m| {
                let p = self.take.position(frame, m)?;
                let (s, z) = self.project(vp, rect, p)?;
                let d = s.distance(pointer);
                (d <= radius).then_some((m, z, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
            .map(|(m, _, _)| m)
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(img) = ctx.input(|i| i.events.iter().find_map(|e| match e {
            egui::Event::Screenshot { image, .. } => Some(image.clone()),
            _ => None,
        })) {
            if let Some(path) = self.screenshot.take() {
                save_png(&img, &path);
            }
        }
        let t_start = Instant::now();
        let now = t_start;
        let n = self.take.n_frames;

        // ---- keyboard
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.playback.toggle(now);
            }
            if i.key_pressed(egui::Key::Escape) {
                self.playback.stop();
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.playback.step(1, n);
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.playback.step(-1, n);
            }
            if i.key_pressed(egui::Key::L) {
                self.show_labels = !self.show_labels;
            }
            if i.key_pressed(egui::Key::F) {
                self.camera = Camera::fit(&self.take);
            }
        });
        let frame = self.playback.tick(now, self.take.fps, n);

        // ---- top bar
        egui::Panel::top("bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button(if self.playback.playing { "⏸" } else { "▶" }).clicked() {
                    self.playback.toggle(now);
                }
                if ui.button("⏹").clicked() {
                    self.playback.stop();
                }
                ui.checkbox(&mut self.playback.looping, "loop");
                ui.checkbox(&mut self.show_labels, "labels");
                ui.add(egui::Slider::new(&mut self.point_size, 3.0..=24.0).text("px"));
                ui.separator();
                ui.label(format!("frame {frame}/{n}  @{:.0} fps  markers {}", self.take.fps, self.take.n_markers()));
                ui.separator();
                let (cpu_avg, cpu_p95, cpu_max) = Stats::summary(&self.stats.cpu_us);
                let (sc_avg, _, sc_max) = Stats::summary(&self.stats.scene_us);
                ui.monospace(format!(
                    "{:>5.1} fps | update {cpu_avg:>6.0} µs avg {cpu_p95:>6.0} p95 {cpu_max:>7.0} max | scene {sc_avg:>5.0} avg {sc_max:>6.0} max",
                    self.stats.fps()
                ));
                if let Some(sel) = self.selected {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(255, 217, 38), &self.take.markers[sel]);
                }
            });
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small(&self.adapter);
                ui.separator();
                ui.small("Space play · Esc stop · ←/→ step · L labels · F fit · LMB orbit · RMB/MMB pan · wheel zoom · click = select");
            });
        });

        // ---- 3D viewport
        let scene_time;
        {
                let t_scene = Instant::now();
                let rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                let ppp = ctx.pixels_per_point();

                // input → camera
                let d = response.drag_delta();
                if response.dragged_by(egui::PointerButton::Primary) {
                    self.camera.orbit(d.x, d.y);
                } else if response.dragged_by(egui::PointerButton::Secondary) || response.dragged_by(egui::PointerButton::Middle) {
                    self.camera.pan(d.x, d.y);
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll);
                    }
                }

                let aspect = rect.width().max(1.0) / rect.height().max(1.0);
                let vp = self.camera.view_proj(aspect);

                if response.clicked_by(egui::PointerButton::Primary) {
                    if let Some(pos) = response.interact_pointer_pos() {
                        self.selected = self.pick(vp, rect, pos, frame);
                    }
                }

                // per-marker state (R5: one small buffer, no per-marker draws)
                let mut states = vec![0u32; self.take.n_markers()];
                if let Some(s) = self.selected {
                    states[s] = 1;
                }

                // trajectory of the selected marker (CPU, tiny)
                let mut trajectory = Vec::new();
                if let Some(s) = self.selected {
                    let lo = frame.saturating_sub(TRAJ_WINDOW);
                    let hi = (frame + TRAJ_WINDOW).min(n - 1);
                    for f in lo..=hi {
                        if let Some(p) = self.take.position(f, s) {
                            let t = if f <= frame { 1.0 } else { 0.45 };
                            trajectory.push(StaticVertex { pos: p.to_array(), color: [1.0 * t, 0.85 * t, 0.15 * t] });
                        }
                    }
                }

                let uniforms = Uniforms {
                    view_proj: vp.to_cols_array_2d(),
                    viewport: [rect.width() * ppp, rect.height() * ppp],
                    frame: frame as u32,
                    n_markers: self.take.n_markers() as u32,
                    point_size: self.point_size * ppp,
                    selected: self.selected.map_or(-1, |s| s as i32),
                    _pad: [0.0; 2],
                };
                ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, SceneCallback { uniforms, states, trajectory }));

                // labels: projected on the CPU, drawn by egui in one text batch
                if self.show_labels {
                    let painter = ui.painter_at(rect);
                    let font = egui::FontId::proportional(11.0);
                    for m in 0..self.take.n_markers() {
                        let Some(p) = self.take.position(frame, m) else { continue };
                        let Some((s, _)) = self.project(vp, rect, p) else { continue };
                        if !rect.contains(s) {
                            continue;
                        }
                        let color = if self.selected == Some(m) { egui::Color32::from_rgb(255, 217, 38) } else { egui::Color32::from_gray(210) };
                        painter.text(s + egui::vec2(6.0, -4.0), egui::Align2::LEFT_BOTTOM, &self.take.markers[m], font.clone(), color);
                    }
                }
                scene_time = t_scene.elapsed();
        }

        // ---- bookkeeping
        self.stats.push(t_start.elapsed(), scene_time, now);
        if self.playback.playing {
            ctx.request_repaint();
        }
        if let Some((t0, dur)) = self.bench {
            if !self.shot_requested && self.screenshot.is_some() && now.duration_since(t0) >= Duration::from_millis(1500) {
                self.shot_requested = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            if now.duration_since(t0) >= dur {
                let (cpu_avg, cpu_p95, cpu_max) = Stats::summary(&self.stats.cpu_us);
                let (sc_avg, sc_p95, sc_max) = Stats::summary(&self.stats.scene_us);
                println!(
                    "{{\"adapter\":\"{}\",\"markers\":{},\"frames\":{},\"fps\":{:.1},\"update_us\":{{\"avg\":{:.1},\"p95\":{:.1},\"max\":{:.1}}},\"scene_us\":{{\"avg\":{:.1},\"p95\":{:.1},\"max\":{:.1}}},\"pixels_per_point\":{}}}",
                    self.adapter, self.take.n_markers(), self.take.n_frames, self.stats.fps(), cpu_avg, cpu_p95, cpu_max, sc_avg, sc_p95, sc_max, ctx.pixels_per_point()
                );
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                self.bench = None;
            }
        }
    }
}

fn save_png(img: &egui::ColorImage, path: &str) {
    let [w, h] = img.size;
    let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
    match image::RgbaImage::from_raw(w as u32, h as u32, bytes).map(|i| i.save(path)) {
        Some(Ok(())) => eprintln!("screenshot saved: {path}"),
        other => eprintln!("screenshot failed: {other:?}"),
    }
}

// ============================================================================
// main
// ============================================================================

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<String> = None;
    let mut stress: Option<(usize, usize)> = None;
    let mut bench: Option<f32> = None;
    let mut screenshot: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--stress" => {
                stress = Some((args[i + 1].parse().expect("markers"), args[i + 2].parse().expect("frames")));
                i += 3;
            }
            "--bench" => {
                bench = Some(args[i + 1].parse().expect("seconds"));
                i += 2;
            }
            "--screenshot" => {
                screenshot = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                path = Some(other.to_string());
                i += 1;
            }
        }
    }

    let take = if let Some((m, f)) = stress {
        Take::synthetic(m, f)
    } else {
        let path = path.unwrap_or_else(|| {
            let local = "tests/test.trc";
            if std::path::Path::new(local).exists() {
                local.to_string()
            } else {
                format!("{}/../../tests/test.trc", env!("CARGO_MANIFEST_DIR"))
            }
        });
        Take::load_trc(&path).unwrap_or_else(|e| panic!("failed to load {path}: {e}"))
    };
    eprintln!(
        "take: {} markers × {} frames @ {} fps, {} skeleton pairs, {:.1} MB on GPU",
        take.n_markers(),
        take.n_frames,
        take.fps,
        take.pairs.len() / 2,
        (take.positions.len() * 16) as f64 / 1e6
    );

    // Large takes need more than the default 128 MiB storage binding.
    let mut wgpu_setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| wgpu::DeviceDescriptor {
        label: Some("mstudio-spike"),
        required_limits: adapter.limits(),
        ..Default::default()
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]).with_title("MStudio spike — wgpu"),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 24,
        multisampling: MSAA as u16,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            surface: egui_wgpu::SurfaceConfig { present_mode: wgpu::PresentMode::AutoVsync, desired_maximum_frame_latency: Some(2) },
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native("mstudio-spike", options, Box::new(move |cc| Ok(Box::new(App::new(cc, take, bench, screenshot)))))
}
