//! MStudio viewport renderer on wgpu.
//!
//! Owns no window and no UI: the host (an egui paint callback, or a headless
//! test) supplies the device, the render pass and the camera; the renderer
//! keeps the take resident on the GPU and issues one draw call per layer
//! (plan rules R1, R2, R5). Text is not drawn here — `camera::project` and
//! `AnalysisOverlay::labels` give the host the screen anchors.

pub mod analysis_overlay;
pub mod camera;
pub mod gpu_take;
pub mod grid;
pub mod picking;
pub mod segments;
pub mod trajectories;

pub use analysis_overlay::{analysis_overlay, AnalysisOverlay};
pub use camera::{coordinate_model, project, Camera};
pub use gpu_take::{pack_outliers, GpuTake, CULL_SENTINEL};
pub use grid::grid_segments;
pub use picking::{distance_to_segment_px, pick_marker};
pub use segments::Segment;
pub use trajectories::trajectory_segments;

use glam::Mat4;
use mstudio_core::{DirtyRange, Take, VisualSettings};
use wgpu::util::DeviceExt;

/// Per-marker display state (matches the shader constants).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarkerState {
    #[default]
    Normal = 0,
    Selected = 1,
    Pattern = 2,
    Analysis = 3,
}

/// What the host's render target looks like; pipelines are built to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderConfig {
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub msaa_samples: u32,
}

/// Everything that changes per frame.
#[derive(Debug, Clone)]
pub struct FrameParams<'a> {
    pub view_proj: Mat4,
    /// Physical pixels.
    pub viewport: [f32; 2],
    pub frame: usize,
    pub selected: Option<usize>,
    pub visual: &'a VisualSettings,
    pub show_skeleton: bool,
    pub show_grid: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    frame: u32,
    n_markers: u32,
    marker_size: f32,
    marker_opacity: f32,
    line_width: f32,
    skeleton_opacity: f32,
    color_normal: [f32; 4],
    color_selected: [f32; 4],
    color_pattern: [f32; 4],
    color_analysis: [f32; 4],
    skel_normal: [f32; 4],
    skel_outlier: [f32; 4],
    selected: i32,
    n_pairs: u32,
    flags: u32,
    _pad: u32,
}

fn rgba(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 1.0]
}

pub struct Renderer {
    config: RenderConfig,
    scene_layout: wgpu::BindGroupLayout,
    segment_layout: wgpu::BindGroupLayout,
    pipe_markers: wgpu::RenderPipeline,
    pipe_skeleton: wgpu::RenderPipeline,
    pipe_segments: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    take: Option<GpuTake>,
    states_buf: wgpu::Buffer,
    pairs_buf: wgpu::Buffer,
    n_pairs: u32,
    outliers_buf: wgpu::Buffer,
    scene_bind_group: Option<wgpu::BindGroup>,
    grid: segments::SegmentBuffer,
    overlay: segments::SegmentBuffer,
    n_markers: u32,
    show_skeleton: bool,
    show_grid: bool,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, config: RenderConfig) -> Renderer {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mstudio viewport"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                storage(4),
            ],
        });
        let segment_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("segments"),
            entries: &[storage(0)],
        });
        let scene_only = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&scene_layout)],
            immediate_size: 0,
        });
        let with_segments = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene+segments"),
            bind_group_layouts: &[Some(&scene_layout), Some(&segment_layout)],
            immediate_size: 0,
        });

        let depth = config.depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let make = |label: &str, layout: &wgpu::PipelineLayout, vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: depth.clone(),
                multisample: wgpu::MultisampleState {
                    count: config.msaa_samples.max(1),
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let empty = |label: &str| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[0u32, 0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };

        Renderer {
            pipe_markers: make("markers", &scene_only, "vs_marker", "fs_marker"),
            pipe_skeleton: make("skeleton", &scene_only, "vs_skeleton", "fs_line"),
            pipe_segments: make("segments", &with_segments, "vs_segments", "fs_line"),
            grid: segments::SegmentBuffer::new(device, &segment_layout, "grid"),
            overlay: segments::SegmentBuffer::new(device, &segment_layout, "overlay"),
            config,
            scene_layout,
            segment_layout,
            uniform_buf,
            take: None,
            states_buf: empty("states"),
            pairs_buf: empty("pairs"),
            n_pairs: 0,
            outliers_buf: empty("outliers"),
            scene_bind_group: None,
            n_markers: 0,
            show_skeleton: true,
            show_grid: true,
        }
    }

    pub fn config(&self) -> RenderConfig {
        self.config
    }

    pub fn has_take(&self) -> bool {
        self.take.is_some()
    }

    /// Upload a take (all frames) and reset per-marker state, pairs and outliers.
    pub fn load_take(&mut self, device: &wgpu::Device, take: &Take) {
        self.take = Some(GpuTake::new(device, take));
        self.n_markers = take.n_markers() as u32;
        let states = vec![0u32; take.n_markers().max(2)];
        self.states_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("states"),
            contents: bytemuck::cast_slice(&states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.n_pairs = 0;
        self.pairs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pairs"),
            contents: bytemuck::cast_slice(&[0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let words = (take.n_frames() * take.n_markers()).div_ceil(32).max(1);
        self.outliers_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("outliers"),
            contents: bytemuck::cast_slice(&vec![0u32; words]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.scene_bind_group = None;
    }

    pub fn clear_take(&mut self) {
        self.take = None;
        self.n_markers = 0;
        self.n_pairs = 0;
        self.scene_bind_group = None;
    }

    /// Patch the frames in `range` after an edit (rule R2).
    pub fn update_frames(&self, queue: &wgpu::Queue, take: &Take, range: DirtyRange) {
        if let Some(g) = &self.take {
            g.write_range(queue, take, range);
        }
    }

    pub fn set_skeleton_pairs(&mut self, device: &wgpu::Device, pairs: &[(usize, usize)]) {
        let flat: Vec<u32> =
            if pairs.is_empty() { vec![0, 0] } else { pairs.iter().flat_map(|&(a, b)| [a as u32, b as u32]).collect() };
        self.pairs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pairs"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.n_pairs = pairs.len() as u32;
        self.scene_bind_group = None;
    }

    /// `[marker, frame]` flags; the whole map is uploaded (it changes only
    /// when detection reruns).
    pub fn set_outliers(&mut self, device: &wgpu::Device, map: &ndarray::Array2<bool>) {
        let words = pack_outliers(map);
        self.outliers_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("outliers"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.scene_bind_group = None;
    }

    /// One state per marker (extra entries ignored, missing ones = Normal).
    pub fn set_marker_states(&self, queue: &wgpu::Queue, states: &[MarkerState]) {
        if self.n_markers == 0 {
            return;
        }
        let mut words = vec![0u32; self.n_markers as usize];
        for (w, s) in words.iter_mut().zip(states) {
            *w = *s as u32;
        }
        queue.write_buffer(&self.states_buf, 0, bytemuck::cast_slice(&words));
    }

    pub fn set_grid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, segments: &[Segment]) {
        self.grid.set(device, queue, &self.segment_layout, segments);
    }

    /// Trajectories + analysis geometry for this frame (rebuilt by the host
    /// only when something in it changes).
    pub fn set_overlay(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, segments: &[Segment]) {
        self.overlay.set(device, queue, &self.segment_layout, segments);
    }

    /// Write the uniforms for this frame and (re)build the scene bind group
    /// if any buffer was replaced.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, params: &FrameParams<'_>) {
        let v = params.visual;
        let u = Uniforms {
            view_proj: params.view_proj.to_cols_array_2d(),
            viewport: params.viewport,
            frame: params.frame as u32,
            n_markers: self.n_markers,
            marker_size: v.marker.size * 2.0, // Python sizes are radii-ish; ×2 ≈ the same on-screen diameter
            marker_opacity: v.marker.opacity,
            line_width: v.skeleton.line_width,
            skeleton_opacity: v.skeleton.opacity,
            color_normal: rgba(v.marker.color_normal),
            color_selected: rgba(v.marker.color_selected),
            color_pattern: rgba(v.marker.color_pattern),
            color_analysis: [0.4, 0.85, 1.0, 1.0],
            skel_normal: rgba(v.skeleton.color_normal),
            skel_outlier: rgba(v.skeleton.color_outlier),
            selected: params.selected.map_or(-1, |s| s as i32),
            n_pairs: self.n_pairs,
            flags: 0,
            _pad: 0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        self.show_skeleton = params.show_skeleton;
        self.show_grid = params.show_grid;
        if self.scene_bind_group.is_none() {
            if let Some(take) = &self.take {
                self.scene_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("scene"),
                    layout: &self.scene_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: take.buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.states_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.pairs_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.outliers_buf.as_entire_binding() },
                    ],
                }));
            }
        }
    }

    /// Issue the draw calls into a pass whose attachments match `config`.
    /// The host sets viewport/scissor. Nothing is drawn before `prepare`.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(bg) = &self.scene_bind_group else { return };
        pass.set_bind_group(0, bg, &[]);
        pass.set_pipeline(&self.pipe_segments);
        if self.show_grid && self.grid.count > 0 {
            pass.set_bind_group(1, &self.grid.bind_group, &[]);
            pass.draw(0..6, 0..self.grid.count);
        }
        if self.show_skeleton && self.n_pairs > 0 {
            pass.set_pipeline(&self.pipe_skeleton);
            pass.draw(0..6, 0..self.n_pairs);
        }
        if self.overlay.count > 0 {
            pass.set_pipeline(&self.pipe_segments);
            pass.set_bind_group(1, &self.overlay.bind_group, &[]);
            pass.draw(0..6, 0..self.overlay.count);
        }
        if self.n_markers > 0 {
            pass.set_pipeline(&self.pipe_markers);
            pass.draw(0..6, 0..self.n_markers);
        }
    }
}
