//! Thick line segments drawn as instanced quads: grid and axes, trajectories,
//! analysis overlays. Small CPU-built lists uploaded to a storage buffer.

/// One segment: endpoints in data space (or view-up space for the grid),
/// width in pixels, RGBA colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Segment {
    pub a: [f32; 4], // xyz + width px
    pub b: [f32; 4],
    pub color: [f32; 4],
}

impl Segment {
    pub fn new(a: [f32; 3], b: [f32; 3], width_px: f32, color: [f32; 4]) -> Segment {
        Segment { a: [a[0], a[1], a[2], width_px], b: [b[0], b[1], b[2], 0.0], color }
    }
}

/// A growable storage buffer of segments with its own bind group.
pub struct SegmentBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    pub count: u32,
    pub bind_group: wgpu::BindGroup,
    label: &'static str,
}

impl SegmentBuffer {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, label: &'static str) -> Self {
        Self::with_capacity(device, layout, label, 64)
    }

    fn with_capacity(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: &'static str,
        capacity: usize,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (capacity * std::mem::size_of::<Segment>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        SegmentBuffer { buffer, capacity, count: 0, bind_group, label }
    }

    /// Replace the contents, growing the buffer (and bind group) if needed.
    pub fn set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        segments: &[Segment],
    ) {
        if segments.len() > self.capacity {
            *self = Self::with_capacity(device, layout, self.label, segments.len().next_power_of_two());
        }
        if !segments.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(segments));
        }
        self.count = segments.len() as u32;
    }
}
