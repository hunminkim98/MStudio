//! The take on the GPU (plan rules R1 / R2): every frame of every marker in
//! one storage buffer, written once at load and patched by frame range after
//! an edit. Frame advance is a uniform write, never a re-upload.

use mstudio_core::{DirtyRange, Take};
use wgpu::util::DeviceExt;

/// Missing samples are stored as this value on every component; the shaders
/// cull anything above `1e29`.
pub const CULL_SENTINEL: f32 = 1.0e30;

const BYTES_PER_MARKER: u64 = 16; // vec4<f32>

pub struct GpuTake {
    pub buffer: wgpu::Buffer,
    pub n_frames: usize,
    pub n_markers: usize,
}

impl GpuTake {
    pub fn new(device: &wgpu::Device, take: &Take) -> GpuTake {
        let data = take.frames_f32(CULL_SENTINEL);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("take positions"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        GpuTake { buffer, n_frames: take.n_frames(), n_markers: take.n_markers() }
    }

    /// Byte size of the buffer a take needs.
    pub fn byte_size(take: &Take) -> u64 {
        (take.n_frames() * take.n_markers()) as u64 * BYTES_PER_MARKER
    }

    pub fn byte_offset(&self, frame: usize) -> u64 {
        (frame * self.n_markers) as u64 * BYTES_PER_MARKER
    }

    /// Upload only the frames in `range` (rule R2). No-op for an empty range.
    pub fn write_range(&self, queue: &wgpu::Queue, take: &Take, range: DirtyRange) {
        debug_assert_eq!(take.n_markers(), self.n_markers);
        let end = range.end.min(self.n_frames);
        if range.start >= end {
            return;
        }
        let bytes = Self::bytes_for_range(take, range.start, end);
        queue.write_buffer(&self.buffer, self.byte_offset(range.start), &bytes);
    }

    /// The f32 image of frames `[start, end)` exactly as stored on the GPU.
    pub fn bytes_for_range(take: &Take, start: usize, end: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity((end - start) * take.n_markers() * BYTES_PER_MARKER as usize);
        for f in start..end {
            for m in 0..take.n_markers() {
                let v = match take.position(f, m) {
                    Some(p) => [p[0] as f32, p[1] as f32, p[2] as f32, 0.0],
                    None => [CULL_SENTINEL, CULL_SENTINEL, CULL_SENTINEL, 0.0],
                };
                out.extend_from_slice(bytemuck::bytes_of(&v));
            }
        }
        out
    }
}

/// Pack a `[marker, frame]` outlier map into bits indexed `frame * n_markers + marker`,
/// matching `is_outlier` in the shader. Always at least one word.
pub fn pack_outliers(map: &ndarray::Array2<bool>) -> Vec<u32> {
    let (n_markers, n_frames) = map.dim();
    let n_bits = n_markers * n_frames;
    let mut words = vec![0u32; n_bits.div_ceil(32).max(1)];
    for f in 0..n_frames {
        for m in 0..n_markers {
            if map[[m, f]] {
                let idx = f * n_markers + m;
                words[idx / 32] |= 1 << (idx % 32);
            }
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn range_bytes_match_full_image_and_offsets() {
        let mut f = Array3::<f64>::zeros((4, 3, 3));
        f[[2, 1, 0]] = 7.5;
        f[[3, 2, 2]] = f64::NAN;
        let take = Take::new(vec!["a".into(), "b".into(), "c".into()], 10.0, f);
        let full: Vec<u8> = bytemuck::cast_slice(&take.frames_f32(CULL_SENTINEL)).to_vec();
        let part = GpuTake::bytes_for_range(&take, 2, 4);
        assert_eq!(part.len(), 2 * 3 * 16);
        assert_eq!(&full[2 * 3 * 16..], &part[..]);
        let v: [f32; 4] = *bytemuck::from_bytes(&part[16..32]);
        assert_eq!(v, [7.5, 0.0, 0.0, 0.0]);
        let missing: [f32; 4] = *bytemuck::from_bytes(&part[(3 + 2) * 16..(3 + 3) * 16]);
        assert_eq!(missing[0], CULL_SENTINEL);
    }

    #[test]
    fn outlier_bits_follow_frame_major_indexing() {
        let mut m = Array2::from_elem((3, 40), false);
        m[[1, 0]] = true; // idx 1
        m[[2, 11]] = true; // idx 11*3+2 = 35 → word 1, bit 3
        let w = pack_outliers(&m);
        assert_eq!(w.len(), 4);
        assert_eq!(w[0], 0b10);
        assert_eq!(w[1], 1 << 3);
        assert_eq!(pack_outliers(&Array2::from_elem((0, 0), false)), vec![0]);
    }
}
