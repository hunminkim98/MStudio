//! C3D reader (via the `c3dio` crate) and a self-contained writer.
//!
//! Reader mirrors `dataLoader.read_data_from_c3d`: labels trimmed, empty and
//! duplicate labels dropped, coordinates mm → m in **f32** (the Python
//! divides a float32 array), points with a negative residual → NaN (what the
//! Python `c3d` package does with `check_nan=True`). Unlike the Python code,
//! a dropped label does not shift the remaining columns.
//!
//! The writer produces the same file layout as the Python `c3d.Writer` used by
//! `dataSaver.save_to_c3d` (Intel byte order, float samples in mm, missing
//! samples as `(0, 0, 0)` with residual word −1). `c3dio`'s own writer emits
//! parameter blocks the reference Python reader cannot parse, so it is not
//! used. Files with more than 65 535 frames carry `POINT:LONG_FRAMES` and
//! 32-bit `TRIAL:ACTUAL_END_FIELD`, as Vicon readers expect.

use std::path::Path;

use c3dio::data::MarkerPoint;
use c3dio::C3d;
use mstudio_core::Take;
use ndarray::Array3;

use crate::{IoError, Result};

pub fn read_c3d(path: impl AsRef<Path>) -> Result<Take> {
    let c3d = C3d::load_path(path.as_ref().to_path_buf()).map_err(|e| IoError::C3d(format!("{e:?}")))?;
    let pts = &c3d.points;
    let n_frames = pts.points.rows();
    let n_cols = pts.points.cols();

    // (column index, trimmed label) for labels that are non-empty and unseen.
    let mut keep: Vec<(usize, String)> = Vec::new();
    for (col, raw) in pts.labels.iter().enumerate().take(n_cols) {
        let name = raw.trim();
        if !name.is_empty() && !keep.iter().any(|(_, k)| k == name) {
            keep.push((col, name.to_string()));
        }
    }
    if keep.is_empty() {
        return Err(IoError::C3d("C3D file has no usable point labels".into()));
    }

    let mut frames = Array3::<f64>::from_elem((n_frames, keep.len(), 3), f64::NAN);
    for f in 0..n_frames {
        for (m, (col, _)) in keep.iter().enumerate() {
            let p: &MarkerPoint = &pts.points[(f, *col)];
            if p.residual < 0.0 {
                continue;
            }
            for k in 0..3 {
                frames[[f, m, k]] = (p.point[k] / 1000.0f32) as f64;
            }
        }
    }

    let fps = pts.frame_rate as f64;
    let first = pts.first_frame as i64;
    let frame_numbers: ndarray::Array1<i64> = (0..n_frames as i64).map(|i| first + i).collect();
    let time = frame_numbers.mapv(|i| i as f64 / fps);
    Ok(Take::with_columns(keep.into_iter().map(|(_, n)| n).collect(), fps, frames, frame_numbers, time))
}

pub fn write_c3d(path: impl AsRef<Path>, take: &Take) -> Result<()> {
    std::fs::write(path, c3d_bytes(take))?;
    Ok(())
}

// ---------------------------------------------------------------- writer --

const BLOCK: usize = 512;
const PROCESSOR_INTEL: u8 = 84;
const CHAR: i8 = -1;
const INT16: i8 = 2;
const FLOAT: i8 = 4;

/// Parameter-section records exactly as the reference reader walks them:
/// `[name_len i8][group_id i8][name][offset i16][body]`, where `offset` is
/// counted from the start of the offset field to the next record.
struct Params {
    buf: Vec<u8>,
}

impl Params {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(BLOCK * 2) }
    }

    fn record(&mut self, name: &str, id: i8, body: &[u8]) {
        let name = ascii(name);
        self.buf.push(name.len() as u8); // i8 with the sign bit clear (unlocked)
        self.buf.push(id as u8);
        self.buf.extend_from_slice(&name);
        self.buf.extend_from_slice(&((2 + body.len()) as i16).to_le_bytes());
        self.buf.extend_from_slice(body);
    }

    fn group(&mut self, id: i8, name: &str, desc: &str) {
        let desc = ascii(desc);
        let mut body = vec![desc.len() as u8];
        body.extend_from_slice(&desc);
        self.record(name, -id, &body);
    }

    fn param(&mut self, group: i8, name: &str, ty: i8, dims: &[u8], data: &[u8], desc: &str) {
        let desc = ascii(desc);
        let mut body = vec![ty as u8, dims.len() as u8];
        body.extend_from_slice(dims);
        body.extend_from_slice(data);
        body.push(desc.len() as u8);
        body.extend_from_slice(&desc);
        self.record(name, group, &body);
    }

    fn int16(&mut self, group: i8, name: &str, v: u16, desc: &str) {
        self.param(group, name, INT16, &[], &v.to_le_bytes(), desc);
    }

    fn float(&mut self, group: i8, name: &str, v: f32, desc: &str) {
        self.param(group, name, FLOAT, &[], &v.to_le_bytes(), desc);
    }

    /// `[width, count]` char array, each string space-padded to `width`.
    fn strings(&mut self, group: i8, name: &str, items: &[Vec<u8>], desc: &str) {
        let width = items.iter().map(Vec::len).max().unwrap_or(0).min(255);
        let mut data = Vec::with_capacity(width * items.len());
        for s in items {
            let mut s = s.clone();
            s.resize(width, b' ');
            data.extend_from_slice(&s[..width]);
        }
        self.param(group, name, CHAR, &[width as u8, items.len() as u8], &data, desc);
    }

    /// Terminator plus zero padding to a block boundary; returns the block count.
    fn finish(mut self, header: [u8; 4]) -> (Vec<u8>, u8) {
        let mut out = header.to_vec();
        out.append(&mut self.buf);
        out.extend_from_slice(&[0, 0]);
        let blocks = out.len().div_ceil(BLOCK);
        out.resize(blocks * BLOCK, 0);
        (out, blocks as u8)
    }
}

/// C3D is an ASCII format; anything else becomes `?` (Latin-1 would be
/// ambiguous across readers).
fn ascii(s: &str) -> Vec<u8> {
    s.chars().map(|c| if c.is_ascii() { c as u8 } else { b'?' }).collect()
}

fn build_params(take: &Take, data_start_block: u16) -> Params {
    let n_frames = take.n_frames();
    let n_markers = take.n_markers();
    let (analog, point, trial) = (1i8, 2i8, 3i8);
    let mut p = Params::new();

    p.group(analog, "ANALOG", "ANALOG group");
    p.float(analog, "GEN_SCALE", 1.0, "Analog general scale factor");
    p.int16(analog, "USED", 0, "Analog channel count");
    p.float(analog, "RATE", 0.0, "Analog samples per second");
    p.param(analog, "SCALE", FLOAT, &[0], &[], "Analog channel scale factors");
    p.param(analog, "OFFSET", INT16, &[0], &[], "Analog channel offsets");
    p.param(analog, "DESCRIPTIONS", CHAR, &[1, 0], &[], "Channel descriptions.");

    p.group(point, "POINT", "POINT group");
    // Labels beyond 255 go to LABELS2, LABELS3, … (C3D convention).
    let labels: Vec<Vec<u8>> = take.markers.iter().map(|m| ascii(m)).collect();
    for (i, chunk) in labels.chunks(255).enumerate() {
        let name = if i == 0 { "LABELS".to_string() } else { format!("LABELS{}", i + 1) };
        p.strings(point, &name, chunk, "Point labels.");
    }
    p.int16(point, "USED", n_markers as u16, "Number of point samples");
    p.int16(point, "FRAMES", n_frames.min(u16::MAX as usize) as u16, "Total frame count");
    if n_frames >= u16::MAX as usize {
        p.float(point, "LONG_FRAMES", n_frames as f32, "Total frame count");
    }
    p.int16(point, "DATA_START", data_start_block, "First data block containing frame samples.");
    p.float(point, "SCALE", -1.0, "Point data scaling factor");
    p.float(point, "RATE", take.fps as f32, "Point data sample rate");
    p.param(point, "UNITS", CHAR, &[4], b"mm  ", "Units used for point data measurements.");
    if n_markers <= 255 {
        let blank = vec![vec![b' ']; n_markers];
        p.strings(point, "DESCRIPTIONS", &blank, "Channel descriptions.");
    }

    p.group(trial, "TRIAL", "TRIAL group");
    let words = |v: u32| [v as u16, (v >> 16) as u16].iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    p.param(trial, "ACTUAL_START_FIELD", INT16, &[2], &words(1), "Actual start frame");
    p.param(trial, "ACTUAL_END_FIELD", INT16, &[2], &words(n_frames as u32), "Actual end frame");
    p
}

/// The complete file as bytes.
pub fn c3d_bytes(take: &Take) -> Vec<u8> {
    let n_frames = take.n_frames();
    let n_markers = take.n_markers();

    // The parameter section's length does not depend on DATA_START's value,
    // so build once to size it, then once more with the real block number.
    let (_, blocks) = build_params(take, 0).finish([0, 0, 0, PROCESSOR_INTEL]);
    let data_start = 2 + blocks as u16;
    let (params, _) = build_params(take, data_start).finish([0, 0, blocks, PROCESSOR_INTEL]);

    let mut header = [0u8; BLOCK];
    header[0] = 2; // parameter section starts at block 2
    header[1] = 0x50;
    header[2..4].copy_from_slice(&(n_markers as u16).to_le_bytes());
    header[4..6].copy_from_slice(&0u16.to_le_bytes()); // analog measurements per frame
    header[6..8].copy_from_slice(&1u16.to_le_bytes()); // first frame
    header[8..10].copy_from_slice(&(n_frames.min(u16::MAX as usize) as u16).to_le_bytes());
    header[10..12].copy_from_slice(&0u16.to_le_bytes()); // max interpolation gap
    header[12..16].copy_from_slice(&(-1.0f32).to_le_bytes()); // negative scale = float samples
    header[16..18].copy_from_slice(&data_start.to_le_bytes());
    header[18..20].copy_from_slice(&0u16.to_le_bytes()); // analog samples per frame
    header[20..24].copy_from_slice(&(take.fps as f32).to_le_bytes());

    let mut out = Vec::with_capacity(BLOCK + params.len() + n_frames * n_markers * 16 + BLOCK);
    out.extend_from_slice(&header);
    out.extend_from_slice(&params);
    debug_assert_eq!(out.len(), (data_start as usize - 1) * BLOCK);
    for f in 0..n_frames {
        for m in 0..n_markers {
            let (xyz, word) = match take.position(f, m) {
                // Python multiplies in f64 and the writer stores float32.
                Some(p) => ([(p[0] * 1000.0) as f32, (p[1] * 1000.0) as f32, (p[2] * 1000.0) as f32], 0.0f32),
                None => ([0.0; 3], -1.0),
            };
            for v in xyz {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    let padded = out.len().div_ceil(BLOCK) * BLOCK;
    out.resize(padded, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_take() -> Take {
        let mut f = Array3::<f64>::zeros((3, 2, 3));
        f[[0, 0, 0]] = 0.5;
        f[[1, 1, 2]] = -1.25;
        f[[2, 1, 0]] = f64::NAN;
        Take::new(vec!["A".into(), "LongerName".into()], 100.0, f)
    }

    /// Walk the parameter section the way the Python reader does.
    fn walk(bytes: &[u8]) -> Vec<(String, i8, Vec<u8>)> {
        let blocks = bytes[514] as usize;
        let end = 512 + blocks * 512;
        let mut pos = 516;
        let mut out = Vec::new();
        while pos < end {
            let (nlen, gid) = (bytes[pos] as i8, bytes[pos + 1] as i8);
            if nlen == 0 || gid == 0 {
                break;
            }
            let name = String::from_utf8(bytes[pos + 2..pos + 2 + nlen as usize].to_vec()).unwrap();
            let p = pos + 2 + nlen as usize;
            let off = i16::from_le_bytes([bytes[p], bytes[p + 1]]) as usize;
            out.push((name, gid, bytes[p + 2..p + off].to_vec()));
            pos = p + off;
        }
        out
    }

    #[test]
    fn header_and_layout_follow_the_reference_writer() {
        let t = small_take();
        let b = c3d_bytes(&t);
        assert_eq!(b.len() % BLOCK, 0);
        assert_eq!((b[0], b[1]), (2, 0x50));
        assert_eq!(u16::from_le_bytes([b[2], b[3]]), 2);
        assert_eq!(u16::from_le_bytes([b[8], b[9]]), 3);
        assert_eq!(f32::from_le_bytes([b[12], b[13], b[14], b[15]]), -1.0);
        let data_start = u16::from_le_bytes([b[16], b[17]]) as usize;
        assert_eq!(data_start, 2 + b[514] as usize);
        assert_eq!(f32::from_le_bytes([b[20], b[21], b[22], b[23]]), 100.0);
        assert_eq!(b[515], PROCESSOR_INTEL);

        let recs = walk(&b);
        let names: Vec<&str> = recs.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"POINT") && names.contains(&"LABELS") && names.contains(&"ACTUAL_END_FIELD"));
        let labels = recs.iter().find(|(n, _, _)| n == "LABELS").unwrap();
        // body = [type, ndims, width, count, data..., desc_len, desc]
        assert_eq!(&labels.2[..4], &[CHAR as u8, 2, 10, 2]);
        assert_eq!(&labels.2[4..24], b"A         LongerName");

        // frame 0 marker 0 = 500 mm, residual word 0; frame 2 marker 1 invalid
        let d = (data_start - 1) * BLOCK;
        assert_eq!(f32::from_le_bytes(b[d..d + 4].try_into().unwrap()), 500.0);
        assert_eq!(f32::from_le_bytes(b[d + 12..d + 16].try_into().unwrap()), 0.0);
        let inv = d + (2 * 2 + 1) * 16;
        assert_eq!(f32::from_le_bytes(b[inv + 12..inv + 16].try_into().unwrap()), -1.0);
    }
}
