// MStudio viewport shaders.
//
// Group 0: the take (all frames, one storage buffer), per-marker state,
//          skeleton pairs and the packed outlier map — uploaded once, read
//          per frame by index (plan rules R1 / R5).
// Group 1: a list of thick segments (grid, trajectories, analysis overlay).

struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    frame: u32,
    n_markers: u32,
    marker_size: f32,
    marker_opacity: f32,
    line_width: f32,
    skeleton_opacity: f32,
    color_normal: vec4<f32>,
    color_selected: vec4<f32>,
    color_pattern: vec4<f32>,
    color_analysis: vec4<f32>,
    skel_normal: vec4<f32>,
    skel_outlier: vec4<f32>,
    selected: i32,
    n_pairs: u32,
    flags: u32,
    _pad: u32,
};

const CULL: f32 = 1.0e29;
const STATE_NORMAL: u32 = 0u;
const STATE_SELECTED: u32 = 1u;
const STATE_PATTERN: u32 = 2u;
const STATE_ANALYSIS: u32 = 3u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> states: array<u32>;
@group(0) @binding(3) var<storage, read> pairs: array<u32>;
@group(0) @binding(4) var<storage, read> outlier_bits: array<u32>;

struct Segment {
    a: vec4<f32>, // xyz, w = width in pixels
    b: vec4<f32>,
    color: vec4<f32>,
};
@group(1) @binding(0) var<storage, read> segments: array<Segment>;

fn position_of(marker: u32) -> vec4<f32> {
    return positions[u.frame * u.n_markers + marker];
}

fn is_outlier(marker: u32) -> bool {
    let idx = u.frame * u.n_markers + marker;
    return ((outlier_bits[idx / 32u] >> (idx % 32u)) & 1u) == 1u;
}

fn culled() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0); // z > w: clipped away
}

// Quad corner for a screen-space thick segment between two clip positions.
fn thick_segment(clip_a: vec4<f32>, clip_b: vec4<f32>, width_px: f32, vi: u32) -> vec4<f32> {
    if (clip_a.w <= 0.0 || clip_b.w <= 0.0) {
        return culled();
    }
    var at_b = false;
    var side = -1.0;
    switch vi {
        case 1u: { at_b = true; side = -1.0; }
        case 2u: { at_b = true; side = 1.0; }
        case 4u: { at_b = true; side = 1.0; }
        case 5u: { at_b = false; side = 1.0; }
        default: { at_b = false; side = -1.0; }
    }
    let sa = clip_a.xy / clip_a.w * u.viewport * 0.5;
    let sb = clip_b.xy / clip_b.w * u.viewport * 0.5;
    var d = sb - sa;
    let len = length(d);
    if (len < 1.0e-6) {
        d = vec2<f32>(1.0, 0.0);
    } else {
        d = d / len;
    }
    let nrm = vec2<f32>(-d.y, d.x) * width_px * 0.5 * side;
    var clip = clip_a;
    if (at_b) {
        clip = clip_b;
    }
    return vec4<f32>(clip.xy + nrm * 2.0 / u.viewport * clip.w, clip.zw);
}

// ---------------------------------------------------------------- markers --

struct MarkerOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));

@vertex
fn vs_marker(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MarkerOut {
    var out: MarkerOut;
    let p = position_of(ii);
    var clip = u.view_proj * vec4<f32>(p.xyz, 1.0);
    if (p.x > CULL || clip.w <= 0.0) {
        out.clip = culled();
        out.uv = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    let state = states[ii];
    var size = u.marker_size;
    var color = u.color_normal;
    if (is_outlier(ii)) {
        color = u.skel_outlier;
    }
    if (state == STATE_PATTERN) {
        color = u.color_pattern;
    } else if (state == STATE_ANALYSIS) {
        color = u.color_analysis;
        size = size * 1.3;
    }
    if (state == STATE_SELECTED || i32(ii) == u.selected) {
        color = u.color_selected;
        size = size * 1.5;
    }
    let corner = QUAD[vi];
    let px = corner * size * 0.5;
    clip = vec4<f32>(clip.xy + px * 2.0 / u.viewport * clip.w, clip.zw);
    out.clip = clip;
    out.uv = corner;
    out.color = vec4<f32>(color.rgb, u.marker_opacity);
    return out;
}

@fragment
fn fs_marker(in: MarkerOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    let nz = sqrt(max(1.0 - r2, 0.0));
    let light = normalize(vec3<f32>(0.4, 0.6, 1.0));
    let shade = 0.35 + 0.65 * max(dot(vec3<f32>(in.uv, nz), light), 0.0);
    return vec4<f32>(in.color.rgb * shade, in.color.a);
}

// --------------------------------------------------------------- skeleton --

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_skeleton(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LineOut {
    var out: LineOut;
    let ma = pairs[ii * 2u];
    let mb = pairs[ii * 2u + 1u];
    let pa = position_of(ma);
    let pb = position_of(mb);
    if (pa.x > CULL || pb.x > CULL) {
        out.clip = culled();
        out.color = vec4<f32>(0.0);
        return out;
    }
    let ca = u.view_proj * vec4<f32>(pa.xyz, 1.0);
    let cb = u.view_proj * vec4<f32>(pb.xyz, 1.0);
    var width = u.line_width;
    var color = u.skel_normal;
    if (is_outlier(ma) || is_outlier(mb)) {
        color = u.skel_outlier;
        width = width * 1.5;
    }
    out.clip = thick_segment(ca, cb, width, vi);
    out.color = vec4<f32>(color.rgb, u.skeleton_opacity);
    return out;
}

// --------------------------------------------------------- segment lists --

@vertex
fn vs_segments(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> LineOut {
    var out: LineOut;
    let s = segments[ii];
    let ca = u.view_proj * vec4<f32>(s.a.xyz, 1.0);
    let cb = u.view_proj * vec4<f32>(s.b.xyz, 1.0);
    out.clip = thick_segment(ca, cb, s.a.w, vi);
    out.color = s.color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}
