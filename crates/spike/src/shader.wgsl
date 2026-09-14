// MStudio spike shaders.
//
// All marker positions for the whole take live in one storage buffer
// (frame-major, vec4 per marker: xyz + pad). The uniform block carries the
// current frame index; nothing per-frame is uploaded except that index.
// NaN positions are stored as CULL_SENTINEL on the CPU side because WGSL
// NaN comparisons are not reliable across backends.

const CULL_SENTINEL: f32 = 1.0e29;

struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,   // physical pixels of the 3D view
    frame: u32,
    n_markers: u32,
    point_size: f32,       // marker diameter in pixels
    selected: i32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> states: array<u32>;
@group(0) @binding(3) var<storage, read> pairs: array<u32>;

fn culled() -> vec4<f32> {
    // z > w puts the vertex outside the clip volume on every backend.
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

fn marker_pos(id: u32) -> vec4<f32> {
    return positions[u.frame * u.n_markers + id];
}

// ---------------------------------------------------------------- markers --

struct MarkerOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

fn state_color(s: u32) -> vec3<f32> {
    switch s {
        case 1u: { return vec3<f32>(1.0, 0.85, 0.15); } // selected
        case 2u: { return vec3<f32>(1.0, 0.30, 0.30); } // outlier frame
        case 3u: { return vec3<f32>(0.40, 0.80, 1.00); } // pattern reference
        default: { return vec3<f32>(0.92, 0.92, 0.92); }
    }
}

@vertex
fn vs_marker(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MarkerOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vi];
    let p = marker_pos(ii);

    var out: MarkerOut;
    out.uv = corner;
    out.color = state_color(states[ii]);

    if (p.x > CULL_SENTINEL) {
        out.clip = culled();
        return out;
    }

    var clip = u.view_proj * vec4<f32>(p.xyz, 1.0);
    var size = u.point_size;
    if (i32(ii) == u.selected) {
        size = size * 1.5;
    }
    // Billboard offset in pixels -> NDC, scaled by w so the quad stays
    // a constant pixel size at any depth.
    let px = corner * size * 0.5;
    clip.x = clip.x + px.x * 2.0 / u.viewport.x * clip.w;
    clip.y = clip.y + px.y * 2.0 / u.viewport.y * clip.w;
    out.clip = clip;
    return out;
}

@fragment
fn fs_marker(in: MarkerOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    // Cheap sphere shading so markers read as balls, not flat discs.
    let n = vec3<f32>(in.uv, sqrt(1.0 - r2));
    let light = normalize(vec3<f32>(0.4, 0.6, 1.0));
    let shade = 0.35 + 0.65 * max(dot(n, light), 0.0);
    return vec4<f32>(in.color * shade, 1.0);
}

// ------------------------------------------------------------------ lines --

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// Skeleton: vertex i reads marker id pairs[i]; pairs are laid out as
// [parent0, child0, parent1, child1, ...] and drawn as a line list.
@vertex
fn vs_skeleton(@builtin(vertex_index) vi: u32) -> LineOut {
    let me = marker_pos(pairs[vi]);
    let other = marker_pos(pairs[vi ^ 1u]);
    var out: LineOut;
    out.color = vec3<f32>(0.35, 0.75, 1.0);
    if (me.x > CULL_SENTINEL || other.x > CULL_SENTINEL) {
        out.clip = culled();
    } else {
        out.clip = u.view_proj * vec4<f32>(me.xyz, 1.0);
    }
    return out;
}

// Static / CPU-fed lines (grid, axes, trajectory): plain vertex buffer.
struct StaticVertex {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_static(v: StaticVertex) -> LineOut {
    var out: LineOut;
    out.clip = u.view_proj * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
