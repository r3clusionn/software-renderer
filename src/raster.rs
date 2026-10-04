//! The rasterizer: clipping in homogeneous space, edge functions in fixed point with the top-left
//! fill rule, a depth buffer, and perspective-correct attributes with their screen-space
//! derivatives (for mipmapping). Triangles are binned into tiles and tiles are drawn in parallel;
//! within a tile triangles keep their submission order, so the image never depends on how many
//! threads drew it.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::math::{v4, Vec3, Vec4};

/// Attributes carried from vertices to pixels.
pub const MAX_ATTRS: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    /// Position in clip space.
    pub pos: Vec4,
    pub attrs: [f32; MAX_ATTRS],
}

/// What a fragment shader sees for one pixel.
pub struct Fragment<'a> {
    pub x: u32,
    pub y: u32,
    /// Depth after the perspective divide, 0 (near) to 1 (far).
    pub depth: f32,
    /// Perspective-correct attributes.
    pub attrs: &'a [f32; MAX_ATTRS],
    /// How each attribute changes per pixel to the right and per pixel down.
    pub ddx: &'a [f32; MAX_ATTRS],
    pub ddy: &'a [f32; MAX_ATTRS],
    /// Index of the triangle in the draw call.
    pub prim: u32,
    pub front_facing: bool,
}

pub trait Shader: Sync {
    /// The colour for one pixel (linear RGB), or `None` to discard it.
    fn shade(&self, f: &Fragment) -> Option<Vec3>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cull {
    None,
    Back,
    Front,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    Replace,
    /// Add to what is there (for coverage counting in tests and for light accumulation).
    Add,
}

#[derive(Clone, Copy, Debug)]
pub struct State {
    pub cull: Cull,
    pub depth_test: bool,
    pub depth_write: bool,
    pub color_write: bool,
    pub blend: Blend,
    /// Added to every depth before testing and writing (for shadow maps).
    pub depth_bias: f32,
    /// Interpolate the given attributes counting how far into each triangle (in screen space)
    /// rather than perspective-correctly. Only tests use this, to show the difference.
    pub affine: bool,
}

impl Default for State {
    fn default() -> State {
        State {
            cull: Cull::Back,
            depth_test: true,
            depth_write: true,
            color_write: true,
            blend: Blend::Replace,
            depth_bias: 0.0,
            affine: false,
        }
    }
}

/// A colour and depth target, plus an optional buffer of which triangle covers each pixel.
pub struct Target {
    pub w: usize,
    pub h: usize,
    pub color: Vec<Vec3>,
    pub depth: Vec<f32>,
    pub ids: Option<Vec<u32>>,
}

impl Target {
    pub fn new(w: usize, h: usize) -> Target {
        Target { w, h, color: vec![Vec3::ZERO; w * h], depth: vec![1.0; w * h], ids: None }
    }

    /// A target already cleared to `c` (one pass over memory instead of two).
    pub fn filled(w: usize, h: usize, c: Vec3) -> Target {
        Target { w, h, color: vec![c; w * h], depth: vec![1.0; w * h], ids: None }
    }

    /// A depth buffer only (for shadow maps); drawing colour into it panics.
    pub fn depth_only(w: usize, h: usize) -> Target {
        Target { w, h, color: Vec::new(), depth: vec![1.0; w * h], ids: None }
    }

    pub fn with_ids(mut self) -> Target {
        self.ids = Some(vec![u32::MAX; self.w * self.h]);
        self
    }

    pub fn clear(&mut self, c: Vec3) {
        self.color.fill(c);
        self.depth.fill(1.0);
        if let Some(ids) = &mut self.ids {
            ids.fill(u32::MAX);
        }
    }
}

const SUB_BITS: i64 = 8;
const SUB: f32 = (1 << SUB_BITS) as f32;
const TILE: usize = 64;
/// Clip against x and y only at this multiple of w (a guard band): smaller triangles are simply
/// limited to the screen by their bounding box, and fixed-point coordinates stay in range.
const GUARD: f32 = 16.0;

/// A triangle after clipping and the perspective divide, ready to rasterize.
struct Prepared {
    /// Screen positions in fixed point (1/256 pixel).
    xs: [i64; 3],
    ys: [i64; 3],
    /// Depth and 1/w at each vertex.
    z: [f32; 3],
    inv_w: [f32; 3],
    /// Attributes divided by w (or plain, for affine interpolation).
    a: [[f32; MAX_ATTRS]; 3],
    nattr: usize,
    prim: u32,
    batch: u32,
    front: bool,
    bbox: (i64, i64, i64, i64),
}

/// One draw: triangles (three vertices each) with their state and shader.
pub struct Batch<'a> {
    pub verts: &'a [Vertex],
    pub nattr: usize,
    pub state: State,
    pub shader: &'a dyn Shader,
}

/// Draws triangles (three vertices each) into `target`, on up to `threads` threads.
pub fn draw(target: &mut Target, verts: &[Vertex], nattr: usize, state: &State, shader: &dyn Shader, threads: usize) {
    draw_batches(target, &[Batch { verts, nattr, state: *state, shader }], threads);
}

/// Draws several batches in one parallel pass: every tile draws every batch's triangles in
/// order, so a small object never leaves threads idle waiting for the next draw, and the result
/// is the same as drawing the batches one after another.
pub fn draw_batches(target: &mut Target, batches: &[Batch], threads: usize) {
    let (w, h) = (target.w, target.h);
    assert!(
        batches.iter().all(|b| !b.state.color_write) || target.color.len() == w * h,
        "a depth-only target cannot take colour"
    );
    let threads = threads.max(1);
    // Clip and set up triangles in parallel chunks, then join them in order.
    let jobs: Vec<(usize, usize, usize)> = batches
        .iter()
        .enumerate()
        .flat_map(|(bi, b)| {
            let n = b.verts.len() / 3;
            let chunk = n.div_ceil(threads * 4).max(256);
            (0..n).step_by(chunk).map(move |s| (bi, s, (s + chunk).min(n)))
        })
        .collect();
    let setup = |&(bi, s, e): &(usize, usize, usize)| {
        let b = &batches[bi];
        let mut out = Vec::with_capacity(e - s);
        for prim in s..e {
            let t = &b.verts[prim * 3..prim * 3 + 3];
            clip(&[t[0], t[1], t[2]], b.nattr, &mut |poly| {
                prepare(poly, b.nattr, prim as u32, bi as u32, w, h, &b.state, &mut out)
            });
        }
        out
    };
    let prepared: Vec<Prepared> = if threads == 1 || jobs.len() == 1 {
        jobs.iter().flat_map(setup).collect()
    } else {
        let mut parts: Vec<Vec<Prepared>> = (0..jobs.len()).map(|_| Vec::new()).collect();
        pool().scope(|sc| {
            for (slot, job) in parts.iter_mut().zip(&jobs) {
                let setup = &setup;
                sc.spawn(move || *slot = setup(job));
            }
        });
        parts.into_iter().flatten().collect()
    };
    // Bin into tiles by bounding box.
    let tx = w.div_ceil(TILE);
    let ty = h.div_ceil(TILE);
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); tx * ty];
    for (i, p) in prepared.iter().enumerate() {
        let (x0, y0, x1, y1) = p.bbox;
        for by in (y0 as usize / TILE)..=((y1 as usize) / TILE).min(ty - 1) {
            for bx in (x0 as usize / TILE)..=((x1 as usize) / TILE).min(tx - 1) {
                bins[by * tx + bx].push(i as u32);
            }
        }
    }
    let next = AtomicUsize::new(0);
    let tiles = tx * ty;
    // Each tile is written by one thread only; the buffers are shared through raw pointers that
    // never touch the same pixel from two threads.
    let color = SyncPtr(target.color.as_mut_ptr());
    let depth = SyncPtr(target.depth.as_mut_ptr());
    let ids = SyncPtr(target.ids.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()));
    let work = || loop {
        let t = next.fetch_add(1, Ordering::Relaxed);
        if t >= tiles {
            return;
        }
        let (bx, by) = (t % tx, t / tx);
        let clip_rect =
            ((bx * TILE) as i64, (by * TILE) as i64, ((bx + 1) * TILE).min(w) as i64 - 1, ((by + 1) * TILE).min(h) as i64 - 1);
        for &i in &bins[t] {
            let p = &prepared[i as usize];
            let b = &batches[p.batch as usize];
            // SAFETY: pixels inside this tile are only touched by the thread that took the tile.
            unsafe { raster(p, clip_rect, w, &b.state, b.shader, &color, &depth, &ids) };
        }
    };
    let threads = threads.min(tiles.max(1));
    if threads == 1 {
        work();
    } else {
        let work = &work;
        pool().scope(|s| {
            for _ in 0..threads {
                s.spawn(work);
            }
        });
    }
}

/// One pool for the whole process, so a frame does not create threads.
pub fn pool() -> &'static workpool::ThreadPool {
    static POOL: std::sync::OnceLock<workpool::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| workpool::ThreadPool::new(std::thread::available_parallelism().map_or(4, |n| n.get())))
}

#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// Sutherland-Hodgman clipping of a triangle against the near and far planes and the guard band,
/// in clip space where every attribute is linear. Hands each triangle of the clipped polygon to
/// `emit` (a triangle entirely inside, nearly always the case, goes straight through).
fn clip(tri: &[Vertex; 3], nattr: usize, emit: &mut dyn FnMut(&[Vertex; 3])) {
    // Each plane as a vector: inside where dot(plane, pos) >= 0.
    const PLANES: [Vec4; 6] = [
        v4(0.0, 0.0, 1.0, 0.0),    // z >= 0 (near)
        v4(0.0, 0.0, -1.0, 1.0),   // z <= w (far)
        v4(1.0, 0.0, 0.0, GUARD),  // x >= -G w
        v4(-1.0, 0.0, 0.0, GUARD), // x <= G w
        v4(0.0, 1.0, 0.0, GUARD),  // y >= -G w
        v4(0.0, -1.0, 0.0, GUARD), // y <= G w
    ];
    let inside_all = |v: &Vertex| PLANES.iter().all(|p| p.dot(v.pos) >= 0.0);
    if tri.iter().all(inside_all) {
        return emit(tri);
    }
    let mut poly: Vec<Vertex> = tri.to_vec();
    for p in PLANES {
        if poly.is_empty() {
            break;
        }
        let mut out = Vec::with_capacity(poly.len() + 2);
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            let (da, db) = (p.dot(a.pos), p.dot(b.pos));
            if da >= 0.0 {
                out.push(a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                let t = da / (da - db);
                let mut v = Vertex { pos: a.pos.lerp(b.pos, t), attrs: a.attrs };
                for k in 0..nattr {
                    v.attrs[k] = a.attrs[k] + (b.attrs[k] - a.attrs[k]) * t;
                }
                out.push(v);
            }
        }
        poly = out;
    }
    for i in 1..poly.len().saturating_sub(1) {
        emit(&[poly[0], poly[i], poly[i + 1]]);
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare(tri: &[Vertex; 3], nattr: usize, prim: u32, batch: u32, w: usize, h: usize, state: &State, out: &mut Vec<Prepared>) {
    let mut xs = [0i64; 3];
    let mut ys = [0i64; 3];
    let mut z = [0f32; 3];
    let mut inv_w = [0f32; 3];
    let mut a = [[0f32; MAX_ATTRS]; 3];
    for i in 0..3 {
        let p = tri[i].pos;
        if p.w <= 0.0 {
            return; // only possible for degenerate input after clipping
        }
        let iw = 1.0 / p.w;
        let sx = (p.x * iw + 1.0) * 0.5 * w as f32;
        let sy = (1.0 - p.y * iw) * 0.5 * h as f32;
        xs[i] = (sx * SUB).round() as i64;
        ys[i] = (sy * SUB).round() as i64;
        z[i] = p.z * iw;
        inv_w[i] = if state.affine { 1.0 } else { iw };
        for (dst, src) in a[i][..nattr].iter_mut().zip(&tri[i].attrs[..nattr]) {
            *dst = src * inv_w[i];
        }
    }
    // Twice the signed area; positive means counter-clockwise on screen (y down flips it).
    let area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0]);
    if area == 0 {
        return;
    }
    // Counter-clockwise in the usual y-up sense is a negative area here.
    let front = area < 0;
    match state.cull {
        Cull::Back if !front => return,
        Cull::Front if front => return,
        _ => {}
    }
    // Rasterize with one winding (positive area, where every edge function is positive inside).
    if area < 0 {
        xs.swap(1, 2);
        ys.swap(1, 2);
        z.swap(1, 2);
        inv_w.swap(1, 2);
        a.swap(1, 2);
    }
    let (minx, maxx) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
    let (miny, maxy) = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap());
    let sub = 1i64 << SUB_BITS;
    let x0 = (minx >> SUB_BITS).max(0);
    let y0 = (miny >> SUB_BITS).max(0);
    let x1 = ((maxx + sub - 1) >> SUB_BITS).min(w as i64 - 1);
    let y1 = ((maxy + sub - 1) >> SUB_BITS).min(h as i64 - 1);
    if x0 > x1 || y0 > y1 {
        return;
    }
    out.push(Prepared { xs, ys, z, inv_w, a, nattr, prim, batch, front, bbox: (x0, y0, x1, y1) });
}

/// Draws one prepared triangle inside `rect` (a tile).
#[allow(clippy::too_many_arguments)]
unsafe fn raster(
    p: &Prepared,
    rect: (i64, i64, i64, i64),
    w: usize,
    state: &State,
    shader: &dyn Shader,
    color: &SyncPtr<Vec3>,
    depth: &SyncPtr<f32>,
    ids: &SyncPtr<u32>,
) {
    let x0 = p.bbox.0.max(rect.0);
    let y0 = p.bbox.1.max(rect.1);
    let x1 = p.bbox.2.min(rect.2);
    let y1 = p.bbox.3.min(rect.3);
    if x0 > x1 || y0 > y1 {
        return;
    }
    let (xs, ys) = (p.xs, p.ys);
    // Edge i runs from vertex i+1 to vertex i+2 and is opposite vertex i, so its function divided
    // by the area is vertex i's barycentric weight; with the winding fixed in `prepare` it is
    // positive inside.
    let edge = |i: usize| {
        let (a, b) = ((i + 1) % 3, (i + 2) % 3);
        let (ax, ay, bx, by) = (xs[a], ys[a], xs[b], ys[b]);
        // E(x, y) = (bx - ax) * (y - ay) - (by - ay) * (x - ax), here with A = -(by - ay)
        // per unit x and B = (bx - ax) per unit y.
        let (ea, eb) = (ay - by, bx - ax);
        // Top-left rule: pixels exactly on an edge belong to it only if it is a top edge
        // (horizontal, the triangle below it) or a left edge (the triangle to its right).
        // The function grows towards the inside, so its gradient (ea, eb) says where that is.
        let top_left = (ea == 0 && eb > 0) || ea > 0;
        let bias = if top_left { 0 } else { -1 };
        (ea, eb, ax, ay, bias)
    };
    let e = [edge(0), edge(1), edge(2)];
    let half = 1i64 << (SUB_BITS - 1);
    let sub = 1i64 << SUB_BITS;
    // Edge values at the first pixel centre, then stepped per pixel.
    let px0 = x0 * sub + half;
    let py0 = y0 * sub + half;
    let mut row = [0i64; 3];
    for i in 0..3 {
        let (ea, eb, ax, ay, bias) = e[i];
        row[i] = ea * (px0 - ax) + eb * (py0 - ay) + bias;
    }
    let total = ((xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])).abs() as f32;
    let inv_area = 1.0 / total;
    // Plane equations in pixels for depth, 1/w and attributes / w: v = c + dx * x + dy * y,
    // derived from the barycentric weights' own steps.
    let step_x = [e[0].0 as f32 * SUB * inv_area, e[1].0 as f32 * SUB * inv_area, e[2].0 as f32 * SUB * inv_area];
    let step_y = [e[0].1 as f32 * SUB * inv_area, e[1].1 as f32 * SUB * inv_area, e[2].1 as f32 * SUB * inv_area];
    let plane = |v: [f32; 3]| {
        (step_x[0] * v[0] + step_x[1] * v[1] + step_x[2] * v[2], step_y[0] * v[0] + step_y[1] * v[1] + step_y[2] * v[2])
    };
    let (dwx, dwy) = plane(p.inv_w);
    let mut dax = [0f32; MAX_ATTRS];
    let mut day = [0f32; MAX_ATTRS];
    for k in 0..p.nattr {
        let (x, y) = plane([p.a[0][k], p.a[1][k], p.a[2][k]]);
        dax[k] = x;
        day[k] = y;
    }
    let mut attrs = [0f32; MAX_ATTRS];
    let mut ddx = [0f32; MAX_ATTRS];
    let mut ddy = [0f32; MAX_ATTRS];
    for y in y0..=y1 {
        let mut ev = row;
        for x in x0..=x1 {
            if ev[0] >= 0 && ev[1] >= 0 && ev[2] >= 0 {
                // Barycentric weights from the (unbiased) edge values.
                let l0 = (ev[0] - e[0].4) as f32 * inv_area;
                let l1 = (ev[1] - e[1].4) as f32 * inv_area;
                let l2 = 1.0 - l0 - l1;
                let z = l0 * p.z[0] + l1 * p.z[1] + l2 * p.z[2] + state.depth_bias;
                let idx = y as usize * w + x as usize;
                let dptr = depth.0.add(idx);
                if !state.depth_test || z < *dptr {
                    let b = l0 * p.inv_w[0] + l1 * p.inv_w[1] + l2 * p.inv_w[2];
                    let ib = 1.0 / b;
                    for k in 0..p.nattr {
                        let av = l0 * p.a[0][k] + l1 * p.a[1][k] + l2 * p.a[2][k];
                        attrs[k] = av * ib;
                        // d(A/B) = (dA * B - A * dB) / B^2
                        ddx[k] = (dax[k] * b - av * dwx) * ib * ib;
                        ddy[k] = (day[k] * b - av * dwy) * ib * ib;
                    }
                    let frag = Fragment {
                        x: x as u32,
                        y: y as u32,
                        depth: z,
                        attrs: &attrs,
                        ddx: &ddx,
                        ddy: &ddy,
                        prim: p.prim,
                        front_facing: p.front,
                    };
                    if let Some(c) = shader.shade(&frag) {
                        if state.depth_write {
                            *dptr = z;
                        }
                        if state.color_write {
                            let cptr = color.0.add(idx);
                            *cptr = match state.blend {
                                Blend::Replace => c,
                                Blend::Add => *cptr + c,
                            };
                        }
                        if !ids.0.is_null() {
                            *ids.0.add(idx) = p.prim;
                        }
                    }
                }
            }
            for i in 0..3 {
                ev[i] += e[i].0 * sub;
            }
        }
        for i in 0..3 {
            row[i] += e[i].1 * sub;
        }
    }
}
