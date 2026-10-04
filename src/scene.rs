//! Scenes and how they are lit: Blinn-Phong shading (flat, per vertex or per pixel) under one
//! directional light with a shadow map, textures, and supersampling.

use std::sync::Arc;
use std::time::Instant;

use crate::math::{v2, v3, Mat4, Vec3};
use crate::mesh::Mesh;
use crate::raster::{draw_batches, Batch, Cull, Fragment, Shader, State, Target, Vertex, MAX_ATTRS};
use crate::texture::{linear_to_srgb8, Filter, Texture};

#[derive(Clone)]
pub struct Material {
    pub albedo: Vec3,
    pub texture: Option<Arc<Texture>>,
    pub specular: f32,
    pub shininess: f32,
}

impl Default for Material {
    fn default() -> Material {
        Material { albedo: v3(0.8, 0.8, 0.8), texture: None, specular: 0.3, shininess: 48.0 }
    }
}

#[derive(Clone)]
pub struct Object {
    pub mesh: Arc<Mesh>,
    pub material: Material,
    pub transform: Mat4,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective(self.fovy, aspect, self.near, self.far) * Mat4::look_at(self.eye, self.target, self.up)
    }
}

pub struct Scene {
    pub objects: Vec<Object>,
    /// Direction towards the light.
    pub light_dir: Vec3,
    pub light: Vec3,
    pub ambient: Vec3,
    pub background: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shading {
    /// One normal per triangle (from the screen-space derivatives of the position).
    Flat,
    /// Lighting at the vertices, interpolated.
    Gouraud,
    /// Lighting at every pixel.
    Phong,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub width: usize,
    pub height: usize,
    pub threads: usize,
    pub shading: Shading,
    pub filter: Filter,
    /// Shadow map size in texels (0 turns shadows off).
    pub shadow_size: usize,
    /// Supersampling factor per axis (1 = off).
    pub ssaa: usize,
    pub wireframe: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            width: 1280,
            height: 720,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            shading: Shading::Phong,
            filter: Filter::Trilinear,
            shadow_size: 2048,
            ssaa: 1,
            wireframe: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub triangles: usize,
    pub shadow_ms: f64,
    pub main_ms: f64,
    pub resolve_ms: f64,
}

// Attribute layout: world position (3), normal (3), uv (2), per-vertex diffuse and specular for
// Gouraud shading (2), barycentric coordinates for the wireframe (3).
const A_POS: usize = 0;
const A_NRM: usize = 3;
const A_UV: usize = 6;
const A_GOURAUD: usize = 8;
const A_BARY: usize = 10;
const NATTR: usize = 13;

struct DepthOnly;

impl Shader for DepthOnly {
    fn shade(&self, _: &Fragment) -> Option<Vec3> {
        Some(Vec3::ZERO)
    }
}

struct Lit<'a> {
    scene: &'a Scene,
    material: &'a Material,
    eye: Vec3,
    shading: Shading,
    filter: Filter,
    shadow: Option<&'a ShadowMap>,
    wireframe: bool,
}

pub struct ShadowMap {
    size: usize,
    depth: Vec<f32>,
    pub light_vp: Mat4,
    /// The width of one shadow-map texel in world units.
    texel: f32,
}

impl ShadowMap {
    /// The fraction of light reaching a point (1 = lit), from a 3x3 percentage-closer filter.
    /// Looks up a world-space point with normal `n`. The point is first moved along its normal by
    /// about a texel ("normal offset"), so surfaces at a grazing angle to the light do not shadow
    /// themselves (shadow acne).
    fn visibility_at(&self, p: Vec3, n: Vec3, slope: f32) -> f32 {
        let q = self.light_vp * (p + n * (self.texel * (0.5 + 1.5 * slope))).extend(1.0);
        self.visibility([q.x, q.y, q.z, q.w], slope)
    }

    fn visibility(&self, lp: [f32; 4], slope: f32) -> f32 {
        if lp[3] <= 0.0 {
            return 1.0;
        }
        let (x, y, z) = (lp[0] / lp[3], lp[1] / lp[3], lp[2] / lp[3]);
        let u = (x + 1.0) * 0.5 * self.size as f32;
        let v = (1.0 - y) * 0.5 * self.size as f32;
        // A bias that grows with the surface's slope to the light avoids shadow acne.
        let bias = 0.0005 + 0.001 * slope;
        let mut lit = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (sx, sy) = ((u as i64 + dx).clamp(0, self.size as i64 - 1), (v as i64 + dy).clamp(0, self.size as i64 - 1));
                if z - bias <= self.depth[sy as usize * self.size + sx as usize] {
                    lit += 1.0;
                }
            }
        }
        lit / 9.0
    }
}

/// The Blinn-Phong terms at a point: (diffuse, specular) as plain factors.
fn blinn_phong(scene: &Scene, m: &Material, eye: Vec3, p: Vec3, n: Vec3) -> (f32, f32) {
    let l = scene.light_dir.normalize();
    let v = (eye - p).normalize();
    let h = (l + v).normalize();
    let diff = n.dot(l).max(0.0);
    let spec = if diff > 0.0 { n.dot(h).max(0.0).powf(m.shininess) * m.specular } else { 0.0 };
    (diff, spec)
}

fn combine(scene: &Scene, albedo: Vec3, diff: f32, spec: f32, shadow: f32) -> Vec3 {
    albedo.mul_elem(scene.ambient) + (albedo * diff + v3(spec, spec, spec)).mul_elem(scene.light) * shadow
}

impl Shader for Lit<'_> {
    fn shade(&self, f: &Fragment) -> Option<Vec3> {
        let a = f.attrs;
        let p = v3(a[A_POS], a[A_POS + 1], a[A_POS + 2]);
        let uv = v2(a[A_UV], a[A_UV + 1]);
        let mut albedo = self.material.albedo;
        if let Some(t) = &self.material.texture {
            let ddx = v2(f.ddx[A_UV], f.ddx[A_UV + 1]);
            let ddy = v2(f.ddy[A_UV], f.ddy[A_UV + 1]);
            albedo = albedo.mul_elem(t.sample(uv, ddx, ddy, self.filter));
        }
        let mut n = match self.shading {
            // The face normal from how the position changes across the screen.
            Shading::Flat => {
                let dpx = v3(f.ddx[A_POS], f.ddx[A_POS + 1], f.ddx[A_POS + 2]);
                let dpy = v3(f.ddy[A_POS], f.ddy[A_POS + 1], f.ddy[A_POS + 2]);
                dpy.cross(dpx).normalize()
            }
            _ => v3(a[A_NRM], a[A_NRM + 1], a[A_NRM + 2]).normalize(),
        };
        if !f.front_facing {
            n = -n;
        }
        let slope = 1.0 - n.dot(self.scene.light_dir.normalize()).abs();
        let shadow = self.shadow.map_or(1.0, |s| s.visibility_at(p, n, slope));
        // Gouraud: the lighting terms were computed at the vertices and interpolated; texture and
        // shadow are still applied per pixel.
        let (diff, spec) = match self.shading {
            Shading::Gouraud => (a[A_GOURAUD], a[A_GOURAUD + 1]),
            _ => blinn_phong(self.scene, self.material, self.eye, p, n),
        };
        let mut c = combine(self.scene, albedo, diff, spec, shadow);
        if self.wireframe {
            // Distance to the nearest edge in pixels: each barycentric coordinate divided by how
            // fast it changes across the screen. Darken within about a pixel of an edge.
            let mut d = f32::MAX;
            for k in 0..3 {
                let b = a[A_BARY + k];
                let g = (f.ddx[A_BARY + k].powi(2) + f.ddy[A_BARY + k].powi(2)).sqrt().max(1e-9);
                d = d.min(b / g);
            }
            let t = (d - 0.5).clamp(0.0, 1.0);
            c = c * (0.25 + 0.75 * t);
        }
        if !(c.x.is_finite() && c.y.is_finite() && c.z.is_finite()) {
            c = Vec3::ZERO;
        }
        Some(c)
    }
}

/// Renders `scene` from `camera`; the result is linear light at the requested size.
pub fn render(scene: &Scene, camera: &Camera, opts: &Options) -> (Target, Stats) {
    let k = opts.ssaa.max(1);
    let (w, h) = (opts.width * k, opts.height * k);
    let mut stats = Stats::default();
    let aspect = opts.width as f32 / opts.height as f32;
    let vp = camera.view_proj(aspect);
    // Shadow map: an orthographic view along the light that covers the whole scene.
    let shadow = if opts.shadow_size > 0 {
        let t = Instant::now();
        let s = shadow_map(scene, opts.shadow_size, opts.threads);
        stats.shadow_ms = t.elapsed().as_secs_f64() * 1e3;
        Some(s)
    } else {
        None
    };
    let t = Instant::now();
    let mut target = Target::filled(w, h, scene.background);
    // Vertex work per object, then one parallel draw of every object (see `draw_batches`).
    let mut prepared: Vec<(Vec<Vertex>, Lit)> = Vec::with_capacity(scene.objects.len());
    for obj in &scene.objects {
        let normal_m = obj.transform.inverse().map_or(obj.transform, |m| m.transpose());
        let mvp = vp * obj.transform;
        let lit = Lit {
            scene,
            material: &obj.material,
            eye: camera.eye,
            shading: opts.shading,
            filter: opts.filter,
            shadow: shadow.as_ref(),
            wireframe: opts.wireframe,
        };
        let mesh = &obj.mesh;
        // Vertex work in parallel chunks of triangles, joined in order.
        let shade_tri = |tri: &[u32; 3], out: &mut Vec<Vertex>| {
            for (corner, &i) in tri.iter().enumerate() {
                let i = i as usize;
                let p = mesh.positions[i];
                let wp = obj.transform.transform_point(p);
                let n = normal_m.transform_dir(*mesh.normals.get(i).unwrap_or(&v3(0.0, 1.0, 0.0))).normalize();
                let uv = *mesh.uvs.get(i).unwrap_or(&v2(0.0, 0.0));
                let mut attrs = [0f32; MAX_ATTRS];
                attrs[A_POS..A_POS + 3].copy_from_slice(&[wp.x, wp.y, wp.z]);
                attrs[A_NRM..A_NRM + 3].copy_from_slice(&[n.x, n.y, n.z]);
                attrs[A_UV..A_UV + 2].copy_from_slice(&[uv.x, uv.y]);
                if opts.shading == Shading::Gouraud {
                    let (d, s) = blinn_phong(scene, &obj.material, camera.eye, wp, n);
                    attrs[A_GOURAUD] = d;
                    attrs[A_GOURAUD + 1] = s;
                }
                attrs[A_BARY + corner] = 1.0;
                out.push(Vertex { pos: mvp * p.extend(1.0), attrs });
            }
        };
        let verts = par_chunks(&mesh.triangles, opts.threads, shade_tri);
        stats.triangles += mesh.triangles.len();
        prepared.push((verts, lit));
    }
    let state = State { cull: Cull::Back, ..State::default() };
    let batches: Vec<Batch> = prepared.iter().map(|(v, lit)| Batch { verts: v, nattr: NATTR, state, shader: lit }).collect();
    draw_batches(&mut target, &batches, opts.threads);
    stats.main_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let out = if k > 1 { downsample(&target, k) } else { target };
    stats.resolve_ms = t.elapsed().as_secs_f64() * 1e3;
    (out, stats)
}

fn shadow_map(scene: &Scene, size: usize, threads: usize) -> ShadowMap {
    // Fit an orthographic box around every object's bounds as seen from the light.
    let dir = scene.light_dir.normalize();
    let up = if dir.y.abs() > 0.99 { v3(0.0, 0.0, 1.0) } else { v3(0.0, 1.0, 0.0) };
    let view = Mat4::look_at(dir * 50.0, Vec3::ZERO, up);
    let (mut lo, mut hi) = (v3(f32::MAX, f32::MAX, f32::MAX), v3(f32::MIN, f32::MIN, f32::MIN));
    for o in &scene.objects {
        let (a, b) = o.mesh.bounds();
        for c in 0..8 {
            let corner =
                v3(if c & 1 == 0 { a.x } else { b.x }, if c & 2 == 0 { a.y } else { b.y }, if c & 4 == 0 { a.z } else { b.z });
            let p = view.transform_point(o.transform.transform_point(corner));
            lo = v3(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = v3(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
    }
    let proj = Mat4::orthographic(lo.x, hi.x, lo.y, hi.y, -hi.z - 1.0, -lo.z + 1.0);
    let light_vp = proj * view;
    let mut t = Target::depth_only(size, size);
    let verts: Vec<Vec<Vertex>> = scene
        .objects
        .iter()
        .map(|o| {
            let m = light_vp * o.transform;
            par_chunks(&o.mesh.triangles, threads, |tri: &[u32; 3], out: &mut Vec<Vertex>| {
                out.extend(
                    tri.iter().map(|&i| Vertex { pos: m * o.mesh.positions[i as usize].extend(1.0), attrs: [0.0; MAX_ATTRS] }),
                )
            })
        })
        .collect();
    // Both faces cast shadows (the floor plane is single-sided).
    let state = State { cull: Cull::None, color_write: false, ..State::default() };
    let batches: Vec<Batch> = verts.iter().map(|v| Batch { verts: v, nattr: 0, state, shader: &DepthOnly }).collect();
    draw_batches(&mut t, &batches, threads);
    ShadowMap { size, depth: t.depth, light_vp, texel: (hi.x - lo.x).max(hi.y - lo.y) / size as f32 }
}

/// Maps each triangle to its three vertices on up to `threads` threads, keeping the order.
fn par_chunks<F>(tris: &[[u32; 3]], threads: usize, f: F) -> Vec<Vertex>
where
    F: Fn(&[u32; 3], &mut Vec<Vertex>) + Sync,
{
    let chunk = tris.len().div_ceil(threads.max(1)).max(2048);
    if threads <= 1 || tris.len() <= chunk {
        let mut out = Vec::with_capacity(tris.len() * 3);
        tris.iter().for_each(|t| f(t, &mut out));
        return out;
    }
    let chunks: Vec<&[[u32; 3]]> = tris.chunks(chunk).collect();
    let mut parts: Vec<Vec<Vertex>> = vec![Vec::new(); chunks.len()];
    crate::raster::pool().scope(|s| {
        for (slot, c) in parts.iter_mut().zip(&chunks) {
            let f = &f;
            s.spawn(move || {
                let mut out = Vec::with_capacity(c.len() * 3);
                c.iter().for_each(|t| f(t, &mut out));
                *slot = out;
            });
        }
    });
    parts.concat()
}

fn downsample(t: &Target, k: usize) -> Target {
    let (w, h) = (t.w / k, t.h / k);
    let mut out = Target::new(w, h);
    let inv = 1.0 / (k * k) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut sum = Vec3::ZERO;
            for dy in 0..k {
                for dx in 0..k {
                    sum = sum + t.color[(y * k + dy) * t.w + x * k + dx];
                }
            }
            out.color[y * w + x] = sum * inv;
        }
    }
    out
}

/// The colour buffer as 8-bit sRGB RGB.
pub fn to_srgb8(t: &Target) -> Vec<u8> {
    let mut out = Vec::with_capacity(t.w * t.h * 3);
    for c in &t.color {
        out.extend_from_slice(&[linear_to_srgb8(c.x), linear_to_srgb8(c.y), linear_to_srgb8(c.z)]);
    }
    out
}

/// The colour buffer as a PNG (through the lumen image library).
pub fn to_png(t: &Target) -> Vec<u8> {
    let img = lumen::image::Image::new8(t.w as u32, t.h as u32, lumen::image::Color::Rgb, to_srgb8(t)).expect("size matches");
    lumen::png::encode(&img, 6)
}
