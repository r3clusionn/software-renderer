//! The rasterizer against independent answers: exact coverage, a ray caster, analytic
//! perspective, and identical output on any number of threads.

use swr::math::{v3, v4, Mat4, Vec3, Vec4};
use swr::raster::{draw, Blend, Cull, Fragment, Shader, State, Target, Vertex, MAX_ATTRS};

struct Count;
impl Shader for Count {
    fn shade(&self, _: &Fragment) -> Option<Vec3> {
        Some(v3(1.0, 0.0, 0.0))
    }
}

struct Id;
impl Shader for Id {
    fn shade(&self, f: &Fragment) -> Option<Vec3> {
        Some(v3(f.prim as f32, f.attrs[0], f.attrs[1]))
    }
}

fn vert(p: Vec4) -> Vertex {
    Vertex { pos: p, attrs: [0.0; MAX_ATTRS] }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A mesh that tiles the screen with no gaps or overlaps must cover every pixel exactly once:
/// random points in clip space, joined into a grid, including vertices exactly on pixel centres
/// and edges exactly along pixel rows (where only the fill rule decides).
#[test]
fn shared_edges_cover_every_pixel_exactly_once() {
    let mut rng = Rng(7);
    for (n, snap) in [(6, false), (13, true), (40, false)] {
        let (w, h) = (97usize, 61usize);
        // Grid points from -1.2 to 1.2 (beyond the screen), jittered by up to 0.15 of a cell,
        // little enough that every cell stays convex (a folded quad would overlap itself).
        let pts: Vec<Vec<(f32, f32)>> = (0..=n)
            .map(|j| {
                (0..=n)
                    .map(|i| {
                        let edge = i == 0 || j == 0 || i == n || j == n;
                        let jx = if edge { 0.0 } else { (rng.f() - 0.5) * 0.3 };
                        let jy = if edge { 0.0 } else { (rng.f() - 0.5) * 0.3 };
                        let mut x = -1.2 + 2.4 * (i as f32 + jx) / n as f32;
                        let mut y = -1.2 + 2.4 * (j as f32 + jy) / n as f32;
                        if snap {
                            // Put points exactly on pixel centres or corners.
                            x = ((x + 1.0) * w as f32 / 2.0).round() / (w as f32 / 2.0) - 1.0;
                            y = ((y + 1.0) * h as f32 / 2.0 * 2.0).round() / (h as f32) - 1.0;
                        }
                        (x, y)
                    })
                    .collect()
            })
            .collect();
        let mut verts = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let p = |a: usize, b: usize| vert(v4(pts[b][a].0, pts[b][a].1, 0.5, 1.0));
                verts.extend([p(i, j), p(i + 1, j), p(i + 1, j + 1)]);
                verts.extend([p(i, j), p(i + 1, j + 1), p(i, j + 1)]);
            }
        }
        let mut t = Target::new(w, h);
        let st = State { cull: Cull::None, depth_test: false, depth_write: false, blend: Blend::Add, ..State::default() };
        draw(&mut t, &verts, 0, &st, &Count, 4);
        let wrong: Vec<(usize, f32)> =
            t.color.iter().enumerate().filter(|(_, c)| c.x != 1.0).map(|(i, c)| (i, c.x)).take(5).collect();
        assert!(wrong.is_empty(), "grid {n}: pixels covered other than once: {wrong:?}");
    }
}

/// Möller-Trumbore: distance along the ray to the triangle, if it is hit.
fn hit(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32)> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = s.dot(p) * inv;
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if u < 0.0 || v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    // Distance to the nearest edge in barycentric terms, to skip pixels on edges.
    let margin = u.min(v).min(1.0 - u - v);
    (t > 0.0).then_some((t, margin))
}

/// The rasterizer's choice of triangle per pixel (depth test, clipping, projection) must agree
/// with casting a ray through each pixel centre, except where the centre is within a hair of an
/// edge or two triangles are at nearly the same depth.
#[test]
fn agrees_with_a_ray_caster() {
    let mut rng = Rng(99);
    let (w, h) = (160usize, 120usize);
    let eye = v3(0.0, 0.0, 5.0);
    let near = 0.5;
    let vp = Mat4::perspective(1.0, w as f32 / h as f32, near, 50.0) * Mat4::look_at(eye, v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0));
    let inv_vp = vp.inverse().unwrap();
    let mut tris = Vec::new();
    for _ in 0..60 {
        let c = v3(rng.f() * 8.0 - 4.0, rng.f() * 6.0 - 3.0, rng.f() * 8.0 - 4.0);
        let mut r = || v3(rng.f() - 0.5, rng.f() - 0.5, rng.f() - 0.5) * 3.0;
        tris.push([c + r(), c + r(), c + r()]);
    }
    // Some triangles that cross the near plane and pass behind the camera.
    tris.push([v3(-2.0, -0.5, 6.0), v3(2.0, -0.5, 6.0), v3(0.0, 0.5, 2.0)]);
    tris.push([v3(-1.0, 1.0, 4.9), v3(1.0, 1.2, 4.2), v3(0.0, -1.5, 3.0)]);
    let verts: Vec<Vertex> = tris.iter().flatten().map(|p| vert(vp * p.extend(1.0))).collect();
    let mut t = Target::new(w, h).with_ids();
    draw(&mut t, &verts, 0, &State { cull: Cull::None, ..State::default() }, &Count, 3);
    let ids = t.ids.unwrap();
    let (mut checked, mut mismatches) = (0, Vec::new());
    for y in 0..h {
        for x in 0..w {
            // The ray through the pixel centre: unproject two depths.
            let nx = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
            let ny = 1.0 - (y as f32 + 0.5) / h as f32 * 2.0;
            let p0 = inv_vp.transform_point(v3(nx, ny, 0.0));
            let p1 = inv_vp.transform_point(v3(nx, ny, 1.0));
            let d = (p1 - p0).normalize();
            let mut hits: Vec<(f32, f32, u32)> = tris
                .iter()
                .enumerate()
                .filter_map(|(i, t)| hit(p0, d, t[0], t[1], t[2]).map(|(dist, m)| (dist, m, i as u32)))
                .collect();
            hits.sort_by(|a, b| a.0.total_cmp(&b.0));
            let expect = hits.first().map_or(u32::MAX, |h| h.2);
            // Skip ambiguous pixels: centre near an edge, or the two nearest within 0.1%.
            if hits.first().is_some_and(|h| h.1 < 1e-3) || (hits.len() > 1 && (hits[1].0 - hits[0].0) / hits[0].0 < 1e-3) {
                continue;
            }
            checked += 1;
            if ids[y * w + x] != expect {
                mismatches.push((x, y, ids[y * w + x], expect));
            }
        }
    }
    assert!(checked > (w * h) * 9 / 10, "too many pixels skipped: checked {checked}");
    assert!(
        mismatches.is_empty(),
        "{} of {checked} pixels differ: {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(8)]
    );
}

/// Texture coordinates across a plane seen at a grazing angle must match where the ray through
/// each pixel meets the plane; screen-space (affine) interpolation must not.
#[test]
fn perspective_correct_attributes() {
    let (w, h) = (128usize, 96usize);
    let eye = v3(0.0, 1.0, 3.0);
    let vp = Mat4::perspective(1.2, w as f32 / h as f32, 0.1, 100.0) * Mat4::look_at(eye, v3(0.0, 0.0, -6.0), v3(0.0, 1.0, 0.0));
    let inv = vp.inverse().unwrap();
    // A floor quad from z = 2 to z = -40, u = x and v = z as attributes.
    let corners = [v3(-6.0, 0.0, 2.0), v3(6.0, 0.0, 2.0), v3(6.0, 0.0, -40.0), v3(-6.0, 0.0, -40.0)];
    let v = |p: Vec3| {
        let mut a = [0.0; MAX_ATTRS];
        a[0] = p.x;
        a[1] = p.z;
        Vertex { pos: vp * p.extend(1.0), attrs: a }
    };
    let verts = vec![v(corners[0]), v(corners[1]), v(corners[2]), v(corners[0]), v(corners[2]), v(corners[3])];
    let measure = |affine: bool| {
        let mut t = Target::new(w, h);
        draw(&mut t, &verts, 2, &State { cull: Cull::None, affine, ..State::default() }, &Id, 2);
        let mut worst: f32 = 0.0;
        for y in 0..h {
            for x in 0..w {
                if t.depth[y * w + x] >= 1.0 {
                    continue;
                }
                let nx = (x as f32 + 0.5) / w as f32 * 2.0 - 1.0;
                let ny = 1.0 - (y as f32 + 0.5) / h as f32 * 2.0;
                let p0 = inv.transform_point(v3(nx, ny, 0.0));
                let p1 = inv.transform_point(v3(nx, ny, 1.0));
                let d = p1 - p0;
                let s = -p0.y / d.y;
                let hitp = p0 + d * s;
                let c = t.color[y * w + x];
                worst = worst.max((c.y - hitp.x).abs()).max((c.z - hitp.z).abs());
            }
        }
        worst
    };
    let correct = measure(false);
    let affine = measure(true);
    assert!(correct < 0.02, "perspective-correct error {correct}");
    assert!(affine > 1.0, "affine interpolation should be visibly wrong here: {affine}");
}

/// The same scene on 1 and on 7 threads gives bit-identical buffers.
#[test]
fn thread_count_does_not_change_the_image() {
    let mut rng = Rng(3);
    let mut verts = Vec::new();
    for _ in 0..3000 {
        let c = v3(rng.f() * 2.0 - 1.0, rng.f() * 2.0 - 1.0, rng.f());
        for _ in 0..3 {
            let mut a = [0.0; MAX_ATTRS];
            a[0] = rng.f();
            a[1] = rng.f();
            verts.push(Vertex { pos: v4(c.x + rng.f() * 0.3 - 0.15, c.y + rng.f() * 0.3 - 0.15, c.z, 1.0), attrs: a });
        }
    }
    let run = |threads: usize| {
        let mut t = Target::new(301, 177);
        draw(&mut t, &verts, 2, &State { cull: Cull::None, ..State::default() }, &Id, threads);
        (t.color, t.depth)
    };
    assert!(run(1) == run(7));
}

#[test]
fn back_faces_are_culled() {
    let ccw = [vert(v4(-0.5, -0.5, 0.5, 1.0)), vert(v4(0.5, -0.5, 0.5, 1.0)), vert(v4(0.0, 0.5, 0.5, 1.0))];
    let cw = [ccw[0], ccw[2], ccw[1]];
    let covered = |v: &[Vertex]| {
        let mut t = Target::new(20, 20);
        draw(&mut t, v, 0, &State::default(), &Count, 1);
        t.color.iter().filter(|c| c.x > 0.0).count()
    };
    assert_eq!(covered(&ccw), 50, "half of the 10 x 10 pixel box");
    assert_eq!(covered(&cw), 0);
}
