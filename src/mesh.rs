//! Triangle meshes: procedural shapes and a Wavefront OBJ loader.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::path::Path;

use crate::math::{v2, v3, Vec2, Vec3};

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    /// Counter-clockwise when seen from the front.
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = v3(f32::MAX, f32::MAX, f32::MAX);
        let mut hi = v3(f32::MIN, f32::MIN, f32::MIN);
        for p in &self.positions {
            lo = v3(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = v3(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        (lo, hi)
    }

    /// Smooth normals: each vertex gets the area-weighted sum of its faces' normals.
    pub fn compute_normals(&mut self) {
        let mut n = vec![Vec3::ZERO; self.positions.len()];
        for t in &self.triangles {
            let [a, b, c] = t.map(|i| self.positions[i as usize]);
            let face = (b - a).cross(c - a);
            for &i in t {
                n[i as usize] = n[i as usize] + face;
            }
        }
        self.normals = n.into_iter().map(|v| v.normalize()).collect();
    }

    /// A square in the XZ plane facing up, `size` across, texture repeated `repeat` times.
    pub fn plane(size: f32, repeat: f32) -> Mesh {
        let h = size / 2.0;
        Mesh {
            positions: vec![v3(-h, 0.0, -h), v3(-h, 0.0, h), v3(h, 0.0, h), v3(h, 0.0, -h)],
            normals: vec![v3(0.0, 1.0, 0.0); 4],
            uvs: vec![v2(0.0, 0.0), v2(0.0, repeat), v2(repeat, repeat), v2(repeat, 0.0)],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        }
    }

    /// A unit cube centred on the origin, with its own normals and texture square per face.
    pub fn cube() -> Mesh {
        let mut m = Mesh::default();
        let faces = [
            (v3(1.0, 0.0, 0.0), v3(0.0, 0.0, -1.0), v3(0.0, 1.0, 0.0)),
            (v3(-1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), v3(0.0, 1.0, 0.0)),
            (v3(0.0, 1.0, 0.0), v3(1.0, 0.0, 0.0), v3(0.0, 0.0, -1.0)),
            (v3(0.0, -1.0, 0.0), v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0)),
            (v3(0.0, 0.0, 1.0), v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)),
            (v3(0.0, 0.0, -1.0), v3(-1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0)),
        ];
        for (n, u, v) in faces {
            let base = m.positions.len() as u32;
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                m.positions.push((n + u * su + v * sv) * 0.5);
                m.normals.push(n);
                m.uvs.push(v2((su + 1.0) / 2.0, (1.0 - sv) / 2.0));
            }
            m.triangles.push([base, base + 1, base + 2]);
            m.triangles.push([base, base + 2, base + 3]);
        }
        m
    }

    /// A UV sphere.
    pub fn sphere(radius: f32, segments: usize, rings: usize) -> Mesh {
        let mut m = Mesh::default();
        for r in 0..=rings {
            let phi = PI * r as f32 / rings as f32;
            for s in 0..=segments {
                let theta = TAU * s as f32 / segments as f32;
                let n = v3(phi.sin() * theta.cos(), phi.cos(), -phi.sin() * theta.sin());
                m.positions.push(n * radius);
                m.normals.push(n);
                m.uvs.push(v2(s as f32 / segments as f32, r as f32 / rings as f32));
            }
        }
        let row = segments as u32 + 1;
        for r in 0..rings as u32 {
            for s in 0..segments as u32 {
                let (a, b) = (r * row + s, (r + 1) * row + s);
                // At the poles one of the two triangles has no area; leave it out.
                if r + 1 < rings as u32 {
                    m.triangles.push([a, b, b + 1]);
                }
                if r > 0 {
                    m.triangles.push([a, b + 1, a + 1]);
                }
            }
        }
        m
    }

    /// A (p, q) torus knot: a tube of radius `tube` around the knot curve.
    pub fn torus_knot(p: f32, q: f32, tube: f32, segments: usize, sides: usize) -> Mesh {
        let curve = |t: f32| {
            let r = 2.0 + (q * t).cos();
            v3(r * (p * t).cos(), r * (p * t).sin(), -(q * t).sin()) * 0.5
        };
        let mut m = Mesh::default();
        for i in 0..=segments {
            let t = TAU * i as f32 / segments as f32;
            let c = curve(t);
            let tan = (curve(t + 1e-3) - curve(t - 1e-3)).normalize();
            let side = tan.cross(c.normalize()).normalize();
            let up = side.cross(tan);
            for j in 0..=sides {
                let a = TAU * j as f32 / sides as f32;
                let n = (side * a.cos() + up * a.sin()).normalize();
                m.positions.push(c + n * tube);
                m.normals.push(n);
                m.uvs.push(v2(i as f32 / segments as f32 * 8.0, j as f32 / sides as f32));
            }
        }
        let row = sides as u32 + 1;
        for i in 0..segments as u32 {
            for j in 0..sides as u32 {
                let (a, b) = (i * row + j, (i + 1) * row + j);
                m.triangles.push([a, b, b + 1]);
                m.triangles.push([a, b + 1, a + 1]);
            }
        }
        // The tube's winding depends on the frame; make it face outwards.
        let t0 = m.triangles[0].map(|i| m.positions[i as usize]);
        let face = (t0[1] - t0[0]).cross(t0[2] - t0[0]);
        if face.dot(m.normals[m.triangles[0][0] as usize]) < 0.0 {
            for t in &mut m.triangles {
                t.swap(1, 2);
            }
        }
        m
    }
}

/// A material from an OBJ's MTL file.
#[derive(Clone, Debug, Default)]
pub struct ObjMaterial {
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
    pub texture: Option<std::path::PathBuf>,
}

type Group = (String, Mesh, HashMap<(usize, usize, usize), u32>);

/// Loads an OBJ file: one mesh per material used. Polygons are split into fans; negative
/// indices count back from the end; missing normals are computed.
pub fn load_obj(path: &Path) -> Result<Vec<(Mesh, ObjMaterial)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let (mut pos, mut tex, mut nrm) = (Vec::new(), Vec::new(), Vec::new());
    let mut materials: HashMap<String, ObjMaterial> = HashMap::new();
    // Per material: its name, the mesh so far, and the vertex index of each (v, vt, vn) seen.
    let mut groups: Vec<Group> = Vec::new();
    let mut current = String::new();
    for (lineno, line) in text.lines().enumerate() {
        let err = |m: &str| format!("{}:{}: {m}", path.display(), lineno + 1);
        let mut it = line.split_whitespace();
        let Some(tag) = it.next() else { continue };
        let nums = |it: std::str::SplitWhitespace| -> Result<Vec<f32>, String> {
            it.map(|s| s.parse::<f32>().map_err(|_| err(&format!("bad number '{s}'")))).collect()
        };
        match tag {
            "v" => {
                let v = nums(it)?;
                if v.len() < 3 {
                    return Err(err("v needs x y z"));
                }
                pos.push(v3(v[0], v[1], v[2]));
            }
            "vt" => {
                let v = nums(it)?;
                // OBJ's v runs up the image; textures here are addressed top down.
                tex.push(v2(*v.first().unwrap_or(&0.0), 1.0 - *v.get(1).unwrap_or(&0.0)));
            }
            "vn" => {
                let v = nums(it)?;
                if v.len() < 3 {
                    return Err(err("vn needs x y z"));
                }
                nrm.push(v3(v[0], v[1], v[2]).normalize());
            }
            "usemtl" => current = it.next().unwrap_or("").to_string(),
            "mtllib" => {
                for f in it {
                    if let Ok(m) = std::fs::read_to_string(dir.join(f)) {
                        parse_mtl(&m, dir, &mut materials);
                    }
                }
            }
            "f" => {
                let resolve = |s: &str, n: usize| -> Result<usize, String> {
                    let i: i64 = s.parse().map_err(|_| err(&format!("bad index '{s}'")))?;
                    let r = if i < 0 { n as i64 + i } else { i - 1 };
                    if r < 0 || r >= n as i64 {
                        return Err(err(&format!("index {i} out of range")));
                    }
                    Ok(r as usize)
                };
                let gi = match groups.iter().position(|g| g.0 == current) {
                    Some(i) => i,
                    None => {
                        groups.push((current.clone(), Mesh::default(), HashMap::new()));
                        groups.len() - 1
                    }
                };
                let mut corner = Vec::new();
                for v in it {
                    let mut parts = v.split('/');
                    let p = resolve(parts.next().unwrap_or(""), pos.len())?;
                    let t = match parts.next() {
                        Some(s) if !s.is_empty() => resolve(s, tex.len())? + 1,
                        _ => 0,
                    };
                    let n = match parts.next() {
                        Some(s) if !s.is_empty() => resolve(s, nrm.len())? + 1,
                        _ => 0,
                    };
                    let (_, mesh, map) = &mut groups[gi];
                    let idx = *map.entry((p, t, n)).or_insert_with(|| {
                        mesh.positions.push(pos[p]);
                        mesh.uvs.push(if t > 0 { tex[t - 1] } else { v2(0.0, 0.0) });
                        mesh.normals.push(if n > 0 { nrm[n - 1] } else { Vec3::ZERO });
                        (mesh.positions.len() - 1) as u32
                    });
                    corner.push(idx);
                }
                if corner.len() < 3 {
                    return Err(err("a face needs three corners"));
                }
                for k in 1..corner.len() - 1 {
                    groups[gi].1.triangles.push([corner[0], corner[k], corner[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if groups.is_empty() {
        return Err(format!("{}: no faces", path.display()));
    }
    Ok(groups
        .into_iter()
        .map(|(name, mut mesh, _)| {
            if mesh.normals.contains(&Vec3::ZERO) {
                mesh.compute_normals();
            }
            let mat = materials.get(&name).cloned().unwrap_or(ObjMaterial {
                diffuse: v3(0.8, 0.8, 0.8),
                specular: v3(0.2, 0.2, 0.2),
                shininess: 32.0,
                texture: None,
            });
            (mesh, mat)
        })
        .collect())
}

fn parse_mtl(text: &str, dir: &Path, out: &mut HashMap<String, ObjMaterial>) {
    let mut name = String::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let Some(tag) = it.next() else { continue };
        let rgb = |it: std::str::SplitWhitespace| {
            let v: Vec<f32> = it.filter_map(|s| s.parse().ok()).collect();
            v3(*v.first().unwrap_or(&0.0), *v.get(1).unwrap_or(&0.0), *v.get(2).unwrap_or(&0.0))
        };
        match tag {
            "newmtl" => {
                name = it.next().unwrap_or("").to_string();
                out.insert(
                    name.clone(),
                    ObjMaterial { diffuse: v3(0.8, 0.8, 0.8), specular: Vec3::ZERO, shininess: 32.0, texture: None },
                );
            }
            "Kd" => {
                if let Some(m) = out.get_mut(&name) {
                    m.diffuse = rgb(it);
                }
            }
            "Ks" => {
                if let Some(m) = out.get_mut(&name) {
                    m.specular = rgb(it);
                }
            }
            "Ns" => {
                if let Some(m) = out.get_mut(&name) {
                    m.shininess = it.next().and_then(|s| s.parse().ok()).unwrap_or(32.0);
                }
            }
            "map_Kd" => {
                if let Some(m) = out.get_mut(&name) {
                    m.texture = it.last().map(|f| dir.join(f));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_meshes_have_outward_faces() {
        for (which, m) in
            [Mesh::cube(), Mesh::sphere(1.0, 16, 8), Mesh::torus_knot(2.0, 3.0, 0.3, 128, 12)].into_iter().enumerate()
        {
            for t in &m.triangles {
                let [a, b, c] = t.map(|i| m.positions[i as usize]);
                let face = (b - a).cross(c - a);
                let n = m.normals[t[0] as usize] + m.normals[t[1] as usize] + m.normals[t[2] as usize];
                if face.length() > 1e-9 {
                    assert!(face.dot(n) > 0.0, "mesh {which}: triangle {t:?} points inwards");
                }
            }
        }
    }

    #[test]
    fn obj_with_polygons_negative_indices_and_materials() {
        let d = std::env::temp_dir().join(format!("swr-obj-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("m.mtl"), "newmtl red\nKd 1 0 0\nNs 64\n").unwrap();
        std::fs::write(
            d.join("q.obj"),
            "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 1\nusemtl red\nf 1/1 2/1 3/2 4/2\nf -4 -2 -1\n",
        )
        .unwrap();
        let parts = load_obj(&d.join("q.obj")).unwrap();
        assert_eq!(parts.len(), 1);
        let (mesh, mat) = &parts[0];
        assert_eq!(mesh.triangles.len(), 3, "a quad is two triangles, plus one more");
        assert_eq!(mat.diffuse, v3(1.0, 0.0, 0.0));
        assert!(mesh.normals.iter().all(|n| (n.z - 1.0).abs() < 1e-6), "computed normals face +z");
        assert!(load_obj(&d.join("missing.obj")).is_err());
        std::fs::write(d.join("bad.obj"), "v 0 0 0\nf 1 2 3\n").unwrap();
        assert!(load_obj(&d.join("bad.obj")).unwrap_err().contains("out of range"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
