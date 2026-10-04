//! The built-in demo scene and loading an OBJ model as a scene.

use std::path::Path;
use std::sync::Arc;

use crate::math::{v3, Mat4, Vec3};
use crate::mesh::{load_obj, Mesh};
use crate::scene::{Camera, Material, Object, Scene};
use crate::texture::Texture;

/// A checkerboard floor, a shiny torus knot, a textured sphere and two cubes, lit from above
/// and to the side so the shadows show.
pub fn demo_scene() -> Scene {
    let checker = Arc::new(Texture::checker(512, 16, v3(0.85, 0.85, 0.85), v3(0.08, 0.09, 0.12)));
    let stripes = Arc::new(Texture::from_linear(
        256,
        256,
        (0..256 * 256).map(|i| if ((i % 256) / 16) % 2 == 0 { v3(0.9, 0.55, 0.1) } else { v3(0.95, 0.9, 0.8) }).collect(),
    ));
    let floor = Object {
        mesh: Arc::new(Mesh::plane(24.0, 12.0)),
        material: Material { albedo: v3(1.0, 1.0, 1.0), texture: Some(checker), specular: 0.1, shininess: 16.0 },
        transform: Mat4::IDENTITY,
    };
    let knot = Object {
        mesh: Arc::new(Mesh::torus_knot(2.0, 3.0, 0.32, 600, 24)),
        material: Material { albedo: v3(0.75, 0.08, 0.1), texture: None, specular: 0.8, shininess: 96.0 },
        transform: Mat4::translate(v3(0.0, 2.1, 0.0)) * Mat4::rotate(v3(1.0, 0.0, 0.0), -1.2),
    };
    let ball = Object {
        mesh: Arc::new(Mesh::sphere(1.0, 64, 32)),
        material: Material { albedo: v3(1.0, 1.0, 1.0), texture: Some(stripes), specular: 0.5, shininess: 64.0 },
        transform: Mat4::translate(v3(3.6, 1.0, 1.6)),
    };
    let cube = |pos: Vec3, angle: f32, colour: Vec3| Object {
        mesh: Arc::new(Mesh::cube()),
        material: Material { albedo: colour, texture: None, specular: 0.2, shininess: 32.0 },
        transform: Mat4::translate(pos) * Mat4::rotate(v3(0.0, 1.0, 0.0), angle) * Mat4::scale(v3(1.4, 1.4, 1.4)),
    };
    Scene {
        objects: vec![
            floor,
            knot,
            ball,
            cube(v3(-3.6, 0.7, 1.4), 0.5, v3(0.15, 0.45, 0.85)),
            cube(v3(-1.8, 0.7, 3.4), -0.3, v3(0.2, 0.7, 0.35)),
        ],
        light_dir: v3(-0.5, 1.0, 0.45),
        light: v3(1.0, 0.97, 0.9),
        ambient: v3(0.12, 0.13, 0.16),
        background: v3(0.02, 0.025, 0.04),
    }
}

pub fn demo_camera() -> Camera {
    Camera { eye: v3(6.5, 4.5, 9.0), target: v3(0.0, 1.4, 0.0), up: v3(0.0, 1.0, 0.0), fovy: 0.75, near: 0.1, far: 100.0 }
}

/// An OBJ model on a floor, scaled to about 4 units and set on it, with a camera framing it.
pub fn obj_scene(path: &Path) -> Result<(Scene, Camera), String> {
    let parts = load_obj(path)?;
    let (mut lo, mut hi) = (v3(f32::MAX, f32::MAX, f32::MAX), v3(f32::MIN, f32::MIN, f32::MIN));
    for (m, _) in &parts {
        let (a, b) = m.bounds();
        lo = v3(lo.x.min(a.x), lo.y.min(a.y), lo.z.min(a.z));
        hi = v3(hi.x.max(b.x), hi.y.max(b.y), hi.z.max(b.z));
    }
    let size = (hi - lo).max_elem().max(1e-6);
    let k = 4.0 / size;
    let centre = (lo + hi) * 0.5;
    let place = Mat4::scale(v3(k, k, k)) * Mat4::translate(v3(-centre.x, -lo.y, -centre.z));
    let mut scene = demo_scene();
    scene.objects.truncate(1);
    for (mesh, m) in parts {
        let texture = match &m.texture {
            Some(p) => Some(Arc::new(Texture::load(p)?)),
            None => None,
        };
        let albedo = if texture.is_some() { v3(1.0, 1.0, 1.0) } else { m.diffuse };
        let specular = m.specular.max_elem().clamp(0.0, 1.0);
        scene.objects.push(Object {
            mesh: Arc::new(mesh),
            material: Material { albedo, texture, specular, shininess: m.shininess.max(1.0) },
            transform: place,
        });
    }
    let h = (hi.y - lo.y) * k;
    Ok((
        scene,
        Camera {
            eye: v3(5.0, h * 0.6 + 2.0, 7.0),
            target: v3(0.0, h * 0.45, 0.0),
            up: v3(0.0, 1.0, 0.0),
            fovy: 0.75,
            near: 0.1,
            far: 100.0,
        },
    ))
}
