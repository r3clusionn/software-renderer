//! Whole scenes: lighting, shadows and OBJ models.

use std::sync::Arc;

use swr::math::{v3, Mat4};
use swr::mesh::Mesh;
use swr::scene::{render, Camera, Material, Object, Options, Scene, Shading};

fn floor_only() -> Scene {
    Scene {
        objects: vec![Object {
            mesh: Arc::new(Mesh::plane(20.0, 1.0)),
            material: Material { specular: 0.0, ..Material::default() },
            transform: Mat4::IDENTITY,
        }],
        light_dir: v3(-0.4, 1.0, 0.3),
        light: v3(1.0, 1.0, 1.0),
        ambient: v3(0.1, 0.1, 0.1),
        background: v3(0.0, 0.0, 0.0),
    }
}

fn camera() -> Camera {
    Camera { eye: v3(0.0, 5.0, 8.0), target: v3(0.0, 0.0, 0.0), up: v3(0.0, 1.0, 0.0), fovy: 0.9, near: 0.1, far: 100.0 }
}

fn opts() -> Options {
    Options { width: 200, height: 120, threads: 4, ..Options::default() }
}

#[test]
fn a_lit_floor_does_not_shadow_itself() {
    // Without specular light every floor pixel gets the same colour; any darker pixel would be
    // shadow acne.
    let (t, _) = render(&floor_only(), &camera(), &opts());
    let floor: Vec<_> = t.color.iter().filter(|c| c.x > 0.0).collect();
    assert!(floor.len() > 10_000);
    let first = floor[0];
    assert!(floor.iter().all(|c| (c.x - first.x).abs() < 1e-5), "the floor has darker spots");
}

#[test]
fn objects_cast_shadows_on_the_floor() {
    let mut s = floor_only();
    // From the side, so seen from above the shadow falls beside the slab, not under it.
    s.light_dir = v3(0.7, 1.0, 0.0);
    s.objects.push(Object {
        mesh: Arc::new(Mesh::cube()),
        material: Material::default(),
        transform: Mat4::translate(v3(0.0, 2.0, 0.0)) * Mat4::scale(v3(2.0, 0.2, 2.0)),
    });
    let cam =
        Camera { eye: v3(0.0, 12.0, 0.01), target: v3(0.0, 0.0, 0.0), up: v3(0.0, 0.0, -1.0), fovy: 0.9, near: 0.1, far: 100.0 };
    let mut bare = floor_only();
    bare.light_dir = s.light_dir;
    let (lit, _) = render(&bare, &cam, &opts());
    let (shadowed, _) = render(&s, &cam, &opts());
    // A pixel near the edge of the view sees floor in both images and the same light.
    let edge = 5 * 200 + 5;
    assert!((lit.color[edge].x - shadowed.color[edge].x).abs() < 1e-5);
    // The same scene without shadows: with them, part of the floor around the slab is darker.
    let shadow_without = Options { shadow_size: 0, ..opts() };
    let (no_shadow, _) = render(&s, &cam, &shadow_without);
    let darker = lit.color.iter().zip(&shadowed.color).zip(&no_shadow.color).filter(|((_, a), b)| a.x + 0.05 < b.x).count();
    assert!(darker > 100, "turning shadows on darkened {darker} pixels");
}

#[test]
fn shading_modes_all_render() {
    let mut s = floor_only();
    s.objects.push(Object {
        mesh: Arc::new(Mesh::sphere(1.0, 24, 12)),
        material: Material::default(),
        transform: Mat4::translate(v3(0.0, 1.0, 0.0)),
    });
    for shading in [Shading::Flat, Shading::Gouraud, Shading::Phong] {
        for wireframe in [false, true] {
            let (t, st) = render(&s, &camera(), &Options { shading, wireframe, ssaa: 2, ..opts() });
            assert_eq!((t.w, t.h), (200, 120));
            assert_eq!(st.triangles, 2 + 24 * 12 * 2 - 48);
            assert!(t.color.iter().all(|c| c.x.is_finite() && c.y.is_finite() && c.z.is_finite()));
        }
    }
}

#[test]
fn an_obj_model_renders() {
    let d = std::env::temp_dir().join(format!("swr-scene-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    // A tetrahedron.
    std::fs::write(d.join("t.obj"), "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n").unwrap();
    let (scene, cam) = swr::demo::obj_scene(&d.join("t.obj")).unwrap();
    let (t, st) = render(&scene, &cam, &opts());
    assert_eq!(st.triangles, 2 + 4);
    let centre = t.color[60 * 200 + 100];
    assert!(centre.x > 0.0, "the model is in the middle of the view");
    std::fs::remove_dir_all(&d).unwrap();
}
