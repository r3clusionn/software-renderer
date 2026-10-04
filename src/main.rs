use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use swr::demo::{demo_camera, demo_scene, obj_scene};
use swr::math::v3;
use swr::scene::{render, to_png, Camera, Options, Scene, Shading};
use swr::texture::{linear_to_srgb8, Filter};

#[derive(Parser)]
#[command(
    name = "swr",
    version,
    about = "A CPU rasterizer: render the demo scene or an OBJ model to a PNG, view it, or benchmark it"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct Look {
    /// An OBJ model to show instead of the demo scene
    #[arg(long)]
    model: Option<PathBuf>,
    /// WIDTHxHEIGHT
    #[arg(long, default_value = "1280x720")]
    size: String,
    /// flat, gouraud or phong
    #[arg(long, default_value = "phong")]
    shading: String,
    /// nearest, bilinear or trilinear
    #[arg(long, default_value = "trilinear")]
    filter: String,
    /// Shadow map size (0 = no shadows)
    #[arg(long, default_value_t = 2048)]
    shadow: usize,
    /// Supersampling per axis (2 renders 4 samples per pixel)
    #[arg(long, default_value_t = 1)]
    ssaa: usize,
    #[arg(long)]
    wireframe: bool,
    /// Threads (default: all)
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Render one frame to a PNG
    Render {
        out: PathBuf,
        #[command(flatten)]
        look: Look,
    },
    /// Open a window: drag to orbit, wheel to zoom, 1-3 shading, F filter, S shadows, W wireframe
    View {
        #[command(flatten)]
        look: Look,
    },
    /// Time frames of the demo scene at each thread count
    Bench {
        #[command(flatten)]
        look: Look,
        #[arg(long, default_value_t = 20)]
        frames: usize,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("swr: {e}");
            ExitCode::from(2)
        }
    }
}

fn setup(l: &Look) -> Result<(Scene, Camera, Options), String> {
    let (w, h) = l
        .size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
        .ok_or("--size must look like 1280x720")?;
    let shading = match l.shading.as_str() {
        "flat" => Shading::Flat,
        "gouraud" => Shading::Gouraud,
        "phong" => Shading::Phong,
        s => return Err(format!("unknown shading '{s}'")),
    };
    let filter = match l.filter.as_str() {
        "nearest" => Filter::Nearest,
        "bilinear" => Filter::Bilinear,
        "trilinear" => Filter::Trilinear,
        s => return Err(format!("unknown filter '{s}'")),
    };
    let mut o = Options {
        width: w,
        height: h,
        shading,
        filter,
        shadow_size: l.shadow,
        ssaa: l.ssaa.clamp(1, 4),
        wireframe: l.wireframe,
        ..Options::default()
    };
    if let Some(t) = l.threads {
        o.threads = t.max(1);
    }
    let (scene, camera) = match &l.model {
        Some(p) => obj_scene(p)?,
        None => (demo_scene(), demo_camera()),
    };
    Ok((scene, camera, o))
}

fn run() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::Render { out, look } => {
            let (scene, camera, opts) = setup(&look)?;
            let t = Instant::now();
            let (target, s) = render(&scene, &camera, &opts);
            let took = t.elapsed();
            std::fs::write(&out, to_png(&target)).map_err(|e| format!("{}: {e}", out.display()))?;
            eprintln!(
                "{}: {}x{}, {} triangles, {:.1} ms (shadow map {:.1}, scene {:.1}, resolve {:.1}) on {} threads",
                out.display(),
                opts.width,
                opts.height,
                s.triangles,
                took.as_secs_f64() * 1e3,
                s.shadow_ms,
                s.main_ms,
                s.resolve_ms,
                opts.threads
            );
            Ok(())
        }
        Cmd::Bench { look, frames } => {
            let (scene, camera, opts) = setup(&look)?;
            let max = opts.threads;
            let mut counts: Vec<usize> = vec![1, 2, 4, 8, 16, 32].into_iter().filter(|&n| n < max).collect();
            counts.push(max);
            println!("{}x{}, {} frames each", opts.width, opts.height, frames);
            let mut base = 0.0;
            for n in counts {
                let o = Options { threads: n, ..opts };
                render(&scene, &camera, &o); // warm up
                let mut times = Vec::new();
                let mut tris = 0;
                for _ in 0..frames {
                    let t = Instant::now();
                    let (_, s) = render(&scene, &camera, &o);
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                    tris = s.triangles;
                }
                times.sort_by(f64::total_cmp);
                let med = times[times.len() / 2];
                if n == 1 {
                    base = med;
                }
                println!(
                    "{n:>3} threads: median {med:7.1} ms per frame ({:5.1} fps), {:5.1} M triangles/s, {:4.1}x",
                    1000.0 / med,
                    tris as f64 / med / 1e3,
                    base / med
                );
            }
            Ok(())
        }
        Cmd::View { look } => view(look),
    }
}

fn view(look: Look) -> Result<(), String> {
    let (scene, mut camera, mut opts) = setup(&look)?;
    let mut win = Window::new("swr", opts.width, opts.height, WindowOptions { resize: true, ..WindowOptions::default() })
        .map_err(|e| e.to_string())?;
    win.set_target_fps(120);
    let target = camera.target;
    let off = camera.eye - target;
    let mut dist = off.length();
    let mut yaw = off.x.atan2(off.z);
    let mut pitch = (off.y / dist).asin();
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut buf: Vec<u32> = Vec::new();
    let mut frame_ms = 0.0;
    while win.is_open() && !win.is_key_down(Key::Escape) {
        for k in win.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Key1 => opts.shading = Shading::Flat,
                Key::Key2 => opts.shading = Shading::Gouraud,
                Key::Key3 => opts.shading = Shading::Phong,
                Key::F => {
                    opts.filter = match opts.filter {
                        Filter::Nearest => Filter::Bilinear,
                        Filter::Bilinear => Filter::Trilinear,
                        Filter::Trilinear => Filter::Nearest,
                    }
                }
                Key::S => opts.shadow_size = if opts.shadow_size > 0 { 0 } else { 2048 },
                Key::W => opts.wireframe = !opts.wireframe,
                Key::A => opts.ssaa = if opts.ssaa > 1 { 1 } else { 2 },
                _ => {}
            }
        }
        if win.get_mouse_down(MouseButton::Left) {
            if let Some((mx, my)) = win.get_mouse_pos(MouseMode::Pass) {
                if let Some((lx, ly)) = last_mouse {
                    yaw -= (mx - lx) * 0.01;
                    pitch = (pitch + (my - ly) * 0.01).clamp(-1.4, 1.4);
                }
                last_mouse = Some((mx, my));
            }
        } else {
            last_mouse = None;
        }
        if let Some((_, wheel)) = win.get_scroll_wheel() {
            dist = (dist * (1.0 - wheel * 0.05)).clamp(1.0, 80.0);
        }
        camera.eye = target + v3(pitch.cos() * yaw.sin(), pitch.sin(), pitch.cos() * yaw.cos()) * dist;
        let (w, h) = win.get_size();
        opts.width = w.max(1);
        opts.height = h.max(1);
        let t = Instant::now();
        let (img, s) = render(&scene, &camera, &opts);
        frame_ms = frame_ms * 0.9 + t.elapsed().as_secs_f64() * 1e3 * 0.1;
        buf.clear();
        buf.extend(
            img.color
                .iter()
                .map(|c| (linear_to_srgb8(c.x) as u32) << 16 | (linear_to_srgb8(c.y) as u32) << 8 | linear_to_srgb8(c.z) as u32),
        );
        win.set_title(&format!(
            "swr - {} triangles, {:.1} ms per frame ({:.0} fps), {:?}, {:?}",
            s.triangles,
            frame_ms,
            1000.0 / frame_ms,
            opts.shading,
            opts.filter
        ));
        win.update_with_buffer(&buf, img.w, img.h).map_err(|e| e.to_string())?;
    }
    Ok(())
}
