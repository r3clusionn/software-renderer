//! Textures in linear light with a mip chain, sampled nearest, bilinear or trilinear with the
//! level of detail taken from the screen-space derivatives of the texture coordinates.

use crate::math::{v3, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Bilinear,
    /// Bilinear on the two nearest mip levels, blended.
    Trilinear,
}

#[derive(Clone, Debug)]
struct Level {
    w: usize,
    h: usize,
    px: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct Texture {
    levels: Vec<Level>,
}

/// sRGB to linear for each 8-bit value.
fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear to sRGB, as an 8-bit value (clamped).
pub fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

impl Texture {
    /// From 8-bit sRGB RGB pixels; builds the mip chain by 2x2 averaging in linear light.
    pub fn from_srgb8(w: usize, h: usize, rgb: &[u8]) -> Texture {
        let lut: Vec<f32> = (0..=255).map(srgb_to_linear).collect();
        let px = rgb.as_chunks::<3>().0.iter().map(|p| v3(lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize])).collect();
        Texture::from_linear(w, h, px)
    }

    pub fn from_linear(w: usize, h: usize, px: Vec<Vec3>) -> Texture {
        assert_eq!(px.len(), w * h);
        let mut levels = vec![Level { w, h, px }];
        while levels.last().is_some_and(|l| l.w > 1 || l.h > 1) {
            let l = levels.last().unwrap();
            let (nw, nh) = ((l.w / 2).max(1), (l.h / 2).max(1));
            let mut px = Vec::with_capacity(nw * nh);
            for y in 0..nh {
                for x in 0..nw {
                    let at = |dx: usize, dy: usize| l.px[(2 * y + dy).min(l.h - 1) * l.w + (2 * x + dx).min(l.w - 1)];
                    px.push((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25);
                }
            }
            levels.push(Level { w: nw, h: nh, px });
        }
        Texture { levels }
    }

    /// Loads a PNG or baseline JPEG with the lumen image library.
    pub fn load(path: &std::path::Path) -> Result<Texture, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let img = if data.starts_with(b"\x89PNG") {
            lumen::png::decode(&data).map_err(|e| format!("{}: {e}", path.display()))?
        } else {
            lumen::jpeg::decode(&data).map_err(|e| format!("{}: {e}", path.display()))?
        };
        let rgb = img.to_8bit().convert(lumen::image::Color::Rgb);
        Ok(Texture::from_srgb8(rgb.width as usize, rgb.height as usize, rgb.data8()))
    }

    /// A checkerboard of `n` x `n` squares, `size` pixels square.
    pub fn checker(size: usize, n: usize, a: Vec3, b: Vec3) -> Texture {
        let cell = (size / n).max(1);
        let px =
            (0..size * size).map(|i| if ((i % size) / cell + (i / size) / cell).is_multiple_of(2) { a } else { b }).collect();
        Texture::from_linear(size, size, px)
    }

    pub fn size(&self) -> (usize, usize) {
        (self.levels[0].w, self.levels[0].h)
    }

    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    fn texel(&self, lvl: usize, x: i64, y: i64) -> Vec3 {
        let l = &self.levels[lvl];
        let xi = x.rem_euclid(l.w as i64) as usize;
        let yi = y.rem_euclid(l.h as i64) as usize;
        l.px[yi * l.w + xi]
    }

    fn bilinear(&self, lvl: usize, uv: Vec2) -> Vec3 {
        let l = &self.levels[lvl];
        let x = uv.x * l.w as f32 - 0.5;
        let y = uv.y * l.h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as i64, y0 as i64);
        let top = self.texel(lvl, xi, yi).lerp(self.texel(lvl, xi + 1, yi), fx);
        let bot = self.texel(lvl, xi, yi + 1).lerp(self.texel(lvl, xi + 1, yi + 1), fx);
        top.lerp(bot, fy)
    }

    /// The level of detail for these derivatives: log2 of the larger footprint in texels.
    pub fn lod(&self, ddx: Vec2, ddy: Vec2) -> f32 {
        let (w, h) = (self.levels[0].w as f32, self.levels[0].h as f32);
        let fx = ((ddx.x * w).powi(2) + (ddx.y * h).powi(2)).sqrt();
        let fy = ((ddy.x * w).powi(2) + (ddy.y * h).powi(2)).sqrt();
        fx.max(fy).max(1e-8).log2()
    }

    /// Samples with wrap-around addressing. `v` runs down the image (row 0 at v = 0).
    pub fn sample(&self, uv: Vec2, ddx: Vec2, ddy: Vec2, filter: Filter) -> Vec3 {
        match filter {
            Filter::Nearest => {
                let l = &self.levels[0];
                self.texel(0, (uv.x * l.w as f32).floor() as i64, (uv.y * l.h as f32).floor() as i64)
            }
            Filter::Bilinear => self.bilinear(0, uv),
            Filter::Trilinear => {
                let lod = self.lod(ddx, ddy).clamp(0.0, (self.levels.len() - 1) as f32);
                let l0 = lod.floor() as usize;
                let l1 = (l0 + 1).min(self.levels.len() - 1);
                let t = lod - l0 as f32;
                if t < 1e-4 || l0 == l1 {
                    self.bilinear(l0, uv)
                } else {
                    self.bilinear(l0, uv).lerp(self.bilinear(l1, uv), t)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::v2;

    #[test]
    fn srgb_round_trip() {
        for v in 0..=255u8 {
            assert_eq!(linear_to_srgb8(srgb_to_linear(v)), v);
        }
    }

    #[test]
    fn mip_chain_averages_to_the_mean() {
        let t = Texture::checker(64, 8, v3(1.0, 1.0, 1.0), v3(0.0, 0.0, 0.0));
        assert_eq!(t.levels(), 7);
        let last = t.levels.last().unwrap();
        assert!((last.px[0].x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lod_follows_the_footprint() {
        let t = Texture::checker(256, 8, v3(1.0, 1.0, 1.0), Vec3::ZERO);
        // One texel per pixel is level 0; four texels per pixel is level 2.
        assert!(t.lod(v2(1.0 / 256.0, 0.0), v2(0.0, 1.0 / 256.0)).abs() < 1e-5);
        assert!((t.lod(v2(4.0 / 256.0, 0.0), v2(0.0, 1.0 / 256.0)) - 2.0).abs() < 1e-5);
        // Far away (many texels per pixel), trilinear filtering gives the average grey.
        let c = t.sample(v2(0.3, 0.7), v2(1.0, 0.0), v2(0.0, 1.0), Filter::Trilinear);
        assert!((c.x - 0.5).abs() < 1e-5);
    }

    #[test]
    fn bilinear_is_exact_at_texel_centres_and_wraps() {
        let t = Texture::from_linear(2, 1, vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)]);
        let z = v2(0.0, 0.0);
        assert_eq!(t.sample(v2(0.25, 0.5), z, z, Filter::Bilinear).x, 0.0);
        assert_eq!(t.sample(v2(0.75, 0.5), z, z, Filter::Bilinear).x, 1.0);
        assert!((t.sample(v2(0.5, 0.5), z, z, Filter::Bilinear).x - 0.5).abs() < 1e-6);
        // At u = 0 the left neighbour wraps round to the right edge.
        assert!((t.sample(v2(0.0, 0.5), z, z, Filter::Bilinear).x - 0.5).abs() < 1e-6);
    }
}
