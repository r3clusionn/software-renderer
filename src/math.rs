//! Small vector and matrix types: what a renderer needs and nothing else. Matrices are column
//! major and multiply column vectors (`m * v`), as in OpenGL and most graphics texts.

use std::ops::{Add, Div, Index, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

pub const fn v2(x: f32, y: f32) -> Vec2 {
    Vec2 { x, y }
}

pub const fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

pub const fn v4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4 { x, y, z, w }
}

impl Vec2 {
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);

    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        if l > 0.0 {
            self / l
        } else {
            self
        }
    }

    pub fn mul_elem(self, o: Vec3) -> Vec3 {
        v3(self.x * o.x, self.y * o.y, self.z * o.z)
    }

    pub fn extend(self, w: f32) -> Vec4 {
        v4(self.x, self.y, self.z, w)
    }

    pub fn max_elem(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    pub fn lerp(self, o: Vec3, t: f32) -> Vec3 {
        self + (o - self) * t
    }
}

impl Vec4 {
    pub fn xyz(self) -> Vec3 {
        v3(self.x, self.y, self.z)
    }

    pub fn dot(self, o: Vec4) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }

    pub fn lerp(self, o: Vec4, t: f32) -> Vec4 {
        v4(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t, self.z + (o.z - self.z) * t, self.w + (o.w - self.w) * t)
    }
}

macro_rules! ops {
    ($t:ident { $($f:ident),* }) => {
        impl Add for $t {
            type Output = $t;
            fn add(self, o: $t) -> $t { $t { $($f: self.$f + o.$f),* } }
        }
        impl Sub for $t {
            type Output = $t;
            fn sub(self, o: $t) -> $t { $t { $($f: self.$f - o.$f),* } }
        }
        impl Mul<f32> for $t {
            type Output = $t;
            fn mul(self, k: f32) -> $t { $t { $($f: self.$f * k),* } }
        }
        impl Div<f32> for $t {
            type Output = $t;
            fn div(self, k: f32) -> $t { $t { $($f: self.$f / k),* } }
        }
        impl Neg for $t {
            type Output = $t;
            fn neg(self) -> $t { $t { $($f: -self.$f),* } }
        }
    };
}

ops!(Vec2 { x, y });
ops!(Vec3 { x, y, z });
ops!(Vec4 { x, y, z, w });

/// A 4x4 matrix, stored by columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub c: [Vec4; 4],
}

impl Index<(usize, usize)> for Mat4 {
    type Output = f32;
    /// `m[(row, col)]`.
    fn index(&self, (r, c): (usize, usize)) -> &f32 {
        let col = &self.c[c];
        match r {
            0 => &col.x,
            1 => &col.y,
            2 => &col.z,
            _ => &col.w,
        }
    }
}

impl Mat4 {
    pub const IDENTITY: Mat4 =
        Mat4 { c: [v4(1.0, 0.0, 0.0, 0.0), v4(0.0, 1.0, 0.0, 0.0), v4(0.0, 0.0, 1.0, 0.0), v4(0.0, 0.0, 0.0, 1.0)] };

    pub fn from_rows(r: [[f32; 4]; 4]) -> Mat4 {
        Mat4 { c: [0, 1, 2, 3].map(|j| v4(r[0][j], r[1][j], r[2][j], r[3][j])) }
    }

    fn row(&self, i: usize) -> Vec4 {
        v4(self[(i, 0)], self[(i, 1)], self[(i, 2)], self[(i, 3)])
    }

    pub fn translate(t: Vec3) -> Mat4 {
        Mat4::from_rows([[1.0, 0.0, 0.0, t.x], [0.0, 1.0, 0.0, t.y], [0.0, 0.0, 1.0, t.z], [0.0, 0.0, 0.0, 1.0]])
    }

    pub fn scale(s: Vec3) -> Mat4 {
        Mat4::from_rows([[s.x, 0.0, 0.0, 0.0], [0.0, s.y, 0.0, 0.0], [0.0, 0.0, s.z, 0.0], [0.0, 0.0, 0.0, 1.0]])
    }

    /// Rotation by `angle` radians around `axis` (right-handed).
    pub fn rotate(axis: Vec3, angle: f32) -> Mat4 {
        let a = axis.normalize();
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        Mat4::from_rows([
            [t * a.x * a.x + c, t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y, 0.0],
            [t * a.x * a.y + s * a.z, t * a.y * a.y + c, t * a.y * a.z - s * a.x, 0.0],
            [t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// A right-handed view matrix looking from `eye` at `target`.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Mat4::from_rows([
            [s.x, s.y, s.z, -s.dot(eye)],
            [u.x, u.y, u.z, -u.dot(eye)],
            [-f.x, -f.y, -f.z, f.dot(eye)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// A perspective projection to clip space with depth from 0 (near) to 1 (far) after the
    /// divide, as Direct3D and Vulkan use. `fovy` is the vertical field of view in radians.
    pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fovy / 2.0).tan();
        Mat4::from_rows([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), near * far / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }

    /// An orthographic projection (for shadow maps), depth 0 to 1.
    pub fn orthographic(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
        Mat4::from_rows([
            [2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)],
            [0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)],
            [0.0, 0.0, 1.0 / (n - f), n / (n - f)],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn transpose(&self) -> Mat4 {
        Mat4 { c: [self.row(0), self.row(1), self.row(2), self.row(3)] }
    }

    /// The inverse by cofactors; `None` if the matrix is singular.
    pub fn inverse(&self) -> Option<Mat4> {
        let m = |r: usize, c: usize| self[(r, c)];
        let mut inv = [[0.0f32; 4]; 4];
        let minor = |r0: usize, c0: usize| -> f32 {
            let rs: Vec<usize> = (0..4).filter(|&r| r != r0).collect();
            let cs: Vec<usize> = (0..4).filter(|&c| c != c0).collect();
            let a = |i: usize, j: usize| m(rs[i], cs[j]);
            a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
                + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
        };
        let mut det = 0.0;
        for c in 0..4 {
            let cof = if c % 2 == 0 { 1.0 } else { -1.0 } * minor(0, c);
            det += m(0, c) * cof;
        }
        if det.abs() < 1e-12 {
            return None;
        }
        for (r, row) in inv.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                // The adjugate is the transposed cofactor matrix.
                let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                *v = sign * minor(c, r) / det;
            }
        }
        Some(Mat4::from_rows(inv))
    }

    /// Transforms a direction (w = 0) and keeps xyz.
    pub fn transform_dir(&self, d: Vec3) -> Vec3 {
        (*self * d.extend(0.0)).xyz()
    }

    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let v = *self * p.extend(1.0);
        v.xyz() / v.w
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, v: Vec4) -> Vec4 {
        self.c[0] * v.x + self.c[1] * v.y + self.c[2] * v.z + self.c[3] * v.w
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, o: Mat4) -> Mat4 {
        Mat4 { c: o.c.map(|col| self * col) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Mat4, b: Mat4) -> bool {
        (0..4).all(|r| (0..4).all(|c| (a[(r, c)] - b[(r, c)]).abs() < 1e-4))
    }

    #[test]
    fn inverse_undoes() {
        let m = Mat4::translate(v3(1.0, -2.0, 3.0)) * Mat4::rotate(v3(1.0, 1.0, 0.3), 0.7) * Mat4::scale(v3(2.0, 0.5, 1.5));
        assert!(close(m * m.inverse().unwrap(), Mat4::IDENTITY));
        assert!(Mat4::scale(v3(1.0, 0.0, 1.0)).inverse().is_none());
    }

    #[test]
    fn perspective_maps_near_and_far_to_0_and_1() {
        let p = Mat4::perspective(1.0, 1.5, 0.5, 100.0);
        let near = p * v4(0.0, 0.0, -0.5, 1.0);
        let far = p * v4(0.0, 0.0, -100.0, 1.0);
        assert!((near.z / near.w).abs() < 1e-6);
        assert!((far.z / far.w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn look_at_puts_the_target_on_the_negative_z_axis() {
        let v = Mat4::look_at(v3(3.0, 4.0, 5.0), v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0));
        let t = v.transform_point(v3(0.0, 0.0, 0.0));
        assert!(t.x.abs() < 1e-5 && t.y.abs() < 1e-5);
        assert!((t.z + 50f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn rotation_is_right_handed() {
        let r = Mat4::rotate(v3(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let x = r.transform_dir(v3(1.0, 0.0, 0.0));
        assert!((x - v3(0.0, 1.0, 0.0)).length() < 1e-6);
    }
}
