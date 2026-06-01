use crate::bvh::ray::Ray;
use bytemuck::{Pod, Zeroable};
use core::ops::BitAnd;
use spirv_std::glam::{Mat4, Vec3A};
/// An Axis-Aligned Bounding Box (AABB) represented by its minimum and maximum points.
#[derive(Default, Clone, Copy, Debug, PartialEq, Zeroable)]
#[repr(C)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

unsafe impl Pod for Aabb {}

impl Aabb {
    /// An invalid (empty) AABB with min set to the maximum possible value
    /// and max set to the minimum possible value.
    pub const INVALID: Self = Self {
        min: Vec3A::splat(f32::MAX),
        max: Vec3A::splat(f32::MIN),
    };

    /// An infinite AABB with min set to negative infinity
    /// and max set to positive infinity.
    pub const LARGEST: Self = Self {
        min: Vec3A::splat(-f32::MAX),
        max: Vec3A::splat(f32::MAX),
    };

    /// An infinite AABB with min set to negative infinity
    /// and max set to positive infinity.
    pub const INFINITY: Self = Self {
        min: Vec3A::splat(-f32::INFINITY),
        max: Vec3A::splat(f32::INFINITY),
    };

    /// Creates a new AABB with the given minimum and maximum points.
    #[inline(always)]
    pub fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }

    /// Creates a new AABB with both min and max set to the given point.
    #[inline(always)]
    pub fn from_point(point: Vec3A) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Creates an AABB that bounds the given set of points.
    #[inline(always)]
    pub fn from_points(points: &[Vec3A]) -> Self {
        let mut points = points.iter();
        let mut aabb = Aabb::from_point(*points.next().unwrap());
        for point in points {
            aabb.extend(*point);
        }
        aabb
    }

    /// Checks if the AABB contains the given point.
    #[inline(always)]
    pub fn contains_point(&self, point: Vec3A) -> bool {
        (point.cmpge(self.min).bitand(point.cmple(self.max))).all()
    }

    /// Extends the AABB to include the given point.
    #[inline(always)]
    pub fn extend(&mut self, point: Vec3A) -> &mut Self {
        *self = self.union(&Self::from_point(point));
        self
    }

    /// Returns the union of this AABB and another AABB.
    #[inline(always)]
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Returns the intersection of this AABB and another AABB.
    ///
    /// The intersection of two AABBs is the overlapping region that is
    /// common to both AABBs. If the AABBs do not overlap, the resulting
    /// AABB will have min and max values that do not form a valid box
    /// (min will not be less than max).
    #[inline(always)]
    pub fn intersection(&self, other: &Self) -> Self {
        Aabb {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Returns the diagonal vector of the AABB.
    #[inline(always)]
    pub fn diagonal(&self) -> Vec3A {
        self.max - self.min
    }

    /// Returns the center point of the AABB.
    #[inline(always)]
    pub fn center(&self) -> Vec3A {
        (self.max + self.min) * 0.5
    }

    /// Returns the center coordinate of the AABB along a specific axis.
    #[inline(always)]
    pub fn center_axis(&self, axis: usize) -> f32 {
        (self.max[axis] + self.min[axis]) * 0.5
    }

    /// Returns the index of the largest axis of the AABB.
    #[inline]
    pub fn largest_axis(&self) -> usize {
        let d = self.diagonal();
        if d.x < d.y {
            if d.y < d.z { 2 } else { 1 }
        } else if d.x < d.z {
            2
        } else {
            0
        }
    }

    /// Returns the index of the smallest axis of the AABB.
    #[inline]
    pub fn smallest_axis(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y {
            if d.y > d.z { 2 } else { 1 }
        } else if d.x > d.z {
            2
        } else {
            0
        }
    }

    /// Returns half the surface area of the AABB.
    #[inline(always)]
    pub fn half_area(&self) -> f32 {
        let d = self.diagonal();
        (d.x + d.y) * d.z + d.x * d.y
    }

    /// Returns the surface area of the AABB.
    #[inline(always)]
    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        2.0 * d.dot(d)
    }

    /// Returns an empty AABB.
    #[inline(always)]
    pub fn empty() -> Self {
        Self {
            min: Vec3A::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vec3A::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    /// Checks if the AABB is valid (i.e., min <= max on all axes).
    pub fn valid(&self) -> bool {
        self.min.cmple(self.max).all()
    }

    /// Checks if this AABB intersects with another AABB.
    #[inline(always)]
    pub fn intersect_aabb(&self, other: &Aabb) -> bool {
        (self.min.cmpgt(other.max) | self.max.cmplt(other.min)).bitmask() == 0
    }

    /// Checks if this AABB intersects with a ray and returns the distance to the intersection point.
    /// Returns `f32::INFINITY` if there is no intersection.
    #[inline(always)]
    pub fn intersect_ray(&self, ray: &Ray) -> f32 {
        // TODO perf: this is faster with #[target_feature(enable = "avx")] without any code changes
        // The compiler emits vsubps instead of mulps which ultimately results in less instructions.
        // Consider using is_x86_feature_detected!("avx") or #[multiversion(targets("x86_64+avx"))] before traversal
        // The manual impl using is_x86_feature_detected directly is a bit faster than multiversion

        let t1 = (self.min - ray.origin) * ray.inv_direction;
        let t2 = (self.max - ray.origin) * ray.inv_direction;

        let tmin = t1.min(t2);
        let tmax = t1.max(t2);

        let tmin_n = tmin.max_element();
        let tmax_n = tmax.min_element();

        if tmax_n >= tmin_n && tmax_n >= 0.0 {
            tmin_n
        } else {
            f32::INFINITY
        }
    }
}

pub trait Boundable {
    fn aabb(&self) -> Aabb;
}

/// A trait for types that can have a matrix transform applied. Primarily for testing/examples.
pub trait Transformable {
    fn transform(&mut self, matrix: &Mat4);
}

/// Apply a function to each component of a type.
#[doc(hidden)]
pub trait PerComponent<C1, C2 = C1, Output = Self> {
    fn per_comp(self, f: impl Fn(C1) -> C2) -> Output;
}

impl<Input, C1, C2, Output> PerComponent<C1, C2, Output> for Input
where
    Input: Into<[C1; 3]>,
    Output: From<[C2; 3]>,
{
    /// Applies a function to each component of the input.
    fn per_comp(self, f: impl Fn(C1) -> C2) -> Output {
        let [x, y, z] = self.into();
        Output::from([f(x), f(y), f(z)])
    }
}
