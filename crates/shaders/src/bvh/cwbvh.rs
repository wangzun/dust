use crate::bvh::aabb::Aabb;
use crate::bvh::faststack::{FastStack, StackStack};
use crate::bvh::ray::{Ray, RayHit};
use bytemuck::{Pod, Zeroable};
use spirv_std::glam::{UVec2, Vec3, Vec3A, uvec2, vec3a};

pub const BRANCHING: usize = 8;

// Corresponds directly to the number of bit patterns created for child ordering
#[allow(dead_code)]
const DIRECTIONS: usize = 8;

#[allow(dead_code)]
const INVALID: u32 = u32::MAX;

const NQ: u32 = 8;
const NQ_SCALE: f32 = ((1 << NQ) - 1) as f32; //255.0
#[allow(dead_code)]
const DENOM: f32 = 1.0 / NQ_SCALE; // 1.0 / 255.0

/// A Compressed Wide BVH8 Node. repr(C), Pod, 80 bytes.
// https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct CwBvhNode {
    /// Min point of node AABB
    pub p: Vec3,

    /// Exponent of child bounding box compression
    /// Max point of node AABB could be calculated ex: `p.x + bitcast<f32>(e[0] << 23) * 255.0`
    pub e: [u8; 3],

    /// Bitmask indicating which children are internal nodes. 1 for internal, 0 for leaf
    pub imask: u8,

    /// Index of first child into `Vec<CwBvhNode>`
    pub child_base_idx: u32,

    /// Index of first primitive into primitive_indices `Vec<u32>`
    pub primitive_base_idx: u32,

    /// Meta data for each child
    /// Empty child slot: The field is set to 00000000
    ///
    /// For leaves nodes: the low 5 bits store the primitive offset [0..24) from primitive_base_idx. And the high
    /// 3 bits store the number of primitives in that leaf in a unary encoding.
    /// A child leaf with 2 primitives with the first primitive starting at primitive_base_idx would be 0b01100000
    /// A child leaf with 3 primitives with the first primitive starting at primitive_base_idx + 2 would be 0b11100010
    /// A child leaf with 1 primitive with the first primitive starting at primitive_base_idx + 1 would be 0b00100001
    ///
    /// For internal nodes: The high 3 bits are set to 001 while the low 5 bits store the child slot index plus 24
    /// i.e., the values range [24..32)
    pub child_meta: [u8; 8],

    // Note: deviation from the paper: the min&max are interleaved here.
    /// Axis planes for each child.
    /// The plane position could be calculated, for example, with `p.x + bitcast<f32>(e[0] << 23) * child_min_x[0]`
    /// But in the actual intersection implementation the ray is transformed instead.
    pub child_min_x: [u8; 8],
    pub child_max_x: [u8; 8],
    pub child_min_y: [u8; 8],
    pub child_max_y: [u8; 8],
    pub child_min_z: [u8; 8],
    pub child_max_z: [u8; 8],
}

pub(crate) const EPSILON: f32 = 0.0001;

impl CwBvhNode {
    #[inline(always)]
    pub fn intersect_ray(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        // #[cfg(all(
        //     any(target_arch = "x86", target_arch = "x86_64"),
        //     target_feature = "sse2"
        // ))]
        // {
        //     self.intersect_ray_simd(ray, oct_inv4)
        // }
        self.intersect_ray_basic(ray, oct_inv4)
    }

    /// Intersects only one child at a time with the given ray. Limited simd usage on platforms that support it. Exists for reference & compatibility.
    /// Traversal times with CwBvhNode::intersect_ray_simd take less than half the time vs intersect_ray_basic.
    #[inline(always)]
    pub fn intersect_ray_basic(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adjusted_ray_dir_inv = self.compute_extent() * ray.inv_direction;
        let adjusted_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;

        let mut hit_mask = 0;

        let rdx = ray.direction.x < 0.0;
        let rdy = ray.direction.y < 0.0;
        let rdz = ray.direction.z < 0.0;

        for child in 0..8 {
            let q_lo_x = self.child_min_x[child];
            let q_lo_y = self.child_min_y[child];
            let q_lo_z = self.child_min_z[child];

            let q_hi_x = self.child_max_x[child];
            let q_hi_y = self.child_max_y[child];
            let q_hi_z = self.child_max_z[child];

            let x_min = if rdx { q_hi_x } else { q_lo_x };
            let x_max = if rdx { q_lo_x } else { q_hi_x };
            let y_min = if rdy { q_hi_y } else { q_lo_y };
            let y_max = if rdy { q_lo_y } else { q_hi_y };
            let z_min = if rdz { q_hi_z } else { q_lo_z };
            let z_max = if rdz { q_lo_z } else { q_hi_z };

            let mut tmin3 = vec3a(x_min as f32, y_min as f32, z_min as f32);
            let mut tmax3 = vec3a(x_max as f32, y_max as f32, z_max as f32);

            // Account for grid origin and scale
            tmin3 = tmin3 * adjusted_ray_dir_inv + adjusted_ray_origin;
            tmax3 = tmax3 * adjusted_ray_dir_inv + adjusted_ray_origin;

            let tmin = tmin3.max_element().max(EPSILON); //ray.tmin?
            let tmax = tmax3.min_element().min(ray.tmax);

            let intersected = tmin <= tmax;
            if intersected {
                let meta = self.child_meta[child] as u32;
                let is_inner = (meta & (meta << 1) & 0b00010000) != 0;
                let bit_index = if is_inner {
                    (meta ^ (oct_inv4 & 0xff)) & 0b00011111
                } else {
                    meta & 0b00011111
                };
                let child_bits = (meta >> 5) & 0b00000111;
                hit_mask |= child_bits << bit_index;
            }
        }

        hit_mask
    }

    #[inline(always)]
    pub fn intersect_aabb(&self, aabb: &Aabb, oct_inv4: u32) -> u32 {
        let extent_rcp = 1.0 / self.compute_extent();
        let p = Vec3A::from(self.p);

        // Transform the query aabb into the node's local space
        let adjusted_aabb = Aabb::new((aabb.min - p) * extent_rcp, (aabb.max - p) * extent_rcp);

        let mut hit_mask = 0;

        let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

        for child in 0..8 {
            if self.local_child_aabb(child).intersect_aabb(&adjusted_aabb) {
                let child_bits = extract_byte64(child_bits8, child);
                let bit_index = extract_byte64(bit_index8, child);
                hit_mask |= child_bits << bit_index;
            }
        }

        hit_mask
    }

    #[inline(always)]
    pub fn contains_point(&self, point: &Vec3A, oct_inv4: u32) -> u32 {
        let extent_rcp = 1.0 / self.compute_extent();
        let p = Vec3A::from(self.p);

        // Transform the query point into the node's local space
        let adjusted_point = (*point - p) * extent_rcp;

        let mut hit_mask = 0;

        let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

        for child in 0..8 {
            if self.local_child_aabb(child).contains_point(adjusted_point) {
                let child_bits = extract_byte64(child_bits8, child);
                let bit_index = extract_byte64(bit_index8, child);
                hit_mask |= child_bits << bit_index;
            }
        }

        hit_mask
    }

    // TODO intersect frustum
    // https://github.com/zeux/niagara/blob/bf90aa8c78e352d3b753b35553a3bcc8c65ef7a0/src/shaders/drawcull.comp.glsl#L71
    // https://iquilezles.org/articles/frustumcorrect/

    #[inline(always)]
    pub fn get_child_and_index_bits(&self, oct_inv4: u32) -> (u64, u64) {
        let mut oct_inv8 = oct_inv4 as u64;
        oct_inv8 |= oct_inv8 << 32;
        let meta8 = u64::from_le_bytes(self.child_meta);

        // (meta8 & (meta8 << 1)) takes advantage of the offset indexing for inner nodes [24..32)
        // [0b00011000..=0b00011111). For leaf nodes [0..24) these two bits (0b00011000) are never both set.
        let inner_mask = 0b0001000000010000000100000001000000010000000100000001000000010000;
        let is_inner8 = (meta8 & (meta8 << 1)) & inner_mask;

        // 00010000 >> 4: 00000001, then 00000001 * 0xff: 11111111
        let inner_mask8 = (is_inner8 >> 4) * 0xffu64;

        // Each byte of bit_index8 contains the traversal priority, biased by 24, for internal nodes, and
        // the triangle offset for leaf nodes. The bit index will later be used to shift the child bits.
        let index_mask = 0b0001111100011111000111110001111100011111000111110001111100011111;
        let bit_index8 = (meta8 ^ (oct_inv8 & inner_mask8)) & index_mask;

        // For internal nodes child_bits8 will just be 1 in each byte, so that bit will then be shifted into the high
        // byte of the node hit_mask (see CwBvhNode::intersect_ray). For leaf nodes it will have the unary encoded
        // leaf primitive count and that will be shifted into the lower 24 bits of node hit_mask.
        let child_mask = 0b0000011100000111000001110000011100000111000001110000011100000111;
        let child_bits8 = (meta8 >> 5) & child_mask;
        (child_bits8, bit_index8)
    }

    /// Get local child aabb position relative to the parent
    #[inline(always)]
    pub fn local_child_aabb(&self, child: usize) -> Aabb {
        Aabb::new(
            vec3a(
                self.child_min_x[child] as f32,
                self.child_min_y[child] as f32,
                self.child_min_z[child] as f32,
            ),
            vec3a(
                self.child_max_x[child] as f32,
                self.child_max_y[child] as f32,
                self.child_max_z[child] as f32,
            ),
        )
    }

    #[inline(always)]
    pub fn child_aabb(&self, child: usize) -> Aabb {
        let e = self.compute_extent();
        let p: Vec3A = self.p.into();
        let mut local_aabb = self.local_child_aabb(child);
        local_aabb.min = local_aabb.min * e + p;
        local_aabb.max = local_aabb.max * e + p;
        local_aabb
    }

    #[inline(always)]
    pub fn aabb(&self) -> Aabb {
        let e = self.compute_extent();
        let p: Vec3A = self.p.into();
        Aabb::new(p, p + e * NQ_SCALE)
    }

    /// Convert stored extent exponent into float vector
    #[inline(always)]
    pub fn compute_extent(&self) -> Vec3A {
        vec3a(
            f32::from_bits((self.e[0] as u32) << 23),
            f32::from_bits((self.e[1] as u32) << 23),
            f32::from_bits((self.e[2] as u32) << 23),
        )
    }

    // If the child is empty this will also return true. If needed also use CwBvh::is_child_empty().
    #[inline(always)]
    pub fn is_leaf(&self, child: usize) -> bool {
        (self.imask & (1 << child)) == 0
    }

    #[inline(always)]
    pub fn is_child_empty(&self, child: usize) -> bool {
        self.child_meta[child] == 0
    }

    /// Returns the primitive starting index and primitive count for the given child.
    #[inline(always)]
    pub fn child_primitives(&self, child: usize) -> (u32, u32) {
        let child_meta = self.child_meta[child];
        let starting_index = self.primitive_base_idx + (self.child_meta[child] & 0b11111) as u32;
        let primitive_count = (child_meta & 0b11100000).count_ones();
        (starting_index, primitive_count)
    }

    /// Returns the node index of the given child.
    #[inline(always)]
    pub fn child_node_index(&self, child: usize) -> u32 {
        let child_meta = self.child_meta[child];
        let slot_index = (child_meta & 0b11111) as usize - 24;
        let relative_index = (self.imask as u32 & !(0xffffffffu32 << slot_index)).count_ones();
        self.child_base_idx + relative_index
    }
}

#[inline(always)]
pub fn extract_byte(x: u32, b: u32) -> u32 {
    (x >> (b * 8)) & 0xFFu32
}

#[inline(always)]
pub fn extract_byte64(x: u64, b: usize) -> u32 {
    ((x >> (b * 8)) as u32) & 0xFFu32
}

#[inline(always)]
pub fn firstbithigh(value: u32) -> u32 {
    31 - value.leading_zeros()
}

/// A Compressed Wide BVH8
#[derive(Clone, Default, PartialEq)]
#[repr(C)]
pub struct CwBvh<'a> {
    pub nodes: &'a [CwBvhNode],
    // pub primitive_indices: &'a [u32],
    // pub total_aabb: Aabb,
    // pub exact_node_aabbs: Option<&'a [Aabb]>,

    // /// Indicates that this BVH is using spatial splits. Large triangles are split into multiple smaller Aabbs, so
    // /// primitives will extend outside the leaf in some cases.
    // /// If the bvh uses splits, a primitive can show up in multiple leaf nodes so there wont be a 1 to 1 correlation
    // /// between the total number of primitives in leaf nodes and in Bvh2::primitive_indices, vs the input triangles.
    // /// If spatial splits are used, some validation steps have to be skipped.
    // pub uses_spatial_splits: bool,
}

const TRAVERSAL_STACK_SIZE: usize = 32;

/// Holds Ray traversal state to allow for dynamic traversal (yield on hit)
pub struct RayTraversal {
    pub stack: StackStack<UVec2, TRAVERSAL_STACK_SIZE>,
    pub current_group: UVec2,
    pub primitive_group: UVec2,
    pub oct_inv4: u32,
    pub ray: Ray,
}

impl RayTraversal {
    #[inline(always)]
    /// Reinitialize traversal state with new ray.
    pub fn reinit(&mut self, ray: Ray) {
        self.stack.clear();
        self.current_group = uvec2(0, 0x80000000);
        self.primitive_group = UVec2::ZERO;
        self.oct_inv4 = ray_get_octant_inv4(&ray.direction);
        self.ray = ray;
    }
}

/// Holds traversal state to allow for dynamic traversal (yield on hit)
pub struct Traversal {
    pub stack: StackStack<UVec2, TRAVERSAL_STACK_SIZE>,
    pub current_group: UVec2,
    pub primitive_group: UVec2,
    pub oct_inv4: u32,
    pub traversal_direction: Vec3A,
    pub primitive_id: u32,
    pub hitmask: u32,
}

impl Default for Traversal {
    fn default() -> Self {
        Self {
            stack: Default::default(),
            current_group: uvec2(0, 0x80000000),
            primitive_group: Default::default(),
            oct_inv4: Default::default(),
            traversal_direction: Default::default(),
            primitive_id: Default::default(),
            hitmask: Default::default(),
        }
    }
}

impl Traversal {
    #[inline(always)]
    /// Reinitialize traversal state with new traversal direction.
    pub fn reinit(&mut self, traversal_direction: Vec3A) {
        self.stack.clear();
        self.current_group = uvec2(0, 0x80000000);
        self.primitive_group = UVec2::ZERO;
        self.oct_inv4 = ray_get_octant_inv4(&traversal_direction);
        self.traversal_direction = traversal_direction;
        self.primitive_id = 0;
        self.hitmask = 0;
    }
}

#[inline(always)]
fn ray_get_octant_inv4(dir: &Vec3A) -> u32 {
    // Ray octant, encoded in 3 bits
    // let oct = (if dir.x < 0.0 { 0b100 } else { 0 })
    //     | (if dir.y < 0.0 { 0b010 } else { 0 })
    //     | (if dir.z < 0.0 { 0b001 } else { 0 });
    // return (7 - oct) * 0x01010101;
    (if dir.x < 0.0 { 0 } else { 0x04040404 }
        | if dir.y < 0.0 { 0 } else { 0x02020202 }
        | if dir.z < 0.0 { 0 } else { 0x01010101 })
}

impl CwBvh<'_> {
    #[inline(always)]
    pub fn new_ray_traversal(&self, ray: Ray) -> RayTraversal {
        //  BVH8's tend to be shallow. A stack of 32 would be very deep even for a large scene with no tlas.
        let stack = StackStack::default();
        let current_group = if self.nodes.is_empty() {
            UVec2::ZERO
        } else {
            uvec2(0, 0x80000000)
        };
        let primitive_group = UVec2::ZERO;
        let oct_inv4 = ray_get_octant_inv4(&ray.direction);

        RayTraversal {
            stack,
            current_group,
            primitive_group,
            oct_inv4,
            ray,
        }
    }

    #[inline(always)]
    /// traversal_direction is used to determine the order of bvh node child traversal. This would typically be the ray direction.
    pub fn new_traversal(&self, traversal_direction: Vec3A) -> Traversal {
        //  BVH8's tend to be shallow. A stack of 32 would be very deep even for a large scene with no tlas.
        let stack = StackStack::default();
        let current_group = if self.nodes.is_empty() {
            UVec2::ZERO
        } else {
            uvec2(0, 0x80000000)
        };
        let primitive_group = UVec2::ZERO;
        let oct_inv4 = ray_get_octant_inv4(&traversal_direction);
        Traversal {
            stack,
            current_group,
            primitive_group,
            oct_inv4,
            traversal_direction,
            primitive_id: 0,
            hitmask: 0,
        }
    }

    /// Traverse the BVH, finding the closest hit.
    /// Returns true if any primitive was hit.
    pub fn ray_traverse<F: FnMut(&Ray, usize) -> f32>(
        &self,
        ray: Ray,
        hit: &mut RayHit,
        mut intersection_fn: F,
    ) -> bool {
        let mut traverse_ray = ray;
        let mut state = self.new_traversal(ray.direction);

        loop {
            while state.primitive_group.y != 0 {
                let local_primitive_index = firstbithigh(state.primitive_group.y);
                state.primitive_group.y &= !(1u32 << local_primitive_index);
                state.primitive_id = state.primitive_group.x + local_primitive_index;

                let t = intersection_fn(&traverse_ray, state.primitive_id as usize);
                if t < traverse_ray.tmax {
                    hit.primitive_id = state.primitive_id;
                    hit.t = t;
                    traverse_ray.tmax = t;
                }
            }
            state.primitive_group = UVec2::ZERO;

            if state.current_group.y & 0xff000000 != 0 {
                let hits_imask = state.current_group.y;
                let child_index_offset = firstbithigh(hits_imask);
                let child_index_base = state.current_group.x;

                state.current_group.y &= !(1u32 << child_index_offset);
                if state.current_group.y & 0xff000000 != 0 {
                    state.stack.push(state.current_group);
                }

                let slot_index = (child_index_offset - 24) ^ (state.oct_inv4 & 0xff);
                let relative_index = (hits_imask & !(0xffffffffu32 << slot_index)).count_ones();
                let child_node_index = child_index_base + relative_index;
                let node = &self.nodes[child_node_index as usize];
                state.hitmask = node.intersect_ray(&traverse_ray, state.oct_inv4);

                state.current_group.x = node.child_base_idx;
                state.primitive_group.x = node.primitive_base_idx;
                state.current_group.y = (state.hitmask & 0xff000000) | (node.imask as u32);
                state.primitive_group.y = state.hitmask & 0x00ffffff;
            } else {
                state.current_group = UVec2::ZERO;
            }

            if state.primitive_group.y == 0 && (state.current_group.y & 0xff000000) == 0 {
                if state.stack.is_empty() {
                    break;
                }

                state.current_group = state.stack.pop_fast();
            }
        }

        hit.t < ray.tmax
    }

    /// Traverse the bvh for a given `Ray`. Returns true if the ray missed all primitives.
    pub fn ray_traverse_miss<F: FnMut(&Ray, usize) -> f32>(
        &self,
        ray: Ray,
        mut intersection_fn: F,
    ) -> bool {
        let mut state = self.new_traversal(ray.direction);

        loop {
            while state.primitive_group.y != 0 {
                let local_primitive_index = firstbithigh(state.primitive_group.y);
                state.primitive_group.y &= !(1u32 << local_primitive_index);
                state.primitive_id = state.primitive_group.x + local_primitive_index;

                let t = intersection_fn(&ray, state.primitive_id as usize);
                if t < ray.tmax {
                    return false;
                }
            }
            state.primitive_group = UVec2::ZERO;

            if state.current_group.y & 0xff000000 != 0 {
                let hits_imask = state.current_group.y;
                let child_index_offset = firstbithigh(hits_imask);
                let child_index_base = state.current_group.x;

                state.current_group.y &= !(1u32 << child_index_offset);
                if state.current_group.y & 0xff000000 != 0 {
                    state.stack.push(state.current_group);
                }

                let slot_index = (child_index_offset - 24) ^ (state.oct_inv4 & 0xff);
                let relative_index = (hits_imask & !(0xffffffffu32 << slot_index)).count_ones();
                let child_node_index = child_index_base + relative_index;
                let node = &self.nodes[child_node_index as usize];
                state.hitmask = node.intersect_ray(&ray, state.oct_inv4);

                state.current_group.x = node.child_base_idx;
                state.primitive_group.x = node.primitive_base_idx;
                state.current_group.y = (state.hitmask & 0xff000000) | (node.imask as u32);
                state.primitive_group.y = state.hitmask & 0x00ffffff;
            } else {
                state.current_group = UVec2::ZERO;
            }

            if state.primitive_group.y == 0 && (state.current_group.y & 0xff000000) == 0 {
                if state.stack.is_empty() {
                    break;
                }

                state.current_group = state.stack.pop_fast();
            }
        }

        true
    }

    /// Traverse the bvh for a given `Ray`. Intersects all primitives along ray (for things like evaluating transparency)
    ///   intersection_fn is called for all intersections. Ray is not updated to allow for evaluating at every hit.
    ///
    /// # Arguments
    /// * `ray` - The ray to be tested for intersection.
    /// * `intersection_fn` - takes the given ray and primitive index.
    pub fn ray_traverse_anyhit<F: FnMut(&Ray, usize)>(&self, ray: Ray, mut intersection_fn: F) {
        let mut state = self.new_traversal(ray.direction);

        loop {
            while state.primitive_group.y != 0 {
                let local_primitive_index = firstbithigh(state.primitive_group.y);
                state.primitive_group.y &= !(1u32 << local_primitive_index);
                state.primitive_id = state.primitive_group.x + local_primitive_index;
                intersection_fn(&ray, state.primitive_id as usize);
            }
            state.primitive_group = UVec2::ZERO;

            if state.current_group.y & 0xff000000 != 0 {
                let hits_imask = state.current_group.y;
                let child_index_offset = firstbithigh(hits_imask);
                let child_index_base = state.current_group.x;

                state.current_group.y &= !(1u32 << child_index_offset);
                if state.current_group.y & 0xff000000 != 0 {
                    state.stack.push(state.current_group);
                }

                let slot_index = (child_index_offset - 24) ^ (state.oct_inv4 & 0xff);
                let relative_index = (hits_imask & !(0xffffffffu32 << slot_index)).count_ones();
                let child_node_index = child_index_base + relative_index;
                let node = &self.nodes[child_node_index as usize];
                state.hitmask = node.intersect_ray(&ray, state.oct_inv4);

                state.current_group.x = node.child_base_idx;
                state.primitive_group.x = node.primitive_base_idx;
                state.current_group.y = (state.hitmask & 0xff000000) | (node.imask as u32);
                state.primitive_group.y = state.hitmask & 0x00ffffff;
            } else {
                state.current_group = UVec2::ZERO;
            }

            if state.primitive_group.y == 0 && (state.current_group.y & 0xff000000) == 0 {
                if state.stack.is_empty() {
                    break;
                }

                state.current_group = state.stack.pop_fast();
            }
        }
    }
}
