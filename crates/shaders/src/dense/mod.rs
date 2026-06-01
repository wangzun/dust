use spirv_std::{
    TypedBuffer,
    glam::{IVec3, UVec3, UVec4, Vec3, Vec4Swizzles},
};

pub const MODEL_SIZE: u32 = 256;
pub const BLOCK_SIZE: u32 = 4;
pub const EMPTY_MATERIAL_REF: u32 = u32::MAX;
pub const MATERIAL_PAGE_FLAG: u32 = 0x8000_0000;
pub const MATERIAL_PAGE_MASK: u32 = !MATERIAL_PAGE_FLAG;
pub const INVALID_MATERIAL: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DenseVoxelGpu {
    pub size: UVec4,
    pub occupancy_handle: u32,
    pub material_refs_handle: u32,
    pub material_pages_handle: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DenseVoxelHit {
    pub voxel: UVec3,
    pub material: u32,
    pub t: f32,
    pub normal: IVec3,
}

impl DenseVoxelHit {
    pub fn miss() -> Self {
        Self {
            voxel: UVec3::ZERO,
            material: INVALID_MATERIAL,
            t: f32::INFINITY,
            normal: IVec3::ZERO,
        }
    }

    pub fn is_hit(self) -> bool {
        self.material != INVALID_MATERIAL
    }
}

pub fn block_extent(model: DenseVoxelGpu) -> UVec3 {
    (model.size.xyz() + UVec3::splat(BLOCK_SIZE - 1)) / UVec3::splat(BLOCK_SIZE)
}

pub fn block_index(block: UVec3, extent: UVec3) -> usize {
    (block.x + block.y * extent.x + block.z * extent.x * extent.y) as usize
}

pub fn local_index(local: UVec3) -> u32 {
    local.x | (local.y << 2) | (local.z << 4)
}

pub fn voxel_block(voxel: UVec3) -> UVec3 {
    voxel / UVec3::splat(BLOCK_SIZE)
}

pub fn voxel_local(voxel: UVec3) -> UVec3 {
    voxel & UVec3::splat(BLOCK_SIZE - 1)
}

pub fn is_page_material_ref(material_ref: u32) -> bool {
    material_ref != EMPTY_MATERIAL_REF && (material_ref & MATERIAL_PAGE_FLAG) != 0
}

pub fn is_uniform_material_ref(material_ref: u32) -> bool {
    material_ref != EMPTY_MATERIAL_REF && (material_ref & MATERIAL_PAGE_FLAG) == 0
}

pub fn material_page_index(material_ref: u32) -> u32 {
    material_ref & MATERIAL_PAGE_MASK
}

pub fn occupancy_bit(occupancy: u64, local: UVec3) -> bool {
    ((occupancy >> local_index(local)) & 1) != 0
}

pub fn material_for_local_voxel(
    material_ref: u32,
    material_pages: &TypedBuffer<[u8]>,
    local: UVec3,
) -> u32 {
    if is_uniform_material_ref(material_ref) {
        return material_ref & 0xff;
    }

    if !is_page_material_ref(material_ref) {
        return INVALID_MATERIAL;
    }

    let page_index = material_page_index(material_ref);
    let byte_index = page_index * 64 + local_index(local);
    material_pages[byte_index as usize] as u32
}

pub fn voxel_material(
    model: DenseVoxelGpu,
    occupancy: &TypedBuffer<[u64]>,
    material_refs: &TypedBuffer<[u32]>,
    material_pages: &TypedBuffer<[u8]>,
    voxel: UVec3,
) -> u32 {
    if !voxel_in_bounds(model, voxel) {
        return INVALID_MATERIAL;
    }

    let block = voxel_block(voxel);
    let local = voxel_local(voxel);
    let block_index = block_index(block, block_extent(model));
    let word = occupancy[block_index];
    if !occupancy_bit(word, local) {
        return INVALID_MATERIAL;
    }

    material_for_local_voxel(material_refs[block_index], material_pages, local)
}

pub fn voxel_material_bindless(model: DenseVoxelGpu, voxel: UVec3) -> u32 {
    let occupancy = dst_heap::storage_buffer_from_u32::<u64>(model.occupancy_handle);
    let material_refs = dst_heap::storage_buffer_from_u32::<u32>(model.material_refs_handle);
    let material_pages = dst_heap::storage_buffer_from_u32::<u8>(model.material_pages_handle);
    voxel_material(model, occupancy, material_refs, material_pages, voxel)
}

pub fn ray_traverse_bindless(
    model: DenseVoxelGpu,
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_min: f32,
    t_max: f32,
) -> DenseVoxelHit {
    let occupancy = dst_heap::storage_buffer_from_u32::<u64>(model.occupancy_handle);
    let material_refs = dst_heap::storage_buffer_from_u32::<u32>(model.material_refs_handle);
    let material_pages = dst_heap::storage_buffer_from_u32::<u8>(model.material_pages_handle);
    ray_traverse(
        model,
        occupancy,
        material_refs,
        material_pages,
        ray_origin,
        ray_dir,
        t_min,
        t_max,
    )
}

pub fn ray_traverse(
    model: DenseVoxelGpu,
    occupancy: &TypedBuffer<[u64]>,
    material_refs: &TypedBuffer<[u32]>,
    material_pages: &TypedBuffer<[u8]>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_min: f32,
    t_max: f32,
) -> DenseVoxelHit {
    let bounds_min = Vec3::ZERO;
    let bounds_max = model.size.xyz().as_vec3();
    let range = intersect_aabb(ray_origin, ray_dir, bounds_min, bounds_max);
    if range.z == 0.0 {
        return DenseVoxelHit::miss();
    }
    let mut t = range.x;
    let exit_t = range.y;

    t = t.max(t_min);
    if t > exit_t || t > t_max {
        return DenseVoxelHit::miss();
    }

    let block_extent = block_extent(model);
    let mut block = position_to_voxel(model, ray_origin + ray_dir * t) / UVec3::splat(BLOCK_SIZE);
    let step = step_vec(ray_dir);
    let mut t_next = first_block_boundary_t(ray_origin, ray_dir, block, step);
    let t_delta = block_t_delta(ray_dir);
    let mut normal = IVec3::ZERO;

    while block.x < block_extent.x && block.y < block_extent.y && block.z < block_extent.z {
        let block_end_t = t_next.x.min(t_next.y).min(t_next.z).min(exit_t).min(t_max);
        let index = block_index(block, block_extent);
        let word = occupancy[index];
        if word != 0 {
            let hit = ray_traverse_block(
                model,
                word,
                material_refs[index],
                material_pages,
                ray_origin,
                ray_dir,
                block,
                t,
                block_end_t,
                normal,
            );
            if hit.is_hit() {
                return hit;
            }
        }

        if block_end_t >= exit_t || block_end_t >= t_max {
            break;
        }

        if t_next.x <= t_next.y && t_next.x <= t_next.z {
            t = t_next.x;
            t_next.x += t_delta.x;
            normal = IVec3::new(-step.x, 0, 0);
            if !step_axis(&mut block.x, step.x, block_extent.x) {
                break;
            }
        } else if t_next.y <= t_next.z {
            t = t_next.y;
            t_next.y += t_delta.y;
            normal = IVec3::new(0, -step.y, 0);
            if !step_axis(&mut block.y, step.y, block_extent.y) {
                break;
            }
        } else {
            t = t_next.z;
            t_next.z += t_delta.z;
            normal = IVec3::new(0, 0, -step.z);
            if !step_axis(&mut block.z, step.z, block_extent.z) {
                break;
            }
        }
    }

    DenseVoxelHit::miss()
}

fn ray_traverse_block(
    model: DenseVoxelGpu,
    occupancy: u64,
    material_ref: u32,
    material_pages: &TypedBuffer<[u8]>,
    ray_origin: Vec3,
    ray_dir: Vec3,
    block: UVec3,
    mut t: f32,
    block_exit_t: f32,
    entry_normal: IVec3,
) -> DenseVoxelHit {
    let block_origin = block * UVec3::splat(BLOCK_SIZE);
    let block_max = (block_origin + UVec3::splat(BLOCK_SIZE)).min(model.size.xyz());
    let mut voxel = position_to_voxel(model, ray_origin + ray_dir * t)
        .clamp(block_origin, block_max - UVec3::ONE);
    let step = step_vec(ray_dir);
    let mut t_next = first_voxel_boundary_t(ray_origin, ray_dir, voxel, step);
    let t_delta = voxel_t_delta(ray_dir);
    let mut normal = entry_normal;

    while voxel.x >= block_origin.x
        && voxel.y >= block_origin.y
        && voxel.z >= block_origin.z
        && voxel.x < block_max.x
        && voxel.y < block_max.y
        && voxel.z < block_max.z
    {
        let local = voxel - block_origin;
        if occupancy_bit(occupancy, local) {
            return DenseVoxelHit {
                voxel,
                material: material_for_local_voxel(material_ref, material_pages, local),
                t,
                normal,
            };
        }

        if t > block_exit_t {
            break;
        }

        if t_next.x <= t_next.y && t_next.x <= t_next.z {
            t = t_next.x;
            if t > block_exit_t {
                break;
            }
            t_next.x += t_delta.x;
            normal = IVec3::new(-step.x, 0, 0);
            if !step_axis_range(&mut voxel.x, step.x, block_origin.x, block_max.x) {
                break;
            }
        } else if t_next.y <= t_next.z {
            t = t_next.y;
            if t > block_exit_t {
                break;
            }
            t_next.y += t_delta.y;
            normal = IVec3::new(0, -step.y, 0);
            if !step_axis_range(&mut voxel.y, step.y, block_origin.y, block_max.y) {
                break;
            }
        } else {
            t = t_next.z;
            if t > block_exit_t {
                break;
            }
            t_next.z += t_delta.z;
            normal = IVec3::new(0, 0, -step.z);
            if !step_axis_range(&mut voxel.z, step.z, block_origin.z, block_max.z) {
                break;
            }
        }
    }

    DenseVoxelHit::miss()
}

fn voxel_in_bounds(model: DenseVoxelGpu, voxel: UVec3) -> bool {
    voxel.x < model.size.x && voxel.y < model.size.y && voxel.z < model.size.z
}

fn position_to_voxel(model: DenseVoxelGpu, position: Vec3) -> UVec3 {
    let max_voxel = model.size.xyz() - UVec3::ONE;
    position.floor().max(Vec3::ZERO).as_uvec3().min(max_voxel)
}

fn intersect_aabb(origin: Vec3, dir: Vec3, bounds_min: Vec3, bounds_max: Vec3) -> Vec3 {
    let inv_dir = Vec3::new(
        safe_inverse(dir.x),
        safe_inverse(dir.y),
        safe_inverse(dir.z),
    );
    let t0 = (bounds_min - origin) * inv_dir;
    let t1 = (bounds_max - origin) * inv_dir;
    let t_min = t0.min(t1);
    let t_max = t0.max(t1);
    let enter = t_min.x.max(t_min.y).max(t_min.z);
    let exit = t_max.x.min(t_max.y).min(t_max.z);
    if exit >= enter.max(0.0) {
        Vec3::new(enter.max(0.0), exit, 1.0)
    } else {
        Vec3::ZERO
    }
}

fn safe_inverse(value: f32) -> f32 {
    if value == 0.0 {
        f32::INFINITY
    } else {
        1.0 / value
    }
}

fn step_vec(dir: Vec3) -> IVec3 {
    IVec3::new(
        if dir.x >= 0.0 { 1 } else { -1 },
        if dir.y >= 0.0 { 1 } else { -1 },
        if dir.z >= 0.0 { 1 } else { -1 },
    )
}

fn first_block_boundary_t(origin: Vec3, dir: Vec3, block: UVec3, step: IVec3) -> Vec3 {
    let base = block * UVec3::splat(BLOCK_SIZE);
    let next = UVec3::new(
        if step.x > 0 {
            base.x + BLOCK_SIZE
        } else {
            base.x
        },
        if step.y > 0 {
            base.y + BLOCK_SIZE
        } else {
            base.y
        },
        if step.z > 0 {
            base.z + BLOCK_SIZE
        } else {
            base.z
        },
    )
    .as_vec3();
    Vec3::new(
        axis_boundary_t(origin.x, dir.x, next.x),
        axis_boundary_t(origin.y, dir.y, next.y),
        axis_boundary_t(origin.z, dir.z, next.z),
    )
}

fn first_voxel_boundary_t(origin: Vec3, dir: Vec3, voxel: UVec3, step: IVec3) -> Vec3 {
    let next = UVec3::new(
        if step.x > 0 { voxel.x + 1 } else { voxel.x },
        if step.y > 0 { voxel.y + 1 } else { voxel.y },
        if step.z > 0 { voxel.z + 1 } else { voxel.z },
    )
    .as_vec3();
    Vec3::new(
        axis_boundary_t(origin.x, dir.x, next.x),
        axis_boundary_t(origin.y, dir.y, next.y),
        axis_boundary_t(origin.z, dir.z, next.z),
    )
}

fn axis_boundary_t(origin: f32, dir: f32, boundary: f32) -> f32 {
    if dir == 0.0 {
        f32::INFINITY
    } else {
        (boundary - origin) / dir
    }
}

fn block_t_delta(dir: Vec3) -> Vec3 {
    Vec3::new(
        axis_delta(dir.x, BLOCK_SIZE as f32),
        axis_delta(dir.y, BLOCK_SIZE as f32),
        axis_delta(dir.z, BLOCK_SIZE as f32),
    )
}

fn voxel_t_delta(dir: Vec3) -> Vec3 {
    Vec3::new(
        axis_delta(dir.x, 1.0),
        axis_delta(dir.y, 1.0),
        axis_delta(dir.z, 1.0),
    )
}

fn axis_delta(dir: f32, cell_size: f32) -> f32 {
    if dir == 0.0 {
        f32::INFINITY
    } else {
        cell_size / dir.abs()
    }
}

fn step_axis(axis: &mut u32, step: i32, extent: u32) -> bool {
    if step > 0 {
        *axis += 1;
        *axis < extent
    } else if *axis == 0 {
        false
    } else {
        *axis -= 1;
        true
    }
}

fn step_axis_range(axis: &mut u32, step: i32, min: u32, max: u32) -> bool {
    if step > 0 {
        *axis += 1;
        *axis < max
    } else if *axis == min {
        false
    } else {
        *axis -= 1;
        true
    }
}
