#![allow(dead_code)]

use spirv_std::{
    TypedBuffer,
    arch::atomic_i_add,
    glam::{UVec3, UVec4, Vec2, Vec3, Vec4},
    memory::Scope,
    num_traits::Float,
    spirv,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Block {
    pub mask: u64,
    pub coords_packed: u32,
    pub material_ptr: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VisibleCluster {
    pub local_min_size: Vec4,
    pub color: Vec4,
    pub meta: UVec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VoxMaterialParams {
    pub base_color: Vec4,
    pub pbr: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RenderParams {
    pub camera_axis_x: Vec4,
    pub camera_axis_y: Vec4,
    pub camera_axis_z: Vec4,
    pub model_row0: Vec4,
    pub model_row1: Vec4,
    pub model_row2: Vec4,
    pub camera_params: Vec4,
    pub mesh_params: UVec4,
    pub resource_handles: UVec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshPushConstants {
    pub params_handle: u32,
    pub params_index: u32,
    pub clusters_handle: u32,
    pub indirect_handle: u32,
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn unpack_block_coords(packed: u32) -> UVec3 {
    UVec3::new((packed >> 20) & 1023, (packed >> 10) & 1023, packed & 1023) << 2
}

fn popcount64(value: u64) -> u32 {
    value.count_ones()
}

fn voxel_id(voxel: UVec3) -> u32 {
    (voxel.x << 4) | (voxel.y << 2) | voxel.z
}

fn occupied(mask: u64, voxel: UVec3) -> bool {
    ((mask >> voxel_id(voxel)) & 1) != 0
}

fn transform_position(params: RenderParams, position: Vec3) -> Vec3 {
    Vec3::new(
        params.model_row0.truncate().dot(position) + params.model_row0.w,
        params.model_row1.truncate().dot(position) + params.model_row1.w,
        params.model_row2.truncate().dot(position) + params.model_row2.w,
    )
}

fn voxel_material(
    material_info: &mut TypedBuffer<[u32]>,
    material_table: &mut TypedBuffer<[VoxMaterialParams]>,
    block: Block,
    hit_voxel_id: u32,
) -> VoxMaterialParams {
    let before_mask = block.mask & ((1u64 << hit_voxel_id) - 1);
    let material_offset = block.material_ptr + popcount64(before_mask);
    let material_word = material_info[(material_offset >> 2) as usize];
    let palette_index = (material_word >> ((material_offset & 3) * 8)) & 255;
    material_table[palette_index.min(255) as usize]
}

fn pack_unorm8(value: f32) -> u32 {
    (saturate(value) * 255.0).round() as u32
}

fn pack_pbr(pbr: Vec4) -> u32 {
    let metallic = pack_unorm8(pbr.x);
    let roughness = pack_unorm8(pbr.y);
    let specular = pack_unorm8(pbr.z);
    let emission = pack_unorm8(pbr.w / 16.0);
    metallic | (roughness << 8) | (specular << 16) | (emission << 24)
}

fn block_visible(params: RenderParams, local_min: Vec3, size: f32) -> bool {
    let center = local_min + Vec3::splat(size * 0.5);
    let world_center = transform_position(params, center);
    let camera_origin = Vec3::new(
        params.camera_axis_x.w,
        params.camera_axis_y.w,
        params.camera_axis_z.w,
    );
    let rel = world_center - camera_origin;
    let camera_x = rel.dot(params.camera_axis_x.truncate());
    let camera_y = rel.dot(params.camera_axis_y.truncate());
    let forward = rel.dot(-params.camera_axis_z.truncate());

    let radius = params
        .model_row0
        .truncate()
        .length()
        .max(
            params
                .model_row1
                .truncate()
                .length()
                .max(params.model_row2.truncate().length()),
        )
        * size
        * 0.9;
    let near_z = params.camera_params.z;
    let far_z = params.camera_params.w;
    if forward + radius <= near_z || forward - radius >= far_z {
        return false;
    }

    let tan_half_fov = params.camera_params.x;
    let aspect = params.camera_params.y;
    if camera_x.abs() > forward.max(near_z) * tan_half_fov * aspect + radius {
        return false;
    }
    if camera_y.abs() > forward.max(near_z) * tan_half_fov + radius {
        return false;
    }
    true
}

fn project_to_screen(params: RenderParams, world_position: Vec3) -> Vec3 {
    let camera_origin = Vec3::new(
        params.camera_axis_x.w,
        params.camera_axis_y.w,
        params.camera_axis_z.w,
    );
    let rel = world_position - camera_origin;
    let camera_x = rel.dot(params.camera_axis_x.truncate());
    let camera_y = rel.dot(params.camera_axis_y.truncate());
    let forward = rel.dot(-params.camera_axis_z.truncate());
    let tan_half_fov = params.camera_params.x;
    let aspect = params.camera_params.y;
    let near_z = params.camera_params.z;
    let far_z = params.camera_params.w;
    let depth = saturate((forward - near_z) / (far_z - near_z).max(0.0001));
    let ndc_x = camera_x / (forward * tan_half_fov * aspect).max(0.0001);
    let ndc_y = -camera_y / (forward * tan_half_fov).max(0.0001);
    Vec3::new(ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5, depth)
}

fn mip_offset(base_width: u32, base_height: u32, mip: u32) -> u32 {
    let mut offset = 0;
    let mut width = base_width;
    let mut height = base_height;
    let mut i = 0;
    while i < mip {
        offset += width * height;
        width = (width >> 1).max(1);
        height = (height >> 1).max(1);
        i += 1;
    }
    offset
}

fn cluster_occluded(control: RenderParams, params: RenderParams, local_min: Vec3, size: f32) -> bool {
    let pyramid_handle = control.resource_handles.x;
    let pyramid_width = control.resource_handles.y;
    let pyramid_height = control.resource_handles.z;
    let mip_count = control.resource_handles.w;
    if pyramid_width == 0 || pyramid_height == 0 || mip_count == 0 {
        return false;
    }

    let mut screen_min = Vec2::splat(1e20);
    let mut screen_max = Vec2::splat(-1e20);
    let mut nearest_depth = 1.0;
    let mut corner = 0;
    while corner < 8 {
        let corner_offset = Vec3::new(
            if (corner & 1) != 0 { size } else { 0.0 },
            if (corner & 2) != 0 { size } else { 0.0 },
            if (corner & 4) != 0 { size } else { 0.0 },
        );
        let world_position = transform_position(params, local_min + corner_offset);
        let camera_origin = Vec3::new(
            params.camera_axis_x.w,
            params.camera_axis_y.w,
            params.camera_axis_z.w,
        );
        let rel = world_position - camera_origin;
        let forward = rel.dot(-params.camera_axis_z.truncate());
        if forward <= params.camera_params.z {
            return false;
        }

        let projected = project_to_screen(params, world_position);
        screen_min = screen_min.min(projected.truncate());
        screen_max = screen_max.max(projected.truncate());
        nearest_depth = f32::min(nearest_depth, projected.z);
        corner += 1;
    }

    screen_min = screen_min.clamp(Vec2::ZERO, Vec2::ONE);
    screen_max = screen_max.clamp(Vec2::ZERO, Vec2::ONE);
    let rect_pixels = ((screen_max - screen_min) * Vec2::new(pyramid_width as f32, pyramid_height as f32))
        .max(Vec2::ONE);
    let mip_float = (rect_pixels.x.max(rect_pixels.y).log2().floor() - 1.0).max(0.0);
    let mip = (mip_float as u32).min(mip_count - 1);
    let mip_width = (pyramid_width >> mip).max(1);
    let mip_height = (pyramid_height >> mip).max(1);
    let offset = mip_offset(pyramid_width, pyramid_height, mip);

    let extent = Vec2::new(mip_width as f32, mip_height as f32);
    let max_pixel = spirv_std::glam::UVec2::new(mip_width - 1, mip_height - 1);
    let p0 = (screen_min * extent).as_uvec2().min(max_pixel);
    let p1 = (screen_max * extent).as_uvec2().min(max_pixel);
    let pyramid = dst_heap::storage_buffer_from_u32::<f32>(pyramid_handle);
    let d0 = pyramid[(offset + p0.y * mip_width + p0.x) as usize];
    let d1 = pyramid[(offset + p0.y * mip_width + p1.x) as usize];
    let d2 = pyramid[(offset + p1.y * mip_width + p0.x) as usize];
    let d3 = pyramid[(offset + p1.y * mip_width + p1.x) as usize];
    let max_depth = d0.max(d1).max(d2.max(d3));
    max_depth < nearest_depth - 0.001
}

fn select_lod_group_size(params: RenderParams, local_min: Vec3) -> u32 {
    let world_center = transform_position(params, local_min + Vec3::splat(2.0));
    let camera_origin = Vec3::new(
        params.camera_axis_x.w,
        params.camera_axis_y.w,
        params.camera_axis_z.w,
    );
    let distance = (world_center - camera_origin).length();

    if distance < 308.0 {
        1
    } else if distance < 880.0 {
        2
    } else {
        4
    }
}

fn emit_cluster(
    params: RenderParams,
    clusters: &mut TypedBuffer<[VisibleCluster]>,
    draw_args: &mut TypedBuffer<[u32]>,
    params_index: u32,
    local_min: Vec3,
    size: f32,
    color: Vec4,
    packed_pbr: u32,
) {
    let cluster_index = unsafe {
        atomic_i_add::<u32, { Scope::Device as u32 }, 0>(&mut draw_args[1], 1)
    };
    if cluster_index >= params.mesh_params.y {
        return;
    }

    clusters[cluster_index as usize] = VisibleCluster {
        local_min_size: local_min.extend(size),
        color,
        meta: UVec4::new(params_index, packed_pbr, size as u32, 0),
    };
}

fn emit_group(
    control: RenderParams,
    params: RenderParams,
    clusters: &mut TypedBuffer<[VisibleCluster]>,
    draw_args: &mut TypedBuffer<[u32]>,
    material_info: &mut TypedBuffer<[u32]>,
    material_table: &mut TypedBuffer<[VoxMaterialParams]>,
    block: Block,
    params_index: u32,
    block_origin: UVec3,
    group_origin: UVec3,
    group_size: u32,
    occlusion_enabled: bool,
) {
    let mut color_sum = Vec4::ZERO;
    let mut pbr_sum = Vec4::ZERO;
    let mut color_count = 0;

    let mut x = 0;
    while x < group_size {
        let mut y = 0;
        while y < group_size {
            let mut z = 0;
            while z < group_size {
                let local_voxel = group_origin + UVec3::new(x, y, z);
                if !local_voxel.cmpge(UVec3::splat(4)).any() && occupied(block.mask, local_voxel) {
                    let material =
                        voxel_material(material_info, material_table, block, voxel_id(local_voxel));
                    color_sum += material.base_color;
                    pbr_sum += material.pbr;
                    color_count += 1;
                }
                z += 1;
            }
            y += 1;
        }
        x += 1;
    }

    if color_count == 0 {
        return;
    }

    let size = group_size as f32;
    let local_min = (block_origin + group_origin).as_vec3();
    if !block_visible(params, local_min, size) {
        return;
    }
    if occlusion_enabled && cluster_occluded(control, params, local_min, size) {
        return;
    }

    let count = color_count as f32;
    emit_cluster(
        params,
        clusters,
        draw_args,
        params_index,
        local_min,
        size,
        color_sum / count,
        pack_pbr(pbr_sum / count),
    );
}

#[spirv(compute(threads(64, 1, 1), entry_point_name = "main"))]
pub fn mesh_main(
    #[spirv(global_invocation_id)] dispatch_thread_id: UVec3,
    #[spirv(push_constant)] constants: &MeshPushConstants,
) {
    let params_buffer = dst_heap::storage_buffer_from_u32::<RenderParams>(constants.params_handle);
    let clusters = dst_heap::storage_buffer_from_u32::<VisibleCluster>(constants.clusters_handle);
    let draw_args = dst_heap::storage_buffer_from_u32::<u32>(constants.indirect_handle);
    let control = params_buffer[constants.params_index as usize];
    let mode = control.mesh_params.z;

    if mode == 0 {
        if dispatch_thread_id.x == 0 {
            draw_args[0] = 36;
            draw_args[1] = 0;
            draw_args[2] = 0;
            draw_args[3] = 0;
        }
        return;
    }

    if mode == 2 {
        if dispatch_thread_id.x == 0 {
            draw_args[0] = 36;
            draw_args[1] = draw_args[1].min(control.mesh_params.y);
            draw_args[2] = 0;
            draw_args[3] = 0;
        }
        return;
    }

    let global_block_index = dispatch_thread_id.x;
    let instance_count = control.mesh_params.x;
    let first_instance_param = control.mesh_params.w;
    let mut low = 0;
    let mut high = instance_count;
    while low < high {
        let mid = (low + high) >> 1;
        let candidate_index = first_instance_param + mid;
        let candidate = params_buffer[candidate_index as usize];
        if candidate.mesh_params.w <= global_block_index {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if low == 0 {
        return;
    }

    let instance_param_index = first_instance_param + low - 1;
    let params = params_buffer[instance_param_index as usize];
    let block_count = params.mesh_params.x;
    let block_base = params.mesh_params.w;
    if global_block_index < block_base || global_block_index >= block_base + block_count {
        return;
    }
    let local_block_index = global_block_index - block_base;

    let geometry_info = dst_heap::storage_buffer_from_u32::<Block>(params.resource_handles.x);
    let material_info = dst_heap::storage_buffer_from_u32::<u32>(params.resource_handles.y);
    let material_table =
        dst_heap::storage_buffer_from_u32::<VoxMaterialParams>(params.resource_handles.z);

    let block = geometry_info[local_block_index as usize];
    if block.mask == 0 {
        return;
    }

    let block_origin = unpack_block_coords(block.coords_packed);
    let group_size = select_lod_group_size(params, block_origin.as_vec3());
    let occlusion_enabled = mode == 4 || mode == 5;
    let mut x = 0;
    while x < 4 {
        let mut y = 0;
        while y < 4 {
            let mut z = 0;
            while z < 4 {
                emit_group(
                    control,
                    params,
                    clusters,
                    draw_args,
                    material_info,
                    material_table,
                    block,
                    instance_param_index,
                    block_origin,
                    UVec3::new(x, y, z),
                    group_size,
                    occlusion_enabled,
                );
                z += group_size;
            }
            y += group_size;
        }
        x += group_size;
    }
}
