#![allow(dead_code)]

use crate::{
    bvh::{
        cwbvh::{CwBvh, CwBvhNode},
        ray::Ray,
    },
    dense::{DenseVoxelGpu, ray_traverse_bindless},
};
use spirv_std::{
    glam::{UVec3, UVec4, Vec3, Vec3A, Vec4},
    spirv,
};

const PI: f32 = core::f32::consts::PI;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VisibleCluster {
    pub local_min_size: Vec4,
    pub color: Vec4,
    pub meta: UVec4,
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
    pub cull_min: UVec4,
    pub cull_max: UVec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RtDenseModel {
    pub inverse_model_row0: Vec4,
    pub inverse_model_row1: Vec4,
    pub inverse_model_row2: Vec4,
    pub size: UVec4,
    pub resource_handles: UVec4,
    pub cull_min: UVec4,
    pub cull_max: UVec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeshRenderPushConstants {
    pub params_handle: u32,
    pub params_index: u32,
    pub clusters_handle: u32,
    pub indirect_handle: u32,
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn saturate_vec3(value: Vec3) -> Vec3 {
    value.clamp(Vec3::ZERO, Vec3::ONE)
}

fn cube_corner(vertex_id: u32) -> Vec3 {
    let tri = vertex_id % 6;
    let corner_id = if tri == 0 {
        0
    } else if tri == 1 {
        1
    } else if tri == 2 {
        2
    } else if tri == 3 {
        0
    } else if tri == 4 {
        2
    } else {
        3
    };

    match vertex_id / 6 {
        0 => [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ][corner_id as usize],
        1 => [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ][corner_id as usize],
        2 => [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ][corner_id as usize],
        3 => [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0),
        ][corner_id as usize],
        4 => [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ][corner_id as usize],
        _ => [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ][corner_id as usize],
    }
}

fn cube_normal(vertex_id: u32) -> Vec3 {
    match vertex_id / 6 {
        0 => Vec3::new(-1.0, 0.0, 0.0),
        1 => Vec3::new(1.0, 0.0, 0.0),
        2 => Vec3::new(0.0, -1.0, 0.0),
        3 => Vec3::new(0.0, 1.0, 0.0),
        4 => Vec3::new(0.0, 0.0, -1.0),
        _ => Vec3::new(0.0, 0.0, 1.0),
    }
}

fn transform_position(params: RenderParams, position: Vec3) -> Vec3 {
    Vec3::new(
        params.model_row0.truncate().dot(position) + params.model_row0.w,
        params.model_row1.truncate().dot(position) + params.model_row1.w,
        params.model_row2.truncate().dot(position) + params.model_row2.w,
    )
}

fn transform_normal(params: RenderParams, normal: Vec3) -> Vec3 {
    Vec3::new(
        params.model_row0.truncate().dot(normal),
        params.model_row1.truncate().dot(normal),
        params.model_row2.truncate().dot(normal),
    )
    .normalize()
}

fn transform_rt_point(model: RtDenseModel, position: Vec3) -> Vec3 {
    Vec3::new(
        model.inverse_model_row0.truncate().dot(position) + model.inverse_model_row0.w,
        model.inverse_model_row1.truncate().dot(position) + model.inverse_model_row1.w,
        model.inverse_model_row2.truncate().dot(position) + model.inverse_model_row2.w,
    )
}

fn transform_rt_vector(model: RtDenseModel, vector: Vec3) -> Vec3 {
    Vec3::new(
        model.inverse_model_row0.truncate().dot(vector),
        model.inverse_model_row1.truncate().dot(vector),
        model.inverse_model_row2.truncate().dot(vector),
    )
}

fn voxel_in_rt_cull(model: RtDenseModel, voxel: UVec3) -> bool {
    voxel.x >= model.cull_min.x
        && voxel.y >= model.cull_min.y
        && voxel.z >= model.cull_min.z
        && voxel.x < model.cull_max.x.min(model.size.x)
        && voxel.y < model.cull_max.y.min(model.size.y)
        && voxel.z < model.cull_max.z.min(model.size.z)
}

fn rt_dense_intersection(
    model: RtDenseModel,
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_min: f32,
    t_max: f32,
) -> f32 {
    let local_origin = transform_rt_point(model, ray_origin);
    let local_dir = transform_rt_vector(model, ray_dir);
    if local_dir.length_squared() < 0.000001 {
        return f32::INFINITY;
    }

    let dense_model = DenseVoxelGpu {
        size: model.size,
        occupancy_handle: model.resource_handles.x,
        material_refs_handle: model.resource_handles.y,
        material_pages_handle: model.resource_handles.z,
        _pad: 0,
    };

    let mut current_min = t_min;
    for _ in 0..8 {
        let hit = ray_traverse_bindless(dense_model, local_origin, local_dir, current_min, t_max);
        if !hit.is_hit() {
            return f32::INFINITY;
        }
        if voxel_in_rt_cull(model, hit.voxel) {
            return hit.t;
        }
        current_min = hit.t + 0.01;
        if current_min >= t_max {
            return f32::INFINITY;
        }
    }

    f32::INFINITY
}

fn rt_scene_any_hit(
    draw_params: RenderParams,
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_max: f32,
) -> bool {
    let node_count = draw_params.resource_handles.w;
    let primitive_count = draw_params.cull_min.x;
    let model_count = draw_params.cull_min.y;
    if node_count == 0 || primitive_count == 0 || model_count == 0 {
        return false;
    }

    let nodes = dst_heap::storage_buffer_from_u32::<CwBvhNode>(draw_params.resource_handles.x);
    let primitive_to_model =
        dst_heap::storage_buffer_from_u32::<u32>(draw_params.resource_handles.y);
    let models = dst_heap::storage_buffer_from_u32::<RtDenseModel>(draw_params.resource_handles.z);
    let bvh = CwBvh { nodes };
    let ray = Ray::new(
        Vec3A::from(ray_origin),
        Vec3A::from(ray_dir.normalize()),
        0.02,
        t_max,
    );

    !bvh.ray_traverse_miss(ray, |_, primitive_id| {
        if primitive_id as u32 >= primitive_count {
            return f32::INFINITY;
        }
        let model_index = primitive_to_model[primitive_id];
        if model_index == u32::MAX || model_index >= model_count {
            return f32::INFINITY;
        }

        rt_dense_intersection(
            models[model_index as usize],
            ray_origin,
            ray_dir,
            0.02,
            t_max,
        )
    })
}

fn tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let tangent = if normal.y.abs() < 0.9 {
        normal.cross(Vec3::Y).normalize()
    } else {
        normal.cross(Vec3::X).normalize()
    };
    (tangent, normal.cross(tangent).normalize())
}

fn rt_ambient_occlusion(draw_params: RenderParams, world_position: Vec3, normal: Vec3) -> f32 {
    let (tangent, bitangent) = tangent_basis(normal);
    let origin = world_position + normal * 0.08;
    let dir0 = (normal * 0.75 + tangent * 0.45 + bitangent * 0.2).normalize();
    let dir1 = (normal * 0.7 - tangent * 0.25 + bitangent * 0.55).normalize();
    let mut hits = 0.0;
    if rt_scene_any_hit(draw_params, origin, dir0, 24.0) {
        hits += 1.0;
    }
    if rt_scene_any_hit(draw_params, origin, dir1, 24.0) {
        hits += 1.0;
    }
    1.0 - hits * 0.18
}

fn unpack_unorm8(packed: u32, shift: u32) -> f32 {
    ((packed >> shift) & 255) as f32 / 255.0
}

fn unpack_pbr(packed: u32) -> Vec4 {
    Vec4::new(
        unpack_unorm8(packed, 0),
        unpack_unorm8(packed, 8).max(0.04),
        unpack_unorm8(packed, 16),
        unpack_unorm8(packed, 24) * 16.0,
    )
}

fn project_position(params: RenderParams, world_position: Vec3) -> Vec4 {
    let camera_origin = Vec3::new(
        params.camera_axis_x.w,
        params.camera_axis_y.w,
        params.camera_axis_z.w,
    );
    let rel = world_position - camera_origin;
    let camera_x = rel.dot(params.camera_axis_x.truncate());
    let camera_y = rel.dot(params.camera_axis_y.truncate());
    let forward = rel.dot(-params.camera_axis_z.truncate());
    let near_z = params.camera_params.z;
    let far_z = params.camera_params.w;
    if forward <= near_z {
        return Vec4::new(2.0, 2.0, 1.0, 1.0);
    }

    let tan_half_fov = params.camera_params.x;
    let aspect = params.camera_params.y;
    let depth = saturate((forward - near_z) / (far_z - near_z).max(0.0001));
    Vec4::new(
        camera_x / (tan_half_fov * aspect).max(0.0001),
        -camera_y / tan_half_fov.max(0.0001),
        depth * forward,
        forward,
    )
}

fn cluster_world_position(params: RenderParams, cluster: VisibleCluster, vertex_id: u32) -> Vec3 {
    let local_position =
        cluster.local_min_size.truncate() + cube_corner(vertex_id) * cluster.local_min_size.w;
    transform_position(params, local_position)
}

fn vertex_id_without_base(vertex_index: u32, base_vertex: u32) -> u32 {
    vertex_index.wrapping_sub(base_vertex)
}

fn instance_id_without_base(instance_index: u32, base_instance: u32) -> u32 {
    instance_index.wrapping_sub(base_instance)
}

#[spirv(vertex(entry_point_name = "vertexMain"))]
pub fn vertex_main(
    #[spirv(vertex_index)] vertex_index: u32,
    #[spirv(base_vertex)] base_vertex: u32,
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(base_instance)] base_instance: u32,
    #[spirv(push_constant)] constants: &MeshRenderPushConstants,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_normal: &mut Vec4,
    #[spirv(location = 2)] out_pbr: &mut Vec4,
    #[spirv(location = 3)] out_view: &mut Vec4,
    #[spirv(location = 4)] out_world_position: &mut Vec4,
    #[spirv(position)] out_position: &mut Vec4,
) {
    let vertex_id = vertex_id_without_base(vertex_index, base_vertex);
    let instance_id = instance_id_without_base(instance_index, base_instance);
    let params_buffer = dst_heap::storage_buffer_from_u32::<RenderParams>(constants.params_handle);
    let clusters = dst_heap::storage_buffer_from_u32::<VisibleCluster>(constants.clusters_handle);
    let draw_params = params_buffer[constants.params_index as usize];
    if vertex_id >= draw_params.mesh_params.x || instance_id >= draw_params.mesh_params.y {
        *out_position = Vec4::new(2.0, 2.0, 1.0, 1.0);
        *out_color = Vec4::ZERO;
        *out_normal = Vec4::ZERO;
        *out_pbr = Vec4::ZERO;
        *out_view = Vec4::ZERO;
        *out_world_position = Vec4::ZERO;
        return;
    }

    let cluster = clusters[instance_id as usize];
    let params = params_buffer[cluster.meta.x as usize];
    let world_position = cluster_world_position(params, cluster, vertex_id);
    let world_normal = transform_normal(params, cube_normal(vertex_id));
    let camera_origin = Vec3::new(
        params.camera_axis_x.w,
        params.camera_axis_y.w,
        params.camera_axis_z.w,
    );

    *out_position = project_position(params, world_position);
    *out_color = Vec4::new(
        cluster.color.x,
        cluster.color.y,
        cluster.color.z,
        cluster.local_min_size.w,
    );
    *out_normal = world_normal.extend(0.0);
    *out_pbr = unpack_pbr(cluster.meta.y);
    *out_view = (camera_origin - world_position).normalize().extend(0.0);
    *out_world_position = world_position.extend(1.0);
}

#[spirv(vertex(entry_point_name = "depthVertexMain"))]
pub fn depth_vertex_main(
    #[spirv(vertex_index)] vertex_index: u32,
    #[spirv(base_vertex)] base_vertex: u32,
    #[spirv(instance_index)] instance_index: u32,
    #[spirv(base_instance)] base_instance: u32,
    #[spirv(push_constant)] constants: &MeshRenderPushConstants,
    #[spirv(position)] out_position: &mut Vec4,
) {
    let vertex_id = vertex_id_without_base(vertex_index, base_vertex);
    let instance_id = instance_id_without_base(instance_index, base_instance);
    let params_buffer = dst_heap::storage_buffer_from_u32::<RenderParams>(constants.params_handle);
    let clusters = dst_heap::storage_buffer_from_u32::<VisibleCluster>(constants.clusters_handle);
    let draw_params = params_buffer[constants.params_index as usize];
    if vertex_id >= draw_params.mesh_params.x || instance_id >= draw_params.mesh_params.y {
        *out_position = Vec4::new(2.0, 2.0, 1.0, 1.0);
        return;
    }

    let cluster = clusters[instance_id as usize];
    let params = params_buffer[cluster.meta.x as usize];
    let world_position = cluster_world_position(params, cluster, vertex_id);
    *out_position = project_position(params, world_position);
}

fn distribution_ggx(normal: Vec3, half_vector: Vec3, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = saturate(normal.dot(half_vector));
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom).max(0.00001)
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    n_dot_v / (n_dot_v * (1.0 - k) + k).max(0.00001)
}

fn geometry_smith(normal: Vec3, view_dir: Vec3, light_dir: Vec3, roughness: f32) -> f32 {
    let n_dot_v = saturate(normal.dot(view_dir));
    let n_dot_l = saturate(normal.dot(light_dir));
    geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness)
}

fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    let x = saturate(1.0 - cos_theta);
    let x2 = x * x;
    f0 + (Vec3::ONE - f0) * x2 * x2 * x
}

#[spirv(fragment(entry_point_name = "fragmentMain"))]
pub fn fragment_main(
    #[spirv(push_constant)] constants: &MeshRenderPushConstants,
    #[spirv(location = 0)] color: Vec4,
    #[spirv(location = 1)] normal: Vec4,
    #[spirv(location = 2)] pbr: Vec4,
    #[spirv(location = 3)] view: Vec4,
    #[spirv(location = 4)] world_position: Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
) {
    let albedo = saturate_vec3(color.truncate()).powf(2.2);
    let metallic = saturate(pbr.x);
    let roughness = pbr.y.clamp(0.04, 1.0);
    let specular_strength = saturate(pbr.z);
    let emission_strength = pbr.w.max(0.0);

    let n = normal.truncate().normalize();
    let l = Vec3::new(0.35, 0.7, 0.55).normalize();
    let n_dot_l = saturate(n.dot(l));
    let params_buffer = dst_heap::storage_buffer_from_u32::<RenderParams>(constants.params_handle);
    let draw_params = params_buffer[constants.params_index as usize];
    let origin = world_position.truncate() + n * 0.08;
    let shadow = if n_dot_l > 0.0 && rt_scene_any_hit(draw_params, origin, l, 512.0) {
        0.18
    } else {
        1.0
    };
    let ao = rt_ambient_occlusion(draw_params, world_position.truncate(), n);
    if color.w > 6.5 {
        let cheap = albedo * (0.08 * ao + 1.45 * n_dot_l * shadow) + albedo * emission_strength;
        *out_color = cheap.extend(1.0);
        return;
    }

    let v = view.truncate().normalize();
    let h = (v + l).normalize();
    let n_dot_v = saturate(n.dot(v));

    let f0 = Vec3::splat(0.08 * specular_strength).lerp(albedo, metallic);
    let f = fresnel_schlick(saturate(h.dot(v)), f0);
    let d = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let specular = (d * g * f) / (4.0 * n_dot_v * n_dot_l).max(0.0001);
    let diffuse = (Vec3::ONE - f) * (1.0 - metallic) * albedo / PI;

    let direct_light = Vec3::splat(2.2) * (diffuse + specular) * n_dot_l * shadow;
    let ambient = albedo * 0.04 * ao;
    let emissive = albedo * emission_strength;
    *out_color = (ambient + direct_light + emissive).extend(1.0);
}
