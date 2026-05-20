#![allow(dead_code)]

use spirv_std::{
    glam::{UVec4, Vec3, Vec4},
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
    #[spirv(location = 0)] color: Vec4,
    #[spirv(location = 1)] normal: Vec4,
    #[spirv(location = 2)] pbr: Vec4,
    #[spirv(location = 3)] view: Vec4,
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
    if color.w > 6.5 {
        let cheap = albedo * (0.08 + 1.45 * n_dot_l) + albedo * emission_strength;
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

    let direct_light = Vec3::splat(2.2) * (diffuse + specular) * n_dot_l;
    let ambient = albedo * 0.04;
    let emissive = albedo * emission_strength;
    *out_color = (ambient + direct_light + emissive).extend(1.0);
}
