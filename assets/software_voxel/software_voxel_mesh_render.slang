[[vk::binding(0, 0)]]
__DynamicResource<__DynamicResourceKind.General> resourceHandles[];

[[vk::binding(0, 1)]]
__DynamicResource<__DynamicResourceKind.Sampler> samplerHandles[];

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle)
    where T : IOpaqueDescriptor
{
    __target_switch
    {
    case spirv:
        if (T.kind == DescriptorKind.Sampler)
            return samplerHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
        else
            return resourceHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
    default:
        return defaultGetDescriptorFromHandle(handle);
    }
}

struct VisibleCluster
{
    float4 local_min_size;
    float4 color;
    uint4 meta;
};

struct RenderParams
{
    float4 camera_axis_x;
    float4 camera_axis_y;
    float4 camera_axis_z;
    float4 model_row0;
    float4 model_row1;
    float4 model_row2;
    float4 camera_params;
    uint4 mesh_params;
    uint4 resource_handles;
};

struct VertexStageOutput
{
    float4 color : COLOR0;
    float4 normal : COLOR1;
    float4 pbr : COLOR2;
    float4 view : COLOR3;
    float4 sv_position : SV_Position;
};

struct DepthVertexStageOutput
{
    float4 sv_position : SV_Position;
};

struct Fragment
{
    float4 color;
};

float3 cube_corner(uint vertex_id)
{
    uint corner_id = 0u;
    uint face = vertex_id / 6u;
    uint tri = vertex_id % 6u;
    if (tri == 0u)
    {
        corner_id = 0u;
    }
    else if (tri == 1u)
    {
        corner_id = 1u;
    }
    else if (tri == 2u)
    {
        corner_id = 2u;
    }
    else if (tri == 3u)
    {
        corner_id = 0u;
    }
    else if (tri == 4u)
    {
        corner_id = 2u;
    }
    else
    {
        corner_id = 3u;
    }

    if (face == 0u)
    {
        float3 corners[4] = { float3(0, 0, 0), float3(0, 0, 1), float3(0, 1, 1), float3(0, 1, 0) };
        return corners[corner_id];
    }
    if (face == 1u)
    {
        float3 corners[4] = { float3(1, 0, 0), float3(1, 1, 0), float3(1, 1, 1), float3(1, 0, 1) };
        return corners[corner_id];
    }
    if (face == 2u)
    {
        float3 corners[4] = { float3(0, 0, 0), float3(1, 0, 0), float3(1, 0, 1), float3(0, 0, 1) };
        return corners[corner_id];
    }
    if (face == 3u)
    {
        float3 corners[4] = { float3(0, 1, 0), float3(0, 1, 1), float3(1, 1, 1), float3(1, 1, 0) };
        return corners[corner_id];
    }
    if (face == 4u)
    {
        float3 corners[4] = { float3(0, 0, 0), float3(0, 1, 0), float3(1, 1, 0), float3(1, 0, 0) };
        return corners[corner_id];
    }
    float3 corners[4] = { float3(0, 0, 1), float3(1, 0, 1), float3(1, 1, 1), float3(0, 1, 1) };
    return corners[corner_id];
}

float3 cube_normal(uint vertex_id)
{
    uint face = vertex_id / 6u;
    if (face == 0u)
    {
        return float3(-1.0, 0.0, 0.0);
    }
    if (face == 1u)
    {
        return float3(1.0, 0.0, 0.0);
    }
    if (face == 2u)
    {
        return float3(0.0, -1.0, 0.0);
    }
    if (face == 3u)
    {
        return float3(0.0, 1.0, 0.0);
    }
    if (face == 4u)
    {
        return float3(0.0, 0.0, -1.0);
    }
    return float3(0.0, 0.0, 1.0);
}

float3 transform_position(RenderParams params, float3 position)
{
    return float3(
        dot(params.model_row0.xyz, position) + params.model_row0.w,
        dot(params.model_row1.xyz, position) + params.model_row1.w,
        dot(params.model_row2.xyz, position) + params.model_row2.w);
}

float3 transform_normal(RenderParams params, float3 normal)
{
    return normalize(float3(
        dot(params.model_row0.xyz, normal),
        dot(params.model_row1.xyz, normal),
        dot(params.model_row2.xyz, normal)));
}

float unpack_unorm8(uint packed, uint shift)
{
    return float((packed >> shift) & 255u) / 255.0;
}

float4 unpack_pbr(uint packed)
{
    return float4(
        unpack_unorm8(packed, 0u),
        max(unpack_unorm8(packed, 8u), 0.04),
        unpack_unorm8(packed, 16u),
        unpack_unorm8(packed, 24u) * 16.0);
}

float4 project_position(RenderParams params, float3 world_position)
{
    float3 camera_origin = float3(params.camera_axis_x.w, params.camera_axis_y.w, params.camera_axis_z.w);
    float3 rel = world_position - camera_origin;
    float camera_x = dot(rel, params.camera_axis_x.xyz);
    float camera_y = dot(rel, params.camera_axis_y.xyz);
    float forward = dot(rel, -params.camera_axis_z.xyz);
    float near_z = params.camera_params.z;
    float far_z = params.camera_params.w;
    if (forward <= near_z)
    {
        return float4(2.0, 2.0, 1.0, 1.0);
    }

    float tan_half_fov = params.camera_params.x;
    float aspect = params.camera_params.y;
    float depth = saturate((forward - near_z) / max(far_z - near_z, 0.0001));
    return float4(
        camera_x / max(tan_half_fov * aspect, 0.0001),
        -camera_y / max(tan_half_fov, 0.0001),
        depth * forward,
        forward);
}

float3 cluster_world_position(RenderParams params, VisibleCluster cluster, uint vertex_id)
{
    float3 local_position = cluster.local_min_size.xyz + cube_corner(vertex_id) * cluster.local_min_size.w;
    return transform_position(params, local_position);
}

bool load_cluster(
    uint vertex_id,
    uint instance_id,
    uint params_handle,
    uint params_index,
    uint clusters_handle,
    out RenderParams params,
    out VisibleCluster cluster,
    out float3 world_position)
{
    StructuredBuffer<RenderParams>.Handle params_buffer = (StructuredBuffer<RenderParams>.Handle)params_handle;
    StructuredBuffer<VisibleCluster>.Handle clusters = (StructuredBuffer<VisibleCluster>.Handle)clusters_handle;
    RenderParams draw_params = params_buffer[params_index];

    if (vertex_id >= draw_params.mesh_params.x || instance_id >= draw_params.mesh_params.y)
    {
        params = draw_params;
        cluster.local_min_size = float4(0.0);
        cluster.color = float4(0.0);
        cluster.meta = uint4(0u);
        world_position = float3(0.0);
        return false;
    }

    cluster = clusters[instance_id];
    params = params_buffer[cluster.meta.x];
    world_position = cluster_world_position(params, cluster, vertex_id);
    return true;
}

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vertex_id: SV_VertexID,
    uint instance_id: SV_InstanceID,
    uniform uint32_t params_handle,
    uniform uint32_t params_index,
    uniform uint32_t clusters_handle,
    uniform uint32_t indirect_handle)
{
    VertexStageOutput output;
    RenderParams params;
    VisibleCluster cluster;
    float3 world_position;
    if (!load_cluster(vertex_id, instance_id, params_handle, params_index, clusters_handle, params, cluster, world_position))
    {
        output.sv_position = float4(2.0, 2.0, 1.0, 1.0);
        output.color = float4(0.0);
        output.normal = float4(0.0);
        output.pbr = float4(0.0);
        output.view = float4(0.0);
        return output;
    }

    float3 world_normal = transform_normal(params, cube_normal(vertex_id));
    float3 camera_origin = float3(params.camera_axis_x.w, params.camera_axis_y.w, params.camera_axis_z.w);
    output.sv_position = project_position(params, world_position);
    output.color = float4(cluster.color.rgb, cluster.local_min_size.w);
    output.normal = float4(world_normal, 0.0);
    output.pbr = unpack_pbr(cluster.meta.y);
    output.view = float4(normalize(camera_origin - world_position), 0.0);
    return output;
}

[shader("vertex")]
DepthVertexStageOutput depthVertexMain(
    uint vertex_id: SV_VertexID,
    uint instance_id: SV_InstanceID,
    uniform uint32_t params_handle,
    uniform uint32_t params_index,
    uniform uint32_t clusters_handle,
    uniform uint32_t indirect_handle)
{
    DepthVertexStageOutput output;
    RenderParams params;
    VisibleCluster cluster;
    float3 world_position;
    if (!load_cluster(vertex_id, instance_id, params_handle, params_index, clusters_handle, params, cluster, world_position))
    {
        output.sv_position = float4(2.0, 2.0, 1.0, 1.0);
        return output;
    }

    output.sv_position = project_position(params, world_position);
    return output;
}

static const float PI = 3.14159265359;

float distribution_ggx(float3 normal, float3 half_vector, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = saturate(dot(normal, half_vector));
    float n_dot_h2 = n_dot_h * n_dot_h;
    float denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 0.00001);
}

float geometry_schlick_ggx(float n_dot_v, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.00001);
}

float geometry_smith(float3 normal, float3 view_dir, float3 light_dir, float roughness)
{
    float n_dot_v = saturate(dot(normal, view_dir));
    float n_dot_l = saturate(dot(normal, light_dir));
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

[shader("fragment")]
Fragment fragmentMain(float4 color: COLOR0, float4 normal: COLOR1, float4 pbr: COLOR2, float4 view: COLOR3) : SV_Target
{
    Fragment output;
    float3 albedo = pow(saturate(color.rgb), float3(2.2));
    float metallic = saturate(pbr.x);
    float roughness = clamp(pbr.y, 0.04, 1.0);
    float specular_strength = saturate(pbr.z);
    float emission_strength = max(pbr.w, 0.0);

    float3 n = normalize(normal.xyz);
    float3 l = normalize(float3(0.35, 0.7, 0.55));
    float n_dot_l = saturate(dot(n, l));
    if (color.a > 6.5)
    {
        float3 cheap = albedo * (0.08 + 1.45 * n_dot_l) + albedo * emission_strength;
        output.color = float4(cheap, 1.0);
        return output;
    }

    float3 v = normalize(view.xyz);
    float3 h = normalize(v + l);
    float n_dot_v = saturate(dot(n, v));

    float3 f0 = lerp(float3(0.08 * specular_strength), albedo, metallic);
    float3 f = fresnel_schlick(saturate(dot(h, v)), f0);
    float d = distribution_ggx(n, h, roughness);
    float g = geometry_smith(n, v, l, roughness);
    float3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    float3 diffuse = (1.0 - f) * (1.0 - metallic) * albedo / PI;

    float3 direct_light = float3(2.2) * (diffuse + specular) * n_dot_l;
    float3 ambient = albedo * 0.04;
    float3 emissive = albedo * emission_strength;
    output.color = float4(ambient + direct_light + emissive, 1.0);
    return output;
}
