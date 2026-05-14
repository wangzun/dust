[[vk::binding(0, 0)]]
__DynamicResource<__DynamicResourceKind.General> resourceHandles[];

[[vk::binding(0, 1)]]
__DynamicResource<__DynamicResourceKind.Sampler> samplerHandles[];

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
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

struct GeneratedVertex {
    float4 position;
    float4 color;
    float4 normal;
};

struct RenderParams {
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

struct VertexStageOutput {
    float4 color : COLOR0;
    float4 normal : COLOR1;
    float4 sv_position : SV_Position;
};

struct Fragment {
    float4 color;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    uint vertex_id : SV_VertexID,
    uniform uint32_t params_handle,
    uniform uint32_t params_index,
    uniform uint32_t vertices_handle,
    uniform uint32_t indirect_handle)
{
    StructuredBuffer<RenderParams>.Handle params_buffer = (StructuredBuffer<RenderParams>.Handle)params_handle;
    StructuredBuffer<GeneratedVertex>.Handle vertices = (StructuredBuffer<GeneratedVertex>.Handle)vertices_handle;
    RenderParams params = params_buffer[params_index];

    VertexStageOutput output;
    if (vertex_id >= params.mesh_params.y)
    {
        output.sv_position = float4(2.0, 2.0, 1.0, 1.0);
        output.color = float4(0.0);
        output.normal = float4(0.0);
        return output;
    }

    GeneratedVertex vertex = vertices[vertex_id];
    float3 camera_origin = float3(params.camera_axis_x.w, params.camera_axis_y.w, params.camera_axis_z.w);
    float3 rel = vertex.position.xyz - camera_origin;
    float camera_x = dot(rel, params.camera_axis_x.xyz);
    float camera_y = dot(rel, params.camera_axis_y.xyz);
    float forward = dot(rel, -params.camera_axis_z.xyz);

    float near_z = params.camera_params.z;
    float far_z = params.camera_params.w;
    if (forward <= near_z)
    {
        output.sv_position = float4(2.0, 2.0, 1.0, 1.0);
        output.color = vertex.color;
        output.normal = vertex.normal;
        return output;
    }

    float tan_half_fov = params.camera_params.x;
    float aspect = params.camera_params.y;
    float depth = saturate((forward - near_z) / max(far_z - near_z, 0.0001));
    output.sv_position = float4(
        camera_x / max(tan_half_fov * aspect, 0.0001),
        -camera_y / max(tan_half_fov, 0.0001),
        depth * forward,
        forward);
    output.color = vertex.color;
    output.normal = vertex.normal;
    return output;
}

[shader("fragment")]
Fragment fragmentMain(float4 color : COLOR0, float4 normal : COLOR1) : SV_Target
{
    Fragment output;
    float3 light_dir = normalize(float3(0.35, 0.7, 0.55));
    float diffuse = saturate(dot(normalize(normal.xyz), light_dir));
    output.color = float4(color.rgb * (0.35 + 0.65 * diffuse), color.a);
    return output;
}
