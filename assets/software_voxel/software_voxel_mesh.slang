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

struct Block {
    uint64_t mask;
    uint coords_packed;
    uint material_ptr;
};

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

uint3 unpack_block_coords(uint packed)
{
    return uint3((packed >> 20) & 1023u, (packed >> 10) & 1023u, packed & 1023u) << 2;
}

uint popcount64(uint64_t value)
{
    return countbits(uint(value & 0xffffffffUL)) + countbits(uint(value >> 32));
}

uint voxel_id(int3 voxel)
{
    return (uint(voxel.x) << 4) | (uint(voxel.y) << 2) | uint(voxel.z);
}

bool occupied(uint64_t mask, int3 voxel)
{
    if (any(voxel < int3(0)) || any(voxel >= int3(4)))
    {
        return false;
    }
    return ((mask >> voxel_id(voxel)) & 1UL) != 0UL;
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

float3 palette_color(RenderParams params, StructuredBuffer<uint32_t>.Handle material_info, StructuredBuffer<uint32_t>.Handle palette_info, Block block, uint hit_voxel_id)
{
    uint64_t before_mask = block.mask & ((1UL << hit_voxel_id) - 1UL);
    uint material_offset = block.material_ptr + popcount64(before_mask);
    uint material_word = material_info[material_offset >> 2];
    uint palette_index = (material_word >> ((material_offset & 3u) * 8u)) & 255u;
    uint packed = palette_info[min(palette_index, 255u)];

    return float3(
        float(packed & 255u),
        float((packed >> 8) & 255u),
        float((packed >> 16) & 255u)) / 255.0;
}

void write_vertex(RenderParams params, RWStructuredBuffer<GeneratedVertex>.Handle vertices, uint index, float3 local_position, float3 color, float3 normal)
{
    vertices[index].position = float4(transform_position(params, local_position), 1.0);
    vertices[index].color = float4(color, 1.0);
    vertices[index].normal = float4(normal, 0.0);
}

float3 face_normal(uint face)
{
    if (face == 0u) { return float3(-1.0, 0.0, 0.0); }
    if (face == 1u) { return float3( 1.0, 0.0, 0.0); }
    if (face == 2u) { return float3(0.0, -1.0, 0.0); }
    if (face == 3u) { return float3(0.0,  1.0, 0.0); }
    if (face == 4u) { return float3(0.0, 0.0, -1.0); }
    return float3(0.0, 0.0, 1.0);
}

void emit_face(RenderParams params, RWStructuredBuffer<GeneratedVertex>.Handle vertices, RWStructuredBuffer<uint>.Handle draw_args, float3 p, uint face, float3 color)
{
    uint base_vertex;
    InterlockedAdd(draw_args[0], 6u, base_vertex);
    if (base_vertex + 6u > params.mesh_params.y)
    {
        return;
    }

    float3 a;
    float3 b;
    float3 c;
    float3 d;
    if (face == 0u)
    {
        a = p + float3(0, 0, 0); b = p + float3(0, 0, 1); c = p + float3(0, 1, 1); d = p + float3(0, 1, 0);
    }
    else if (face == 1u)
    {
        a = p + float3(1, 0, 0); b = p + float3(1, 1, 0); c = p + float3(1, 1, 1); d = p + float3(1, 0, 1);
    }
    else if (face == 2u)
    {
        a = p + float3(0, 0, 0); b = p + float3(1, 0, 0); c = p + float3(1, 0, 1); d = p + float3(0, 0, 1);
    }
    else if (face == 3u)
    {
        a = p + float3(0, 1, 0); b = p + float3(0, 1, 1); c = p + float3(1, 1, 1); d = p + float3(1, 1, 0);
    }
    else if (face == 4u)
    {
        a = p + float3(0, 0, 0); b = p + float3(0, 1, 0); c = p + float3(1, 1, 0); d = p + float3(1, 0, 0);
    }
    else
    {
        a = p + float3(0, 0, 1); b = p + float3(1, 0, 1); c = p + float3(1, 1, 1); d = p + float3(0, 1, 1);
    }

    float3 normal = transform_normal(params, face_normal(face));
    write_vertex(params, vertices, base_vertex + 0u, a, color, normal);
    write_vertex(params, vertices, base_vertex + 1u, b, color, normal);
    write_vertex(params, vertices, base_vertex + 2u, c, color, normal);
    write_vertex(params, vertices, base_vertex + 3u, a, color, normal);
    write_vertex(params, vertices, base_vertex + 4u, c, color, normal);
    write_vertex(params, vertices, base_vertex + 5u, d, color, normal);
}

[numthreads(64, 1, 1)]
void meshMain(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    uniform uint32_t params_handle,
    uniform uint32_t params_index,
    uniform uint32_t vertices_handle,
    uniform uint32_t indirect_handle)
{
    StructuredBuffer<RenderParams>.Handle params_buffer = (StructuredBuffer<RenderParams>.Handle)params_handle;
    RWStructuredBuffer<GeneratedVertex>.Handle vertices = (RWStructuredBuffer<GeneratedVertex>.Handle)vertices_handle;
    RWStructuredBuffer<uint>.Handle draw_args = (RWStructuredBuffer<uint>.Handle)indirect_handle;
    RenderParams params = params_buffer[params_index];
    uint mode = params.mesh_params.z;
    if (mode == 0u)
    {
        if (dispatch_thread_id.x == 0u)
        {
            draw_args[0] = 0u;
            draw_args[1] = 1u;
            draw_args[2] = 0u;
            draw_args[3] = 0u;
        }
        return;
    }

    if (mode == 2u)
    {
        if (dispatch_thread_id.x == 0u)
        {
            draw_args[0] = min(draw_args[0], params.mesh_params.y);
            draw_args[1] = 1u;
            draw_args[2] = 0u;
            draw_args[3] = 0u;
        }
        return;
    }

    uint block_index = dispatch_thread_id.x;
    StructuredBuffer<Block>.Handle geometry_info = (StructuredBuffer<Block>.Handle)params.resource_handles.x;
    StructuredBuffer<uint32_t>.Handle material_info = (StructuredBuffer<uint32_t>.Handle)params.resource_handles.y;
    StructuredBuffer<uint32_t>.Handle palette_info = (StructuredBuffer<uint32_t>.Handle)params.resource_handles.z;

    if (block_index >= params.mesh_params.x)
    {
        return;
    }

    Block block = geometry_info[block_index];
    if (block.mask == 0UL)
    {
        return;
    }

    uint3 block_origin = unpack_block_coords(block.coords_packed);
    for (uint voxel = 0u; voxel < 64u; voxel++)
    {
        if (((block.mask >> voxel) & 1UL) == 0UL)
        {
            continue;
        }

        int3 local_voxel = int3(int(voxel >> 4), int((voxel >> 2) & 3u), int(voxel & 3u));
        float3 p = float3(block_origin + uint3(local_voxel));
        float3 color = palette_color(params, material_info, palette_info, block, voxel);

        if (!occupied(block.mask, local_voxel + int3(-1, 0, 0))) { emit_face(params, vertices, draw_args, p, 0u, color); }
        if (!occupied(block.mask, local_voxel + int3( 1, 0, 0))) { emit_face(params, vertices, draw_args, p, 1u, color); }
        if (!occupied(block.mask, local_voxel + int3(0, -1, 0))) { emit_face(params, vertices, draw_args, p, 2u, color); }
        if (!occupied(block.mask, local_voxel + int3(0,  1, 0))) { emit_face(params, vertices, draw_args, p, 3u, color); }
        if (!occupied(block.mask, local_voxel + int3(0, 0, -1))) { emit_face(params, vertices, draw_args, p, 4u, color); }
        if (!occupied(block.mask, local_voxel + int3(0, 0,  1))) { emit_face(params, vertices, draw_args, p, 5u, color); }
    }
}
