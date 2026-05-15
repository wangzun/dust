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

struct Block
{
    uint64_t mask;
    uint coords_packed;
    uint material_ptr;
};

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

uint3 unpack_block_coords(uint packed)
{
    return uint3((packed >> 20) & 1023u, (packed >> 10) & 1023u, packed & 1023u) << 2;
}

uint popcount64(uint64_t value)
{
    return countbits(uint(value & 0xffffffffUL)) + countbits(uint(value >> 32));
}

uint voxel_id(uint3 voxel)
{
    return (voxel.x << 4) | (voxel.y << 2) | voxel.z;
}

bool occupied(uint64_t mask, uint3 voxel)
{
    return ((mask >> voxel_id(voxel)) & 1UL) != 0UL;
}

float3 transform_position(RenderParams params, float3 position)
{
    return float3(
        dot(params.model_row0.xyz, position) + params.model_row0.w,
        dot(params.model_row1.xyz, position) + params.model_row1.w,
        dot(params.model_row2.xyz, position) + params.model_row2.w);
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
               float((packed >> 16) & 255u)) /
           255.0;
}

bool block_visible(RenderParams params, float3 local_min, float size)
{
    float3 center = local_min + float3(size * 0.5);
    float3 world_center = transform_position(params, center);
    float3 camera_origin = float3(params.camera_axis_x.w, params.camera_axis_y.w, params.camera_axis_z.w);
    float3 rel = world_center - camera_origin;
    float camera_x = dot(rel, params.camera_axis_x.xyz);
    float camera_y = dot(rel, params.camera_axis_y.xyz);
    float forward = dot(rel, -params.camera_axis_z.xyz);

    float radius = max(length(params.model_row0.xyz), max(length(params.model_row1.xyz), length(params.model_row2.xyz))) * size * 0.9;
    float near_z = params.camera_params.z;
    float far_z = params.camera_params.w;
    if (forward + radius <= near_z || forward - radius >= far_z)
    {
        return false;
    }

    float tan_half_fov = params.camera_params.x;
    float aspect = params.camera_params.y;
    if (abs(camera_x) > max(forward, near_z) * tan_half_fov * aspect + radius)
    {
        return false;
    }
    if (abs(camera_y) > max(forward, near_z) * tan_half_fov + radius)
    {
        return false;
    }
    return true;
}

static const float32_t distance_lod_thresholds[3] = { 36.0, 110.0, 10000.0 };

uint select_lod_group_size(RenderParams params, float3 local_min)
{
    float3 world_center = transform_position(params, local_min + float3(2.0));
    float3 camera_origin = float3(params.camera_axis_x.w, params.camera_axis_y.w, params.camera_axis_z.w);
    float distance = length(world_center - camera_origin);

    if (distance < distance_lod_thresholds[0])
    {
        return 1u;
    }
    if (distance < distance_lod_thresholds[1])
    {
        return 2u;
    }
    return 4u;
}

void emit_cluster(
    RenderParams params,
    RWStructuredBuffer<VisibleCluster>.Handle clusters,
    RWStructuredBuffer<uint>.Handle draw_args,
    uint params_index,
    float3 local_min,
    float size,
    float3 color)
{
    uint cluster_index;
    InterlockedAdd(draw_args[1], 1u, cluster_index);
    if (cluster_index >= params.mesh_params.y)
    {
        return;
    }

    clusters[cluster_index].local_min_size = float4(local_min, size);
    clusters[cluster_index].color = float4(color, 1.0);
    clusters[cluster_index].meta = uint4(params_index, 0u, 0u, 0u);
}

void emit_group(
    RenderParams params,
    RWStructuredBuffer<VisibleCluster>.Handle clusters,
    RWStructuredBuffer<uint>.Handle draw_args,
    StructuredBuffer<uint32_t>.Handle material_info,
    StructuredBuffer<uint32_t>.Handle palette_info,
    Block block,
    uint params_index,
    uint3 block_origin,
    uint3 group_origin,
    uint group_size)
{
    float3 color_sum = float3(0.0);
    uint color_count = 0u;

    for (uint x = 0u; x < group_size; x++)
    {
        for (uint y = 0u; y < group_size; y++)
        {
            for (uint z = 0u; z < group_size; z++)
            {
                uint3 local_voxel = group_origin + uint3(x, y, z);
                if (any(local_voxel >= uint3(4u)))
                {
                    continue;
                }
                if (!occupied(block.mask, local_voxel))
                {
                    continue;
                }
                color_sum += palette_color(params, material_info, palette_info, block, voxel_id(local_voxel));
                color_count++;
            }
        }
    }

    if (color_count == 0u)
    {
        return;
    }

    float size = float(group_size);
    float3 local_min = float3(block_origin + group_origin);
    if (!block_visible(params, local_min, size))
    {
        return;
    }
    emit_cluster(params, clusters, draw_args, params_index, local_min, size, color_sum / float(color_count));
}

[numthreads(64, 1, 1)]
void meshMain(
    uint3 dispatch_thread_id: SV_DispatchThreadID,
    uniform uint32_t params_handle,
    uniform uint32_t params_index,
    uniform uint32_t clusters_handle,
    uniform uint32_t indirect_handle)
{
    StructuredBuffer<RenderParams>.Handle params_buffer = (StructuredBuffer<RenderParams>.Handle)params_handle;
    RWStructuredBuffer<VisibleCluster>.Handle clusters = (RWStructuredBuffer<VisibleCluster>.Handle)clusters_handle;
    RWStructuredBuffer<uint>.Handle draw_args = (RWStructuredBuffer<uint>.Handle)indirect_handle;
    RenderParams control = params_buffer[params_index];
    uint mode = control.mesh_params.z;

    if (mode == 0u)
    {
        if (dispatch_thread_id.x == 0u)
        {
            draw_args[0] = 36u;
            draw_args[1] = 0u;
            draw_args[2] = 0u;
            draw_args[3] = 0u;
        }
        return;
    }

    if (mode == 2u)
    {
        if (dispatch_thread_id.x == 0u)
        {
            draw_args[0] = 36u;
            draw_args[1] = min(draw_args[1], control.mesh_params.y);
            draw_args[2] = 0u;
            draw_args[3] = 0u;
        }
        return;
    }

    uint global_block_index = dispatch_thread_id.x;
    uint instance_count = control.mesh_params.x;
    uint first_instance_param = control.mesh_params.w;
    uint low = 0u;
    uint high = instance_count;
    while (low < high)
    {
        uint mid = (low + high) >> 1u;
        uint candidate_index = first_instance_param + mid;
        RenderParams candidate = params_buffer[candidate_index];
        if (candidate.mesh_params.w <= global_block_index)
        {
            low = mid + 1u;
        }
        else
        {
            high = mid;
        }
    }
    if (low == 0u)
    {
        return;
    }

    uint instance_param_index = first_instance_param + low - 1u;
    RenderParams params = params_buffer[instance_param_index];
    uint block_count = params.mesh_params.x;
    uint block_base = params.mesh_params.w;
    if (global_block_index < block_base || global_block_index >= block_base + block_count)
    {
        return;
    }
    uint local_block_index = global_block_index - block_base;

    StructuredBuffer<Block>.Handle geometry_info = (StructuredBuffer<Block>.Handle)params.resource_handles.x;
    StructuredBuffer<uint32_t>.Handle material_info = (StructuredBuffer<uint32_t>.Handle)params.resource_handles.y;
    StructuredBuffer<uint32_t>.Handle palette_info = (StructuredBuffer<uint32_t>.Handle)params.resource_handles.z;

    Block block = geometry_info[local_block_index];
    if (block.mask == 0UL)
    {
        return;
    }

    uint3 block_origin = unpack_block_coords(block.coords_packed);
    uint group_size = select_lod_group_size(params, float3(block_origin));
    for (uint x = 0u; x < 4u; x += group_size)
    {
        for (uint y = 0u; y < 4u; y += group_size)
        {
            for (uint z = 0u; z < 4u; z += group_size)
            {
                emit_group(
                    params,
                    clusters,
                    draw_args,
                    material_info,
                    palette_info,
                    block,
                    instance_param_index,
                    block_origin,
                    uint3(x, y, z),
                    group_size);
            }
        }
    }
}
