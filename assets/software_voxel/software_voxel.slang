struct Block {
    uint64_t mask;
    uint coords_packed;
    uint material_ptr;
};

struct RenderParams {
    float4 camera_row0;
    float4 camera_row1;
    float4 camera_row2;
    float4 model_translation_unit;
    float4 camera_params;
    Block* geometry_info;
    uint32_t* material_info;
    uint32_t* palette_info;
    uint block_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
    uint _pad4;
};

[[vk::binding(0, 0)]] ConstantBuffer<RenderParams> params;

[format("rgba8")]
[[vk::binding(1, 0)]] RWTexture2D<float4> output_image;

uint3 unpack_block_coords(uint packed)
{
    return uint3((packed >> 20) & 1023u, (packed >> 10) & 1023u, packed & 1023u) << 2;
}

float3 sky_color(float3 direction)
{
    float t = saturate(direction.y * 0.5 + 0.5);
    return lerp(float3(0.025, 0.03, 0.04), float3(0.46, 0.62, 0.86), t);
}

bool intersect_aabb(
    float3 ray_origin,
    float3 inv_dir,
    float3 box_min,
    float3 box_max,
    out float t_min,
    out float t_max)
{
    float3 t0 = (box_min - ray_origin) * inv_dir;
    float3 t1 = (box_max - ray_origin) * inv_dir;
    float3 t_near = min(t0, t1);
    float3 t_far = max(t0, t1);

    t_min = max(max(t_near.x, t_near.y), max(t_near.z, 0.001));
    t_max = min(t_far.x, min(t_far.y, t_far.z));
    return t_max >= t_min;
}

uint popcount64(uint64_t value)
{
    return countbits(uint(value & 0xffffffffUL)) + countbits(uint(value >> 32));
}

uint voxel_id(int3 voxel)
{
    return (uint(voxel.x) << 4) | (uint(voxel.y) << 2) | uint(voxel.z);
}

bool voxel_occupied(uint64_t mask, int3 voxel)
{
    uint id = voxel_id(voxel);
    return ((mask >> id) & 1UL) != 0UL;
}

bool trace_block(
    Block block,
    float3 ray_origin,
    float3 ray_dir,
    float t_enter,
    float t_exit,
    float3 block_min,
    float voxel_size,
    out float hit_t,
    out float3 hit_normal,
    out uint hit_voxel_id)
{
    hit_t = 0.0;
    hit_voxel_id = 0u;
    float start_t = max(t_enter, 0.001);
    float3 local_origin = (ray_origin - block_min) / voxel_size;
    float3 local_dir = ray_dir / voxel_size;
    float3 local_pos = local_origin + local_dir * start_t;
    int3 voxel = int3(clamp(floor(local_pos), float3(0.0), float3(3.0)));

    int3 step_dir = int3(
        local_dir.x >= 0.0 ? 1 : -1,
        local_dir.y >= 0.0 ? 1 : -1,
        local_dir.z >= 0.0 ? 1 : -1);

    float3 next_boundary = float3(
        step_dir.x > 0 ? float(voxel.x + 1) : float(voxel.x),
        step_dir.y > 0 ? float(voxel.y + 1) : float(voxel.y),
        step_dir.z > 0 ? float(voxel.z + 1) : float(voxel.z));
    float3 t_max = (next_boundary - local_origin) / local_dir;
    float3 t_delta = abs(1.0 / local_dir);

    t_max = select(abs(local_dir) > float3(1e-7), t_max, float3(1e30));
    t_delta = select(abs(local_dir) > float3(1e-7), t_delta, float3(1e30));

    hit_normal = -sign(ray_dir);
    for (uint step_index = 0; step_index < 16u; step_index++)
    {
        if (all(voxel >= int3(0)) && all(voxel < int3(4)) && voxel_occupied(block.mask, voxel))
        {
            hit_t = start_t;
            hit_voxel_id = voxel_id(voxel);
            return true;
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z)
        {
            start_t = t_max.x;
            voxel.x += step_dir.x;
            t_max.x += t_delta.x;
            hit_normal = float3(-float(step_dir.x), 0.0, 0.0);
        }
        else if (t_max.y <= t_max.z)
        {
            start_t = t_max.y;
            voxel.y += step_dir.y;
            t_max.y += t_delta.y;
            hit_normal = float3(0.0, -float(step_dir.y), 0.0);
        }
        else
        {
            start_t = t_max.z;
            voxel.z += step_dir.z;
            t_max.z += t_delta.z;
            hit_normal = float3(0.0, 0.0, -float(step_dir.z));
        }

        if (start_t > t_exit)
        {
            break;
        }
    }

    return false;
}

float3 palette_color(Block block, uint hit_voxel_id)
{
    uint64_t before_mask = block.mask & ((1UL << hit_voxel_id) - 1UL);
    uint material_offset = block.material_ptr + popcount64(before_mask);
    uint material_word = params.material_info[material_offset >> 2];
    uint palette_index = (material_word >> ((material_offset & 3u) * 8u)) & 255u;
    uint packed = params.palette_info[min(palette_index, 255u)];

    return float3(
        float(packed & 255u),
        float((packed >> 8) & 255u),
        float((packed >> 16) & 255u)) / 255.0;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint width;
    uint height;
    output_image.GetDimensions(width, height);
    if (dispatch_thread_id.x >= width || dispatch_thread_id.y >= height)
    {
        return;
    }

    float2 uv = (float2(dispatch_thread_id.xy) + 0.5) / float2(width, height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    ndc.x *= float(width) / max(float(height), 1.0);
    ndc *= params.camera_params.x;

    float3 camera_x = params.camera_row0.xyz;
    float3 camera_y = params.camera_row1.xyz;
    float3 camera_z = params.camera_row2.xyz;
    float3 ray_origin = float3(params.camera_row0.w, params.camera_row1.w, params.camera_row2.w);
    float3 ray_dir = normalize(camera_x * ndc.x + camera_y * ndc.y - camera_z);
    float3 inv_dir = 1.0 / ray_dir;

    float best_t = 1e30;
    float3 best_normal = float3(0.0, 1.0, 0.0);
    float3 best_color = float3(0.0);

    if (params.geometry_info != nullptr && params.material_info != nullptr && params.palette_info != nullptr)
    {
        float voxel_size = max(params.model_translation_unit.w, 0.0001);
        float3 model_translation = params.model_translation_unit.xyz;

        for (uint block_index = 0; block_index < params.block_count; block_index++)
        {
            Block block = params.geometry_info[block_index];
            if (block.mask == 0UL)
            {
                continue;
            }

            float3 block_min = model_translation + float3(unpack_block_coords(block.coords_packed)) * voxel_size;
            float3 block_max = block_min + float3(4.0 * voxel_size);
            float t_enter;
            float t_exit;
            if (!intersect_aabb(ray_origin, inv_dir, block_min, block_max, t_enter, t_exit) || t_enter >= best_t)
            {
                continue;
            }

            float hit_t;
            float3 hit_normal;
            uint hit_voxel_id;
            if (trace_block(block, ray_origin, ray_dir, t_enter, min(t_exit, best_t), block_min, voxel_size, hit_t, hit_normal, hit_voxel_id)
                && hit_t < best_t)
            {
                best_t = hit_t;
                best_normal = hit_normal;
                best_color = palette_color(block, hit_voxel_id);
            }
        }
    }

    float3 color = sky_color(ray_dir);
    if (best_t < 1e29)
    {
        float3 light_dir = normalize(float3(-0.35, 0.85, -0.38));
        float diffuse = saturate(dot(best_normal, light_dir));
        float rim = pow(saturate(1.0 - abs(dot(best_normal, -ray_dir))), 3.0);
        color = best_color * (0.22 + diffuse * 0.78) + rim * 0.08;
    }

    output_image[int2(dispatch_thread_id.xy)] = float4(color, 1.0);
}
