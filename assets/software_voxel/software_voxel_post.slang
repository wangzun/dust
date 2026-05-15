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

struct VertexOutput
{
    float2 uv : TEXCOORD0;
    float4 sv_position : SV_Position;
};

struct Fragment
{
    float4 color;
};

[shader("vertex")]
VertexOutput vertexMain(uint32_t vertex_id : SV_VertexID)
{
    float2 uv = float2((vertex_id << 1u) & 2u, vertex_id & 2u);

    VertexOutput output;
    output.uv = uv;
    output.sv_position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

float bayer4(uint2 pixel)
{
    uint2 p = pixel & 3u;
    uint index = p.x + p.y * 4u;
    if (index == 0u) return 0.0 / 16.0;
    if (index == 1u) return 8.0 / 16.0;
    if (index == 2u) return 2.0 / 16.0;
    if (index == 3u) return 10.0 / 16.0;
    if (index == 4u) return 12.0 / 16.0;
    if (index == 5u) return 4.0 / 16.0;
    if (index == 6u) return 14.0 / 16.0;
    if (index == 7u) return 6.0 / 16.0;
    if (index == 8u) return 3.0 / 16.0;
    if (index == 9u) return 11.0 / 16.0;
    if (index == 10u) return 1.0 / 16.0;
    if (index == 11u) return 9.0 / 16.0;
    if (index == 12u) return 15.0 / 16.0;
    if (index == 13u) return 7.0 / 16.0;
    if (index == 14u) return 13.0 / 16.0;
    return 5.0 / 16.0;
}

float3 quantize_color(float3 color, uint2 pixel)
{
    const float levels = 7.0;
    float threshold = (bayer4(pixel) - 0.5) / levels;
    return saturate(floor(saturate(color) * levels + 0.5 + threshold) / levels);
}

float3 sample_color(Texture2D.Handle image, SamplerState.Handle sampler, float2 uv)
{
    return image.Sample(sampler, saturate(uv)).rgb;
}

[shader("fragment")]
Fragment fragmentMain(
    VertexOutput input,
    uniform uint32_t color_handle,
    uniform uint32_t sampler_handle,
    uniform float2 extent,
    uniform float pixel_size,
    uniform float outline_strength)
{
    Texture2D.Handle color_texture = (Texture2D.Handle)color_handle;
    SamplerState.Handle color_sampler = (SamplerState.Handle)sampler_handle;

    float2 safe_extent = max(extent, float2(1.0));
    float block_size = max(pixel_size, 1.0);
    float2 screen_pixel = input.uv * safe_extent;
    float2 block = floor(screen_pixel / block_size);
    float2 block_center = (block + 0.5) * block_size;
    float2 block_uv = block_center / safe_extent;
    float2 block_texel = float2(block_size, block_size) / safe_extent;

    float3 color = sample_color(color_texture, color_sampler, block_uv);
    float3 right = sample_color(color_texture, color_sampler, block_uv + float2(block_texel.x, 0.0));
    float3 left = sample_color(color_texture, color_sampler, block_uv - float2(block_texel.x, 0.0));
    float3 up = sample_color(color_texture, color_sampler, block_uv + float2(0.0, block_texel.y));
    float3 down = sample_color(color_texture, color_sampler, block_uv - float2(0.0, block_texel.y));

    float edge_delta = max(max(length(color - right), length(color - left)), max(length(color - up), length(color - down)));
    float edge = smoothstep(0.10, 0.28, edge_delta * outline_strength);

    uint2 block_pixel = uint2(block);
    float3 quantized = quantize_color(color, block_pixel);

    float2 cell = frac(screen_pixel / block_size);
    float bevel_shadow = max(1.0 - min(cell.x, cell.y) * 10.0, 0.0);
    float bevel_light = max(max(cell.x, cell.y) * 10.0 - 9.0, 0.0);
    float bevel = 1.0 - bevel_shadow * 0.08 + bevel_light * 0.06;

    float3 outlined = lerp(quantized * bevel, quantized * 0.18, edge);

    Fragment output;
    output.color = float4(saturate(outlined), 1.0);
    return output;
}
