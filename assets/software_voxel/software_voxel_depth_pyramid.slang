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

[numthreads(8, 8, 1)]
void depthPyramidMain(
    uint3 dispatch_thread_id: SV_DispatchThreadID,
    uniform uint32_t pyramid_handle,
    uniform uint32_t src_offset,
    uniform uint32_t dst_offset,
    uniform uint32_t src_width,
    uniform uint32_t src_height,
    uniform uint32_t dst_width,
    uniform uint32_t dst_height,
    uniform uint32_t _pad)
{
    if (dispatch_thread_id.x >= dst_width || dispatch_thread_id.y >= dst_height)
    {
        return;
    }

    RWStructuredBuffer<float>.Handle pyramid = (RWStructuredBuffer<float>.Handle)pyramid_handle;
    uint2 dst = dispatch_thread_id.xy;
    uint2 src0 = min(dst * 2u, uint2(src_width - 1u, src_height - 1u));
    uint2 src1 = min(src0 + uint2(1u, 0u), uint2(src_width - 1u, src_height - 1u));
    uint2 src2 = min(src0 + uint2(0u, 1u), uint2(src_width - 1u, src_height - 1u));
    uint2 src3 = min(src0 + uint2(1u, 1u), uint2(src_width - 1u, src_height - 1u));

    float d0 = pyramid[src_offset + src0.y * src_width + src0.x];
    float d1 = pyramid[src_offset + src1.y * src_width + src1.x];
    float d2 = pyramid[src_offset + src2.y * src_width + src2.x];
    float d3 = pyramid[src_offset + src3.y * src_width + src3.x];
    pyramid[dst_offset + dst.y * dst_width + dst.x] = max(max(d0, d1), max(d2, d3));
}
