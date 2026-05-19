use spirv_std::{glam::UVec3, spirv};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DepthPyramidPushConstants {
    pub pyramid_handle: u32,
    pub src_offset: u32,
    pub dst_offset: u32,
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub _pad: u32,
}

#[spirv(compute(threads(8, 8, 1), entry_point_name = "main"))]
pub fn depth_pyramid_main(
    #[spirv(global_invocation_id)] dispatch_thread_id: UVec3,
    #[spirv(push_constant)] constants: &DepthPyramidPushConstants,
) {
    if dispatch_thread_id.x >= constants.dst_width || dispatch_thread_id.y >= constants.dst_height {
        return;
    }

    let dst_x = dispatch_thread_id.x;
    let dst_y = dispatch_thread_id.y;
    let src_max_x = constants.src_width - 1;
    let src_max_y = constants.src_height - 1;
    let src_x0 = dst_x * 2;
    let src_y0 = dst_y * 2;
    let src_x = if src_x0 < src_max_x {
        src_x0
    } else {
        src_max_x
    };
    let src_y = if src_y0 < src_max_y {
        src_y0
    } else {
        src_max_y
    };
    let src_x1_raw = src_x + 1;
    let src_y1_raw = src_y + 1;
    let src_x1 = if src_x1_raw < src_max_x {
        src_x1_raw
    } else {
        src_max_x
    };
    let src_y1 = if src_y1_raw < src_max_y {
        src_y1_raw
    } else {
        src_max_y
    };

    let src_offset = constants.src_offset as usize;
    let dst_offset = constants.dst_offset as usize;
    let src_width = constants.src_width as usize;
    let dst_width = constants.dst_width as usize;
    let pyramid = dst_heap::storage_buffer_from_u32::<f32>(constants.pyramid_handle);

    let src_index0 = src_offset + src_y as usize * src_width + src_x as usize;
    let src_index1 = src_offset + src_y as usize * src_width + src_x1 as usize;
    let src_index2 = src_offset + src_y1 as usize * src_width + src_x as usize;
    let src_index3 = src_offset + src_y1 as usize * src_width + src_x1 as usize;
    let d0 = pyramid[src_index0];
    let d1 = pyramid[src_index1];
    let d2 = pyramid[src_index2];
    let d3 = pyramid[src_index3];
    let max01 = if d0 > d1 { d0 } else { d1 };
    let max23 = if d2 > d3 { d2 } else { d3 };
    let max_depth = if max01 > max23 { max01 } else { max23 };

    let dst_index = dst_offset + dst_y as usize * dst_width + dst_x as usize;
    pyramid[dst_index] = max_depth;
}
