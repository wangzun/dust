#![allow(dead_code)]

use spirv_std::{
    glam::{UVec2, Vec2, Vec3, Vec4},
    image::Image2d,
    spirv,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostPushConstants {
    pub color_handle: u32,
    pub sampler_handle: u32,
    pub extent: Vec2,
    pub pixel_size: f32,
    pub outline_strength: f32,
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn saturate_vec2(value: Vec2) -> Vec2 {
    value.clamp(Vec2::ZERO, Vec2::ONE)
}

fn saturate_vec3(value: Vec3) -> Vec3 {
    value.clamp(Vec3::ZERO, Vec3::ONE)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = saturate((value - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

fn bayer4(pixel: UVec2) -> f32 {
    let p = pixel & UVec2::splat(3);
    let index = p.x + p.y * 4;
    if index == 0 {
        0.0 / 16.0
    } else if index == 1 {
        8.0 / 16.0
    } else if index == 2 {
        2.0 / 16.0
    } else if index == 3 {
        10.0 / 16.0
    } else if index == 4 {
        12.0 / 16.0
    } else if index == 5 {
        4.0 / 16.0
    } else if index == 6 {
        14.0 / 16.0
    } else if index == 7 {
        6.0 / 16.0
    } else if index == 8 {
        3.0 / 16.0
    } else if index == 9 {
        11.0 / 16.0
    } else if index == 10 {
        1.0 / 16.0
    } else if index == 11 {
        9.0 / 16.0
    } else if index == 12 {
        15.0 / 16.0
    } else if index == 13 {
        7.0 / 16.0
    } else if index == 14 {
        13.0 / 16.0
    } else {
        5.0 / 16.0
    }
}

fn quantize_color(color: Vec3, pixel: UVec2) -> Vec3 {
    let levels = 7.0;
    let threshold = (bayer4(pixel) - 0.5) / levels;
    saturate_vec3((saturate_vec3(color) * levels + Vec3::splat(0.5 + threshold)).floor() / levels)
}

fn sample_color(color_handle: u32, sampler_handle: u32, uv: Vec2) -> Vec3 {
    let image = dst_heap::image_from_u32::<Image2d>(color_handle);
    let sampler = dst_heap::sampler_from_u32(sampler_handle);
    image.sample(sampler, saturate_vec2(uv)).truncate()
}

#[spirv(vertex(entry_point_name = "vertexMain"))]
pub fn vertex_main(
    #[spirv(vertex_index)] vertex_index: u32,
    #[spirv(base_vertex)] base_vertex: u32,
    #[spirv(location = 0)] out_uv: &mut Vec2,
    #[spirv(position)] out_position: &mut Vec4,
) {
    let vertex_id = vertex_index.wrapping_sub(base_vertex);
    let uv = Vec2::new(
        ((vertex_id << 1) & 2) as f32,
        (vertex_id & 2) as f32,
    );

    *out_uv = uv;
    *out_position = Vec4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
}

#[spirv(fragment(entry_point_name = "fragmentMain"))]
pub fn fragment_main(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(push_constant)] constants: &PostPushConstants,
    #[spirv(location = 0)] out_color: &mut Vec4,
) {
    let safe_extent = constants.extent.max(Vec2::ONE);
    let block_size = constants.pixel_size.max(1.0);
    let screen_pixel = uv * safe_extent;
    let block = (screen_pixel / block_size).floor();
    let block_center = (block + Vec2::splat(0.5)) * block_size;
    let block_uv = block_center / safe_extent;
    let block_texel = Vec2::splat(block_size) / safe_extent;

    let color = sample_color(constants.color_handle, constants.sampler_handle, block_uv);
    let right = sample_color(
        constants.color_handle,
        constants.sampler_handle,
        block_uv + Vec2::new(block_texel.x, 0.0),
    );
    let left = sample_color(
        constants.color_handle,
        constants.sampler_handle,
        block_uv - Vec2::new(block_texel.x, 0.0),
    );
    let up = sample_color(
        constants.color_handle,
        constants.sampler_handle,
        block_uv + Vec2::new(0.0, block_texel.y),
    );
    let down = sample_color(
        constants.color_handle,
        constants.sampler_handle,
        block_uv - Vec2::new(0.0, block_texel.y),
    );

    let edge_delta = (color - right)
        .length()
        .max((color - left).length())
        .max((color - up).length().max((color - down).length()));
    let edge = smoothstep(0.10, 0.28, edge_delta * constants.outline_strength);

    let block_pixel = block.as_uvec2();
    let quantized = quantize_color(color, block_pixel);

    let cell = (screen_pixel / block_size).fract();
    let bevel_shadow = (1.0 - cell.x.min(cell.y) * 10.0).max(0.0);
    let bevel_light = (cell.x.max(cell.y) * 10.0 - 9.0).max(0.0);
    let bevel = 1.0 - bevel_shadow * 0.08 + bevel_light * 0.06;

    let outlined = (quantized * bevel).lerp(quantized * 0.18, edge);
    *out_color = saturate_vec3(outlined).extend(1.0);
}
