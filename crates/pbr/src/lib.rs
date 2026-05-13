pub mod camera;

use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use dust_vox::{VoxGeometry, VoxMaterial, VoxModel, VoxPalette};
use pumicite::{
    Allocator,
    ash::vk,
    buffer::BufferLike,
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, ImageLike},
    sync::GPUMutex,
    tracking::{Access, ResourceState},
    utils::AsVkHandle,
};

use crate::camera::Camera;

pub struct PbrRenderPlugin;

impl Plugin for PbrRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SoftwareVoxelRenderPlugin);
    }
}

pub struct SoftwareVoxelRenderPlugin;

impl Plugin for SoftwareVoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<pumicite::device::DeviceBuilder>| {
                device_builder
                    .enable_extension::<pumicite::ash::khr::push_descriptor::Meta>()
                    .unwrap();
                device_builder
                    .enable_extension::<pumicite::ash::khr::buffer_device_address::Meta>()
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFeatures| {
                        &mut features.shader_int64
                    })
                    .unwrap();
                device_builder
                    .enable_feature::<vk::PhysicalDeviceBufferDeviceAddressFeatures>(|features| {
                        &mut features.buffer_device_address
                    })
                    .unwrap();
            })
            .before(bevy_pumicite::CreateDevice),
        );

        app.add_systems(Startup, setup.after(bevy_pumicite::CreateDevice));
        app.add_systems(
            PostUpdate,
            (ensure_render_target, render)
                .chain()
                .in_set(DefaultRenderSet),
        );
    }
}

#[derive(Resource)]
struct SoftwareVoxelPipeline {
    draw: Handle<ComputePipeline>,
}

#[derive(Resource)]
struct SoftwareRenderTarget {
    view: GPUMutex<FullImageView<Image>>,
    state: ResourceState,
    extent: UVec2,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftwareVoxelUniform {
    camera_rows: [[f32; 4]; 3],
    model_translation_unit: [f32; 4],
    camera_params: [f32; 4],
    geometry_addr: u64,
    material_addr: u64,
    palette_addr: u64,
    block_count: u32,
    _pad: [u32; 5],
}

const _: () = assert!(std::mem::size_of::<SoftwareVoxelUniform>() == 128);

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SoftwareVoxelPipeline {
        draw: asset_server.load("software_voxel/software_voxel.comp.pipeline.ron"),
    });
}

fn ensure_render_target(
    mut commands: Commands,
    current_target: Option<Res<SoftwareRenderTarget>>,
    allocator: Res<Allocator>,
    swapchain_images: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain_image) = swapchain_images.single() else {
        return;
    };
    let Some(current) = swapchain_image.current_image() else {
        return;
    };
    let extent = UVec2::new(current.extent().x, current.extent().y);
    if current_target
        .as_ref()
        .is_some_and(|target| target.extent == extent)
    {
        return;
    }

    let image = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"Software Voxel Render Target");

    commands.insert_resource(SoftwareRenderTarget {
        view: GPUMutex::new(
            image
                .create_full_view()
                .unwrap()
                .with_name(c"Software Voxel Render Target View"),
        ),
        state: ResourceState::default(),
        extent,
    });
}

fn render(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: SubmissionState,
    pipeline: Res<SoftwareVoxelPipeline>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut ring_buffer: ResMut<UniformRingBuffer>,
    mut target: ResMut<SoftwareRenderTarget>,
    cameras: Query<(&Camera, &GlobalTransform), With<bevy::window::PrimaryWindow>>,
    models: Query<(&VoxModel, Option<&GlobalTransform>)>,
    geometries: Res<Assets<VoxGeometry>>,
    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(pipeline) = compute_pipelines.get(&pipeline.draw).cloned() else {
        return;
    };

    let mut uniform = empty_uniform(camera, camera_transform);
    for (model, transform) in models.iter() {
        let Some(geometry) = geometries.get(&model.geometry) else {
            continue;
        };
        let Some(material) = materials.get(&model.material) else {
            continue;
        };
        let Some(palette) = palettes.get(&model.palette) else {
            continue;
        };
        let block_count = geometry.tree.pools()[0].used_capacity();
        if block_count == 0 {
            continue;
        }

        let translation = transform
            .map(|transform| transform.translation())
            .unwrap_or(Vec3::ZERO);
        uniform.model_translation_unit = [
            translation.x,
            translation.y,
            translation.z,
            geometry.unit_size,
        ];
        uniform.geometry_addr = geometry.tree.pools()[0].storage().device_address();
        uniform.material_addr = material.buffer.device_address();
        uniform.palette_addr = palette.device_address();
        uniform.block_count = block_count;
        break;
    }

    state.record(|encoder| {
        let mut uniform_buffer =
            ring_buffer.allocate_buffer(std::mem::size_of::<SoftwareVoxelUniform>() as u64, 128);
        uniform_buffer
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(bytemuck::bytes_of(&uniform));
        let uniform_buffer = encoder.retain(uniform_buffer);

        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let output_view = encoder.lock(&target.view, vk::PipelineStageFlags2::COMPUTE_SHADER);

        encoder.use_image_resource(
            output_view.image(),
            &mut target.state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        encoder.emit_barriers();

        let pipeline = encoder.retain(pipeline.into_inner());
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: uniform_buffer.vk_handle(),
            offset: uniform_buffer.offset(),
            range: uniform_buffer.size(),
        };
        let image_info = vk::DescriptorImageInfo {
            image_view: output_view.vk_handle(),
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        };
        encoder.push_descriptor_set(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout(),
            0,
            &[
                vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &buffer_info,
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: &image_info,
                    ..Default::default()
                },
            ],
        );

        encoder.dispatch(UVec3::new(
            target.extent.x.div_ceil(8),
            target.extent.y.div_ceil(8),
            1,
        ));

        let current_swapchain_image =
            encoder.lock(current_swapchain_image, vk::PipelineStageFlags2::BLIT);
        encoder.use_image_resource(
            output_view.image(),
            &mut target.state,
            Access::BLIT_SRC,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::BLIT_DST,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder.blit_image_with_layout(
            output_view.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            current_swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                src_offsets: [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: target.extent.x as i32,
                        y: target.extent.y as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                dst_offsets: [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: current_swapchain_image.extent().x as i32,
                        y: current_swapchain_image.extent().y as i32,
                        z: 1,
                    },
                ],
            }],
            vk::Filter::NEAREST,
        );
    });
}

fn empty_uniform(camera: &Camera, transform: &GlobalTransform) -> SoftwareVoxelUniform {
    let affine = transform.affine();
    let x = affine.matrix3.x_axis;
    let y = affine.matrix3.y_axis;
    let z = affine.matrix3.z_axis;
    let w = affine.translation;

    SoftwareVoxelUniform {
        camera_rows: [
            [x.x, y.x, z.x, w.x],
            [x.y, y.y, z.y, w.y],
            [x.z, y.z, z.z, w.z],
        ],
        model_translation_unit: [0.0, 0.0, 0.0, 1.0],
        camera_params: [camera.tan_half_fov(), 0.0, 0.0, 0.0],
        geometry_addr: 0,
        material_addr: 0,
        palette_addr: 0,
        block_count: 0,
        _pad: [0; 5],
    }
}
