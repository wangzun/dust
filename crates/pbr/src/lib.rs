pub mod camera;

use std::{collections::HashMap, sync::Arc};

use bevy::math::Affine3A;
use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use dust_vox::{
    BindlessBufferHandle, BufferDescriptor, VoxGeometry, VoxMaterial, VoxModel, VoxPalette,
};
use pumicite::{
    Allocator,
    ash::vk,
    buffer::{Buffer, BufferLike},
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, ImageLike},
    sync::GPUMutex,
    tracking::{Access, ResourceState},
};

use crate::camera::{Camera, SoftwareVoxelCamera};

const MAX_VISIBLE_CLUSTERS: u32 = 2 * 1024 * 1024;
const MAX_MESH_PARAMS: usize = 4096;
const VISIBLE_CLUSTER_SIZE: u64 = 48;
const INDIRECT_BUFFER_SIZE: u64 = std::mem::size_of::<vk::DrawIndirectCommand>() as u64;

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
                    .enable_feature(|features: &mut vk::PhysicalDeviceFeatures| {
                        &mut features.shader_int64
                    })
                    .unwrap();
            })
            .before(bevy_pumicite::CreateDevice),
        );

        app.add_systems(Startup, setup.after(bevy_pumicite::CreateDevice));
        app.add_systems(
            PostUpdate,
            (ensure_mesh_resources, render)
                .chain()
                .in_set(DefaultRenderSet)
                .after(bevy::transform::TransformSystems::Propagate),
        );
    }
}

#[derive(Resource)]
struct SoftwareVoxelPipeline {
    mesh: Handle<ComputePipeline>,
    draw: Handle<GraphicsPipeline>,
}

#[derive(Resource)]
struct SoftwareMeshResources {
    clusters: Arc<Buffer>,
    indirect: Arc<Buffer>,
    params: Arc<Buffer>,
    clusters_handle: BindlessBufferHandle,
    indirect_handle: BindlessBufferHandle,
    params_handle: BindlessBufferHandle,
    depth: GPUMutex<FullImageView<Image>>,
    cluster_state: ResourceState,
    indirect_state: ResourceState,
    params_state: ResourceState,
    depth_state: ResourceState,
    extent: UVec2,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftwareVoxelMeshUniform {
    camera_axes: [[f32; 4]; 3],
    model_rows: [[f32; 4]; 3],
    camera_params: [f32; 4],
    mesh_params: [u32; 4],
    resource_handles: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<SoftwareVoxelMeshUniform>() == 144);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftwareVoxelPushConstants {
    params_handle: u32,
    params_index: u32,
    clusters_handle: u32,
    indirect_handle: u32,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SoftwareVoxelPipeline {
        mesh: asset_server.load("software_voxel/software_voxel_mesh.comp.pipeline.ron"),
        draw: asset_server.load("software_voxel/software_voxel_mesh.gfx.pipeline.ron"),
    });
}

fn ensure_mesh_resources(
    mut commands: Commands,
    current_resources: Option<Res<SoftwareMeshResources>>,
    allocator: Res<Allocator>,
    heap: Res<DescriptorHeap>,
    swapchain_images: Query<&SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(swapchain_image) = swapchain_images.single() else {
        return;
    };
    let Some(current) = swapchain_image.current_image() else {
        return;
    };
    let extent = UVec2::new(current.extent().x, current.extent().y);
    if current_resources
        .as_ref()
        .is_some_and(|resources| resources.extent == extent)
    {
        return;
    }

    let clusters = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            MAX_VISIBLE_CLUSTERS as u64 * VISIBLE_CLUSTER_SIZE,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap()
        .with_name(c"Software Voxel Visible Clusters"),
    );
    let indirect = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            INDIRECT_BUFFER_SIZE,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel Indirect Draw"),
    );
    let params = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            MAX_MESH_PARAMS as u64 * std::mem::size_of::<SoftwareVoxelMeshUniform>() as u64,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel Params"),
    );

    let resource_heap = heap.resource_heap();
    let clusters_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(clusters.as_ref())).unwrap();
    let indirect_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(indirect.as_ref())).unwrap();
    let params_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(params.as_ref())).unwrap();

    let depth = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"Software Voxel Depth");

    commands.insert_resource(SoftwareMeshResources {
        clusters,
        indirect,
        params,
        clusters_handle,
        indirect_handle,
        params_handle,
        depth: GPUMutex::new(
            depth
                .create_full_view()
                .unwrap()
                .with_name(c"Software Voxel Depth View"),
        ),
        cluster_state: ResourceState::default(),
        indirect_state: ResourceState::default(),
        params_state: ResourceState::default(),
        depth_state: ResourceState::default(),
        extent,
    });
}

fn render(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: SubmissionState,
    pipeline: Res<SoftwareVoxelPipeline>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    graphics_pipelines: Res<Assets<GraphicsPipeline>>,
    heap: Res<DescriptorHeap>,
    mut staging: ResMut<HostVisibleRingBuffer>,
    mut resources: ResMut<SoftwareMeshResources>,
    cameras: Query<(&Camera, &GlobalTransform), With<SoftwareVoxelCamera>>,
    models: Query<(&VoxModel, Option<&GlobalTransform>, Option<&Transform>)>,
    geometries: Res<Assets<VoxGeometry>>,
    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>,
    mut bounds_cache: Local<HashMap<AssetId<VoxGeometry>, Option<(Vec3, Vec3)>>>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(mesh_pipeline) = compute_pipelines.get(&pipeline.mesh).cloned() else {
        return;
    };
    let Some(draw_pipeline) = graphics_pipelines.get(&pipeline.draw).cloned() else {
        return;
    };

    let base_uniform = empty_uniform(camera, camera_transform, resources.extent);
    let mut mesh_uniforms = Vec::new();
    let mut total_blocks = 0u32;
    for (model, global_transform, local_transform) in models.iter() {
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

        let affine = global_transform
            .map(GlobalTransform::affine)
            .unwrap_or_else(|| {
                local_transform
                    .map(Transform::compute_affine)
                    .unwrap_or(Affine3A::IDENTITY)
            });
        if let Some((local_min, local_max)) =
            geometry_bounds(&mut bounds_cache, model.geometry.id(), geometry)
        {
            if !aabb_visible(
                camera,
                camera_transform,
                resources.extent,
                affine,
                geometry.unit_size,
                local_min,
                local_max,
            ) {
                continue;
            }
        }
        let mut uniform = base_uniform;
        uniform.model_rows = affine_rows(affine, geometry.unit_size);
        uniform.mesh_params = [block_count, MAX_VISIBLE_CLUSTERS, 0, total_blocks];
        let (Some(geometry_handle), Some(material_handle), Some(palette_handle)) = (
            geometry.bindless_handle(),
            material.bindless_handle(),
            palette.bindless_handle(),
        ) else {
            continue;
        };
        uniform.resource_handles = [geometry_handle, material_handle, palette_handle, 0];
        mesh_uniforms.push(uniform);
        total_blocks = total_blocks.saturating_add(block_count);
        if mesh_uniforms.len() + 4 >= MAX_MESH_PARAMS {
            break;
        }
    }

    state.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };

        let _clusters = encoder.retain(resources.clusters.clone());
        let indirect = encoder.retain(resources.indirect.clone());
        let params = encoder.retain(resources.params.clone());
        encoder.use_resource::<()>(&mut resources.indirect_state, Access::COPY_WRITE);
        encoder.use_resource::<()>(&mut resources.params_state, Access::COPY_WRITE);
        encoder.update_buffer(indirect.as_ref(), &[0; INDIRECT_BUFFER_SIZE as usize]);
        let mut dispatch_params = Vec::with_capacity(mesh_uniforms.len() + 4);
        let mut init_uniform = base_uniform;
        init_uniform.mesh_params = [0, MAX_VISIBLE_CLUSTERS, 0, 0];
        dispatch_params.push(init_uniform);
        let mut emit_uniform = base_uniform;
        emit_uniform.mesh_params = [mesh_uniforms.len() as u32, MAX_VISIBLE_CLUSTERS, 1, 2];
        dispatch_params.push(emit_uniform);
        dispatch_params.extend(mesh_uniforms.iter().copied());
        let mut finalize_uniform = base_uniform;
        finalize_uniform.mesh_params = [0, MAX_VISIBLE_CLUSTERS, 2, 0];
        dispatch_params.push(finalize_uniform);
        let mut draw_uniform = base_uniform;
        draw_uniform.mesh_params = [36, MAX_VISIBLE_CLUSTERS, 3, 0];
        dispatch_params.push(draw_uniform);
        upload_params_buffer(
            encoder,
            &mut staging,
            params.as_ref(),
            bytemuck::cast_slice(&dispatch_params),
        );
        let params_shader_read = Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        encoder.use_resource::<()>(&mut resources.cluster_state, Access::COMPUTE_WRITE);
        encoder.use_resource::<()>(&mut resources.indirect_state, Access::COMPUTE_WRITE);
        encoder.use_resource::<()>(&mut resources.params_state, params_shader_read);
        encoder.emit_barriers();

        let mesh_pipeline = encoder.retain(mesh_pipeline.into_inner());
        heap.bind(encoder, vk::PipelineBindPoint::COMPUTE);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &mesh_pipeline);

        dispatch_mesh_pipeline(encoder, &mesh_pipeline, &resources, 0, UVec3::ONE);
        compute_mesh_barrier(encoder);

        if total_blocks > 0 && !mesh_uniforms.is_empty() {
            dispatch_mesh_pipeline(
                encoder,
                &mesh_pipeline,
                &resources,
                1,
                UVec3::new(total_blocks.div_ceil(64), 1, 1),
            );
            compute_mesh_barrier(encoder);
        }

        dispatch_mesh_pipeline(
            encoder,
            &mesh_pipeline,
            &resources,
            (mesh_uniforms.len() + 2) as u32,
            UVec3::ONE,
        );

        let cluster_shader_read = Access {
            stage: vk::PipelineStageFlags2::VERTEX_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        let indirect_read = Access {
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
        };
        encoder.use_resource::<()>(&mut resources.cluster_state, cluster_shader_read);
        encoder.use_resource::<()>(&mut resources.indirect_state, indirect_read);

        let current_swapchain_image = encoder.lock(
            current_swapchain_image,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );
        let depth = encoder.lock(
            &resources.depth,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        );
        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            depth.image(),
            &mut resources.depth_state,
            Access::EARLY_FRAGMENT_TEST_WRITE,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let mut pass = encoder
            .begin_rendering()
            .color_attachment(0, |mut builder| {
                builder
                    .clear(Vec4::new(0.025, 0.03, 0.04, 1.0))
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(current_swapchain_image.srgb_view().unwrap());
            })
            .depth_attachment(|mut builder| {
                builder
                    .clear(1.0)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(depth);
            })
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .begin();

        let draw_pipeline = pass.retain(draw_pipeline.into_inner());
        heap.bind(&mut pass, vk::PipelineBindPoint::GRAPHICS);
        pass.bind_pipeline(draw_pipeline);
        pass.set_viewport(
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: current_swapchain_image.extent().x as f32,
                height: current_swapchain_image.extent().y as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        pass.set_scissor(
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: current_swapchain_image.extent().x,
                    height: current_swapchain_image.extent().y,
                },
            }],
        );

        pass.push_constants(
            draw_pipeline.layout(),
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&SoftwareVoxelPushConstants {
                params_handle: resources.params_handle.get(),
                params_index: (mesh_uniforms.len() + 3) as u32,
                clusters_handle: resources.clusters_handle.get(),
                indirect_handle: resources.indirect_handle.get(),
            }),
        );
        pass.draw_indirect(indirect.as_ref(), 1, INDIRECT_BUFFER_SIZE as u32);
        pass.end();
    });
}

fn dispatch_mesh_pipeline<'a>(
    encoder: &mut pumicite::command::CommandEncoder<'a>,
    pipeline: &'a Arc<pumicite::pipeline::Pipeline>,
    resources: &SoftwareMeshResources,
    params_index: u32,
    groups: UVec3,
) {
    encoder.push_constants(
        pipeline.layout(),
        vk::ShaderStageFlags::ALL,
        0,
        bytemuck::bytes_of(&SoftwareVoxelPushConstants {
            params_handle: resources.params_handle.get(),
            params_index,
            clusters_handle: resources.clusters_handle.get(),
            indirect_handle: resources.indirect_handle.get(),
        }),
    );
    encoder.dispatch(groups);
}

fn upload_params_buffer<'a>(
    encoder: &mut pumicite::command::CommandEncoder<'a>,
    staging: &mut HostVisibleRingBuffer,
    params: &'a Buffer,
    data: &[u8],
) {
    if data.len() <= 65536 {
        encoder.update_buffer(params, data);
        return;
    }

    let mut staging_buffer = staging.allocate_buffer(data.len() as u64, 16);
    staging_buffer
        .as_slice_mut()
        .expect("host-visible staging buffer must be mapped")[..data.len()]
        .copy_from_slice(data);
    let staging_buffer = encoder.retain(staging_buffer);
    encoder.copy_buffer_region(staging_buffer, 0, params, 0, data.len() as u64);
}

fn compute_mesh_barrier(encoder: &mut pumicite::command::CommandEncoder<'_>) {
    encoder.memory_barrier(
        Access::COMPUTE_WRITE,
        Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        },
    );
    encoder.emit_barriers();
}

fn geometry_bounds(
    cache: &mut HashMap<AssetId<VoxGeometry>, Option<(Vec3, Vec3)>>,
    id: AssetId<VoxGeometry>,
    geometry: &VoxGeometry,
) -> Option<(Vec3, Vec3)> {
    *cache.entry(id).or_insert_with(|| {
        let mut iter = geometry.tree.iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for voxel in iter {
            min = min.min(voxel);
            max = max.max(voxel);
        }
        Some((min.as_vec3(), max.as_vec3() + Vec3::ONE))
    })
}

fn aabb_visible(
    camera: &Camera,
    camera_transform: &GlobalTransform,
    extent: UVec2,
    affine: Affine3A,
    unit_size: f32,
    local_min: Vec3,
    local_max: Vec3,
) -> bool {
    let mut model_affine = affine;
    model_affine.matrix3.x_axis *= unit_size;
    model_affine.matrix3.y_axis *= unit_size;
    model_affine.matrix3.z_axis *= unit_size;

    let mut world_min = Vec3A::splat(f32::INFINITY);
    let mut world_max = Vec3A::splat(f32::NEG_INFINITY);
    for x in [local_min.x, local_max.x] {
        for y in [local_min.y, local_max.y] {
            for z in [local_min.z, local_max.z] {
                let corner = model_affine.transform_point3a(Vec3A::new(x, y, z));
                world_min = world_min.min(corner);
                world_max = world_max.max(corner);
            }
        }
    }

    let center = (world_min + world_max) * 0.5;
    let radius = (world_max - center).length();
    let camera_affine = camera_transform.affine();
    let rel = center - camera_affine.translation;
    let camera_x = rel.dot(camera_affine.matrix3.x_axis);
    let camera_y = rel.dot(camera_affine.matrix3.y_axis);
    let forward = rel.dot(-camera_affine.matrix3.z_axis);

    if forward + radius <= camera.depth.start || forward - radius >= camera.depth.end {
        return false;
    }

    let tan_half_fov = camera.tan_half_fov();
    let aspect = extent.x as f32 / extent.y.max(1) as f32;
    let forward = forward.max(camera.depth.start);
    camera_x.abs() <= forward * tan_half_fov * aspect + radius
        && camera_y.abs() <= forward * tan_half_fov + radius
}

fn empty_uniform(
    camera: &Camera,
    transform: &GlobalTransform,
    extent: UVec2,
) -> SoftwareVoxelMeshUniform {
    let affine = transform.affine();
    let x = affine.matrix3.x_axis;
    let y = affine.matrix3.y_axis;
    let z = affine.matrix3.z_axis;
    let w = affine.translation;

    SoftwareVoxelMeshUniform {
        camera_axes: [
            [x.x, x.y, x.z, w.x],
            [y.x, y.y, y.z, w.y],
            [z.x, z.y, z.z, w.z],
        ],
        model_rows: affine_rows(Affine3A::IDENTITY, 1.0),
        camera_params: [
            camera.tan_half_fov(),
            extent.x as f32 / extent.y.max(1) as f32,
            camera.depth.start,
            camera.depth.end,
        ],
        mesh_params: [0, MAX_VISIBLE_CLUSTERS, 0, 0],
        resource_handles: [0; 4],
    }
}

fn affine_rows(affine: Affine3A, unit_size: f32) -> [[f32; 4]; 3] {
    let mut affine = affine;
    affine.matrix3.x_axis *= unit_size;
    affine.matrix3.y_axis *= unit_size;
    affine.matrix3.z_axis *= unit_size;
    let x = affine.matrix3.x_axis;
    let y = affine.matrix3.y_axis;
    let z = affine.matrix3.z_axis;
    let w = affine.translation;
    [
        [x.x, y.x, z.x, w.x],
        [x.y, y.y, z.y, w.y],
        [x.z, y.z, z.z, w.z],
    ]
}
