pub mod camera;

use std::{collections::HashMap, sync::Arc};

use avian3d::collider_tree::{ColliderTreeType, ColliderTrees, ProxyId};
use bevy::ecs::system::SystemParam;
use bevy::math::Affine3A;
use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use dust_dense::{
    BindlessBufferHandle, BufferDescriptor, DenseVoxelGeometry, DenseVoxelMaterial, DenseVoxelModel,
};
use obvhs::cwbvh::{bvh2_to_cwbvh::bvh2_to_cwbvh, node::CwBvhNode as CpuCwBvhNode};
use pumicite::{
    Allocator, HasDevice,
    ash::{VkResult, vk},
    bindless::{ImageAccessMode, ResourceHeap, SamplerHandle},
    buffer::{Buffer, BufferLike},
    debug::DebugObject,
    image::{FullImageView, Image, ImageExt, ImageLike},
    sync::GPUMutex,
    tracking::{Access, ResourceState},
};

use crate::camera::{Camera, SoftwareVoxelCamera};

const MAX_VISIBLE_CLUSTERS: u32 = 2 * 1024 * 1024;
const MAX_MESH_PARAMS: usize = 4096;
const MAX_RT_NODES: usize = 65_536;
const MAX_RT_PRIMITIVES: usize = 65_536;
const VISIBLE_CLUSTER_SIZE: u64 = 48;
const INDIRECT_BUFFER_SIZE: u64 = std::mem::size_of::<vk::DrawIndirectCommand>() as u64;
const DEPTH_PYRAMID_TEXEL_SIZE: u64 = 4;
type DenseBoundsCacheKey = (AssetId<DenseVoxelGeometry>, [u32; 3], [u32; 3]);

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
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFloat16Int8FeaturesKHR| {
                        &mut features.shader_int8
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDevice8BitStorageFeatures| {
                        &mut features.storage_buffer8_bit_access
                    })
                    .unwrap();
            })
            .before(bevy_pumicite::CreateDevice),
        );

        app.init_asset::<DenseVoxelGeometry>()
            .init_asset::<DenseVoxelMaterial>()
            .register_type::<DenseVoxelModel>();
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
    depth_pyramid: Handle<ComputePipeline>,
    depth_draw: Handle<GraphicsPipeline>,
    draw: Handle<GraphicsPipeline>,
    post: Handle<GraphicsPipeline>,
}

#[derive(Resource)]
struct SoftwareMeshResources {
    clusters: Arc<Buffer>,
    indirect: Arc<Buffer>,
    params: Arc<Buffer>,
    rt_nodes: Arc<Buffer>,
    rt_primitive_to_model: Arc<Buffer>,
    rt_models: Arc<Buffer>,
    depth_pyramid: Arc<Buffer>,
    previous_depth_pyramid: Arc<Buffer>,
    clusters_handle: BindlessBufferHandle,
    indirect_handle: BindlessBufferHandle,
    params_handle: BindlessBufferHandle,
    rt_nodes_handle: BindlessBufferHandle,
    rt_primitive_to_model_handle: BindlessBufferHandle,
    rt_models_handle: BindlessBufferHandle,
    depth_pyramid_handle: BindlessBufferHandle,
    previous_depth_pyramid_handle: BindlessBufferHandle,
    color_handle: BindlessImageHandle,
    post_sampler: SamplerHandle,
    color: GPUMutex<FullImageView<Image>>,
    depth: GPUMutex<FullImageView<Image>>,
    cluster_state: ResourceState,
    indirect_state: ResourceState,
    params_state: ResourceState,
    rt_nodes_state: ResourceState,
    rt_primitive_to_model_state: ResourceState,
    rt_models_state: ResourceState,
    depth_pyramid_state: ResourceState,
    previous_depth_pyramid_state: ResourceState,
    color_state: ResourceState,
    depth_state: ResourceState,
    extent: UVec2,
    depth_mips: Vec<DepthPyramidMip>,
    previous_depth_valid: bool,
    previous_depth_camera_axes: Option<[[f32; 4]; 3]>,
}

#[derive(Clone, Copy)]
struct DepthPyramidMip {
    offset: u32,
    width: u32,
    height: u32,
}

struct BindlessImageHandle {
    heap: ResourceHeap,
    handle: u32,
}

#[derive(SystemParam)]
struct DenseRenderParams<'w, 's> {
    models: Query<
        'w,
        's,
        (
            Entity,
            &'static DenseVoxelModel,
            Option<&'static GlobalTransform>,
            Option<&'static Transform>,
        ),
    >,
    parents: Query<'w, 's, &'static ChildOf>,
    geometries: ResMut<'w, Assets<DenseVoxelGeometry>>,
    materials: ResMut<'w, Assets<DenseVoxelMaterial>>,
    bounds_cache: Local<'s, HashMap<DenseBoundsCacheKey, Option<(Vec3, Vec3)>>>,
}

impl BindlessImageHandle {
    fn sampled(
        heap: &ResourceHeap,
        image: &impl ImageLike,
        image_layout: vk::ImageLayout,
    ) -> VkResult<Self> {
        Ok(Self {
            heap: heap.clone(),
            handle: heap.add_image_with_layout(image, image_layout, ImageAccessMode::Sampled)?,
        })
    }

    fn get(&self) -> u32 {
        self.handle
    }

    fn replace_sampled(
        &mut self,
        image: &impl ImageLike,
        image_layout: vk::ImageLayout,
    ) -> VkResult<()> {
        self.heap.update_image_with_layout(
            self.handle,
            image,
            image_layout,
            ImageAccessMode::Sampled,
        )
    }
}

impl Drop for BindlessImageHandle {
    fn drop(&mut self) {
        self.heap.remove(self.handle);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftwareVoxelMeshUniform {
    camera_axes: [[f32; 4]; 3],
    model_rows: [[f32; 4]; 3],
    camera_params: [f32; 4],
    mesh_params: [u32; 4],
    resource_handles: [u32; 4],
    cull_min: [u32; 4],
    cull_max: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<SoftwareVoxelMeshUniform>() == 176);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtDenseModel {
    inverse_model_rows: [[f32; 4]; 3],
    size: [u32; 4],
    resource_handles: [u32; 4],
    cull_min: [u32; 4],
    cull_max: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftwareVoxelPushConstants {
    params_handle: u32,
    params_index: u32,
    clusters_handle: u32,
    indirect_handle: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthPyramidPushConstants {
    pyramid_handle: u32,
    src_offset: u32,
    dst_offset: u32,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftwareVoxelPostPushConstants {
    color_handle: u32,
    sampler_handle: u32,
    extent: [f32; 2],
    pixel_size: f32,
    outline_strength: f32,
    _pad: [u32; 2],
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(SoftwareVoxelPipeline {
        mesh: asset_server.load("software_voxel/software_voxel_mesh.comp.pipeline.ron"),
        depth_pyramid: asset_server
            .load("software_voxel/software_voxel_depth_pyramid.comp.pipeline.ron"),
        depth_draw: asset_server.load("software_voxel/software_voxel_mesh_depth.gfx.pipeline.ron"),
        draw: asset_server.load("software_voxel/software_voxel_mesh.gfx.pipeline.ron"),
        post: asset_server.load("software_voxel/software_voxel_post.gfx.pipeline.ron"),
    });
}

fn ensure_mesh_resources(
    mut commands: Commands,
    mut current_resources: Option<ResMut<SoftwareMeshResources>>,
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
    if let Some(resources) = current_resources.as_deref_mut() {
        if resources.extent != extent {
            resize_mesh_resources(resources, allocator.as_ref(), extent);
        }
        return;
    }

    let depth_mips = depth_pyramid_mips(extent);
    let depth_pyramid_size = depth_mips
        .last()
        .map(|mip| {
            (mip.offset as u64 + mip.width as u64 * mip.height as u64) * DEPTH_PYRAMID_TEXEL_SIZE
        })
        .unwrap_or(DEPTH_PYRAMID_TEXEL_SIZE);

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
    let rt_nodes = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            MAX_RT_NODES as u64 * std::mem::size_of::<CpuCwBvhNode>() as u64,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel RT CwBvh Nodes"),
    );
    let rt_primitive_to_model = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            MAX_RT_PRIMITIVES as u64 * std::mem::size_of::<u32>() as u64,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel RT Primitive Model Map"),
    );
    let rt_models = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            MAX_MESH_PARAMS as u64 * std::mem::size_of::<RtDenseModel>() as u64,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel RT Dense Models"),
    );
    let depth_pyramid = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            depth_pyramid_size,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel Depth Pyramid"),
    );
    let previous_depth_pyramid = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            depth_pyramid_size,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel Previous Depth Pyramid"),
    );

    let resource_heap = heap.resource_heap();
    let clusters_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(clusters.as_ref())).unwrap();
    let indirect_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(indirect.as_ref())).unwrap();
    let params_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(params.as_ref())).unwrap();
    let rt_nodes_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(rt_nodes.as_ref())).unwrap();
    let rt_primitive_to_model_handle = BindlessBufferHandle::new(
        resource_heap,
        BufferDescriptor::new(rt_primitive_to_model.as_ref()),
    )
    .unwrap();
    let rt_models_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(rt_models.as_ref()))
            .unwrap();
    let depth_pyramid_handle =
        BindlessBufferHandle::new(resource_heap, BufferDescriptor::new(depth_pyramid.as_ref()))
            .unwrap();
    let previous_depth_pyramid_handle = BindlessBufferHandle::new(
        resource_heap,
        BufferDescriptor::new(previous_depth_pyramid.as_ref()),
    )
    .unwrap();

    let color = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::B8G8R8A8_SRGB,
            extent: vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"Software Voxel Post Color");
    let color_handle = BindlessImageHandle::sampled(
        resource_heap,
        &color,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )
    .unwrap();
    let post_sampler = SamplerHandle::new(
        heap.sampler_heap().clone(),
        &vk::SamplerCreateInfo {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            min_lod: 0.0,
            max_lod: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

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
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC,
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
        rt_nodes,
        rt_primitive_to_model,
        rt_models,
        depth_pyramid,
        previous_depth_pyramid,
        clusters_handle,
        indirect_handle,
        params_handle,
        rt_nodes_handle,
        rt_primitive_to_model_handle,
        rt_models_handle,
        depth_pyramid_handle,
        previous_depth_pyramid_handle,
        color_handle,
        post_sampler,
        color: GPUMutex::new(
            color
                .create_full_view()
                .unwrap()
                .with_name(c"Software Voxel Post Color View"),
        ),
        depth: GPUMutex::new(
            depth
                .create_full_view()
                .unwrap()
                .with_name(c"Software Voxel Depth View"),
        ),
        cluster_state: ResourceState::default(),
        indirect_state: ResourceState::default(),
        params_state: ResourceState::default(),
        rt_nodes_state: ResourceState::default(),
        rt_primitive_to_model_state: ResourceState::default(),
        rt_models_state: ResourceState::default(),
        depth_pyramid_state: ResourceState::default(),
        previous_depth_pyramid_state: ResourceState::default(),
        color_state: ResourceState::default(),
        depth_state: ResourceState::default(),
        extent,
        depth_mips,
        previous_depth_valid: false,
        previous_depth_camera_axes: None,
    });
}

fn resize_mesh_resources(
    resources: &mut SoftwareMeshResources,
    allocator: &Allocator,
    extent: UVec2,
) {
    unsafe {
        allocator.device().device_wait_idle().unwrap();
    }

    let depth_mips = depth_pyramid_mips(extent);
    let depth_pyramid_size = depth_mips
        .last()
        .map(|mip| {
            (mip.offset as u64 + mip.width as u64 * mip.height as u64) * DEPTH_PYRAMID_TEXEL_SIZE
        })
        .unwrap_or(DEPTH_PYRAMID_TEXEL_SIZE);

    let depth_pyramid = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            depth_pyramid_size,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel Depth Pyramid"),
    );
    let previous_depth_pyramid = Arc::new(
        Buffer::new_private(
            allocator.clone(),
            depth_pyramid_size,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .unwrap()
        .with_name(c"Software Voxel Previous Depth Pyramid"),
    );

    let color = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::B8G8R8A8_SRGB,
            extent: vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"Software Voxel Post Color");

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
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap()
    .with_name(c"Software Voxel Depth");

    resources
        .depth_pyramid_handle
        .replace(BufferDescriptor::new(depth_pyramid.as_ref()))
        .unwrap();
    resources
        .previous_depth_pyramid_handle
        .replace(BufferDescriptor::new(previous_depth_pyramid.as_ref()))
        .unwrap();
    resources
        .color_handle
        .replace_sampled(&color, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .unwrap();

    resources.depth_pyramid = depth_pyramid;
    resources.previous_depth_pyramid = previous_depth_pyramid;
    resources.color = GPUMutex::new(
        color
            .create_full_view()
            .unwrap()
            .with_name(c"Software Voxel Post Color View"),
    );
    resources.depth = GPUMutex::new(
        depth
            .create_full_view()
            .unwrap()
            .with_name(c"Software Voxel Depth View"),
    );
    resources.depth_pyramid_state = ResourceState::default();
    resources.previous_depth_pyramid_state = ResourceState::default();
    resources.color_state = ResourceState::default();
    resources.depth_state = ResourceState::default();
    resources.extent = extent;
    resources.depth_mips = depth_mips;
    resources.previous_depth_valid = false;
    resources.previous_depth_camera_axes = None;
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
    cameras: Query<(&Camera, &Projection, &GlobalTransform), With<SoftwareVoxelCamera>>,
    collider_trees: Option<Res<ColliderTrees>>,
    mut dense: DenseRenderParams,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let Some(current_extent) = swapchain_image.current_image().map(|image| {
        let extent = image.extent();
        UVec2::new(extent.x, extent.y)
    }) else {
        return;
    };
    if current_extent != resources.extent {
        resources.previous_depth_valid = false;
        resources.previous_depth_camera_axes = None;
        return;
    }
    let Ok((camera, projection, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(mesh_pipeline) = compute_pipelines.get(&pipeline.mesh).cloned() else {
        return;
    };
    let Some(depth_pyramid_pipeline) = compute_pipelines.get(&pipeline.depth_pyramid).cloned()
    else {
        return;
    };
    let Some(depth_draw_pipeline) = graphics_pipelines.get(&pipeline.depth_draw).cloned() else {
        return;
    };
    let Some(draw_pipeline) = graphics_pipelines.get(&pipeline.draw).cloned() else {
        return;
    };
    let pixel_art = camera.pixel_art;
    let post_pipeline = if pixel_art.enabled {
        let Some(post_pipeline) = graphics_pipelines.get(&pipeline.post).cloned() else {
            return;
        };
        Some(post_pipeline)
    } else {
        None
    };

    let base_uniform = empty_uniform(projection, camera_transform, resources.extent);
    let previous_depth_valid = resources.previous_depth_valid
        && resources
            .previous_depth_camera_axes
            .is_some_and(|axes| previous_depth_camera_matches(axes, base_uniform.camera_axes));
    let mut mesh_uniforms = Vec::new();
    let mut rt_models = Vec::new();
    let mut entity_to_rt_model = HashMap::new();
    let mut total_blocks = 0u32;
    let resource_heap = heap.resource_heap();
    for (entity, model, global_transform, local_transform) in dense.models.iter() {
        let Some(geometry) = dense.geometries.get_mut(&model.occupancy) else {
            continue;
        };
        let Some(material) = dense.materials.get_mut(&model.material) else {
            continue;
        };
        if geometry.size() != material.size() {
            continue;
        }
        if geometry.register_bindless(resource_heap).is_err()
            || material.register_bindless(resource_heap).is_err()
        {
            continue;
        }

        let local_bounds = dense_geometry_bounds(
            &mut dense.bounds_cache,
            model.occupancy.id(),
            geometry,
            model.cull_min,
            model.cull_max,
        );
        let block_count = dense_block_count(geometry.size());
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
        let Some((local_min, local_max)) = local_bounds else {
            continue;
        };

        let Some(dense_gpu) = DenseVoxelModel::gpu_descriptor(geometry, material) else {
            continue;
        };
        if rt_models.len() < MAX_MESH_PARAMS {
            let rt_model_index = rt_models.len() as u32;
            entity_to_rt_model.insert(entity, rt_model_index);
            rt_models.push(RtDenseModel {
                inverse_model_rows: affine_rows(affine.inverse(), 1.0),
                size: [
                    geometry.size()[0],
                    geometry.size()[1],
                    geometry.size()[2],
                    0,
                ],
                resource_handles: [
                    dense_gpu.occupancy_handle,
                    dense_gpu.material_refs_handle,
                    dense_gpu.material_pages_handle,
                    0,
                ],
                cull_min: [model.cull_min.x, model.cull_min.y, model.cull_min.z, 0],
                cull_max: [model.cull_max.x, model.cull_max.y, model.cull_max.z, 0],
            });
        }

        if !aabb_visible(
            projection,
            camera_transform,
            resources.extent,
            affine,
            1.0,
            local_min,
            local_max,
        ) {
            continue;
        }

        if mesh_uniforms.len() + 8 >= MAX_MESH_PARAMS {
            continue;
        }
        let mut uniform = base_uniform;
        uniform.model_rows = affine_rows(affine, 1.0);
        uniform.mesh_params = [block_count, MAX_VISIBLE_CLUSTERS, 0, total_blocks];
        uniform.resource_handles = [
            dense_gpu.occupancy_handle,
            dense_gpu.material_refs_handle,
            dense_gpu.material_pages_handle,
            pack_dense_size(geometry.size()),
        ];
        uniform.cull_min = [
            model.cull_min.x,
            model.cull_min.y,
            model.cull_min.z,
            dense_gpu.material_params_handle,
        ];
        uniform.cull_max = [model.cull_max.x, model.cull_max.y, model.cull_max.z, 0];
        mesh_uniforms.push(uniform);
        total_blocks = total_blocks.saturating_add(block_count);
    }
    let cpu_start = std::time::Instant::now();
    let (rt_nodes, rt_primitive_to_model) = build_rt_scene(
        collider_trees.as_deref(),
        &entity_to_rt_model,
        &dense.parents,
    );
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
    println!("build bvh : {cpu_ms:.3} ms");
    let rt_model_count = rt_models.len() as u32;
    let rt_node_count = rt_nodes.len() as u32;
    let rt_primitive_count = rt_primitive_to_model.len() as u32;
    state.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let swapchain_extent = current_swapchain_image.extent();

        let _clusters = encoder.retain(resources.clusters.clone());
        let indirect = encoder.retain(resources.indirect.clone());
        let params = encoder.retain(resources.params.clone());
        let rt_nodes_buffer = encoder.retain(resources.rt_nodes.clone());
        let rt_primitive_to_model_buffer = encoder.retain(resources.rt_primitive_to_model.clone());
        let rt_models_buffer = encoder.retain(resources.rt_models.clone());
        let depth_pyramid = encoder.retain(resources.depth_pyramid.clone());
        let previous_depth_pyramid = encoder.retain(resources.previous_depth_pyramid.clone());
        encoder.use_resource::<()>(&mut resources.indirect_state, Access::COPY_WRITE);
        encoder.use_resource::<()>(&mut resources.params_state, Access::COPY_WRITE);
        encoder.use_resource::<()>(&mut resources.rt_nodes_state, Access::COPY_WRITE);
        encoder.use_resource::<()>(
            &mut resources.rt_primitive_to_model_state,
            Access::COPY_WRITE,
        );
        encoder.use_resource::<()>(&mut resources.rt_models_state, Access::COPY_WRITE);
        encoder.update_buffer(indirect.as_ref(), &[0; INDIRECT_BUFFER_SIZE as usize]);
        upload_or_clear_buffer(
            encoder,
            &mut staging,
            rt_nodes_buffer.as_ref(),
            bytemuck::cast_slice(&rt_nodes),
            std::mem::size_of::<CpuCwBvhNode>(),
        );
        upload_or_clear_buffer(
            encoder,
            &mut staging,
            rt_primitive_to_model_buffer.as_ref(),
            bytemuck::cast_slice(&rt_primitive_to_model),
            std::mem::size_of::<u32>(),
        );
        upload_or_clear_buffer(
            encoder,
            &mut staging,
            rt_models_buffer.as_ref(),
            bytemuck::cast_slice(&rt_models),
            std::mem::size_of::<RtDenseModel>(),
        );
        let mut dispatch_params = Vec::with_capacity(mesh_uniforms.len() + 8);
        let init_prepass_index = dispatch_params.len() as u32;
        let mut init_prepass_uniform = base_uniform;
        init_prepass_uniform.mesh_params = [0, MAX_VISIBLE_CLUSTERS, 0, 0];
        dispatch_params.push(init_prepass_uniform);
        let emit_prepass_index = dispatch_params.len() as u32;
        let mut emit_prepass_uniform = base_uniform;
        emit_prepass_uniform.mesh_params = [
            mesh_uniforms.len() as u32,
            MAX_VISIBLE_CLUSTERS,
            if previous_depth_valid { 5 } else { 1 },
            2,
        ];
        if previous_depth_valid {
            emit_prepass_uniform.resource_handles = [
                resources.previous_depth_pyramid_handle.get(),
                resources.extent.x,
                resources.extent.y,
                resources.depth_mips.len() as u32,
            ];
        }
        dispatch_params.push(emit_prepass_uniform);
        let first_instance_index = dispatch_params.len() as u32;
        dispatch_params.extend(mesh_uniforms.iter().copied());
        let finalize_prepass_index = dispatch_params.len() as u32;
        let mut finalize_prepass_uniform = base_uniform;
        finalize_prepass_uniform.mesh_params = [0, MAX_VISIBLE_CLUSTERS, 2, 0];
        dispatch_params.push(finalize_prepass_uniform);
        let draw_prepass_index = dispatch_params.len() as u32;
        let mut draw_prepass_uniform = base_uniform;
        draw_prepass_uniform.mesh_params = [36, MAX_VISIBLE_CLUSTERS, 3, 0];
        dispatch_params.push(draw_prepass_uniform);
        let init_final_index = dispatch_params.len() as u32;
        let mut init_final_uniform = base_uniform;
        init_final_uniform.mesh_params = [0, MAX_VISIBLE_CLUSTERS, 0, 0];
        dispatch_params.push(init_final_uniform);
        let emit_final_index = dispatch_params.len() as u32;
        let mut emit_final_uniform = base_uniform;
        emit_final_uniform.mesh_params = [
            mesh_uniforms.len() as u32,
            MAX_VISIBLE_CLUSTERS,
            4,
            first_instance_index,
        ];
        emit_final_uniform.resource_handles = [
            resources.depth_pyramid_handle.get(),
            resources.extent.x,
            resources.extent.y,
            resources.depth_mips.len() as u32,
        ];
        dispatch_params.push(emit_final_uniform);
        let finalize_final_index = dispatch_params.len() as u32;
        let mut finalize_final_uniform = base_uniform;
        finalize_final_uniform.mesh_params = [0, MAX_VISIBLE_CLUSTERS, 2, 0];
        dispatch_params.push(finalize_final_uniform);
        let draw_final_index = dispatch_params.len() as u32;
        let mut draw_final_uniform = base_uniform;
        draw_final_uniform.mesh_params = [36, MAX_VISIBLE_CLUSTERS, 3, 0];
        draw_final_uniform.resource_handles = [
            resources.rt_nodes_handle.get(),
            resources.rt_primitive_to_model_handle.get(),
            resources.rt_models_handle.get(),
            rt_node_count,
        ];
        draw_final_uniform.cull_min = [rt_primitive_count, rt_model_count, 0, 0];
        dispatch_params.push(draw_final_uniform);
        upload_params_buffer(
            encoder,
            &mut staging,
            params.as_ref(),
            bytemuck::cast_slice(&dispatch_params),
        );
        let params_shader_read = Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        encoder.use_resource::<()>(&mut resources.cluster_state, Access::COMPUTE_WRITE);
        encoder.use_resource::<()>(&mut resources.indirect_state, Access::COMPUTE_WRITE);
        encoder.use_resource::<()>(&mut resources.params_state, params_shader_read);
        if previous_depth_valid {
            encoder.use_resource::<()>(
                &mut resources.previous_depth_pyramid_state,
                Access {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_STORAGE_READ,
                },
            );
        }
        encoder.emit_barriers();

        let mesh_pipeline = encoder.retain(mesh_pipeline.into_inner());
        heap.bind(encoder, vk::PipelineBindPoint::COMPUTE);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &mesh_pipeline);

        dispatch_mesh_pipeline(
            encoder,
            &mesh_pipeline,
            &resources,
            init_prepass_index,
            UVec3::ONE,
        );
        compute_mesh_barrier(encoder);

        if total_blocks > 0 && !mesh_uniforms.is_empty() {
            dispatch_mesh_pipeline(
                encoder,
                &mesh_pipeline,
                &resources,
                emit_prepass_index,
                UVec3::new(total_blocks.div_ceil(64), 1, 1),
            );
            compute_mesh_barrier(encoder);
        }

        dispatch_mesh_pipeline(
            encoder,
            &mesh_pipeline,
            &resources,
            finalize_prepass_index,
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

        let depth = encoder.lock(
            &resources.depth,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::TRANSFER,
        );
        encoder.use_image_resource(
            depth.image(),
            &mut resources.depth_state,
            Access {
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            },
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();

        let mut pass = encoder
            .begin_rendering()
            .depth_attachment(|mut builder| {
                builder
                    .clear(1.0)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(depth);
            })
            .render_area(IVec2::ZERO, swapchain_extent.xy())
            .begin();

        let prepass_draw_pipeline = pass.retain(depth_draw_pipeline.into_inner());
        heap.bind(&mut pass, vk::PipelineBindPoint::GRAPHICS);
        pass.bind_pipeline(prepass_draw_pipeline);
        pass.set_viewport(
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain_extent.x as f32,
                height: swapchain_extent.y as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        pass.set_scissor(
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: swapchain_extent.x,
                    height: swapchain_extent.y,
                },
            }],
        );

        pass.push_constants(
            prepass_draw_pipeline.layout(),
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&SoftwareVoxelPushConstants {
                params_handle: resources.params_handle.get(),
                params_index: draw_prepass_index,
                clusters_handle: resources.clusters_handle.get(),
                indirect_handle: resources.indirect_handle.get(),
            }),
        );
        pass.draw_indirect(indirect.as_ref(), 1, INDIRECT_BUFFER_SIZE as u32);
        pass.end();

        let depth_mip0 = resources.depth_mips[0];
        encoder.use_image_resource(
            depth.image(),
            &mut resources.depth_state,
            Access::COPY_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_resource::<()>(&mut resources.depth_pyramid_state, Access::COPY_WRITE);
        encoder.emit_barriers();
        encoder.copy_image_to_buffer_with_layout(
            depth.image(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            depth_pyramid.as_ref(),
            &[vk::BufferImageCopy {
                buffer_offset: depth_mip0.offset as u64 * DEPTH_PYRAMID_TEXEL_SIZE,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: depth_mip0.width,
                    height: depth_mip0.height,
                    depth: 1,
                },
            }],
        );

        let depth_pyramid_pipeline = encoder.retain(depth_pyramid_pipeline.into_inner());
        encoder.use_resource::<()>(
            &mut resources.depth_pyramid_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            },
        );
        encoder.emit_barriers();
        heap.bind(encoder, vk::PipelineBindPoint::COMPUTE);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &depth_pyramid_pipeline);
        for mip_index in 1..resources.depth_mips.len() {
            let src = resources.depth_mips[mip_index - 1];
            let dst = resources.depth_mips[mip_index];
            encoder.push_constants(
                depth_pyramid_pipeline.layout(),
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&DepthPyramidPushConstants {
                    pyramid_handle: resources.depth_pyramid_handle.get(),
                    src_offset: src.offset,
                    dst_offset: dst.offset,
                    src_width: src.width,
                    src_height: src.height,
                    dst_width: dst.width,
                    dst_height: dst.height,
                    _pad: 0,
                }),
            );
            encoder.dispatch(UVec3::new(dst.width.div_ceil(8), dst.height.div_ceil(8), 1));
            compute_mesh_barrier(encoder);
        }

        encoder.use_resource::<()>(&mut resources.cluster_state, Access::COMPUTE_WRITE);
        encoder.use_resource::<()>(&mut resources.indirect_state, Access::COMPUTE_WRITE);
        encoder.use_resource::<()>(
            &mut resources.depth_pyramid_state,
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_STORAGE_READ,
            },
        );
        encoder.emit_barriers();
        heap.bind(encoder, vk::PipelineBindPoint::COMPUTE);
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &mesh_pipeline);
        dispatch_mesh_pipeline(
            encoder,
            &mesh_pipeline,
            &resources,
            init_final_index,
            UVec3::ONE,
        );
        compute_mesh_barrier(encoder);
        if total_blocks > 0 && !mesh_uniforms.is_empty() {
            dispatch_mesh_pipeline(
                encoder,
                &mesh_pipeline,
                &resources,
                emit_final_index,
                UVec3::new(total_blocks.div_ceil(64), 1, 1),
            );
            compute_mesh_barrier(encoder);
        }
        dispatch_mesh_pipeline(
            encoder,
            &mesh_pipeline,
            &resources,
            finalize_final_index,
            UVec3::ONE,
        );

        let rt_shader_read = Access {
            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access: vk::AccessFlags2::SHADER_STORAGE_READ,
        };
        encoder.use_resource::<()>(&mut resources.cluster_state, cluster_shader_read);
        encoder.use_resource::<()>(&mut resources.indirect_state, indirect_read);
        encoder.use_resource::<()>(&mut resources.rt_nodes_state, rt_shader_read);
        encoder.use_resource::<()>(&mut resources.rt_primitive_to_model_state, rt_shader_read);
        encoder.use_resource::<()>(&mut resources.rt_models_state, rt_shader_read);
        if pixel_art.enabled {
            let color = encoder.lock(
                &resources.color,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            );
            encoder.use_image_resource(
                color.image(),
                &mut resources.color_state,
                Access::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                0..1,
                0..1,
                true,
            );
            encoder.use_image_resource(
                depth.image(),
                &mut resources.depth_state,
                Access {
                    stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                },
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                0..1,
                0..1,
                true,
            );
            encoder.emit_barriers();

            let mut pass = encoder
                .begin_rendering()
                .color_attachment(0, |mut builder| {
                    builder
                        .clear(Vec4::new(0.025, 0.03, 0.04, 1.0))
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .store(true)
                        .view(color);
                })
                .depth_attachment(|mut builder| {
                    builder
                        .clear(1.0)
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .store(true)
                        .view(depth);
                })
                .render_area(IVec2::ZERO, swapchain_extent.xy())
                .begin();

            let draw_pipeline = pass.retain(draw_pipeline.into_inner());
            heap.bind(&mut pass, vk::PipelineBindPoint::GRAPHICS);
            pass.bind_pipeline(draw_pipeline);
            pass.set_viewport(
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: swapchain_extent.x as f32,
                    height: swapchain_extent.y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            pass.set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: swapchain_extent.x,
                        height: swapchain_extent.y,
                    },
                }],
            );

            pass.push_constants(
                draw_pipeline.layout(),
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&SoftwareVoxelPushConstants {
                    params_handle: resources.params_handle.get(),
                    params_index: draw_final_index,
                    clusters_handle: resources.clusters_handle.get(),
                    indirect_handle: resources.indirect_handle.get(),
                }),
            );
            pass.draw_indirect(indirect.as_ref(), 1, INDIRECT_BUFFER_SIZE as u32);
            pass.end();

            encoder.use_image_resource(
                color.image(),
                &mut resources.color_state,
                Access::FRAGMENT_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
            let current_swapchain_image = encoder.lock(
                current_swapchain_image,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
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
                .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
                .begin();

            let post_pipeline =
                post_pipeline.expect("post pipeline is loaded when pixel art is enabled");
            let post_pipeline = pass.retain(post_pipeline.into_inner());
            heap.bind(&mut pass, vk::PipelineBindPoint::GRAPHICS);
            pass.bind_pipeline(post_pipeline);
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
                post_pipeline.layout(),
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&SoftwareVoxelPostPushConstants {
                    color_handle: resources.color_handle.get(),
                    sampler_handle: resources.post_sampler.id(),
                    extent: [
                        current_swapchain_image.extent().x as f32,
                        current_swapchain_image.extent().y as f32,
                    ],
                    pixel_size: pixel_art.pixel_size,
                    outline_strength: pixel_art.outline_strength,
                    _pad: [0; 2],
                }),
            );
            pass.draw(0..3, 0..1);
            pass.end();
        } else {
            let current_swapchain_image = encoder.lock(
                current_swapchain_image,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
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
                Access {
                    stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                },
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                0..1,
                0..1,
                true,
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
                    params_index: draw_final_index,
                    clusters_handle: resources.clusters_handle.get(),
                    indirect_handle: resources.indirect_handle.get(),
                }),
            );
            pass.draw_indirect(indirect.as_ref(), 1, INDIRECT_BUFFER_SIZE as u32);
            pass.end();
        }

        encoder.use_resource::<()>(&mut resources.depth_pyramid_state, Access::COPY_READ);
        encoder.use_resource::<()>(
            &mut resources.previous_depth_pyramid_state,
            Access::COPY_WRITE,
        );
        encoder.emit_barriers();
        encoder.copy_buffer(depth_pyramid.as_ref(), previous_depth_pyramid.as_ref());
    });

    resources.previous_depth_valid = true;
    resources.previous_depth_camera_axes = Some(base_uniform.camera_axes);
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

fn build_rt_scene(
    collider_trees: Option<&ColliderTrees>,
    entity_to_rt_model: &HashMap<Entity, u32>,
    parents: &Query<&ChildOf>,
) -> (Vec<CpuCwBvhNode>, Vec<u32>) {
    let Some(collider_trees) = collider_trees else {
        return (Vec::new(), Vec::new());
    };
    let tree = collider_trees.tree_for_type(ColliderTreeType::Static);
    if tree.bvh.primitive_indices.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let cwbvh = bvh2_to_cwbvh(&tree.bvh, 3, true, false);
    if cwbvh.nodes.len() > MAX_RT_NODES || cwbvh.primitive_indices.len() > MAX_RT_PRIMITIVES {
        return (Vec::new(), Vec::new());
    }

    let mut primitive_to_model = Vec::with_capacity(cwbvh.primitive_indices.len());
    for proxy_id in cwbvh.primitive_indices.iter().copied() {
        let model_index = tree
            .proxies
            .get(ProxyId::new(proxy_id).index())
            .and_then(|proxy| resolve_rt_model(proxy.collider, entity_to_rt_model, parents))
            .unwrap_or(u32::MAX);
        primitive_to_model.push(model_index);
    }

    (cwbvh.nodes, primitive_to_model)
}

fn resolve_rt_model(
    entity: Entity,
    entity_to_rt_model: &HashMap<Entity, u32>,
    parents: &Query<&ChildOf>,
) -> Option<u32> {
    let mut current = entity;
    for _ in 0..16 {
        if let Some(model) = entity_to_rt_model.get(&current).copied() {
            return Some(model);
        }
        current = parents.get(current).ok()?.parent();
    }
    None
}

fn upload_or_clear_buffer<'a>(
    encoder: &mut pumicite::command::CommandEncoder<'a>,
    staging: &mut HostVisibleRingBuffer,
    buffer: &'a Buffer,
    data: &[u8],
    clear_bytes: usize,
) {
    if data.is_empty() {
        let clear = vec![0; clear_bytes];
        encoder.update_buffer(buffer, &clear);
    } else {
        upload_params_buffer(encoder, staging, buffer, data);
    }
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

fn depth_pyramid_mips(extent: UVec2) -> Vec<DepthPyramidMip> {
    let mut mips = Vec::new();
    let mut offset = 0u32;
    let mut width = extent.x.max(1);
    let mut height = extent.y.max(1);
    loop {
        mips.push(DepthPyramidMip {
            offset,
            width,
            height,
        });
        offset = offset.saturating_add(width.saturating_mul(height));
        if width == 1 && height == 1 {
            break;
        }
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    mips
}

fn previous_depth_camera_matches(previous: [[f32; 4]; 3], current: [[f32; 4]; 3]) -> bool {
    let previous_position = Vec3::new(previous[0][3], previous[1][3], previous[2][3]);
    let current_position = Vec3::new(current[0][3], current[1][3], current[2][3]);
    if previous_position.distance_squared(current_position) > 1.0 {
        return false;
    }

    for axis in 0..3 {
        let previous_axis = Vec3::new(previous[axis][0], previous[axis][1], previous[axis][2]);
        let current_axis = Vec3::new(current[axis][0], current[axis][1], current[axis][2]);
        if previous_axis.dot(current_axis) < 0.9995 {
            return false;
        }
    }
    true
}

fn dense_geometry_bounds(
    cache: &mut HashMap<DenseBoundsCacheKey, Option<(Vec3, Vec3)>>,
    id: AssetId<DenseVoxelGeometry>,
    geometry: &DenseVoxelGeometry,
    cull_min: UVec3,
    cull_max: UVec3,
) -> Option<(Vec3, Vec3)> {
    let key = (
        id,
        [cull_min.x, cull_min.y, cull_min.z],
        [cull_max.x, cull_max.y, cull_max.z],
    );
    *cache.entry(key).or_insert_with(|| {
        let size = geometry.size();
        let size = UVec3::new(size[0], size[1], size[2]);
        let cull_min = cull_min.min(size);
        let cull_max = cull_max.min(size);
        if cull_min.cmpge(cull_max).any() {
            return None;
        }

        let mut min = UVec3::MAX;
        let mut max = UVec3::ZERO;
        for (index, word) in geometry.occupancy().iter().copied().enumerate() {
            if word == 0 {
                continue;
            }

            let block = dense_storage_block_coords(index as u32, dense_block_extent(size));
            let block_min = block * UVec3::splat(4);
            let block_max = (block_min + UVec3::splat(4)).min(size);
            if block_max.x <= cull_min.x
                || block_max.y <= cull_min.y
                || block_max.z <= cull_min.z
                || block_min.x >= cull_max.x
                || block_min.y >= cull_max.y
                || block_min.z >= cull_max.z
            {
                continue;
            }

            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros();
                let local = UVec3::new(bit & 3, (bit >> 2) & 3, (bit >> 4) & 3);
                let voxel = block_min + local;
                remaining &= remaining - 1;
                if !voxel.cmplt(size).all()
                    || voxel.cmplt(cull_min).any()
                    || voxel.cmpge(cull_max).any()
                {
                    continue;
                }
                min = min.min(voxel);
                max = max.max(voxel + UVec3::ONE);
            }
        }

        if min == UVec3::MAX {
            None
        } else {
            Some((min.as_vec3(), max.as_vec3()))
        }
    })
}

fn dense_block_count(size: [u32; 3]) -> u32 {
    let extent = dense_block_extent(UVec3::new(size[0], size[1], size[2]));
    extent[0]
        .saturating_mul(extent[1])
        .saturating_mul(extent[2])
}

fn dense_block_extent(size: UVec3) -> UVec3 {
    (size + UVec3::splat(3)) / UVec3::splat(4)
}

fn dense_storage_block_coords(index: u32, block_extent: UVec3) -> UVec3 {
    let x = index % block_extent.x;
    let yz = index / block_extent.x;
    let y = yz % block_extent.y;
    let z = yz / block_extent.y;
    UVec3::new(x, y, z)
}

fn pack_dense_size(size: [u32; 3]) -> u32 {
    debug_assert!(size.iter().all(|axis| *axis > 0 && *axis <= 1024));
    (size[0] - 1) | ((size[1] - 1) << 10) | ((size[2] - 1) << 20)
}

fn aabb_visible(
    projection: &Projection,
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
    let projection_params = software_projection_params(projection, extent);

    if forward + radius <= projection_params.near || forward - radius >= projection_params.far {
        return false;
    }

    let tan_half_fov = projection_params.tan_half_fov;
    let aspect = projection_params.aspect;
    let forward = forward.max(projection_params.near);
    camera_x.abs() <= forward * tan_half_fov * aspect + radius
        && camera_y.abs() <= forward * tan_half_fov + radius
}

fn empty_uniform(
    projection: &Projection,
    transform: &GlobalTransform,
    extent: UVec2,
) -> SoftwareVoxelMeshUniform {
    let affine = transform.affine();
    let x = affine.matrix3.x_axis;
    let y = affine.matrix3.y_axis;
    let z = affine.matrix3.z_axis;
    let w = affine.translation;
    let projection_params = software_projection_params(projection, extent);

    SoftwareVoxelMeshUniform {
        camera_axes: [
            [x.x, x.y, x.z, w.x],
            [y.x, y.y, y.z, w.y],
            [z.x, z.y, z.z, w.z],
        ],
        model_rows: affine_rows(Affine3A::IDENTITY, 1.0),
        camera_params: [
            projection_params.tan_half_fov,
            projection_params.aspect,
            projection_params.near,
            projection_params.far,
        ],
        mesh_params: [0, MAX_VISIBLE_CLUSTERS, 0, 0],
        resource_handles: [0; 4],
        cull_min: [0; 4],
        cull_max: [u32::MAX; 4],
    }
}

struct SoftwareProjectionParams {
    tan_half_fov: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

fn software_projection_params(projection: &Projection, extent: UVec2) -> SoftwareProjectionParams {
    match projection {
        Projection::Perspective(projection) => SoftwareProjectionParams {
            tan_half_fov: (projection.fov * 0.5).tan(),
            aspect: projection.aspect_ratio,
            near: projection.near,
            far: projection.far,
        },
        _ => SoftwareProjectionParams {
            tan_half_fov: Camera::default().tan_half_fov(),
            aspect: extent.x as f32 / extent.y.max(1) as f32,
            near: Camera::default().depth.start,
            far: Camera::default().depth.end,
        },
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
