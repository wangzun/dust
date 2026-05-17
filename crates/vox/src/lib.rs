#![feature(generic_const_exprs)]

use bevy::math::U8Vec4;
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::{
    asset::{Asset, Handle},
    ecs::{bundle::Bundle, component::Component},
    reflect::TypePath,
    transform::components::{GlobalTransform, Transform},
};
use bevy_pumicite::{CreateDevice, DefaultTransferSet, DescriptorHeap, SubmissionState};
use dust_vdb::hierarchy;
use pumicite::Allocator;
use pumicite::ash::{VkResult, vk};
use pumicite::bindless::{BufferAccessMode, ResourceHeap};
use pumicite::buffer::{BufferLike, ManagedBuffer};
use pumicite::device::DeviceBuilder;
use pumicite::utils::AsVkHandle;
use std::ops::{Deref, DerefMut};

use avian3d::prelude::*;

mod geometry;
mod loader;
mod material;
mod runtime;

pub use material::{VoxLeafNode, VoxMaterial};
pub use runtime::{
    RuntimeVoxel, RuntimeVoxelModel, RuntimeVoxelModelId, RuntimeVoxelModelRef, RuntimeVoxelWorld,
};

/// Leaf node size: 96 bytes
type TreeRoot = hierarchy!(3, 3, 2, VoxLeafNode);

type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

pub use geometry::VoxGeometry;

#[derive(Clone, Copy)]
pub struct BufferDescriptor {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    device_address: vk::DeviceAddress,
}

impl BufferDescriptor {
    pub fn new(buffer: &impl BufferLike) -> Self {
        Self {
            buffer: buffer.vk_handle(),
            offset: buffer.offset(),
            size: buffer.size(),
            device_address: buffer.device_address(),
        }
    }
}

impl AsVkHandle for BufferDescriptor {
    type Handle = vk::Buffer;

    fn vk_handle(&self) -> Self::Handle {
        self.buffer
    }
}

impl BufferLike for BufferDescriptor {
    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    fn as_slice(&self) -> Option<&[u8]> {
        None
    }

    fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        None
    }

    fn flush(&mut self, _range: impl std::ops::RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        Ok(())
    }

    fn invalidate(&mut self, _range: impl std::ops::RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        Ok(())
    }
}

pub struct BindlessBufferHandle {
    heap: ResourceHeap,
    handle: u32,
}

impl BindlessBufferHandle {
    pub fn new(heap: &ResourceHeap, buffer: impl BufferLike) -> VkResult<Self> {
        Ok(Self {
            heap: heap.clone(),
            handle: heap.add_buffer(buffer, BufferAccessMode::Storage)?,
        })
    }

    pub fn get(&self) -> u32 {
        self.handle
    }
}

impl Drop for BindlessBufferHandle {
    fn drop(&mut self) {
        self.heap.remove(self.handle);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxMaterialParam {
    pub base_color: [f32; 4],
    /// metallic, roughness, specular strength, emissive strength.
    pub pbr: [f32; 4],
}

impl Default for VoxMaterialParam {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            pbr: [0.0, 0.6, 0.5, 0.0],
        }
    }
}

#[derive(Asset, TypePath)]
pub struct VoxPalette(ManagedBuffer, Option<BindlessBufferHandle>);

impl Deref for VoxPalette {
    type Target = [U8Vec4];
    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(self.0.as_slice())
    }
}
impl DerefMut for VoxPalette {
    fn deref_mut(&mut self) -> &mut Self::Target {
        bytemuck::cast_slice_mut(self.0.as_slice_mut())
    }
}
impl VoxPalette {
    pub(crate) fn from_buffer(buffer: ManagedBuffer) -> Self {
        Self(buffer, None)
    }

    pub fn colorful(allocator: pumicite::Allocator) -> VkResult<Self> {
        use bevy::color::{Hsva, Srgba};
        let mut hue = 0.0;
        let saturation = 0.8;
        let value = 0.9;

        let mut arr: Box<[U8Vec4; 256]> = Box::new([U8Vec4::ZERO; 256]);
        for x in 0..256 {
            let color = Hsva::new(hue, saturation, value, 1.0);
            let rgb_color: Srgba = color.into();
            let rgb_color: [u8; 4] = rgb_color.to_u8_array();
            arr[x] = U8Vec4::from_array(rgb_color);
            hue += 360.0 / 256.0;
        }

        let mut buffer =
            ManagedBuffer::new(allocator, 256 * 4, 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        buffer
            .as_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&*arr));
        Ok(Self(buffer, None))
    }

    pub fn register_bindless(&mut self, heap: &ResourceHeap) -> VkResult<()> {
        if self.1.is_none() {
            self.1 = Some(BindlessBufferHandle::new(
                heap,
                BufferDescriptor::new(&self.0),
            )?);
        }
        Ok(())
    }

    pub fn bindless_handle(&self) -> Option<u32> {
        self.1.as_ref().map(BindlessBufferHandle::get)
    }

    pub fn flush(&self, encoder: &mut pumicite::command::CommandEncoder) {
        self.0.flush(encoder);
    }
}

#[derive(Asset, TypePath)]
pub struct VoxMaterialTable(ManagedBuffer, Option<BindlessBufferHandle>);

impl VoxMaterialTable {
    pub(crate) fn from_entries(
        allocator: Allocator,
        entries: &[VoxMaterialParam; 256],
    ) -> VkResult<Self> {
        let mut buffer = ManagedBuffer::new(
            allocator,
            std::mem::size_of_val(entries) as u64,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        buffer
            .as_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(entries));
        Ok(Self(buffer, None))
    }

    pub fn register_bindless(&mut self, heap: &ResourceHeap) -> VkResult<()> {
        if self.1.is_none() {
            self.1 = Some(BindlessBufferHandle::new(
                heap,
                BufferDescriptor::new(&self.0),
            )?);
        }
        Ok(())
    }

    pub fn bindless_handle(&self) -> Option<u32> {
        self.1.as_ref().map(BindlessBufferHandle::get)
    }

    pub fn flush(&self, encoder: &mut pumicite::command::CommandEncoder) {
        self.0.flush(encoder);
    }
}

/// Marker component for Vox instances
#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct VoxInstance;

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct VoxModel {
    pub geometry: Handle<VoxGeometry>,
    pub material: Handle<VoxMaterial>,
    pub palette: Handle<VoxPalette>,
    pub material_table: Handle<VoxMaterialTable>,
}

#[derive(Bundle, Default)]
pub struct VoxModelBundle {
    pub model: VoxModel,
}

#[derive(Bundle, Default)]
pub struct VoxInstanceBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub instance: VoxInstance,
}

pub struct VoxPlugin;
impl Plugin for VoxPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default());

        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterialTable>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();
        runtime::runtime_voxel_systems(app);

        if app
            .world()
            .resource::<pumicite::physical_device::PhysicalDevice>()
            .properties()
            .device_type
            != vk::PhysicalDeviceType::INTEGRATED_GPU
        {
            app.add_systems(PostUpdate, sync_buffers_system.in_set(DefaultTransferSet));
        }

        app.add_systems(
            Startup,
            (|mut device_builder: ResMut<DeviceBuilder>| {
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFeatures| {
                        &mut features.shader_int64
                    })
                    .unwrap();
                device_builder
                    .enable_feature(|features: &mut vk::PhysicalDeviceFeatures| {
                        &mut features.shader_int16
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
            .before(CreateDevice),
        );

        app.add_systems(
            Startup,
            (|allocator: Res<Allocator>,
              asset_server: Res<AssetServer>,
              heap: Res<DescriptorHeap>| {
                asset_server.register_loader(VoxLoader::new(
                    allocator.clone(),
                    heap.resource_heap().clone(),
                ));
            })
            .after(CreateDevice),
        );
    }
}

fn sync_buffers_system(
    mut ctx: SubmissionState,
    mut material_events: MessageReader<AssetEvent<VoxMaterial>>,
    mut palette_events: MessageReader<AssetEvent<VoxPalette>>,
    mut material_table_events: MessageReader<AssetEvent<VoxMaterialTable>>,

    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>,
    material_tables: Res<Assets<VoxMaterialTable>>,
) {
    ctx.record(|encoder| {
        for event in material_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let material = materials.get(*id).unwrap();
                    material.buffer.flush(encoder);
                }
                _ => (),
            }
        }
        for event in palette_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let palette = palettes.get(*id).unwrap();
                    palette.flush(encoder);
                }
                _ => (),
            }
        }
        for event in material_table_events.read() {
            match event {
                AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                    let material_table = material_tables.get(*id).unwrap();
                    material_table.flush(encoder);
                }
                _ => (),
            }
        }
    });
}
