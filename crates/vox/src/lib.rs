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
use bevy_pumicite::{CreateDevice, DefaultTransferSet, SubmissionState};
use dust_vdb::hierarchy;
use pumicite::Allocator;
use pumicite::ash::{VkResult, vk};
use pumicite::buffer::{BufferLike, ManagedBuffer};
use pumicite::device::DeviceBuilder;
use std::ops::{Deref, DerefMut};

mod geometry;
mod loader;
mod material;

pub use material::{VoxLeafNode, VoxMaterial};

/// Leaf node size: 96 bytes
type TreeRoot = hierarchy!(3, 3, 2, VoxLeafNode);

type Tree = dust_vdb::Tree<TreeRoot>;

pub use loader::*;

pub use geometry::VoxGeometry;

#[derive(Asset, TypePath)]
pub struct VoxPalette(ManagedBuffer);

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

        let mut buffer = ManagedBuffer::new(
            allocator,
            256 * 4,
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        buffer
            .as_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&*arr));
        Ok(Self(buffer))
    }

    pub fn device_address(&self) -> u64 {
        self.0.device_address()
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
        app.init_asset::<VoxGeometry>()
            .init_asset::<VoxPalette>()
            .init_asset::<VoxMaterial>()
            .register_type::<VoxInstance>()
            .register_type::<VoxModel>();

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
                    .enable_extension::<pumicite::ash::khr::push_descriptor::Meta>()
                    .unwrap();
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
            (|allocator: Res<Allocator>, asset_server: Res<AssetServer>| {
                asset_server.register_loader(VoxLoader::new(allocator.clone()));
            })
            .after(CreateDevice),
        );
    }
}

fn sync_buffers_system(
    mut ctx: SubmissionState,
    mut material_events: MessageReader<AssetEvent<VoxMaterial>>,
    mut palette_events: MessageReader<AssetEvent<VoxPalette>>,

    materials: Res<Assets<VoxMaterial>>,
    palettes: Res<Assets<VoxPalette>>,
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
                    palette.0.flush(encoder);
                }
                _ => (),
            }
        }
    });
}
