use std::collections::HashMap;

use bevy::asset::AssetId;
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use bevy::math::Affine3A;
use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use bevy_pumicite::PumiciteApp;
use dust_pbr::camera::{Camera as SoftwareCamera, SoftwareVoxelCamera, SoftwareVoxelPixelArt};
use dust_vox::{VoxGeometry, VoxModel};
use pumicite::ash::vk;

const TEAPOT_CENTER: Vec3 = Vec3::new(62.5, 30.0, 40.0);
const TEAPOT_CAMERA_POSITION: Vec3 = Vec3::new(62.5, 35.0, 220.0);
const SCENE_PATH: &str = "castle.vox";

fn main() {
    let mut app = bevy::app::App::new();

    app.add_plugins(dust_app::DustApp)
        .add_plugins(bevy::DefaultPlugins)
        .add_plugins(bevy_pumicite::SurfacePlugin::default())
        .add_plugins(bevy_pumicite::DebugUtilsPlugin::default())
        .add_plugins(bevy_pumicite::PumicitePlugin::default())
        .add_plugins(bevy_pumicite::swapchain::SwapchainPlugin);

    app.add_plugins(FreeCameraPlugin);

    // Dust plugins
    app.add_plugins(dust_pbr::PbrRenderPlugin)
        .add_plugins(dust_vox::VoxPlugin);
    app.enable_bindless().expect("Bindless not supported");

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.world_mut().spawn((
        bevy::camera::Camera {
            is_active: false,
            ..default()
        },
        SoftwareCamera {
            pixel_art: SoftwareVoxelPixelArt {
                enabled: false,
                pixel_size: 2.0,
                outline_strength: 2.0,
            },
            ..default()
        },
        SoftwareVoxelCamera,
        Transform::from_translation(TEAPOT_CAMERA_POSITION).looking_at(TEAPOT_CENTER, Vec3::Y),
        FreeCamera {
            sensitivity: 0.2,
            friction: 25.0,
            walk_speed: 60.0,
            run_speed: 180.0,
            ..default()
        },
    ));

    app.add_systems(Startup, startup_system.after(CreateDevice));
    app.add_systems(Update, frame_camera_to_loaded_scene);

    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>) {
    let scene: Handle<Scene> = asset_server.load(SCENE_PATH);
    commands.spawn(SceneRoot(scene));
}

fn frame_camera_to_loaded_scene(
    mut framed: Local<bool>,
    mut bounds_cache: Local<HashMap<AssetId<VoxGeometry>, Option<(Vec3, Vec3)>>>,
    geometries: Res<Assets<VoxGeometry>>,
    models: Query<
        (&VoxModel, Option<&GlobalTransform>, Option<&Transform>),
        Without<SoftwareVoxelCamera>,
    >,
    mut cameras: ParamSet<(
        Query<(Entity, &SoftwareCamera, &mut Transform), With<SoftwareVoxelCamera>>,
        Query<&mut FreeCameraState>,
    )>,
) {
    if *framed {
        return;
    }

    let Ok((camera_entity, tan_half_fov)) = cameras
        .p0()
        .single_mut()
        .map(|(entity, camera, _)| (entity, camera.tan_half_fov()))
    else {
        return;
    };
    let mut pending_transform = None;

    let mut scene_min = Vec3::splat(f32::INFINITY);
    let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
    let mut saw_model = false;
    let mut all_ready = true;

    for (model, global_transform, local_transform) in models.iter() {
        let Some(geometry) = geometries.get(&model.geometry) else {
            all_ready = false;
            continue;
        };
        let Some((local_min, local_max)) =
            geometry_bounds(&mut bounds_cache, model.geometry.id(), geometry)
        else {
            continue;
        };

        let affine = global_transform
            .map(GlobalTransform::affine)
            .unwrap_or_else(|| {
                local_transform
                    .map(Transform::compute_affine)
                    .unwrap_or(Affine3A::IDENTITY)
            });
        for corner in aabb_corners(
            local_min * geometry.unit_size,
            local_max * geometry.unit_size,
        ) {
            let point = affine.transform_point3(corner);
            scene_min = scene_min.min(point);
            scene_max = scene_max.max(point);
        }
        saw_model = true;
    }

    if !all_ready || !saw_model {
        return;
    }

    let center = (scene_min + scene_max) * 0.5;
    let radius = ((scene_max - scene_min).length() * 0.5).max(1.0);
    let distance = radius / tan_half_fov.max(0.001) * 1.6;
    let position = center + Vec3::new(0.0, radius * 0.25, distance);
    let new_transform = Transform::from_translation(position).looking_at(center, Vec3::Y);

    if let Ok((_, _, mut camera_transform)) = cameras.p0().get_mut(camera_entity) {
        *camera_transform = new_transform;
        pending_transform = Some(*camera_transform);
    }

    if let (Some(transform), Ok(mut free_camera_state)) =
        (pending_transform, cameras.p1().get_mut(camera_entity))
    {
        let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
        free_camera_state.yaw = yaw;
        free_camera_state.pitch = pitch;
        free_camera_state.velocity = Vec3::ZERO;
    }

    *framed = true;
}

fn geometry_bounds(
    cache: &mut HashMap<AssetId<VoxGeometry>, Option<(Vec3, Vec3)>>,
    id: AssetId<VoxGeometry>,
    geometry: &VoxGeometry,
) -> Option<(Vec3, Vec3)> {
    *cache.entry(id).or_insert_with(|| {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for coord in geometry.tree.iter() {
            let p = coord.as_vec3();
            min = min.min(p);
            max = max.max(p + Vec3::ONE);
        }
        min.is_finite().then_some((min, max))
    })
}

fn aabb_corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ]
}
