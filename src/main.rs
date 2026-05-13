use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use pumicite::ash::vk;

const TEAPOT_CENTER: Vec3 = Vec3::new(62.5, 30.0, 40.0);
const TEAPOT_CAMERA_POSITION: Vec3 = Vec3::new(62.5, 35.0, 220.0);

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

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert((
            dust_pbr::camera::Camera::default(),
            GlobalTransform::default(),
            Transform::from_translation(TEAPOT_CAMERA_POSITION).looking_at(TEAPOT_CENTER, Vec3::Y),
            FreeCamera {
                sensitivity: 0.2,
                friction: 25.0,
                walk_speed: 60.0,
                run_speed: 180.0,
                ..default()
            },
        ))
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.add_systems(Startup, startup_system.after(CreateDevice));

    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>) {
    let teapot: Handle<Scene> = asset_server.load("teapot.vox");
    commands.spawn(SceneRoot(teapot));
}
