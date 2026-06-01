use avian3d::collider_tree::{ColliderTreeType, ColliderTrees};
use avian3d::prelude::*;
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::picking::{
    backend::ray::RayMap,
    pointer::{PointerId, PointerInteraction},
};
use bevy::prelude::*;
use bevy_pumicite::CreateDevice;
use bevy_pumicite::PumiciteApp;
use dust_dense::{DenseVoxelGeometry, DenseVoxelMaterial, DenseVoxelModel};
use dust_pbr::camera::{Camera as SoftwareCamera, SoftwareVoxelCamera, SoftwareVoxelPixelArt};
use obvhs::cwbvh::bvh2_to_cwbvh::bvh2_to_cwbvh;
use pumicite::ash::vk;

const TEAPOT_CENTER: Vec3 = Vec3::new(62.5, 30.0, 40.0);
const TEAPOT_CAMERA_POSITION: Vec3 = Vec3::new(62.5, 35.0, 220.0);
// const SCENE_PATH: &str = "teapot.vox";
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
    app.add_plugins(PhysicsPlugins::default());
    app.add_plugins(PhysicsPickingPlugin);

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
        Camera3d::default(),
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
    app.add_systems(Update, delete_voxel_by_hit);
    app.add_systems(Update, get_collider_tree_bvh_to_raw_data);
    // app.add_systems(Update, frame_camera_to_loaded_scene);

    app.run();
}

fn startup_system(mut commands: Commands, asset_server: Res<bevy::asset::AssetServer>) {
    let scene: Handle<Scene> = asset_server.load(SCENE_PATH);
    commands.spawn((SceneRoot(scene.clone()),));
}

fn delete_voxel_by_hit(
    buttons: Res<ButtonInput<MouseButton>>,
    ray_map: Res<RayMap>,
    pointers: Query<(&PointerId, &PointerInteraction)>,
    parents: Query<&ChildOf>,
    models: Query<(&DenseVoxelModel, &GlobalTransform)>,
    mut colliders: Query<&mut Collider>,
    mut dense_geometries: ResMut<Assets<DenseVoxelGeometry>>,
    mut dense_materials: ResMut<Assets<DenseVoxelMaterial>>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    for (pointer_id, interaction) in pointers.iter() {
        let Some((hit_entity, hit)) = interaction.get_nearest_hit() else {
            continue;
        };
        println!("Hit entity {hit_entity:?} at {hit:?}");
        let Some(ray) = ray_map
            .iter()
            .find(|(ray_id, _)| ray_id.camera == hit.camera && ray_id.pointer == *pointer_id)
            .map(|(_, ray)| *ray)
        else {
            continue;
        };

        let Some((model_entity, model, transform)) =
            find_dense_model_entity(*hit_entity, &models, &parents)
        else {
            continue;
        };

        let Some(geometry) = dense_geometries.get_mut(&model.occupancy) else {
            continue;
        };

        let inverse = transform.affine().inverse();
        let local_origin = inverse.transform_point3(ray.origin);
        let local_dir = inverse.transform_vector3(*ray.direction);
        let Some(coords) =
            raycast_dense_voxels(local_origin, local_dir, hit.depth, geometry, model)
        else {
            continue;
        };

        let Some(material) = dense_materials.get_mut(&model.material) else {
            continue;
        };
        println!(
            "Clearing voxel at coords {:?} in model entity {model_entity:?}",
            coords
        );
        DenseVoxelModel::clear_voxel(geometry, material, coords);

        if let Ok(mut collider) = colliders.get_mut(model_entity) {
            println!("Updating collider for model entity {model_entity:?}");
            // *collider = Collider::voxels(Vec3::ONE, &dense_collider_voxels(geometry));
        }

        return;
    }
}

fn find_dense_model_entity<'a>(
    entity: Entity,
    models: &'a Query<(&DenseVoxelModel, &GlobalTransform)>,
    parents: &Query<&ChildOf>,
) -> Option<(Entity, &'a DenseVoxelModel, &'a GlobalTransform)> {
    let mut current = entity;
    loop {
        if let Ok((model, transform)) = models.get(current) {
            return Some((current, model, transform));
        }
        current = parents.get(current).ok()?.parent();
    }
}

fn raycast_dense_voxels(
    origin: Vec3,
    dir: Vec3,
    hit_t: f32,
    geometry: &DenseVoxelGeometry,
    model: &DenseVoxelModel,
) -> Option<[u32; 3]> {
    let dir_len = dir.length();
    if dir_len <= f32::EPSILON {
        return None;
    }

    let size = geometry.size();
    let size = UVec3::new(size[0], size[1], size[2]);
    let bounds_min = model.cull_min.min(size);
    let bounds_max = model.cull_max.min(size);
    if bounds_min.cmpge(bounds_max).any() {
        return None;
    }

    let (enter_t, exit_t) =
        intersect_bounds(origin, dir, bounds_min.as_vec3(), bounds_max.as_vec3())?;
    let start_t = hit_t.max(enter_t);
    if start_t > exit_t {
        return None;
    }
    let bounds_min = bounds_min.as_ivec3();
    let bounds_max = bounds_max.as_ivec3();
    let start = origin + dir * (start_t + 0.001 / dir_len);
    let mut voxel = start
        .floor()
        .as_ivec3()
        .clamp(bounds_min, bounds_max - IVec3::ONE);
    let step = IVec3::new(axis_step(dir.x), axis_step(dir.y), axis_step(dir.z));
    let mut next_t = Vec3::new(
        next_voxel_boundary_t(origin.x, dir.x, voxel.x, step.x),
        next_voxel_boundary_t(origin.y, dir.y, voxel.y, step.y),
        next_voxel_boundary_t(origin.z, dir.z, voxel.z, step.z),
    );
    let delta_t = Vec3::new(
        voxel_axis_delta_t(dir.x),
        voxel_axis_delta_t(dir.y),
        voxel_axis_delta_t(dir.z),
    );

    while voxel.cmpge(bounds_min).all() && voxel.cmplt(bounds_max).all() {
        let coords = [voxel.x as u32, voxel.y as u32, voxel.z as u32];
        if geometry.is_occupied(coords) {
            return Some(coords);
        }

        let axis_t = next_t.x.min(next_t.y).min(next_t.z);
        if axis_t > exit_t {
            break;
        }

        if next_t.x == axis_t {
            voxel.x += step.x;
            next_t.x += delta_t.x;
        }
        if next_t.y == axis_t {
            voxel.y += step.y;
            next_t.y += delta_t.y;
        }
        if next_t.z == axis_t {
            voxel.z += step.z;
            next_t.z += delta_t.z;
        }
    }

    None
}

fn intersect_bounds(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<(f32, f32)> {
    let mut enter: f32 = 0.0;
    let mut exit = f32::INFINITY;

    for (origin, dir, min, max) in [
        (origin.x, dir.x, min.x, max.x),
        (origin.y, dir.y, min.y, max.y),
        (origin.z, dir.z, min.z, max.z),
    ] {
        if dir.abs() <= f32::EPSILON {
            if origin < min || origin > max {
                return None;
            }
            continue;
        }

        let mut axis_enter = (min - origin) / dir;
        let mut axis_exit = (max - origin) / dir;
        if axis_enter > axis_exit {
            std::mem::swap(&mut axis_enter, &mut axis_exit);
        }
        enter = enter.max(axis_enter);
        exit = exit.min(axis_exit);
        if enter > exit {
            return None;
        }
    }

    Some((enter, exit))
}

fn axis_step(dir: f32) -> i32 {
    if dir > 0.0 {
        1
    } else if dir < 0.0 {
        -1
    } else {
        0
    }
}

fn next_voxel_boundary_t(origin: f32, dir: f32, voxel: i32, step: i32) -> f32 {
    if step == 0 {
        return f32::INFINITY;
    }
    let boundary = if step > 0 { voxel + 1 } else { voxel } as f32;
    (boundary - origin) / dir
}

fn voxel_axis_delta_t(dir: f32) -> f32 {
    if dir == 0.0 {
        f32::INFINITY
    } else {
        1.0 / dir.abs()
    }
}

fn dense_collider_voxels(geometry: &DenseVoxelGeometry) -> Vec<IVec3> {
    let size = geometry.size();
    let block_extent = dense_block_extent(size);
    let mut voxels = Vec::new();
    for (index, word) in geometry.occupancy().iter().copied().enumerate() {
        let block = dense_block_coords(index as u32, block_extent);
        let block_min = block * UVec3::splat(4);
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros();
            remaining &= remaining - 1;
            let local = UVec3::new(bit & 3, (bit >> 2) & 3, (bit >> 4) & 3);
            let voxel = block_min + local;
            if voxel.x < size[0] && voxel.y < size[1] && voxel.z < size[2] {
                voxels.push(IVec3::new(voxel.x as i32, voxel.y as i32, voxel.z as i32));
            }
        }
    }
    voxels
}

fn dense_block_extent(size: [u32; 3]) -> UVec3 {
    (UVec3::new(size[0], size[1], size[2]) + UVec3::splat(3)) / UVec3::splat(4)
}

fn dense_block_coords(index: u32, block_extent: UVec3) -> UVec3 {
    let x = index % block_extent.x;
    let yz = index / block_extent.x;
    let y = yz % block_extent.y;
    let z = yz / block_extent.y;
    UVec3::new(x, y, z)
}

fn get_collider_tree_bvh_to_raw_data(mut collider_trees: ResMut<ColliderTrees>) {
    let tree = collider_trees.tree_for_type_mut(ColliderTreeType::Static);
    let bvh = &mut tree.bvh;
    let cwbvh = bvh2_to_cwbvh(&bvh, 3, true, false);
    let raw_data = cwbvh.primitive_indices;
    // println!("Raw BVH data: {raw_data:?}");
}
