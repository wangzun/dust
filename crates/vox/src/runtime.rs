use std::collections::HashMap;

use bevy::{
    asset::AssetId,
    ecs::schedule::IntoScheduleConfigs,
    math::UVec3,
    prelude::{Commands, Component, Entity, Handle, Query, Res, ResMut, Resource},
};
use bevy_pumicite::{DefaultTransferSet, DescriptorHeap, SubmissionState};
use pumicite::{Allocator, ash::VkResult, bindless::ResourceHeap};

use crate::{VoxGeometry, VoxMaterial, VoxMaterialTable, VoxModel};

#[derive(Clone, Copy, Debug)]
pub struct RuntimeVoxel {
    pub coords: UVec3,
    /// 0-based palette/material table index.
    pub material: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeVoxelModelId(pub AssetId<VoxGeometry>);

#[derive(Component, Clone, Copy, Debug)]
pub struct RuntimeVoxelModelRef {
    pub id: RuntimeVoxelModelId,
}

pub struct RuntimeVoxelModel {
    pub geometry: VoxGeometry,
    pub material: VoxMaterial,
    pub material_table: Handle<VoxMaterialTable>,
    pub voxels: Vec<RuntimeVoxel>,
    lookup: HashMap<UVec3, usize>,
    bounds: Option<(bevy::math::Vec3, bevy::math::Vec3)>,
    pub revision: u64,
    dirty: bool,
}

impl RuntimeVoxelModel {
    pub fn from_assets(
        allocator: Allocator,
        heap: &ResourceHeap,
        geometry: &VoxGeometry,
        material: &VoxMaterial,
        material_table: Handle<VoxMaterialTable>,
    ) -> VkResult<Self> {
        let voxels = geometry.export_voxels(material);
        Self::from_voxels(allocator, heap, geometry.unit_size, material_table, voxels)
    }

    pub fn from_voxels(
        allocator: Allocator,
        heap: &ResourceHeap,
        unit_size: f32,
        material_table: Handle<VoxMaterialTable>,
        voxels: Vec<RuntimeVoxel>,
    ) -> VkResult<Self> {
        let lookup = voxel_lookup(&voxels);
        let bounds = voxel_bounds(&voxels);
        let (geometry, material) = build_gpu_voxels(allocator, heap, unit_size, &voxels)?;
        Ok(Self {
            geometry,
            material,
            material_table,
            voxels,
            lookup,
            bounds,
            revision: 0,
            dirty: false,
        })
    }

    pub fn bounds(&self) -> Option<(bevy::math::Vec3, bevy::math::Vec3)> {
        self.bounds
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn set_voxel(&mut self, coords: UVec3, material: u8) {
        if let Some(index) = self.lookup.get(&coords).copied() {
            self.voxels[index].material = material;
        } else {
            self.lookup.insert(coords, self.voxels.len());
            self.voxels.push(RuntimeVoxel { coords, material });
            self.bounds = voxel_bounds(&self.voxels);
        }
        self.dirty = true;
    }

    pub fn remove_voxel(&mut self, coords: UVec3) -> Option<RuntimeVoxel> {
        let index = self.lookup.remove(&coords)?;
        let removed = self.voxels.swap_remove(index);
        if let Some(swapped) = self.voxels.get(index) {
            self.lookup.insert(swapped.coords, index);
        }
        self.bounds = voxel_bounds(&self.voxels);
        self.dirty = true;
        Some(removed)
    }

    pub fn rebuild_if_dirty(&mut self, allocator: Allocator, heap: &ResourceHeap) -> VkResult<()> {
        if !self.dirty {
            return Ok(());
        }
        let (geometry, material) =
            build_gpu_voxels(allocator, heap, self.geometry.unit_size, &self.voxels)?;
        self.geometry = geometry;
        self.material = material;
        self.lookup = voxel_lookup(&self.voxels);
        self.bounds = voxel_bounds(&self.voxels);
        self.revision = self.revision.wrapping_add(1);
        self.dirty = false;
        Ok(())
    }
}

#[derive(Resource, Default)]
pub struct RuntimeVoxelWorld {
    pub models: HashMap<RuntimeVoxelModelId, RuntimeVoxelModel>,
}

impl RuntimeVoxelWorld {
    pub fn model(&self, id: RuntimeVoxelModelId) -> Option<&RuntimeVoxelModel> {
        self.models.get(&id)
    }

    pub fn model_mut(&mut self, id: RuntimeVoxelModelId) -> Option<&mut RuntimeVoxelModel> {
        self.models.get_mut(&id)
    }

    pub fn ensure_model_from_assets(
        &mut self,
        id: RuntimeVoxelModelId,
        allocator: Allocator,
        heap: &ResourceHeap,
        geometry: &VoxGeometry,
        material: &VoxMaterial,
        material_table: Handle<VoxMaterialTable>,
    ) -> VkResult<&mut RuntimeVoxelModel> {
        if !self.models.contains_key(&id) {
            let runtime_model = RuntimeVoxelModel::from_assets(
                allocator,
                heap,
                geometry,
                material,
                material_table,
            )?;
            self.models.insert(id, runtime_model);
        }
        Ok(self.models.get_mut(&id).expect("runtime model inserted"))
    }
}

pub(crate) fn runtime_voxel_systems(app: &mut bevy::prelude::App) {
    app.init_resource::<RuntimeVoxelWorld>().add_systems(
        bevy::prelude::PostUpdate,
        sync_runtime_voxel_world.in_set(DefaultTransferSet),
    );
}

fn sync_runtime_voxel_world(
    mut commands: Commands,
    mut ctx: SubmissionState,
    allocator: Res<Allocator>,
    heap: Res<DescriptorHeap>,
    mut runtime_world: ResMut<RuntimeVoxelWorld>,
    mut models: Query<(Entity, &VoxModel, Option<&mut RuntimeVoxelModelRef>)>,
) {
    let heap = heap.resource_heap();
    let mut flush_models = Vec::new();
    for (id, model) in runtime_world.models.iter_mut() {
        if model.is_dirty() {
            model
                .rebuild_if_dirty(allocator.clone(), heap)
                .expect("failed to rebuild dirty runtime voxel model");
            flush_models.push(*id);
        }
    }

    for (entity, model, runtime_ref) in models.iter_mut() {
        if runtime_ref.is_some() {
            continue;
        }

        commands.entity(entity).insert(RuntimeVoxelModelRef {
            id: RuntimeVoxelModelId(model.geometry.id()),
        });
    }

    ctx.record(|encoder| {
        for id in flush_models {
            let Some(model) = runtime_world.models.get(&id) else {
                continue;
            };
            model.material.buffer.flush(encoder);
        }
    });
}

fn build_gpu_voxels(
    allocator: Allocator,
    heap: &ResourceHeap,
    unit_size: f32,
    voxels: &[RuntimeVoxel],
) -> VkResult<(VoxGeometry, VoxMaterial)> {
    let mut geometry = VoxGeometry::new(allocator.clone(), unit_size);
    let mut material = VoxMaterial::new(allocator);
    {
        let mut accessor = geometry.tree.accessor_mut(&mut material);
        for voxel in voxels {
            accessor.set(voxel.coords, voxel.material.saturating_add(1));
        }
        accessor.end();
    }
    geometry.register_bindless(heap)?;
    material.register_bindless(heap)?;
    Ok((geometry, material))
}

fn voxel_lookup(voxels: &[RuntimeVoxel]) -> HashMap<UVec3, usize> {
    voxels
        .iter()
        .enumerate()
        .map(|(index, voxel)| (voxel.coords, index))
        .collect()
}

fn voxel_bounds(voxels: &[RuntimeVoxel]) -> Option<(bevy::math::Vec3, bevy::math::Vec3)> {
    let mut iter = voxels.iter();
    let first = iter.next()?;
    let mut min = first.coords;
    let mut max = first.coords;
    for voxel in iter {
        min = min.min(voxel.coords);
        max = max.max(voxel.coords);
    }
    Some((min.as_vec3(), max.as_vec3() + bevy::math::Vec3::ONE))
}
