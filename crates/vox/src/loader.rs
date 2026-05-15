use std::collections::{BTreeMap, BTreeSet};

use bevy::{asset::AssetLoader, math::Vec3A, prelude::*, reflect::TypePath};
use bevy_pumicite::DescriptorHeap;
use dot_vox::{DotVoxData, Rotation, SceneNode};
use pumicite::bindless::ResourceHeap;
use pumicite::{Allocator, ash::vk, buffer::ManagedBuffer};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    VoxGeometry, VoxInstance, VoxInstanceBundle, VoxMaterial, VoxMaterialParam, VoxMaterialTable,
    VoxModel, VoxPalette,
};

enum WorldOrParent<'w, 'q> {
    World(&'w mut World),
    Parent(&'w mut ChildSpawner<'q>),
}

impl<'w, 'q> WorldOrParent<'w, 'q> {
    fn spawn(self, bundle: impl Bundle + Send + Sync + 'static) -> EntityWorldMut<'w> {
        match self {
            WorldOrParent::World(world) => world.spawn(bundle),
            WorldOrParent::Parent(parent) => parent.spawn(bundle),
        }
    }
}

struct SceneGraphTraverser<'a> {
    scene: &'a DotVoxData,
    models: BTreeSet<u32>,
    instances: Vec<(u32, Entity)>,
}

impl<'a> SceneGraphTraverser<'a> {
    fn traverse(
        &mut self,
        node: u32,
        parent: WorldOrParent<'_, '_>,
        translation: IVec3,
        rotation: Rotation,
        name: Option<&str>,
    ) {
        if self.scene.scenes.is_empty() {
            // Shape nodes are leafs and correspond to models
            assert_eq!(self.scene.models.len(), 1);
            let model = &self.scene.models[0];
            if model.voxels.len() == 0 {
                return;
            }
            let entity = parent
                .spawn(VoxInstanceBundle {
                    transform: Transform::default(),
                    global_transform: GlobalTransform::default(),
                    instance: VoxInstance,
                })
                .id();
            self.instances.push((0, entity));
            self.models.insert(0);
            return;
        }
        self.traverse_recursive(node, parent, translation, rotation, name);
    }
    fn traverse_recursive(
        &mut self,
        node: u32,
        parent: WorldOrParent<'_, '_>,
        translation: IVec3,
        rotation: Rotation,
        _name: Option<&str>,
    ) {
        let node = &self.scene.scenes[node as usize];
        match node {
            SceneNode::Transform {
                attributes,
                frames,
                child,
                layer_id: _,
            } => {
                if frames.len() != 1 {
                    unimplemented!("Multiple frame in transform node");
                }
                let name = attributes.get("_name").map(String::as_str);
                let frame = &frames[0];
                let this_translation = frame
                    .position()
                    .map(|position| IVec3 {
                        x: position.x,
                        y: position.y,
                        z: position.z,
                    })
                    .unwrap_or(IVec3::ZERO);

                let this_rotation = frame.orientation().unwrap_or(Rotation::IDENTITY);
                //let rotation = rotation * this_rotation; // reverse?
                let translation = translation + this_translation;

                self.traverse_recursive(*child, parent, translation, this_rotation, name);
            }
            SceneNode::Group {
                attributes: _,
                children,
            } => {
                parent
                    .spawn((
                        self.to_transform(translation, rotation, UVec3::ZERO),
                        GlobalTransform::default(),
                    ))
                    .with_children(|builder| {
                        for &i in children {
                            self.traverse_recursive(
                                i,
                                WorldOrParent::Parent(builder),
                                IVec3::ZERO,
                                Rotation::IDENTITY,
                                None,
                            );
                        }
                    });
            }
            SceneNode::Shape {
                attributes: _,
                models,
            } => {
                // Shape nodes are leafs and correspond to models
                if models.len() != 1 {
                    unimplemented!("Multiple shape models in Shape node");
                }
                let shape_model = &models[0];
                let model = &self.scene.models[shape_model.model_id as usize];
                if model.voxels.len() == 0 {
                    return;
                }
                let size = self.scene.models[shape_model.model_id as usize].size;
                let entity = parent
                    .spawn(VoxInstanceBundle {
                        transform: self.to_transform(
                            translation,
                            rotation,
                            UVec3 {
                                x: size.x,
                                y: size.y,
                                z: size.z,
                            },
                        ),
                        ..Default::default()
                    })
                    .id();
                self.instances.push((shape_model.model_id, entity));
                self.models.insert(shape_model.model_id);
            }
        }
    }

    fn to_transform(&self, translation: IVec3, rotation: Rotation, size: UVec3) -> Transform {
        let mut translation = translation.as_vec3a().xzy();
        translation.z *= -1.0;

        let (quat, scale) = rotation.to_quat_scale();
        let quat = Quat::from_array(quat);
        let quat = Quat::from_xyzw(quat.x, quat.z, -quat.y, quat.w);
        let scale = Vec3A::from_array(scale).xzy(); // no need to negate scale.y because scale is not a coordinate

        let mut offset = Vec3A::new(
            if size.x % 2 == 0 { 0.0 } else { 0.5 },
            if size.z % 2 == 0 { 0.0 } else { 0.5 },
            if size.y % 2 == 0 { 0.0 } else { -0.5 },
        );
        offset = quat.mul_vec3a(offset); // If another seam shows up in the future, try multiplying this with `scale`
        let center = quat * (size.xzy().as_vec3a() / 2.0);
        Transform {
            translation: (translation - center * scale + offset).into(),
            rotation: quat,
            scale: scale.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VoxLoadingError {
    #[error("parse error: {0}")]
    ParseError(&'static str),
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("vulkan error: {0}")]
    VulkanError(#[from] vk::Result),
}

#[derive(TypePath)]
pub struct VoxLoader {
    allocator: Allocator,
    heap: ResourceHeap,
}
impl VoxLoader {
    pub fn new(allocator: Allocator, heap: ResourceHeap) -> Self {
        Self { allocator, heap }
    }
}
impl FromWorld for VoxLoader {
    fn from_world(world: &mut World) -> Self {
        Self {
            allocator: world.resource::<Allocator>().clone(),
            heap: world.resource::<DescriptorHeap>().resource_heap().clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxLoaderSettings {
    pub unit_size: f32,
}
impl Default for VoxLoaderSettings {
    fn default() -> Self {
        Self { unit_size: 1.0 }
    }
}

impl AssetLoader for VoxLoader {
    type Asset = Scene;
    type Settings = VoxLoaderSettings;
    type Error = VoxLoadingError;
    fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext,
    ) -> impl bevy::tasks::ConditionalSendFuture<Output = Result<Scene, VoxLoadingError>> {
        async {
            tracing::info!("Loading vox file {:?}", load_context.path());
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).await?;
            let mut file = dot_vox::load_bytes(buffer.as_slice())
                .map_err(|reason| VoxLoadingError::ParseError(reason))?;
            tracing::info!("Vox file deserialized: {} models", file.models.len());

            let mut world = World::default();
            let mut traverser = SceneGraphTraverser {
                scene: &file,
                models: BTreeSet::new(),
                instances: Vec::new(),
            };
            traverser.traverse(
                0,
                WorldOrParent::World(&mut world),
                IVec3::ZERO,
                Rotation::IDENTITY,
                None,
            );
            let referenced_models = std::mem::take(&mut traverser.models);
            let referenced_instances = std::mem::take(&mut traverser.instances);
            drop(traverser);

            tracing::info!(
                "Scene graph traversed: {} models, {} instances",
                referenced_models.len(),
                referenced_instances.len()
            );

            let material_table_entries = material_table_entries(&file.palette, &file.materials);
            let material_table_handle = load_context.add_labeled_asset("MaterialTable".into(), {
                let mut material_table =
                    VoxMaterialTable::from_entries(self.allocator.clone(), &material_table_entries)
                        .map_err(VoxLoadingError::VulkanError)?;
                material_table
                    .register_bindless(&self.heap)
                    .map_err(VoxLoadingError::VulkanError)?;
                material_table
            });

            let palette_handle = load_context.add_labeled_asset("Palette".into(), {
                let mut palette = VoxPalette::from_buffer(unsafe {
                    let arr = std::mem::take(&mut file.palette).into_boxed_slice();
                    assert_eq!(arr.len(), 256);
                    let mut buffer = ManagedBuffer::new(
                        self.allocator.clone(),
                        256 * 4,
                        4,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .map_err(VoxLoadingError::VulkanError)?;
                    buffer
                        .as_slice_mut()
                        .copy_from_slice(std::slice::from_raw_parts::<u8>(
                            arr.as_ptr() as *const u8,
                            256 * 4,
                        ));
                    buffer
                });
                palette
                    .register_bindless(&self.heap)
                    .map_err(VoxLoadingError::VulkanError)?;
                palette
            });

            let model_handles: BTreeMap<u32, VoxModel> = {
                // Add models
                let mut models: Vec<_> = std::mem::take(&mut file.models)
                    .into_iter()
                    .map(|a| Some(a))
                    .collect();
                let models = referenced_models
                    .iter()
                    .map(|model_id| {
                        (
                            *model_id,
                            models.get_mut(*model_id as usize).unwrap().take().unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                let handles = models
                    .par_iter()
                    .map(|(model_id, model)| -> Result<_, VoxLoadingError> {
                        let (mut tree, mut attribute_allocator) =
                            self.model_to_tree(model, settings.unit_size);
                        tree.register_bindless(&self.heap)
                            .map_err(VoxLoadingError::VulkanError)?;
                        attribute_allocator
                            .register_bindless(&self.heap)
                            .map_err(VoxLoadingError::VulkanError)?;
                        Ok((*model_id, (tree, attribute_allocator)))
                    })
                    .collect_vec_list();
                let handles = handles
                    .into_iter()
                    .flat_map(|a| a)
                    .collect::<Result<Vec<_>, _>>()?;
                BTreeMap::from_iter(handles.into_iter().map(|(model_id, (tree, material))| {
                    let geometry =
                        load_context.add_labeled_asset(format!("Geometry{}", model_id), tree);
                    let material =
                        load_context.add_labeled_asset(format!("Material{}", model_id), material);
                    (
                        model_id,
                        VoxModel {
                            geometry,
                            material,
                            palette: palette_handle.clone(),
                            material_table: material_table_handle.clone(),
                        },
                    )
                }))
            };

            for (model_id, entity) in referenced_instances {
                let Some(model) = model_handles.get(&model_id) else {
                    continue;
                };
                world.entity_mut(entity).insert(VoxModel {
                    geometry: model.geometry.clone(),
                    material: model.material.clone(),
                    palette: model.palette.clone(),
                    material_table: model.material_table.clone(),
                });
            }

            let scene = bevy::scene::Scene::new(world);

            tracing::info!("Scene spawned");
            Ok(scene)
        }
    }

    fn extensions(&self) -> &[&str] {
        &["vox"]
    }
}
impl VoxLoader {
    fn model_to_tree(&self, model: &dot_vox::Model, unit_size: f32) -> (VoxGeometry, VoxMaterial) {
        let mut geometry = VoxGeometry::new(self.allocator.clone(), unit_size);
        let mut material = VoxMaterial::new(self.allocator.clone());

        // Create 256x256x256 grid
        let mut accessor = geometry.tree.accessor_mut(&mut material);
        let size_y = model.size.y;

        let mut min = UVec3::MAX;
        let mut max = UVec3::MIN;
        for voxel in model.voxels.iter() {
            let voxel = dot_vox::Voxel {
                x: voxel.x,
                y: voxel.z,
                z: (size_y - voxel.y as u32 - 1) as u8,
                i: voxel.i,
            };
            let coords: UVec3 = UVec3 {
                x: voxel.x as u32,
                y: voxel.y as u32,
                z: voxel.z as u32,
            };

            accessor.set(coords, voxel.i + 1);
            min = min.min(coords);
            max = max.max(coords);
        }

        accessor.end();
        // TODO: material.0.buffer_mut().flush(..);

        (geometry, material)
    }
}

fn material_table_entries(
    palette: &[dot_vox::Color],
    materials: &[dot_vox::Material],
) -> [VoxMaterialParam; 256] {
    let mut entries = [VoxMaterialParam::default(); 256];
    for (index, entry) in entries.iter_mut().enumerate() {
        if let Some(color) = palette.get(index) {
            entry.base_color = [
                color.r as f32 / 255.0,
                color.g as f32 / 255.0,
                color.b as f32 / 255.0,
                1.0,
            ];
        }
    }

    for material in materials {
        let Some(index) = material_table_index(material.id) else {
            continue;
        };

        let material_type = material.properties.get("_type").map(String::as_str);
        let weight = material_float(material, "_weight")
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let metallic = match material_type {
            Some("_metal") => weight,
            _ => material_float(material, "_metal").unwrap_or(0.0),
        }
        .clamp(0.0, 1.0);
        let roughness = material_float(material, "_rough")
            .unwrap_or(entries[index].pbr[1])
            .clamp(0.04, 1.0);
        let mut specular = material_float(material, "_spec")
            .or_else(|| material_float(material, "_sp"))
            .unwrap_or(entries[index].pbr[2])
            .clamp(0.0, 1.0);
        if let Some(ior) = material_float(material, "_ior") {
            let ior = ior.max(1.0);
            let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
            specular = (f0 / 0.08).clamp(0.0, 1.0);
        }

        let emission = match material_type {
            Some("_emit") => weight.max(material_float(material, "_emit").unwrap_or(0.0)),
            _ => material_float(material, "_emit").unwrap_or(0.0),
        };
        let emission = (emission * material_float(material, "_flux").unwrap_or(1.0)).max(0.0);
        entries[index].pbr = [metallic, roughness, specular, emission];
    }

    entries
}

fn material_float(material: &dot_vox::Material, key: &str) -> Option<f32> {
    material.properties.get(key)?.parse().ok()
}

fn material_table_index(material_id: u32) -> Option<usize> {
    let index = if material_id == 0 {
        0
    } else {
        material_id.checked_sub(1)? as usize
    };
    (index < 256).then_some(index)
}
