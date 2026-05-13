use std::sync::Arc;

use crate::Tree;
use bevy::prelude::*;
use dust_vdb::pool::PoolStorage;
use pumicite::{
    Allocator,
    ash::vk,
    buffer::{Buffer, BufferLike},
    debug::DebugObject,
};

#[derive(Asset, TypePath)]
pub struct VoxGeometry {
    pub tree: Tree,

    /// Model space size of each voxel
    pub unit_size: f32,
}

pub struct VoxGeometryLeafStorage {
    allocator: Allocator,
    // A host-cached buffer that is preferably device-visible.
    // TODO: make this a managed buffer.
    buffer: Option<Arc<Buffer>>,
    alignment: usize,
    size: usize,
}
impl VoxGeometryLeafStorage {
    pub fn new(allocator: Allocator, alignment: usize) -> Self {
        Self {
            allocator,
            buffer: None,
            alignment,
            size: 0,
        }
    }
}
impl PoolStorage for VoxGeometryLeafStorage {
    fn device_address(&self) -> u64 {
        if let Some(buffer) = self.buffer.as_ref() {
            buffer.device_address()
        } else {
            0
        }
    }
    fn resize(&mut self, size: usize) -> *mut u8 {
        let mut new_buffer = Buffer::new_dynamic(
            self.allocator.clone(),
            size as u64,
            self.alignment as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )
        .unwrap()
        .with_name(c"VoxGeometryLeafStorage");
        unsafe {
            if let Some(old_buffer) = self.buffer.take() {
                std::ptr::copy_nonoverlapping(
                    old_buffer.as_ptr(),
                    new_buffer.as_mut_ptr(),
                    self.size.min(size),
                );
            }
        }

        let ptr = new_buffer.as_mut_ptr();
        assert!(!ptr.is_null());
        self.buffer = Some(Arc::new(new_buffer));
        self.size = size;
        ptr
    }
}

impl VoxGeometry {
    pub fn new(allocator: Allocator, unit_size: f32) -> Self {
        let tree = crate::Tree::new_with_leaf_storage(Box::new(VoxGeometryLeafStorage::new(
            allocator,
            crate::Tree::metas()[0].layout.align(),
        )));

        Self { tree, unit_size }
    }
}
