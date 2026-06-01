use bevy::prelude::*;
use pumicite::{
    Allocator,
    ash::{VkResult, vk},
    bindless::{BufferAccessMode, ResourceHeap},
    buffer::{BufferLike, ManagedBuffer},
    command::CommandEncoder,
    utils::AsVkHandle,
};

pub const MODEL_SIZE: u32 = 256;
pub const BLOCK_SIZE: u32 = 4;
pub const BLOCK_AXIS: u32 = MODEL_SIZE / BLOCK_SIZE;
pub const VOXELS_PER_BLOCK: usize = (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) as usize;
pub const MATERIAL_PAGE_SIZE: usize = VOXELS_PER_BLOCK;
pub const INITIAL_MATERIAL_PAGE_CAPACITY: usize = 16;

pub const EMPTY_MATERIAL_REF: u32 = u32::MAX;
pub const MATERIAL_PAGE_FLAG: u32 = 0x8000_0000;
pub const MATERIAL_PAGE_MASK: u32 = !MATERIAL_PAGE_FLAG;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DenseMaterialParam {
    pub base_color: [f32; 4],
    pub pbr: [f32; 4],
}

impl Default for DenseMaterialParam {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            pbr: [0.0, 0.6, 0.5, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DenseVoxelGpu {
    pub size: [u32; 4],
    pub occupancy_handle: u32,
    pub material_refs_handle: u32,
    pub material_pages_handle: u32,
    pub material_params_handle: u32,
}

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

    pub fn replace(&mut self, buffer: impl BufferLike) -> VkResult<()> {
        self.heap
            .update_buffer(self.handle, buffer, BufferAccessMode::Storage)
    }
}

impl Drop for BindlessBufferHandle {
    fn drop(&mut self) {
        self.heap.remove(self.handle);
    }
}

#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct DenseVoxelModel {
    pub size: [u32; 3],
    pub occupancy: Handle<DenseVoxelGeometry>,
    pub material: Handle<DenseVoxelMaterial>,
    pub cull_min: UVec3,
    pub cull_max: UVec3,
}

impl DenseVoxelModel {
    pub fn new(
        occupancy: Handle<DenseVoxelGeometry>,
        material: Handle<DenseVoxelMaterial>,
        size: [u32; 3],
    ) -> Self {
        Self {
            size,
            occupancy,
            material,
            cull_min: UVec3::ZERO,
            cull_max: UVec3::new(size[0], size[1], size[2]),
        }
    }

    pub fn gpu_descriptor(
        geometry: &DenseVoxelGeometry,
        material: &DenseVoxelMaterial,
    ) -> Option<DenseVoxelGpu> {
        assert_eq!(geometry.size, material.size);
        Some(DenseVoxelGpu {
            size: [
                geometry.size[0],
                geometry.size[1],
                geometry.size[2],
                material.material_page_count as u32,
            ],
            occupancy_handle: geometry.occupancy_bindless_handle()?,
            material_refs_handle: material.material_refs_bindless_handle()?,
            material_pages_handle: material.material_pages_bindless_handle()?,
            material_params_handle: material.material_params_bindless_handle()?,
        })
    }

    pub fn set_voxel(
        geometry: &mut DenseVoxelGeometry,
        material: &mut DenseVoxelMaterial,
        coords: [u32; 3],
        material_id: u8,
    ) {
        material.set_voxel(geometry, coords, material_id);
    }

    pub fn clear_voxel(
        geometry: &mut DenseVoxelGeometry,
        material: &mut DenseVoxelMaterial,
        coords: [u32; 3],
    ) {
        geometry.clear_voxel(coords);
        material.clear_voxel(geometry, coords);
    }
}

#[derive(Asset, TypePath)]
pub struct DenseVoxelGeometry {
    size: [u32; 3],
    occupancy: ManagedBuffer,
    occupancy_handle: Option<BindlessBufferHandle>,
}

impl DenseVoxelGeometry {
    pub fn new(allocator: Allocator) -> VkResult<Self> {
        Self::with_size(allocator, [MODEL_SIZE, MODEL_SIZE, MODEL_SIZE])
    }

    pub fn with_size(allocator: Allocator, size: [u32; 3]) -> VkResult<Self> {
        assert!(size.iter().all(|axis| *axis > 0 && *axis <= MODEL_SIZE));

        let mut occupancy = ManagedBuffer::new(
            allocator.clone(),
            occupancy_bytes(size) as u64,
            std::mem::align_of::<u64>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        occupancy.as_slice_mut().fill(0);

        Ok(Self {
            size,
            occupancy,
            occupancy_handle: None,
        })
    }

    pub fn size(&self) -> [u32; 3] {
        self.size
    }

    pub fn occupancy(&self) -> &[u64] {
        bytemuck::cast_slice(self.occupancy.as_slice())
    }

    pub fn occupancy_mut(&mut self) -> &mut [u64] {
        bytemuck::cast_slice_mut(self.occupancy.as_slice_mut())
    }

    pub fn occupancy_buffer(&self) -> &ManagedBuffer {
        &self.occupancy
    }

    pub fn register_bindless(&mut self, heap: &ResourceHeap) -> VkResult<()> {
        if self.occupancy_handle.is_none() {
            self.occupancy_handle = Some(BindlessBufferHandle::new(
                heap,
                BufferDescriptor::new(&self.occupancy),
            )?);
        }
        Ok(())
    }

    pub fn occupancy_bindless_handle(&self) -> Option<u32> {
        self.occupancy_handle
            .as_ref()
            .map(BindlessBufferHandle::get)
    }

    pub fn flush(&self, encoder: &mut CommandEncoder<'_>) {
        self.occupancy.flush(encoder);
    }

    pub fn set_occupied(&mut self, coords: [u32; 3]) {
        let index = voxel_index(coords, self.size);
        let bit = 1u64 << index.local_index;
        self.occupancy_mut()[index.block_index] |= bit;
    }

    pub fn clear_voxel(&mut self, coords: [u32; 3]) {
        let index = voxel_index(coords, self.size);
        let bit = 1u64 << index.local_index;
        self.occupancy_mut()[index.block_index] &= !bit;
    }

    pub fn is_occupied(&self, coords: [u32; 3]) -> bool {
        let index = voxel_index(coords, self.size);
        ((self.occupancy()[index.block_index] >> index.local_index) & 1) != 0
    }

    pub fn block_occupancy(&self, block_index: usize) -> u64 {
        self.occupancy()[block_index]
    }
}

#[derive(Asset, TypePath)]
pub struct DenseVoxelMaterial {
    size: [u32; 3],
    material_refs: ManagedBuffer,
    material_refs_handle: Option<BindlessBufferHandle>,
    material_pages: ManagedBuffer,
    material_pages_handle: Option<BindlessBufferHandle>,
    material_params: ManagedBuffer,
    material_params_handle: Option<BindlessBufferHandle>,
    material_page_count: usize,
}

impl DenseVoxelMaterial {
    pub fn new(allocator: Allocator) -> VkResult<Self> {
        Self::with_size(allocator, [MODEL_SIZE, MODEL_SIZE, MODEL_SIZE])
    }

    pub fn with_size(allocator: Allocator, size: [u32; 3]) -> VkResult<Self> {
        assert!(size.iter().all(|axis| *axis > 0 && *axis <= MODEL_SIZE));

        let mut material_refs = ManagedBuffer::new(
            allocator.clone(),
            material_refs_bytes(size) as u64,
            std::mem::align_of::<u32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        bytemuck::cast_slice_mut::<u8, u32>(material_refs.as_slice_mut()).fill(EMPTY_MATERIAL_REF);

        let mut material_pages = ManagedBuffer::new(
            allocator.clone(),
            material_pages_bytes(INITIAL_MATERIAL_PAGE_CAPACITY) as u64,
            std::mem::align_of::<u8>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        material_pages.as_slice_mut().fill(0);

        let entries = [DenseMaterialParam::default(); 256];
        let mut material_params = ManagedBuffer::new(
            allocator,
            std::mem::size_of_val(&entries) as u64,
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        material_params
            .as_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(&entries));

        Ok(Self {
            size,
            material_refs,
            material_refs_handle: None,
            material_pages,
            material_pages_handle: None,
            material_params,
            material_params_handle: None,
            material_page_count: 0,
        })
    }

    pub fn size(&self) -> [u32; 3] {
        self.size
    }

    pub fn material_refs_bindless_handle(&self) -> Option<u32> {
        self.material_refs_handle
            .as_ref()
            .map(BindlessBufferHandle::get)
    }

    pub fn material_pages_bindless_handle(&self) -> Option<u32> {
        self.material_pages_handle
            .as_ref()
            .map(BindlessBufferHandle::get)
    }

    pub fn material_params_bindless_handle(&self) -> Option<u32> {
        self.material_params_handle
            .as_ref()
            .map(BindlessBufferHandle::get)
    }

    pub fn material_refs(&self) -> &[u32] {
        bytemuck::cast_slice(self.material_refs.as_slice())
    }

    pub fn material_refs_mut(&mut self) -> &mut [u32] {
        bytemuck::cast_slice_mut(self.material_refs.as_slice_mut())
    }

    pub fn material_pages(&self) -> &[[u8; VOXELS_PER_BLOCK]] {
        bytemuck::cast_slice(&self.material_pages.as_slice()[..self.material_pages_len_bytes()])
    }

    pub fn material_pages_bytes(&self) -> &[u8] {
        &self.material_pages.as_slice()[..self.material_pages_len_bytes()]
    }

    pub fn material_page_count(&self) -> usize {
        self.material_page_count
    }

    pub fn material_refs_buffer(&self) -> &ManagedBuffer {
        &self.material_refs
    }

    pub fn material_pages_buffer(&self) -> &ManagedBuffer {
        &self.material_pages
    }

    pub fn material_params_buffer(&self) -> &ManagedBuffer {
        &self.material_params
    }

    pub fn set_material_params(&mut self, entries: &[DenseMaterialParam; 256]) {
        self.material_params
            .as_slice_mut()
            .copy_from_slice(bytemuck::cast_slice(entries));
    }

    pub fn register_bindless(&mut self, heap: &ResourceHeap) -> VkResult<()> {
        if self.material_refs_handle.is_none() {
            self.material_refs_handle = Some(BindlessBufferHandle::new(
                heap,
                BufferDescriptor::new(&self.material_refs),
            )?);
        }
        if self.material_pages_handle.is_none() {
            self.material_pages_handle = Some(BindlessBufferHandle::new(
                heap,
                BufferDescriptor::new(&self.material_pages),
            )?);
        }
        if self.material_params_handle.is_none() {
            self.material_params_handle = Some(BindlessBufferHandle::new(
                heap,
                BufferDescriptor::new(&self.material_params),
            )?);
        }
        Ok(())
    }

    pub fn flush(&self, encoder: &mut CommandEncoder<'_>) {
        self.material_refs.flush(encoder);
        self.material_pages.flush(encoder);
        self.material_params.flush(encoder);
    }

    pub fn set_voxel(&mut self, geometry: &mut DenseVoxelGeometry, coords: [u32; 3], material: u8) {
        assert_eq!(self.size, geometry.size);
        let index = voxel_index(coords, self.size);
        geometry.set_occupied(coords);

        let material_ref = self.material_refs()[index.block_index];
        if material_ref == EMPTY_MATERIAL_REF {
            self.material_refs_mut()[index.block_index] = material as u32;
            return;
        }

        if is_uniform_material_ref(material_ref) {
            if material_ref as u8 == material {
                return;
            }

            let mut page = [material_ref as u8; VOXELS_PER_BLOCK];
            page[index.local_index as usize] = material;
            let page_index = self.push_material_page(page);
            self.material_refs_mut()[index.block_index] = MATERIAL_PAGE_FLAG | page_index as u32;
            return;
        }

        let page_index = material_page_index(material_ref);
        self.material_page_mut(page_index)[index.local_index as usize] = material;
    }

    pub fn clear_voxel(&mut self, geometry: &DenseVoxelGeometry, coords: [u32; 3]) {
        assert_eq!(self.size, geometry.size);
        let index = voxel_index(coords, self.size);
        if geometry.block_occupancy(index.block_index) == 0 {
            self.material_refs_mut()[index.block_index] = EMPTY_MATERIAL_REF;
        }
    }

    pub fn material(&self, geometry: &DenseVoxelGeometry, coords: [u32; 3]) -> Option<u8> {
        assert_eq!(self.size, geometry.size);
        let index = voxel_index(coords, self.size);
        if ((geometry.occupancy()[index.block_index] >> index.local_index) & 1) == 0 {
            return None;
        }

        let material_ref = self.material_refs()[index.block_index];
        if is_uniform_material_ref(material_ref) {
            return Some(material_ref as u8);
        }

        Some(self.material_pages()[material_page_index(material_ref)][index.local_index as usize])
    }

    pub fn rebuild_material_pages(&mut self, geometry: &DenseVoxelGeometry) {
        assert_eq!(self.size, geometry.size);
        let old_pages = self.material_pages().to_vec();
        let mut new_pages = Vec::new();
        let mut refs = self.material_refs().to_vec();

        for (block_index, material_ref) in refs.iter_mut().enumerate() {
            let occupancy = geometry.occupancy()[block_index];
            if occupancy == 0 {
                *material_ref = EMPTY_MATERIAL_REF;
                continue;
            }

            if is_uniform_material_ref(*material_ref) {
                continue;
            }

            let old_page = old_pages[material_page_index(*material_ref)];
            if let Some(material) = uniform_material_for_occupied_voxels(occupancy, &old_page) {
                *material_ref = material as u32;
                continue;
            }

            let new_page_index = new_pages.len();
            assert!(new_page_index <= MATERIAL_PAGE_MASK as usize);
            new_pages.push(old_page);
            *material_ref = MATERIAL_PAGE_FLAG | new_page_index as u32;
        }

        self.replace_material_pages(&new_pages);
        self.material_refs_mut().copy_from_slice(&refs);
    }

    fn material_pages_capacity(&self) -> usize {
        self.material_pages.size() as usize / MATERIAL_PAGE_SIZE
    }

    fn material_pages_len_bytes(&self) -> usize {
        material_pages_bytes(self.material_page_count)
    }

    fn material_page_mut(&mut self, page_index: usize) -> &mut [u8; VOXELS_PER_BLOCK] {
        assert!(page_index < self.material_page_count);
        let start = page_index * MATERIAL_PAGE_SIZE;
        let end = start + MATERIAL_PAGE_SIZE;
        bytemuck::from_bytes_mut(&mut self.material_pages.as_slice_mut()[start..end])
    }

    fn push_material_page(&mut self, page: [u8; VOXELS_PER_BLOCK]) -> usize {
        let page_index = self.material_page_count;
        assert!(page_index <= MATERIAL_PAGE_MASK as usize);
        self.reserve_material_pages(page_index + 1);

        let start = page_index * MATERIAL_PAGE_SIZE;
        let end = start + MATERIAL_PAGE_SIZE;
        self.material_pages.as_slice_mut()[start..end].copy_from_slice(&page);
        self.material_page_count += 1;
        page_index
    }

    fn replace_material_pages(&mut self, pages: &[[u8; VOXELS_PER_BLOCK]]) {
        self.reserve_material_pages(pages.len());
        let len = material_pages_bytes(pages.len());
        self.material_pages.as_slice_mut()[..len].copy_from_slice(bytemuck::cast_slice(pages));
        self.material_page_count = pages.len();
    }

    fn reserve_material_pages(&mut self, required: usize) {
        if required <= self.material_pages_capacity() {
            return;
        }

        let new_capacity = required
            .next_power_of_two()
            .max(INITIAL_MATERIAL_PAGE_CAPACITY);
        let mut new_buffer = ManagedBuffer::new(
            self.material_pages.allocator().clone(),
            material_pages_bytes(new_capacity) as u64,
            std::mem::align_of::<u8>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();

        let used_bytes = self.material_pages_len_bytes();
        new_buffer.as_slice_mut()[..used_bytes]
            .copy_from_slice(&self.material_pages.as_slice()[..used_bytes]);
        self.material_pages = new_buffer;

        if let Some(heap) = self
            .material_pages_handle
            .as_ref()
            .map(|bindless_handle| bindless_handle.heap.clone())
        {
            self.material_pages_handle = Some(
                BindlessBufferHandle::new(&heap, BufferDescriptor::new(&self.material_pages))
                    .unwrap(),
            );
        }
    }
}

fn block_count(size: [u32; 3]) -> usize {
    let extent = block_extent(size);
    (extent[0] * extent[1] * extent[2]) as usize
}

fn block_extent(size: [u32; 3]) -> [u32; 3] {
    [
        size[0].div_ceil(BLOCK_SIZE),
        size[1].div_ceil(BLOCK_SIZE),
        size[2].div_ceil(BLOCK_SIZE),
    ]
}

fn occupancy_bytes(size: [u32; 3]) -> usize {
    block_count(size) * std::mem::size_of::<u64>()
}

fn material_refs_bytes(size: [u32; 3]) -> usize {
    block_count(size) * std::mem::size_of::<u32>()
}

fn material_pages_bytes(page_count: usize) -> usize {
    page_count * MATERIAL_PAGE_SIZE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VoxelIndex {
    block_index: usize,
    local_index: u32,
}

fn voxel_index(coords: [u32; 3], size: [u32; 3]) -> VoxelIndex {
    assert!(coords[0] < size[0]);
    assert!(coords[1] < size[1]);
    assert!(coords[2] < size[2]);

    let block = [
        coords[0] / BLOCK_SIZE,
        coords[1] / BLOCK_SIZE,
        coords[2] / BLOCK_SIZE,
    ];
    let local = [
        coords[0] & (BLOCK_SIZE - 1),
        coords[1] & (BLOCK_SIZE - 1),
        coords[2] & (BLOCK_SIZE - 1),
    ];

    VoxelIndex {
        block_index: block_index(block, size),
        local_index: local_index(local),
    }
}

pub fn block_index(block: [u32; 3], size: [u32; 3]) -> usize {
    let extent = block_extent(size);
    assert!(
        block
            .iter()
            .zip(extent)
            .all(|(axis, extent)| *axis < extent)
    );
    (block[0] + block[1] * extent[0] + block[2] * extent[0] * extent[1]) as usize
}

pub fn local_index(local: [u32; 3]) -> u32 {
    assert!(local.iter().all(|axis| *axis < BLOCK_SIZE));
    local[0] | (local[1] << 2) | (local[2] << 4)
}

pub fn is_page_material_ref(material_ref: u32) -> bool {
    material_ref != EMPTY_MATERIAL_REF && (material_ref & MATERIAL_PAGE_FLAG) != 0
}

pub fn is_uniform_material_ref(material_ref: u32) -> bool {
    material_ref != EMPTY_MATERIAL_REF && (material_ref & MATERIAL_PAGE_FLAG) == 0
}

pub fn material_page_index(material_ref: u32) -> usize {
    assert!(is_page_material_ref(material_ref));
    (material_ref & MATERIAL_PAGE_MASK) as usize
}

fn uniform_material_for_occupied_voxels(
    occupancy: u64,
    page: &[u8; VOXELS_PER_BLOCK],
) -> Option<u8> {
    let first = occupancy.trailing_zeros();
    if first >= VOXELS_PER_BLOCK as u32 {
        return None;
    }

    let material = page[first as usize];
    let mut remaining = occupancy & !(1u64 << first);
    while remaining != 0 {
        let bit = remaining.trailing_zeros();
        if page[bit as usize] != material {
            return None;
        }
        remaining &= remaining - 1;
    }
    Some(material)
}
