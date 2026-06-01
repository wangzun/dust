use std::{alloc::Layout, mem::MaybeUninit};

use glam::UVec3;

use crate::{AabbU32, Node, NodeMeta, pool::Pool, pool::PoolStorage};

enum RootStorage<ROOT: Node> {
    Inline(ROOT),
    External {
        storage: Box<dyn PoolStorage>,
        ptr: *mut ROOT,
    },
}

unsafe impl<ROOT: Node> Send for RootStorage<ROOT> {}
unsafe impl<ROOT: Node> Sync for RootStorage<ROOT> {}

impl<ROOT: Node> RootStorage<ROOT> {
    fn inline() -> Self {
        Self::Inline(ROOT::default())
    }

    fn external(mut storage: Box<dyn PoolStorage>, layout: Layout) -> Self {
        let layout = layout.pad_to_align();
        let ptr = storage.resize(layout.size()) as *mut ROOT;
        assert!(!ptr.is_null());
        debug_assert_eq!((ptr as usize) % layout.align(), 0);
        unsafe {
            ptr.write(ROOT::default());
        }
        Self::External { storage, ptr }
    }

    #[inline]
    fn as_ref(&self) -> &ROOT {
        match self {
            Self::Inline(root) => root,
            Self::External { ptr, .. } => unsafe { &**ptr },
        }
    }

    #[inline]
    fn as_mut(&mut self) -> &mut ROOT {
        match self {
            Self::Inline(root) => root,
            Self::External { ptr, .. } => unsafe { &mut **ptr },
        }
    }

    #[inline]
    fn storage(&self) -> Option<&dyn PoolStorage> {
        match self {
            Self::Inline(_) => None,
            Self::External { storage, .. } => Some(&**storage),
        }
    }

    #[inline]
    fn storage_mut(&mut self) -> Option<&mut dyn PoolStorage> {
        match self {
            Self::Inline(_) => None,
            Self::External { storage, .. } => Some(&mut **storage),
        }
    }
}

impl<ROOT: Node> Drop for RootStorage<ROOT> {
    fn drop(&mut self) {
        if let Self::External { ptr, .. } = self {
            unsafe {
                ptr.drop_in_place();
            }
        }
    }
}

pub struct Tree<ROOT: Node>
where
    [(); ROOT::LEVEL as usize]: Sized,
{
    root: RootStorage<ROOT>,
    pub(crate) pool: [Pool; ROOT::LEVEL as usize],
    pub(crate) aabb: AabbU32,
}

/// ```
/// #![feature(generic_const_exprs)]
/// use dust_vdb::{hierarchy, Node, Tree};
/// use glam::UVec3;
/// let mut tree = Tree::<hierarchy!(2, 2)>::new();
/// tree.set_value(UVec3{x: 0, y: 4, z: 0}, Some(true));
/// tree.set_value(UVec3{x: 0, y: 2, z: 2}, Some(false));
/// assert_eq!(tree.get_value(UVec3::new(0, 4, 0)), Some(true));
/// assert_eq!(tree.get_value(UVec3::new(0, 3, 0)), None);
/// assert_eq!(tree.get_value(UVec3::new(0, 2, 2)), Some(false));
/// ```
impl<ROOT: Node> Tree<ROOT>
where
    [(); ROOT::LEVEL as usize]: Sized,
    [(); ROOT::LEVEL as usize + 1]: Sized,
{
    pub fn new() -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize];
        let metas = Self::metas();
        for (i, meta) in metas.iter().take(ROOT::LEVEL).enumerate() {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }

        let pools: [Pool; ROOT::LEVEL as usize] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: RootStorage::inline(),
            pool: pools,
            aabb: AabbU32::default(),
        }
    }
    pub fn new_with_leaf_storage(storage: Box<dyn PoolStorage>) -> Self
    where
        ROOT: Node,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize];
        let metas = Self::metas();
        for (i, meta) in metas.iter().take(ROOT::LEVEL).enumerate().skip(1) {
            // Create CPU pool for levels 1..LEVEL. 1024 internal nodes at each level
            let pool = Pool::new(meta.layout);
            pools[i].write(pool);
        }
        pools[0].write(Pool::new_with_storage(metas[0].layout, storage));

        let pools: [Pool; ROOT::LEVEL as usize] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root: RootStorage::inline(),
            pool: pools,
            aabb: AabbU32::default(),
        }
    }
    /// Create a tree whose root and all pooled node levels use caller-provided storage.
    ///
    /// The storage factory is called once for each node level:
    /// - level `0` is the leaf pool.
    /// - levels `1..ROOT::LEVEL` are internal node pools.
    /// - level `ROOT::LEVEL` is the root node storage.
    ///
    /// Unlike [`Tree::new_with_leaf_storage`], this stores the root node itself in the
    /// supplied storage. The root buffer contains exactly one `ROOT` value at offset 0.
    pub fn new_with_node_storage<F>(mut storage_for_level: F) -> Self
    where
        ROOT: Node,
        F: FnMut(usize, Layout) -> Box<dyn PoolStorage>,
    {
        let mut pools: [MaybeUninit<Pool>; ROOT::LEVEL as usize] =
            [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize];
        let metas = Self::metas();
        for (i, meta) in metas.iter().take(ROOT::LEVEL).enumerate() {
            let pool = Pool::new_with_storage(meta.layout, storage_for_level(i, meta.layout));
            pools[i].write(pool);
        }

        let root_meta = &metas[ROOT::LEVEL];
        let root = RootStorage::external(
            storage_for_level(ROOT::LEVEL, root_meta.layout),
            root_meta.layout,
        );
        let pools: [Pool; ROOT::LEVEL as usize] = unsafe { MaybeUninit::array_assume_init(pools) };
        Self {
            root,
            pool: pools,
            aabb: AabbU32::default(),
        }
    }
    /// Alias for [`Tree::new_with_node_storage`].
    pub fn new_with_all_node_storage<F>(storage_for_level: F) -> Self
    where
        ROOT: Node,
        F: FnMut(usize, Layout) -> Box<dyn PoolStorage>,
    {
        Self::new_with_node_storage(storage_for_level)
    }
    pub fn pools(&self) -> &[Pool] {
        &self.pool
    }
    pub fn root_storage(&self) -> Option<&dyn PoolStorage> {
        self.root.storage()
    }
    pub fn root_storage_mut(&mut self) -> Option<&mut dyn PoolStorage> {
        self.root.storage_mut()
    }
    pub fn root_device_address(&self) -> u64 {
        self.root.storage().map_or(0, PoolStorage::device_address)
    }
    pub(crate) fn root_and_pools(&self) -> (&ROOT, &[Pool]) {
        (self.root.as_ref(), &self.pool)
    }
    pub(crate) fn root_mut_and_pools_mut(&mut self) -> (&mut ROOT, &mut [Pool]) {
        (self.root.as_mut(), &mut self.pool)
    }
    pub unsafe fn alloc_node<CHILD: Node>(&mut self) -> u32 {
        unsafe {
            if ROOT::LEVEL <= CHILD::LEVEL {
                panic!("Can not allocate root node");
            }
            let pool = &mut self.pool[CHILD::LEVEL as usize];
            pool.alloc::<CHILD>()
        }
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node<CHILD: Node>(&self, ptr: u32) -> &CHILD {
        unsafe {
            if CHILD::LEVEL == ROOT::LEVEL {
                // specialization for root
                return &*(self.root.as_ref() as *const ROOT as *const CHILD);
            }
            &*(self.pool[CHILD::LEVEL as usize].get(ptr) as *const CHILD)
        }
    }

    /// Safety: ptr must point to a valid region of memory in the pool of CHILD.
    #[inline]
    pub unsafe fn get_node_mut<CHILD: Node>(&mut self, ptr: u32) -> &mut CHILD {
        unsafe {
            if CHILD::LEVEL == ROOT::LEVEL {
                // specialization for root
                return &mut *(self.root.as_mut() as *mut ROOT as *mut CHILD);
            }
            &mut *(self.pool[CHILD::LEVEL as usize].get_mut(ptr) as *mut CHILD)
        }
    }

    /// ```
    /// #![feature(generic_const_exprs)]
    /// use dust_vdb::{Tree, hierarchy};
    /// use glam::UVec3;
    /// let mut tree = Tree::<hierarchy!(4, 2)>::new();
    /// tree.set_value(UVec3::new(0, 1, 2), Some(true));
    /// tree.set_value(UVec3::new(63, 1, 3), Some(true));
    /// tree.set_value(UVec3::new(63, 63, 63), Some(true));
    /// let mut iter = tree.iter();
    /// assert_eq!(iter.next().unwrap(), UVec3::new(0, 1, 2));
    /// assert_eq!(iter.next().unwrap(), UVec3::new(63, 1, 3));
    /// assert_eq!(iter.next().unwrap(), UVec3::new(63, 63, 63));
    /// assert!(iter.next().is_none());
    ///
    /// ```
    pub fn iter<'a>(&'a self) -> ROOT::Iterator<'a> {
        self.root
            .as_ref()
            .iter(&self.pool, UVec3 { x: 0, y: 0, z: 0 })
    }

    pub fn iter_leaf<'a>(&'a self) -> impl Iterator<Item = (UVec3, &'a <ROOT as Node>::LeafType)> {
        self.root
            .as_ref()
            .iter_leaf(&self.pool, UVec3 { x: 0, y: 0, z: 0 })
            .map(|(position, leaf)| unsafe {
                let leaf: &'a ROOT::LeafType = &*leaf.get();
                (position, leaf)
            })
    }

    pub fn iter_leaf_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (UVec3, &'a mut ROOT::LeafType)> {
        self.root
            .as_ref()
            .iter_leaf(&mut self.pool, UVec3 { x: 0, y: 0, z: 0 })
            .map(|(position, leaf)| unsafe {
                let leaf: &'a mut ROOT::LeafType = &mut *leaf.get();
                (position, leaf)
            })
    }

    pub fn count_leaves(&self) -> usize {
        self.root.as_ref().count_leaves(&self.pool)
    }

    pub fn metas() -> [NodeMeta<ROOT::LeafType>; ROOT::LEVEL as usize + 1] {
        let mut arr = [const { MaybeUninit::uninit() }; ROOT::LEVEL as usize + 1];
        ROOT::write_meta(&mut arr);
        unsafe { MaybeUninit::array_assume_init(arr) }
    }
}
