# Dust

A Rust-based voxel application built on [Bevy](https://bevyengine.org/) and Vulkan through [pumicite](https://github.com/dust-engine/pumicite).

## Features

- Hierarchical voxel spatial index (VDB-style) with const-generic configurable tree shape
- MagicaVoxel `.vox` file loading
- Voxel geometry, material, and palette asset types
- Fly camera demo scene with a castle, teapot, and animated rainbow data

The previous DLSS denoiser, ray-tracing renderer, BLAS/TLAS builders, tone-mapping pass, Bazel build files, shader pipeline assets, and repo-owned C/C++ helper sources have been removed.

## Requirements

- Rust nightly
- Vulkan SDK/runtime required by `pumicite`

## Building

```sh
cargo +nightly check
cargo +nightly run
```

Tests for the `dust_vdb` crate run under Cargo:

```sh
cargo +nightly test -p dust_vdb
```

## Workspace Layout

| Crate | Purpose |
| --- | --- |
| `crates/vdb` (`dust_vdb`) | Hierarchical voxel spatial index. Generic tree with configurable depth via the `hierarchy!` macro. Bit-packed node storage and pooling. |
| `crates/vox` (`dust_vox`) | Voxel geometry, materials, palettes, and `.vox` loading. Defines the tree as `hierarchy!(3, 3, 2, VoxLeafNode)`. |
| `crates/pbr` (`dust_pbr`) | Camera component and placeholder renderer plugin. |
| `crates/app` (`dust_app`) | Application-level Bevy plugin glue. |
| `src/main.rs` | Demo entry point. |
