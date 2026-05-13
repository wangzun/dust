## Project Overview

Dust is a Rust-based voxel application built on Bevy and Vulkan through pumicite.

## Build System

Cargo is the build system.

- **Check:** `cargo +nightly check`
- **Run:** `cargo +nightly run`
- **Run tests:** `cargo +nightly test`

## Architecture

### Workspace Crates

- **`crates/vdb/`** (`dust_vdb`) — Hierarchical voxel spatial index. A generic tree structure with configurable depth using const generics. Handles node pooling, bit-packed storage, and tree traversal.

- **`crates/vox/`** (`dust_vox`) — Voxel geometry, materials, palettes, and MagicaVoxel `.vox` file loading. Defines the voxel tree hierarchy as `hierarchy!(3, 3, 2, VoxLeafNode)`.

- **`crates/pbr/`** (`dust_pbr`) — Camera component and placeholder renderer plugin after removal of the ray-tracing renderer.

- **`crates/app/`** (`dust_app`) — Application-level Bevy plugin glue.

- **`src/main.rs`** — Demo entry point.

### Removed Systems

The DLSS denoiser, ray-tracing renderer, BLAS/TLAS builders, tone-mapping pass, Bazel build files, shader pipeline assets, and repo-owned C/C++ helper sources have been removed.

### Key Dependencies

- **Bevy 0.17.0-dev** — Patched from the `dust-engine/bevy` fork (`release-0.17.3` branch).
- **pumicite / bevy_pumicite** — Vulkan abstraction layer and Bevy integration.
- **ash** — Vulkan FFI custom fork.
