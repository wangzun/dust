use std::{
    env, fs,
    path::{Path, PathBuf},
};

use spirv_builder::{Capability, ModuleResult, SpirvBuilder};

struct RustShaderTarget<'a> {
    crate_path: &'a str,
    feature: &'a str,
    output: &'a str,
    capabilities: &'a [Capability],
}

const RUST_SHADERS: &[RustShaderTarget<'_>] = &[
    RustShaderTarget {
        crate_path: "crates/shaders",
        feature: "depth-pyramid",
        output: "assets/software_voxel/software_voxel_depth_pyramid.spv",
        capabilities: &[],
    },
    RustShaderTarget {
        crate_path: "crates/shaders",
        feature: "mesh-render",
        output: "assets/software_voxel/software_voxel_mesh_render.spv",
        capabilities: &[
            Capability::DrawParameters,
            Capability::Int8,
            Capability::Int64,
        ],
    },
    RustShaderTarget {
        crate_path: "crates/shaders",
        feature: "mesh",
        output: "assets/software_voxel/software_voxel_mesh.spv",
        capabilities: &[
            Capability::Int8,
            Capability::Int64,
            Capability::VulkanMemoryModelDeviceScope,
        ],
    },
    RustShaderTarget {
        crate_path: "crates/shaders",
        feature: "post",
        output: "assets/software_voxel/software_voxel_post.spv",
        capabilities: &[Capability::DrawParameters],
    },
];

fn main() {
    for shader in RUST_SHADERS {
        println!("cargo:rerun-if-changed={}/Cargo.toml", shader.crate_path);
        println!("cargo:rerun-if-changed={}/src", shader.crate_path);
    }
    println!("cargo:rerun-if-changed=../rust-gpu/crates/dst_heap/src");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for shader in RUST_SHADERS {
        compile_rust_shader(&manifest_dir, shader);
    }
}

fn compile_rust_shader(manifest_dir: &Path, shader: &RustShaderTarget<'_>) {
    let source = manifest_dir.join(shader.crate_path);

    let mut builder = SpirvBuilder::new(&source, "spirv-unknown-vulkan1.3");
    builder.build_script.defaults = true;
    builder.capabilities.extend_from_slice(shader.capabilities);
    builder.shader_crate_features.default_features = false;
    builder.shader_crate_features.features = vec![shader.feature.to_owned()];
    builder.target_dir_path = Some(PathBuf::from("rust-gpu-shaders"));

    let compile_result = builder.build().unwrap_or_else(|error| {
        panic!(
            "failed to compile rust-gpu shader {} with feature {}: {}",
            shader.crate_path, shader.feature, error
        )
    });

    let spv_path = match compile_result.module {
        ModuleResult::SingleModule(path) => path,
        ModuleResult::MultiModule(modules) => {
            if modules.len() != 1 {
                panic!(
                    "expected one rust-gpu shader module for {}, got {}",
                    shader.crate_path,
                    modules.len()
                );
            }
            modules
                .into_iter()
                .next()
                .map(|(_, path)| path)
                .expect("module count was checked")
        }
    };

    let output = manifest_dir.join(shader.output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create shader output directory {}: {}",
                parent.display(),
                error
            )
        });
    }

    fs::copy(&spv_path, &output).unwrap_or_else(|error| {
        panic!(
            "failed to copy rust-gpu shader {} to {}: {}",
            spv_path.display(),
            output.display(),
            error
        )
    });
}
