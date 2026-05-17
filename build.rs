use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use spirv_builder::{ModuleResult, SpirvBuilder};

struct ShaderTarget<'a> {
    source: &'a str,
    output: &'a str,
    entries: &'a [(&'a str, Option<&'a str>)],
}

struct RustShaderTarget<'a> {
    crate_path: &'a str,
    output: &'a str,
}

const SHADERS: &[ShaderTarget<'_>] = &[
    ShaderTarget {
        source: "assets/software_voxel/software_voxel_mesh.slang",
        output: "assets/software_voxel/software_voxel_mesh.spv",
        entries: &[("meshMain", Some("compute"))],
    },
    ShaderTarget {
        source: "assets/software_voxel/software_voxel_mesh_render.slang",
        output: "assets/software_voxel/software_voxel_mesh_render.spv",
        entries: &[
            ("vertexMain", None),
            ("fragmentMain", None),
            ("depthVertexMain", None),
        ],
    },
    ShaderTarget {
        source: "assets/software_voxel/software_voxel_post.slang",
        output: "assets/software_voxel/software_voxel_post.spv",
        entries: &[("vertexMain", None), ("fragmentMain", None)],
    },
];

const RUST_SHADERS: &[RustShaderTarget<'_>] = &[RustShaderTarget {
    crate_path: "crates/shaders",
    output: "assets/software_voxel/software_voxel_depth_pyramid.spv",
}];

fn main() {
    println!("cargo:rerun-if-env-changed=SLANGC");
    for shader in SHADERS {
        println!("cargo:rerun-if-changed={}", shader.source);
    }
    for shader in RUST_SHADERS {
        println!("cargo:rerun-if-changed={}/Cargo.toml", shader.crate_path);
        println!("cargo:rerun-if-changed={}/src", shader.crate_path);
    }
    println!("cargo:rerun-if-changed=../rust-gpu/crates/dst_heap/src");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_owned());

    for shader in SHADERS {
        compile_shader(&slangc, &manifest_dir, shader);
    }
    for shader in RUST_SHADERS {
        compile_rust_shader(&manifest_dir, shader);
    }
}

fn compile_shader(slangc: &str, manifest_dir: &Path, shader: &ShaderTarget<'_>) {
    let source = manifest_dir.join(shader.source);
    let output = manifest_dir.join(shader.output);
    let mut command = Command::new(slangc);
    command
        .arg(&source)
        .args(["-target", "spirv", "-profile", "sm_6_6"]);

    for (entry, stage) in shader.entries {
        command.args(["-entry", entry]);
        if let Some(stage) = stage {
            command.args(["-stage", stage]);
        }
    }

    let output_result = command.arg("-o").arg(&output).output();
    let output_result = match output_result {
        Ok(output_result) => output_result,
        Err(error) => {
            panic!(
                "failed to run `{}` while compiling {}: {}",
                slangc, shader.source, error
            );
        }
    };

    if !output_result.status.success() {
        panic!(
            "failed to compile shader {}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            shader.source,
            output_result.status,
            String::from_utf8_lossy(&output_result.stdout),
            String::from_utf8_lossy(&output_result.stderr)
        );
    }
}

fn compile_rust_shader(manifest_dir: &Path, shader: &RustShaderTarget<'_>) {
    let source = manifest_dir.join(shader.crate_path);
    let output = manifest_dir.join(shader.output);

    let mut builder = SpirvBuilder::new(&source, "spirv-unknown-vulkan1.3");
    builder.build_script.defaults = true;
    builder.target_dir_path = Some(PathBuf::from("rust-gpu-shaders"));

    let compile_result = builder.build().unwrap_or_else(|error| {
        panic!(
            "failed to compile rust-gpu shader {}: {}",
            shader.crate_path, error
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
