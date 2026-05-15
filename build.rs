use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

struct ShaderTarget<'a> {
    source: &'a str,
    output: &'a str,
    entries: &'a [(&'a str, Option<&'a str>)],
}

const SHADERS: &[ShaderTarget<'_>] = &[
    ShaderTarget {
        source: "assets/software_voxel/software_voxel_mesh.slang",
        output: "assets/software_voxel/software_voxel_mesh.spv",
        entries: &[("meshMain", Some("compute"))],
    },
    ShaderTarget {
        source: "assets/software_voxel/software_voxel_depth_pyramid.slang",
        output: "assets/software_voxel/software_voxel_depth_pyramid.spv",
        entries: &[("depthPyramidMain", Some("compute"))],
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

fn main() {
    println!("cargo:rerun-if-env-changed=SLANGC");
    for shader in SHADERS {
        println!("cargo:rerun-if-changed={}", shader.source);
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let slangc = env::var("SLANGC").unwrap_or_else(|_| "slangc".to_owned());

    for shader in SHADERS {
        compile_shader(&slangc, &manifest_dir, shader);
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
