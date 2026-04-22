using Pkg.Artifacts
using ArtifactUtils

# This is the path to the Artifacts.toml we will manipulate
artifact_toml = joinpath(@__DIR__, "Artifacts.toml")
add_artifact!(
    "Artifacts.toml",
    "kernels",
    "https://github.com/federicacapellino/2d_hic_simulation/releases/download/v1.0.0/fastreso_Fj_kernels.tar.gz",
    force = true,
)

import Pkg; Pkg.ensure_artifact_installed("kernels", "Artifacts.toml")

kernels = artifact"kernels"
