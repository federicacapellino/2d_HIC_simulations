using MonteCarloGlauber
using Fluidum
using MuladdMacro

include("../utils/MCglauber.jl")
include("../utils/hdf5_io.jl")
include("../utils/observables.jl")
include("../utils/fastreso.jl")


kernels = Fluidum.root_kernels
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_211_total_T0.1560_Fj.out"))
const particle_full_π = particle_full("pion", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_2212_total_T0.1560_Fj.out"))
const particle_full_p = particle_full("proton", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_321_total_T0.1560_Fj.out"))
const particle_full_k = particle_full("kaon", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/Dc1865zer_total_T0.1560_Fj.out"))
const particle_full_D0 = particle_full("D0", Fj[4], 1, 1, Fj[1])

species_list = [particle_full_π, particle_full_p, particle_full_k, particle_full_D0]
pTlists = pt_list.(species_list)
delta_lists = increment.(pTlists)

@inbounds for i in eachindex(pTlists[1])
    delta = delta_lists[1][i]
    pT = (-delta + pTlists[1][i],delta + pTlists[1][i])
    println(pTlists[1][i])
    println("integration limits in pT ", pT)
end
