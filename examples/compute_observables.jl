using MonteCarloGlauber
using Fluidum
using MuladdMacro
using HDF5
using LaTeXStrings
using Plots
using Statistics


include("../utils/MCglauber.jl")
include("../utils/hdf5_io.jl")
include("../utils/observables.jl")
include("../utils/fastreso.jl")

include("../plots/plotting_macros.jl")

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

data = hdf5_to_ObservableResult(pwd()*"/results/event_by_event_results_debug_all_particle_charm.h5")

centrality_bins=[5,10,20,30,40,50,60]
data_chunks = centralities_selection_events(data,centrality_bins)

#spectra


charged_particle_spectra=spectra(data_chunks[1][1])
charged_particle_spectra_cc=spectra(data_chunks[1])
spectra_average_cc = spectra_average(data_chunks[1])
spectra_average_all_cc = spectra_average(data_chunks)

plot_identified_particle_spectra(data_chunks,species_list[1:3], centrality_bins)
plot_identified_particle_spectra_average(data_chunks,species_list[1:3], centrality_bins)


#multiplicity
mult = multiplicity(data_chunks[1][1])
mult_cc = multiplicity(data_chunks[1])
mult_all_cc = multiplicity_event(data_chunks)

total_mult = get_total_multiplicity(mult_cc)
pion_mult = get_identified_multiplicity(mult_cc, 1)
id_mult = get_identified_multiplicity(mult_cc, species_list)

total_mult_avg = total_multiplicity_average(total_mult)
id_mult_avg = identified_multiplicity_average(id_mult)

mult_cc = multiplicity(data_chunks[2])
id_mult = get_identified_multiplicity(mult_cc, species_list)


plot_total_multiplicity(data_chunks,centrality_bins)
plot_total_multiplicity_average(data_chunks,centrality_bins)
plot_identified_multiplicity_average(data_chunks,centrality_bins,species_list[1:3])


#Q vector

#q vector

#<2>

#vn
