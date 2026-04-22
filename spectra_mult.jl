using Plots
using Fluidum
using MonteCarloGlauber
using StaticArrays
using LinearAlgebra
using Integrals
using Cuba
using JLD2
using HDF5
using MuladdMacro
using OhMyThreads
using Base.Threads
using YAML
using Statistics
include("MCglauber.jl")
include("hdf5_io.jl")
include("observables.jl")
include("fastreso.jl")


default(linewidth = 2,
    markersize = 8,
    grid = false,size = (700, 450),
    guidefontsize = 12,
    tickfontsize = 10,
    dpi = 150)



#data1 = hdf5_to_ObservableResult_old(pwd()*"/event_by_event_results_debug.h5")
#data2 = hdf5_to_ObservableResult(pwd()*"/event_by_event_results_debug_2.h5")
#data3 = hdf5_to_ObservableResult(pwd()*"/event_by_event_results_debug_b.h5")
data4 = hdf5_to_ObservableResult(pwd()*"/event_by_event_D0_debug.h5")
d = extract_vn(data4)
sum(d[3,3,:,1,4])

data=data4

#data = vcat(data1,data2,data3)

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

centrality_bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
data_chunks = centralities_selection_events(data,centrality_bins)

pt_list_pi = particle_full_π.pt_list

   

function plot_multiplicity_vs_centrality(data_chunks, centrality_bins, species_list::Vector)
    
    multiplicities = map(Iterators.product(eachindex(centrality_bins),eachindex(species_list))) do I
    cc, k = I
    mean([sum(extract_vn(data_chunks[cc])[i,3,:,1,k]) for i in eachindex(data_chunks[cc])])
    end
    labels = ["$(species_list[k].name)" for k in eachindex(species_list)]
    pl = scatter(centrality_bins, multiplicities[:,1], xlabel="Centrality (%)", ylabel="Multiplicity", label=labels[1])
    if length(species_list) > 1
    for i in 2:length(species_list)
        scatter!(centrality_bins, multiplicities[:,i], label=labels[i])
    end
    end
    display(pl)
end

function plot_multiplicity_vs_centrality(data_chunks, centrality_bins, idx_species::Int)
    
    multiplicities = map(eachindex(centrality_bins)) do cc  
    mean([sum(extract_vn(data_chunks[cc])[i,3,:,1,idx_species]) for i in eachindex(data_chunks[cc])])
    end
    pl = scatter(centrality_bins, multiplicities[:], xlabel="Centrality (%)", ylabel="Multiplicity")
    display(pl)
end

plot_multiplicity_vs_centrality(data_chunks, centrality_bins[1:5], species_list)
plot_multiplicity_vs_centrality(data_chunks, centrality_bins[1:5], 4)
# 2. Define a color palette (10 distinct colors)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]

# 3. Loop through your chunks
for i in 1:10
    vni = spectra(data_chunks[i],species_list)
    cc = i*10
    plot(pt_list_pi, vni[:, 1], color=colors[i], label="$cc %")
end
display(pl)
pl = plot(xlabel="cc", ylabel="multiplicity", legend=:topright)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]
for i in 1:1
    # Extract the specific slice
    vni = extract_vn(data_chunks[i])
    mult = [sum(vni[i,3,:,1,4]) for i in axes(vni,1)]
    cc = i*10
    # Plot all lines for this chunk in the same color, no labels
    plot!(range(cc-10,cc,length(mult)),mult,color=colors[i],yscale=:log10)
end
display(pl)

pl = plot(xlabel="cc", ylabel="multiplicity", legend=:topright)

# 2. Define a color palette (10 distinct colors)
colors = palette(:tab10); # or [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :black, :gray]

# 3. Loop through your chunks
for i in 1:10
    # Extract the specific slice
    vni = extract_vn(data_chunks[i])
    mult = [sum(vni[i,3,:,1,4]) for i in axes(vni,1)]
    cc = i*10
    # Plot all lines for this chunk in the same color, no labels
    plot(range(cc-10,cc,length(mult)),mult,color=colors[i],label="$cc %")
end
display(pl)
