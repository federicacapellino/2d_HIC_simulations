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
using LaTeXStrings
using Statistics
using DataFrames

default(lw = 2, size=(800,600),xtickfontsize=16,ytickfontsize=16,xlabelfontsize=16,ylabelfontsize=16,legendfontsize=16,grid=false,framestyle=:box)

include("MCglauber.jl")
include("hdf5_io.jl")
include("observables.jl")
include("fastreso.jl")

data1 = hdf5_to_ObservableResult(pwd()*"/event_by_event_results_debug_all_particle.h5")

data = vcat(data1)

glauber_vec = extract_glauber_multiplicity(data)
Nev= length(glauber_vec)
artifact_toml = joinpath(@__DIR__, "Artifacts.toml")
ensure_artifact_installed("kernels", "Artifacts.toml")
kernels = artifact"kernels"
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_211_total_T0.1560_Fj.out"))
const particle_full_π = particle_full("pion", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_2212_total_T0.1560_Fj.out"))
const particle_full_p = particle_full("proton", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_321_total_T0.1560_Fj.out"))
const particle_full_k = particle_full("kaon", Fj[4], 1, 1, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/Dc1865zer_total_T0.1560_Fj.out"))
const particle_full_D0 = particle_full("D0", Fj[4], 1, 1, Fj[1])

species_list = [particle_full_π, particle_full_p, particle_full_k, particle_full_D0]


centrality_bins=[5,10,20,30,40,50,60]
data_chunks = centralities_selection_events(data,centrality_bins)

pt_list_pi = particle_full_π.pt_list

q_vector_event_pt_dependent(data_chunks[1][1],species_list,[2,3])
#q_vector_event_pt_dependent_im(data_chunks[1][1],species_list,[2,3])
multiplicity_event(data_chunks[1][1],species_list)
multiplicity_event(data_chunks[1][2],species_list)

g = g_species_event_pt_dependent(data_chunks[2][190],species_list)
sum(g)
q_vector_event_integrated_complex(data_chunks[2][1],species_list,[2,3])

vni = extract_vn(data_chunks[1])
mult = mean([sum(vni[i,3,:,1,1]) for i in axes(vni,1)])
mult_per_bin = mean(vni, dims = 1)[1,3,:,1,1]
vn_pt = real.(vns[4].vm_result[:,1,1])
sum(transpose(vn_pt)*mult_per_bin)/mult
#plot!([vns[i].vm_result_integrated[2,1] for i in 1:10],xlabel = "Centrality class", label = L"v_2\, \mathrm{D^0}")

plot!([real(vns[i].vm_result_integrated[1,2]) for i in 1:4])



#single events

    pTlists = pt_list.(species_list)        
    pt_length_max = maximum(length.(pt_list.(species_list)))
    
    vm_result = zeros(pt_length_max,length(species_list),length(wavenum_list))
    vm_result_integrated = zeros(length(species_list),length(wavenum_list))
    vm_result_charged = zeros(pt_length_max,length(wavenum_list))
    vm_result_charged_integrated = zeros(length(wavenum_list))
        
    for result in event_list        
        q_vector_pt = q_vector_event_pt_dependent(result,species_list,wavenum_list)
        q_vector_total = q_vector_event_integrated(result,species_list,wavenum_list)

        for k in eachindex(species_list)
            pTlist = pTlists[k]
            for i in eachindex(pTlist)
                for wavenum in eachindex(wavenum_list)
                    vm_result[i,k,wavenum] += q_vector_pt[i,k,wavenum]*q_vector_total[wavenum]/length(event_list)
                end
            end
        end

    end



# 1. Load the raw data
# Replace 'data.yaml' with your actual filename or the string content

function parse_hep_data(data)
    # Extract independent variable (pT)
    indep = data["independent_variables"][1]["values"]
    pT_values = [mean([v["low"],v["high"]]) for v in indep]
    
    # Extract dependent variable (v2)
    dep_var = data["dependent_variables"][1]
    dep_values = dep_var["values"]
    
    # Initialize containers for our columns
    v2_values = Float64[]
    stat_err = Float64[]
    sys_err  = Float64[]
    
    for entry in dep_values
        push!(v2_values, entry["value"])
        
        # Parse errors
        for err in entry["errors"]
            if haskey(err, "label") && err["label"] == "stat"
                push!(stat_err, err["symerror"])
            elseif haskey(err, "label") && err["label"] == "sys"
                # Taking the absolute value of the minus/plus asymmetry 
                # (since they are equal in your data)
                #push!(sys_err, abs(err["asymerror"]["minus"]))
                push!(sys_err, abs(err["symerror"]))
           
           end
        end
    end
    
    # Create the DataFrame
    df = DataFrame(
        pT = pT_values,
        v2 = v2_values,
        stat_error = stat_err,
        sys_error = sys_err
    )
    
    return df
end

# 2. Process and Display
raw_data276 = [YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_1.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_2.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_3.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_4.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_5.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_6.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1297103-v1-Table_7.yaml")]

raw_data = [YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_1.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_2.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_3.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_4.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_5.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_6.yaml"),
            YAML.load_file("/home/alice/2d_hic_simulation/expdata/HEPData-ins1672822-v1-Table_7.yaml")]
df = parse_hep_data.(raw_data)

raw_data[1]["dependent_variables"][1]["qualifiers"][4]
colors = Plots.palette(:darkrainbow,7)
begin
    
p = scatter(df[1].pT, df[1].v2, 
    yerror = df[1].stat_error,              # Adds the error bars
    xlabel = "p_T (GeV/c)",
    ylabel = "v_2",
    label  = raw_data[1]["dependent_variables"][1]["qualifiers"][4]["value"],
    title  = "Elliptic Flow v₂ at √s_NN = 5.02 TeV",
    marker = (:circle, 4, colors[1]),  # Shape, size, color
    capsize = 2,                    # Adds the 'caps' to the error bars
    grid = :true,
    legend = :topleft,
    size = (800, 500)
)

for i in 2:length(df)
    scatter!(p,df[i].pT, df[i].v2, 
        yerror = df[i].stat_error,              # Adds the error bars
        label  = raw_data[i]["dependent_variables"][1]["qualifiers"][4]["value"],
        marker = (:circle, 4, colors[i]),  # Shape, size, color
        capsize = 2, ylims=(-0.05,0.4), xlims=(0,7) , legend=:topright             # Adds the 'caps' to the error bars
    )
end

display(p)


vns = [harmonic_coefficient_complex(data_chunks[i],species_list,[2,3]) for i in eachindex(centrality_bins)]
plot!(pt_list_pi, (real.(vns[1].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 0-5", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[1]) 
plot!(pt_list_pi, (real.(vns[2].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 5-10", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[2])
plot!(pt_list_pi,(real.(vns[3].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 10-20", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[3])
plot!(pt_list_pi,(real.(vns[4].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 20-30", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[4])
plot!(pt_list_pi,(real.(vns[5].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 30-40", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[5])
plot!(pt_list_pi,(real.(vns[6].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 40-50", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[6])
plot!(pt_list_pi,(real.(vns[7].vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 50-60", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[7])

end

vn = harmonic_coefficient_complex([data_chunks[1][2]],species_list,[2,3])
plot(pt_list_pi, (real.(vn.vm_result[:,1,1])), label = L"v_2\, \mathrm{\pi} 0-5", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[1]) 
plot!(pt_list_pi, (real.(vn.vm_result[:,2,1])), label = L"v_2\, \mathrm{\pi} 5-10", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[2])
plot!(pt_list_pi,(real.(vn.vm_result[:,3,1])), label = L"v_2\, \mathrm{\pi} 10-20", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[3])
plot!(pt_list_pi,(real.(vn.vm_result[:,4,1])), label = L"v_2\, \mathrm{\pi} 20-30", xlabel = L"p_T\, \mathrm{[GeV]}", ylabel = L"v_n", color = colors[4])
