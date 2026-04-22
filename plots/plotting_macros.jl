
using Plots

default(lw = 2, size=(600,400),xtickfontsize=16,ytickfontsize=16,xlabelfontsize=16,ylabelfontsize=16,legendfontsize=14,grid=false,framestyle=:box,margins=5Plots.mm)


function plot_identified_particle_spectra(data_chunks, particle_species, centrality_bins)
    pTlists = pt_list.(species_list)
    names = name.(species_list)
    colors = Plots.palette(:darkrainbow,length(data_chunks));
      
    for k in eachindex(particle_species)
        charged_particle_spectra_cc=spectra(data_chunks[1])
        spectra_average_cc = spectra_average(data_chunks[1])
        ptrange = pTlists[k]
        p = plot()
        for i in eachindex(charged_particle_spectra_cc)
            plot!(p, ptrange, charged_particle_spectra_cc[i][:,k], label ="", c=colors[1], alpha=0.3)
        end
        plot!(p, ptrange, spectra_average_cc[:,k], lw = 3, c=colors[1], alpha = 1, yscale =:log10,  label ="cc 0-5%")
        for cc in 2:7
            charged_particle_spectra_cc=spectra(data_chunks[cc])
            spectra_average_cc = spectra_average(data_chunks[cc])
            for i in eachindex(charged_particle_spectra_cc)
                plot!(p, ptrange, charged_particle_spectra_cc[i][:,k], label ="", c=colors[cc], alpha=0.3)
            end
            label = "cc $(centrality_bins[cc-1])-$(centrality_bins[cc])%"
            plot!(p, ptrange, spectra_average(data_chunks[cc])[:,k], lw = 3, alpha = 1, c=colors[cc], yscale =:log10, label = label, xlabel=L"$p_T$ (GeV)", ylabel = L"\frac{dN}{dyp_Tdp_T} (GeV^{-2})", title = "$(names[k]) spectra")
        end
        savefig(p, "./plots/spectra_$(names[k])_cc.pdf")
    end
end

function plot_identified_particle_spectra_average(data_chunks, particle_species, centrality_bins;)
    pTlists = pt_list.(species_list)
    names = name.(species_list)
    colors = Plots.palette(:darkrainbow,length(data_chunks));
    q = plot()
         
    for k in eachindex(particle_species)
        spectra_average_cc = spectra_average(data_chunks[1])
        ptrange = pTlists[k]
        p = plot(ptrange, spectra_average_cc[:,k], lw = 3, c=colors[1], alpha = 1, yscale =:log10,  label ="cc 0-5%")
        plot!(q, ptrange, spectra_average_cc[:,k], lw = 3, c=colors[1], alpha = 1, yscale =:log10,  label ="cc 0-5%")
        
        for cc in 2:7
            spectra_average_cc = spectra_average(data_chunks[cc])
            label = "cc $(centrality_bins[cc-1])-$(centrality_bins[cc])%"
            plot!(p, ptrange, spectra_average(data_chunks[cc])[:,k], lw = 3, alpha = 1, c=colors[cc], yscale =:log10, label = label, xlabel=L"$p_T$ (GeV)", ylabel = L"\frac{dN}{dyp_Tdp_T} (GeV^{-2})", title = "$(names[k]) spectra")
            plot!(q, ptrange, spectra_average(data_chunks[cc])[:,k], lw = 3, alpha = 1, c=colors[cc], yscale =:log10, label = label, xlabel=L"$p_T$ (GeV)", ylabel = L"\frac{dN}{dyp_Tdp_T} (GeV^{-2})", title = "$(names[k]) spectra")
        end
        savefig(p, "./plots/spectra_all_charged_particles_cc_average.pdf")
    end
    savefig(q, "./plots/spectra_all_charged_particles_cc_average.pdf")
end


function plot_total_multiplicity_average(data_chunks, centrality_bins)
    
    p = scatter()
    total_mult_avg = zeros(length(centrality_bins))
    total_mult_std = zeros(length(centrality_bins))
    for cc in eachindex(centrality_bins)
        mult_cc = multiplicity(data_chunks[cc])
        total_mult = get_total_multiplicity(mult_cc)
        total_mult_avg[cc] = total_multiplicity_average(total_mult)
        total_mult_std[cc] = std(total_mult)
    end

    scatter!(p,centrality_bins, total_mult_avg, yerr = total_mult_std, lw = 3, alpha = 1, label = "", xlabel="cc", ylabel = "multiplicity")
    savefig(p, "./plots/total_multiplicity_avg.pdf")
    
end

function plot_total_multiplicity(data_chunks, centrality_bins)
    
    p = scatter()
    colors = Plots.palette(:darkrainbow,length(data_chunks));
    
    for cc in eachindex(centrality_bins)
        mult_cc = multiplicity(data_chunks[cc])
        total_mult = get_total_multiplicity(mult_cc)
        if cc>1 
            label = "cc $(centrality_bins[cc-1])-$(centrality_bins[cc])%" 
            else label = "cc 0-5%"
        end
        scatter!(p,total_mult, lw = 3, c=colors[cc], alpha = 1, label = label, xlabel="N event", ylabel = "multiplicity")
        savefig(p, "./plots/total_multiplicity_ebe.pdf")
    end
end

function plot_identified_multiplicity_average(data_chunks, centrality_bins, species_list)
    
    p = scatter()
    id_mult_avg = zeros(length(centrality_bins),length(species_list))
    id_mult_std = zeros(length(centrality_bins),length(species_list))
    for cc in eachindex(centrality_bins)
        mult_cc = multiplicity(data_chunks[cc])
        id_mult = get_identified_multiplicity(mult_cc, species_list)
        id_mult_avg[cc,:] = identified_multiplicity_average(id_mult)[:,1]
        id_mult_std[cc,:] = std(id_mult, dims=2)[:,1]
    end
    names = name.(species_list)
    for k in eachindex(species_list)
    scatter!(p,centrality_bins, id_mult_avg[:,k], 
    yerr = id_mult_std[:,k], 
    lw = 3, alpha = 1, label = names[k], xlabel="cc", ylabel = "multiplicity")
    end
    savefig(p, "./plots/identified_multiplicity_avg.pdf")
    
end