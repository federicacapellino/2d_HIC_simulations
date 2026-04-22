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
@info "Julia threads" nthreads()

include("MCglauber.jl")
include("hdf5_io.jl")
include("observables.jl")
include("fastreso.jl")

# the convention here are T, ux, uy, piyy, pizz, pixy, piB this has to match with the matrix defined
twod_visc_hydro = Fields(
    NDField((:even, :ghost), (:even, :ghost), :temperature),
    NDField((:ghost, :ghost), (:ghost, :ghost), :ux),
    NDField((:ghost, :ghost), (:ghost, :ghost), :uy),
    NDField((:ghost, :ghost), (:even, :ghost), :piyy),
    NDField((:even, :ghost), (:even, :ghost), :pizz),
    NDField((:ghost, :ghost), (:ghost, :ghost), :pixy),
    NDField((:even, :ghost), (:even, :ghost), :piB),
    NDField((:even, :ghost), (:even, :ghost), :mu),
    NDField((:ghost, :ghost), (:ghost, :ghost), :nux),
    NDField((:ghost, :ghost), (:ghost, :ghost), :nuy)
)


# Load configuration from YAML
# Get the config file path from command line arguments, or use default
config_file = if length(ARGS) > 0
    ARGS[1]
else
    @error "Missing parameter file! Try with julia /path/to/collision_event_yaml.jl /path/to/config.yaml"
end

@info "Loading configuration from: $config_file"
config = YAML.load_file(config_file)
physics_params = config["physics parameter"]
run_params = config["run parameter"]

# Run parameters
Nev = run_params["Nev"]
const checkpoint_interval = run_params["checkpoint_interval"]
const checkpoint_file = run_params["checkpoint_file"]
const gridpoints = run_params["gridpoints"]
const xmax = run_params["xmax"]
const wavenum_m = run_params["wavenum"]

n_batches = ceil(Int, Nev / checkpoint_interval)

# Fluid evolution parameters
const tau0 = physics_params["fluid evolution"]["tau0"]
const tmax = physics_params["fluid evolution"]["tmax"]
const norm = physics_params["fluid evolution"]["norm"]
const eta_over_s = physics_params["fluid evolution"]["eta_over_s"]
const zeta_over_s = physics_params["fluid evolution"]["zeta_over_s"]
const DsT = physics_params["fluid evolution"]["DsT"]
const Tfo = physics_params["fluid evolution"]["Tfo"]

# Charm parameters
const dσ_QQdy = physics_params["charm parameter"]["dσ_QQdy"]
const σ_in = physics_params["charm parameter"]["σ_in"]

# Initial condition parameters
const s_NN = physics_params["initial conditions"]["s_NN"]
const w = physics_params["initial conditions"]["w"]
const k = physics_params["initial conditions"]["k"]
const p = physics_params["initial conditions"]["p"]
b = physics_params["initial conditions"]["b"]


# Additional parameters
const tau_eta_par = physics_params["additional parameters"]["tau_eta_par"]
const tau_zeta_par = physics_params["additional parameters"]["tau_zeta_par"]
const hq_mass = physics_params["additional parameters"]["hq_mass"]

# Set up nuclear parameters
spec = physics_params["initial conditions"]["n1"]  # e.g. "Lead()"
const n1 = eval(Meta.parse(spec))
spec = physics_params["initial conditions"]["n2"]  # e.g. "Lead()"
const n2 = eval(Meta.parse(spec))

# Create discretization
discretization = CartesianDiscretization(Fluidum.SymmetricInterval(gridpoints, xmax), Fluidum.SymmetricInterval(gridpoints, xmax))
# Prepare the field with the discretization
twod_visc_hydro_discrete = DiscreteFields(twod_visc_hydro, discretization, Float64)

if b == "minBias"
    participants=Participants(n1,n2,w,s_NN,k,p)
else
    b_tuple = eval(Meta.parse(b))
    participants=Participants(n1,n2,w,s_NN,k,p,b_tuple)
end

pion = particle_simple("pion", 0.13957, 1, 0)
D0 = particle_simple("D0", 1.86483, 1, 1)

Fj = fastreso_reader(pwd() * "/examples/event-by-event/PDGid_211_total_T0.1560_Fj.out")
const particle_full_π = particle_full("pion", 0.13957, 1, 0, Fj[1])
Fj = fastreso_reader(pwd() * "/examples/event-by-event/Dc1865zer_total_T0.1560_Fj.out")
const particle_full_D0 = particle_full("D0", 1.86483, 1, 1, Fj[1])


function run_event(
        participants, twod_visc_hydro_discrete, norm; eta_p = 0.0,
        wavenum_m = [2, 3], species_list = [pion, D0]
    )
    discretization = twod_visc_hydro_discrete.discretization
    #create event
    event = rand(participants)
    #compute center of mass
    mult, x_com, y_com = center_of_mass(event, 100, 50)
    xcm = x_com / mult
    ycm = y_com / mult
    profile = map(discretization.grid) do y
        y = y .+ (xcm, ycm)
        event(y...)
    end
    ncoll_event = event.n_coll

    println("     🎲 Glauber mult: $(round(mult, digits = 2)) | Ncoll: $ncoll_event")


    tspan = (tau0, tmax)
    #set up fluid properties
    #dσ_QQdy = 0.4087 #in mb FONLL #0.04087 with NNPDF_nlo_as_0118
    #σ_in = 70.0 #in mb
    ccbar = ncoll_event * 2 * dσ_QQdy / σ_in #charm pair number density at tau0
    ccbar_norm = 2 * dσ_QQdy / σ_in / tau0
    eos = Heavy_Quark(readresonancelist(), ccbar)
    viscosity = QGPViscosity(eta_over_s, tau_eta_par)
    bulk = SimpleBulkViscosity(zeta_over_s, tau_zeta_par)
    diffusion = ZeroDiffusion() #maybe diffusion too large?
    @show charm_number_hard(ncoll_event;xmax=20.,ncoll_norm = ccbar_norm)
    fluidproperty = FluidProperties(eos, viscosity, bulk, diffusion)

    #setup fields
    temperature_func = trento_event_eos(profile, norm = norm, exp_tail = false)
    fug_func = fug_(temperature_func, ncoll_event, eos, discretization; ncoll_norm = ccbar_norm)

    phi = set_array(temperature_func .+ 0.01, :temperature, twod_visc_hydro_discrete)  #maybe offset too large?
    set_array!(phi, fug_func, :mu, twod_visc_hydro_discrete)
    
    # Plot phi[8,:,:] and save to PDF
    #p = heatmap(phi[8,:,:], title="phi[8,:,:]", xlabel="x", ylabel="y")
    #savefig(p, "phi_8_plot.pdf")

    simulation_pars = (
        dσ_QQdy = dσ_QQdy,
        viscosity = viscosity.ηs,
        bulk = bulk.ζs,
        diffusion = 0.0, #diffusion.DsT,
        t0 = tspan[1],
        Tfo = Tfo,
        species_list = name.(species_list),
        pTlist = pt_lists(species_list),
        length_pTlist = length.(pt_list.(species_list)),
        wavenum_m = wavenum_m,
    )

    result = Fluidum.isosurface(twod_visc_hydro_discrete, Fluidum.matrix2d_visc_HQ!, fluidproperty, phi, tspan, :temperature, Tfo)
    cha = Fluidum.Chart(Fluidum.Surface(result[:surface]), (t, x, y) -> Fluidum.SVector{2}(atan(t, hypot(y, x)), atan(y, x)))
    if length(cha.points) == 0
        return null_observable(wavenum_m, species_list), simulation_pars
    end
    fo_bg = Fluidum.freezeout_interpolation(cha, sort_index = 2, ndim_tuple = 50)

    #run observables
    vn = dvn_dp_list_delta(fo_bg, species_list, eta_p, wavenum_m; eta_min = -5.0, eta_max = 5.0)

    return ObservableResult(mult, vn.u), simulation_pars
end


function run_event_by_event(Nev)
    return tmap(1:Nev) do i
        println("Running event $i / $Nev")
        result = run_event(
            participants,
            twod_visc_hydro_discrete,
            norm; species_list = [particle_full_π, particle_full_D0]
        )
        result
    end
end


function progress_bar(fraction; width = 30)
    filled = round(Int, fraction * width)
    empty = width - filled
    return "▓"^filled * "░"^empty
end

function format_time(seconds)
    if seconds < 60
        return "$(round(seconds, digits = 1))s"
    elseif seconds < 3600
        m, s = divrem(seconds, 60)
        return "$(Int(m))m $(round(s, digits = 0))s"
    else
        h, rem = divrem(seconds, 3600)
        m, s = divrem(rem, 60)
        return "$(Int(h))h $(Int(m))m"
    end
end
begin
    println()
    println("  ╔══════════════════════════════════════════════════════════╗")
    println("  ║     🌊  FLUIDUM Event-by-Event Simulation  🌊            ║")
    println("  ╚══════════════════════════════════════════════════════════╝")
    println()
    println("  ┌─────────────────────────────────────────────────────────┐")
    println("  │  📊 Simulation Parameters                               │")
    println("  ├─────────────────────────────────────────────────────────┤")
    println("  │  🎯 Total events:        $(lpad(Nev, 10))                   │")
    println("  │  💾 Checkpoint interval: $(lpad(checkpoint_interval, 10))                   │")
    println("  │  📦 Number of batches:   $(lpad(n_batches, 10))                   │")
    println("  │  🧵 Threads available:   $(lpad(Threads.nthreads(), 10))                   │")
    println("  │  📁 Output file:         $(rpad(checkpoint_file, 20))    │")
    println("  └─────────────────────────────────────────────────────────┘")
    println()
end;
start_time = time()
batch_times = Float64[]

for batch in 1:n_batches
    batch_start = (batch - 1) * checkpoint_interval + 1
    batch_end = min(batch * checkpoint_interval, Nev)
    batch_size = batch_end - batch_start + 1

    println("  ┌─────────────────────────────────────────────────────────┐")
    println("  │  🚀 Batch $batch/$n_batches │ Events $batch_start → $batch_end ($batch_size events)")
    println("  └─────────────────────────────────────────────────────────┘")


    sim = run_event_by_event(batch_size)
    result = getindex.(sim, 1)
    batch_time = @elapsed data = result
    push!(batch_times, batch_time)
    metadata = sim[1][2]
    append_to_h5(checkpoint_file, data, metadata)

    completed = min(batch * checkpoint_interval, Nev)
    elapsed = time() - start_time
    rate = completed / elapsed
    eta = (Nev - completed) / rate
    fraction = completed / Nev

    avg_mult = sum(d.glauber_multiplicity for d in data) / length(data)

    println()
    println("     $(progress_bar(fraction)) $(round(100 * fraction, digits = 1))%")
    println()
    println("     ⏱️  Batch time:     $(format_time(batch_time))")
    println("     ⏳ Total elapsed:  $(format_time(elapsed))")
    println("     🏁 Completed:      $completed / $Nev events")
    println("     ⚡ Rate:           $(round(rate, digits = 2)) events/s")
    println("     🔮 ETA:            $(format_time(eta))")
    println("     📈 Avg multiplicity: $(round(avg_mult, digits = 2))")
    println("     💾 Checkpoint saved ✅")
    println()
end

total_time = time() - start_time
avg_batch_time = sum(batch_times) / length(batch_times)

println("  ╔══════════════════════════════════════════════════════════╗")
println("  ║              🎉 SIMULATION COMPLETED! 🎉                 ║")
println("  ╚══════════════════════════════════════════════════════════╝")
println()
println("  ┌─────────────────────────────────────────────────────────┐")
println("  │  📊 Final Statistics                                    │")
println("  ├─────────────────────────────────────────────────────────┤")
println("  │  ⏱️  Total time:        $(lpad(format_time(total_time), 15))               │")
println("  │  ⚡ Average rate:      $(lpad(round(Nev / total_time, digits = 2), 12)) events/s    │")
println("  │  📦 Avg batch time:    $(lpad(format_time(avg_batch_time), 15))               │")
println("  │  🧵 Threads used:      $(lpad(Threads.nthreads(), 15))               │")
println("  │  📁 Results saved to:  $(rpad(checkpoint_file, 20))    │")
println("  └─────────────────────────────────────────────────────────┘")
println()
println("  🌟 Thank you for using Fluidum! 🌟")
println()
println("="^60)
