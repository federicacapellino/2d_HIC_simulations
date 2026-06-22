using Fluidum
using MonteCarloGlauber
using HaloArrays
using Printf
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
using Fluidum.Bessels
using Plots
using LaTeXStrings
using DelimitedFiles

using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK



default(aspect_ratio = :equal,
    size = (300, 300),
    dpi = 200,linewidth = 2, axis=true,
    markersize = 8
    , left_margin = 1Plots.mm,
    right_margin = 7Plots.mm,
    top_margin = 1Plots.mm,
    bottom_margin =1Plots.mm, legendfont = "sans-serif", titlefontsize = 12, tickfontsize = 10, legendfontsize = 10)

import Fluidum.QuadGK:quadgk
import Fluidum.HCubature:hcubature
@info "Julia threads" nthreads()

kernels = Fluidum.root_kernels

include("../utils/MCglauber.jl")
include("../utils/hdf5_io.jl")
include("../utils/observables.jl")
include("../utils/fastreso.jl")
#include("../utils/finite_volume.jl")
include("../utils/finite_volum_diffeq.jl")

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

config = YAML.load_file("config.yaml")

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
disc = twod_visc_hydro_discrete.discretization

if b == "minBias"
    participants=Participants(n1,n2,w,s_NN,k,p)
else
    b_tuple = eval(Meta.parse(b))
    participants=Participants(n1,n2,w,s_NN,k,p,b_tuple)
end


Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_211_total_T0.1560_Fj.out"))
const particle_full_π = particle_full("pion", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_2212_total_T0.1560_Fj.out"))
const particle_full_p = particle_full("proton", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_321_total_T0.1560_Fj.out"))
const particle_full_k = particle_full("kaon", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/Dc1865zer_total_T0.1560_Fj.out"))
const particle_full_D0 = particle_full("D0", Fj[4], 1, 1, Fj[1])


species_list = [particle_full_π, particle_full_p, particle_full_k, particle_full_D0]


#grid and temperature
discretization = twod_visc_hydro_discrete.discretization
event = rand(participants)
ncoll_event = event.n_coll
mult, x_com, y_com = center_of_mass(event, 100, 50)
xcm = x_com / mult
ycm = y_com / mult
profile = map(discretization.grid) do y
    y = y .+ (xcm, ycm)
    event(y...)
    end

#Fluid properties setup
begin
    @show ccbar = 2* ncoll_event * dσ_QQdy / σ_in #charm pair number density at tau0
    ccbar_norm = 2 * dσ_QQdy / σ_in / tau0
    particle_list = "./particles_D0.data";
    eos = Heavy_Quark(particle_list, ccbar)
    viscosity = QGPViscosity(eta_over_s, tau_eta_par)
    bulk = SimpleBulkViscosity(zeta_over_s, tau_zeta_par)
    viscosity = ZeroViscosity()
    bulk = ZeroBulkViscosity()
    diffusion = ZeroDiffusion() 
    fluidproperty = FluidProperties(eos, viscosity, bulk, diffusion)
end


tmap, nhardmap, fugmap, μmap = MCGlauber_to_fields(event, discretization, σ_in,dσ_QQdy,tau0,eos, offset=1e-4)

heatmap(tmap)
heatmap(nhardmap)
heatmap(fugmap)
heatmap(μmap)

maximum(μmap)
maximum(tmap)
minimum(μmap)
minimum(tmap)

tmap
maximum(fugmap)
isinf.(fugmap)

ncoll_int=hcubature(b->ncoll_fluctuating_thickness(b[1],b[2],event,σ_in),(-20.0, -20.0), (20.0, 20.0), rtol=1e-3, atol=1e-3)
ncoll_event    

ccbar_norm = 2. /tau0/σ_in*dσ_QQdy
ncoll_int[1]*ccbar_norm*tau0
ncoll_event*ccbar_norm*tau0


begin
N = 0.
for i in eachindex(axes(tmap,1))
    for j in eachindex(axes(tmap,2))
        N+=thermodynamic(tmap[i,j],fugmap[i,j],eos.hadron_list).pressure
    end
end
println(N*tau0*0.2*0.2)
end

begin
N = 0.
for i in eachindex(axes(tmap,1))
    for j in eachindex(axes(tmap,2))
        N+=nhardmap[i,j]+ 1e-4
    end
end
println(N*tau0*0.2*0.2)
end


@inline charge_density(T,α,eos) = thermodynamic(T,α,eos.hadron_list).pressure
@inline fluid_pressure(T,eos) = Fluidum.pressure(T,eos)
@inline fluid_energy_density(T,eos) = T*Fluidum.pressure_derivative(T,Val(1),eos) - p
@inline enthalpy_density(T,eos) = fluid_energy_density(T,eos) + fluid_pressure(T,eos)
@inline sound_speed_sq(eos) = 1. / 3.            # c_s² = 1/3
@inline function sound_speed_sq(T, eos)
    dp_dT  = Fluidum.pressure_derivative(T, Val(1), eos)
    d2p_dT2 = Fluidum.pressure_derivative(T, Val(2), eos)
    
    # cs^2 = (dp/dT) / (de/dT) = dp_dT / (T * d2p_dT2)
    return dp_dT / (T * d2p_dT2)
end
 #we define some random intial condition 
function temperature(r)
       0.4(1+0.3)/(exp(abs(r))+1 )+0.01
end
#we set the array corresponding to the temperature 
phi=Fluidum.set_array((x,y)->temperature(hypot(x,y)),:temperature,twod_visc_hydro_discrete);
heatmap(phi[4,60:80,60:80])
heatmap(interior_view(sol[1][:,:]))

phi[1,70:75,70:75]
zeros(10,10)
run_2d_ideal_charge(eos,phi[1,70:75,70:75], 0. *phi[1,70:75,70:75], t_end=.6, cfl =0.1)

sol = run_2d_ideal_charge_diffeq(eos,tmap[60:80,60:80], μmap[60:80,60:80], t_end=2., cfl =0.3)
sol
final_fields = similar(tmap)
final_fields = get_primitive_variables(sol,eos)

heatmap(final_fields.T[:,:])
heatmap(final_fields.vx[:,:])
heatmap(final_fields.vy[:,:])
heatmap(final_fields.μ[:,:])

heatmap(interior_view(sol[4][:,:]))
maximum(interior_view(sol[1][:,:]))
maximum(interior_view(sol[4][:,:]))

maximum(phi[1,60:80,60:80])
using StaticArrays
using LinearAlgebra

using Printf
using StaticArrays

function verify_2x2_conservation(eos)
    println("==================================================")
    println("  RUNNING 2x2 HYDRODYNAMICS FLUX DISCRETIZATION TEST")
    println("==================================================")
    
    n = 2
    dx = 1.0 / n
    dy = 1.0 / n

    # Initialize a 2x2 grid using your actual LocalMultiHaloArray allocation scheme
    # Halo width = 1, boundary condition = :repeating
    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:N, :Mx, :My, :E), boundary_condition=:repeating)
    du = similar(u)

    # Populate the 2x2 cells with an asymmetric high-pressure spot at cell (1,1)
    # to guarantee non-zero fluxes across all inner and outer periodic faces.
    for j in 1:n, i in 1:n
        if i == 1 && j == 1
            # "Spark" cell primitives -> converted to conserved
            U = prim_to_cons(0.3, 0.5, 0.1, -0.1, eos)
        else
            # "Atmosphere" cells
            U = prim_to_cons(0.1, 0.1, 0.0, 0.0, eos)
        end
        interior_view(u.N)[i, j]  = U[1]
        interior_view(u.Mx)[i, j] = U[2]
        interior_view(u.My)[i, j] = U[3]
        interior_view(u.E)[i, j]  = U[4]
    end

    # Run your code's actual halo wrapping logic
    synchronize_halo!(u)

    # Compute spatial derivatives/flux divergence updates across the grid using your exact RHS routine
    rel_rhs!(du, u, eos, dx, dy)

    # Extract raw structural arrays including the halo buffers to check totals
    pN  = parent(du).N
    pMx = parent(du).Mx
    pMy = parent(du).My
    pE  = parent(du).E

    # Check total sums across the entire memory structure (including halos)
    total_sum_N  = sum(pN)
    total_sum_Mx = sum(pMx)
    total_sum_My = sum(pMy)
    total_sum_E  = sum(pE)

    # Check sums strictly within the active domain region (ignoring halos)
    # Since n=2 and halo=1, interior slices are rows/cols 2:3
    interior_sum_N  = sum(pN[2:3, 2:3])
    interior_sum_Mx = sum(pMx[2:3, 2:3])
    interior_sum_My = sum(pMy[2:3, 2:3])
    interior_sum_E  = sum(pE[2:3, 2:3])

    @printf("\n--- GLOBAL PARENT MEMORY STORAGE SUMS ---\n")
    @printf("  Total N Array Sum : %+.15e\n", total_sum_N)
    @printf("  Total E Array Sum : %+.15e\n", total_sum_E)
    @printf("  Total Mx Array Sum: %+.15e\n", total_sum_Mx)
    @printf("  Total My Array Sum: %+.15e\n", total_sum_My)

    @printf("\n--- ACTIVE INTERIOR PHYSICAL DOMAIN SUMS ---\n")
    @printf("  Interior N Sum    : %+.15e\n", interior_sum_N)
    @printf("  Interior E Sum    : %+.15e\n", interior_sum_E)
    @printf("  Interior Mx Sum   : %+.15e\n", interior_sum_Mx)
    @printf("  Interior My Sum   : %+.15e\n", interior_sum_My)

    println("\n================ DIAGNOSTIC REPORT ================")
    if abs(interior_sum_E) < 1e-13 && abs(interior_sum_N) < 1e-13
        println("  ✓ SUCCESS: Your active interior cells balance to machine precision.")
        println("             No matter changes, global conservation laws will hold perfectly.")
    elseif abs(total_sum_E) < 1e-13 && abs(interior_sum_E) > 1e-13
        println("  ✗ FAILURE: The complete memory array balances out, but your active")
        println("             interior domain is leaking values into the ghost rows/columns.")
        println("             This confirms boundary fluxes are overwritten by `synchronize_halo!`.")
    else
        println("  ✗ FAILURE: Even the raw memory buffer arrays do not balance to zero.")
        println("             Check your `accumulate_flux_divergence!` loop layout; it is likely")
        println("             miscounting or dropping face loops entirely.")
    end
    println("===================================================\n")
end

verify_2x2_conservation(eos)

n = size(temperature_map,1)
dx = 1.0 / n;  dy = 1.0 / n
# Allocate physical variables and Runge-Kutta workspaces
u  = LocalMultiHaloArray(Float64, (n, n), 1;
    fields=(:N, :Mx, :My, :E), boundary_condition=:repeating)
u1 = similar(u)
du = similar(u)
# Convert primitive variables to conserved variables and populate the grid
for j in 1:n, i in 1:n
    T = temperature_map[i, j]
    μ = μ_map[i, j]
    
    # Initial condition assumes zero initial fluid velocity (vx=0, vy=0)
    U = cons_from_prim(eos, T, μ, 0.0, 0.0)
    
    interior_view(u.N)[i, j]  = U[1]
    interior_view(u.Mx)[i, j] = U[2]
    interior_view(u.My)[i, j] = U[3]
    interior_view(u.E)[i, j]  = U[4]
end
synchronize_halo!(u)


#why NaN here?
q0, e0, _ = diagnostics(u, eos, dx, dy)