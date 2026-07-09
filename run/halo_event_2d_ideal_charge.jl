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
#include("../utils/finite_volum_diffeq_multithread.jl")

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


tmap, nhardmap, fugmap, μmap = MCGlauber_to_fields(event, discretization, σ_in, dσ_QQdy,tau0,eos, offset=1e-4)

heatmap(tmap)
heatmap(nhardmap)
heatmap(fugmap)
heatmap(μmap)

maximum(μmap)
maximum(tmap)
minimum(μmap)
minimum(tmap)

maximum(fugmap)
isinf.(fugmap)

ncoll_int=hcubature(b->σ_in*MonteCarloGlauber.ncoll_fluctuating_thickness(b[1],b[2],event),(-20.0, -20.0), (20.0, 20.0), rtol=1e-3, atol=1e-3)
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
@inline fluid_energy_density(T,eos) = T*Fluidum.pressure_derivative(T,Val(1),eos) - fluid_pressure(T,eos)
@inline enthalpy_density(T,eos) = fluid_energy_density(T,eos) + fluid_pressure(T,eos)
@inline sound_speed_sq(eos) = 1. / 3.            # c_s² = 1/3
@inline function sound_speed_sq(T, eos)
    dp_dT  = Fluidum.pressure_derivative(T, Val(1), eos)
    d2p_dT2 = Fluidum.pressure_derivative(T, Val(2), eos)
    
    # cs^2 = (dp/dT) / (de/dT) = dp_dT / (T * d2p_dT2)
    return dp_dT / (T * d2p_dT2)
end
 #we define some random intial condition 

#this now takes a long time
#sol = run_2d_ideal_charge_diffeq(eos,tmap, μmap, t_end=.6, cfl =0.1)
sol = run_2d_ideal_charge_diffeq(eos,tmap[60:80,60:80], μmap[60:80,60:80], t_end=.6, cfl =0.1)

#alternatively

phi = map(Iterators.product((-1:0.1:1),(-1:0.1:1))) do I
    x,y = Tuple(I)
    0.4*(1+0.3)/(exp(abs(x^2+y^2)/0.1))+1e-4
end
phi
heatmap(phi)

sol = run_2d_ideal_charge_diffeq(eos,phi, 0. *phi, t_end=.6, cfl =0.1)
sol = run_2d_ideal_charge_diffeq_threaded(eos,tmap[60:70,60:70], μmap[60:70,60:70], t_end=.6, cfl =0.3)
