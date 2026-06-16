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
    eos = Heavy_Quark(readresonancelist(;name_file=particle_list), ccbar)
    viscosity = QGPViscosity(eta_over_s, tau_eta_par)
    bulk = SimpleBulkViscosity(zeta_over_s, tau_zeta_par)
    viscosity = ZeroViscosity()
    bulk = ZeroBulkViscosity()
    diffusion = ZeroDiffusion() 
    fluidproperty = FluidProperties(eos, viscosity, bulk, diffusion)
end


tmap, nhardmap, fugmap, μmap = MCGlauber_to_fields(event, discretization, σ_in,dσ_QQdy,tau0,eos)

heatmap(tmap)
heatmap(nhardmap)
heatmap(fugmap)
heatmap(μmap)

maximum(μmap)
maximum(fugmap)

ncoll_int=hcubature(b->ncoll_fluctuating_thickness(b[1],b[2],event,σ_in),(-20.0, -20.0), (20.0, 20.0), rtol=1e-3, atol=1e-3)
ncoll_event    

ccbar_norm = 2. /tau0/σ_in*dσ_QQdy
ncoll_int[1]*ccbar_norm*tau0
ncoll_event*ccbar_norm*tau0

isinf.(fugmap)

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
        N+=nhardmap[i,j]+ 1e-5
    end
end
println(N*tau0*0.2*0.2)
end

ccbar


@inline charge_density(T,μ,eos) = thermodynamic(T,μ,eos.hadron_list).pressure
@inline fluid_pressure(T,eos) = Fluidum.pressure(T,eos)
@inline fluid_energy_density(T,eos) = T*Fluidum.pressure_derivative(T,Val(1),eos) - p
@inline enthalpy_density(T,eos) = fluid_energy_density(T,eos) + fluid_pressure(T,eos)
@inline sound_speed_sq(eos) = 1. / 3.            # c_s² = 1/3

# ─── Primitive ↔ conserved ────────────────────────────────────────────────────

@inline function cons_from_prim(eos, T, μ, vx, vy)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2)
    p = fluid_pressure(T, eos)
    n = charge_density(T, μ, eos)
    w = enthalpy_density(T, eos)
    N  = n * W
    Mx = w * W^2 * vx
    My = w * W^2 * vy
    E  = w * W^2 - p
    return SVector(N, Mx, My, E)
end

N, Mx, My, E = cons_from_prim(eos, 0.3,4.,0.4,.3)
#dpt = Fluidum.pressure_derivative(1.,Val(1),eos) #entropy
#dptt = Fluidum.pressure_derivative(1,Val(2),eos)

# Two residuals enforcing the charge and enthalpy identities at trial (T, μ),
# with |M|² entering exactly as in 1-D (here |M|² = Mₓ² + M_y²).
#should be (0,0) if the primitives are correct

@inline function _residuals(eos, E, M2, N, T, μ)
    p = fluid_pressure(T, eos)
    n = charge_density(T, μ, eos)
    w = enthalpy_density(T, eos)
    X = E + p
    Z = sqrt(max(X^2 - M2, 1.0e-30))
    W = X / Z
    return n * W - N, w * W^2 - X
end

_residuals(eos, E+15., Mx^2+My^2,N+10, 0.5, 1.)

function prim_from_cons(eos, U; maxit=500, tol=1.0e-11)
    N, Mx, My, E = U
    M2 = Mx^2 + My^2

    #initial trial seeds for primitives
    T = max((max(E^(0.25), 1.0e-12)), 1.0e-8)             # W≈1 seed (lorentz gamma)
    mass = eos.hadron_list[1].Mass
    b2 = besselkx(2,mass/T)
    μ = T * log(max(max(N*2*pi^2, 1.0e-12) / (mass^2*T*b2), 1.0e-300))
    #μ = T * log(max(max(N, 1.0e-12) / (T^3), 1.0e-300))

    for _ in 1:maxit
        R1, R2 = _residuals(eos, E, M2, N, T, μ)
        #guess update
        δT = max(1.0e-7 * abs(T), 1.0e-9)
        δμ = max(1.0e-7 * abs(μ), 1.0e-9)
        #compute new residuals
        R1T, R2T = _residuals(eos, E, M2, N, T + δT, μ)
        R1m, R2m = _residuals(eos, E, M2, N, T, μ + δμ)
        #compute jacobian of the residual
        J11 = (R1T - R1) / δT;  J21 = (R2T - R2) / δT
        J12 = (R1m - R1) / δμ;  J22 = (R2m - R2) / δμ
        det = J11 * J22 - J12 * J21
        abs(det) < 1.0e-300 && break
        #from dR to dT, dμ
        dT = (-R1 * J22 + R2 * J12) / det
        dμ = ( R1 * J21 - R2 * J11) / det
        T  = max(T + dT, 1.0e-10)
        μ  = μ + dμ
        (abs(dT) < tol * (abs(T) + 1.0e-12) && abs(dμ) < tol * (abs(μ) + 1.0e-12)) && break
    end

    p = fluid_pressure(T, eos)
    X = E + p
    invX = 1.0 / max(X, 1.0e-30)
    return T, μ, Mx * invX, My * invX                        # T, μ, vx, vy
end

prim_from_cons(eos, (N, Mx, My, E))
N, Mx, My, E
# ─── Direction-aware flux and signal speed ────────────────────────────────────

@inline function physical_flux(eos, U, dim::Int)
    T, μ, vx, vy = prim_from_cons(eos, U)
    p = fluid_pressure(T, eos)
    N, Mx, My, _ = U
    if dim == 1
        return SVector(N * vx, Mx * vx + p, My * vx, Mx)
    else
        return SVector(N * vy, Mx * vy, My * vy + p, My)
    end
end


physical_flux(eos, (N, Mx, My, E), 1) #the dim index selects the component

@inline function max_wave_speed(eos, U, dim::Int)
    _, _, vx, vy = prim_from_cons(eos, U)
    v2  = vx^2 + vy^2
    vn  = dim == 1 ? vx : vy
    cs2 = sound_speed_sq(eos)
    disc = (1.0 - v2) * ((1.0 - v2 * cs2) - vn^2 * (1.0 - cs2))
    root = sqrt(max(disc, 0.0)) * sqrt(cs2)
    den  = 1.0 - v2 * cs2
    return max(abs((vn * (1.0 - cs2) + root) / den), abs((vn * (1.0 - cs2) - root) / den))
end

max_wave_speed(eos, (N, Mx, My, E), 2)

@inline function rusanov_flux(eos, UL, UR, dim::Int)
    smax = max(max_wave_speed(eos, UL, dim), max_wave_speed(eos, UR, dim))
    return 0.5 * (physical_flux(eos, UL, dim) + physical_flux(eos, UR, dim)) -
           0.5 * smax * (UR - UL)
end

# ─── Field access (conserved fields N, Mx, My, E) ─────────────────────────────

@inline conserved_cell(d, I) = SVector(d.N[I], d.Mx[I], d.My[I], d.E[I])

@inline function add_conserved!(d, I, scale, U)
    d.N[I]  += scale * U[1]
    d.Mx[I] += scale * U[2]
    d.My[I] += scale * U[3]
    d.E[I]  += scale * U[4]
    return d
end

# ─── RHS (x then y sweep) and SSP-RK2 ─────────────────────────────────────────

function rel_rhs!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)                  # :repeating fills the outflow ghosts
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
    return du
end

#predict derivative at n+1 and evaluate u at n+1 by  summing u(n+1) = u(n) + 0.5*(u'(n)+u'(n+1))*dt
#u'(n) is computed with RHS, u'(n+1) is predicted 
function ssprk2_step!(u, u1, du, eos, dt, dx, dy)
    rel_rhs!(du, u, eos, dx, dy)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, dx, dy)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────

function cfl_dt(u, eos, dx, dy, cfl)
    d = parent(u)
    amax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        amax = max(amax, max_wave_speed(eos, U, 1) / dx + max_wave_speed(eos, U, 2) / dy)
    end
    return cfl / max(amax, 1.0e-14)
end

function diagnostics(u, eos, dx, dy)
    d = parent(u)
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.E))
        U = conserved_cell(d, I)
        _, _, vx, vy = prim_from_cons(eos, U)

        if isnan(vx) || isnan(vy)
            # Print the culprits to find out exactly what numbers are breaking the solver
            @error "NaN detected at cell $I  Conserved values are: N=$(U[1]), Mx=$(U[2]), My=$(U[3]), E=$(U[4])"
            # Fallback to zero velocity so the diagnostic script doesn't crash completely
            vx = isnan(vx) ? 0.0 : vx
            vy = isnan(vy) ? 0.0 : vy
        end

        charge += U[1] * dx * dy
        energy += U[4] * dx * dy
        vmax    = max(vmax, sqrt(vx^2 + vy^2))
    end
    return charge, energy, vmax
end

function probe_n(u, eos, i, j)
    d = parent(u)
    h = halo_width(u.E)
    T, μ, _, _ = prim_from_cons(eos, conserved_cell(d, CartesianIndex(i + h, j + h)))
    return charge_density(T, μ, eos)
end


# ─── Driver: 2-D charge-carrying relativistic blast wave ──────────────────────
function run_2d_ideal_charge(eos, T_init, μ_init; cfl=0.3, t_end=0.20)
    # Check that input matrices are valid and square
    @assert size(T_init) == size(μ_init) "T_init and μ_init grids must have matching dimensions!"
    n = size(T_init, 1)
    @assert size(T_init, 2) == n "This solver assumes a square grid (n × n)!"

    dx = 1.0 / n;  dy = 1.0 / n

    # Define a thermodynamically stable, consistent atmosphere baseline
    # (Adjust these base constants if your simulation scale demands it)
   
    T_atmo = 1e-5
    μ_atmo = T_atmo

    # Allocate physical variables and Runge-Kutta workspaces
    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:N, :Mx, :My, :E), boundary_condition=:repeating)
    u1 = similar(u)
    du = similar(u)

    # Convert primitive variables to conserved variables and populate the grid
    for j in 1:n, i in 1:n
        T = T_init[i, j]
        μ = μ_init[i, j]
        
        # --- ATMOSPHERE INJECTION ---
        # If the input temperature represents an absolute vacuum or unphysical numerical noise,
        # reset the cell's primitives to our robust baseline before creating conserved states.
        if T <= 1e-5 || isnan(T) || isnan(μ)
            T = T_atmo
            μ = μ_atmo
        end
        # -----------------------------

        # Initial condition assumes zero initial fluid velocity (vx=0, vy=0)
        U = cons_from_prim(eos, T, μ, 0.0, 0.0)
        
        interior_view(u.N)[i, j]  = U[1]
        interior_view(u.Mx)[i, j] = U[2]
        interior_view(u.My)[i, j] = U[3]
        interior_view(u.E)[i, j]  = U[4]
    end
    synchronize_halo!(u)

    # Capture initial totals for checking physical conservation laws
    q0, e0, _ = diagnostics(u, eos, dx, dy)
    @printf("2-D Relativistic Hydro — Evolving Custom Initial Conditions\n")
    @printf("  grid=%d×%d  t_end=%.2f  initial charge=%.6f energy=%.6f\n", n, n, t_end, q0, e0)

    # Main hydrodynamic evolution loop
    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, dy, cfl), t_end - t)
        
        # If cfl_dt accidentally hits a NaN due to transient calculations, 
        # force a tiny step fallback rather than crashing completely
        if isnan(dt)
            @warn "CFL timestep calculation returned NaN at step $step. Forcing minimal fallback dt."
            dt = min(1e-4, t_end - t)
        end

        ssprk2_step!(u, u1, du, eos, dt, dx, dy)
        t += dt; step += 1
    end

    # Post-evolution metrics
    q1, e1, vmax = diagnostics(u, eos, dx, dy)
    @printf("  final    charge=%.6f (Δ=%.2e)  energy=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step)

    # Strict physical validation: checks conservation profiles and fluid flow
    charge_conserved = abs(q1 - q0) / q0 < 1e-3
    energy_conserved = abs(e1 - e0) / e0 < 1e-3
    fluid_moved = (vmax > 0.0) && !isnan(vmax)

    if charge_conserved && energy_conserved && fluid_moved
        println("  ✓ custom initial conditions: charge & energy conserved, evolution completed successfully")
    else
        println("  ✗ conservation laws violated, NaN detected, or fluid failed to evolve dynamically")
    end

    return u
end

run_2d_ideal_charge(eos,tmap, μmap)

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