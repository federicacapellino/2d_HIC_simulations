
using ForwardDiff
import Fluidum.Bessels # or import SpecialFunctions depending on what you use
using NonlinearSolve
import SimpleNonlinearSolve as SNLS
import StaticArrays as SA
using DifferentialEquations
import OrdinaryDiffEq
# 1. Intercept besselk1x at the top-level API
function Bessels.besselk1x(x::ForwardDiff.Dual{T, V, N}) where {T, V, N}
    val = ForwardDiff.value(x)
    
    # Evaluate using standard numbers to bypass internal _besselkx calls safely
    fx = Bessels.besselk1x(val)
    fk0 = Bessels.besselk0x(val)
    
    # Analytical derivative: d/dx [K_1(x) e^x]
    dfx = fx - fk0 - fx / val
    
    # Reconstruct the dual number via chain rule
    return ForwardDiff.Dual{T}(fx, dfx * ForwardDiff.partials(x))
end

# 2. Intercept besselk0x (since the derivative above relies on it!)
function Bessels.besselk0x(x::ForwardDiff.Dual{T, V, N}) where {T, V, N}
    val = ForwardDiff.value(x)
    
    fx = Bessels.besselk0x(val)
    fk1 = Bessels.besselk1x(val)
    
    # Analytical derivative: d/dx [K_0(x) e^x]
    dfx = fx - fk1
    
    return ForwardDiff.Dual{T}(fx, dfx * ForwardDiff.partials(x))
end

# Intercept the internal _besselkx function directly for any order nu
function Bessels._besselkx(nu, x::ForwardDiff.Dual{T, V, N}) where {T, V, N}
    # 1. Extract the raw Float64 value
    val = ForwardDiff.value(x)
    
    # 2. Compute the primal value using standard numbers
    fx = Bessels._besselkx(nu, val)
    
    # 3. Compute the analytical derivative: 
    # d/dx [K_ν(x) e^x] = K_ν(x)e^x - K_{ν-1}(x)e^x - (ν/x)K_ν(x)e^x
    # We use abs(nu - 1) because K_{-ν}(x) = K_ν(x)
    dfx = fx - Bessels._besselkx(abs(nu - 1), val) - (nu / val) * fx
    
    # 4. Reconstruct and return the dual number via the chain rule
    return ForwardDiff.Dual{T}(fx, dfx * ForwardDiff.partials(x))
end

# --- Conserved to Primitive Transformation ---
function cons_to_primitive_rhds(u, p)
    T, μ = u              
    α = μ/T
    E, M2, N, eos = p     # Unpack parameters including eos
    
    en = fluid_energy_density(T, eos)
    pr = fluid_pressure(T, eos)
    n  = charge_density(T, α, eos)
    
    q  = E + pr            # = (e+p)γ²
    W  = q^2
    
    fe = (en + pr) * q - (W - M2)
    fn = n * q - N * sqrt(W - M2)   
    
    return SA.SVector{2}(fe, fn)
end


# --- Primitive to Conserved Transformation ---
function prim_to_cons(T, μ, v_x, v_y, eos)
    α = μ/T
    en  = fluid_energy_density(T, eos)
    pr  = fluid_pressure(T, eos)
    n = charge_density(T, α, eos)
    
    w     = en + pr #enthalpy
    gamma = 1.0 / sqrt(1.0 - v_x^2 - v_y^2)
    w_gamma2 = w * gamma^2
    
    Mx = v_x * w_gamma2
    My = v_y * w_gamma2
    E  = w_gamma2 - pr
    N  = n * gamma
    
    return (N, Mx, My, E)
end 


function cons_to_primitive(U, eos)
    N, Mx, My, E = U
    
    # FIX: M2 must be squared to prevent W - M2 from becoming negative under the sqrt
    M2 = hypot(Mx, My)^2 
    
    # initial trial seeds for primitives
    T = 0.5             
    μ = 1.
    
    cons = (E, M2, N, eos) # Passed eos along into the parameter tuple
    u0 = SA.SVector{2}(T, μ) # Clean static vector for NonlinearSolve

    prob = NonlinearProblem(cons_to_primitive_rhds, u0, cons)
    sol = solve(prob, SNLS.SimpleNewtonRaphson())
    if SciMLBase.successful_retcode(sol.retcode) == false || T <= 1e-5 || isnan(T) || isnan(μ)
#        @warn "cons to primitive is failing"
        T_atmo = 1e-5
        return T_atmo, T_atmo, 0.0, 0.0
    end
    T, μ = sol
    pr = fluid_pressure(T, eos)
    W = E + pr
    
    vx = Mx / W
    vy = My / W
    return T, μ, vx, vy                        
end

# Verification with matching function names:
#N, Mx, My, E = prim_to_cons(0.1, 0.1, 0.1, 0.1, eos)

#cons_to_primitive((N, Mx, My, E), eos)

# ─── Direction-aware flux and signal speed ────────────────────────────────────


@inline function physical_flux(eos, U, dim::Int)
    T, μ, vx, vy = cons_to_primitive(U, eos)
    p = fluid_pressure(T, eos)
    N, Mx, My, _ = U
    if dim == 1
        return SVector{4}(N * vx, Mx * vx + p, My * vx, Mx)
    else
        return SVector{4}(N * vy, Mx * vy, My * vy + p, My)
    end
    #return SVector(U[1] * v, U[2] * v + p, U[2])
end

@inline function max_wave_speed(eos, U, dim::Int)
    _, _, vx, vy = cons_to_primitive(U, eos)
    v2  = vx^2 + vy^2
    vn  = dim == 1 ? vx : vy
    cs2 = sound_speed_sq(eos)
    disc = (1.0 - v2) * ((1.0 - v2 * cs2) - vn^2 * (1.0 - cs2))
    root = sqrt(max(disc, 0.0)) * sqrt(cs2)
    den  = 1.0 - v2 * cs2
    return max(abs((vn * (1.0 - cs2) + root) / den), abs((vn * (1.0 - cs2) - root) / den))
end


@inline function rusanov_flux(eos, UL, UR, dim::Int)
    smax = max(max_wave_speed(eos, UL, dim), max_wave_speed(eos, UR, dim))
    return 0.5 * (physical_flux(eos, UL, dim) + physical_flux(eos, UR, dim)) - 0.5 * smax * (UR - UL)
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
function rhs!(du, u, params, t)
    eos, dx, dy = params
    fill!(du, 0.0)
    synchronize_halo!(u)              # :repeating fills the outflow ghosts
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
    return nothing
end
function rel_rhs!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)                  
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
    return du
end
#=
function rhs!(du, u, params, t)
    eos, dx, dy = params
    fill!(du, 0.0)
    
    # 1. Bring the halos up to date from the previous state interior
    synchronize_halo!(u)              
    
    # 2. Accumulate fluxes into parent(du)
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
        
    # 3. CRITICAL FIX FOR DifferentialEquations.jl:
    # Zero out the halos in the derivative array `du` so that the ODE solver 
    # never propagates ghost-cell residuals into the internal RK stages.
    # We find the full parent arrays and wipe out the non-interior rows/columns.
    
    h = halo_width(u.E)
    n = size(interior_view(u.E), 1)
    
    # Identify ghost index sets (assuming a 1-cell halo wrapper at indices 1 and n+2)
    # Clear out rows and columns belonging to the halo zone
    for fields in (:N, :Mx, :My, :E)
        p_du = getfield(parent(du), fields)
        
        # Zero out left/right ghost columns
        p_du[1:h, :] .= 0.0
        p_du[(h + n + 1):end, :] .= 0.0
        
        # Zero out top/bottom ghost rows
        p_du[:, 1:h] .= 0.0
        p_du[:, (h + n + 1):end] .= 0.0
    end

    return nothing
end
=#
function ssprk2_step!(u, u1, du, eos, dt, dx, dy)
    rel_rhs!(du, u, eos, dx, dy)
    @. u1 = u + dt * du
    rel_rhs!(du, u1, eos, dx, dy)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

#=function ssprk2_step!(u, u1, du, eos, dt, dx, dy)
    params = (eos, dx, dy)
    
    # --- STAGE 1 ---
    rhs!(du, u, params, 0.0)
    
    # Cleanly update interior of u1 without corrupting halos via broadcasting
    interior_view(u1.N)  .= interior_view(u.N)  .+ dt .* interior_view(du.N)
    interior_view(u1.Mx) .= interior_view(u.Mx) .+ dt .* interior_view(du.Mx)
    interior_view(u1.My) .= interior_view(u.My) .+ dt .* interior_view(du.My)
    interior_view(u1.E)  .= interior_view(u.E)  .+ dt .* interior_view(du.E)
    synchronize_halo!(u1) # Keep stage 1 bounds isolated
    
    # --- STAGE 2 ---
    # Create a clean temporary array for the second stage derivative
    du2 = similar(du) 
    rhs!(du2, u1, params, 0.0)
    
    # --- FINAL BLENDING ---
    # u_new = 0.5 * u + 0.5 * u1 + 0.5 * dt * du2
    interior_view(u.N)  .= 0.5 .* interior_view(u.N)  .+ 0.5 .* interior_view(u1.N)  .+ (0.5 * dt) .* interior_view(du2.N)
    interior_view(u.Mx) .= 0.5 .* interior_view(u.Mx) .+ 0.5 .* interior_view(u1.Mx) .+ (0.5 * dt) .* interior_view(du2.Mx)
    interior_view(u.My) .= 0.5 .* interior_view(u.My) .+ 0.5 .* interior_view(u1.My) .+ (0.5 * dt) .* interior_view(du2.My)
    interior_view(u.E)  .= 0.5 .* interior_view(u.E)  .+ 0.5 .* interior_view(u1.E)  .+ (0.5 * dt) .* interior_view(du2.E)
    
    synchronize_halo!(u)
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
        _, _, vx, vy = cons_to_primitive(U, eos)

        if isnan(vx) || isnan(vy)
            @error "NaN detected at cell $I. Conserved values are: N=$(U[1]), Mx=$(U[2]), My=$(U[3]), E=$(U[4])"
            vx = isnan(vx) ? 0.0 : vx
            vy = isnan(vy) ? 0.0 : vy
        end

        charge += U[1] * dx * dy
        energy += U[4] * dx * dy
        vmax    = max(vmax, sqrt(vx^2 + vy^2))
    end
    return charge, energy, vmax
end=#

function probe_n(u, eos, i, j)
    d = parent(u)
    h = halo_width(u.E)
    U = conserved_cell(d, CartesianIndex(i + h, j + h))
    T, μ, _, _ = cons_to_primitive(U, eos)
    
    # FIX: Your charge_density(T, α, eos) expects α = μ/T, not raw μ
    return charge_density(T, μ / T, eos)
end

# ─── Driver: 2-D charge-carrying relativistic blast wave ──────────────────────

# ─── Driver: 2-D charge-carrying relativistic blast wave ──────────────────────
function run_2d_ideal_charge_diffeq(eos, T_init, μ_init; cfl=0.3, t_end=0.20)


    @assert size(T_init) == size(μ_init) "T_init and μ_init grids must have matching dimensions!"
    n = size(T_init, 1)
    @assert size(T_init, 2) == n "This solver assumes a square grid (n × n)!"

    dx = 1.0 / n;  dy = 1.0 / n
   
    T_atmo = 1e-5
    μ_atmo = T_atmo

    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:N, :Mx, :My, :E), boundary_condition=:repeating)
   
    for j in 1:n, i in 1:n
        T = T_init[i, j]
        μ = μ_init[i, j]
        
    #    if T <= 1e-5 || isnan(T) || isnan(μ)
    #        T = T_atmo
    #        μ = μ_atmo
    #    end

        U = prim_to_cons(T, μ, zero(T), zero(T), eos)
        
        interior_view(u.N)[i, j]  = U[1]
        interior_view(u.Mx)[i, j] = U[2]
        interior_view(u.My)[i, j] = U[3]
        interior_view(u.E)[i, j]  = U[4]
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, dx, dy)
    @printf("2-D Relativistic Hydro — Evolving Custom Initial Conditions\n")
    @printf("  grid=%d×%d  t_end=%.2f  initial charge=%.6f energy=%.6f\n", n, n, t_end, q0, e0)

    dt   = cfl * dx
    prob = ODEProblem{true}(rhs!, u, (0.0, t_end), (eos, dx, dy))
    sol  = solve(prob, OrdinaryDiffEqSSPRK.SSPRK33(); dt=dt, adaptive=false, save_everystep=false)

    u = sol.u[end]
    synchronize_halo!(u)

    q1, e1, vmax = diagnostics(u, eos, dx, dy)
    @printf("  final    charge=%.6f (Δ=%.2e)  energy=%.6f (Δ=%.2e)  vmax=%.4f \n",
        q1, q1 - q0, e1, e1 - e0, vmax )

    @show charge_conserved = abs(q1 - q0) / q0 < 1e-6
    @show energy_conserved = abs(e1 - e0) / e0 < 1e-6
    @show vmax
    @show fluid_moved = (vmax > 0.0) && !isnan(vmax)

    if charge_conserved && energy_conserved && fluid_moved
        println("  ✓ custom initial conditions: charge & energy conserved, evolution completed successfully")
    else
        println("  ✗ conservation laws violated, NaN detected, or fluid failed to evolve dynamically")
    end

    return u
end

"""
    get_primitive_variables(u, eos)

Takes the final `LocalMultiHaloArray` output `u` from your simulation, 
runs your `cons_to_primitive` inversion over the active interior grid,
and returns a NamedTuple containing 2D matrices for `T`, `μ`, `vx`, and `vy`.
"""
function get_primitive_variables(u, eos)
    # 1. Access the interior views of your conserved fields (ignoring halos)
    dN  = interior_view(u.N)
    dMx = interior_view(u.Mx)
    dMy = interior_view(u.My)
    dE  = interior_view(u.E)

    # 2. Allocate output matrices matching the physical grid dimensions


    # 3. Fast column-major iteration loop over the physical domain
    @inbounds for j in 1:nx
        for i in 1:ny
            # Construct the 4-element conserved vector for this cell
            U_cell = SVector{4, Float64}(dN[i, j], dMx[i, j], dMy[i, j], dE[i, j])
            
            # Execute your primitive variable recovery logic
            T, μ, vx, vy = cons_to_primitive(U_cell, eos)
            
            # Write out directly to our result matrices
            T_grid[i, j]  = T
            μ_grid[i, j]  = μ
            vx_grid[i, j] = vx
            vy_grid[i, j] = vy
        end
    end

    # Return as a clean, easily accessible NamedTuple
    return (T = T_grid, μ = μ_grid, vx = vx_grid, vy = vy_grid)
end

#=
function run_2d_ideal_charge_diffeq(eos, T_init, μ_init; cfl=0.3, t_end=0.20)
    
    n = size(T_init, 1)
    dx = 1.0 / n;  dy = 1.0 / n
   
    T_atmo = 1e-5
    μ_atmo = T_atmo

    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:N, :Mx, :My, :E), boundary_condition=:periodic)
    u1 = similar(u)
    du = similar(u)

    for j in 1:n, i in 1:n
        T = T_init[i, j]
        μ = μ_init[i, j]
        
        if T <= 1e-5 || isnan(T) || isnan(μ)
            T = T_atmo
            μ = μ_atmo
        end

        # FIX: Changed from cons_from_prim(eos, T, μ, 0.0, 0.0) to your exact definition format
        U = prim_to_cons(T, μ, zero(T), zero(T), eos)
        
        interior_view(u.N)[i, j]  = U[1]
        interior_view(u.Mx)[i, j] = U[2]
        interior_view(u.My)[i, j] = U[3]
        interior_view(u.E)[i, j]  = U[4]
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, dx, dy)
    @printf("2-D Relativistic Hydro — Evolving Custom Initial Conditions\n")
    @printf("  grid=%d×%d  t_end=%.2f  initial charge=%.6f energy=%.6f\n", n, n, t_end, q0, e0)

    # Calculate initial dt using your CFL function
    dt = cfl_dt(u, eos, dx, dy, cfl)
    
    println("  -> Starting manual SSP-RK2 explicit loop...")
    t = 0.0
    step_count = 0
    
step_count = 0
while t < t_end
    dt_step = min(dt, t_end - t)
    u = ssprk2_step!(u, u1, du, eos, dt_step, dx, dy)
    
    t += dt_step
    step_count += 1 # Increments correctly every iteration
    
    # Print progress every 10 steps to see immediate trends
    if step_count % 10 == 0
        q_current, e_current, v_current = diagnostics(u, eos, dx, dy)
        @printf("    Step %4d | t = %.4f | Charge Δ = %+.2e | Energy Δ = %+.2e\n", 
            step_count, t, q_current - q0, e_current - e0)
    end
    dt = cfl_dt(u, eos, dx, dy, cfl)
end
    # --- Final Evaluation ---
    synchronize_halo!(u)
    q1, e1, vmax = diagnostics(u, eos, dx, dy)
    
    @printf("\n  final    charge=%.6f (Δ=%.2e)  energy=%.6f (Δ=%.2e)  vmax=%.4f \n",
        q1, q1 - q0, e1, e1 - e0, vmax)

    charge_conserved = abs(q1 - q0) / q0 < 1e-11 # Tighten threshold to machine precision!
    energy_conserved = abs(e1 - e0) / e0 < 1e-11
    @show charge_conserved
    @show energy_conserved
    @show vmax

    if charge_conserved && energy_conserved
        println("  ✓ SUCCESS: System is perfectly closed. Conservation laws held down to machine precision!")
    else
        println("  ✗ FAILURE: Conservation laws violated.")
    end

    return u
end
=#
#=

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

=#

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

