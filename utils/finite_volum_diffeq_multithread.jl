
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
        @warn "cons to primitive is failing"
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
    
    # 1. Bring the halos up to date from the previous state interior
    synchronize_halo!(u)              
    
    # 2. Get the layout ranges and internal configurations
    fr    = FaceRanges(u)
    cells = get_owned_cells(CellRanges(u)) # Per-tile owned cells configuration

    # 3. Parallelize across all tiles using HaloArrays' thread backend
    tile_foreach(thread_backend(u.N), tile_id -> begin
        # Extract the local NamedTuple of arrays for this specific tile
        du_data = tile_parent(du, tile_id)     
        u_data  = tile_parent(u, tile_id)
        
        # Initialize this tile's derivative entries to zero before accumulation
        # (This replaces the global fill!(du, 0.0) safely for threads)
        for field in (:N, :Mx, :My, :E)
            fill!(getfield(du_data, field), 0.0)
        end
        
        # Accumulate X-flux divergence into this tile (-∂_x F_x)
        accumulate_flux_divergence!(du_data, u_data, fr, 1, inv(dx),
            (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
            
        # Accumulate Y-flux divergence into this tile (-∂_y F_y)
        accumulate_flux_divergence!(du_data, u_data, fr, 2, inv(dy),
            (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
            
    end, 1:tile_count(u); scheduler=:static)

    return nothing
end

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
   h = halo_width(u.N)
    cells = get_owned_cells(CellRanges(u))
    charge = 0.0; energy = 0.0; vmax = 0.0
    for tile_id in 1:tile_count(u)
        d = tile_parent(u, tile_id)
        @inbounds for I in cells
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
end
    return charge, energy, vmax
end

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
using Base.Threads: @threads
using Printf

using Printf
using Base.Threads: @threads

function run_2d_ideal_charge_diffeq_threaded(eos, T_init, μ_init; 
        ntiles_x=max(1, Threads.nthreads()), ntiles_y=1, cfl=0.3, t_end=0.20)

    @assert size(T_init) == size(μ_init) "T_init and μ_init grids must have matching dimensions!"
    nx, ny = size(T_init)
    
    nx % ntiles_x == 0 || error("nx=$nx must be divisible by ntiles_x=$ntiles_x")
    ny % ntiles_y == 0 || error("ny=$ny must be divisible by ntiles_y=$ntiles_y")

    dx = 1.0 / nx;  dy = 1.0 / ny
   
    # Allocate a Threaded multi-halo array split into tiles
    # The domain dimension size is the size per individual tile: (nx ÷ ntiles_x, ny ÷ ntiles_y)
    u = ThreadedMultiHaloArray(Float64, (nx ÷ ntiles_x, ny ÷ ntiles_y), 1; 
        dims=(ntiles_x, ntiles_y),
        fields=(:N, :Mx, :My, :E), 
        boundary_condition=:repeating)
   
    h = halo_width(u.N)
    
    # Run loop over tiles using the library's parallel layout patterns
    for tile_id in 1:tile_count(u)
        dN  = tile_parent(u.N,  tile_id)
        dMx = tile_parent(u.Mx, tile_id)
        dMy = tile_parent(u.My, tile_id)
        dE  = tile_parent(u.E,  tile_id)
        
        local_size = tile_size(u) # (tile_nx, tile_ny)
        
        for j in 1:local_size[2]
            for i in 1:local_size[1]
                # Map local tile indices (i, j) to global input grid indices (gi, gj)
                gi, gj = owned_to_global_index(u.N, tile_id, (i, j))
                
                T = T_init[gi, gj]
                μ = μ_init[gi, gj]
                
                # Conserved variables from primitives (assumes 2D requires v_x=0, v_y=0)
                U = prim_to_cons(T, μ, zero(T), zero(T), eos)
                
                # Store variables in the tile parent arrays accounting for halo offsets
                dN[i + h, j + h]  = U[1]
                dMx[i + h, j + h] = U[2]
                dMy[i + h, j + h] = U[3]
                dE[i + h, j + h]  = U[4]
            end
        end
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics(u, eos, dx, dy)
    @printf("2-D Relativistic Hydro — Evolving Custom Initial Conditions (Threaded)\n")
    @printf("  grid=%d×%d (tiles: %d×%d)  t_end=%.2f  initial charge=%.6f energy=%.6f\n", 
        nx, ny, ntiles_x, ntiles_y, t_end, q0, e0)

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


function oioi(u, eos)
    # 1. Get the full global size of the simulation grid (excluding halos)
   _, nx_global, ny_global = owned_size(u)
    
    # 2. Allocate global flat output matrices
    T_flat  = Matrix{Float64}(undef, nx_global, ny_global)
    μ_flat  = Matrix{Float64}(undef, nx_global, ny_global)
    vx_flat = Matrix{Float64}(undef, nx_global, ny_global)
    vy_flat = Matrix{Float64}(undef, nx_global, ny_global)
    
    @show range = interior_range(u.N)  #(2:end-1,2:end-1)
    @show h     = halo_width(u.N)
    
    @show tile_nx = length(range[1])
    tile_ny = length(range[2])
    @show tile_count(u)
    # 3. Process tiles in parallel
    @tasks for tile_id in 1:tile_count(u)
        u_data = tile_parent(u, tile_id) #with halo
        @show coords = tile_coordinates(u.N, tile_id)
        for oi in tile_size(u)[1]
        @show owned_to_global_index(u.N, tile_id, (oi,oi))[1]
        end
        global_start_i = (coords[1] - 1) * tile_nx
        global_start_j = (coords[2] - 1) * tile_ny
        
        for I in CartesianIndices(range)
            U = conserved_cell(u_data, I) 
            
            # Updated to your specific function signature
            T, μ, vx, vy = cons_to_primitive(U, eos)
            
            gi = global_start_i + (I[1] - h + 1)
            gj = global_start_j + (I[2] - h + 1)
            
            T_flat[gi, gj]  = T
            μ_flat[gi, gj]  = μ
            vx_flat[gi, gj] = vx
            vy_flat[gi, gj] = vy
        end
    end
    
    return T_flat, μ_flat, vx_flat, vy_flat
end
