using ForwardDiff
import Fluidum.Bessels # or import SpecialFunctions depending on what you use
using NonlinearSolve
import SimpleNonlinearSolve as SNLS
import StaticArrays as SA

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
    if SciMLBase.successful_retcode(sol.retcode) == false
        @warn "not converged"
        @show sol.retcode
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
    # Changed from prim_from_cons(eos, U) to your cons_to_primitive(U, eos)
    T, μ, vx, vy = cons_to_primitive(U, eos)
    p = fluid_pressure(T, eos)
    N, Mx, My, _ = U
    if dim == 1
        return SVector{4}(N * vx, Mx * vx + p, My * vx, Mx)
    else
        return SVector{4}(N * vy, Mx * vy, My * vy + p, My)
    end
end

@inline function max_wave_speed(eos, U, dim::Int)
    # Changed to match cons_to_primitive(U, eos)
    # Changed to match cons_to_primitive(U, eos)
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
    synchronize_halo!(u)                  
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 1), conserved_cell, add_conserved!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux(eos, UL, UR, 2), conserved_cell, add_conserved!)
    return du
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
function run_2d_viscous_charge(eos, T_init, μ_init; cfl=0.3, t_end=0.20)

    @assert size(T_init) == size(μ_init) "T_init and μ_init grids must have matching dimensions!"
    n = size(T_init, 1)
    @assert size(T_init, 2) == n "This solver assumes a square grid (n × n)!"

    dx = 1.0 / n;  dy = 1.0 / n
   
    T_atmo = 1e-5
    μ_atmo = T_atmo

    u  = LocalMultiHaloArray(Float64, (n, n), 1;
        fields=(:N, :Mx, :My, :E), boundary_condition=:repeating)
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

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt(u, eos, dx, dy, cfl), t_end - t)
        
        if isnan(dt)
            @warn "CFL timestep calculation returned NaN at step $step. Forcing minimal fallback dt."
            dt = min(1e-4, t_end - t)
        end

        ssprk2_step!(u, u1, du, eos, dt, dx, dy)
        t += dt; step += 1
    end

    q1, e1, vmax = diagnostics(u, eos, dx, dy)
    @printf("  final    charge=%.6f (Δ=%.2e)  energy=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step)

    @show charge_conserved = abs(q1 - q0) / q0 < 1e-3
    @show energy_conserved = abs(e1 - e0) / e0 < 1e-3
    @show vmax
    @show fluid_moved = (vmax > 0.0) && !isnan(vmax)

    if charge_conserved && energy_conserved && fluid_moved
        println("  ✓ custom initial conditions: charge & energy conserved, evolution completed successfully")
    else
        println("  ✗ conservation laws violated, NaN detected, or fluid failed to evolve dynamically")
    end

    return u
end

