using StaticArrays: SVector
import StaticArrays as SA
using NonlinearSolve
import SimpleNonlinearSolve as SNLS
using LinearAlgebra: opnorm
using Printf
using Test

# ==============================================================================
# 1. Equation of State Setup
# ==============================================================================
struct UltraRelGas
    A::Float64        # degeneracy g/π²
end

@inline _exp_muT(μ, T) = exp(clamp(μ / T, -700.0, 700.0))

@inline pressure(eos::UltraRelGas, T, μ)         = eos.A * T^4 * _exp_muT(μ, T)
@inline charge_density(eos::UltraRelGas, T, μ)   = eos.A * T^3 * _exp_muT(μ, T)
@inline enthalpy_density(eos::UltraRelGas, T, μ) = 4.0 * pressure(eos, T, μ)

@inline bar_p(eos, τ, a)   = pressure(eos, exp(τ), a * exp(τ))
@inline bar_nB(eos, τ, a)  = charge_density(eos, exp(τ), a * exp(τ))
@inline bar_eps(eos, τ, a) = enthalpy_density(eos, exp(τ), a * exp(τ)) - bar_p(eos, τ, a)

# ==============================================================================
# 2. Forward Hydrodynamics Transformation (Primitives -> Conserved)
# ==============================================================================
# General viscous, diffusive case. Returns (C0,Cx,Cy,Cn), matching init_state's field order
# below: C0 = √-g T^{0τ}, Cx = √-g T^{0x}, Cy = √-g T^{0y}, Cn = √-g N^0.
@inline function cons_from_prim(eos, qx, qy, τ, a, bulk_Pi, pixx, pixy, piyy, nux, nuy)
    q2 = qx^2 + qy^2
    W  = sqrt(1.0 + q2)
    vx = qx / W
    vy = qy / W

    p   = bar_p(eos, τ, a)
    n_B = bar_nB(eos, τ, a)
    eps = bar_eps(eos, τ, a)

    # Total pressure includes bulk viscous pressure: P_total = p + Π
    P_tot = p + bulk_Pi
    w_tot = eps + P_tot

    # Cn = n_B*W + ν^0 where ν^0 = v_i * ν^i
    nu0 = vx * nux + vy * nuy
    Cn  = n_B * W + nu0

    # Cx,Cy = [ε + P_total] * W^2 * v_i + π_ij * v^j
    Cx = w_tot * W^2 * vx + (pixx * vx + pixy * vy)
    Cy = w_tot * W^2 * vy + (pixy * vx + piyy * vy)

    # C0 = [ε + P_total] * W^2 - P_total + v_i * v_j * π^ij
    v_v_pi = vx^2 * pixx + 2.0 * vx * vy * pixy + vy^2 * piyy
    C0     = w_tot * W^2 - P_tot + v_v_pi

    return SVector(C0, Cx, Cy, Cn)
end

# Ideal, non-diffusive fallback: Π = π^ij = ν^i = 0.
@inline cons_from_prim_ideal(eos, qx, qy, τ, a) =
    cons_from_prim(eos, qx, qy, τ, a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# ==============================================================================
# 3. Backward Inversion (Conserved -> Primitives Nonlinear 4D Core)
# ==============================================================================
function cons_to_primitive_rhds(u, p_tuple)
    qx, qy, τ, a = u
    C0, Cx, Cy, Cn, bulk_Pi, pixx, pixy, piyy, nux, nuy, eos = p_tuple

    q2 = qx^2 + qy^2
    W  = sqrt(1.0 + q2)

    pr = bar_p(eos, τ, a)
    en = bar_eps(eos, τ, a)
    nB = bar_nB(eos, τ, a)

    q_q_pi = qx^2 * pixx + 2.0 * qx * qy * pixy + qy^2 * piyy
    Q_q    = q_q_pi / (1.0 + q2)

    S_q  = Cx * qx + Cy * qy
    q_nu = qx * nux + qy * nuy

    # Core factor for F_i: [C0 + p + Π - Q(q)]
    core_factor = C0 + pr + bulk_Pi - Q_q

    # F_i Residuals
    fx = (1.0 / W) * (core_factor * qx + (pixx * qx + pixy * qy)) - Cx
    fy = (1.0 / W) * (core_factor * qy + (pixy * qx + piyy * qy)) - Cy

    # F_E Residual
    fe = en - C0 + (S_q / W)

    # F_D Residual
    fn = nB - (Cn / W) + (q_nu / (1.0 + q2))

    return SA.SVector{4}(fx, fy, fe, fn)
end

# ==============================================================================
# 4. Backward Inversion Wrapper & Property Reconstruction
# ==============================================================================
function cons_to_primitive(U, DISSP, eos)
    C0, Cx, Cy, Cn = U
    bulk_Pi, pixx, pixy, piyy, nux, nuy = DISSP

    u0 = SA.SVector{4}(0.0, 0.0, log(0.5), 1.0 / 0.5)
    p_tuple = (C0, Cx, Cy, Cn, bulk_Pi, pixx, pixy, piyy, nux, nuy, eos)

    prob = NonlinearProblem(cons_to_primitive_rhds, u0, p_tuple)
    sol = solve(prob, SNLS.SimpleNewtonRaphson())
    #if too slow, implement the gradients to update the initial guess

    if SciMLBase.successful_retcode(sol.retcode) == false
        @warn "Primitive inversion did not converge"
        @show sol.retcode
    end

    qx, qy, τ, a = sol

    q2 = qx^2 + qy^2
    W  = sqrt(1.0 + q2)
    vx = qx / W
    vy = qy / W
    T  = exp(τ)
    μ  = a * T

    p_eq   = bar_p(eos, τ, a)
    n_eq   = bar_nB(eos, τ, a)
    eps_eq = bar_eps(eos, τ, a)

    q_pi_x = qx * pixx + qy * pixy
    q_pi_y = qx * pixy + qy * piyy
    q_q_pi = qx^2 * pixx + 2.0 * qx * qy * pixy + qy^2 * piyy
    q_nu   = qx * nux + qy * nuy

    pi0x = q_pi_x / W
    pi0y = q_pi_y / W
    pi00 = q_q_pi / (W^2)
    nu0  = q_nu / W

    return qx, qy, τ, a, W, vx, vy, T, μ, p_eq, eps_eq, n_eq, pi0x, pi0y, pi00, nu0
end

# Ideal, non-diffusive fallback: same core solve with DISSP = 0, dissipative time-components
# (pi0x,pi0y,pi00,nu0) dropped from the return since they don't exist for an ideal fluid.
function cons_to_primitive_ideal(U, eos)
    qx, qy, τ, a, W, vx, vy, T, μ, p_eq, eps_eq, n_eq, _, _, _, _ =
        cons_to_primitive(U, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), eos)
    return qx, qy, τ, a, W, vx, vy, T, μ, p_eq, eps_eq, n_eq
end

# ==============================================================================
# 5. Automated Consistency Verification Test Suite
# ==============================================================================
function run_variable_consistency_test()
    eos = UltraRelGas(1.0)

    @testset "Bounded Primitive Inversion (2D+1 with Diffusion)" begin
        qx_ref = 0.45
        qy_ref = -0.25
        τ_ref  = log(0.38)
        a_ref  = 0.12 / 0.38

        bulk_Pi_ref = 0.005
        pixx_ref    = 0.015
        pixy_ref    = -0.003
        piyy_ref    = 0.022
        nux_ref     = 0.008
        nuy_ref     = -0.004
        DISSP = (bulk_Pi_ref, pixx_ref, pixy_ref, piyy_ref, nux_ref, nuy_ref)

        q2_ref  = qx_ref^2 + qy_ref^2
        W_ref   = sqrt(1.0 + q2_ref)
        vx_ref  = qx_ref / W_ref
        vy_ref  = qy_ref / W_ref
        T_ref   = exp(τ_ref)
        μ_ref   = a_ref * T_ref

        p_ref   = bar_p(eos, τ_ref, a_ref)
        n_ref   = bar_nB(eos, τ_ref, a_ref)
        eps_ref = bar_eps(eos, τ_ref, a_ref)

        pi0x_ref = (qx_ref * pixx_ref + qy_ref * pixy_ref) / W_ref
        pi0y_ref = (qx_ref * pixy_ref + qy_ref * piyy_ref) / W_ref
        pi00_ref = (qx_ref^2 * pixx_ref + 2.0 * qx_ref * qy_ref * pixy_ref + qy_ref^2 * piyy_ref) / (W_ref^2)
        nu0_ref  = (qx_ref * nux_ref + qy_ref * nuy_ref) / W_ref

        # Forward Pass
        U_conserved = cons_from_prim(
            eos, qx_ref, qy_ref, τ_ref, a_ref,
            bulk_Pi_ref, pixx_ref, pixy_ref, piyy_ref, nux_ref, nuy_ref
        )

        # Backward Pass
        qx_rec, qy_rec, τ_rec, a_rec, W_rec, vx_rec, vy_rec, T_rec, μ_rec,
        p_rec, eps_rec, n_rec, pi0x_rec, pi0y_rec, pi00_rec, nu0_rec =
            cons_to_primitive(U_conserved, DISSP, eos)

        @testset "Core Variables Inversion" begin
            @test qx_rec ≈ qx_ref atol=1e-8
            @test qy_rec ≈ qy_ref atol=1e-8
            @test τ_rec  ≈ τ_ref  atol=1e-8
            @test a_rec  ≈ a_ref  atol=1e-8
        end

        @testset "Standard Primitives Mapping" begin
            @test W_rec  ≈ W_ref  atol=1e-8
            @test vx_rec ≈ vx_ref atol=1e-8
            @test vy_rec ≈ vy_ref atol=1e-8
            @test T_rec  ≈ T_ref  atol=1e-8
            @test μ_rec  ≈ μ_ref  atol=1e-8
        end

        @testset "Reconstructed Thermodynamics & Dissipative Time Components" begin
            @test p_rec    ≈ p_ref    atol=1e-8
            @test eps_rec  ≈ eps_ref  atol=1e-8
            @test n_rec    ≈ n_ref    atol=1e-8
            @test pi0x_rec ≈ pi0x_ref atol=1e-8
            @test pi0y_rec ≈ pi0y_ref atol=1e-8
            @test pi00_rec ≈ pi00_ref atol=1e-8
            @test nu0_rec  ≈ nu0_ref  atol=1e-8
        end
    end
end

function run_ideal_variable_consistency_test()
    eos = UltraRelGas(1.0)

    @testset "Bounded Primitive Inversion (2D+1, ideal non-diffusive)" begin
        qx_ref = 0.45
        qy_ref = -0.25
        τ_ref  = log(0.38)
        a_ref  = 0.12 / 0.38

        q2_ref  = qx_ref^2 + qy_ref^2
        W_ref   = sqrt(1.0 + q2_ref)
        vx_ref  = qx_ref / W_ref
        vy_ref  = qy_ref / W_ref
        T_ref   = exp(τ_ref)
        μ_ref   = a_ref * T_ref

        p_ref   = bar_p(eos, τ_ref, a_ref)
        n_ref   = bar_nB(eos, τ_ref, a_ref)
        eps_ref = bar_eps(eos, τ_ref, a_ref)

        # Forward Pass
        U_conserved = cons_from_prim_ideal(eos, qx_ref, qy_ref, τ_ref, a_ref)

        # Backward Pass
        qx_rec, qy_rec, τ_rec, a_rec, W_rec, vx_rec, vy_rec, T_rec, μ_rec,
        p_rec, eps_rec, n_rec = cons_to_primitive_ideal(U_conserved, eos)

        @testset "Core Variables Inversion" begin
            @test qx_rec ≈ qx_ref atol=1e-8
            @test qy_rec ≈ qy_ref atol=1e-8
            @test τ_rec  ≈ τ_ref  atol=1e-8
            @test a_rec  ≈ a_ref  atol=1e-8
        end

        @testset "Standard Primitives Mapping" begin
            @test W_rec  ≈ W_ref  atol=1e-8
            @test vx_rec ≈ vx_ref atol=1e-8
            @test vy_rec ≈ vy_ref atol=1e-8
            @test T_rec  ≈ T_ref  atol=1e-8
            @test μ_rec  ≈ μ_ref  atol=1e-8
        end

        @testset "Reconstructed Thermodynamics" begin
            @test p_rec   ≈ p_ref   atol=1e-8
            @test eps_rec ≈ eps_ref atol=1e-8
            @test n_rec   ≈ n_ref   atol=1e-8
        end
    end
end

run_variable_consistency_test()
run_ideal_variable_consistency_test()

# ─── Grid storage (Sec. VIII split: C-sector C0,Cx,Cy,Cn / W-sector dissipative) ──
#
# C-sector (C0,Cx,Cy,Cn — energy-momentum + charge conserved densities, √-g folded in) and
# W-sector (bulk,pixx,pixy,piyy,nux,nuy — dissipative, path-conservative per Sec VIII A-C)
# share one LocalMultiHaloArray when viscous=true. LocalMultiHaloArray only requires fields to
# share geometry/halo (not update rule), so flux-divergence-updated (C) and locally-updated
# (W) fields can coexist — precedented by HaloArrays' own relativistic_hydro_cylindrical_1d.jl
# example, which interleaves accumulate_flux_divergence! with a plain per-cell local source
# loop on the same array. viscous=false gives the ideal, non-diffusive fallback grid with only
# the C-sector fields.
#
# boundary_condition=:repeating is a placeholder: qx/pixy/nux are odd under x-reflection (and
# qy/pixy/nuy odd under y-reflection), matching the parity convention already used for
# Fluidum's NDField((:odd,...)) fields — per-field, per-dimension boundary_conditions=(...)
# should replace this once the boundary-condition step is addressed.
const C_FIELDS = (:C0, :Cx, :Cy, :Cn)
const W_FIELDS = (:bulk, :pixx, :pixy, :piyy, :nux, :nuy)

function init_state(n::Integer; halo::Integer=1, viscous::Bool=true)
    fields = viscous ? (C_FIELDS..., W_FIELDS...) : C_FIELDS
    return LocalMultiHaloArray(Float64, (n, n), halo; fields=fields, boundary_condition=:repeating)
end

init_dnmr_state(n::Integer; halo::Integer=1)  = init_state(n; halo=halo, viscous=true)
init_ideal_state(n::Integer; halo::Integer=1) = init_state(n; halo=halo, viscous=false)

# ─── Primitive recovery — shared step, threaded through the whole RHS ────────
# Bundles cons_to_primitive's long return tuple so it's recovered once per cell and passed
# around, instead of every downstream flux/source stub re-solving the same Newton problem.
struct PrimitiveState
    qx::Float64; qy::Float64; τ::Float64; a::Float64
    Wlorentz::Float64; vx::Float64; vy::Float64
    Temp::Float64; μ::Float64
    p::Float64; eps::Float64; n::Float64
    pi0x::Float64; pi0y::Float64; pi00::Float64; nu0::Float64
end

@inline function recover_primitives(C, DISSP, eos)
    qx, qy, τ, a, Wl, vx, vy, Temp, μ, p, eps, n, pi0x, pi0y, pi00, nu0 =
        cons_to_primitive(C, DISSP, eos)
    return PrimitiveState(qx, qy, τ, a, Wl, vx, vy, Temp, μ, p, eps, n, pi0x, pi0y, pi00, nu0)
end

# A "full cell state" S = (C, prim, W): raw C-sector conserved densities, cached recovered
# primitives, raw W-sector dissipative fields — everything a flux/source formula could need,
# computed once and threaded through rather than re-derived inside each call. read_W is
# dissipative_cell_W on the viscous grid, or a zero-fill on the ideal grid (no W-sector
# fields to read at all) — see build_primitive_cache, which decides which one applies.
const ZERO_W_STATE = SA.SVector{6}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
@inline zero_dissipative_cell_W(d, I) = ZERO_W_STATE

@inline full_cell(d, prim_cache, I, read_W) =
    (conserved_cell_C(d, I), prim_cache[I], read_W(d, I))

# Recovers primitives for every cell in the padded (ghost-inclusive) grid, once per RHS
# evaluation — needed at ghost cells too, since domain-boundary faces read from them. Uses
# field_storages(u), NOT parent(u): interior_faces/interior_range (used throughout this file
# by accumulate_flux_divergence!/accumulate_path_conservative_W!/cfl_dt_full/diagnostics_full)
# hand out storage-space (ghost-inclusive) CartesianIndices, while parent(u)'s per-field
# arrays are sized/indexed logically (1:n) and throw a BoundsError on those — field_storages(u)
# is the padded (n+2halo)-sized accessor that matches. Falls back to zero DISSP when u has no
# W-sector fields (the ideal grid), so this one function serves both init_ideal_state and
# init_dnmr_state grids. Also returns has_W so callers can build a matching full_cell reader
# without re-checking propertynames themselves.
function build_primitive_cache(u, eos)
    d = field_storages(u)
    has_W = :bulk in propertynames(u)
    prim_cache = Array{PrimitiveState}(undef, size(d.C0))
    for I in CartesianIndices(d.C0)
        C = conserved_cell_C(d, I)
        Wdiss = has_W ? dissipative_cell_W(d, I) : ZERO_W_STATE
        prim_cache[I] = recover_primitives(C, Wdiss, eos)
    end
    return prim_cache, has_W
end

# ─── C-sector physical flux & wave speed — equations supplied later ──────────
# S = (C, prim, W) at one cell/state, C=(C0,Cx,Cy,Cn). dim: 1 = x-flux, 2 = y-flux. Kept as
# named stubs (rather than an inlined flat-space formula) because C0,Cx,Cy,Cn fold in a √-g
# metric factor, so the actual flux/wave-speed formulas need the geometric factors/source
# terms supplied separately.
function physical_flux_C(eos, S, dim::Int)
    error("physical_flux_C: equations not yet supplied")
end

# flux_jacobian_C(eos, S, dim) -> 4x4 matrix, A(U) = ∂F_C/∂C. Entries supplied later.
function flux_jacobian_C(eos, S, dim::Int)
    error("flux_jacobian_C: equations not yet supplied")
end

# max_wave_speed_C: Sec VIII E, Eq. 104 — α(U) ≃ sqrt(‖A(U)‖_1 ‖A(U)‖_∞), a cheap upper bound
# on the spectral radius via induced matrix norms (opnorm(A,1) = max column-abs-sum,
# opnorm(A,Inf) = max row-abs-sum), avoiding an eigenvalue solve at every interface. Mechanical
# once flux_jacobian_C is supplied — this function itself needs no further equations.
@inline function max_wave_speed_C(eos, S, dim::Int)
    A = flux_jacobian_C(eos, S, dim)
    return sqrt(opnorm(A, 1) * opnorm(A, Inf))
end

# ─── Rusanov flux (Eq. 66) — fully mechanical, no physics assumptions ────────
# smax follows Eq. 104 literally: αLR ≃ sqrt(‖A(U*)‖_1 ‖A(U*)‖_∞) at the interface-averaged
# state U* = (U_L+U_R)/2, not max(α(U_L), α(U_R)) — the two only coincide when α is linear in U.
@inline function rusanov_flux_C(eos, SL, SR, dim::Int)
    CL, CR = SL[1], SR[1]
    WL, WR = SL[3], SR[3]
    C_mid = 0.5 * (CL + CR)
    W_mid = 0.5 * (WL + WR)
    S_mid = (C_mid, recover_primitives(C_mid, W_mid, eos), W_mid)
    smax = max_wave_speed_C(eos, S_mid, dim)
    return 0.5 * (physical_flux_C(eos, SL, dim) + physical_flux_C(eos, SR, dim)) -
           0.5 * smax * (CR - CL)
end

# ─── Field access (C-sector only — untouched W-sector fields, if present on the same
# LocalMultiHaloArray, are simply never referenced here) ──────────────────────
@inline conserved_cell_C(d, I) = SVector(d.C0[I], d.Cx[I], d.Cy[I], d.Cn[I])

@inline function add_conserved_C!(d, I, scale, U)
    d.C0[I] += scale * U[1]
    d.Cx[I] += scale * U[2]
    d.Cy[I] += scale * U[3]
    d.Cn[I] += scale * U[4]
    return d
end

# ─── RHS (x then y sweep) — C-sector only, isolated single-sector testing ────
# Primitives are always recovered assuming zero DISSP here (build_primitive_cache falls back
# to that when u has no W-sector fields, but even called against init_dnmr_state's 10-field
# grid this stays a pure C-sector-only evolution — use rhs_full! for the true coupled system.
function rhs_C!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    d = field_storages(u)
    prim_cache, has_W = build_primitive_cache(u, eos)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    read_full = (data, I) -> full_cell(data, prim_cache, I, read_W_fn)

    accumulate_flux_divergence!(field_storages(du), d, fr, 1, inv(dx),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 1), read_full, add_conserved_C!)
    accumulate_flux_divergence!(field_storages(du), d, fr, 2, inv(dy),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 2), read_full, add_conserved_C!)
    return du
end

# ─── SSP-RK2 time step ────────────────────────────────────────────────────────
function ssprk2_step_C!(u, u1, du, eos, dt, dx, dy)
    rhs_C!(du, u, eos, dx, dy)
    @. u1 = u + dt * du
    rhs_C!(du, u1, eos, dx, dy)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── W-sector path-conservative update (Eqs. 99, 100, 102) — equations later ─
#
# Sec VIII C keeps the two sectors structurally separate: the C-sector is a genuine balance
# law (already handled above by rusanov_flux_C, no B-matrix), while the W-sector is purely
# non-conservative, ∂_tW^a + B^{x,a}_B(U)∂_xU^B = S^a_W(U) (Eq. 101) — no flux term at all, so
# unlike the generic Eq. 99 there's no F(U_R)-F(U_L) piece here, only the path integral.

# path_integrand_W(eos, S_mid, ΔC, ΔW, dim) -> SVector{6}
#   Straight-line, midpoint-quadrature approximation of Eq. 99's path integral
#   N^a_LR = ∫_0^1 B^{x,a}_B(Φ(s)) ∂_sΦ^B ds ≈ B^{x,a}_B((U_L+U_R)/2) · ΔU^B, restricted to the
#   W-sector: combines both the B_WC piece (C-gradients coupling in through the reconstructed
#   u^μ, e.g. θ,σ^μν,ω^μν) and the B_WW piece (self-coupling of W-gradients) in one call.
#   S_mid = full state (C,prim,W) recovered fresh at the face midpoint Φ(s=1/2); ΔC,ΔW =
#   right-minus-left jumps. dim: 1=x, 2=y.
function path_integrand_W(eos, S_mid, ΔC, ΔW, dim::Int)
    error("path_integrand_W: equations not yet supplied")
end

# mixed_flux_jacobian(eos, S, dim) -> 10x10 matrix, A_eff (Eq. 105's block form:
#   [[∂F_C/∂C, ∂F_C/∂W], [H∂q/∂C, G+H∂q/∂W]]) for the combined (C,W) state. Kept distinct from
#   flux_jacobian_C since it must account for the B-matrix coupling, not just the C-sector flux
#   Jacobian. Entries supplied later.
function mixed_flux_jacobian(eos, S, dim::Int)
    error("mixed_flux_jacobian: equations not yet supplied")
end

# max_wave_speed_W: Sec VIII E, Eq. 104/105 — same sqrt(‖A_eff‖_1 ‖A_eff‖_∞) norm-product bound
# as max_wave_speed_C, applied to the mixed operator A_eff instead of the plain C-sector
# Jacobian. Mechanical once mixed_flux_jacobian is supplied.
@inline function max_wave_speed_W(eos, S, dim::Int)
    A = mixed_flux_jacobian(eos, S, dim)
    return sqrt(opnorm(A, 1) * opnorm(A, Inf))
end

# source_W(eos, S) -> SVector{6}
#   Local DNMR relaxation source, Eq. 101's S^a_W(U) (e.g. relaxation-time-driven return to
#   the Navier-Stokes value) — a function of local state only, no direction/gradient argument.
function source_W(eos, S)
    error("source_W: equations not yet supplied")
end

# ─── Path-conservative Rusanov splitting (Eq. 100), restricted to the W-sector ───────────────
# D^∓_LR = (1/2)[N_LR ∓ α_LR ΔW]  (Eq. 100 with D_LR -> N_LR, since there's no flux term here).
# Both N_LR and αLR are evaluated at the same freshly-recovered face-midpoint state S_mid
# (Φ(s=1/2) generally isn't a cached cell), per Eq. 99's path integral and Eq. 104's αLR.
@inline function path_conservative_split_W(eos, SL, SR, dim::Int)
    CL, _, WL = SL
    CR, _, WR = SR
    C_mid = 0.5 * (CL + CR)
    W_mid = 0.5 * (WL + WR)
    ΔC = CR - CL
    ΔW = WR - WL
    S_mid = (C_mid, recover_primitives(C_mid, W_mid, eos), W_mid)

    N_LR = path_integrand_W(eos, S_mid, ΔC, ΔW, dim)
    # Eq. 104: αLR ≃ sqrt(‖A_eff(U*)‖_1 ‖A_eff(U*)‖_∞) at the already-computed midpoint S_mid,
    # not max(α(U_L), α(U_R)).
    αLR  = max_wave_speed_W(eos, S_mid, dim)

    Dminus = 0.5 * (N_LR - αLR * ΔW)
    Dplus  = 0.5 * (N_LR + αLR * ΔW)
    return Dminus, Dplus
end

# ─── Field access for the W-sector accumulation ──────────────────────────────
@inline dissipative_cell_W(d, I) =
    SVector(d.bulk[I], d.pixx[I], d.pixy[I], d.piyy[I], d.nux[I], d.nuy[I])

@inline function add_dissipative_W!(d, I, scale, W)
    d.bulk[I] += scale * W[1]
    d.pixx[I] += scale * W[2]
    d.pixy[I] += scale * W[3]
    d.piyy[I] += scale * W[4]
    d.nux[I]  += scale * W[5]
    d.nuy[I]  += scale * W[6]
    return d
end

# ─── Face accumulation for the W-sector (Eq. 102's D^-/D^+ deposit) ──────────────────────────
# Cannot reuse accumulate_flux_divergence!, which deposits one symmetric value with opposite
# signs at the two faces (dU_i/dt gets ∓F/Δx); here the deposit is asymmetric — the left cell
# of a face gets -D^-/Δx and the right cell gets -D^+/Δx, generally different values from the
# same face (Eq. 102: dW^a_i/dt = -(1/Δx)(D^{-a}_{i+1/2}+D^{+a}_{i-1/2}) + S^a_W(U_i)) — so this
# mirrors accumulate_flux_divergence!'s own face loop (same interior_faces/unit_vector
# primitives from HaloArrays) with a custom two-value scatter instead.
function accumulate_path_conservative_W!(du, u, ranges, dim, scale, eos, read_full, scatter_W!)
    e = unit_vector(ranges, dim)
    for IL in interior_faces(ranges, dim)
        IR = IL + e
        SL = read_full(u, IL)
        SR = read_full(u, IR)
        Dminus, Dplus = path_conservative_split_W(eos, SL, SR, dim)
        scatter_W!(du, IL, -scale, Dminus)
        scatter_W!(du, IR, -scale, Dplus)
    end
    return du
end

# ─── RHS (x then y sweep) + local source — W-sector only, isolated single-sector testing ─────
# Standalone/self-contained like rhs_C! (own fill!(du,0.0)), mirroring its structure for
# Eq. 102, but uses the real (non-zeroed) W-sector state from u for primitive recovery, since
# the W-sector genuinely depends on its own current values.
function rhs_W!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    d = field_storages(u)
    prim_cache, has_W = build_primitive_cache(u, eos)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    read_full = (data, I) -> full_cell(data, prim_cache, I, read_W_fn)

    accumulate_path_conservative_W!(field_storages(du), d, fr, 1, inv(dx), eos, read_full, add_dissipative_W!)
    accumulate_path_conservative_W!(field_storages(du), d, fr, 2, inv(dy), eos, read_full, add_dissipative_W!)

    for I in CartesianIndices(interior_range(u.bulk))
        S = full_cell(d, prim_cache, I, read_W_fn)
        add_dissipative_W!(field_storages(du), I, 1.0, source_W(eos, S))
    end
    return du
end

# ─── SSP-RK2 time step ────────────────────────────────────────────────────────
function ssprk2_step_W!(u, u1, du, eos, dt, dx, dy)
    rhs_W!(du, u, eos, dx, dy)
    @. u1 = u + dt * du
    rhs_W!(du, u1, eos, dx, dy)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Combined RHS: recover primitives -> C-sector fluxes/sources -> W-sector
# path-conservative update -> combine — the actual coupled DNMR right-hand side ────────────────
# One fill!/synchronize_halo!/FaceRanges/primitive-cache build shared by both sectors (unlike
# rhs_C!/rhs_W!, which each do their own), so this is the RHS to hand to a solver/stepper for
# the real coupled system, not just isolated single-sector testing. Needs init_dnmr_state's
# 10-field grid (both sectors present) — calling this against init_ideal_state's 4-field grid
# will error on the missing W-sector fields, same as rhs_W! already does.
function rhs_full!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    d = field_storages(u)

    # Step 1: recover primitives once per cell (including ghosts), cached so every flux/source
    # evaluation below reuses them instead of re-solving the Newton problem.
    prim_cache, has_W = build_primitive_cache(u, eos)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    read_full = (data, I) -> full_cell(data, prim_cache, I, read_W_fn)

    # Step 2: C-sector fluxes (Rusanov flux divergence, Eq. 66-67).
    accumulate_flux_divergence!(field_storages(du), d, fr, 1, inv(dx),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 1), read_full, add_conserved_C!)
    accumulate_flux_divergence!(field_storages(du), d, fr, 2, inv(dy),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 2), read_full, add_conserved_C!)

    # Step 3: W-sector path-conservative update (Eq. 99-100, 102).
    accumulate_path_conservative_W!(field_storages(du), d, fr, 1, inv(dx), eos, read_full, add_dissipative_W!)
    accumulate_path_conservative_W!(field_storages(du), d, fr, 2, inv(dy), eos, read_full, add_dissipative_W!)

    # Step 4: local DNMR relaxation source for W.
    for I in CartesianIndices(interior_range(u.bulk))
        S = full_cell(d, prim_cache, I, read_W_fn)
        add_dissipative_W!(field_storages(du), I, 1.0, source_W(eos, S))
    end

    # Step 5: combine — du now holds the full coupled (C,W) right-hand side, ready to be
    # wrapped up in a solver (SSP-RK2 below, or an implicit/IMEX stepper later for the
    # relaxation-stiff W-sector source).
    return du
end

# ─── SSP-RK2 time step (coupled) ──────────────────────────────────────────────
function ssprk2_step_full!(u, u1, du, eos, dt, dx, dy)
    rhs_full!(du, u, eos, dx, dy)
    @. u1 = u + dt * du
    rhs_full!(du, u1, eos, dx, dy)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────
# Mirrors cfl_dt/diagnostics in utils/finite_volume_viscous_2d.jl, adapted to the bundled
# full_cell state and to the two grid types (ideal vs. viscous) via has_W, same pattern as
# rhs_C!/rhs_W!/rhs_full!.
function cfl_dt_full(u, eos, dx, dy, cfl)
    d = field_storages(u)
    has_W = :bulk in propertynames(u)
    prim_cache, _ = build_primitive_cache(u, eos)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    amax = 0.0
    for I in CartesianIndices(interior_range(u.C0))
        S = full_cell(d, prim_cache, I, read_W_fn)
        sx = has_W ? max_wave_speed_W(eos, S, 1) : max_wave_speed_C(eos, S, 1)
        sy = has_W ? max_wave_speed_W(eos, S, 2) : max_wave_speed_C(eos, S, 2)
        amax = max(amax, sx / dx + sy / dy)
    end
    return cfl / max(amax, 1.0e-14)
end

function diagnostics_full(u, eos, dx, dy)
    d = field_storages(u)
    has_W = :bulk in propertynames(u)
    prim_cache, _ = build_primitive_cache(u, eos)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.C0))
        C, prim, _ = full_cell(d, prim_cache, I, read_W_fn)
        charge += C[4] * dx * dy   # Cn
        energy += C[1] * dx * dy   # C0
        vmax = max(vmax, hypot(prim.vx, prim.vy))
    end
    return charge, energy, vmax
end

# ─── Driver: 2-D DNMR viscous hydro from (T,μ) initial data, zero initial flow/dissipation ───
function run_2d_dnmr(eos, T_init, μ_init; cfl=0.3, t_end=0.20)
    @assert size(T_init) == size(μ_init) "T_init and μ_init grids must have matching dimensions!"
    n = size(T_init, 1)
    @assert size(T_init, 2) == n "This solver assumes a square grid (n × n)!"

    dx = 1.0 / n; dy = 1.0 / n

    u  = init_dnmr_state(n)
    u1 = similar(u)
    du = similar(u)

    for j in 1:n, i in 1:n
        T = T_init[i, j]; μ = μ_init[i, j]
        C = cons_from_prim_ideal(eos, 0.0, 0.0, log(T), μ / T)
        interior_view(u.C0)[i, j]   = C[1]
        interior_view(u.Cx)[i, j]   = C[2]
        interior_view(u.Cy)[i, j]   = C[3]
        interior_view(u.Cn)[i, j]   = C[4]
        interior_view(u.bulk)[i, j] = 0.0
        interior_view(u.pixx)[i, j] = 0.0
        interior_view(u.pixy)[i, j] = 0.0
        interior_view(u.piyy)[i, j] = 0.0
        interior_view(u.nux)[i, j]  = 0.0
        interior_view(u.nuy)[i, j]  = 0.0
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics_full(u, eos, dx, dy)
    @printf("2-D DNMR viscous hydro — grid=%d×%d  t_end=%.2f  initial Cn=%.6f C0=%.6f\n",
        n, n, t_end, q0, e0)

    t = 0.0; step = 0
    while t < t_end
        dt = min(cfl_dt_full(u, eos, dx, dy, cfl), t_end - t)
        if isnan(dt)
            @warn "CFL timestep calculation returned NaN at step $step. Forcing minimal fallback dt."
            dt = min(1e-4, t_end - t)
        end
        ssprk2_step_full!(u, u1, du, eos, dt, dx, dy)
        t += dt; step += 1
    end

    q1, e1, vmax = diagnostics_full(u, eos, dx, dy)
    @printf("  final  Cn=%.6f (Δ=%.2e)  C0=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step)
    return u
end
