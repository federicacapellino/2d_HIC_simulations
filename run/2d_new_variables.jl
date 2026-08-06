using StaticArrays: SVector
import StaticArrays as SA
using NonlinearSolve
import SimpleNonlinearSolve as SNLS
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

# ─── C-sector physical flux & wave speed — equations supplied later ──────────
# U = (C0, Cx, Cy, Cn). dim: 1 = x-flux, 2 = y-flux. Kept as named stubs (rather than an
# inlined flat-space formula) because C0,Cx,Cy,Cn fold in a √-g metric factor, so the actual
# flux/wave-speed formulas need the geometric factors/source terms supplied separately.
function physical_flux_C(eos, U, dim::Int)
    error("physical_flux_C: equations not yet supplied")
end

function max_wave_speed_C(eos, U, dim::Int)
    error("max_wave_speed_C: equations not yet supplied")
end

# ─── Rusanov flux (Eq. 66) — fully mechanical, no physics assumptions ────────
@inline function rusanov_flux_C(eos, UL, UR, dim::Int)
    smax = max(max_wave_speed_C(eos, UL, dim), max_wave_speed_C(eos, UR, dim))
    return 0.5 * (physical_flux_C(eos, UL, dim) + physical_flux_C(eos, UR, dim)) -
           0.5 * smax * (UR - UL)
end

# ─── Field access for accumulate_flux_divergence! (C-sector only — untouched W-sector
# fields, if present on the same LocalMultiHaloArray, are simply never referenced here) ──────
@inline conserved_cell_C(d, I) = SVector(d.C0[I], d.Cx[I], d.Cy[I], d.Cn[I])

@inline function add_conserved_C!(d, I, scale, U)
    d.C0[I] += scale * U[1]
    d.Cx[I] += scale * U[2]
    d.Cy[I] += scale * U[3]
    d.Cn[I] += scale * U[4]
    return d
end

# ─── RHS (x then y sweep) ─────────────────────────────────────────────────────
function rhs_C!(du, u, eos, dx, dy)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 1, inv(dx),
        (UL, UR) -> rusanov_flux_C(eos, UL, UR, 1), conserved_cell_C, add_conserved_C!)
    accumulate_flux_divergence!(parent(du), parent(u), fr, 2, inv(dy),
        (UL, UR) -> rusanov_flux_C(eos, UL, UR, 2), conserved_cell_C, add_conserved_C!)
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
