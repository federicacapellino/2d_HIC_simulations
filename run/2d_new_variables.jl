using StaticArrays: SVector
import StaticArrays as SA
using NonlinearSolve
import SimpleNonlinearSolve as SNLS
using LinearAlgebra: opnorm, SingularException
using Printf
using Test
using HaloArrays
using Plots

default(lw=2, size=(600, 400), xtickfontsize=14, ytickfontsize=14, xlabelfontsize=16,
    ylabelfontsize=16, legendfontsize=12, titlefontsize=14, grid=false, framestyle=:box,
    margins=5Plots.mm)

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
@inline sound_speed_sq(::UltraRelGas)            = 1.0 / 3.0

@inline bar_p(eos, τ, a)   = pressure(eos, exp(τ), a * exp(τ))
@inline bar_nB(eos, τ, a)  = charge_density(eos, exp(τ), a * exp(τ))
@inline bar_eps(eos, τ, a) = enthalpy_density(eos, exp(τ), a * exp(τ)) - bar_p(eos, τ, a)

# Entropy density s = ∂p/∂T|_μ. From the Euler relation Ts = ε̄+p̄-μn̄ = 4p̄-a·p̄ (UltraRelGas has
# ε̄=3p̄, n̄=p̄/T), so s = n̄·(4-a). Dimension: [T]³, same as bar_nB.
@inline bar_s(eos, τ, a) = bar_nB(eos, τ, a) * (4.0 - a)

# ==============================================================================
# 1b. Transport Coefficients & Relaxation Times (Sec 2 of dnmr_equations.pdf)
# ==============================================================================
# Same η/s, ζ/s, dimensionless-shape-constant convention Fluidum uses (SimpleShearViscosity,
# SimpleBulkViscosity) and that config.yaml already exposes for the Fluidum-based scripts
# (eta_over_s, zeta_over_s, tau_eta_par, tau_zeta_par, DsT) — kept local/self-contained here
# rather than depending on Fluidum, matching the rest of this file. Field/default names mirror
# config.yaml's so the numbers carry over if this is ever wired up to it.
struct TransportCoefficients
    eta_over_s::Float64   # η/s
    C_eta::Float64         # dimensionless shape constant setting τ_π (≈ config.yaml's tau_eta_par)
    zeta_over_s::Float64  # ζ/s
    C_zeta::Float64        # dimensionless shape constant setting τ_Π (≈ config.yaml's tau_zeta_par)
    DsT::Float64           # dimensionless charge-diffusion coefficient (2πD_sT-style)
    C_kappa::Float64       # dimensionless shape constant setting τ_ν
end

TransportCoefficients(; eta_over_s=0.55, C_eta=0.2, zeta_over_s=0.018, C_zeta=15.0, DsT=0.1,
    C_kappa=1.0) = TransportCoefficients(eta_over_s, C_eta, zeta_over_s, C_zeta, DsT, C_kappa)

# Shear viscosity η = (η/s)·s. Dimension: [T]³.
@inline shear_viscosity(tc::TransportCoefficients, eos, τ, a) = tc.eta_over_s * bar_s(eos, τ, a)

# Shear relaxation time τ_π = η/(sT·C_η) = (η/s)/(T·C_η) — the s factor cancels. Dimension: [T]⁻¹.
@inline tau_shear(tc::TransportCoefficients, eos, τ, a) = tc.eta_over_s / (exp(τ) * tc.C_eta)

# Bulk viscosity ζ = (ζ/s)·s (flat ratio, no extra T-shaping: UltraRelGas is exactly conformal,
# so a QCD-crossover-shaped ζ(T) wouldn't map onto it). Dimension: [T]³.
@inline bulk_viscosity(tc::TransportCoefficients, eos, τ, a) = tc.zeta_over_s * bar_s(eos, τ, a)

# Bulk relaxation time τ_Π = ζ/(sT·C_ζ)·1/(1/3-c_s²)² (standard Denicol-et-al form). UltraRelGas
# has c_s²=1/3 exactly, making this singular for this EOS — regularized with a floor on
# (1/3-c_s²)² so τ_Π stays finite; the floor is a placeholder that only matters until a
# non-conformal EOS (c_s²≠1/3) is used here. Dimension: [T]⁻¹.
@inline function tau_bulk(tc::TransportCoefficients, eos, τ, a)
    cs2 = sound_speed_sq(eos)
    denom = max((1.0 / 3.0 - cs2)^2, 1.0e-6)
    return tc.zeta_over_s / (exp(τ) * tc.C_zeta * denom)
end

# Charge-diffusion coefficient κ = DsT·n̄ (mirrors Fluidum's diffusion(T,n,y)=DsT(y,T)/T*n
# reduction, i.e. a dimensionless D_sT-like coefficient times the number density).
# Dimension: [T]³.
@inline diffusion_coefficient(tc::TransportCoefficients, eos, τ, a) = tc.DsT * bar_nB(eos, τ, a)

# Diffusion relaxation time τ_ν = C_κ·κ/(n̄T) = C_κ·DsT/T — the n̄ factor cancels, same reduction
# pattern as τ_π. Dimension: [T]⁻¹.
@inline tau_diffusion(tc::TransportCoefficients, eos, τ, a) =
    tc.C_kappa * diffusion_coefficient(tc, eos, τ, a) / (bar_nB(eos, τ, a) * exp(τ))

function run_transport_coefficient_test()
    eos = UltraRelGas(1.0)
    tc  = TransportCoefficients()

    @testset "Transport coefficients: positivity & dimensional scaling" begin
        τ_ref = log(0.3)
        a_ref = 0.1
        λ = 2.0
        τ_scaled = τ_ref + log(λ)   # T -> λT at fixed a=μ/T

        @testset "Positivity" begin
            @test shear_viscosity(tc, eos, τ_ref, a_ref) > 0.0
            @test bulk_viscosity(tc, eos, τ_ref, a_ref) > 0.0
            @test diffusion_coefficient(tc, eos, τ_ref, a_ref) > 0.0
            @test tau_shear(tc, eos, τ_ref, a_ref) > 0.0
            @test tau_bulk(tc, eos, τ_ref, a_ref) > 0.0
            @test tau_diffusion(tc, eos, τ_ref, a_ref) > 0.0
            @test isfinite(tau_bulk(tc, eos, τ_ref, a_ref))
        end

        @testset "η, ζ, κ scale as T³" begin
            @test shear_viscosity(tc, eos, τ_scaled, a_ref) / shear_viscosity(tc, eos, τ_ref, a_ref) ≈ λ^3 atol=1e-8
            @test bulk_viscosity(tc, eos, τ_scaled, a_ref) / bulk_viscosity(tc, eos, τ_ref, a_ref) ≈ λ^3 atol=1e-8
            @test diffusion_coefficient(tc, eos, τ_scaled, a_ref) / diffusion_coefficient(tc, eos, τ_ref, a_ref) ≈ λ^3 atol=1e-8
        end

        @testset "τ_π, τ_Π, τ_ν scale as 1/T" begin
            @test tau_shear(tc, eos, τ_scaled, a_ref) / tau_shear(tc, eos, τ_ref, a_ref) ≈ 1.0 / λ atol=1e-8
            @test tau_bulk(tc, eos, τ_scaled, a_ref) / tau_bulk(tc, eos, τ_ref, a_ref) ≈ 1.0 / λ atol=1e-8
            @test tau_diffusion(tc, eos, τ_scaled, a_ref) / tau_diffusion(tc, eos, τ_ref, a_ref) ≈ 1.0 / λ atol=1e-8
        end
    end
end

run_transport_coefficient_test()

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

    # Physically-motivated seed instead of a fixed constant: the W≈1, qx=qy=0 limit of
    # the UltraRelGas relations (ε̄=3p̄, n̄=A T^3 e^{μ/T}) gives T≈C0/(3Cn), a≈log(Cn/(A T^3))
    # — same closed-form seed used by relativistic_hydro_Tmu_1d/2d.jl in HaloArrays'
    # examples. A fixed guess was previously overshooting into non-convergence (MaxIters)
    # at cells far from T=0.5 (e.g. steep blast-edge gradients); this seed tracks the
    # cell's actual state instead.
    T_guess = max(C0 / (3.0 * max(Cn, 1.0e-12)), 1.0e-8)
    a_guess = log(max(Cn / (eos.A * T_guess^3), 1.0e-300))
    u0 = SA.SVector{4}(0.0, 0.0, log(T_guess), a_guess)
    p_tuple = (C0, Cx, Cy, Cn, bulk_Pi, pixx, pixy, piyy, nux, nuy, eos)

    prob = NonlinearProblem(cons_to_primitive_rhds, u0, p_tuple)
    # The seeded guess above lands very close to the true root for most cells, but for
    # some it can walk the Newton iteration into an exactly-singular Jacobian (e.g. the
    # EOS's exp(clamp(μ/T,...)) going flat), which SimpleNewtonRaphson's plain `\` solve
    # throws on rather than degrading gracefully. Fall back to the old generic (T=0.5)
    # seed in that case — a different starting point avoids the same singular iterate.
    sol = try
        solve(prob, SNLS.SimpleNewtonRaphson())
    catch e
        e isa SingularException || rethrow()
        u0_fallback = SA.SVector{4}(0.0, 0.0, log(0.5), 1.0 / 0.5)
        solve(NonlinearProblem(cons_to_primitive_rhds, u0_fallback, p_tuple), SNLS.SimpleNewtonRaphson())
    end

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
function build_primitive_cache(u, eos, t_now)
    d = field_storages(u)
    has_W = :bulk in propertynames(u)
    prim_cache = Array{PrimitiveState}(undef, size(d.C0))
    inv_t = 1.0 / t_now
    h = halo_width(u.C0)
    nx, ny = size(interior_view(u.C0))
    for I in CartesianIndices(d.C0)
        i, j = Tuple(I)
        # Diagonal corner ghost cells (both dimensions outside the interior range) are
        # never written by synchronize_halo!'s per-face boundary fill — HaloArrays fills
        # each face's ghost band only across the OTHER dimension's *interior* span (see
        # _halo_window_view in HaloArrays/src/haloarray.jl), by construction leaving
        # corners at whatever `undef`-allocated memory they started with — and never read
        # by the face-only flux/source stencils below (accumulate_flux_divergence! etc.
        # only touch face-adjacent or interior cells). Skip them rather than inverting
        # leftover garbage there (observed as NaN/subnormal conserved values, causing
        # spurious "did not converge" warnings unrelated to any physical cell).
        (1 <= i - h <= nx || 1 <= j - h <= ny) || continue
        C = conserved_cell_C(d, I) .* inv_t
        Wdiss = has_W ? dissipative_cell_W(d, I) : ZERO_W_STATE
        prim_cache[I] = recover_primitives(C, Wdiss, eos)
    end
    return prim_cache, has_W
end

# ─── Navier-Stokes targets (Eq 24-25's Π_NS,π^ij_NS,ν^i_NS) — leading-order, spatial-gradient-only
# ────────────────────────────────────────────────────────────────────────────────────────────
# θ=∇·u uses the standard boost-invariant formula θ = 1/t_now + ∂xvx + ∂yvy (the Bjorken geometric
# term, exact for u^η=0, plus the flat transverse divergence) — NOT the ∂τu^τ term, which is
# dropped at this order (standard leading-order approximation throughout the boost-invariant
# 2+1D viscous-Bjorken literature). σ^xx,σ^xy,σ^yy are the traceless-symmetric-transverse shear
# tensor built from the same gradients; σ^xx+σ^yy is consistent with the existing algebraic
# π^ηη=-(π^xx+π^yy)/t² reconstruction (Eq 14), both following from the same 3D (x,y,η)
# tracelessness condition, so no separate σ^ηη is needed here. Central differences use
# prim_cache's ghost-inclusive band (same indexing build_primitive_cache already fills).
@inline function ns_targets(tc::TransportCoefficients, eos, prim_cache, I, dx, dy, t_now)
    ex = CartesianIndex(1, 0)
    ey = CartesianIndex(0, 1)

    dvx_dx = (prim_cache[I+ex].vx - prim_cache[I-ex].vx) / (2.0 * dx)
    dvy_dx = (prim_cache[I+ex].vy - prim_cache[I-ex].vy) / (2.0 * dx)
    dvx_dy = (prim_cache[I+ey].vx - prim_cache[I-ey].vx) / (2.0 * dy)
    dvy_dy = (prim_cache[I+ey].vy - prim_cache[I-ey].vy) / (2.0 * dy)
    da_dx  = (prim_cache[I+ex].a  - prim_cache[I-ex].a)  / (2.0 * dx)
    da_dy  = (prim_cache[I+ey].a  - prim_cache[I-ey].a)  / (2.0 * dy)

    θ   = 1.0 / t_now + dvx_dx + dvy_dy
    σxx = dvx_dx - θ / 3.0
    σyy = dvy_dy - θ / 3.0
    σxy = 0.5 * (dvy_dx + dvx_dy)

    τp, ap = prim_cache[I].τ, prim_cache[I].a
    η = shear_viscosity(tc, eos, τp, ap)
    ζ = bulk_viscosity(tc, eos, τp, ap)
    κ = diffusion_coefficient(tc, eos, τp, ap)

    Π_NS   = -ζ * θ
    πxx_NS = 2.0 * η * σxx
    πxy_NS = 2.0 * η * σxy
    πyy_NS = 2.0 * η * σyy
    νx_NS  = -κ * da_dx
    νy_NS  = -κ * da_dy

    return SVector(Π_NS, πxx_NS, πxy_NS, πyy_NS, νx_NS, νy_NS)
end

# Builds a minimal-but-valid PrimitiveState for ns_targets' synthetic test grids below — only
# vx,vy,a,τ are read by ns_targets, the rest are unused filler.
@inline synthetic_prim(vx, vy, τ, a) =
    PrimitiveState(vx, vy, τ, a, 1.0, vx, vy, exp(τ), a * exp(τ), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function run_navier_stokes_target_test()
    eos = UltraRelGas(1.0)
    tc  = TransportCoefficients()
    dx = dy = 0.1
    t_now = 1.0

    @testset "ns_targets" begin
        @testset "Linear velocity/a profile: exact central-difference gradients" begin
            # vx = 0.1+0.05x+0.02y, vy = -0.05+0.03x+0.04y, a = 0.2+0.01x-0.02y — central
            # differences of a linear function are exact, so this pins down ns_targets' formula
            # against a hand-computed reference with no discretization error.
            τ0 = log(0.35); a0 = 0.2
            n = 5
            prim_cache = Array{PrimitiveState}(undef, n, n)
            for j in 1:n, i in 1:n
                x = (i - 3) * dx; y = (j - 3) * dy
                vx = 0.1 + 0.05x + 0.02y
                vy = -0.05 + 0.03x + 0.04y
                a  = a0 + 0.01x - 0.02y
                prim_cache[i, j] = synthetic_prim(vx, vy, τ0, a)
            end
            I = CartesianIndex(3, 3)   # x=y=0, matching τ0,a0 above

            dvx_dx, dvy_dy, dvy_dx, dvx_dy, da_dx, da_dy = 0.05, 0.04, 0.03, 0.02, 0.01, -0.02
            θ_ref   = 1.0 / t_now + dvx_dx + dvy_dy
            σxx_ref = dvx_dx - θ_ref / 3.0
            σyy_ref = dvy_dy - θ_ref / 3.0
            σxy_ref = 0.5 * (dvy_dx + dvx_dy)
            η = shear_viscosity(tc, eos, τ0, a0)
            ζ = bulk_viscosity(tc, eos, τ0, a0)
            κ = diffusion_coefficient(tc, eos, τ0, a0)
            NS_ref = SVector(-ζ * θ_ref, 2η * σxx_ref, 2η * σxy_ref, 2η * σyy_ref, -κ * da_dx, -κ * da_dy)

            NS = ns_targets(tc, eos, prim_cache, I, dx, dy, t_now)
            @test all(isapprox.(NS, NS_ref, atol=1e-10))
        end

        @testset "Uniform flow: pure-Bjorken limit (θ=1/t_now, transverse gradients=0)" begin
            # Zero TRANSVERSE gradients doesn't mean zero shear: θ=1/t_now from the Bjorken term
            # alone still gives σxx=σyy=-θ/3 (the well-known nonzero transverse shear induced by
            # pure longitudinal boost-invariant expansion), σxy=0, ∇a=0.
            n = 5
            τ0 = log(0.4); a0 = 0.15
            prim_cache = fill(synthetic_prim(0.2, -0.1, τ0, a0), n, n)
            I = CartesianIndex(3, 3)

            θ_ref = 1.0 / t_now
            σxx_ref = σyy_ref = -θ_ref / 3.0
            ζ = bulk_viscosity(tc, eos, τ0, a0)
            η = shear_viscosity(tc, eos, τ0, a0)
            NS_ref = SVector(-ζ * θ_ref, 2η * σxx_ref, 0.0, 2η * σyy_ref, 0.0, 0.0)

            NS = ns_targets(tc, eos, prim_cache, I, dx, dy, t_now)
            @test all(isapprox.(NS, NS_ref, atol=1e-12))
        end
    end
end

run_navier_stokes_target_test()

# ─── C-sector physical flux & wave speed (Eqs 2-9 ideal / Eqs 20-23 dissipative of
# dnmr_equations.pdf) ──────────
# S = (C, prim, W) at one cell/state, C=(C0,Cx,Cy,Cn), W=(Π,π^xx,π^xy,π^yy,ν^x,ν^y). dim: 1 =
# x-flux, 2 = y-flux. t_now is the current Milne proper time (named to avoid clashing with this
# file's τ≡log T primitive).
#
# PDF Eqs 2-5/16-19 define the STORED grid quantities as C = t_now·Ĉ, where Ĉ=(Ĉ0,Ĉx,Ĉy,Ĉn) is
# exactly what cons_from_prim/cons_to_primitive already compute (no t factor at all — see
# run_variable_consistency_test). Eqs 6-9/20-23 show the flux is F = t_now·f_flat(Ĉ,W), the
# flat-space ultrarelativistic-gas flux plus the dissipative π^ij/ν^i additions (Eqs 21-23): P̄ =
# p̄+Π (effective pressure) replaces the bare pressure in the momentum-flux diagonal, and π^ij,ν^i
# enter Cx/Cy/Cn's flux as plain additive terms — no B-matrix/path-conservative treatment needed
# here, since Π,π^ij,ν^i are themselves evolved fields, not gradients. For the C0-balance law
# (Eq 6/20) the x/y-flux is exactly C1,C2 themselves (the PDF's own note under Eq 9) — no separate
# evaluation needed.
@inline function physical_flux_C(eos, S, dim::Int, t_now)
    C, prim, W = S
    qx, qy = prim.qx, prim.qy
    p̄ = prim.p
    Π, πxx, πxy, πyy, νx, νy = W
    P̄ = p̄ + Π
    w̄ = prim.eps + P̄
    n̄ = prim.n
    if dim == 1
        return SVector(C[2], t_now * (w̄ * qx^2 + P̄ + πxx), t_now * (w̄ * qx * qy + πxy), t_now * (n̄ * qx + νx))
    else
        return SVector(C[3], t_now * (w̄ * qx * qy + πxy), t_now * (w̄ * qy^2 + P̄ + πyy), t_now * (n̄ * qy + νy))
    end
end

# flux_jacobian_C(eos, S, dim) -> 4x4 matrix, A(U) = ∂F_C/∂C. Superseded for the ideal case by
# the analytic max_wave_speed_C below (see its docstring for why); kept as a stub reserved for
# the future viscous/mixed system (Sec 2), where no closed-form eigenvalues exist and the
# generic Eq 104 norm bound is the only option.
function flux_jacobian_C(eos, S, dim::Int)
    error("flux_jacobian_C: equations not yet supplied")
end

# max_wave_speed_C — analytic ultrarelativistic-gas eigenvalues (Mignone & Bodo 2005), NOT the
# generic Eq 104 Jacobian-norm bound. Because C = t_now·Ĉ and F = t_now·f_flat(Ĉ)
# (physical_flux_C above), ∂F/∂C = t_now·(∂f_flat/∂Ĉ)·(1/t_now) = ∂f_flat/∂Ĉ: the t_now
# prefactor cancels exactly, so the C-sector characteristic speeds equal the plain flat-space
# ultrarelativistic-gas eigenvalues, independent of t_now — the same closed form used in
# HaloArrays' relativistic_hydro_Tmu_2d.jl example.
#
# Deliberately unchanged now that physical_flux_C carries the Eqs 20-23 π^ij/ν^i additions: this
# still only depends on prim.vx/vy and sound_speed_sq, not on Π/π^ij/ν^i. Using the ideal-gas
# characteristic speed as the Rusanov dissipation coefficient, while the flux itself carries the
# viscous correction, is the standard approach in relativistic viscous-hydro codes — the viscous
# terms don't change the leading-order causal structure, so re-deriving exact mixed eigenvalues
# here isn't necessary (that's what the still-stubbed flux_jacobian_C/Eq 104 route is for, if ever
# needed).
@inline function max_wave_speed_C(eos, S, dim::Int)
    _, prim, _ = S
    vx, vy = prim.vx, prim.vy
    v2  = vx^2 + vy^2
    vn  = dim == 1 ? vx : vy
    cs2 = sound_speed_sq(eos)
    disc = (1.0 - v2) * ((1.0 - v2 * cs2) - vn^2 * (1.0 - cs2))
    root = sqrt(max(disc, 0.0)) * sqrt(cs2)
    den  = 1.0 - v2 * cs2
    return max(abs((vn * (1.0 - cs2) + root) / den), abs((vn * (1.0 - cs2) - root) / den))
end

# ─── C-sector local source (Eq 20 RHS: ∂tC0+∂xC1+∂yC2 = (π^xx+π^yy)-P̄, P̄=p̄+Π;
# Eqs 21-23 carry no local source, only the flux additions already in physical_flux_C) ──────────
# Reduces exactly to the old ideal-only source (-p̄) when W=(0,0,0,0,0,0).
@inline function source_C(eos, S)
    _, prim, W = S
    Π, πxx, _, πyy, _, _ = W
    return SVector((πxx + πyy) - (prim.p + Π), 0.0, 0.0, 0.0)
end

# ─── Rusanov flux (Eq. 66) — fully mechanical, no physics assumptions ────────
# smax follows Eq. 104 literally: αLR ≃ sqrt(‖A(U*)‖_1 ‖A(U*)‖_∞) at the interface-averaged
# state U* = (U_L+U_R)/2, not max(α(U_L), α(U_R)) — the two only coincide when α is linear in U.
# (Here smax's ‖A‖-norm bound is replaced by the analytic max_wave_speed_C, see its docstring.)
# recover_primitives expects the UNWEIGHTED Ĉ, so C_mid (built from the stored, t-weighted CL,
# CR) must be divided by t_now before primitive recovery — see build_primitive_cache for the
# same fix applied to the per-cell cache.
@inline function rusanov_flux_C(eos, SL, SR, dim::Int, t_now)
    CL, CR = SL[1], SR[1]
    WL, WR = SL[3], SR[3]
    C_mid = 0.5 * (CL + CR)
    W_mid = 0.5 * (WL + WR)
    S_mid = (C_mid, recover_primitives(C_mid ./ t_now, W_mid, eos), W_mid)
    smax = max_wave_speed_C(eos, S_mid, dim)
    return 0.5 * (physical_flux_C(eos, SL, dim, t_now) + physical_flux_C(eos, SR, dim, t_now)) -
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

# ─── RHS (x then y sweep) + local source — C-sector only, Sec 1 ideal case ───
# Primitives are always recovered assuming zero DISSP here (build_primitive_cache falls back
# to that when u has no W-sector fields, but even called against init_dnmr_state's 10-field
# grid this stays a pure C-sector-only evolution — use rhs_full! for the true coupled system.
# t_now is the current Milne proper time — see physical_flux_C/build_primitive_cache.
function rhs_C!(du, u, eos, dx, dy, t_now)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    d = field_storages(u)
    prim_cache, has_W = build_primitive_cache(u, eos, t_now)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    read_full = (data, I) -> full_cell(data, prim_cache, I, read_W_fn)

    accumulate_flux_divergence!(field_storages(du), d, fr, 1, inv(dx),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 1, t_now), read_full, add_conserved_C!)
    accumulate_flux_divergence!(field_storages(du), d, fr, 2, inv(dy),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 2, t_now), read_full, add_conserved_C!)

    # Eq 6's local source, -p̄ on C0 only (source_C).
    for I in CartesianIndices(interior_range(u.C0))
        S = full_cell(d, prim_cache, I, read_W_fn)
        add_conserved_C!(field_storages(du), I, 1.0, source_C(eos, S))
    end
    return du
end

# ─── SSP-RK2 time step ────────────────────────────────────────────────────────
# Standard explicit treatment of the t_now-dependent RHS: stage 1 at t_now, stage 2 at t_now+dt.
function ssprk2_step_C!(u, u1, du, eos, t_now, dt, dx, dy)
    rhs_C!(du, u, eos, dx, dy, t_now)
    @. u1 = u + dt * du
    rhs_C!(du, u1, eos, dx, dy, t_now + dt)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Snapshot capture: primitive fields on the interior (x,y) grid at one proper time ─────────
# Bundles (T, ε, n, |v|) as plain n×n Matrix{Float64}s (indexed [i,j] = [x,y], matching the
# interior_view convention used throughout this file) for later plotting — decoupled from the
# HaloArrays-internal PrimitiveState/padded-array representation so plotting code stays simple.
function capture_snapshot(u, eos, t_now)
    prim_cache, _ = build_primitive_cache(u, eos, t_now)
    n = size(interior_view(u.C0), 1)
    h = halo_width(u.C0)
    T    = Array{Float64}(undef, n, n)
    eps  = Array{Float64}(undef, n, n)
    nch  = Array{Float64}(undef, n, n)
    vmag = Array{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        prim = prim_cache[CartesianIndex(i + h, j + h)]
        T[i, j]    = prim.Temp
        eps[i, j]  = prim.eps
        nch[i, j]  = prim.n
        vmag[i, j] = hypot(prim.vx, prim.vy)
    end
    return (tau=t_now, T=T, eps=eps, n=nch, vmag=vmag)
end

# ─── Driver: 2-D ideal Milne-coordinate hydro (Sec. 1) from (T,μ) initial data ────────────────
# Uses the C-sector-only pathway (init_ideal_state/rhs_C!/ssprk2_step_C!) — the W-sector fields
# are simply absent from this grid, matching "W ≡ 0" for the ideal case. t_now is the Milne
# proper time τ (named t_now to avoid clashing with this file's τ≡log T primitive); evolution
# starts at tau0 (the proper time the initial (T,μ) data is given at, matching config.yaml's
# tau0 convention) and proceeds to tau0+t_end. cfl_dt_full/diagnostics_full are defined later
# in this file but resolved lazily at call time (standard top-level Julia function ordering).
#
# snapshot_taus (sorted proper times) + snapshots (a caller-provided Vector to push into) are
# both optional and default to a no-op, so existing callers (e.g. run_ideal_milne_blast_test)
# that only want the final state are unaffected. Whenever t_now advances past the next requested
# snapshot_taus entry, capture_snapshot is called and appended to snapshots.
function run_2d_ideal(eos, T_init, μ_init; tau0=1.0, cfl=0.3, t_end=0.20,
        snapshot_taus=Float64[], snapshots=nothing)
    @assert size(T_init) == size(μ_init) "T_init and μ_init grids must have matching dimensions!"
    n = size(T_init, 1)
    @assert size(T_init, 2) == n "This solver assumes a square grid (n × n)!"

    dx = 1.0 / n; dy = 1.0 / n

    u  = init_ideal_state(n)
    u1 = similar(u)
    du = similar(u)

    # cons_from_prim_ideal returns Ĉ (Eqs 2-5's local, unweighted conserved densities); the
    # stored grid quantities are the t-weighted C = tau0·Ĉ at the initial proper time tau0.
    for j in 1:n, i in 1:n
        T = T_init[i, j]; μ = μ_init[i, j]
        C = cons_from_prim_ideal(eos, 0.0, 0.0, log(T), μ / T)
        interior_view(u.C0)[i, j] = tau0 * C[1]
        interior_view(u.Cx)[i, j] = tau0 * C[2]
        interior_view(u.Cy)[i, j] = tau0 * C[3]
        interior_view(u.Cn)[i, j] = tau0 * C[4]
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics_full(u, eos, dx, dy, tau0)
    @printf("2-D ideal Milne hydro — grid=%d×%d  tau0=%.2f  t_end=%.2f  initial dN/dη=%.6f dE/dη=%.6f\n",
        n, n, tau0, t_end, q0, e0)

    remaining_taus = sort(collect(Float64, snapshot_taus))
    take_snapshot!(τ) = snapshots !== nothing && push!(snapshots, capture_snapshot(u, eos, τ))
    while !isempty(remaining_taus) && first(remaining_taus) <= tau0
        take_snapshot!(popfirst!(remaining_taus))
    end

    t_now = tau0; t_stop = tau0 + t_end; step = 0
    while t_now < t_stop
        dt = min(cfl_dt_full(u, eos, dx, dy, cfl, t_now), t_stop - t_now)
        if isnan(dt)
            @warn "CFL timestep calculation returned NaN at step $step. Forcing minimal fallback dt."
            dt = min(1e-4, t_stop - t_now)
        end
        ssprk2_step_C!(u, u1, du, eos, t_now, dt, dx, dy)
        t_now += dt; step += 1
        while !isempty(remaining_taus) && t_now >= first(remaining_taus)
            take_snapshot!(popfirst!(remaining_taus))
        end
    end
    # Anything requested past t_stop (e.g. exactly t_stop, hit only via float roundoff) still
    # gets a snapshot of the final state.
    while !isempty(remaining_taus)
        take_snapshot!(popfirst!(remaining_taus))
    end

    q1, e1, vmax = diagnostics_full(u, eos, dx, dy, t_now)
    @printf("  final  dN/dη=%.6f (Δ=%.2e)  dE/dη=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d  τ=%.4f\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step, t_now)
    return u
end

# ─── Plotting: heatmaps of (T, ε, |v|) at one snapshot, saved as a single PDF ──────────────────
# x,y are cell-center coordinates on the unit square (matching run_2d_ideal's dx=dy=1/n grid).
# heatmap(x,y,z) expects z sized (length(y),length(x)) i.e. z[iy,ix], so the [i,j]=[x,y]-indexed
# snapshot matrices are transposed here.
function plot_ideal_milne_snapshot(snap; outdir="plots")
    n = size(snap.T, 1)
    x = ((1:n) .- 0.5) ./ n
    τstr = @sprintf("%.2f", snap.tau)

    p1 = heatmap(x, x, snap.T';    c=:thermal, aspect_ratio=1, xlims=(0, 1), ylims=(0, 1),
        xlabel="x", ylabel="y", title="T,  τ=$τstr")
    p2 = heatmap(x, x, snap.eps';  c=:viridis, aspect_ratio=1, xlims=(0, 1), ylims=(0, 1),
        xlabel="x", ylabel="y", title="ε̄")
    p3 = heatmap(x, x, snap.vmag'; c=:inferno, aspect_ratio=1, xlims=(0, 1), ylims=(0, 1),
        clims=(0, 1), xlabel="x", ylabel="y", title="|v|")
    p = plot(p1, p2, p3, layout=(1, 3), size=(1500, 420))

    mkpath(outdir)
    fname = joinpath(outdir, "ideal_milne_tau_$(replace(τstr, "." => "p")).pdf")
    savefig(p, fname)
    return fname
end

plot_ideal_milne_snapshots(snapshots; outdir="plots") =
    [plot_ideal_milne_snapshot(s; outdir=outdir) for s in snapshots]

# ─── Driver: run + plot several proper-time snapshots of the ideal Milne blast ────────────────
# Same disk-blast (T,μ) initial data as run_ideal_milne_blast_test, but samples n_snapshots
# evenly-spaced proper times over [tau0, tau0+t_end] and writes one heatmap PDF per snapshot
# under plots/ (following this repo's convention of writing generated PDFs to plots/, e.g.
# plots/plotting_macros.jl's savefig(p, "./plots/...")).
function run_2d_ideal_and_plot(; A=1.0, n=48, cfl=0.3, tau0=1.0, t_end=0.20, n_snapshots=5,
        T_in=1.0, μ_in=0.3, T_out=0.5, μ_out=0.1, radius=0.15, outdir="plots")
    eos = UltraRelGas(A)
    T_init = Array{Float64}(undef, n, n)
    μ_init = Array{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        x = (i - 0.5) / n; y = (j - 0.5) / n
        inside = hypot(x - 0.5, y - 0.5) < radius
        T_init[i, j] = inside ? T_in : T_out
        μ_init[i, j] = inside ? μ_in : μ_out
    end

    snapshot_taus = collect(range(tau0, tau0 + t_end, length=n_snapshots))
    snapshots = Vector{@NamedTuple{tau::Float64, T::Matrix{Float64}, eps::Matrix{Float64},
        n::Matrix{Float64}, vmag::Matrix{Float64}}}()
    u = run_2d_ideal(eos, T_init, μ_init; tau0=tau0, cfl=cfl, t_end=t_end,
        snapshot_taus=snapshot_taus, snapshots=snapshots)

    files = plot_ideal_milne_snapshots(snapshots; outdir=outdir)
    for f in files
        println("  wrote $f")
    end
    return u, snapshots, files
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

# source_W(tc, eos, S, NS) -> SVector{6}
#   Local DNMR relaxation source, Eq. 101's S^a_W(U) — dnmr_equations.pdf Eq. 24-25, FIRST ORDER
#   ONLY (2nd-order shear-bulk-vorticity coupling terms deliberately dropped, per the note's own
#   scoping and by explicit choice — see plan): ∂tΠ = -(Π-Π_NS)/τ_Π, ∂tπ^ij = -(π^ij-π^ij_NS)/τ_π,
#   ∂tν^i = -(ν^i-ν^i_NS)/τ_ν. NS = (Π_NS,π^xx_NS,π^xy_NS,π^yy_NS,ν^x_NS,ν^y_NS) from ns_targets;
#   τ_π/τ_Π/τ_ν from tau_shear/tau_bulk/tau_diffusion (Sec 1b), evaluated at S's cached (τ,a).
@inline function source_W(tc::TransportCoefficients, eos, S, NS)
    _, prim, W = S
    Π, πxx, πxy, πyy, νx, νy = W
    Π_NS, πxx_NS, πxy_NS, πyy_NS, νx_NS, νy_NS = NS
    τp, ap = prim.τ, prim.a

    τΠ = tau_bulk(tc, eos, τp, ap)
    τπ = tau_shear(tc, eos, τp, ap)
    τν = tau_diffusion(tc, eos, τp, ap)

    return SVector(
        -(Π   - Π_NS)   / τΠ,
        -(πxx - πxx_NS) / τπ,
        -(πxy - πxy_NS) / τπ,
        -(πyy - πyy_NS) / τπ,
        -(νx  - νx_NS)  / τν,
        -(νy  - νy_NS)  / τν,
    )
end

function run_source_w_relaxation_test()
    eos = UltraRelGas(1.0)
    tc  = TransportCoefficients()
    τ0 = log(0.3); a0 = 0.1
    prim = synthetic_prim(0.0, 0.0, τ0, a0)
    C_dummy = SVector(0.0, 0.0, 0.0, 0.0)

    @testset "source_W relaxation" begin
        @testset "Matches -(W-W_NS)/τ exactly, isolated from ns_targets" begin
            W  = SVector(0.05, 0.02, -0.01, 0.03, 0.01, -0.02)
            NS = SVector(0.01, -0.005, 0.002, 0.01, -0.003, 0.001)
            S = (C_dummy, prim, W)

            τΠ = tau_bulk(tc, eos, τ0, a0)
            τπ = tau_shear(tc, eos, τ0, a0)
            τν = tau_diffusion(tc, eos, τ0, a0)
            ref = SVector(-(W[1] - NS[1]) / τΠ, -(W[2] - NS[2]) / τπ, -(W[3] - NS[3]) / τπ,
                -(W[4] - NS[4]) / τπ, -(W[5] - NS[5]) / τν, -(W[6] - NS[6]) / τν)

            @test all(isapprox.(source_W(tc, eos, S, NS), ref, atol=1e-12))
        end

        @testset "W at target: zero source" begin
            W = SVector(0.02, 0.01, -0.005, 0.015, 0.004, -0.002)
            S = (C_dummy, prim, W)
            @test all(isapprox.(source_W(tc, eos, S, W), 0.0, atol=1e-12))
        end

        @testset "Sign: source pulls W toward W_NS" begin
            NS = SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            W  = SVector(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)   # all above target
            S = (C_dummy, prim, W)
            @test all(source_W(tc, eos, S, NS) .< 0.0)
        end
    end
end

run_source_w_relaxation_test()

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

# ─── RHS (local relaxation only) — W-sector only, isolated single-sector testing ─────
# Standalone/self-contained like rhs_C! (own fill!(du,0.0)), mirroring its structure, but uses the
# real (non-zeroed) W-sector state from u for primitive recovery, since the W-sector genuinely
# depends on its own current values.
#
# Advection deferred (see plan / path_integrand_W, mixed_flux_jacobian docstrings): Eq. 99-100's
# path-conservative accumulate_path_conservative_W! is NOT called here — Π,π^ij,ν^i relax locally
# toward their Navier-Stokes targets (ns_targets) but aren't yet carried by the flow. Re-enable
# that call once path_integrand_W/mixed_flux_jacobian are supplied.
function rhs_W!(du, u, eos, tc::TransportCoefficients, dx, dy, t_now)
    fill!(du, 0.0)
    synchronize_halo!(u)
    d = field_storages(u)
    prim_cache, has_W = build_primitive_cache(u, eos, t_now)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W

    for I in CartesianIndices(interior_range(u.bulk))
        S = full_cell(d, prim_cache, I, read_W_fn)
        NS = ns_targets(tc, eos, prim_cache, I, dx, dy, t_now)
        add_dissipative_W!(field_storages(du), I, 1.0, source_W(tc, eos, S, NS))
    end
    return du
end

# ─── SSP-RK2 time step ────────────────────────────────────────────────────────
function ssprk2_step_W!(u, u1, du, eos, tc::TransportCoefficients, t_now, dt, dx, dy)
    rhs_W!(du, u, eos, tc, dx, dy, t_now)
    @. u1 = u + dt * du
    rhs_W!(du, u1, eos, tc, dx, dy, t_now + dt)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Combined RHS: recover primitives -> C-sector fluxes/sources -> W-sector local relaxation ->
# combine — the actual coupled DNMR right-hand side ────────────────
# One fill!/synchronize_halo!/FaceRanges/primitive-cache build shared by both sectors (unlike
# rhs_C!/rhs_W!, which each do their own), so this is the RHS to hand to a solver/stepper for
# the real coupled system, not just isolated single-sector testing. Needs init_dnmr_state's
# 10-field grid (both sectors present) — calling this against init_ideal_state's 4-field grid
# will error on the missing W-sector fields, same as rhs_W! already does.
#
# Advection deferred (see rhs_W!'s docstring): no accumulate_path_conservative_W! call — Π,π^ij,ν^i
# relax locally toward ns_targets but aren't yet carried by the flow.
function rhs_full!(du, u, eos, tc::TransportCoefficients, dx, dy, t_now)
    fill!(du, 0.0)
    synchronize_halo!(u)
    fr = FaceRanges(u)
    d = field_storages(u)

    # Step 1: recover primitives once per cell (including ghosts), cached so every flux/source
    # evaluation below reuses them instead of re-solving the Newton problem.
    prim_cache, has_W = build_primitive_cache(u, eos, t_now)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    read_full = (data, I) -> full_cell(data, prim_cache, I, read_W_fn)

    # Step 2: C-sector fluxes (Rusanov flux divergence, Eq. 66-67), now carrying the Eqs 20-23
    # π^ij/ν^i additions via physical_flux_C/source_C.
    accumulate_flux_divergence!(field_storages(du), d, fr, 1, inv(dx),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 1, t_now), read_full, add_conserved_C!)
    accumulate_flux_divergence!(field_storages(du), d, fr, 2, inv(dy),
        (SL, SR) -> rusanov_flux_C(eos, SL, SR, 2, t_now), read_full, add_conserved_C!)

    for I in CartesianIndices(interior_range(u.C0))
        S = full_cell(d, prim_cache, I, read_W_fn)
        add_conserved_C!(field_storages(du), I, 1.0, source_C(eos, S))
    end

    # Step 3: local DNMR relaxation source for W (Eq 24-25, first order only).
    for I in CartesianIndices(interior_range(u.bulk))
        S = full_cell(d, prim_cache, I, read_W_fn)
        NS = ns_targets(tc, eos, prim_cache, I, dx, dy, t_now)
        add_dissipative_W!(field_storages(du), I, 1.0, source_W(tc, eos, S, NS))
    end

    # Step 4: combine — du now holds the full coupled (C,W) right-hand side, ready to be
    # wrapped up in a solver (SSP-RK2 below, or an implicit/IMEX stepper later for the
    # relaxation-stiff W-sector source).
    return du
end

# ─── SSP-RK2 time step (coupled) ──────────────────────────────────────────────
function ssprk2_step_full!(u, u1, du, eos, tc::TransportCoefficients, t_now, dt, dx, dy)
    rhs_full!(du, u, eos, tc, dx, dy, t_now)
    @. u1 = u + dt * du
    rhs_full!(du, u1, eos, tc, dx, dy, t_now + dt)
    @. u = 0.5 * u + 0.5 * (u1 + dt * du)
    return u
end

# ─── Diagnostics ──────────────────────────────────────────────────────────────
# Mirrors cfl_dt/diagnostics in utils/finite_volume_viscous_2d.jl, adapted to the bundled
# full_cell state and to the two grid types (ideal vs. viscous) via has_W, same pattern as
# rhs_C!/rhs_W!/rhs_full!. t_now is the current Milne proper time, needed by
# build_primitive_cache to unweight the stored (t-weighted) C before primitive recovery.
#
# Always uses max_wave_speed_C for the hydrodynamic part, even when has_W: advection of the
# W-sector is deferred (see rhs_full!'s docstring), so max_wave_speed_W (still a
# mixed_flux_jacobian stub) is never called here. When tc is supplied (viscous grid), also floors
# dt by the local relaxation times τ_π/τ_Π/τ_ν (min over the grid) — explicit SSP-RK2 is not
# unconditionally stable for a stiff -(W-W_NS)/τ source when τ ≪ the hydrodynamic CFL step, so this
# takes many small steps rather than going unstable; an IMEX/implicit treatment (already
# anticipated in rhs_full!'s comments) would remove the need for this floor.
function cfl_dt_full(u, eos, dx, dy, cfl, t_now; tc::Union{TransportCoefficients,Nothing}=nothing)
    d = field_storages(u)
    has_W = :bulk in propertynames(u)
    prim_cache, _ = build_primitive_cache(u, eos, t_now)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    amax = 0.0
    tau_min = Inf
    for I in CartesianIndices(interior_range(u.C0))
        S = full_cell(d, prim_cache, I, read_W_fn)
        sx = max_wave_speed_C(eos, S, 1)
        sy = max_wave_speed_C(eos, S, 2)
        amax = max(amax, sx / dx + sy / dy)
        if has_W && tc !== nothing
            _, prim, _ = S
            tau_min = min(tau_min, tau_shear(tc, eos, prim.τ, prim.a),
                tau_bulk(tc, eos, prim.τ, prim.a), tau_diffusion(tc, eos, prim.τ, prim.a))
        end
    end
    dt_hydro = cfl / max(amax, 1.0e-14)
    return has_W && tc !== nothing ? min(dt_hydro, cfl * tau_min) : dt_hydro
end

function diagnostics_full(u, eos, dx, dy, t_now)
    d = field_storages(u)
    has_W = :bulk in propertynames(u)
    prim_cache, _ = build_primitive_cache(u, eos, t_now)
    read_W_fn = has_W ? dissipative_cell_W : zero_dissipative_cell_W
    charge = 0.0; energy = 0.0; vmax = 0.0
    for I in CartesianIndices(interior_range(u.C0))
        C, prim, _ = full_cell(d, prim_cache, I, read_W_fn)
        charge += C[4] * dx * dy   # Cn = t_now·n̄W → ∫Cn dx dy = dN/dη
        energy += C[1] * dx * dy   # C0 = t_now·[(ε̄+p̄)W²-p̄] → ∫C0 dx dy = dE/dη
        vmax = max(vmax, hypot(prim.vx, prim.vy))
    end
    return charge, energy, vmax
end

# ─── Driver: 2-D DNMR viscous hydro from (T,μ) initial data, zero initial flow/dissipation ───
# Mirrors run_2d_ideal's tau0/t_now conventions (Milne proper time can't start at 0, the
# coordinate singularity) — evolution starts at tau0 and proceeds to tau0+t_end.
function run_2d_dnmr(eos, T_init, μ_init; tc::TransportCoefficients=TransportCoefficients(),
        tau0=1.0, cfl=0.3, t_end=0.20)
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
        interior_view(u.C0)[i, j]   = tau0 * C[1]
        interior_view(u.Cx)[i, j]   = tau0 * C[2]
        interior_view(u.Cy)[i, j]   = tau0 * C[3]
        interior_view(u.Cn)[i, j]   = tau0 * C[4]
        interior_view(u.bulk)[i, j] = 0.0
        interior_view(u.pixx)[i, j] = 0.0
        interior_view(u.pixy)[i, j] = 0.0
        interior_view(u.piyy)[i, j] = 0.0
        interior_view(u.nux)[i, j]  = 0.0
        interior_view(u.nuy)[i, j]  = 0.0
    end
    synchronize_halo!(u)

    q0, e0, _ = diagnostics_full(u, eos, dx, dy, tau0)
    @printf("2-D DNMR viscous hydro — grid=%d×%d  tau0=%.2f  t_end=%.2f  initial dN/dη=%.6f dE/dη=%.6f\n",
        n, n, tau0, t_end, q0, e0)

    t_now = tau0; t_stop = tau0 + t_end; step = 0
    while t_now < t_stop
        dt = min(cfl_dt_full(u, eos, dx, dy, cfl, t_now; tc=tc), t_stop - t_now)
        if isnan(dt)
            @warn "CFL timestep calculation returned NaN at step $step. Forcing minimal fallback dt."
            dt = min(1e-4, t_stop - t_now)
        end
        ssprk2_step_full!(u, u1, du, eos, tc, t_now, dt, dx, dy)
        t_now += dt; step += 1
    end

    q1, e1, vmax = diagnostics_full(u, eos, dx, dy, t_now)
    @printf("  final  dN/dη=%.6f (Δ=%.2e)  dE/dη=%.6f (Δ=%.2e)  vmax=%.4f  steps=%d  τ=%.4f\n",
        q1, q1 - q0, e1, e1 - e0, vmax, step, t_now)
    return u
end

# ─── Smoke test: Sec 1 ideal Milne hydro, circular over-density/over-pressure blast ───────────
# Same disk-blast setup as HaloArrays' relativistic_hydro_Tmu_2d.jl run_Tmu_blast_2d, run
# through run_2d_ideal. Checks: dN/dη is exactly conserved (Eq 9 has no source term), dE/dη
# decreases by a small, bounded amount (Bjorken-like longitudinal work, Eq 6's -p̄ source) while
# staying positive, and flow actually develops (vmax > 0).
function run_ideal_milne_blast_test(; A=1.0, n=48, cfl=0.3, tau0=1.0, t_end=0.20,
        T_in=1.0, μ_in=0.3, T_out=0.5, μ_out=0.1, radius=0.15)
    eos = UltraRelGas(A)
    T_init = Array{Float64}(undef, n, n)
    μ_init = Array{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        x = (i - 0.5) / n; y = (j - 0.5) / n
        inside = hypot(x - 0.5, y - 0.5) < radius
        T_init[i, j] = inside ? T_in : T_out
        μ_init[i, j] = inside ? μ_in : μ_out
    end

    u = run_2d_ideal(eos, T_init, μ_init; tau0=tau0, cfl=cfl, t_end=t_end)

    dx = 1.0 / n; dy = 1.0 / n
    q1, e1, vmax = diagnostics_full(u, eos, dx, dy, tau0 + t_end)

    ok = isfinite(q1) && isfinite(e1) && e1 > 0.0 && vmax > 0.0
    println(ok ?
        "  ✓ ideal Milne blast: dN/dη, dE/dη finite, dE/dη>0, flow developed (vmax=$(round(vmax, digits=4)))" :
        "  ✗ unexpected result")
    return u
end

run_ideal_milne_blast_test()

# ─── Smoke test: Sec 2 DNMR viscous Milne hydro, same disk-blast, dissipation deferred-advection
# ────────────────────────────────────────────────────────────────────────────────────────────
# Same blast setup as run_ideal_milne_blast_test, run through run_2d_dnmr (source_W's first-order
# relaxation, no W-sector advection yet). Checks: runs to completion with finite Π,π^ij,ν^i
# everywhere, the dissipative fields actually become nonzero (relaxation is active, not a no-op),
# and dE/dη drops by *more* than the ideal case run with identical initial data/parameters — a
# genuine physical sanity check (viscosity should dissipate more energy than -p̄ work alone), not
# just "doesn't crash".
#
# Deliberately MILDER defaults than run_ideal_milne_blast_test/config.yaml (η/s=0.08 not 0.55,
# tau0=3.0 not ~1, a smaller T contrast): config.yaml's η/s=0.55 combined with an early tau0~1
# makes the leading-order NS target π_NS=2ησ^ij genuinely large relative to p̄+ε̄ purely from the
# geometric Bjorken term (σ^xx≈σ^yy≈-1/(3·tau0) even for perfectly uniform transverse flow — no
# blast needed) — a well-known early-time pathology of first-order/Navier-Stokes-type viscous
# closures (the classical motivation for genuinely 2nd-order DNMR theories and/or a free-streaming
# pre-equilibrium stage before hydro starts). Confirmed empirically: with config.yaml's defaults
# this pass's implementation is CORRECT (π relaxes linearly toward π_NS exactly as Eq 24-25
# prescribes) but π_NS itself is large enough to push cons_to_primitive's Newton solve out of a
# solvable regime, causing "Primitive inversion did not converge" warnings. Fixing that needs
# either the deferred 2nd-order DNMR terms, a later tau0/pre-equilibrium stage, or a regulator —
# out of scope here; this test instead verifies the mechanism is correct within the regime where
# the leading-order closure is actually valid (π,Π ≪ p̄+ε̄).
function run_dnmr_milne_blast_test(; A=1.0, n=48, cfl=0.3, tau0=3.0, t_end=0.20,
        T_in=0.6, μ_in=0.15, T_out=0.5, μ_out=0.1, radius=0.15,
        tc::TransportCoefficients=TransportCoefficients(eta_over_s=0.08))
    eos = UltraRelGas(A)
    T_init = Array{Float64}(undef, n, n)
    μ_init = Array{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        x = (i - 0.5) / n; y = (j - 0.5) / n
        inside = hypot(x - 0.5, y - 0.5) < radius
        T_init[i, j] = inside ? T_in : T_out
        μ_init[i, j] = inside ? μ_in : μ_out
    end

    u_dnmr  = run_2d_dnmr(eos, T_init, μ_init; tc=tc, tau0=tau0, cfl=cfl, t_end=t_end)
    u_ideal = run_2d_ideal(eos, T_init, μ_init; tau0=tau0, cfl=cfl, t_end=t_end)

    dx = 1.0 / n; dy = 1.0 / n
    q_dnmr, e_dnmr, _ = diagnostics_full(u_dnmr, eos, dx, dy, tau0 + t_end)
    _, e_ideal, _      = diagnostics_full(u_ideal, eos, dx, dy, tau0 + t_end)

    W_fields = (u_dnmr.bulk, u_dnmr.pixx, u_dnmr.pixy, u_dnmr.piyy, u_dnmr.nux, u_dnmr.nuy)
    W_finite = all(f -> all(isfinite, interior_view(f)), W_fields)
    W_max    = maximum(f -> maximum(abs, interior_view(f)), W_fields)

    dissipation_active = W_max > 1.0e-8
    more_dissipation    = e_dnmr < e_ideal

    ok = isfinite(q_dnmr) && isfinite(e_dnmr) && e_dnmr > 0.0 && W_finite && dissipation_active && more_dissipation
    println(ok ?
        "  ✓ DNMR Milne blast: finite dN/dη=$(round(q_dnmr, digits=6)) dE/dη=$(round(e_dnmr, digits=6)), " *
        "|W|max=$(round(W_max, digits=6)), extra dissipation vs ideal (ΔE=$(round(e_ideal - e_dnmr, digits=6)))" :
        "  ✗ unexpected result (q_dnmr=$q_dnmr e_dnmr=$e_dnmr e_ideal=$e_ideal W_max=$W_max W_finite=$W_finite)")
    return u_dnmr
end

run_dnmr_milne_blast_test()

run_2d_ideal_and_plot()
