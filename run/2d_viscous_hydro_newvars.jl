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
@inline function cons_from_prim_q(eos, qx, qy, τ, a, bulk_Pi, pixx, pixy, piyy, nux, nuy)
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
    
    # D = n_B * W + ν^0 where ν^0 = v_i * ν^i
    nu0 = vx * nux + vy * nuy
    D   = n_B * W + nu0
    
    # S_i = [ε + P_total] * W^2 * v_i + π_ij * v^j
    Mx = w_tot * W^2 * vx + (pixx * vx + pixy * vy)
    My = w_tot * W^2 * vy + (pixy * vx + piyy * vy)
    
    # E = [ε + P_total] * W^2 - P_total + v_i * v_j * π^ij
    v_v_pi = vx^2 * pixx + 2.0 * vx * vy * pixy + vy^2 * piyy
    E      = w_tot * W^2 - P_tot + v_v_pi
    
    return SVector(D, Mx, My, E)
end

# ==============================================================================
# 3. Backward Inversion (Conserved -> Primitives Nonlinear 4D Core)
# ==============================================================================
function cons_to_primitive_rhds_q(u, p_tuple)
    qx, qy, τ, a = u              
    D, Mx, My, E, bulk_Pi, pixx, pixy, piyy, nux, nuy, eos = p_tuple     
    
    q2 = qx^2 + qy^2
    W  = sqrt(1.0 + q2)
    
    pr  = bar_p(eos, τ, a)
    en  = bar_eps(eos, τ, a)
    nB  = bar_nB(eos, τ, a)
    
    q_q_pi = qx^2 * pixx + 2.0 * qx * qy * pixy + qy^2 * piyy
    Q_q    = q_q_pi / (1.0 + q2)
    
    S_q = Mx * qx + My * qy
    q_nu = qx * nux + qy * nuy
    
    # Core factor for F_i: [E + p + Π - Q(q)]
    core_factor = E + pr + bulk_Pi - Q_q
    
    # F_i Residuals
    fx = (1.0 / W) * (core_factor * qx + (pixx * qx + pixy * qy)) - Mx
    fy = (1.0 / W) * (core_factor * qy + (pixy * qx + piyy * qy)) - My
    
    # F_E Residual
    fe = en - E + (S_q / W)
    
    # F_D Residual
    fn = nB - (D / W) + (q_nu / (1.0 + q2))
    
    return SA.SVector{4}(fx, fy, fe, fn)
end

# ==============================================================================
# 4. Backward Inversion Wrapper & Property Reconstruction
# ==============================================================================
function cons_to_primitive_q(U, DISSP, eos)
    D, Mx, My, E = U
    bulk_Pi, pixx, pixy, piyy, nux, nuy = DISSP
    
    u0 = SA.SVector{4}(0.0, 0.0, log(0.5), 1.0 / 0.5)
    p_tuple = (D, Mx, My, E, bulk_Pi, pixx, pixy, piyy, nux, nuy, eos)

    prob = NonlinearProblem(cons_to_primitive_rhds_q, u0, p_tuple)
    sol = solve(prob, SNLS.SimpleNewtonRaphson())
    
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

# ==============================================================================
# 5. Automated Consistency Verification Test Suite
# ==============================================================================
function run_q_variable_consistency_test()
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
        U_conserved = cons_from_prim_q(
            eos, qx_ref, qy_ref, τ_ref, a_ref, 
            bulk_Pi_ref, pixx_ref, pixy_ref, piyy_ref, nux_ref, nuy_ref
        )
        
        # Backward Pass
        qx_rec, qy_rec, τ_rec, a_rec, W_rec, vx_rec, vy_rec, T_rec, μ_rec, 
        p_rec, eps_rec, n_rec, pi0x_rec, pi0y_rec, pi00_rec, nu0_rec = 
            cons_to_primitive_q(U_conserved, DISSP, eos)

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

run_q_variable_consistency_test()