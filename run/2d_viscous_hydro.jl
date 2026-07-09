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

# ==============================================================================
# 2. Forward Hydrodynamics Transformation (Primitives -> Conserved)
# ==============================================================================
@inline function cons_from_prim(eos, T, μ, vx, vy, bulk_Pi, pixx, pixy, piyy)
    W = 1.0 / sqrt(1.0 - vx^2 - vy^2)
    p = pressure(eos, T, μ)
    n = charge_density(eos, T, μ)
    eps = enthalpy_density(eos, T, μ) - p  # ε = w - p
    
    # Total pressure includes bulk viscous pressure: P_total = p + Π
    P_tot = p + bulk_Pi
    w_tot = eps + P_tot
    
    # D = n * W
    N  = n * W
    
    # S_i = [ε + P_tot] * W^2 * v_i + π_ij * v^j
    Mx = w_tot * W^2 * vx + (pixx * vx + pixy * vy)
    My = w_tot * W^2 * vy + (pixy * vx + piyy * vy)
    
    # E = [ε + P_tot] * W^2 - P_tot + v_i * v_j * π^ij
    v_v_pi = vx^2 * pixx + 2.0 * vx * vy * pixy + vy^2 * piyy
    E      = w_tot * W^2 - P_tot + v_v_pi
    
    return SVector(N, Mx, My, E)
end

# ==============================================================================
# 3. Backward Inversion (Conserved -> Primitives Nonlinear Core)
# ==============================================================================
function cons_to_primitive_rhds(u, p_tuple)
    vx, vy, T, μ = u              
    N, Mx, My, E, bulk_Pi, pixx, pixy, piyy, eos = p_tuple     
    
    p   = pressure(eos, T, μ)
    n   = charge_density(eos, T, μ)
    eps = enthalpy_density(eos, T, μ) - p
    
    v2 = vx^2 + vy^2
    v_v_pi = vx^2 * pixx + 2.0 * vx * vy * pixy + vy^2 * piyy
    S_v = Mx * vx + My * vy
    
    # Core factor for F_i: [E + p + Π - v_k v_l π^{kl}]
    core_factor = E + p + bulk_Pi - v_v_pi
    
    # F_i Residuals
    fx = core_factor * vx + (pixx * vx + pixy * vy) - Mx
    fy = core_factor * vy + (pixy * vx + piyy * vy) - My
    
    # F_E Residual
    fe = eps - (E - S_v)
    
    # F_D Residual
    fn = n - N * sqrt(1.0 - v2)
    
    return SA.SVector{4}(fx, fy, fe, fn)
end

# ==============================================================================
# 4. Backward Inversion Wrapper & Property Reconstruction
# ==============================================================================
function cons_to_primitive(U, DISSP, eos)
    N, Mx, My, E = U
    bulk_Pi, pixx, pixy, piyy = DISSP
    
    u0 = SA.SVector{4}(0.0, 0.0, 0.5, 1.0)
    p_tuple = (N, Mx, My, E, bulk_Pi, pixx, pixy, piyy, eos)

    prob = NonlinearProblem(cons_to_primitive_rhds, u0, p_tuple)
    sol = solve(prob, SNLS.SimpleNewtonRaphson())
    
    if SciMLBase.successful_retcode(sol.retcode) == false
        @warn "Primitive inversion did not converge"
        @show sol.retcode
    end
    
    vx, vy, T, μ = sol
    
    v2 = vx^2 + vy^2
    W = 1.0 / sqrt(1.0 - v2)
    
    p_eq   = pressure(eos, T, μ)
    n_eq   = charge_density(eos, T, μ)
    eps_eq = enthalpy_density(eos, T, μ) - p_eq
    
    pi0x = vx * pixx + vy * pixy
    pi0y = vx * pixy + vy * piyy
    pi00 = vx^2 * pixx + 2.0 * vx * vy * pixy + vy^2 * piyy
    
    return T, μ, vx, vy, W, p_eq, eps_eq, n_eq, pi0x, pi0y, pi00
end

# ==============================================================================
# 5. Automated Consistency Verification Test Suite
# ==============================================================================
function run_consistency_test()
    eos = UltraRelGas(1.0)

    @testset "Standard Hydrodynamics Inversion Consistency (2D+1)" begin
        T_ref  = 0.40
        μ_ref  = 0.15
        vx_ref = 0.35
        vy_ref = -0.25
        
        bulk_Pi_ref = 0.005
        pixx_ref    = 0.012
        pixy_ref    = -0.005
        piyy_ref    = 0.018
        DISSP = (bulk_Pi_ref, pixx_ref, pixy_ref, piyy_ref)

        v2_ref  = vx_ref^2 + vy_ref^2
        W_ref   = 1.0 / sqrt(1.0 - v2_ref)
        p_ref   = pressure(eos, T_ref, μ_ref)
        n_ref   = charge_density(eos, T_ref, μ_ref)
        eps_ref = enthalpy_density(eos, T_ref, μ_ref) - p_ref
        
        pi0x_ref = vx_ref * pixx_ref + vy_ref * pixy_ref
        pi0y_ref = vx_ref * pixy_ref + vy_ref * piyy_ref
        pi00_ref = vx_ref^2 * pixx_ref + 2.0 * vx_ref * vy_ref * pixy_ref + vy_ref^2 * piyy_ref

        # Forward Pass
        U_conserved = cons_from_prim(eos, T_ref, μ_ref, vx_ref, vy_ref, bulk_Pi_ref, pixx_ref, pixy_ref, piyy_ref)
        
        # Backward Pass
        T_rec, μ_rec, vx_rec, vy_rec, W_rec, p_rec, eps_rec, n_rec, pi0x_rec, pi0y_rec, pi00_rec = 
            cons_to_primitive(U_conserved, DISSP, eos)

        @testset "Core Primitives Inversion" begin
            @test vx_rec ≈ vx_ref atol=1e-8
            @test vy_rec ≈ vy_ref atol=1e-8
            @test T_rec  ≈ T_ref  atol=1e-8
            @test μ_rec  ≈ μ_ref  atol=1e-8
        end

        @testset "Reconstructed Thermodynamic Quantities" begin
            @test W_rec   ≈ W_ref   atol=1e-8
            @test p_rec   ≈ p_ref   atol=1e-8
            @test eps_rec ≈ eps_ref atol=1e-8
            @test n_rec   ≈ n_ref   atol=1e-8
        end

        @testset "Shear Tensor Time-Components" begin
            @test pi0x_rec ≈ pi0x_ref atol=1e-8
            @test pi0y_rec ≈ pi0y_ref atol=1e-8
            @test pi00_rec ≈ pi00_ref atol=1e-8
        end
    end
end

run_consistency_test()