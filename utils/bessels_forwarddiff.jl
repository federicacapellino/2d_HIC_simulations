using ForwardDiff
import Fluidum.Bessels # or import SpecialFunctions depending on what you use

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
