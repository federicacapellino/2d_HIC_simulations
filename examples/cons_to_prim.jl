using Fluidum
include("../utils/finite_volume.jl")

particle_list = "./particles_D0.data";
eos = Heavy_Quark(particle_list, 10.)
    

N, Mx, My, E = primitive_to_cons(0.1, 0.1, 0.1, 0.1, eos)

u0   = SA.SVector{2}(0.1, 0.1)
M2   = hypot(Mx, My)^2
cons    = (E, M2, N, eos) # Passed eos along into the parameter tuple

prob = NonlinearProblem(cons_to_primitive_rhds, u0, cons)
sol  = solve(prob, SNLS.SimpleNewtonRaphson())
