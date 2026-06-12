function nhard_profile(x,y,ncoll;sigma = 1,ncoll_norm = 2*0.4087/(70.)/0.4)
    ncoll_norm*ncoll*exp(-(x^2 + y^2)/(2*sigma^2))/(2*pi*sigma^2)
end

function charm_number_hard(ncoll, tau0;xmax=20.,ncoll_norm = 2*0.4087/(70.)/0.4)
    domain = ([-xmax,-xmax],[xmax,xmax])
    function f(u,p) 
        ncoll = p
        nhard_profile(u[1],u[2],ncoll,ncoll_norm=ncoll_norm)*tau0
    end
    p = (ncoll)
    prob = IntegralProblem(f,domain,p)
    result = solve(prob, HCubatureJL(), reltol=1e-3, abstol=1e-6)
    return result
end

function fugacity(x,y,T,eos; sigma = 1.)
    if hypot(x,y)<sigma
        return log(nhard_profile(x, y, ncoll; sigma = sigma, ncoll_norm = ncoll_norm)/(thermodynamic(T,0.0,eos.hadron_list).pressure))
    else
        return 0.0
    end
end

function fug_(profile, ncoll, eos, discretization; sigma, ncoll_norm = 2*0.4087/(70.)/0.4)   
    function fugacity(x,y,T)
        return log(nhard_profile(x, y, ncoll; sigma = sigma, ncoll_norm = ncoll_norm)/(thermodynamic(T,0.0,eos.hadron_list).pressure))
    end
    fug_map = map(zip(discretization.grid,profile)) do x
            return fugacity(x[1][1],x[1][2],x[2])
    end

    return fug_map
end

function ncoll_fluctuating_thickness(x::Num1, y::Num2, f::Participant{T, S, V, M, C, D, F}, N1::Int, N2::Int, σin::Float64) where {Num1 <: Real, Num2 <: Real, T, S, V, M, C, D, F}

    part1 = f.part1 #already includes the impact parameter shift
    part2 = f.part2
    shape1 = f.shape1
    shape2 = f.shape2
    w = f.sub_nucleon_width
    p = f.p

    ta = zero(eltype(f))
    tb = zero(eltype(f))

    @inbounds @fastmath for i in eachindex(part1)
        pa_x, pa_y = part1[i]
        ga = shape1[i]
        ta +=  ga * MonteCarloGlauber.Tp(x - pa_x, y - pa_y, w) 
    end

    @inbounds @fastmath for i in eachindex(part2)
        pa_x, pa_y = part2[i]
        ga = shape2[i]
        tb += ga * MonteCarloGlauber.Tp(x - pa_x, y - pa_y, w)
    end

    return σin*ta*tb 
end


function T_A(x::Num1, y::Num2, f::Participant{T, S, V, M, C, D, F}) where {Num1 <: Real, Num2 <: Real, T, S, V, M, C, D, F}

    part1 = f.part1 #already includes the impact parameter shift
    shape1 = f.shape1
    w = f.sub_nucleon_width
   
    ta = zero(eltype(f))
   
    @inbounds @fastmath for i in eachindex(part1)
        pa_x, pa_y = part1[i]
        ga = shape1[i]
        ta +=  ga * MonteCarloGlauber.Tp(x - pa_x, y - pa_y, w) 
    end

    return ta
end

# hcubature(b->T_A(b[1],b[2],event),(-20.0, -20.0), (20.0, 20.0), rtol=1e-3, atol=1e-3) == Npart_A


@inline function mean2(a, b)
    return a * b
end

function ncoll_(event, discretization, N1, N2, σin)   
   
    ncoll_map = map(discretization.grid) do x
            return ncoll_fluctuating_thickness(x[1], x[2], event, N1, N2, σin)
    end

    return ncoll_map
end

function nhard_profile_(event, discretization, N1, N2, σin, norm; offset = 0.001)
    ncoll_map = ncoll_(event, discretization, N1, N2, σin)  
    return ncoll_map.*norm .+ offset
end

function fug_(event, discretization, N1, N2, σin, dσ_QQdy, tau0, eos; offset = 10e-5)   
    ccbar_norm = 2. /tau0/σ_in*dσ_QQdy
    fug_map = map(zip(discretization.grid,profile)) do x
        nhard_= ncoll_fluctuating_thickness(x[1][1], x[1][2], event, N1, N2, σin) * ccbar_norm
        log((nhard_+ offset)/(thermodynamic(x[2],0.0,eos.hadron_list).pressure+offset))
    end

    return fug_map
end

function trento_event_eos(profile;norm=10,exp_tail = false, xgrid = -10:0.5:10, ygrid = -10:0.5:10, kwarg...)
 
    temperature_profile = InverseFunction(x->pressure_derivative(x,Val(1),FluiduMEoS())).(norm.*profile)
    
    if exp_tail
        temp_exp = [Fluidum.exponential_tail_pointlike(temperature_funct, x, y; offset=0.01) for x in xgrid, y in ygrid]
        temp_exp_funct = linear_interpolation((xgrid,ygrid), temp_exp; extrapolation_bc=Flat()) 
        return  temp_exp_funct
    else
        return temperature_profile
    end
  
end

function set_array(phi::AbstractArray,express::S,disc::DiscreteFields{T,total_dimensions,space_dimension,N_field,Sizes_ghosted,Sizes,Lengths,DXS,M,S,N_field2}) where {S,T,total_dimensions,space_dimension,N_field,Sizes_ghosted,Sizes,Lengths,DXS,M,N_field2}
    array=Fluidum.get_array(disc)
    i=Fluidum.get_index(express,disc.fields)
    for I in disc.index_structure.interior
        array[i,I]= phi[I]
    end
    Fluidum.boundary_condition!(array,disc)
    return array
end

function set_array!(array,phi::AbstractArray,express::S,disc::DiscreteFields{T,total_dimensions,space_dimension,N_field,Sizes_ghosted,Sizes,Lengths,DXS,M,S,N_field2}) where {S,T,total_dimensions,space_dimension,N_field,Sizes_ghosted,Sizes,Lengths,DXS,M,N_field2}
    i=Fluidum.get_index(express,disc.fields)
    for I in disc.index_structure.interior
        array[i,I]= phi[I]
    end

    Fluidum.boundary_condition!(array,disc)

    return array
end
