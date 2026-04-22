using Fluidum
using MonteCarloGlauber
using StaticArrays
using LinearAlgebra
using Integrals
using Cuba
using JLD2
using HDF5
using MuladdMacro
using OhMyThreads
using Base.Threads
using YAML

using Plots
using LaTeXStrings
default(aspect_ratio = :equal,
    size = (300, 300),
    dpi = 200,linewidth = 2, axis=true,
    markersize = 8
    , left_margin = 1Plots.mm,
    right_margin = 7Plots.mm,
    top_margin = 1Plots.mm,
    bottom_margin =1Plots.mm, legendfont = "sans-serif", titlefontsize = 12, tickfontsize = 10, legendfontsize = 10)

import Fluidum.QuadGK:quadgk
import Fluidum.HCubature:hcubature
@info "Julia threads" nthreads()

kernels = Fluidum.root_kernels

include("../utils/MCglauber.jl")
include("../utils/hdf5_io.jl")
include("../utils/observables.jl")
include("../utils/fastreso.jl")

# the convention here are T, ux, uy, piyy, pizz, pixy, piB this has to match with the matrix defined
twod_visc_hydro = Fields(
    NDField((:even, :ghost), (:even, :ghost), :temperature),
    NDField((:ghost, :ghost), (:ghost, :ghost), :ux),
    NDField((:ghost, :ghost), (:ghost, :ghost), :uy),
    NDField((:ghost, :ghost), (:even, :ghost), :piyy),
    NDField((:even, :ghost), (:even, :ghost), :pizz),
    NDField((:ghost, :ghost), (:ghost, :ghost), :pixy),
    NDField((:even, :ghost), (:even, :ghost), :piB),
    NDField((:even, :ghost), (:even, :ghost), :mu),
    NDField((:ghost, :ghost), (:ghost, :ghost), :nux),
    NDField((:ghost, :ghost), (:ghost, :ghost), :nuy)
)

config = YAML.load_file("config.yaml")

physics_params = config["physics parameter"]
run_params = config["run parameter"]

# Run parameters
Nev = run_params["Nev"]
const checkpoint_interval = run_params["checkpoint_interval"]
const checkpoint_file = run_params["checkpoint_file"]
const gridpoints = run_params["gridpoints"]
const xmax = run_params["xmax"]
const wavenum_m = run_params["wavenum"]

n_batches = ceil(Int, Nev / checkpoint_interval)

# Fluid evolution parameters
const tau0 = physics_params["fluid evolution"]["tau0"]
const tmax = physics_params["fluid evolution"]["tmax"]
const norm = physics_params["fluid evolution"]["norm"]
const eta_over_s = physics_params["fluid evolution"]["eta_over_s"]
const zeta_over_s = physics_params["fluid evolution"]["zeta_over_s"]
const DsT = physics_params["fluid evolution"]["DsT"]
const Tfo = physics_params["fluid evolution"]["Tfo"]

# Charm parameters
const dσ_QQdy = physics_params["charm parameter"]["dσ_QQdy"]
const σ_in = physics_params["charm parameter"]["σ_in"]

# Initial condition parameters
const s_NN = physics_params["initial conditions"]["s_NN"]
const w = physics_params["initial conditions"]["w"]
const k = physics_params["initial conditions"]["k"]
const p = physics_params["initial conditions"]["p"]
b = physics_params["initial conditions"]["b"]

# Additional parameters
const tau_eta_par = physics_params["additional parameters"]["tau_eta_par"]
const tau_zeta_par = physics_params["additional parameters"]["tau_zeta_par"]
const hq_mass = physics_params["additional parameters"]["hq_mass"]

# Set up nuclear parameters
spec = physics_params["initial conditions"]["n1"]  # e.g. "Lead()"
const n1 = eval(Meta.parse(spec))
spec = physics_params["initial conditions"]["n2"]  # e.g. "Lead()"
const n2 = eval(Meta.parse(spec))

grid_spacing_fine_ends(L,beta,epsilon) = L/2*(1-tanh(beta*(1-2*epsilon))/tanh(beta))
grid_spacing_fine_end_coarse_end(L,beta,epsilon) = L*(1-tanh(beta*(1-epsilon))/tanh(beta))

# Create discretization
discretization = CartesianDiscretization(Fluidum.SymmetricInterval(gridpoints, xmax), Fluidum.SymmetricInterval(gridpoints, xmax))
#discretization = CartesianDiscretization(Fluidum.CartesianDiscretization((gridpoints,),(promote(-xmax,xmax),), 1.), 
#Fluidum.CartesianDiscretization((gridpoints,),(promote(-xmax,xmax),), 1.))

# Prepare the field with the discretization
twod_visc_hydro_discrete = DiscreteFields(twod_visc_hydro, discretization, Float64)

if b == "minBias"
    participants=Participants(n1,n2,w,s_NN,k,p)
else
    b_tuple = eval(Meta.parse(b))
    participants=Participants(n1,n2,w,s_NN,k,p,b_tuple)
end


Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_211_total_T0.1560_Fj.out"))
const particle_full_π = particle_full("pion", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_2212_total_T0.1560_Fj.out"))
const particle_full_p = particle_full("proton", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/PDGid_321_total_T0.1560_Fj.out"))
const particle_full_k = particle_full("kaon", Fj[4], 1, 0, Fj[1])
Fj = fastreso_reader(joinpath(kernels, "./kernels/Dc1865zer_total_T0.1560_Fj.out"))
const particle_full_D0 = particle_full("D0", Fj[4], 1, 1, Fj[1])


species_list = [particle_full_π, particle_full_p, particle_full_k, particle_full_D0]


#grid and temperature
begin
discretization = twod_visc_hydro_discrete.discretization
event = rand(participants)
ncoll_event = event.n_coll;
mult, x_com, y_com = center_of_mass(event, 100, 50)
xcm = x_com / mult
ycm = y_com / mult
profile = map(discretization.grid) do y
    y = y .+ (xcm, ycm)
    event(y...)
    end
temperature_func = trento_event_eos(profile, norm = norm, exp_tail = false) .+0.01
end


xgrid = getindex.(discretization.grid[:,1],1)

heatmap(xgrid, xgrid, temperature_func, xlabel = "x (fm)", ylabel = "y (fm)", title = L"T \ \mathrm{(GeV)}, \tau =\tau_0 ", seriescolor = cgrad(:inferno, [0, 0.156, 0.6]), clim = (0.,0.6))
minimum(temperature_func)
savefig("temperature_profile.pdf")

x_coord_1 = getindex.(event.part1,1)
y_coord_1 = getindex.(event.part1,2)

scatter(x_coord_1,y_coord_1)

x_coord_1 = getindex.(event.part2,1)
y_coord_1 = getindex.(event.part2,2)

scatter!(x_coord_1,y_coord_1)

part1 = event.part1
shape1 = event.shape1
part2 = event.part2
shape2 = event.shape2

function ta_f(x,y,part1,shape1)
ta = 0.
@inbounds @fastmath for i in eachindex(part1)
        pa_x, pa_y = part1[i]
        ga = shape1[i]
        ta +=  ga * MonteCarloGlauber.Tp(x - pa_x, y - pa_y, w) #TODO check this shift
    end
    return ta
end

ta_map = map(Iterators.product(-10:0.1:10, -10:0.1:10)) do (x, y)
    ta_f(x, y, part1, shape1)
end
tb_map = map(Iterators.product(-10:0.1:10, -10:0.1:10)) do (x, y)
    ta_f(x, y, part2, shape2)
end

heatmap(ta_map)
heatmap(tb_map)

#Fluid properties setup
begin
    @show ccbar = 2* ncoll_event * dσ_QQdy / σ_in #charm pair number density at tau0
    ccbar_norm = 2 * dσ_QQdy / σ_in / tau0
    particle_list = "./particles_D0.data";
    eos = Heavy_Quark(readresonancelist(;name_file=particle_list), ccbar)
    viscosity = QGPViscosity(eta_over_s, tau_eta_par)
    bulk = SimpleBulkViscosity(zeta_over_s, tau_zeta_par)
    viscosity = ZeroViscosity()
    bulk = ZeroBulkViscosity()
    diffusion = ZeroDiffusion() 
    fluidproperty = FluidProperties(eos, viscosity, bulk, diffusion)
end


#ncoll
begin
    ncoll_int=hcubature(b->ncoll_fluctuating_thickness(b[1],b[2],event,n1.N_nucleon,n2.N_nucleon,σ_in),(-20.0, -20.0), (20.0, 20.0), rtol=1e-3, atol=1e-3)
    @show ccbar = 2* ncoll_int[1] * dσ_QQdy / σ_in #charm pair number density at tau0
    println("ncoll from event: $ncoll_event, ncoll from integration: $(ncoll_int[1])")
    if isapprox(ncoll_event, ncoll_int[1], atol=0.1*ncoll_event)
        println("ncoll from event and integration are compatible!")
    end
end

#nhard
begin
    b = impactParameter(event)
    sigma = max(((event.R1+event.R2)-b)/2.,0.01);
    nhard_analytic = map(discretization.grid) do y
        nhard_profile(y...,ncoll_event, sigma=sigma, ncoll_norm = ccbar_norm)
        end
    #display(heatmap(nhard_analytic))
    nhard_fluct = nhard_profile_(event, discretization, n1.N_nucleon, n2.N_nucleon, σ_in,ccbar_norm, offset=0.001)
    #@show 0.001*0.4*0.4*102*102
    #display(heatmap(nhard_fluct))
    println("number of ccbar $ccbar")
    println("nhard analytic integral: $(sum(nhard_analytic)*0.4*0.4*tau0), nhard fluctuating integral: $(sum(nhard_fluct)*0.4*0.4*tau0)")
    if isapprox(sum(nhard_analytic),sum(nhard_fluct), atol=0.1*sum(nhard_analytic))
        println("integrals of nhard analytic and fluctuating are compatible!")
    end
    println("nhard analytic min: $(minimum(nhard_analytic)), max: $(maximum(nhard_analytic))")
    println("nhard fluctuating min: $(minimum(nhard_fluct)), max: $(maximum(nhard_fluct))")
end;

begin
    n_thermal = map(CartesianIndices(temperature_func)) do x
      i, j = Tuple(x)
      thermodynamic(temperature_func[i, j],0.,eos.hadron_list).pressure
      end
    println("nhard fluctuating min: $(minimum(nhard_fluct)), max: $(maximum(nhard_fluct))")
    println("n_thermal min: $(minimum(n_thermal)), max: $(maximum(n_thermal))")
    gamma = nhard_fluct ./ n_thermal;
    mu_over_T = log.(gamma);
    println("ratio min: $(minimum(gamma)), max: $(maximum(gamma))")
end


sum(n_thermal)*tau0*tau0*dx*dx
sum(nhard_fluct.*2)*tau0*tau0*dx*dx

heatmap(gamma)
heatmap(mu_over_T)
heatmap(temperature_func)

#fugacity and hydro density
begin
  #  fug_func = fug_(temperature_func, ncoll_event, eos, discretization; sigma = sigma, ncoll_norm = ccbar_norm)
  #  fu = fug_(event, discretization, n1.N_nucleon,n1.N_nucleon, σ_in, ccbar_norm,eos)

  #  n_analytic = map(CartesianIndices(temperature_func)) do x
  #      i, j = Tuple(x)
  #      thermodynamic(temperature_func[i, j],fug_func[i,j],eos.hadron_list).pressure
  #      end
    n_fluct = map(CartesianIndices(temperature_func)) do x
        i, j = Tuple(x)
        thermodynamic(temperature_func[i, j],mu_over_T[i,j],eos.hadron_list).pressure 
        end
    println("number of ccbar $ccbar")
    println("nhard integral: $(sum(nhard_fluct)*0.4*0.4*tau0), n hydro integral: $(sum(n_fluct)*0.4*0.4*tau0)")
    if isapprox(sum(n_fluct),sum(nhard_fluct), atol=0.1*sum(nhard_fluct))
        println("integrals of nhard and nhydro are compatible!")
    end
   

end

minimum(n_fluct)
heatmap(n_fluct)
#compute center of mass
println("     🎲 Glauber mult: $(round(mult, digits = 2)) | Ncoll: $ncoll_event")
tspan = (tau0, 30.)
phi = set_array(temperature_func, :temperature, twod_visc_hydro_discrete)  #maybe offset too large?
set_array!(phi, mu_over_T, :mu, twod_visc_hydro_discrete)

heatmap(phi[1,:,:])
heatmap(phi[8,:,:])
minimum(phi[1,:,:])
minimum(phi[8,:,:])


begin
    n_0 = map(CartesianIndices(temperature_func)) do x
      i, j = Tuple(x)
      temp = phi[1,:,:]
      fug = phi[8,:,:]
      thermodynamic(temp[i, j],fug[i,j],eos.hadron_list).pressure
      end
end

sum(n_0)*0.4*0.4*dx*dx
default(aspect_ratio = :equal,
    size = (300, 300),
    dpi = 200,linewidth = 2, axis=true,
    markersize = 8
    , left_margin = 1Plots.mm,
    right_margin = 7Plots.mm,
    top_margin = 1Plots.mm,
    bottom_margin =1Plots.mm, legendfont = "sans-serif", titlefontsize = 12, tickfontsize = 10, legendfontsize = 10)
xgrid = getindex.(discretization.grid[:,1],1)
heatmap(xgrid, xgrid, n_0, xlims=(-20.,20.), ylims=(-20.,20.))

simulation_pars = (
    dσ_QQdy = dσ_QQdy,
    viscosity = viscosity.ηs,
    bulk = bulk.ζs,
    diffusion = 0.0, #diffusion.DsT,
    t0 = tspan[1],
    Tfo = Tfo,
    species_list = name.(species_list),
    pTlist = pt_lists(species_list),
    length_pTlist = length.(pt_list.(species_list)),
    wavenum_m = wavenum_m,
)

result = Fluidum.isosurface(twod_visc_hydro_discrete, Fluidum.matrix2d_visc_HQ!, fluidproperty, phi, tspan, :temperature, Tfo)
cha = Fluidum.Chart(Fluidum.Surface(result[:surface]), (t, x, y) -> Fluidum.SVector{2}(atan(t, hypot(y, x)), atan(y, x)))
fo_bg = Fluidum.freezeout_interpolation(cha, sort_index = 2, ndim_tuple = 50)


#fields at freezout
field = fo_bg.fields
llim = Fluidum.leftbounds(field)
hlim = Fluidum.rightbounds(field)
alpha = range(llim[1], hlim[1], length=50)
beta = range(llim[2], hlim[2], length=50)
field_on_grid = map(CartesianIndices((10,length(alpha), length(beta)))) do I
    Nfield, i, j = Tuple(I)
    field(alpha[i], beta[j])[Nfield]
    end
heatmap(field_on_grid[8,:,:])
point = fo_bg.x
coord_on_grid = map(CartesianIndices((length(alpha), length(beta)))) do I
    i, j = Tuple(I)
    point(alpha[i], beta[j])[1]
    end

    plot(alpha, beta,coord_on_grid[1,:,:],:surface)
heatmap(coord_on_grid[1,:,:])
heatmap(coord_on_grid[1,:,:])


t_fo=map(result.surface) do x
    x.X[1]
end

x_fo=map(result.surface) do x
    x.X[2]
end

y_fo=map(result.surface) do x
    x.X[3]
end

r_fo=map(result.surface) do x
    sqrt(x.X[2]^2+x.X[3]^2)
end


alpha_fo =map(result.surface) do x
    sqrt(x.X[1]^2+x.X[2]^2+x.X[3]^2)
end

phi_fo =map(result.surface) do x
    atan(x.X[3],x.X[2])
end

using Surrogates
n_samples = 100

coordinates = map(result.surface) do x
   atan(x.X[1],hypot(x.X[3],x.X[2])),atan(x.X[3],x.X[2])
end
lower_bound = (extrema(alpha_fo)[1],extrema(phi_fo)[1])
upper_bound = (extrema(alpha_fo)[2],extrema(phi_fo)[2])

radial_basis=RadialBasis(coordinates, t_fo,lower_bound, upper_bound)
x = range(lower_bound[1],upper_bound[1],length=100)
y = range(lower_bound[2],upper_bound[2],length=100)
z = map(zip(x,y)) do I
    radial_basis(I)
end

surface(x, y, (x, y) -> radial_basis([x y]),markerstrokewidth=0,markersize=1,c=:viridis,camera=(60,20), markercolor=:viridis)

scatter(x, y, z,camera=(60,20),c=:viridis)

scatter(t_fo)
scatter(x_fo)
scatter(y_fo)

default()
scatter3d(x_fo,y_fo,t_fo,c=:viridis,xlabel="x (fm)",ylabel="y (fm)",zlabel="t (fm/c)",markersize=2,markerstrokewidth=0,marker_z = t_fo)
savefig("freezeout_surface.pdf")
x,phi_fo=fo_bg  

lb=Fluidum.leftbounds(x)
rb=Fluidum.rightbounds(x)
coord1 = range(lb[1],rb[1],length=100)
coord2 = range(lb[2],rb[2],length=100)
fo_surface = map(Iterators.product(eachindex(coord1),eachindex(coord2))) do (a,b)
    x(coord1[a],coord2[b])[1]
end

plot(coord1, coord2, fo_surface,st=:surface,camera=(20,20))
#run observables
vn = dvn_dp_list_delta(fo_bg, species_list, 0., [2]; eta_min = -5.0, eta_max = 5.0)
d = extract_vn([ObservableResult(mult, b, vn.u)])
plot(d[1,3,:,1,4])
sum(d[1,3,:,1,4])
sum(d[1,3,:,1,3])
sum(d[1,3,:,1,1])
sum(d[1,3,:,1,2])


tspan = (tau0, 6.)
result_oneshoot = Fluidum.oneshoot(twod_visc_hydro_discrete, Fluidum.matrix2d_visc_HQ!, fluidproperty, phi, tspan,saveat=[3.,6.])
heatmap(xgrid, xgrid, result_oneshoot[1][1,:,:], seriescolor = cgrad([:black, :blue, :red], [0, 0.156, 1.]),
 xlabel = "x (fm)", ylabel = "y (fm)", title = L"T \ \mathrm{(GeV)}, \tau =3.0 fm/c")
savefig("temperature_profile_3fm.pdf")

heatmap(xgrid, xgrid, result_oneshoot[2][1,:,:]
, xlabel = "x (fm)", ylabel = "y (fm)", title = L"T \ \mathrm{(GeV)}, \tau =6.0 fm/c")
savefig("temperature_profile_6fm.pdf")

heatmap(xgrid, xgrid, result_oneshoot[1][1,:,:]
, xlabel = "x (fm)", ylabel = "y (fm)", title = L"T \ \mathrm{(GeV)}, \tau =9.0 fm/c", seriescolor = cgrad(:inferno, [0, 0.156, 0.6]), clim = (0.,0.6))
savefig("temperature_profile_9fm.pdf")



heatmap(result_oneshoot[1][8,:,:])
heatmap(result_oneshoot(4.)[8,:,:])
heatmap(result_oneshoot(6.)[8,:,:])
heatmap(result_oneshoot(3.)[1,:,:])
heatmap(result_oneshoot(4.)[1,:,:])
heatmap(result_oneshoot(6.)[1,:,:])
result_oneshoot(3.)[8,:,:]

dx = discretization.grid[2][1] - discretization.grid[1][1]
dy = discretization.grid[1][2] - discretization.grid[1][1]

res = result_oneshoot(3.)
begin
    n_evolution = map(CartesianIndices(temperature_func)) do x
      i, j = Tuple(x)
      temp = res[1,i,j]
      fug = res[8,i,j]
      thermodynamic(temp,fug,eos.hadron_list).pressure
      end
end
sum(n_evolution)*dx*dx*3.

heatmap(xgrid, xgrid, n_evolution.*3.,right_margin = 20Plots.mm)


res = result_oneshoot(6.)
begin
    n_evolution = map(CartesianIndices(temperature_func)) do x
      i, j = Tuple(x)
      temp = res[1,i,j]
      fug = res[8,i,j]
      thermodynamic(temp,fug,eos.hadron_list).pressure
      end
end
sum(n_evolution)*dx*dx*6.

heatmap(xgrid, xgrid, n_evolution.*6.,right_margin = 20Plots.mm)


begin
    n_evolution = map(CartesianIndices(temperature_func)) do x
      i, j = Tuple(x)
      temp = result_oneshoot[1][1,:,:]
      fug = result_oneshoot[1][8,:,:]
      thermodynamic(temp[i, j],fug[i,j],eos.hadron_list).pressure
      end
end

sum(n_evolution)*9*9*dx*dx
default(aspect_ratio = :equal,
    size = (300, 300),
    dpi = 200,linewidth = 2, axis=true,
    markersize = 8
    , left_margin = 1Plots.mm,
    right_margin = 7Plots.mm,
    top_margin = 1Plots.mm,
    bottom_margin =1Plots.mm, legendfont = "sans-serif", titlefontsize = 12, tickfontsize = 10, legendfontsize = 10)
xgrid = getindex.(discretization.grid[:,1],1)
magma = cgrad(:magma).colors
heatmap(xgrid, xgrid, temperature_func, xlabel = "x (fm)", ylabel = "y (fm)", title = L"T \ \mathrm{(GeV)}, \tau =\tau_0 ", seriescolor = cgrad(:inferno, [0, 0.156, 0.6]), clim = (0.,0.6))

heatmap(xgrid, xgrid, n_evolution,right_margin = 20Plots.mm)