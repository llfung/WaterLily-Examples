using WaterLily,StaticArrays,GLMakie
import LinearAlgebra: cross, norm

function sphere(L=2^5;Re=5e2,mem=Array,U=1,T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    R = T(L/4); ν = U*R/Re; center = SA[L/2,L/2,5*L];

    # Build jelly from a mapped sphere and plane
    body = AutoBody((x,t)->abs(√sum(abs2, x .- center) - R)) 

    # Return initialized simulation
    Simulation((L,L,6L),(0,0,-U),R;ν,body,mem,T)
end

function ω!(arr, sim)
    a = sim.flow.σ
    WaterLily.@inside a[I] = WaterLily.ω_mag(I,sim.flow.u)
    copyto!(arr, a[inside(a)]) # copy to CPU
end

function get_forces!(sim,t)
    sim_step!(sim,t,remeasure=false)
    force = WaterLily.total_force(sim)
    force./(0.5sim.L*sim.U^2) # scale the forces!
end
function get_moment!(sim,t,x₀=SA[sim.L/2,sim.L/2,5*sim.L])
    sim_step!(sim,t,remeasure=false)
    force = viscous_moment(x₀,sim)
    force./(0.5sim.L*sim.U^2) # scale the forces!
end

"""
    viscous_moment(x₀,sim::Simulation)

Computes the viscous moment on an immersed body relative to point x₀.
"""
viscous_moment(x₀,sim) = viscous_moment(x₀,sim.flow,sim.body)
viscous_moment(x₀,flow,body) = viscous_moment(x₀,flow.u,flow.ν,flow.f,body,WaterLily.time(flow))
function viscous_moment(x₀,u,ν,df,body,t=0)
    Tu = eltype(u); To = promote_type(Float64,Tu)
    df .= zero(Tu)
    WaterLily.@loop df[I,:] .= -2ν*cross(loc(0,I,Tu)-x₀,WaterLily.S(I,u)*WaterLily.nds(body,loc(0,I,Tu),t)) over I ∈ WaterLily.inside_u(u)
    sum(To,df,dims=ntuple(i->i,ndims(u)-1))[:] |> Array
end

##
using CUDA
# make sim and run
L=2^6 
sim = sphere(L;Re=1e3, mem=CuArray)
t₀ = sim_time(sim)
duration = 100.0
step = 0.01

# Run for visualization
viz!(sim;f=ω!,duration,step,video="sphere.mp4",algorithm=:mip,colormap=:algae)

## Run for force measurement
times = collect(t₀:step:t₀+duration)
forces = [get_forces!(sim,t) for t in times]
moments = [get_moment!(sim,t) for t in times]

# Plots
import Plots
Plots.plot(times, getindex.(forces,3);
    xlabel="tU/L",
    ylabel="Pressure force coefficients")

Plots.plot(times, getindex.(moments,1);
    xlabel="tU/L",
    ylabel="Moment force coefficients")
