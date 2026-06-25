using WaterLily,StaticArrays, JLD2
import LinearAlgebra: cross, norm
import Plots

step = 0.005
import WaterLily: CFL

function CFL(a::WaterLily.AbstractFlow)
    step
end

function spheroid(L=2^5;Re=5e2,mem=Array,U=1,T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    R = T(L/16); ν = U*R/Re; center = SA[L/2,L/2,L];
    a=R
    b=R
    c=2*R
    # Build jelly from a mapped spheroid and plane
    # body = AutoBody((x,t)->(
    #     abs2((x[1]-center[1])/a) 
    #     + abs2(((x[2]-center[2])-(x[3]-center[3]))/b) 
    #     + abs2(((x[3]-center[3])+(x[2]-center[2]))/c) 
    #     -1)) 
    body = AutoBody((x,t)->(
        abs2((x[1]-center[1])/a) 
        + abs2(((x[2]-center[2])/sqrt(2)+(x[3]-center[3])/sqrt(2))/b) 
        + abs2(((x[3]-center[3])/sqrt(2)-(x[2]-center[2])/sqrt(2))/c) 
        -1)) 
    # Return initialized simulation
    Simulation((L,L,2*L),(0,0,-U),c;ν,body,mem,T, Δt=step)
end

function get_forces!(sim,t)
    sim_step!(sim,t,remeasure=false)
    force = WaterLily.total_force(sim)
    force./(sim.L^2*sim.U^2) # scale the forces!
end
function get_moment!(sim,t,x₀=SA[L/2,L/2,L])
    sim_step!(sim,t,remeasure=false)
    force = viscous_moment(x₀,sim)
    force./(sim.L^3*sim.U^2) # scale the forces!
end
function plot_ωx!(sim,t,x₀=Int(L/2+1))
    sim_step!(sim,t,remeasure=true)
    R = inside(sim.flow.p)
    @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(1,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[R[x₀,:,:]] |> Array; clims=(-2,2), 
    title = "ωx at tU/L=$(round(t,digits=2))",
    xlim=(0,L),ylim=(0,2*L))
    println("tU/L=",round(t,digits=8),
            ", Δt=",round(sim.flow.Δt[end],digits=8))
    return viscous_moment(SA[L/2,L/2,L],sim)./(sim.L^3*sim.U^2), WaterLily.pressure_moment(SA[L/2,L/2,L],sim)./(sim.L^3*sim.U^2)
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
L=2^7
sim = spheroid(L;Re=3e-1, mem=CuArray)
t₀ = sim_time(sim)
duration = 3.0
# step = 0.01

## Run for omega_x measurement
vis_moments = []
p_moments = []
Plots.@gif for ti in range(t₀,t₀+duration;step) 
    tmp_v, tmp_p = plot_ωx!(sim,ti)
    push!(vis_moments, tmp_v)
    push!(p_moments, tmp_p)
end

## Run for force measurement
# times = collect(t₀:step:t₀+duration)
# forces = [get_forces!(sim,t) for t in times]
# moments = [get_moment!(sim,t) for t in times]

# Plots
# Plots.plot(times, getindex.(forces,3);
#     xlabel="tU/L",
#     ylabel="Pressure force coefficients")

# Plots.plot(times, getindex.(moments,1);
#     xlabel="tU/L",
#     ylabel="Moment force coefficients")
