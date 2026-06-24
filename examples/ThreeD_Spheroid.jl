using WaterLily,StaticArrays,GLMakie
import LinearAlgebra: cross, norm
import Plots

function sphere(L=2^5;Re=5e2,mem=Array,U=1,T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    R = T(L/8); ν = U*R/Re; center = SA[L/2,L/2,L/2];
    a=R
    b=R
    c=1.5*R
    # Build jelly from a mapped sphere and plane
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
    Simulation((L,L,L),(0,0,-U),R;ν,body,mem,T)
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
function get_moment!(sim,t,x₀=SA[L/2,L/2,L/2])
    sim_step!(sim,t,remeasure=false)
    force = viscous_moment(x₀,sim)
    force./(0.5sim.L^2*sim.U^2) # scale the forces!
end
function plot_ωx!(sim,t,x₀=Int(L/2+1))
    sim_step!(sim,t,remeasure=true)
    R = inside(sim.flow.p)
    @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(1,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[R[x₀,:,:]] |> Array; clims=(-2,2), 
    title = "ωx at tU/L=$(round(t,digits=2))",
    xlim=(0,L),ylim=(0,2*L))
    println("tU/L=",round(t,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    return viscous_moment(SA[L/2,L/2,L/2],sim)./(0.5sim.L^2*sim.U^2)
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
sim = sphere(L;Re=32, mem=CuArray)
t₀ = sim_time(sim)
duration = 50.0
step = 0.01

# Run for visualization
# viz!(sim;f=ω!,duration,step,video="spheroid.mp4",algorithm=:mip,colormap=:algae)
# Run for omega_x measurement
moments = []
Plots.@gif for ti in range(t₀,t₀+duration;step) 
    tmp = plot_ωx!(sim,ti)
    push!(moments, tmp)
end
save!("spheroid_test_start.jld2",sim)
save!("spheroid_test.jld2",sim)
##
Re_Array = 1.0./(0.04:0.01:5.0) |> collect
for Re in Re_Array
    global moments
    sim_ = sphere(L;Re, mem=CuArray)
    load!(sim_;fname="spheroid_test.jld2")
    t₀ = sim_time(sim_)
    duration = 50.0
    step = 0.1
    Plots.@gif for ti in range(t₀,t₀+duration;step) 
        tmp = plot_ωx!(sim_,ti)
        push!(moments, tmp)
    end
    save!("spheroid_test.jld2",sim_)
end
# sim2 = sphere(L;Re=1e1, mem=CuArray)
# load!(sim2; fname="spheroid_test.jld2")
## Run for force measurement
# times = collect(t₀:step:t₀+duration)
# forces = [get_forces!(sim,t) for t in times]
# moments = [get_moment!(sim,t) for t in times]

# Plots
import Plots
# Plots.plot(times, getindex.(forces,3);
#     xlabel="tU/L",
#     ylabel="Pressure force coefficients")

# Plots.plot(times, getindex.(moments,1);
#     xlabel="tU/L",
#     ylabel="Moment force coefficients")
