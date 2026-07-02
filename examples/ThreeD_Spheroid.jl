using WaterLily,StaticArrays, JLD2
using CUDA
import Plots

include("../CVMetrics.jl")

step = 0.005
import WaterLily: CFL

function CFL(a::WaterLily.AbstractFlow)
    step
end

function spheroid_body(center, a, b, c, ϕ)
    AutoBody((x,t)->(
        abs2((x[1]-center[1])/a) 
        + abs2(((x[2]-center[2])/sqrt(2)+(x[3]-center[3])/sqrt(2))/b) 
        + abs2(((x[3]-center[3])/sqrt(2)-(x[2]-center[2])/sqrt(2))/c) 
        -1))
end
function sim_init(body,L,R;Re=3e0,mem=Array,U=1,T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    R = T(R);
    ν = U*R/Re;

    # a=R; b=R; c=3*R; center = SA{T}[L/2,L/2,L/2];

    # body = AutoBody((x,t)->(
    #     abs2((x[1]-center[1])/a) 
    #     + abs2(((x[2]-center[2])/sqrt(2)+(x[3]-center[3])/sqrt(2))/b) 
    #     + abs2(((x[3]-center[3])/sqrt(2)-(x[2]-center[2])/sqrt(2))/c) 
    #     -1))
    # Return initialized simulation
    Simulation((L,L,L),(0,0,-U),R;ν,body,mem,T, Δt=step)
end
## Additional helpers for viscous moment calculation
function get_forces(sim,measure_body)
    force = pressure_force(sim.flow.p,sim.flow.f,measure_body) + 
    viscous_force(sim.flow.u,sim.flow.ν,sim.flow.f,measure_body) + 
    flux_force(sim.flow.u,sim.flow.f,measure_body)
    force./(sim.L^2*sim.U^2) # scale the forces!
end
function get_moment(sim,measure_body;x₀=center)
    moment = pressure_moment(x₀,sim.flow.p,sim.flow.f,measure_body) + 
    viscous_moment(x₀,sim.flow.u,sim.flow.ν,sim.flow.f,measure_body) + 
    flux_moment(x₀,sim.flow.u,sim.flow.f,measure_body)
    moment./(sim.L^3*sim.U^2) # scale the moments!
end
function plot_field(sim,t;contour_plane=Int(L/2+1))
    R = inside(sim.flow.p)
    @WaterLily.inside sim.flow.σ[I] = sim.flow.p[I]
    flood(sim.flow.σ[R[contour_plane,:,:]] |> Array; clims=(-2,2), 
    title = "ωx at tU/L=$(round(t,digits=2))",
    xlim=(0,L),ylim=(0,L))
end
function steps_w_export!(sim,t;measure_body=measure_body,contour_plane=Int(L/2+1))
    sim_step!(sim,t,remeasure=true)
    plot_field(sim,t;contour_plane=contour_plane)
    println("tU/L=",round(t,digits=8),
            ", Δt=",round(sim.flow.Δt[end],digits=8))
    return  get_forces(sim,measure_body), get_moment(sim,measure_body;x₀=center)
end

## Setup simulation 
L=2^8
R=L/64 |> Float32
center = SA{Float32}[L/2,L/2,L/2]
body = spheroid_body(center, R, R, 3*R, Float32(pi/4))
measure_body = spheroid_body(center, R+2, R+2, 3*(R+2), Float32(pi/4))
sim = sim_init(body, L, 3*R; Re=3f0, mem=Array) # Re=3.0
t₀ = sim_time(sim)
duration = 10.0
# step = 0.01

## Run simulation and export forces and moments
forces = []
moments = []
Plots.@gif for ti in range(t₀,t₀+duration;step) 
    tmp_v, tmp_p = steps_w_export!(sim,ti)
    push!(forces, tmp_v)
    push!(moments, tmp_p)
end

