using WaterLily,StaticArrays,GLMakie

function sphere(L=2^5;Re=5e2,mem=Array,U=1,T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    R = T(L/4); ν = U*R/Re; center = SA[L/2,L/2-0.01,5*L];

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
    force = WaterLily.pressure_force(sim)
    force./(0.5sim.L*sim.U^2) # scale the forces!
end

# using CUDA
# make sim and run
sim = sphere(2^6;Re=1e4)#; mem=CuArray)
t₀ = sim_time(sim)
duration = 100.0
step = 0.05

# Run for visualization
viz!(sim;f=ω!,duration,step,video="sphere.mp4",algorithm=:mip,colormap=:algae)

## Run for force measurement
times = collect(t₀:step:t₀+duration)
forces = [get_forces!(sim,t) for t in times]

# Plots
import Plots
Plots.plot(times, getindex.(forces,3);
    xlabel="tU/L",
    ylabel="Pressure force coefficients")


