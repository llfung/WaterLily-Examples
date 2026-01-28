using WaterLily
using CUDA
using StaticArrays
using Plots

function sim_gif_custom!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
    WaterLily.sim_step!(sim,tᵢ;remeasure)
    @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    flood(sim.flow.σ[R] |> Array; kv...)
    plotbody && body_plot!(sim)
    verbose && println("tU/L=",round(tᵢ,digits=4),
            ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

function circle(n,m;Re=50,U=30,T=Float32,mem=Array)
    radius, center = m/8, m/2-1 # The minus 1 breaks the symmetry to get vortex street
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, T, mem)
end

# Initialize the simulation with GPU Array
sim = circle(3*2^5,2^6; T=Float32,mem=CuArray);

cID = "2DCircle"
WaterLily.logger(cID) # Log the residual of pressure solver
#= NOTE: 
If you want to log residuals during a GPU simulation, it's better to include the following line. 
Otherwise, Julia will generate excessive debugging messages, which can significantly slow down the simulation. 
=#
using Logging; disable_logging(Logging.Debug)

## Run the simulation (For CUDA, using custom plotting func becuase internal one is not CUDA ready)
sim_gif_custom!(sim,duration=30,clims=(-5,5),plotbody=true)

# Remember to call Plots package (already done in Line 2). This will let WaterLily
# knows you want to plot sth like residual and will compile the funciton for you.
# NOTE: Comment out this line if you want to see gif animation!
# plot_logger("$(cID).log")