using WaterLily
using Plots

cID = "2DCircle"

function circle(n,m;Re=50,U=30,mem=Array)
    radius, center = m/8, m/2-1 # The minus 1 breaks the symmetry to get vortex street
    body = AutoBody((x,t)->√sum(abs2, x .- center) - radius)
    Simulation((n,m), (U,0), radius; ν=U*radius/Re, body, mem)
end

# Initialize the simulation with GPU Array
# using CUDA
sim = circle(3*2^5,2^6; mem=Array);

WaterLily.logger(cID) # Log the residual of pressure solver
#= NOTE: 
If you want to log residuals during a GPU simulation, it's better to include the following line. 
Otherwise, Julia will generate excessive debugging messages, which can significantly slow down the simulation. 
=#
using Logging; disable_logging(Logging.Debug)

# Run the simulation
sim_gif!(sim,duration=500,clims=(-5,5),plotbody=true)

# Remember to call Plots package (already done in Line 2). This will let WaterLily
# knows you want to plot sth like residual and will compile the funciton for you.
# NOTE: Comment out this line if you want to see gif animation!
# plot_logger("$(cID).log")