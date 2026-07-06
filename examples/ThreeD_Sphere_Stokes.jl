using WaterLily,StaticArrays, JLD2
using CUDA
import Plots

include("../CVMetrics.jl")

step = 0.005
import WaterLily: CFL, conv_diff!, lowerBoundary!, upperBoundary!, BC!

## Overwriting WaterLily's internal
# Custom Time Stepper
function CFL(a::WaterLily.AbstractFlow)
    step
end

# Unsteady Stokes to replace Navier-Stokes (remove convective term)
function conv_diff!(r,u,Φ,λ::F;ν=0.1,perdir=()) where {F}
    r .= zero(eltype(r))
    N,n = WaterLily.size_u(u)
    for i ∈ 1:n, j ∈ 1:n
        # if it is periodic direction
        tagper = (j in perdir)
        # treatment for bottom boundary with BCs
        lowerBoundary!(r,u,Φ,ν,i,j,N,λ,WaterLily.Val{tagper}())
        # inner cells
        WaterLily.@loop (Φ[I] = - ν*WaterLily.∂(j,WaterLily.CI(I,i),u);
               r[I,i] += Φ[I]) over I ∈ WaterLily.inside_u(N,j)
        WaterLily.@loop r[I-δ(j,I),i] -= Φ[I] over I ∈ WaterLily.inside_u(N,j)
        # treatment for upper boundary with BCs
        upperBoundary!(r,u,Φ,ν,i,j,N,λ,WaterLily.Val{tagper}())
    end
end
# Neumann BC Building block
lowerBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{false}) = WaterLily.@loop r[I,i] += - ν*WaterLily.∂(j,WaterLily.CI(I,i),u) over I ∈ WaterLily.slice(N,2,j,2)
upperBoundary!(r,u,Φ,ν,i,j,N,λ,::Val{false}) = WaterLily.@loop r[I-δ(j,I),i] += ν*WaterLily.∂(j,WaterLily.CI(I,i),u) over I ∈ WaterLily.slice(N,N[j],j,2)
# function BC!(a,uBC::Function,saveexit=false,perdir=(),t=0)
#     N,n = WaterLily.size_u(a)
#     for i ∈ 1:n, j ∈ 1:n
#         if j in perdir
#             WaterLily.@loop a[I,i] = a[WaterLily.CIj(j,I,N[j]-1),i] over I ∈ WaterLily.slice(N,1,j)
#             WaterLily.@loop a[I,i] = a[WaterLily.CIj(j,I,2),i] over I ∈ WaterLily.slice(N,N[j],j)
#         else
#             # for s ∈ (1,2)
#                 WaterLily.@loop a[I,i] = uBC(i,WaterLily.loc(i,I),t) over I ∈ WaterLily.slice(N,1,j)
#             # end
#             # for s ∈ (N[j]-1,N[j])
#                 WaterLily.@loop a[I,i] = uBC(i,WaterLily.loc(i,I),t) over I ∈ WaterLily.slice(N,N[j],j)
#             # end
#         end
#     end
# end



## Normal WaterLily Calls
function sphere_body(center, R)
    AutoBody((x,t)->(
        abs2((x[1]-center[1])/R) 
        + abs2((x[2]-center[2])/R) 
        + abs2((x[3]-center[3])/R) 
        -1)) 
end
function sphere(body, L, R;Re=5e2,mem=Array,U=1, center = SA{Float32}[L/2,L/2,L/2], T=Float32)
    # Define simulation size, geometry dimensions, & viscosity
    R = T(R);
    ν = U*R/Re
    ## Analytical solution for Stokes flow past a sphere
    function stokes_sphere(i,xyz,t)
        # center = SA[L/2,L/2,L/2]
        x,y,z = @. xyz 
        rx = (x - center[1])/R; ry = (y - center[2])/R; rz = (z - center[3])/R
        rnorm = (rx^2 + ry^2 + rz^2) |> sqrt
        if rnorm < 1f0
             return 0
        end
        i==1 && return (rx*rz/rnorm^3)*3/4 - (3*rx*rz/rnorm^5)/4
        i==2 && return (ry*rz/rnorm^3)*3/4 - (3*ry*rz/rnorm^5)/4
        return (rnorm^(-1) + rz^2/rnorm^3)*3/4  + (rnorm^(-3) - 3*(rz^2)/rnorm^5)/4 - 1
    end

    uλ(i,xyz) = stokes_sphere(i,xyz,0)
    # Return initialized simulation
    Simulation((L,L,L),stokes_sphere,R;U=U,uλ=uλ,ν,body,mem,T, Δt=step,ϵ=0.1)
end

## Additional helpers for viscous moment calculation
function get_forces(sim,measure_body)
    p = sim.flow.p |> Array
    u = sim.flow.u |> Array
    f = sim.flow.f |> Array
    force = pressure_force(p,f,measure_body) + 
    viscous_force(u,sim.flow.ν,f,measure_body) + 
    flux_force(u,f,measure_body)
    force./(sim.L^2*sim.U^2) # scale the forces!
end
function get_moment(sim,measure_body;x₀=center)
    p = sim.flow.p |> Array
    u = sim.flow.u |> Array
    f = sim.flow.f |> Array
    moment = pressure_moment(x₀,p,f,measure_body) + 
    viscous_moment(x₀,u,sim.flow.ν,f,measure_body) + 
    flux_moment(x₀,u,f,measure_body)
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

##
# make sim and run
L=2^7
R = L/32 |> Float32
center = SA{Float32}[L/2,L/2,L/2]
body = sphere_body(center, R)
measure_body = sphere_body(center, R + 2.83f0)
sim = sphere(body, L, R;Re=1e0, mem=CuArray)
t₀ = sim_time(sim)
duration = 5.0
# step = 0.01

## Run simulation and export forces and moments
forces = []
moments = []
Plots.@gif for ti in range(t₀,t₀+duration;step) 
    tmp_v, tmp_p = steps_w_export!(sim,ti)
    push!(forces, tmp_v)
    push!(moments, tmp_p)
end
