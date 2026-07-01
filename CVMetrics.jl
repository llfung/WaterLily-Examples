using WaterLily,StaticArrays, JLD2
# import LinearAlgebra: cross, norm, dot

import WaterLily: viscous_force, pressure_force, pressure_moment, cross, dot

"""
    viscous_moment(x₀,sim::Simulation)

Computes the viscous moment on an immersed body relative to point x₀.
"""
viscous_moment(x₀,sim) = viscous_moment(x₀,sim.flow,sim.body)
viscous_moment(x₀,flow,body) = viscous_moment(x₀,flow.u,flow.ν,flow.f,body,WaterLily.time(flow))
function viscous_moment(x₀,u,ν,df,body,t=0)
    Tu = eltype(u); To = promote_type(Float64,Tu)
    df .= zero(Tu)
    WaterLily.@loop df[I,:] .= -2ν*WaterLily.cross(WaterLily.loc(0,I,Tu)-x₀,WaterLily.S(I,u)*WaterLily.nds(body,WaterLily.loc(0,I,Tu),t)) over I ∈ WaterLily.inside_u(u)
    sum(To,df,dims=ntuple(i->i,ndims(u)-1))[:] |> Array
end

# Flux through CV surface
flux_force(sim) = flux_force(sim.flow,sim.body)
flux_force(flow,body) = flux_force(flow.u,flow.f,body,WaterLily.time(flow))
function flux_force(u,df,body,t=0)
    Tu = eltype(u); To = promote_type(Float64,Tu)
    df .= zero(Tu)
    WaterLily.@loop df[I,:] .= WaterLily.dot(u[I,:],WaterLily.nds(body,WaterLily.loc(0,I,Tu),t))*u[I,:] over I ∈ WaterLily.inside_u(u)
    sum(To,df,dims=ntuple(i->i,ndims(u)-1))[:] |> Array
end

flux_moment(x₀,sim) = flux_moment(x₀,sim.flow,sim.body)
flux_moment(x₀,flow,body) = flux_moment(x₀,flow.u,flow.f,body,WaterLily.time(flow))
function flux_moment(x₀,u,df,body,t=0)
    Tu = eltype(u); To = promote_type(Float64,Tu)
    df .= zero(Tu)
    WaterLily.@loop df[I,:] .= WaterLily.cross(WaterLily.loc(0,I,Tu)-x₀,WaterLily.dot(u[I,:],WaterLily.nds(body,WaterLily.loc(0,I,Tu),t))*u[I,:]) over I ∈ WaterLily.inside_u(u)
    sum(To,df,dims=ntuple(i->i,ndims(u)-1))[:] |> Array
end

