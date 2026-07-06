using WaterLily,StaticArrays, JLD2
## CV
function slice(dims::NTuple{N},i,j,low,up) where N
    CartesianIndices(ntuple( k-> k==j ? (i:i) : (low:up), N))
end
function unit_vector(i,T)
    i==1 && return SA{T}[1.0,0.0,0.0]
    i==2 && return SA{T}[0.0,1.0,0.0]
    return SA{T}[0.0,0.0,1.0]
end

function CV_force_vis(sim,low_lim,up_lim)
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)
    for i ∈ 1:n
        unit_vec = unit_vector(i,Tu)
        WaterLily.@loop sim.flow.f[I,:] .= 2*sim.flow.ν*WaterLily.S(I,sim.flow.u)*unit_vec over I ∈ slice(N,low_lim,i,low_lim,up_lim)
        WaterLily.@loop sim.flow.f[I,:] .= -2*sim.flow.ν*WaterLily.S(I,sim.flow.u)*unit_vec over I ∈ slice(N,up_lim,i,low_lim,up_lim)
    end
    sum(To,sim.flow.f,dims=ntuple(i->i,n))[:] |> Array
end
function CV_force_p(sim,low_lim,up_lim)
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)
    for i ∈ 1:n
        unit_vec = unit_vector(i,Tu)
        WaterLily.@loop sim.flow.f[I,:] .= -sim.flow.p[I]*unit_vec over I ∈ slice(N,low_lim,i,low_lim,up_lim)
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*unit_vec over I ∈ slice(N,up_lim,i,low_lim,up_lim)
    end
    sum(To,sim.flow.f,dims=ntuple(i->i,n))[:] |> Array
end
@inline flux1(i,I,u) = @inbounds u[I,i]
@inline flux2(I,u) = @inbounds u[I,:]
function CV_force_flux(sim,low_lim,up_lim)
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)
    for i ∈ 1:n
        WaterLily.@loop sim.flow.f[I,:] .= -sim.flow.u[I,i]*sim.flow.u[I,:] over I ∈ slice(N,low_lim,i,low_lim,up_lim)
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.u[I,i]*sim.flow.u[I,:] over I ∈ slice(N,up_lim,i,low_lim,up_lim)
    end
    sum(To,sim.flow.f,dims=ntuple(i->i,n))[:] |> Array
end
function CV_moment_vis(x₀,sim,low_lim,up_lim)
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)
    for i ∈ 1:n
        unit_vec = unit_vector(i,Tu)
        WaterLily.@loop sim.flow.f[I,:] .= 2*sim.flow.ν*cross(loc(0,I,Tu)-x₀,WaterLily.S(I,sim.flow.u)*unit_vec) over I ∈ slice(N,low_lim,i,low_lim,up_lim)
        WaterLily.@loop sim.flow.f[I,:] .= -2*sim.flow.ν*cross(loc(0,I,Tu)-x₀,WaterLily.S(I,sim.flow.u)*unit_vec) over I ∈ slice(N,up_lim,i,low_lim,up_lim)
    end
    sum(To,sim.flow.f,dims=ntuple(i->i,n))[:] |> Array
end
function CV_moment_p(x₀,sim,low_lim,up_lim)
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)
    for i ∈ 1:n
        unit_vec = unit_vector(i,Tu)
        WaterLily.@loop sim.flow.f[I,:] .= -sim.flow.p[I]*cross(loc(0,I,Tu)-x₀,unit_vec) over I ∈ slice(N,low_lim,i,low_lim,up_lim)
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.p[I]*cross(loc(0,I,Tu)-x₀,unit_vec) over I ∈ slice(N,up_lim,i,low_lim,up_lim)
    end
    sum(To,sim.flow.f,dims=ntuple(i->i,n))[:] |> Array
end
function CV_moment_flux(x₀,sim,low_lim,up_lim)
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)
    for i ∈ 1:n
        WaterLily.@loop sim.flow.f[I,:] .= -sim.flow.u[I,i]*cross(loc(0,I,Tu)-x₀,sim.flow.u[I,:]) over I ∈ slice(N,low_lim,i,low_lim,up_lim)
        WaterLily.@loop sim.flow.f[I,:] .= sim.flow.u[I,i]*cross(loc(0,I,Tu)-x₀,sim.flow.u[I,:]) over I ∈ slice(N,up_lim,i,low_lim,up_lim)
    end
    sum(To,sim.flow.f,dims=ntuple(i->i,n))[:] |> Array
end