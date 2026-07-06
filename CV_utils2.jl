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

## CV
function CV_coords(n_pts,Llow,Lhigh,Tp)
    positions = Vector{SVector{3,Tp}}(undef,n_pts^2*2*3)
    normals = Vector{SVector{3,Tp}}(undef,n_pts^2*2*3)
    weights = Vector{Tp}(undef,n_pts^2*2*3)

    ranges  = range(Llow,Lhigh;length=n_pts)
    xrange = vcat(reshape(ones(n_pts,1)*ranges',n_pts^2,1),
    reshape(ones(n_pts,1)*ranges',n_pts^2,1))
    yrange = vcat(reshape(ranges*ones(1,n_pts),n_pts^2,1),
    reshape(ranges*ones(1,n_pts),n_pts^2,1))
    zrange = vcat(ones(n_pts^2,1)*Llow,
                    ones(n_pts^2,1)*Lhigh)
    weight_rng = ones(Tp,n_pts,n_pts)*((Lhigh-Llow)/(n_pts-1))^2
    weight_rng[1,:] .*= Tp(0.5)
    weight_rng[:,1] .*= Tp(0.5)
    weight_rng[end,:] .*= Tp(0.5)
    weight_rng[:,end] .*= Tp(0.5)
    weight_rng = vcat(reshape(weight_rng,n_pts^2,1),reshape(weight_rng,n_pts^2,1))
    for ix in 1:n_pts^2*2
        positions[ix] = SVector{3,Tp}(Tp(zrange[ix]),Tp(xrange[ix]),Tp(yrange[ix]))
        weights[ix] = weight_rng[ix]
    end
    for ix in 1:n_pts^2
        normals[ix] = SVector{3,Tp}(Tp(-1.0),Tp(0.0),Tp(0.0))
        normals[ix+n_pts^2] = SVector{3,Tp}(Tp(1.0),Tp(0.0),Tp(0.0))
    end
    for ix in 1:n_pts^2*2
        positions[n_pts^2*2+ix] = SVector{3,Tp}(Tp(yrange[ix]),Tp(zrange[ix]),Tp(xrange[ix]))
        weights[n_pts^2*2+ix] = weight_rng[ix]
    end
    for ix in 1:n_pts^2
        normals[n_pts^2*2+ix] = SVector{3,Tp}(Tp(0.0),Tp(-1.0),Tp(0.0))
        normals[ix+n_pts^2*3] = SVector{3,Tp}(Tp(0.0),Tp(1.0),Tp(0.0))
    end
    for ix in 1:n_pts^2*2
        positions[n_pts^2*4+ix] = SVector{3,Tp}(Tp(xrange[ix]),Tp(yrange[ix]),Tp(zrange[ix]))
        weights[n_pts^2*4+ix] = weight_rng[ix]
    end
    for ix in 1:n_pts^2
        normals[n_pts^2*4+ix] = SVector{3,Tp}(Tp(0.0),Tp(0.0),Tp(-1.0))
        normals[ix+n_pts^2*5] = SVector{3,Tp}(Tp(0.0),Tp(0.0),Tp(1.0))
    end   
    return positions, normals, weights
end

_cv_interp_batch(positions::Vector{SVector{3, Float32}}, field) = [WaterLily.interp(pos, field) for pos in positions]

function CV_force_p(sim,positions, normals, weights)
    pres = _cv_interp_batch(positions, sim.flow.p) 
    sum(weights[i]*pres[i]*normals[i] for i in 1:length(positions))
end
function CV_force_flux(sim,positions, normals, weights)
    u = _cv_interp_batch(positions, sim.flow.u) 
    sum(weights[i]*u[i]*dot(u[i],normals[i]) for i in 1:length(positions))
end
function CV_force_vis(sim,positions, normals, weights)
    n_pts = length(positions)÷6
    N,n = WaterLily.size_u(sim.flow.u)
    Tu = eltype(sim.flow.u); To = promote_type(Float64,Tu)
    sim.flow.f .= zero(Tu)

    vis_force = zeros(Tu,3)
    unit_vec = unit_vector(1,Tu)
    WaterLily.@loop sim.flow.f[I,:] .= 2*sim.flow.ν*WaterLily.S(I,sim.flow.u)*unit_vec over I ∈ WaterLily.inside_u(sim.flow.u)
    τn = -_cv_interp_batch(positions[1:n_pts], sim.flow.f) 
    vis_force .+= sum(weights[i] * τn[i] for i in 1:n_pts)
    τn = _cv_interp_batch(positions[n_pts+1:2*n_pts], sim.flow.f) 
    vis_force .+= sum(weights[i] * τn[i-n_pts] for i in n_pts+1:2*n_pts)
    unit_vec = unit_vector(2,Tu)
    WaterLily.@loop sim.flow.f[I,:] .= 2*sim.flow.ν*WaterLily.S(I,sim.flow.u)*unit_vec over I ∈ WaterLily.inside_u(sim.flow.u)
    τn = -_cv_interp_batch(positions[2*n_pts+1:3*n_pts], sim.flow.f) 
    vis_force .+= sum(weights[i] * τn[i-2*n_pts] for i in 2*n_pts+1:3*n_pts)
    τn = _cv_interp_batch(positions[3*n_pts+1:4*n_pts], sim.flow.f) 
    vis_force .+= sum(weights[i] * τn[i-3*n_pts] for i in 3*n_pts+1:4*n_pts)
    unit_vec = unit_vector(3,Tu)
    WaterLily.@loop sim.flow.f[I,:] .= 2*sim.flow.ν*WaterLily.S(I,sim.flow.u)*unit_vec over I ∈ WaterLily.inside_u(sim.flow.u)
    τn = -_cv_interp_batch(positions[4*n_pts+1:5*n_pts], sim.flow.f) 
    vis_force .+= sum(weights[i] * τn[i-4*n_pts] for i in 4*n_pts+1:5*n_pts)
    τn = _cv_interp_batch(positions[5*n_pts+1:6*n_pts], sim.flow.f) 
    vis_force .+= sum(weights[i] * τn[i-5*n_pts] for i in 5*n_pts+1:6*n_pts)
        
    return vis_force
end