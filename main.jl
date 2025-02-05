using Distributed
addprocs(100)
@everywhere begin
    
include(joinpath(dirname(@__DIR__), "tools.jl"))
using ITensors, ITensorMPS
using JLD2
using ProgressMeter
using Optim
using Dates

mutable struct Observer <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64
    min_sweep::Int64
    info::AbstractVector

    Observer(energy_tol=1e-6; min_sweep=5, max_sweep=2000) = new(energy_tol, 1000., min_sweep, [max_sweep])
end

function ITensorMPS.checkdone!(o::Observer;kwargs...)
    sweep  = kwargs[:sweep]
    energy = kwargs[:energy]
    if abs(energy-o.last_energy)/abs(energy) < o.energy_tol && sweep > o.min_sweep
        o.info ∋ (sweep=sweep)
        return true
    end

    o.last_energy = energy
    return false
end

function Rydberg(params)
    (; δ, rb, nsites, cutoff, ener_tol, min_sweep, psi, d_cutoff) = params
    outputlv = 0
    v        = rb^6
    sites    = siteinds("S=1/2", nsites)

    os = OpSum()
    for i = 1:nsites
        os += 1/2, "X", i
        os += -δ, "ProjUp", i
    end

    for d = 1:d_cutoff
        for i = 1:nsites-d
            os += v / d^6, "ProjUp", i, "ProjUp", i+d
        end
    end

    ham   = MPO(os, sites)
    psi0  = randomMPS(sites, linkdims=6)

    nsweeps = 10000
    maxdim  = [10,20,100,100,800, 1000]
    cutoff  = [cutoff]
    noise   = [1E-7, 1E-8, 1E-9, 1E-10, 1E-11, 0]
    obs     = Observer(ener_tol)

    if psi !== nothing
        sites  = siteinds(psi)
        ham    = MPO(os, sites)
        psi0   = deepcopy(psi)
        obs    = Observer(ener_tol, min_sweep=min_sweep)
        maxdim = append!(maxdim[end] * ones(Int64, min_sweep-6), [1])
    end

    ener, psi = dmrg(ham, psi0; nsweeps, maxdim, cutoff, noise, observer=obs, outputlevel=outputlv)

    return psi
end

function Berry(params)
    (; i, j, dir) = params
    @show i j
    path = joinpath(dir, "gstate_$(i)_$(j).jld2")
    psi = load(path, "gstate").psi
    mag = expect(psi, "Sz") |> sum
    return (mag, i, j)
end

function Entanglement(psi, b)
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi, b)) )
    SVN = 0.0
    for n=1: dim(S, 1)
        p = S[n, n]^2
        if p > 0.0
        SVN = (SVN - p * log(p) )
        end
    end
    return SVN
end

function prodState(psi, θ_i, sign=1)
    sites = siteinds(psi)
    psi0 = MPS(sites, ["Up" for n=1:len(psi)])
    psi0[1][:, 1] = [cos(θ_i[1]), sign * sin(θ_i[1])]
    psi0[end][1, :] = [cos(θ_i[end]), sign * sin(θ_i[end])]
    for i = 2:len(psi0)-1
        psi0[i][1, :, 1] = [cos(θ_i[i]), sign * sin(θ_i[i])]
    end

    return psi0
end

function clProd(psi, initial, sign=1)
    f(θ)    = -abs(inner(psi, prodState(psi, θ, sign)))
    res     = optimize(f, initial, LBFGS())
    return res
end

function geoEntMag0(params)
    (; δ, rb, nsites, cutoff, ener_tol, d_cutoff) = params
    @show rb δ

    params_1 = (δ=δ, rb=rb, nsites=nsites, cutoff=cutoff, ener_tol=ener_tol, min_sweep=nothing, psi=nothing, d_cutoff)
    psi = Rydberg(params_1)

    initial = 2 * ones(len(psi))
    clprod_res = clProd(psi, initial)
    ini_which = 1

    if Optim.iterations(clprod_res) == 0
        θs      = range(0, π, 101)
        ovlps   = [abs(inner(psi, prodState(psi, θ * ones(nsites)))) for θ in θs]
        initial = θs[findmax(ovlps)[2]] * ones(nsites)
        clprod_res = clProd(psi, initial)
        ini_which = 2
    end
    

    szs = expect(psi, "Sz")
    
    if Optim.iterations(clprod_res) == 0
        initial = 1/2 * acos.(2*szs)
        clprod_res = clProd(psi, initial)
        ini_which = 3
    end
    

    if Optim.iterations(clprod_res) == 0
        initial = 1/2 * acos.(2*szs)
        clprod_res = clProd(psi, initial, -1)
        ini_which = 4
    end

    geo_ent = -log(2, minimum(clprod_res)^2)

    # return geo_ent, sum(szs), ent, Optim.minimizer(clprod_res), ini_which
    return geo_ent, sum(szs), clprod_res, ini_which
end

function geoEntMag(params)
    (; δ, rb, nsites, cutoff, ener_tol, d_cutoff) = params
    @show rb δ

    params_1 = (δ=δ, rb=rb, nsites=nsites, cutoff=cutoff, ener_tol=ener_tol, min_sweep=nothing, psi=nothing, d_cutoff)
    psi = Rydberg(params_1)

    clprod_s = []

    θs      = range(0, π, 101)
    ovlps   = [abs(inner(psi, prodState(psi, θ * ones(nsites)))) for θ in θs]
    initial = θs[findmax(ovlps)[2]] * ones(nsites)
    clprod_s  ∋ clProd(psi, initial)
    
    szs = expect(psi, "Sz")
    
    initial = 1/2 * acos.(2*szs)
    clprod_s ∋ clProd(psi, initial)    

    clprod_s ∋ clProd(psi, initial, -1)

    ovlp, ini_which = findmin(minimum.(clprod_s))

    geo_ent = -log(2, ovlp^2)

    # return geo_ent, sum(szs), clprod_s[ini_which], ini_which
    return geo_ent, sum(szs), ini_which
end

function joinvecs(vecs...)
    joined = []
    for vec in vecs[1:end-1]
        joined ∋ vec[1:end-1]...
    end
    joined ∋ vecs[end]...
    return joined
end



end



# nsites_s  = [61:6:121..., 139:12:199...]
nsites_s  = 61:6:301
# nsites_s  = 199:6:253
# nsites_s  = 259:6:301
# nsites_s  = [61, 91]
cutoff    = 1e-11
ener_tol  = 1e-11
d_cutoff  = 5

params = []

# rb_s    = 0.8:0.01:3.2
# δ_vert  = [2.5, 3.5]
# for δ in δ_vert
#     params ∋ [(δ=δ, rb = rb_s[i], nsites=nsites_s[j], cutoff=cutoff, ener_tol=ener_tol, d_cutoff=d_cutoff) for i = 1:len(rb_s),  j = 1:len(nsites_s)]
# end

rb_hori = 1.6

rb_hori == 1.4 && (δ_s = 0.7:0.005:1.2)
rb_hori == 1.6 && (δ_s = 0.9:0.005:1.4)
rb_hori == 2.3 && (δ_s = 1.5:0.005:2.)

@show nsites_s rb_hori


for rb in rb_hori
    params ∋ [(δ=δ_s[i], rb = rb, nsites=nsites_s[j], cutoff=cutoff, ener_tol=ener_tol, d_cutoff=d_cutoff) for i = 1:len(δ_s), j = 1:len(nsites_s)]
end

geo_ent_s = []
mag_s     = []
clprod_s  = []
ini_which_s = []

@time for i = 1:len(params)
    data = pmap(geoEntMag, params[i])
    geo_ent_s ∋ [el[1] for el in data]
    mag_s ∋ [el[2] for el in data]
    ini_which_s ∋ [el[3] for el in data]
end

jldsave(
    joinpath(@__DIR__, "data/$(today())_geo_ent_mag_scaling_rb_$(rb_hori)_d_$(d_cutoff)_n_$(nsites_s[1])_$(nsites_s[end]).jld2"), 
    nsites_s    = nsites_s,
    δ_s         = δ_s,
    geo_ent_s   = geo_ent_s,
    mag_s       = mag_s,
    ini_which_s = ini_which_s,
    d_cutoff    = d_cutoff
    )

# jldsave(joinpath(@__DIR__, "data/geo_ent_mag_scaling_δ_$(δ_vert)_d_$(d_cutoff)_$(today()).jld2"), nsites_s=nsites_s, rb_s=rb_s, δ_s=δ_s, geo_ent_s=geo_ent_s, mag_s=mag_s, clprod_s=clprod_s, ini_which_s=ini_which_s, d_cutoff=d_cutoff)


