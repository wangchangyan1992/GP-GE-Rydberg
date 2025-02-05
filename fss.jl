include(joinpath(dirname(@__DIR__), "tools.jl"))
using JLD2
using PyPlot
using ProgressMeter
using LsqFit
using Optim
using BenchmarkTools
pygui(true)
# pygui(false)
fssa = pyimport("fssa")




#%%------------------------------------------------------------
function deriData(data, δs, nsites_s)
    data = [data[:, i] / nsites_s[i] for i = 1:len(nsites_s)]
    d_data =  [(el[2:end] - el[1:end-1]) ./ (δs[2:end] - δs[1:end-1]) for el in data]
    return d_data    
end

function deriData(data, δs, nsites_s, di=1, nsites_sel=nothing)
    if nsites_sel !== nothing
        inds_sel = [findfirst(isequal(el), nsites_s) for el in nsites_sel]
    else
        inds_sel = 1:len(nsites_s)
    end
    δs     = δs[1:di:end]
    data   = [data[1:di:end, i] / nsites_s[i] for i in inds_sel]
    d_data = [(el[2:end] - el[1:end-1]) ./ (δs[2:end] - δs[1:end-1]) for el in data]
    δs     = (δs[1:end-1] + δs[2:end]) / 2
    return δs, d_data
end

freeEner(x, y) = 1 - exp(x - y)

function ijPrimes(x, i, j)
    ijprimes = []
    for i_prime in 1:len(x)
        if i_prime != i
            j_prime = findfirst(j_prime->x[i_prime][j_prime] <= x[i][j] <= x[i_prime][j_prime+1], 1:len(x[i_prime])-1)
            j_prime != nothing && (ijprimes ∋ [i_prime, j_prime]; ijprimes ∋ [i_prime, j_prime+1])
        end
    end

    return ijprimes    
end

function linearFit(xij, xyls)
    xls, yls, dyls = xyls
    wls = @. 1 / dyls^2
    k = sum(wls)
    kx = wls' * xls
    ky = wls' * yls
    kxx = wls' * (xls .^ 2)
    kxy = wls .* xls .* yls |> sum
    Δ = k * kxx - kx^2

    ma_yij = (kxx * ky - kx * kxy) / Δ + xij * (k * kxy - kx * ky) / Δ
    d_ma_yij2 = (kxx - 2 * xij * kx + xij^2 * k) / Δ

    return ma_yij, d_ma_yij2
end

xyL(els, ijprimes) = [els[i][j] for (i, j) in ijprimes]

function Quality(x, y, dy)
    ma_y_num = 0
    quality = 0
    
    for i = 1:len(x)
        quality_i = 0
        # ma_y_n_i = 0
        for j in 1:len(x[i])
            ijprimes = ijPrimes(x, i, j)
            if ijprimes != []
                xyls = [xyL(x, ijprimes), xyL(y, ijprimes), xyL(dy, ijprimes)]
                ma_yij, d_ma_yij2 = linearFit(x[i][j], xyls)
                # ma_y_n_i += 1
                ma_y_num += 1
                quality_i += (y[i][j] - ma_yij)^2 / (dy[i][j]^2 + d_ma_yij2)
            end
        end
        # quality_i != 0 && (quality += quality_i / ma_y_n_i; ma_y_num += 1)
        quality_i != 0 && (quality += quality_i)
    end
    return quality / ma_y_num
end


function autoScal(nsites_s, δs, free_ener, dy, init)
    @show len(nsites_s) == len(δs)
    function goal(c)
        x = [nsites_s[i]^(1/c[2]) * (δs[i] .- c[1]) for i = 1:len(nsites_s)]
        return Quality(x, free_ener, dy)
    end

    res = optimize(goal, init)
    return res
end

function dataSel(δs, y, dy, ylim)
    δs_sel = []
    y_sel  = []
    dy_sel = []

    ind = findfirst(x->δs[x]>1.605, 1:len(δs))
    δs  = δs[ind:end]
    y   = [y[i][ind:end] for i = 1:len(y)]
    dy  = [dy[i][ind:end] for i = 1:len(dy)]

    for i = 1:len(y)
        inds = findall(j->y[i][j]<ylim, 1:len(y[i]))
        δs_sel ∋ δs[inds]
        y_sel ∋ y[i][inds]
        dy_sel ∋ dy[i][inds]
    end
    return δs_sel, y_sel, dy_sel
end


function pseudoCrit(data, δs0, nsites_s, di, sel_inds)
    δs, d_data = deriData(data, δs0, nsites_s, di)

    ind    = findfirst(x->δs[x]>1.605, 1:len(δs))
    δs     = δs[ind:end]
    d_data = [d_data[i][ind:end] for i = 1:len(d_data)]

    d_data_max = [maximum(el) for el in d_data[sel_inds]]
    nsites_s = nsites_s[sel_inds]
    model(x, p) = p[1] * x .+ p[2]
    fit = curve_fit(model, log.(nsites_s), d_data_max, [1., 1.])
    x = log(nsites_s[1]-1):0.01:log(nsites_s[end]+10)
    y = model(x, fit.param)
    @show fit.param
    return d_data_max, x, y
end

function finSizeScal(d_data, nsites_s, δs, ylim, sel_inds)
    d_data_max = [maximum(el) for el in d_data]
    free_ener  = [freeEner.(d_data[i], d_data_max[i]) for i = sel_inds]
    ls         = nsites_s[sel_inds]
    dy         = [1e-10*ones(len(el)) for el in free_ener]
    δs_sel, y_sel, dy_sel = dataSel(δs, free_ener, dy, ylim)
    res = autoScal(ls, δs_sel, y_sel, dy_sel, [1., 0.8])
    δc, nuc = Optim.minimizer(res)
    x = [ls[i]^(1/nuc) * (δs_sel[i] .- δc) for i = 1:len(ls)]
    
    @show δc nuc
    return δc, nuc, x, y_sel
end




#%%------------------------------------------------------------
data      = load(joinpath(@__DIR__, "data/2024-08-24_geo_ent_mag_scaling_rb_2.3_d_5_n_61_301.jld2"))
# data      = load(joinpath(@__DIR__, "data/geo_ent_mag_scaling_rb_1.4_d_6_2024-06-12.jld2"))
δs0       = data["δ_s"]
nsites_s  = data["nsites_s"]

#%%------------------------------------------------------------
di = 1

δs, d_berry = deriData(2π * data["mag_s"][1], δs0, nsites_s, di)
δs, d_geo_ent = deriData(data["geo_ent_s"][1], δs0, nsites_s, di)
# d_berry = [-el for el in d_geo_ent[1:end]]

; 
#%%------------------------------------------------------------
sel_inds = 10:1:len(nsites_s)
nsites_sel = nsites_s[sel_inds]

ylim = 0.5
δc, nuc, x, y_sel = finSizeScal(d_berry, nsites_s, δs, ylim, sel_inds)

; 

#%%------------------------------------------------------------
fig = figr()
for i = 1:len(sel_inds)
    line(fig, x[i], y_sel[i], mode="markers", name=nsites_sel[i])
end
plot(fig)


#%%------------------------------------------------------------
berry_max = [maximum(el) for el in d_berry]
δ_m = [δs[findmax(el)[2]] for el in d_berry]
fig = figr()

for i = 2:len(nsites_s)
    x = nsites_s[i] * (δs .- δ_m[i])
    y = freeEner.(d_berry[i], berry_max[i])
    line(fig, x, y, mode="markers", name=nsites_s[i])
end

plot(fig)



#%%------------------------------------------------------------
fig = figr()
[line(fig, δs, d_berry[i], name=nsites_s[i]) for i in 1:len(nsites_s)]
plot(fig)




#%%------------------------------------------------------------






#%%------------------------------------------------------------
data      = load(joinpath(@__DIR__, "data/2024-08-24_geo_ent_mag_scaling_rb_2.3_d_5_n_61_301.jld2"))
# data      = load(joinpath(@__DIR__, "data/2025-01-17_geo_ent_mag_scaling_rb_2.3_d_5_n_61_301.jld2"))
nsites_s  = data["nsites_s"]
δs0       = data["δ_s"]

di = 1

δs, d_berry = deriData(2π * data["mag_s"][1], δs0, nsites_s, di)
δs, d_geo = deriData(data["geo_ent_s"][1], δs0, nsites_s, di)
d_geo = [-el for el in d_geo[1:end]]

sel_inds_berry = 25:1:len(nsites_s)-12
sel_inds_geo = 25:1:len(nsites_s)-12
ylim_berry = 0.5
ylim_geo = 0.2

δc, nuc, x_berry, y_berry = finSizeScal(d_berry, nsites_s, δs, ylim_berry, sel_inds_berry)
δc, nuc, x_geo, y_geo = finSizeScal(d_geo, nsites_s, δs, ylim_geo, sel_inds_geo)

; 

#%%------------------------------------------------------------
markers = ["o", "v", "^", "<", ">", "s", "D", "*", "p", "P"]
markers = repeat(markers, 5)
ax = pfig(1, 2, figsize=(16, 6), fontsize=35, framewidth=2.5, ticksize=8)

for i = 1:len(sel_inds_berry)
    ax[1].plot(x_berry[i], y_berry[i], markers[i])
end

layout(
    ax[1], 
    # xlim=[-9, 3], ylim=[-0.002, 0.06], 
    xlabel=L"L^{1/\nu}(\tilde{δ}-\tilde{δ}_c)", ylabel=L"1-e^{(\phi_B' - \phi_{B, m}')}", 
    ydtick=0.25, 
    legend=[L"L=%$(nsites_s[el])" for el in sel_inds_berry], 
    fontsize=25
)

for i = 1:len(sel_inds_geo)
    ax[2].plot(x_geo[i], y_geo[i], markers[i])
end

layout(
    ax[2], 
    # xlim=[-9, 3], ylim=[-0.002, 0.06], 
    xlabel=L"L^{1/\nu}(\tilde{δ}-\tilde{δ}_c)", ylabel=L"1-e^{(-\mathcal{E}' + \mathcal{E}_m')}", 
    ydtick=0.05, 
    legend=[L"L=%$(nsites_s[el])" for el in sel_inds_geo], 
    fontsize=25
)


savefig(joinpath(@__DIR__, "data_collapse_2.3.pdf"))




#%%------------------------------------------------------------











#%%------------------------------------------------------------
# data      = load(joinpath(@__DIR__, "data/2024-07-16_geo_ent_mag_scaling_rb_1.4_d_5.jld2"))
data      = load(joinpath(@__DIR__, "data/2025-01-17_geo_ent_mag_scaling_rb_2.3_d_5_n_61_301.jld2"))
nsites_s  = data["nsites_s"]
δs0       = data["δ_s"]

nsites_sel = 85:30:260
di = 1
δs, d_berry   = deriData(2π * data["mag_s"][1], δs0, nsites_s, di, nsites_sel)
δs, d_geo_ent = deriData(data["geo_ent_s"][1], δs0, nsites_s, di, nsites_sel)
d_geo_ent = [-el for el in d_geo_ent]

sel_inds = [5:len(nsites_s)-9...]
b_max, x_berry, y_berry = pseudoCrit(2π * data["mag_s"][1], δs0, nsites_s, 1, sel_inds)
g_max, x_geo, y_geo = pseudoCrit(-data["geo_ent_s"][1], δs0, nsites_s, 1, sel_inds)

nsites_s = nsites_s[sel_inds]
; 

#%%------------------------------------------------------------
ax = pfig(1, 2, figsize=(16, 6), fontsize=35, framewidth=2.5, ticksize=8)
# δlim = [0.8, 1.2]
δlim = [1.61, 2]
inds = [findfirst(x->x>δlim[1], δs), findlast(x->x<δlim[2], δs)]

for el in d_berry
    ax[1].plot(δs[inds[1]:inds[2]], el[inds[1]:inds[2]])
end

axin1 = ax[1].inset_axes([0.63, 0.55, 0.37, 0.45])
axin1.plot(log.(nsites_s), b_max, ".")
axin1.plot(x_berry, y_berry)

layout(
    ax[1], 
    # xlim=[0.85, 1.2], 
    xlabel=L"\delta/\Omega", 
    # ylabel=L"\partial \phi_B / \partial (\delta/\Omega)", 
    ylabel=L"\phi_B'", 
    ydtick=0.5, 
    legend=[L"L=%$(nsites_sel[i])" for i = 1:len(nsites_sel)], 
    loc="lower left", 
    fontsize=25
)
axin1.tick_params(labelsize=14)
axin1.set_xlabel(L"\ln L", fontsize=20)
axin1.set_ylabel(L"\phi_{B, m}'", fontsize=20)


for el in d_geo_ent
    ax[2].plot(δs[inds[1]:inds[2]], el[inds[1]:inds[2]])
end
axin2 = ax[2].inset_axes([0.63, 0.55, 0.37, 0.45])
axin2.plot(log.(nsites_s), g_max, ".")
axin2.plot(x_geo, y_geo)

layout(
    ax[2], 
    xlabel=L"\delta/\Omega", 
    # ylabel=L"\partial \phi_B / \partial (\delta/\Omega)", 
    ylabel=L"-\mathcal{E}'", 
    legend=[L"L=%$(nsites_sel[i])" for i = 1:len(nsites_sel)], 
    loc = "lower left", 
    fontsize=25
)
axin2.tick_params(labelsize=14)
axin2.set_xlabel(L"\ln L", fontsize=20)
axin2.set_ylabel(L"-\mathcal{E}_{m}'", fontsize=20)


# savefig(joinpath(@__DIR__, "berry_geo_2.3.pdf"))














#%%------------------------------------------------------------
using Dates

rb_hori = 2.3
data = []
data ∋ load(joinpath(@__DIR__, "data/2024-07-16_geo_ent_mag_scaling_rb_$(rb_hori)_d_5_n_31_193.jld2"))
data ∋ load(joinpath(@__DIR__, "data/2024-07-16_geo_ent_mag_scaling_rb_$(rb_hori)_d_5_n_199_253.jld2"))
data ∋ load(joinpath(@__DIR__, "data/2024-07-16_geo_ent_mag_scaling_rb_$(rb_hori)_d_5_n_259_301.jld2"))

d_cutoff = 5
δ_s = data[1]["δ_s"]

geo_ent_s = data[1]["geo_ent_s"][1]
ini_which_s = data[1]["ini_which_s"][1]
mag_s = data[1]["mag_s"][1]
nsites_s = data[1]["nsites_s"]

for i = 2:3
    geo_ent_s = hcat(geo_ent_s, data[i]["geo_ent_s"][1])
    ini_which_s = hcat(ini_which_s, data[i]["ini_which_s"][1])
    mag_s = hcat(mag_s, data[i]["mag_s"][1])
    nsites_s = [nsites_s..., data[i]["nsites_s"]...]
end

jldsave(
    joinpath(@__DIR__, "data/$(today())_geo_ent_mag_scaling_rb_$(rb_hori)_d_$(d_cutoff).jld2"), 
    nsites_s    = nsites_s,
    δ_s         = δ_s,
    geo_ent_s   = [geo_ent_s],
    mag_s       = [mag_s],
    ini_which_s = [ini_which_s],
    d_cutoff    = d_cutoff
)



















#%%------------------------------------------------------------
