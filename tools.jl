using PyCall
using LaTeXStrings
using Chain: @chain

go = pyimport("plotly.graph_objects")
px = pyimport("plotly.express")

len(arr)              = length(arr)
figr                  = go.Figure
plot(fig::PyObject)   = (pyimport("plotly.offline").plot(fig, include_mathjax="cdn"); return nothing)
Base.:∋(vec, x)       = push!(vec, x)
Base.:∋(vec, x...)    = push!(vec, x...)
dict(; args...)       = Dict(args)
dict(keys, vals)      = Dict(Pair.(keys, vals))
Base.getindex(d::Dict, keys...) = [d[key] for key in keys]
find(x, arr)          = findfirst(==(x), arr)

macro str(vec...)
    return ["$x" for x in vec]
end

macro sym(vec...)
    return vec
end

toarr(vec::Vector{Vector{T}}) where T = hcat(vec...)
function toarr(vec::Vector)
    dims, l = size(vec[1]), len(vec)
    arr = zeros(typeof(vec[1][1]), (dims..., l)...)
    for i = 1:l
        arr[CartesianIndices((dims..., i:i))] = vec[i]
    end
    return arr    
end

function pfig(nrows=1, ncols=1; returnfig=false, figsize=(6, 4), fontsize=18, fontfamily="Times New Roman", texfont="cm", framewidth=1, ticksize=5, kwargs...)
    rc("font", size=fontsize, family=fontfamily)
    rc("mathtext", fontset=texfont)
    rc("axes", linewidth=framewidth, prop_cycle=pyimport("cycler").cycler(color=px.colors.qualitative.Plotly))
    rc("xtick", direction="in", top=true)
    rc("ytick", direction="in", right=true)
    rc("xtick.major", width=framewidth, size=ticksize)
    rc("xtick.minor", visible=true, width=framewidth*0.8, size=ticksize*0.6)
    rc("ytick.major", width=framewidth, size=ticksize)
    rc("ytick.minor", visible=true, width=framewidth*0.8, size=ticksize*0.6)
    fig, ax = subplots(nrows, ncols; figsize=figsize, layout="constrained", kwargs...)

    if nrows*ncols > 1
        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
        for i in 1:nrows, j in 1:ncols
            ax[(j-1)*nrows+i].text(0.02, 0.91, labels[(i-1)*ncols+j], transform=ax[(j-1)*nrows+i].transAxes)
            ax[(j-1)*nrows+i].tick_params(axis="both", direction="in", top=true, right=true, width=framewidth)
        end
    end

    returnfig ? (return fig, ax) : (return ax)
end


function layout(ax; xlabel=nothing, ylabel=nothing, title=nothing, legend=nothing, xticks=nothing, yticks=nothing, xdtick=nothing, ydtick=nothing, xlim=nothing, ylim=nothing, xscale=nothing, yscale=nothing, xscilimits=nothing, yscilimits=nothing, kwargs...)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend !== nothing 
        if typeof(legend[1]) <: Vector
            ax.legend(legend[1]; title=legend[2], handlelength=.6, handletextpad=0.2, labelspacing=0.2, borderpad=0.2, kwargs...)
        else
            ax.legend(legend; handlelength=.6, handletextpad=0.2, labelspacing=0.2, borderpad=0.2, kwargs...)
        end
    end

    if xticks !== nothing 
        typeof(xticks[1]) <: Vector ? ax.set_xticks(xticks...) : ax.set_xticks(xticks)
    end

    if yticks !== nothing 
        typeof(yticks[1]) <: Vector ? ax.set_yticks(yticks...) : ax.set_yticks(yticks)
    end

    if xlim !== nothing 
        typeof(xlim) <: Vector ? ax.set_xlim(xlim) : ax.set_xlim(right=xlim)
    end

    if ylim !== nothing 
        typeof(ylim) <: Vector ? ax.set_ylim(ylim) : ax.set_ylim(top=ylim)
    end

    xdtick !== nothing && ax.xaxis.set_major_locator(pyimport("matplotlib.ticker").MultipleLocator(xdtick))
    ydtick !== nothing && ax.yaxis.set_major_locator(pyimport("matplotlib.ticker").MultipleLocator(ydtick))
    
    xscale !== nothing && ax.set_xscale(xscale)
    yscale !== nothing && ax.set_yscale(yscale)
    xscilimits !== nothing && ax.ticklabel_format(axis="y", style="sci", scilimits=xscilimits, useMathText=true)
    yscilimits !== nothing && ax.ticklabel_format(axis="y", style="sci", scilimits=yscilimits, useMathText=true)

    return nothing
end

