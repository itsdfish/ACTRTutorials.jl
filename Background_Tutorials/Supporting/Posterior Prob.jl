cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Distributions, Plots

β(θ, N, n_sim) = rand(Beta(θ * N, (1 - θ) * N), n_sim)

function η(μ, f, n_sim)
    v = rand(Normal(log(μ / (1 - μ)), 2 * log(f) / 3), n_sim)
    @. v = exp(v)
    @. v = v / (1 + v)
    return v
end

function posterior(f; br, tp, fp, N)
    baserate = f(br..., N)
    truePos = f(tp..., N)
    falsePos = f(fp..., N)
    v = @. baserate / (1 - baserate) * (truePos / falsePos)
    @. v = v / (1 + v)
    return v
end

parms = (br = (.01,1000),fp = (.096,500),tp = (.80,500))
samples1 = posterior(β; parms..., N=10^5)
pyplot()
histogram(samples1, grid=false, color=:grey, norm=true, xlims=(0,1), bins=30,
    leg=false, xlabel="Theta", ylabel="Density", xaxis=font(8), yaxis=font(8),
    size=(300,150))
savefig("PosteriorBC.eps")
