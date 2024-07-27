# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../..")
# load the required packages

using Distributions, ACTRModels, Random, Plots

Random.seed!(87545)
# create chunks of declarative knowledge
chunks = [
    Chunk(; name = :Bob, department = :accounting),
    Chunk(; name = :Alice, department = :HR)
]

# initialize declarative memory
declarative = Declarative(memory = chunks)

# specify model parameters: partial matching, noise, mismatch penalty, activation noise
Θ = (mmp = true, noise = true, δ = 0.5, s = 0.2)

# create an ACT-R object with activation noise and partial matching
actr = ACTR(; declarative, Θ...)

# compute activation for each chunk
compute_activation!(actr; department = :accounting)
# get mean activation
μ = get_mean_activations(actr)
# standard deviation 
σ = Θ.s * pi / sqrt(3)
# lognormal race distribution object
dist = LNR(; μ = -μ, σ, ϕ = 0.0)

# index for accounting
idx = find_index(actr; department = :accounting)
# generate retrieval times
rts = rand(dist, 10^5)
# extract rts for accounting
acc_rts = filter(x -> x[1] == idx, rts) .|> x -> x[2]
# probability of retrieving accounting
p_acc = length(acc_rts) / length(rts)

pyplot()
font_size = 12
plots = plot(color = :grey, grid = false, size = (800, 400), legend = true,
    bins = 100, xlabel = "Mean Reaction Time", ylabel = "", xlims = (0.5, 2),
    layout = (2, 1), linewidth = 2, xaxis = font(font_size), yaxis = font(font_size),
    legendfontsize = 10)
plot!(subplot = 1, title = "Old")
means = [mean(acc_rts), mean(acc_rts) + 0.1]
vline!([means[1]], color = :darkred, label = "model", linewidth = 2)
vline!([means[2]], color = :black, label = "data", linewidth = 2)
plot!(
    means,
    [2, 2],
    linestyle = :dash,
    color = :grey,
    subplot = 1,
    label = "Difference",
    linewidth = 2
)

# index for HR
idx = find_index(actr; department = :HR)
# extract rts for accounting
hr_rts = filter(x -> x[1] == idx, rts) .|> x -> x[2]
plot!(subplot = 2, title = "New")
means = [mean(hr_rts), mean(hr_rts) + 0.1]
vline!([means[1]], color = :darkred, label = "model", subplot = 2, linewidth = 2)
vline!([means[2]], color = :black, label = "data", subplot = 2, linewidth = 2)
plot!(
    means,
    [2, 2],
    linestyle = :dash,
    color = :grey,
    subplot = 2,
    label = "Difference",
    linewidth = 2
)
