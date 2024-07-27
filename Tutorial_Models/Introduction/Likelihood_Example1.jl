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
# histogram of retrieval times
hist = plot(layout = (2, 1))
histogram!(hist, acc_rts, color = :grey, leg = false, grid = false, size = (800, 400),
    bins = 100, norm = true, xlabel = "Reaction Time", ylabel = "Density",
    title = "Old", linewidth = 2, xaxis = font(font_size), yaxis = font(font_size),
    legendfontsize = 10)
# weight histogram according to retrieval probability
hist[1][1][:y] *= p_acc
# collection of retrieval time values
x = 0:0.01:2.5
# density for each x value
dens = pdf.(dist, idx, x)
# overlay PDF on histogram
plot!(hist, x, dens, color = :darkorange, linewidth = 1.5, xlims = (0, 2.5))

# index for accounting
idx = find_index(actr; department = :HR)
# extract rts for HR
hr_rts = filter(x -> x[1] == idx, rts) .|> x -> x[2]

histogram!(hist, hr_rts, color = :grey, leg = false, grid = false, size = (800, 400),
    bins = 100, norm = true, xlabel = "Reaction Time", ylabel = "Density", subplot = 2,
    title = "New", linewidth = 2, xaxis = font(font_size), yaxis = font(font_size),
    legendfontsize = 10)
# weight histogram according to retrieval probability
hist[2][1][:y] *= (1 - p_acc)
# density for each x value
dens = pdf.(dist, idx, x)
# overlay PDF on histogram
plot!(hist, x, dens, color = :darkorange, linewidth = 1.5, xlims = (0, 2.5), subplot = 2)
