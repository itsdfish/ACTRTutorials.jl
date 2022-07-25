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
    Chunk(;name=:Bob, department=:accounting),
    Chunk(;name=:Alice, department=:HR)
    ]

# initialize declarative memory
declarative = Declarative(memory=chunks)

# specify model parameters: partial matching, noise, mismatch penalty, activation noise
Θ = (mmp=true, noise=true, δ=.5, s=.2)  

# create an ACT-R object with activation noise and partial matching
actr = ACTR(;declarative, Θ...)

# compute activation for each chunk
compute_activation!(actr; department=:accounting)
# get mean activation
μ = get_mean_activations(actr)
# standard deviation 
σ = Θ.s * pi / sqrt(3)
# lognormal race distribution object
dist = LNR(;μ=-μ, σ, ϕ=0.0)


# index for accounting
idx = find_index(actr; department=:accounting)
# generate retrieval times
rts = rand(dist, 10^5)
# extract rts for accounting
acc_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
# probability of retrieving accounting
p_acc = length(acc_rts)/length(rts)
# histogram of retrieval times
hist = plot(layout=(2,1))
x = 0:.01:3
# density for each x value
dens = pdf.(dist, idx, x)
# overlay PDF on histogram
font_size = 12
plot!(hist, x, dens, color=:darkorange, linewidth=2, xlims=(0,2.5), leg=false,
    title="Old", grid=false, ylims=(0,1.3), xlabel="RT (seconds)", ylabel="Density",
    size=(800,400), xaxis=font(font_size), yaxis=font(font_size), legendfontsize=10)
# index for accounting
idx = find_index(actr; department=:HR)
# generate retrieval times
# extract rts for accounting
hr_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
# density for each x value
dens = pdf.(dist, idx, x)
# overlay PDF on histogram
plot!(hist, x, dens, color=:darkorange, linewidth=2, xlims=(0,2.5), subplot=2,
    grid=false, leg=false, ylims=(0,1.3), xlabel="RT (seconds)", ylabel="Density",
    title="New", xaxis=font(font_size), yaxis=font(font_size), legendfontsize=10)

# add density lines to correct distribution
x_line1 = [.6,1,1.6]
density_max1 = pdf.(dist, 1, x_line1)
density_min1 = fill(0.0, length(x_line1))
plot!(x_line1', [density_min1';density_max1'], color=:black, subplot=1,
    linestyle=:dash)


# add density lines to incorrect distribution
x_line2 = [.9,1.3]
density_max2 = pdf.(dist, 2, x_line2)
density_min2 = fill(0.0, length(x_line2))
plot!(x_line2', [density_min2';density_max2'], color=:black, subplot=2,
    linestyle=:dash)
savefig("Likelihood_Example2.png")
