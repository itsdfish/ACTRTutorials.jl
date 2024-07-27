#######################################################################################
#                                 Load Packages
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../../")
using Revise, AdaptiveDesignOptimization, Random, ACTRModels, Distributions
using Parameters
include("Simple_Learning.jl")
Random.seed!(19104)
#######################################################################################
#                                  Define Design
#######################################################################################
# τ: retrieval threshold
# s: activation noise
fixed_parms = (τ = 0.5, s = 0.4, bll = true, noise = true)

# model object
model = Model(; loglike, fixed_parms...)

# d: decay rate
# blc: base level constant
parm_list = (d = range(0.1, 0.9, length = 15),
    blc = range(0.0, 2.5, length = 15)
)

# delay: time between practice and test
# N: number of practices
design_list = (
    delay = range(0.1, 15, length = 20) .|> x -> x^2,
    N = range(1, 10, length = 10) .|> Int
)

# retrieved or not
data_list = (retrieved = [true, false],)
#######################################################################################
#                              Optimize Experiment
#######################################################################################
using DataFrames
# true parameter values
true_parms = (d = 0.5, blc = 1.5)
# number of trials
n_trials = 100
# define optimizer object
optimizer = Optimizer(; design_list, parm_list, data_list, model);
# define best design
design = optimizer.best_design
# initialize a dataframe for storing posterior mean and standard deviation of each parameter
df = DataFrame(design = Symbol[], trial = Int[], mean_d = Float64[], mean_blc = Float64[],
    std_d = Float64[], std_blc = Float64[], delay = Float64[], N = Int[])
# initial values
new_data = [:optimal, 0, mean_post(optimizer)..., std_post(optimizer)...,
    design...]
push!(df, new_data)
for trial = 1:n_trials
    # simulate a trial
    data = simulate(true_parms..., design...; fixed_parms...)
    # update posterior distribution and return best design for next trial
    design = update!(optimizer, data)
    # add mean and standard deviation of posterior distributions to dataframe
    new_data = [:optimal, trial, mean_post(optimizer)..., std_post(optimizer)...,
        design...]
    push!(df, new_data)
end
#######################################################################################
#                              Random Experiment
#######################################################################################
randomizer = Optimizer(; design_list, parm_list, data_list, model, design_type = Randomize);
design = randomizer.best_design
new_data = [:random, 0, mean_post(randomizer)..., std_post(randomizer)...,
    design...]
push!(df, new_data);

for trial = 1:n_trials
    data = simulate(true_parms..., design...; fixed_parms...)
    design = update!(randomizer, data)
    new_data = [:random, trial, mean_post(randomizer)..., std_post(randomizer)...,
        design...]
    push!(df, new_data)
end
#######################################################################################
#                                 Plot Results
#######################################################################################
using StatsPlots
pyplot()
@df df plot(:trial, :std_d, xlabel = "trial", ylabel = "σ of d", grid = false,
    group = :design, ylims = (0, 0.3),
    xaxis = font(12), yaxis = font(12), legendfont = font(9), linewidth = 1.5,
    size = (800, 400))
hline!([0.08], label = "termination point", color = :black, line = :dash)

@df df plot(:trial, :std_blc, xlabel = "trial", ylabel = "σ of blc", grid = false,
    group = :design, ylims = (0, 1),
    xaxis = font(12), yaxis = font(12), legendfont = font(9), linewidth = 1.5,
    size = (800, 400))

@df df plot(:trial, :mean_d, xlabel = "Trial", ylabel = "Mean d", group = :design,
    grid = false,
    xaxis = font(12), yaxis = font(12), linewidth = 1.5, legendfont = font(9),
    size = (800, 400))
hline!([true_parms.d], label = "true")

@df df plot(:trial, :mean_blc, xlabel = "trial", ylabel = "mean blc", group = :design,
    grid = false,
    xaxis = font(12), yaxis = font(12), legendfont = font(9), linewidth = 1.5,
    size = (800, 400))
hline!([true_parms.blc], label = "true")

@df df histogram2d(:delay, :N, group = :design, grid = false, bins = 5,
    xlabel = "Delay (seconds)", ylabel = "Practices",
    xaxis = font(12), yaxis = font(12), title = ["Optimal" "Random"], layout = (2, 1),
    size = (800, 400))

max_delay = 300
n_retrievals = 10
delays = 0.01:0.01:max_delay
d = 0.5
blc = 1.5
noise = false
Ns = [1, 5, 10]
sim(N) = activation_dynamics(delays, N; blc, d, fixed_parms..., noise)
activations = map(n -> sim(n), Ns)
plot(delays, activations, grid = false, xlabel = "Delay (seconds)", ylabel = "Activation",
    size(800, 400), linewidth = 1.5,
    xaxis = font(12), yaxis = font(12), legendfont = font(9), labeltitle = "N",
    labels = ["1" "5" "10"])
