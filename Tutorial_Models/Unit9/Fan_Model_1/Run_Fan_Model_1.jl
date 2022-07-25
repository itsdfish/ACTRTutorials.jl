#######################################################################################
#                                   Load Packages
#######################################################################################
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../../")
# load the required packages
using CmdStan, Distributions, Random, StatsPlots, ACTRModels, CSV, DataFrames, MCMCChains
seed = 684478
Random.seed!(seed)
using Turing, StatsPlots, ACTRModels, DataFrames
include("../Common_Functions/Chunks.jl")
include("../Common_Functions/Stimuli.jl")
include("../Common_Functions/Utilities.jl")
include("../Common_Functions/Plotting.jl")
include("Fan_Model_1.jl")
Random.seed!(45240)
#######################################################################################
#                                   Generate Data
#######################################################################################
# True value for the maximum association parameter
γ = 1.6
# Number of repeated blocks
n_reps = 5
# Fixed parameters used in the model
parms = (blc=0.3, τ=-0.5, sa=true, noise=true, s=0.2, ter=0.845)
# Generates data for n_reps. Slots contains the slot-value pairs to populate memory
temp = simulate(stimuli, slots, parms, n_reps; γ)
# Forces the data into a concrete type for improved performance
data = vcat(temp...)
#######################################################################################
#                                    Define Model
#######################################################################################
#Creates a model object and passes it to each processor
@model model(data, slots, parms) = begin
    #Prior distribution for the maximum association parameter
    γ ~ truncated(Normal(1.6, 0.8), 0, 4)
    data ~ Fan(γ, parms ,slots)
end
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# Settings of the NUTS sampler.
n_samples = 2000
n_adapt = 1000
n_chains = 4
# #Collects sampler configuration options
specs = NUTS(n_adapt, 0.8)
#Start sampling.
chain = sample(model(data, slots, parms), specs, MCMCThreads(), n_samples, n_chains, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
ch = group(chain, :γ)
font_size = 12
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(600,600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
sim(p) = simulate(stimuli, slots, parms, n_reps; p...)
preds = posterior_predictive(x->sim(x), chain, 1000, summarize)
df = vcat(preds...)
fan_effect = filter(x->x.resp == :yes && x.trial == :target, df)
df_data = DataFrame(data)
filter!(x->x.resp == :yes && x.trial == :target, df_data)
groups = groupby(df_data, [:fanPlace,:fanPerson])
data_means = combine(groups, :rt=>mean).rt_mean
title = [string("place: ",i," ","person: ",j) for i in 1:3 for j in 1:3]
title = reshape(title, 1, 9)
p4 = @df fan_effect histogram(:MeanRT,group=(:fanPlace,:fanPerson), ylabel="Density",
    xaxis=font(font_size), yaxis=font(font_size), grid=false, norm=true, color=:grey, leg=false, xticks=[1.0,1.3,1.6,1.9],
    titlefont=font(font_size), title=title, layout=9, xlims=(1.0,2.0), ylims=(0,8), bins=15)
vline!(p4, data_means', color=:darkred, size=(800,600))