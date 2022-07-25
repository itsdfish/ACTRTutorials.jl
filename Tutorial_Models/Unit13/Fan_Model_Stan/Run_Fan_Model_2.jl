#######################################################################################
#                                   Load Packages
#######################################################################################
# set the working directory to the directory in which this file is contained
cd(@__DIR__)
# load the package manager
using Pkg
# activate the project environment
Pkg.activate("../../../")
# load the required packages
using CmdStan, Distributions, Random, StatsPlots, ACTRModels, CSV, DataFrames, MCMCChains
proj_dir = pwd()
n_chains = 4
include("Fan_Model_2.jl")
include("../Stan_Utilities/Chunks.jl")
include("../Stan_Utilities/Stimuli.jl")
include("../Stan_Utilities/Utilities.jl")
seed = 684478
Random.seed!(seed)
#######################################################################################
#                                   Generate Data
#######################################################################################
# true value for mismatch penalty parameter 
δ = 0.5
# true value for maximum association strength
γ = 1.6
n_reps = 5
# fixed parameters used in the model
parms = (blc=0.3, τ=-0.5, noise=true, sa=true, mmp=true, s=0.2, ter=.845)
# Generates data for Nblocks. Slots contains the slot-value pairs to populate memory
#stimuli contains the target and foil trials.
temp = simulate(stimuli, slots, parms, n_reps; γ, δ)
#Forces the data into a concrete type for improved performance
data = vcat(temp...)
allVals = [people places]
uvals = unique(allVals)
memory_values = stan_memory_values(allVals, uvals)
memory_slots = [fill(1.0,length(places)) fill(2.0,length(places))]
rts,resp,stimuli_slots,stimuli_values = parse_data_stan(data, uvals)

stan_input = Dict("mp"=>1, "bll"=>0,"sa"=>1, "ter"=>parms.ter, "s"=>parms.s, "tau"=>parms.τ, "blc"=>parms.blc, 
  "resp"=>resp, "rts"=>rts, "n_obs"=>length(data),"memory_slots"=>memory_slots, "memory_values"=>memory_values,
   "n_slots"=>2, "stimuli_slots"=>stimuli_slots,"stimuli_values"=>stimuli_values, "n_chunks"=>length(slots.people))
#######################################################################################
#                                     Load Model
#######################################################################################
stream = open("Fan_Model_2.stan", "r")
Model = read(stream, String)
close(stream)
stanmodel = Stanmodel(Sample(save_warmup=false, num_warmup=1000,
  num_samples=1000, thin=1), nchains=n_chains, name="Fan_Model_2", model=Model,
  printsummary=false, output_format=:mcmcchains, random = CmdStan.Random(seed))
#######################################################################################
#                                     Run Model
#######################################################################################
rc, chain, cnames = stan(stanmodel, stan_input, proj_dir)
chain = replacenames(chain, Dict("gamma"=>"γ", "delta"=>"δ"))
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
font_size = 12
ch = group(chain, :γ)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(600,600))

ch = group(chain, :δ)
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