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
include("Siegler_Model_Choice.jl")
include("../../../Utilities/Utilities.jl")
# project directory is current directory
proj_dir = pwd()
# rng seed
seed = 558
Random.seed!(seed)
#######################################################################################
#                                   Generate Data
#######################################################################################
# mismatch penalty
δ = 16.0
# retrieval threshold
τ = -0.45
# logistic scalar 
s = 0.5
parms = (mmp = true,noise = true)
stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
    (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
temp = mapreduce(x -> simulate(stimuli, parms; δ, τ, s), vcat, 1:5)
# get unique data points with counts to improve efficiency
data = unique_data(temp)
data = vcat(data...)
n_obs = length(data)
choice_reps = map(x->x.N, data)
choice = map(x->x.resp, data)

chunks = populate_memory()
n_chunks = length(chunks)
# slots indexed numerically 1,2,3
memory_slots = repeat([1:3;]', outer=n_chunks)
# convert chunks to a matrix with chunks as rows and addend1, addend2 and addend3 as columns
memory_values = [[values(c.slots)...]  for c in chunks]
memory_values = hcat(memory_values...)'
# convert chunks to a matrix with chunks as rows and addend1 and addend2 as columns
stimuli_values = [[d.num1, d.num2] for d in data]
stimuli_values = hcat(stimuli_values...)'
# slots indexed numerically 1,2,3
stimuli_slots = repeat([1:2;]', outer=n_obs)

stan_input = Dict("mp"=>1, "bll"=>0,"sa"=>0, "n_obs"=>n_obs,
  "memory_slots"=>memory_slots, "memory_values"=>memory_values, "n_slots"=>3, 
  "stimuli_slots"=>stimuli_slots,"stimuli_values"=>stimuli_values, "n_chunks"=>n_chunks,
  "choice"=>choice, "choice_reps"=>choice_reps)
#######################################################################################
#                                     Load Model
#######################################################################################
stream = open("Siegler_Choice_Model.stan","r")
Model = read(stream,String)
close(stream)

# number of chains
n_chains = 4
stanmodel = Stanmodel(Sample(save_warmup=false, num_warmup=1000,
  num_samples=1000, thin=1), nchains=n_chains, name="Siegler_Choice_Model", model=Model,
  printsummary=false, output_format=:mcmcchains, random = CmdStan.Random(seed))
#######################################################################################
#                                     Run Model
#######################################################################################
rc, chain, cnames = stan(stanmodel, stan_input, proj_dir)
chain = replacenames(chain, Dict(:tau=>:τ, :delta=>:δ))
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
font_size = 12
ch = group(chain, :δ)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(800,600))

ch = group(chain, :τ)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(800,600))

ch = group(chain, :s)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
pcτ = plot(p1, p2, p3, layout=(3,1), size=(800,600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(stimuli, parms; x...), chain, 1000)
preds = vcat(vcat(preds...)...)
df = DataFrame(preds)
p5 = response_histogram(df, stimuli)