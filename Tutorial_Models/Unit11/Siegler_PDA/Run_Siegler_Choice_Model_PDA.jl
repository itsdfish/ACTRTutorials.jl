#######################################################################################
#                                   Load Packages
#######################################################################################
# set current directory to directory that contains this file
cd(@__DIR__)
using Pkg, Distributed
# activate tutorial environment
Pkg.activate("../../..")
# load packages
using StatsPlots, DataFrames, MCMCChains
using ACTRModels, Distributions, DifferentialEvolutionMCMC
# specify the number of processors
addprocs(4)

@everywhere begin
  # path to the tutorial environment
  push!(LOAD_PATH, "../../..")
  # required packages
  using ACTRModels, Distributions, DifferentialEvolutionMCMC
  # required model code
  include("Siegler_Model_Choice_PDA.jl")
  include("../../../Utilities/Utilities.jl")
  # seed the random number generator
  Random.seed!(774145)
end
#run(`for pid in $(pgrep "julia"); do cpulimit -l 50 -b -p $pid; done`)
#######################################################################################
#                                   Generate Data
#######################################################################################
n_blocks = 5
parms = (δ=16.0,)
fixed_parms = (s=.5, τ=-.45, mmp = true,noise = true,mmp_fun = sim_fun,ter = 2.05)
stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
    (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
temp = mapreduce(_->simulate(stimuli, fixed_parms; parms...), vcat, 1:n_blocks)
temp = unique_data(temp)
sort!(temp)
data = map(x->filter(y->y.num1==x.num1 && y.num2==x.num2 , temp), stimuli)
#######################################################################################
#                                    Define Model
#######################################################################################
# prior distribution over δ
priors = (
    δ = (Normal(16, 8),),
)

# boundaries for δ
bounds = ((-Inf,Inf),)

# model object.
model = DEModel(stimuli, fixed_parms; priors, model=loglike, data)
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
de = DE(;bounds, burnin=1000, priors, n_groups=1, Np=8)
n_iter = 2000
chain = sample(model, de, n_iter, progress=true)
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
pooleddensity(ch, grid=false, xaxis=font(font_size), yaxis=font(font_size), size=(800,250))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(stimuli, fixed_parms; x...), chain, 1000)
preds = vcat(vcat(preds...)...)
df = DataFrame(preds)
p5 = response_histogram(df, stimuli)