#######################################################################################
#                                   Load Packages
#######################################################################################
cd(@__DIR__)
using Pkg, Distributed
Pkg.activate("../../..")
using StatsPlots, DataFrames, MCMCChains
addprocs(4)
@everywhere begin
  push!(LOAD_PATH, "../../..")
  using ACTRModels, Distributions, DifferentialEvolutionMCMC
  include("Siegler_Model_Choice.jl")
  include("../../../Utilities/Utilities.jl")
  Random.seed!(774145)
end
#run(`for pid in $(pgrep "julia"); do cpulimit -l 50 -b -p $pid; done`)
#######################################################################################
#                                   Generate Data
#######################################################################################
n_blocks = 5
parms = (δ=16.0,)
fixed_parms = (d=.5, s=.5, τ=-.45,bll=false, mmp = true,noise = true,mmpFun = sim_fun,ter = 2.05)
stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
    (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
data = map(_->simulate(stimuli, fixed_parms; parms...), 1:n_blocks)
#data = map(x->filter(y->y.num1==x.num1 && y.num2==x.num2 , temp), stimuli)
#######################################################################################
#                                    Define Model
#######################################################################################
priors = (
    δ = (Normal(16, 8),),
)

bounds = ((-Inf,Inf),)
#loglike(δ, stimuli, fixed_parms, data; n_sim=1000)
model = DEModel(stimuli, fixed_parms; priors, model=loglike, data)
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
de = DE(;bounds, burnin=1000, priors, n_groups=1, Np=8)
n_iter = 2000
chain = sample(model, de, MCMCThreads(), n_iter, progress=true)
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
p4 = rt_histogram(df, stimuli)
p5 = response_histogram(df, stimuli)
savefig(p4, "Siegler_RT_Predictions.eps")
savefig(p5, "Siegler_Response_Predictions.eps")
savefig(posteriors, "Siegler_Posteriors.eps")