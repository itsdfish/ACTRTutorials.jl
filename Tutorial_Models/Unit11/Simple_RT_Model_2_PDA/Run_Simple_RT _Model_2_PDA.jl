#######################################################################################
#                                   Load Packages
#######################################################################################
# change directory to this files containing folder
cd(@__DIR__)
# import package manager
using Pkg
# activate project environment
Pkg.activate("../../..")
# import required packages
using DifferentialEvolutionMCMC, StatsPlots, Revise, ACTRModels, MCMCChains
# import required model functions
include("Simple_RT_Model_2_PDA.jl")
# initialize random number generator
Random.seed!(66501);
#######################################################################################
#                                   Generate Data
#######################################################################################
# the number of trials
n_trials = 50
# true value of blc
blc = 1.25
# true value of τ
τ = 0.5
# perceptual-motor time
ter = (0.05 + 0.085 + 0.05) + (0.05 + 0.06)
parms = (noise = true,s = 0.3,ter = ter)
# generate data
data = map(x -> simulate(parms; blc, τ), 1:n_trials);
#######################################################################################
#                                 Estimate Parameters
#######################################################################################
# prior on blc and τ
priors = (
    blc = (Normal(1.25,.5),),
    τ = (Normal(.5, .5),)
)

# lower and upper bounds for blc and τ
bounds = ((-Inf,Inf),(-Inf,Inf))
# generate model object
model = DEModel(; priors, model=loglike, data, parms)

# generate sampler object
de = DE(;bounds, burnin=1000, priors)
n_iter = 2000
# perform parameter estimation 
chain = sample(model, de, MCMCThreads(), n_iter, progress=true)
#######################################################################################
#                                         Plot
#######################################################################################
pyplot()
pyplot()
ch = group(chain, :blc)
font_size = 12
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
p4 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:pooleddensity),
  grid=false, size=(250,100), titlefont=font(font_size), color=:black)
pcblc = plot(p1, p2, p3, p4, layout=(4,1), size=(600,600))

ch = group(chain, :τ)
p1 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:traceplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p2 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:autocorplot),
  grid=false, size=(250,100), titlefont=font(font_size))
p3 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:mixeddensity),
  grid=false, size=(250,100), titlefont=font(font_size))
p4 = plot(ch, xaxis=font(font_size), yaxis=font(font_size), seriestype=(:pooleddensity),
  grid=false, size=(250,100), titlefont=font(font_size), color=:black)
pcτ = plot(p1, p2, p3, p4, layout=(4,1), size=(600,600))
#######################################################################################
#                                  Posterior Predictive
#######################################################################################
preds = posterior_predictive(x -> simulate(parms; x...), chain, 1000)
correct = filter(x-> x.resp == 1, preds)
rts = map(x->x.rt, correct)
p_correct = mean(x->x.resp == 1, preds)
correct_dist = histogram(rts, xlabel="RT", ylabel="Density", xaxis=font(font_size), yaxis=font(font_size),
    grid=false, norm=true, color=:grey, leg=false, size=(600,300), title="Correct", titlefont=font(font_size),
    xlims=(0,1.5))
correct_dist[1][1][:y] *= p_correct

incorrect = filter(x-> x.resp == 2, preds)
rts = map(x->x.rt, incorrect)
incorrect_dist = histogram(rts, xlabel="RT", ylabel="Density", xaxis=font(font_size), yaxis=font(font_size),
    grid=false, norm=true, color=:grey, leg=false, size=(600,300), title="Incorrect", titlefont=font(font_size),
    xlims=(0,1.5))
incorrect_dist[1][1][:y] *= (1 - p_correct)
plot(correct_dist, incorrect_dist, layout=(2,1), ylims=(0,3), size=(800, 500))