using Turing, ACTRModels, Test, Distributions, Random, StatsPlots
    include("../Tutorial_Models/Unit6/Siegler_RT/Siegler_Model_RT.jl")
    Random.seed!(215977)
    #######################################################################################
    #                                   Generate Data
    #######################################################################################
    δ = 16.0
    τ = -.45
    s = .5
    parms = (mmp = true,noise = true,mmpFun = sim_fun,ter = 2.05)
    stimuli = [(num1 = 1,num2 = 1), (num1 = 1,num2 = 2), (num1 = 1,num2 = 3), (num1 = 2,num2 = 2),
        (num1 = 2,num2 = 3), (num1 = 3,num2 = 3)]
    temp = simulate(stimuli, parms; δ=δ, τ=τ, s=s)
    data = vcat(temp...)
    #######################################################################################
    #                                    Define Model
    #######################################################################################
    @model model(data, parms) = begin
        δ ~ Normal(16, 8)
        τ = -.45
        s = .5
        data ~ Siegler(δ, τ, s, parms)
    end
    #######################################################################################
    #                                 Estimate Parameters
    #######################################################################################
    # Settings of the NUTS sampler.
    n_samples = 2000
    delta = .85
    n_adapt = 1000
    specs = NUTS(n_adapt, delta)
    # Start sampling.
    chain = sample(model(data, parms), specs, n_samples, progress=true)