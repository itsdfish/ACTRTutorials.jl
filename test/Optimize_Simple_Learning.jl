cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Optimize Simple Learning Model" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit12/Optimize_Simple_Learning/Simple_Learning.jl")
    Random.seed!(5045)
    n_trials = 10_000
    d = 0.5
    blc = 1.5
    fixed_parms = (τ = 0.5,s = 0.4,bll = true,noise = true)
    N = 5
    delay = 50.0
    data = map(x->simulate(d, blc, delay, N; fixed_parms...), 1:10_000)
    x = range(.8*blc, 1.2*blc, length=100)
    y = map(x->all_LL(data, d, x, delay, N; fixed_parms...), x)
    _,idx = findmax(y)
    blc′ = x[idx]
    @test blc ≈ blc′ atol = 5e-2
    x = range(.8*d, 1.2*d, length=100)
    y = map(x->all_LL(data, x, blc, delay, N; fixed_parms...), x)
    _,idx = findmax(y)
    d′ =  x[idx]
    @test d ≈ d′ atol = 5e-2
end