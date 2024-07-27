cd(@__DIR__)
using Pkg
Pkg.activate("../")
using SafeTestsets

@safetestset "Siegler PDA" begin
    using ACTRModels, Test, Parameters, Random
    include("../Tutorial_Models/Unit11/Siegler_PDA/Siegler_Model_Choice_PDA.jl")
    include("../Utilities/Utilities.jl")
    Random.seed!(5045)
    n_blocks = 1000
    δ = 16.0
    parms = (δ = δ,)
    fixed_parms =
        (s = 0.5, τ = -0.45, mmp = true, noise = true, mmp_fun = sim_fun, ter = 2.05)
    stimuli = [(num1 = 1, num2 = 1), (num1 = 1, num2 = 2), (num1 = 1, num2 = 3),
        (num1 = 2, num2 = 2),
        (num1 = 2, num2 = 3), (num1 = 3, num2 = 3)]
    temp = mapreduce(_ -> simulate(stimuli, fixed_parms; parms...), vcat, 1:n_blocks)
    temp = unique_data(temp)
    sort!(temp)
    data = map(x -> filter(y -> y.num1 == x.num1 && y.num2 == x.num2, temp), stimuli)

    x = range(0.8 * δ, 1.2 * δ, length = 100)
    y = map(x -> loglike(data, stimuli, fixed_parms, x; n_sim = 10^3), x)
    mxv, mxi = findmax(y)
    δ′ = x[mxi]
    @test δ ≈ δ′ atol = 0.2
end
