cd(@__DIR__)
using Pkg
Pkg.activate("../../../../")
using SafeTestsets

@safetestset "Fan Model 2" begin
    using ACTRModels, Test, Distributions, DataFrames, Random, CmdStan
    include("../../Stan_Utilities/Chunks.jl")
    include("../../Stan_Utilities/Stimuli.jl")
    include("../../Stan_Utilities/Utilities.jl")
    include("Stan_Test_Functions.jl")
    include("../../Fan_Model_Stan/Fan_Model_2.jl")
    seed = 17644
    Random.seed!(seed)
    expose("Fan_Model_2.stan")
    δ = 0.5
    γ = 1.6
    n_blocks = 1
    parms = (τ=-.5, noise=true, s=.3, mmp=true, sa=true, blc=.3, ter=.845)
    temp = simulate(stimuli, slots, parms, n_blocks; δ, γ)
    data = vcat(temp...)
    allVals = [people places]
    uvals = unique(allVals)
    memory_values = stan_memory_values(allVals, uvals)
    memory_slots = [fill(1.0,length(places)) fill(2.0, length(places))]
    rts,resp,stimuli_slots,stimuli_values = parse_data_stan(data, uvals)

    stanLLs = map(1:length(data)) do i
        stanLL(0, .3, δ, γ, 1, 0, 0, parms.s*pi/sqrt(3), parms.τ, parms.ter, 1, 1, 0,
        length(slots.people), 2, memory_slots, memory_values,
        stimuli_slots[i,:], stimuli_values[i,:], resp[i], rts[i])
    end
    turingLLs = map(x->computeLL(parms, slots, [x];  δ, γ), data)
    @test turingLLs ≈ stanLLs atol=1e-6
end