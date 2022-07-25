cd(@__DIR__)
using Pkg 
Pkg.activate("../")
using SafeTestsets

@safetestset "Sampler Test" begin
    include("Sampler_Example.jl")
end
