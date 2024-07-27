cd(@__DIR__)
test_files = readdir()
filter!(x -> x != "runtests.jl" && x != "Sampler_Example.jl", test_files)
map(x -> include(x), test_files)
