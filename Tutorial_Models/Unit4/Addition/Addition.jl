using Parameters, Distributions, StatsFuns

function simulate(N; s, blc)
    # construct the model
    model = construct_model(blc, s)
    # return random samples
    return rand(model, N)
end

function construct_model(α, s)
    λ = s * π / sqrt(3)
    # convolve perceptual motor processes
    μ, σ = convolve_normal(; motor = (μ = 0.06, N = 1), cr = (μ = 0.05, N = 13))
    # convolve perceptual motor and memory components
    model =
        Normal(μ, σ) + LogNormal(-α, λ) + LogNormal(-α, λ) +
        LogNormal(-α, λ) + LogNormal(-α, λ) + LogNormal(-α, λ)
    return model
end

function loglike(data, blc, s)
    # construct the model
    model = construct_model(blc, s)
    # convolve the distributions
    convolve!(model)
    # evaluate the log likelihood of the data
    LL = logpdf.(model, data)
    return sum(LL)
end
