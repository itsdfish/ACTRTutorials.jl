using Parameters, Distributions, StatsFuns

function simulate(N; s, blc)
    # construct the model object
    model = construct_model(blc, s)
    # generate N simulated trials
    return rand(model, N)
end

function construct_model(α, s)
    # standard deviation of activation noise in log space
    λ = s * π / sqrt(3)
    # μ and σ parameters for perceptual motor processes
    μ, σ = convolve_normal(motor = (μ = 0.21, N = 1), cr = (μ = 0.05, N = 11),
        visual = (μ = 0.085, N = 2),
        imaginal = (μ = 0.2, N = 1))
    # create a model object: perceptual motor processes with two memory retrievals
    model = Normal(μ, σ) + LogNormal(-α, λ) + LogNormal(-α, λ)
    return model
end

function loglike(data, blc, s)
    # construct model object
    model = construct_model(blc, s)
    # convolve the component distributions
    convolve!(model)
    # compute the sum log likelihood across all data points
    LL = logpdf.(model, data)
    return sum(LL)
end
