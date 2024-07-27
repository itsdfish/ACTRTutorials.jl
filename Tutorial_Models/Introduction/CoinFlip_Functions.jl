function simulate!(θ, n, sim_data)
    # number of successes
    h = 0
    # simulate n trials
    for t ∈ 1:n
        # increment h if successfull
        h += rand() ≤ θ ? 1 : 0
    end
    # update distribution count vector
    sim_data[h + 1] += 1
    return nothing
end

function computational(data, θ, n, n_sim)
    sim_data = fill(0, n + 1)
    map(x -> simulate!(0.5, n, sim_data), 1:n_sim)
    sim_data /= n_sim
    LL = 0.0
    for d in data
        LL += log(sim_data[d + 1])
    end
    return LL
end

function analytic(data, n, θ)
    LL = 0.0
    for d in data
        LL += logpdf(Binomial(n, θ), d)
    end
    return LL
end
