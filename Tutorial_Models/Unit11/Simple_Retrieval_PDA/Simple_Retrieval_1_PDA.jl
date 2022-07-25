using Parameters, Distributions, StatsFuns

function simulate(parms, n_trials; τ)
    θ = get_retrieval_prob(parms, n_trials; τ)
    # Simulate n_trials
    data = rand(Binomial(n_trials, θ))
    return data
end

function get_retrieval_prob(parms, n_trials; τ)
    # Create a chunk object
    chunk = Chunk()
    # Create a declarative memory object
    memory = Declarative(;memory=[chunk])
    # Create an ACTR object
    actr = ACTR(;declarative=memory, parms..., τ)
    # Compute the retrieval probability of the chunk
    θ,_ = retrieval_prob(actr, chunk)
    return θ
end

function loglike(k, τ; n_trials, parms, n_sim = 10^4)
    # return the retrieval probability 
    θ = get_retrieval_prob(parms, n_trials; τ)
    # indicator function that returns 1 if simulation returns k retrievals
    counter(_) = rand(Binomial(n_trials, θ)) == k ? 1 : 0
    # repeat simulation n_sim times while counting the number of cases in which 
    # k retrievals were found
    cnt = mapreduce(counter, +, 1:n_sim)
    # return the approximate log probability of k retrievals
    return log(cnt / n_sim)
end
