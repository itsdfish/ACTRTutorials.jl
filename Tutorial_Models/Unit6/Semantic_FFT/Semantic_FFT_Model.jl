using Parameters, StatsBase, NamedTupleTools

function populate_memory(act = 0.0)
    chunks = [
        Chunk(object = :shark, attribute = :dangerous, value = :True, act = act),
        Chunk(object = :shark, attribute = :locomotion, value = :swimming, act = act),
        Chunk(object = :shark, attribute = :category, value = :fish, act = act),
        Chunk(object = :salmon, attribute = :edible, value = :True, act = act),
        Chunk(object = :salmon, attribute = :locomotion, value = :swimming, act = act),
        Chunk(object = :salmon, attribute = :category, value = :fish, act = act),
        Chunk(object = :fish, attribute = :breath, value = :gills, act = act),
        Chunk(object = :fish, attribute = :locomotion, value = :swimming, act = act),
        Chunk(object = :fish, attribute = :category, value = :animal, act = act),
        Chunk(object = :animal, attribute = :moves, value = :True, act = act),
        Chunk(object = :animal, attribute = :skin, value = :True, act = act),
        Chunk(object = :canary, attribute = :color, value = :yellow, act = act),
        Chunk(object = :canary, attribute = :sings, value = :True, act = act),
        Chunk(object = :canary, attribute = :category, value = :bird, act = act),
        Chunk(object = :ostritch, attribute = :flies, value = :False, act = act),
        Chunk(object = :ostritch, attribute = :height, value = :tall, act = act),
        Chunk(object = :ostritch, attribute = :category, value = :bird, act = act),
        Chunk(object = :bird, attribute = :wings, value = :True, act = act),
        Chunk(object = :bird, attribute = :locomotion, value = :flying, act = act),
        Chunk(object = :bird, attribute = :category, value = :animal, act = act)
    ]
    return chunks
end

function simulate(fixed_parms, stimulus, n_reps; blc, δ)
    # generate chunks 
    chunks = populate_memory()
    # add chunks to declarative memory
    memory = Declarative(; memory = chunks)
    # add declarative memory and parameters to ACT-R object
    actr = ACTR(; declarative = memory, fixed_parms..., blc, δ)
    # rts for yes responses
    yes_rts = Float64[]
    # rates for no respones
    no_rts = Float64[]
    for rep = 1:n_reps
        # simulate a single trial
        resp, rt = simulate_trial(actr, stimulus)
        # save simulated data
        resp == :yes ? push!(yes_rts, rt) : push!(no_rts, rt)
    end
    return (stimulus = stimulus, yes_rts = yes_rts, no_rts = no_rts)
end

function simulate_trial(actr, stimulus)
    retrieving = true
    probe = stimulus
    response = :_
    # add conflict resolution times
    rt = mapreduce(_ -> process_time(0.05), +, 1:7)
    # add stimulus encoding times
    rt += mapreduce(_ -> process_time(0.085), +, 1:2)
    while retrieving
        # conflict resolution
        rt += process_time(0.05)
        chunk = retrieve(actr; object = probe.object, attribute = :category)
        rt += compute_RT(actr, chunk)
        # retrieval failure, respond "no"
        if isempty(chunk)
            # add motor execution time
            rt += process_time(0.05) + process_time(0.21)
            retrieving = false
            response = :no
            # can respond "yes"
        elseif direct_verify(chunk[1], probe)
            # add motor execution time
            rt += process_time(0.05) + process_time(0.21)
            retrieving = false
            response = :yes
            # category chain
        elseif chain_category(chunk[1], probe)
            probe = delete(probe, :object)
            # update memory probe for category chain
            probe = (object = chunk[1].slots.value, probe...)
        else
            response = :no
            rt += process_time(0.05) + process_time(0.21)
            retrieving = false
        end
    end
    return response, rt
end

process_time(μ) = rand(Uniform(μ * (2 / 3), μ * (4 / 3)))

function direct_verify(chunk, stim)
    return match(chunk, object = stim.object,
        value = stim.category, attribute = :category)
end

function chain_category(chunk, stim)
    return match(chunk, ==, !=, ==, object = stim.object,
        value = stim.category, attribute = :category)
end

function get_stimuli()
    stimuli = NamedTuple[]
    push!(stimuli, (object = :canary, category = :bird, ans = :yes))
    push!(stimuli, (object = :canary, category = :animal, ans = :yes))
    push!(stimuli, (object = :bird, category = :fish, ans = :no))
    push!(stimuli, (object = :canary, category = :fish, ans = :no))
    return vcat(stimuli...)
end

function loglike(data, fixed_parms, blc, δ)
    # populate memory
    chunks = populate_memory()
    # add chunks to declarative memory object
    memory = Declarative(; memory = chunks)
    # add declarative memory and parameters to ACT-R object
    actr = ACTR(; declarative = memory, fixed_parms..., blc, δ, noise = false)
    LL = 0.0
    for d in data
        stimulus = d.stimulus
        # evaluate log likelihood based on maximum number of catory chains
        if (stimulus.object == :canary) && (stimulus.category == :bird)
            LL += zero_chains(actr, stimulus, d)
        elseif (stimulus.object == :canary) && (stimulus.category == :fish)
            LL += two_chains(actr, stimulus, d)
        else
            LL += one_chain(actr, stimulus, d)
        end
    end
    return LL
end

function zero_chains(actr, stimulus, data)
    # this function computes yes and no responses that can be answered directly
    # no category chaining
    # log likelihood for yes responses
    LL = zero_chain_yes(actr, stimulus, data.yes_rts)
    # log likelihood for no responses
    LL += zero_chain_no(actr, stimulus, data.no_rts)
    return LL
end

function one_chain(actr, stimulus, data)
    # handles yes and no responses for a single category chain
    LL = one_chain_yes(actr, stimulus, data.yes_rts)
    LL += one_chain_no(actr, stimulus, data.no_rts)
    return LL
end

function two_chains(actr, stimulus, data)
    # no is the only response for two category chains
    return two_chains_no(actr, stimulus, data.no_rts)
end

function one_chain_no(actr, stimulus, rts)
    # likelihood of respond no after initial retrieval
    likelihoods = one_chain_no_branch1(actr, stimulus, rts)
    # likelihood of responding no after first category chain
    likelihoods .+= one_chain_no_branch2(actr, stimulus, rts)
    return sum(log.(likelihoods))
end

function two_chains_no(actr, stimulus, rts)
    # no category chains
    likelihoods = two_chains_no_branch1(actr, stimulus, rts)
    # one category chain
    likelihoods .+= two_chains_no_branch2(actr, stimulus, rts)
    # two category chains
    likelihoods .+= two_chains_no_branch3(actr, stimulus, rts)
    return sum(log.(likelihoods))
end

function two_chains_no_branch1(actr, stimulus, rts)
    return one_chain_no_branch1(actr, stimulus, rts)
end

function two_chains_no_branch2(actr, stimulus, rts)
    return one_chain_no_branch2(actr, stimulus, rts)
end

function zero_chain_yes(actr, stimulus, rts)
    @unpack τ, s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm, σpm = convolve_normal(
        motor = (μ = 0.21, N = 1),
        cr = (μ = 0.05, N = 9),
        visual = (μ = 0.085, N = 2)
    )
    compute_activation!(actr; object = get_object(stimulus), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    yes_idx =
        find_index(actr, object = get_object(stimulus), value = get_category(stimulus))
    retrieval_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = yes_idx)
    model = Normal(μpm, σpm) + retrieval_dist
    convolve!(model)
    LLs = logpdf.(model, rts)
    return sum(LLs)
end

function zero_chain_no(actr, stimulus, rts)
    @unpack τ, s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    # normal approximation for perceptual-motor time
    μpm, σpm = convolve_normal(
        motor = (μ = 0.21, N = 1),
        cr = (μ = 0.05, N = 9),
        visual = (μ = 0.085, N = 2)
    )
    # compute the activation for all chunks
    compute_activation!(actr; object = get_object(stimulus), attribute = :category)
    # extract the mean activation values
    μ = map(x -> x.act, chunks)
    # add retrieval threshold to mean activation values
    push!(μ, τ)
    # find the chunk index corresponding to a "yes" response
    yes_idx =
        find_index(actr, object = get_object(stimulus), value = get_category(stimulus))
    # Initialize likelihood
    n_resp = length(rts)
    # initialize likelihoods for each no response
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i = 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != yes_idx
            # create Lognormal race distribution for chunk i
            retrieval_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = i)
            # sum the percptual-motor distribution and the retrieval distribution
            model = Normal(μpm, σpm) + retrieval_dist
            # convolve the distributions
            convolve!(model)
            # compute likelihood for each rt using "." broadcasting
            likelihoods .+= pdf.(model, rts)
        end
    end
    return sum(log.(likelihoods))
end

function one_chain_yes(actr, stimulus, rts)
    probe = stimulus
    @unpack τ, s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm, σpm = convolve_normal(
        motor = (μ = 0.21, N = 1),
        cr = (μ = 0.05, N = 10),
        visual = (μ = 0.085, N = 2)
    )
    compute_activation!(actr; object = get_object(stimulus), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain_idx = find_index(
        actr,
        ==,
        !=,
        ==,
        object = get_object(probe),
        value = get_category(probe),
        attribute = :category
    )
    chain1_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = chain_idx)

    probe = (object = get_chunk_value(chunks[chain_idx]), delete(probe, :object)...)
    compute_activation!(actr; object = get_object(probe), attribute = :category)
    yes_idx = find_index(actr, object = get_object(probe), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    yes_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = yes_idx)
    model = Normal(μpm, σpm) + chain1_dist + yes_dist
    convolve!(model)
    LLs = logpdf.(model, rts)
    return sum(LLs)
end

function one_chain_no_branch1(actr, stimulus, rts)
    # respond no after initial retrieval
    probe = stimulus
    @unpack τ, s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    μpm, σpm = convolve_normal(
        motor = (μ = 0.21, N = 1),
        cr = (μ = 0.05, N = 9),
        visual = (μ = 0.085, N = 2)
    )
    compute_activation!(actr; object = get_object(stimulus), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain_idx = find_index(
        actr,
        ==,
        !=,
        ==,
        object = get_object(probe),
        value = get_category(probe),
        attribute = :category
    )
    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i = 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != chain_idx
            no_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = i)
            model = Normal(μpm, σpm) + no_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return likelihoods
end

function one_chain_no_branch2(actr, stimulus, rts)
    # respond no after first category chain
    probe = stimulus
    @unpack τ, s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    # perceptual motor time
    μpm, σpm = convolve_normal(
        motor = (μ = 0.21, N = 1),
        cr = (μ = 0.05, N = 10),
        visual = (μ = 0.085, N = 2)
    )
    # compute activations
    compute_activation!(actr; object = get_object(stimulus), attribute = :category)
    # get activations
    μ = map(x -> x.act, chunks)
    # add retrieval threshold
    push!(μ, τ)
    # index to chain chunk
    chain_idx = find_index(
        actr,
        ==,
        !=,
        ==,
        object = get_object(probe),
        value = get_category(probe),
        attribute = :category
    )
    chain1_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = chain_idx)

    # change probe for second retrieval
    probe = (object = get_chunk_value(chunks[chain_idx]), delete(probe, :object)...)
    compute_activation!(actr; object = get_object(probe), attribute = :category)
    yes_idx = find_index(actr, object = get_object(probe), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)

    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i = 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != yes_idx
            no_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = i)
            model = Normal(μpm, σpm) + chain1_dist + no_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return likelihoods
end

function two_chains_no_branch3(actr, stimulus, rts)
    probe = stimulus
    @unpack τ, s = actr.parms
    chunks = actr.declarative.memory
    σ = s * π / sqrt(3)
    # perceptual motor time
    μpm, σpm = convolve_normal(
        motor = (μ = 0.21, N = 1),
        cr = (μ = 0.05, N = 11),
        visual = (μ = 0.085, N = 2)
    )
    # compute activations
    compute_activation!(actr; object = get_object(stimulus), attribute = :category)
    # get activations
    μ = map(x -> x.act, chunks)
    # add retrieval threshold
    push!(μ, τ)
    # index to chain chunk
    chain_idx1 = find_index(
        actr,
        ==,
        !=,
        ==,
        object = get_object(probe),
        value = get_category(probe),
        attribute = :category
    )
    chain1_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = chain_idx1)

    # change probe for second retrieval
    probe = (object = get_chunk_value(chunks[chain_idx1]), delete(probe, :object)...)
    compute_activation!(actr; object = get_object(probe), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain2_idx = find_index(actr, object = get_object(probe), attribute = :category)
    chain2_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = chain2_idx)

    # change probe for third retrieval
    probe = (object = get_chunk_value(chunks[chain2_idx]), delete(probe, :object)...)
    compute_activation!(actr; object = get_object(probe), attribute = :category)
    μ = map(x -> x.act, chunks)
    push!(μ, τ)
    chain3_idx = find_index(actr, object = get_object(probe), attribute = :category)

    # Initialize likelihood
    n_resp = length(rts)
    likelihoods = fill(0.0, n_resp)
    Nc = length(chunks) + 1
    # Marginalize over all of the possible chunks that could have lead to the
    # observed response
    for i = 1:Nc
        # Exclude the chunk representing the stimulus because the response was "no"
        if i != chain3_idx
            no_dist = LNRC(; μ = -μ, σ = σ, ϕ = 0.0, c = i)
            model = Normal(μpm, σpm) + chain1_dist + chain2_dist + no_dist
            convolve!(model)
            likelihoods .+= pdf.(model, rts)
        end
    end
    return likelihoods
end

get_object(x) = x.object
get_category(x) = x.category
get_chunk_value(x) = x.slots.value

function merge(data)
    yes = map(x -> x.yes_rts, data) |> x -> vcat(x...)
    no = map(x -> x.no_rts, data) |> x -> vcat(x...)
    return (yes = yes, no = no)
end

function grid_plot(preds, stimuli; kwargs...)
    posterior_plots = Plots.Plot[]
    for (pred, stimulus) in zip(preds, stimuli)
        prob_yes = length(pred.yes) / (length(pred.yes) + length(pred.no))
        object = stimulus.object
        category = stimulus.category
        hist = histogram(layout = (1, 2), xlims = (0, 2.5), ylims = (0, 3.5),
            title = "$object-$category",
            grid = false, titlefont = font(12), xaxis = font(12), yaxis = font(12),
            xlabel = "Yes RT",
            xticks = 0:2, yticks = 0:3)

        if !isempty(pred.yes)
            histogram!(hist, pred.yes, xlabel = "Yes RT", norm = true, grid = false,
                color = :grey, leg = false,
                size = (300, 250), subplot = 1)
            hist[1][1][:y] *= prob_yes
        end

        prob_no = 1 - prob_yes
        object = stimulus.object
        category = stimulus.category
        histogram!(hist, pred.no, xlabel = "No RT", norm = true, grid = false,
            color = :grey, leg = false,
            size = (300, 250), subplot = 2)
        hist[2][1][:y] *= prob_no
        push!(posterior_plots, hist)
    end
    return posterior_plot =
        plot(posterior_plots..., layout = (2, 2), size = (800, 600); kwargs...)
end
