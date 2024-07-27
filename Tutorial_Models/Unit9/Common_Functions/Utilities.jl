"""
Computes the fan values for the stimulus set. Returns a NamedTuple
    for each trial.
"""
function count_fan(vals)
    un = (unique(vals)...,)
    uc = map(y -> count(x -> x == y, vals), un)
    return NamedTuple{un}(uc)
end

"""
Returns fan values for a given person-place pair
"""
function get_fan(vals, person, place)
    return (fanPerson = vals[:people][person], fanPlace = vals[:places][place])
end

"""
Computes mean RT for each fan condition. Used in posterior predictive
"""
summarize(vals) = summarize(DataFrame(vcat(vals...)))

function summarize(df::DataFrame)
    g = groupby(df, [:trial, :resp, :fanPerson, :fanPlace])
    return combine(g, :rt => mean => :MeanRT)
end

"""
Computes accuracy for each fan condition. Used in posterior predictive
"""
accuracy(vals) = accuracy(DataFrame(vcat(vals...)))
accuracy(df::DataFrame) = accuracy(df, [:trial, :fanPerson, :fanPlace])

function accuracy(df::DataFrame, factors)
    g = groupby(df, factors)
    compute_correct!(df)
    pred_correct = combine(g, accuracy = :correct => mean)
    return pred_correct
end

function compute_correct!(df)
    df[:, :correct] =
        (df[:, :resp] .== :yes) .& (df[:, :trial] .== :target) .|
        (df[:, :resp] .== :no) .& (df[:, :trial] .== :foil)
end

"""
Formats simulated data for Stan
"""
function parseDataStan(data, uvals)
    df = DataFrame(data)
    Nrows = size(df, 1)
    df = DataFrame(data)
    rts = df[!, :rt]
    idx = df[!, :resp] .== :yes
    resp = fill(1, Nrows)
    resp[idx] .+= 1
    person = fill(0.0, Nrows)
    for (i, u) in enumerate(uvals)
        idx = df[!, :person] .== u
        person[idx] .= i
    end
    place = fill(0.0, Nrows)
    for (i, u) in enumerate(uvals)
        idx = df[!, :place] .== u
        place[idx] .= i
    end
    stimuliValues = [person place]
    stimuliSlots = [fill(1.0, Nrows) fill(2.0, Nrows)]
    return rts, resp, stimuliSlots, stimuliValues
end

"""
Transforms symbols into integer ids for person-place memory values
"""
function stanMemoryValues(allVals, uvals)
    memoryValues = fill(0.0, size(allVals))
    for (i, u) in enumerate(uvals)
        idx = allVals .== u
        memoryValues[idx] .= i
    end
    return memoryValues
end

function show_predictions(
    trial,
    request,
    stimulus,
    slots,
    parms,
    sim_fun;
    add_rts = true,
    kwargs...
)
    fanCount = map(x -> countFan(x), slots)
    chunks = [Chunk(; person = pe, place = pl) for (pe, pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(; memory = chunks, parms..., kwargs...)
    #Initialize imaginal buffer
    imaginal = Imaginal(chunk = chunks[1])
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(; declarative = memory, imaginal = imaginal)
    imaginal.chunk = Chunk(; person = stimulus.person, place = stimulus.place)
    actr.declarative.parms.noise = true
    mean_rt_yes = NaN
    mean_rt_no = NaN
    if add_rts
        preds = map(x -> sim_fun(actr, [stimulus], slots), 1:(10^5))
        preds = vcat(preds...)
        rts_yes = filter(x -> x.resp == :yes, preds)
        rts_no = filter(x -> x.resp == :no, preds)
        isempty(rts_yes) ? (mean_rt_yes = -Inf) : (mean_rt_yes = mean(x -> x.rt, rts_yes))
        mean_rt_no = mean(x -> x.rt, rts_no)
    end
    actr.declarative.parms.noise = false
    map(x -> x.act_noise = 0.0, chunks)
    computeActivation!(actr; request...)
    results = retrievalRequest(actr; request...)
    fans = getFan(fanCount, stimulus.person, stimulus.place)
    f(; kwargs...) = (
        request = request,
        stimulus = stimulus,
        trial = trial,
        fans...,
        mean_rt_yes = mean_rt_yes,
        mean_rt_no = mean_rt_no,
        kwargs...
    )
    vals = map(
        x -> f(chunk = x.slots, act_pm = x.act_pm, act_sa = x.act_sa, act = x.act),
        results
    )
    return DataFrame(vals)
end

"""
Processes data so that it can be used with optimized Stan Full Fan model.
"""
function simplify_data(data, slots, parms; Θ...)
    df = DataFrame()
    for (i, d) in enumerate(data)
        temp = simplify_data_trial(i, d, slots, parms; Θ...)
        append!(df, temp)
    end
    return df
end

function simplify_data_trial(trial_id, stimulus, slots, parms; Θ...)
    fanCount = map(x -> countFan(x), slots)
    request = (person = stimulus.person, place = stimulus.place)
    chunks = [Chunk(; person = pe, place = pl) for (pe, pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(; memory = chunks, parms..., Θ...)
    #Initialize imaginal buffer
    imaginal = Imaginal(chunk = chunks[1])
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(; declarative = memory, imaginal = imaginal)
    imaginal.chunk = Chunk(; request...)
    memory.parms.noise = false
    map(x -> x.act_noise = 0.0, chunks)
    computeActivation!(actr; request...)
    results = retrievalRequest(actr; request...)
    stimulus_fans = getFan(fanCount, stimulus.person, stimulus.place)
    get_penalty(c::Chunk, r) = get_penalty(c.slots, r)
    get_penalty(c, r) = mapreduce(k -> count(c[k] != r[k]), +, keys(r))
    resp = stimulus.resp == :yes ? 2 : 1
    g(; kwargs...) =
        (stimulus = request, trial_id = trial_id, trial = stimulus.trial, rt = stimulus.rt,
            resp = resp,
            stimulus_fans..., kwargs...)
    vals = map(
        x -> g(;
            chunk = x.slots,
            activation = x.act,
            penalty = get_penalty(x, request),
            get_chunk_fan(x, request, stimulus_fans)...
        ),
        results
    )
    return DataFrame(vals)
end

"""
Computes the person and place fan values
"""
function get_chunk_fan(c, r, fans)
    personFan, placeFan = 0, 0
    if c.person == r.person
        personFan = fans.fanPerson
    end
    if c.place == r.place
        placeFan = fans.fanPlace
    end
    return (chunkPersonFan = personFan, chunkPlaceFan = placeFan)
end

get_chunk_fan(c::Chunk, r, fans) = get_chunk_fan(c.slots, r, fans)

function add_activation_id!(df)
    n = unique([(row.activation) for row in eachrow(df)])
    df.activation_id = [findfirst(x -> (row.activation) == x, n)
                        for row in eachrow(df)]
    return nothing
end

function add_problem_id!(df)
    n = unique([(row.trial, row.fanPerson, row.fanPlace) for row in eachrow(df)])
    df.problem_id = [
        findfirst(x -> (row.trial, row.fanPerson, row.fanPlace) == x, n)
        for row in eachrow(df)
    ]
    return nothing
end

function add_production_id!(df)
    n = unique([(row.penalty) for row in eachrow(df)])
    df.production_id = [findfirst(x -> (row.penalty) == x, n)
                        for row in eachrow(df)]
    return nothing
end

function get_response_data(df)
    g = groupby(df, [:trial_id, :problem_id])
    f(x) = x[1]
    return combine(g, :rt => f => :rt, :resp => f => :resp)
end

# add response log prob id
function get_stimulus_info(df)
    g = groupby(df, [:trial_id, :problem_id, :activation_id, :production_id])
    temp = combine(g, :activation_id => length => :count)
    max_id = maximum(temp.activation_id)
    g = groupby(temp, :problem_id)
    thresholds = combine(g, :activation_id => (x -> max_id + 1) => :activation_id,
        :production_id => (x -> 4) => :production_id,
        :count => (x -> 1) => :counts)
    g = groupby(temp, [:problem_id, :activation_id, :production_id])
    f2(x) = Int(mean(x))
    stimulus_info = combine(g, :count => f2 => :counts)
    append!(stimulus_info, thresholds)
    sort!(stimulus_info, [:problem_id, :activation_id, :production_id])
    return stimulus_info
end

function get_activation_info(df)
    f(x) = x[1]
    g = groupby(df, :activation_id)
    temp = combine(g, :penalty => f => :penalty,
        :chunkPersonFan => f => :chunkPersonFan,
        :chunkPlaceFan => f => :chunkPlaceFan)
    max_id = maximum(temp.activation_id)
    push!(
        temp,
        (activation_id = max_id + 1, penalty = 0, chunkPersonFan = 0, chunkPlaceFan = 0)
    )
    sort!(temp, :activation_id)
    return temp
end

function get_production_info(df)
    g = groupby(df, :production_id)
    temp = combine(g, :penalty => (x -> x[1]) => :penalty)
    max_id = maximum(temp.production_id)
    # case for retrieval failure.
    push!(temp, (production_id = max_id + 1, penalty = -100))
    sort!(temp, :production_id)
    return temp
end
