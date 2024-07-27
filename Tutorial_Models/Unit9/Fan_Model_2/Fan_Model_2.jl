using Parameters, StatsBase, Parameters
import Distributions: logpdf, loglikelihood

"""
Distribution constructor for the full fan model.

* `δ`: mismatch penalty parameter
* `γ`: maximum association parameter
* `parms`: contains all fixed parameters
* `slots`: slot-value pairs used to populate ACT-R's declarative memory
"""
struct Fan{T1, T2, T3, T4} <: ContinuousUnivariateDistribution
    δ::T1
    γ::T2
    parms::T3
    slots::T4
end

#Keyword constructor
Fan(; δ, γ, parms, slots) = Fan(δ, γ, parms, slots)

function logpdf(d::Fan, data::Array{<:NamedTuple, 1})
    LL = computeLL(d.parms, d.slots, data; δ = d.δ, γ = d.γ)
    return LL
end

loglikelihood(d::Fan, data::Array{<:NamedTuple, 1}) = logpdf(d, data)

"""
Simulates multiple blocks of trials for the fan experiment.

* `δ`: mismatch penalty parameter
* `γ`: maximum association parameter
* `parms`: contains all fixed parameters
* `slots`: slot-value pairs used to populate ACT-R's declarative memory
* `stimuli`: vector consisting of slot-value pairs and trial type for the experimental stimuli
"""
function simulate(stimuli, slots, parms, n_blocks; δ, γ)
    #Creates an array of chunks for declarative memory
    chunks = [Chunk(; person = pe, place = pl) for (pe, pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(; memory = chunks)
    #Initialize imaginal buffer
    imaginal = Imaginal(buffer = chunks[1])
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(; declarative = memory, imaginal = imaginal, parms..., δ, γ)
    data = Array{Array{<:NamedTuple, 1}, 1}(undef, n_blocks)
    for b = 1:n_blocks
        data[b] = simulate_block(actr, stimuli, slots)
    end
    return vcat(data...)
end

function simulate_block(actr, stimuli, slots)
    @unpack declarative, imaginal = actr
    #Extract ter parameter for encoding and motor response
    ter = get_parm(actr, :ter)
    resp = :_
    data = Array{NamedTuple, 1}(undef, length(stimuli))
    i = 0
    #Counts the fan for each person-place pair
    fanCount = map(x -> count_fan(x), slots)
    for (trial, person, place) in stimuli
        i += 1
        #Encode stimulus into imaginal buffer
        imaginal.buffer[1] = Chunk(; person, place)
        #Retrieve chunk given person-place retrieval request
        chunk = retrieve(actr; person, place)
        #Compute the retrieval time of the retrieved chunk and add ter
        rt = compute_RT(actr, chunk) + ter
        if isempty(chunk) || !match(chunk[1]; person, place)
            resp = :no
        else
            resp = :yes
        end
        #Get the fan for the person and place
        fan = get_fan(fanCount, person, place)
        #Record all of the simulation output for the ith trial
        data[i] =
            (trial = trial, person = person, place = place, fan..., rt = rt, resp = resp)
    end
    return data
end

"""
Computes the log likelihood of the data

* `δ`: mismatch penalty parameter
* `γ`: maximum association parameter
* `parms`: contains all fixed parameters
* `slots`: slot-value pairs used to populate ACT-R's declarative memory
* `stimuli`: vector consisting of slot-value pairs and trial type for the experimental stimuli
"""
function computeLL(parms, slots, data; δ, γ)
    act = zero(γ)
    #Creates an array of chunks for declarative memory
    chunks = [Chunk(person = pe, place = pl, act = act) for (pe, pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(; memory = chunks)
    #Initialize imaginal module
    imaginal = Imaginal(buffer = chunks[1])
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(; declarative = memory, imaginal, δ, γ, parms...)
    #Don't add noise to activation values
    actr.parms.noise = false
    #Initializes the log likelihood
    LL = zero(γ)
    #Iterate over each trial in data and compute the Loglikelihood based on the response yes, no and
    for v in data
        #Add stimulus person-pair to imaginal buffer
        imaginal.buffer[1] = Chunk(act = act, person = v.person, place = v.place)
        if v.resp == :yes
            LL += loglike_yes(actr, v.rt; person = v.person, place = v.place)
        else
            LL += loglike_no(actr, v.rt; person = v.person, place = v.place)
        end
    end
    return LL
end

"""
Computes the simple likelihood of a "yes" response

* `actr`: actr object
* `rt`: observed response time
* `request`: NamedTuple containing retrieval request (either person or place)
"""
function loglike_yes(actr, rt; request...)
    #Extract required parameters
    @unpack s, τ, ter = actr.parms
    #Subset of chunks that match retrieval request
    chunks = actr.declarative.memory
    #Find index corresponding to "yes" response, which is the stimulus
    choice = find_index(chunks; request...)
    #Compute the activation for each of the matching chunks
    compute_activation!(actr; request...)
    #Collect activation values into a vector
    μ = map(x -> x.act, chunks)
    #Add threshold as the last response
    push!(μ, τ)
    #Map the s parameter to the standard deviation for
    #comparability to Lisp ACTR models.
    σ = s * pi / sqrt(3)
    #Create a distribution object for the LogNormal Race model
    dist = LNR(; μ = -μ, σ, ϕ = ter)
    #Compute likelihood of choice and rt given the parameters.
    return logpdf(dist, choice, rt)
end

"""
Computes the simple likelihood of a "no" response. This function
marginalizes over all of the possible chunks that could be resulted in "no".

* `actr`: actr object
* `rt`: observed response time
* `request`: NamedTuple containing retrieval request (either person or place)
"""
function loglike_no(actr, rt; request...)
    #Extract required parameters
    @unpack s, τ, ter, δ = actr.parms
    #Subset of chunks that match retrieval request
    chunks = actr.declarative.memory
    #Compute the activation for all chunks
    compute_activation!(actr; request...)
    #Collect activation values into a vector
    μ = map(x -> x.act, chunks)
    #Add threshold as the last response
    push!(μ, τ)
    #Map the s parameter to the standard deviation for
    #comparability to Lisp ACTR models.
    σ = s * pi / sqrt(3)
    #Create a distribution object for the LogNormal Race model
    dist = LNR(; μ = -μ, σ, ϕ = ter)
    #Index of the chunk that represents the stimulus
    idx = find_index(chunks; request...)
    #Initialize likelihood
    LLs = Array{typeof(δ), 1}()
    N = length(chunks) + 1
    #Marginalize over all of the possible chunks that could have lead to the
    #observed response
    for i = 1:N
        #Exclude the chunk representing the stimulus because the response was "no"
        if i != idx
            push!(LLs, logpdf(dist, i, rt))
        end
    end
    return logsumexp(LLs)
end
