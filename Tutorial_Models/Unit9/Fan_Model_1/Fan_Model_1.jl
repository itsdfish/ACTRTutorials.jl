using Parameters, StatsBase
import Distributions: logpdf, rand, loglikelihood

"""
Distribution constructor for the baseline fan model.

* γ: maximum association parameter
* parms: contains all fixed parameters
* slots: slot-value pairs used to populate ACT-R's declarative memory
"""
struct Fan{T1,T2,T3} <: ContinuousUnivariateDistribution
  γ::T1
  parms::T2
  slots::T3
end

#Keyword constructor
Fan(;γ,ter,parms,slots) = Fan(γ,ter,parms,slots)

"""
Computes the log likelihood for the baseline fan model.
"""
function logpdf(d::Fan, data::Array{<:NamedTuple,1})
    LL = computeLL(d.parms, d.slots, data; γ=d.γ)
    return LL
end

loglikelihood(d::Fan, data::Array{<:NamedTuple,1}) = logpdf(d, data)

"""
Simulates multiple blocks of trials for the fan experiment.

* γ: maximum association parameter
* parms: contains all fixed parameters
* slots: slot-value pairs used to populate ACT-R's declarative memory
* stimuli: vector consisting of slot-value pairs and trial type for the experimental stimuli
"""
function simulate(stimuli, slots, parms, n_blocks; γ)
    #Creates an array of chunks for declarative memory
    chunks = [Chunk(;person=pe, place=pl) for (pe,pl) in zip(slots...)]
    #Creates a declarative memory object that holds an array of chunks and model parameters
    memory = Declarative(;memory=chunks)
    #Initialize imaginal module
    imaginal = Imaginal(buffer=chunks[1])
    #Creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(;declarative=memory, imaginal, parms..., γ)
    data = Array{Array{>:NamedTuple,1},1}(undef,n_blocks)
    for b in 1:n_blocks
        data[b] = simulate_block(actr, stimuli, slots)
    end
    return vcat(data...)
end

function simulate_block(actr, stimuli, slots)
    @unpack declarative,imaginal = actr
    #Extract ter parameter for encoding and motor response
    ter = get_parm(actr, :ter)
    resp = :_
    data = Array{NamedTuple,1}(undef,length(stimuli))
    i = 0
    chunk = Chunk()
    #Counts the fan for each person-place pair
    fanCount = map(x->count_fan(x), slots)
    for (trial,person,place) in stimuli
        i+=1
        #Encode stimulus into imaginal buffer
        imaginal.buffer[1] = Chunk(;person, place)
        #Randomly select a production rule for person or place retrieval request.
        if rand(Bool)
            chunk = retrieve(actr; person)
        else
            chunk = retrieve(actr; place)
        end
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
        data[i] = (trial=trial, person=person, place=place, fan...,rt=rt, resp=resp)
    end
    return data
end

"""
Computes the log likelihood of the data

* γ: maximum association parameter
* parms: contains all fixed parameters
* slots: slot-value pairs used to populate ACT-R's declarative memory
* stimuli: vector consisting of slot-value pairs and trial type for the experimental stimuli
"""
function computeLL(parms ,slots, data; γ)
    act = zero(γ)
    # creates an array of chunks for declarative memory
    chunks = [Chunk(;person=pe, place=pl, act) for (pe,pl) in zip(slots...)]
    # creates a declarative memory object that holds an array of chunks
    memory = Declarative(;memory=chunks)
    imaginal = Imaginal(buffer=chunks[1])
    # creates an ACTR object that holds declarative memory and other modules as needed
    actr = ACTR(;declarative=memory, imaginal, γ, parms...)
    # don't add noise to activation values
    actr.parms.noise = false
    # initializes the log likelihood
    LL = 0.0
    LLs = Array{typeof(γ), 1}(undef,2)
    # iterate over each trial in data and compute the Loglikelihood based on the response yes, no and
    # rf (retrieval failure)
    # each response is a mixture of person vs. place retrieval request
    for v in data
        #Add stimulus to imaginal buffer
        imaginal.buffer[1] = Chunk(;act, person=v.person, place=v.place)
        if v.resp == :yes
            # log likelihood of retrieval using person slot
            LLs[1] = loglike_yes(actr, v.person, v.place,v.rt; person=v.person)
            # log likelihood of retrieval using place slot
            LLs[2] = loglike_yes(actr, v.person, v.place, v.rt; place=v.place)
            # log likelihood of retrieving with person vs. place slot
            LLs .+= log(.5)
            # compute log likelihood of mixture
            LL += logsumexp(LLs)
        else
            # log likelihood of retrieval using person slot
            LLs[1] = loglike_no(actr, v.person, v.place, v.rt; person=v.person)
            # log likelihood of retrieval using place slot
            LLs[2] = loglike_no(actr, v.person, v.place, v.rt; place=v.place)
            # log likelihood of retrieving with person vs. place slot
            LLs .+= log(.5)
            # compute log likelihood of mixture
            LL += logsumexp(LLs)
        end
    end
    return LL
end

"""
Computes the simple likelihood of a "yes" response

* actr: actr object
* person: person value of the selected response
* place: place value of the selected response
* rt: observed response time
* request: NamedTuple containing retrieval request (either person or place)
"""
function loglike_yes(actr, person, place, rt; request...)
    # extract required parameters
    @unpack s,τ,ter = actr.parms
    # subset of chunks that match retrieval request
    chunks = get_chunks(actr; request...)
    # find index corresponding to "yes" response, which is the stimulus
    choice = find_index(chunks; person, place)
    # compute the activation for each of the matching chunks
    compute_activation!(actr, chunks; request...)
    # collect activation values into a vector
    μ = map(x->x.act, chunks)
    # add threshold as the last response
    push!(μ, τ)
    # map the s parameter to the standard deviation for
    # comparability to Lisp ACTR models.
    σ = s*pi/sqrt(3)
    # create a distribution object for the LogNormal Race model
    dist = LNR(;μ=-μ, σ, ϕ=ter)
    # compute likelihood of choice and rt given the parameters.
    return logpdf(dist, choice, rt)
end

"""
Computes the simple likelihood of a "no" response. This function
marginalizes over all of the possible chunks that could be resulted in "no".

* actr: actr object
* person: person value of the selected response
* place: place value of the selected response
* rt: observed response time
* request: NamedTuple containing retrieval request (either person or place)
"""
function loglike_no(actr, person, place, rt; request...)
    # extract required parameters
    @unpack s,τ,ter,γ = actr.parms
    # subset of chunks that match retrieval request
    chunks = get_chunks(actr; request...)
    # compute the activation for each of the matching chunks
    compute_activation!(actr, chunks; request...)
    # collect activation values into a vector
    μ = map(x->x.act, chunks)
    # add threshold as the last response
    push!(μ, τ)
    # map the s parameter to the standard deviation for
    # comparability to Lisp ACTR models.
    σ = s*pi/sqrt(3)
    dist = LNR(;μ=-μ, σ, ϕ=ter)
    # index of the chunk that represents the stimulus
    idx = find_index(chunks; person=person, place=place)
    # marginalize over all of the possible chunks that could have lead to the
    # observed response
    N = length(chunks) + 1
    LLs = Array{typeof(γ), 1}()
    for i in 1:N
        # exclude the chunk representing the stimulus because the response was "no"
        if i != idx
            push!(LLs, logpdf(dist, i, rt))
        end
    end
    return logsumexp(LLs)
end