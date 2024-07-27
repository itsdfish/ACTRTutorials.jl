### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ bee213aa-db64-4ae2-b65b-8ecbde45a0da
begin
    using CmdStan,
        Distributions, Random, StatsPlots, ACTRModels, CSV, DataFrames, MCMCChains
    using PlutoUI
    TableOfContents()
end

# ╔═╡ c08fffe0-ecf8-11ec-05eb-15b906fa1db0
md"""
# Introduction

In this tutorial, we use the Stan language to estimate the parameters of the fan effect model. Much of the material below is reproduced from the pure Julia implementation in Unit 9.

## Task

The classic fan experiment was a paired associate recognition task in which subjects learned a series of statements regarding people and places, such as "The lawyer is in the store" (Anderson, 1974). During the test phase, subjects attempted discriminate between target statements, which were in the study list, and foil statements, which were novel statements formed by mismatching people and places used in the target statements (e.g. "The lawyer is in the park"). A key manipulation in this experiment is termed the "fan", which is the number of statements in which a particular person or place appeared. The primary finding, termed the fan effect, was that increasing the fan lead to an increase in reaction time.


| Target    |        | Foil    |       |
|-----------|--------|---------|-------|
| Person    | Place  | Person  | Place |
|-----------|--------|---------|-------|
| lawyer    | store  | fireman | store |
| captain   | cave   | captain | store |
| hippie    | church | giant   | store |
| debutante | bank   | fireman | bank  |
| earl      | castle | captain | bank  |
| hippie    | bank   | giant   | bank  |
| fireman   | park   | lawyer  | park  |
| captain   | park   | earl    | park  |
| hippie    | park   | giant   | park  |


In our simulated fan experiment, we will use a block of nine target and nine foil statements found in the standard ACT-R tutorial. All of the person-place pairs on which the statements are based are listed in table above. On each trial, a person-place pair is presented and the model decides whether the pair was in the study list, responding either "Yes" or  "No" by pressing a key associated with the given response. We simulated the block of trials five times, yielding a total of 90 trials.   

# Fan Model 2


## Declarative memory

Declarative memory $M$ consists 13 chunks representing the person-place relationships. Let $Q= \rm \{person,place\}$ be the set of slots that define the chunk. Note that nine of the chunks correspond to the stimuli in the Target column of the table above.  

### Encoding

On each trial, a person-place pair is presented on a screen. The model locates and encodes each stimulus into a chunk that is stored in the imaginal buffer. Formally, we define the encoded chunk as:

$\begin{align}
\mathbf{c_{s,\rm \mathbf{imaginal}}} = \{ (\rm person, v_{s,1}), (place, v_{s,2})\}
\end{align}$

where $v_{s,1}$ and $v_{s,2}$ are the values associated with the person and place slots of the chunk in the imaginal buffer. 

### Retrieval Request

Un like Fan Model 1, the present model includes both the person and place slots in the retrievial request. The retrieval request is formed from the stimulus encoded as a chunk in the imaginal buffer, and is defined as: 

$\begin{align}
\mathbf{r} = \{(\rm person,c_{s, imaginal}(\rm person),(place,c_{s, imaginal}( place) \} 
\end{align}$


### Activation

In the fan model, memory activation for chunk $m$ is defined as: 

$\begin{equation}
a_m = {\rm blc} + S_m + \epsilon_m
\end{equation}$

where blc is the base-level constant parameter, $S_m$ is the spreading activation component, and $\epsilon \sim \rm normal(0, \sigma)$ is a noise term.
"""

# ╔═╡ 831d4f96-65f1-4360-b218-912fd0ec3dce
md"""
Utilities
"""

# ╔═╡ cf104abb-1d6b-4974-964d-85040a0415c2
begin
    people = [:hippie, :hippie, :hippie, :captain, :captain, :debutante, :fireman, :giant,
        :giant, :giant, :earl, :earl, :lawyer]

    places = [:park, :church, :bank, :park, :cave, :bank, :park, :beach, :castle, :dungeon,
        :castle, :forest, :store]

    slots = (people = people, places = places)

    trial = [:target, :target, :target, :target, :target,
        :target, :target, :target, :target, :foil, :foil, :foil, :foil,
        :foil, :foil, :foil, :foil, :foil]
    pep = [:lawyer, :captain, :hippie, :debutante, :earl, :hippie,
        :fireman, :captain, :hippie, :fireman, :captain, :giant,
        :fireman, :captain, :giant, :lawyer, :earl, :giant]
    pla = [:store, :cave, :church, :bank, :castle, :bank, :park,
        :park, :park, :store, :store, :store,
        :bank, :bank, :bank, :park, :park, :park]

    stimuli = [(trial = t, person = p, place = pl) for (t, p, pl) in zip(trial, pep, pla)]

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
    function parse_data_stan(data, uvals)
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
        stimuli_values = [person place]
        stimuli_slots = [fill(1.0, Nrows) fill(2.0, Nrows)]
        return rts, resp, stimuli_slots, stimuli_values
    end

    """
    Transforms symbols into integer ids for person-place memory values
    """
    function stan_memory_values(allVals, uvals)
        memory_values = fill(0.0, size(allVals))
        for (i, u) in enumerate(uvals)
            idx = allVals .== u
            memory_values[idx] .= i
        end
        return memory_values
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
        fanCount = map(x -> count_fan(x), slots)
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
        g(; kwargs...) = (stimulus = request, trial_id = trial_id, trial = stimulus.trial,
            rt = stimulus.rt, resp = resp,
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
            (
                activation_id = max_id + 1,
                penalty = 0,
                chunkPersonFan = 0,
                chunkPlaceFan = 0
            )
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
end

# ╔═╡ 07eeccba-9e2d-4b9a-9f00-b38383cb8930
md"""
#### Spreading Activation

Spreading activation term for chunk $m$ is defined as:

$\begin{align}
S_m = \sum_{b \in B} \sum_{k\in Q_b} W_{b,k}S_{m,k}
\end{align}$


where $b$ is a buffer in buffer set $B$, $k$ is a slot of the chunk in buffer $b$ and $Q_b$ is the set of all slots in the chunk in buffer $b$, $W_{b,k}$ is the association weight for slot value $k$ of the chunk in buffer $b$, which is defined as $\frac{1}{|Q_b|}$ by default, and  $S_{m,k}$ is the strength of association for slot value $k$ of chunk $m$. In general, activation can spread from any number of buffers $B$ containing a chunk. In the present model, the set of buffers is defined as $B=\{\rm imaginal\}$ because activation spreads only from the imaginal buffer. Strength of association is defined as:

$\begin{align}
S_{m,k} = \gamma + \log(\textrm{fan}_{m,k})
\end{align}$

where $\gamma$ is the maximum association value. The fan term is a ratio of the occurrences of slot value $c_{b}(k)$ in chunk $m$ over the number of occurrences of $c_{b}(k)$ across all chunks in declarative memory plus one,

$\begin{align}
\textrm{fan}_{m,k} = \frac{C_{b,m,k}}{1+C_{b,k}}
\end{align}$


The term $C_{b,m,k}$ represents the number of occurrences of slot value $c_{b}(k)$ in chunk $m$: 

$\begin{align}
C_{b,m,k} = \sum_{q \in Q_m} I(c_m(q),c_{b}(k))
\end{align}$

where $j$ is a slot value index for chunk $m$, and the indicator function is defined as:

$$I(x,y) =
  \begin{cases}
    1      & x = y\\
    0  & x \neq y
  \end{cases}$$

The term $C_{b,k}$ refers to the number of occurrences of slot value slot value $c_{b}(k)$ across all chunks in declarative memory:

$\begin{align}
C_{b,k} = \sum_{i \in M} C_{b,i,k}
\end{align}$



## Response Mapping 

In Fan Model 1, there is a one-to-one response mapping between the retrieved chunk and the "yes" response and a many-to-one response mapping between the retrieved chunk and the "no" response. The production rule for a yes response $\mathbf{p} = \rm \{(person,c_{s,imaginal}(person), (place,c_{s,imaginal}(place)\}$ is triggered only when the retrieved chunk matches all conditions. All other cases map to a "no" response. More formally, we define the following response mappings for "yes" and "no" responses:

$\begin{align}
H_{\rm yes} &= \{\mathbf{c}_m \in M: \forall q \in Q c_m(q) = p(q) \}\\
H_{\rm no} &= \{\mathbf{c}_m \in M: \exists q \in Q \textrm{ s.t. } c_m(q) = p(q) \} \cup \mathbf{c_{m^{\prime}}} = \{M \setminus H_{\rm yes} \} \cup \mathbf{c_{m^{\prime}}}\\
\end{align}$

On target trials, set $H_{\rm yes}$ only the chunk whose slots match the stimulus. Otherwise, $H_{\rm yes} =\emptyset$. $H_{\rm no}$ is the complimentary set containing all elements in $M$ that are not in $H_{\rm yes}$ plus the retrieval failure $\mathbf{c_{m^{\prime}}}$
"""

# ╔═╡ 63a1af39-3180-4db9-b223-b4942666b409
md"""
## Conflict resolution and perceptual-motor time

As in earlier tutorials, we treat the sum of conflict resolution and perceptual-motor processing time as a constant. Using default ACT-R parameters, this constant sums to:

$\begin{align}
 t_\textrm{er} = .845 \rm seconds
\end{align}$

See the first figure above for details.

## Likelihood Function 

The likelihood function for Fan Model 1 has two functions: one for "yes" responses, and one for "no" responses. The likelihood of responding "yes" in rt seconds is:
 
$\begin{align}
f_{\textrm{yes}}(\textrm{rt}) 
= \sum_{\mathbf{c_{m}} \in H_{\rm yes}}
g(\textrm{rt}-\textrm{t}_{\textrm{er}}|-\mu_m,\sigma)
\prod_{\mathbf{c_{z}} \in \left\{M\cup \mathbf{c_{m^{\prime}}}  \right\}\setminus \mathbf{c_{m}} }\left[1-G(\textrm{rt}-\textrm{t}_{\textrm{er}}|-\mu_z,\sigma)\right].
\end{align}$


The corresponding likelihood for responding "no" in rt seconds is given by:
$\begin{align}
f_{\textrm{no}}(\textrm{rt}) 
= \sum_{\mathbf{c_{m}} \in H_{\rm no}}
g(\textrm{rt}-\textrm{t}_{\textrm{er}}|-\mu_m,\sigma)
\prod_{\mathbf{c_{z}}  \in \left\{M\cup \mathbf{c_{m^{\prime}}}  \right\}\setminus \mathbf{c_{m}} }\left[1-G(\textrm{rt}-\textrm{t}_{\textrm{er}}|-\mu_z,\sigma)\right].
\end{align}$
 
In both functions, the outer summation marginalizes over the set of chunks $H_{\rm yes}$ or $H_{\rm no}$.

The following summarizes the assumption of the model:

1. The retrieval request is based on both the person and place slots of the encoded stimulus
2. The fan effect is due to spreading activation
3. Each response is independent of others
4. Retrieval times follow a Lognormal Normal Race process 
"""

# ╔═╡ ea8b58e1-d137-4c13-b5ae-ac9d7ca35c09
md"""
# Generate Data

Two functions are used to simulate data from Fan Model 1: `simulate` and `simulate_block`. The creates the required chunks for declarative memory, adds them to the ACT-R object, and calls the `simulate_block` function for each repetition. `simulate` function accepts parms the following arguments

- stimuli: an array of `NamedTuples` containing stimulus information
- slots: a `NamedTuple` of person and place slot values
- parms: a `NamedTuple` of fixed parameters
- n_blocks: the number of simulated blocks of trials
-  $\gamma$: the maximum association parameter 

The function `simulate_block` accepts the following argument:

- actr: an ACT-R object containing declarative memory, the imaginal module and parameters
- stimuli: an array of `NamedTuples` containing stimulus information
- slots: a `NamedTuple` of person and place slot values

For each stimulus, the function `simulate_block` performs the following operations:

1. Encode simulus into imaginal buffer
2. Randomly choose between retrieving a chunk using the person or place slot as retrieval request
3. Compute the reaction time of the retrieved chunk
4. Record a "yes" response if the slots of the retrieved chunk match the chunk in the imaginal buffer. Otherwise, record a "no" response.
5. Add the reaction time, response and fan values to a `NamedTuple`.
"""

# ╔═╡ b3808ebd-6969-4830-950e-2ec4dde2df2a
begin
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
            data[i] = (
                trial = trial,
                person = person,
                place = place,
                fan...,
                rt = rt,
                resp = resp
            )
        end
        return data
    end
end

# ╔═╡ 2b4092e2-005c-4c38-9dc7-540d54dbaff2
md"""
In the code block below, we will generate 5 blocks of simulated data, producing a total of 90 trials. 
"""

# ╔═╡ 07bc0184-b1c2-4297-b60c-06451038c1f0
begin
    seed = 684478
    Random.seed!(seed)
    # true value for mismatch penalty parameter 
    δ = 0.5
    # true value for maximum association strength
    γ = 1.6
    n_reps = 5
    # fixed parameters used in the model
    parms = (blc = 0.3, τ = -0.5, noise = true, sa = true, mmp = true, s = 0.2, ter = 0.845)
    # Generates data for Nblocks. Slots contains the slot-value pairs to populate memory
    #stimuli contains the target and foil trials.
    temp = simulate(stimuli, slots, parms, n_reps; γ, δ)
    #Forces the data into a concrete type for improved performance
    data = vcat(temp...)
end

# ╔═╡ e0485b53-47ab-488a-b5e4-d0df84f7295a
md"""
Next, the data must be reformatted for Stan
"""

# ╔═╡ 717e576c-6900-4cd6-b8b2-fc8eb58f4b5d
begin
    allVals = [people places]
    uvals = unique(allVals)
    memory_values = stan_memory_values(allVals, uvals)
    memory_slots = [fill(1.0, length(places)) fill(2.0, length(places))]
    rts, resp, stimuli_slots, stimuli_values = parse_data_stan(data, uvals)
end

# ╔═╡ 787aa115-592b-42e9-b29a-5634686d30d2
md"""
All of the variables declared in the data block are collected in a `Dictionary` called `stan_input`. `stan_input` includes:

- mp: use mismatch penalty if 1
- bll: use base level learning if 1
- sa: use spreading activation if 1
- s: logistic scale parameter for activation noise
- tau: retrieval threshold
- blc: base level constant
- resp: vector of responses
- rts: vector of response times
- n_obs: number of observations
- memory_slots: matrix of memory slots where rows correspond to chunks and columns correspond to slots
- memory_values: matrix of memory values where rows correspond to chunks and columns correspond to values
- stimulus_slots: matrix of stimulus slots where rows correspond to chunks and columns correspond to slots
- stimuli_values: matrix of stimulus values where rows correspond to chunks and columns correspond to values
- n_slots: the number of slot-value pairs
- n_chunks: the number of chunks in declarative memory
"""

# ╔═╡ dce50f6a-3dce-4009-8f20-20962c4727f8
stan_input = Dict("mp" => 1, "bll" => 0, "sa" => 1, "ter" => parms.ter, "s" => parms.s,
    "tau" => parms.τ, "blc" => parms.blc,
    "resp" => resp, "rts" => rts, "n_obs" => length(data),
    "memory_slots" => memory_slots, "memory_values" => memory_values,
    "n_slots" => 2, "stimuli_slots" => stimuli_slots,
    "stimuli_values" => stimuli_values, "n_chunks" => length(slots.people))

# ╔═╡ cae7c912-cf39-4433-8f05-c23e7f855f62
md"""
## Define Model

The model and prior distributions are summarized as follows:

$\delta \sim \rm normal(.5.,.25)$

$\gamma \sim \rm normal(1.6, .8)$

$\mathbf{\mu} = [\mu_1, \mu_2,\dots, \mu_{n}, \tau]$
 

In computer code, the model is specified as follows:

### Data Block

As shown below, the data block contains variables that match the entries in the `stan_input` dictionary. Three types of variables are included in the data block: variables for the data, variables representing chunks in declarative memory, and fixed parameters and settings. 

### Parameter Block 

Three parameters are declared in the parameter block: `delta` and `gamma`.

### Model Block

Five local variables are declared at the top of the model block. With expection of `sigma`, the variables can be disregarded because they are not used in the current model. However, they are required inputs into the various ACT-R functions, and thus must be declared. 

The second group of statements is for the prior distributions of `delta` and `gamma`. The log likelihood calculations are performed in the for loop at the bottom of the model block. Inside the for loop, the custom function `computeLL` is called, which returns the log likelihood of a given observation and adds it to the `target` variable, which accumulates the log likelihoods. 

### Function Block 

The function block consists of many functions for ACT-R. We will focus on the description of three functions for the log likelihood function, located towards the bottom of the function block.

#### LLYes

The function `LLYes` computes the log likelihood of a yes response. Yes responses have a one-to-one mapping between the retrieved chunk and response.

#### LLNo

The function `LLNo` computes the log likelihood of a no response. Because a no response has a many-to-one response mapping, the log likelihood is marginalized across the all chunks except the chunk corresponding to a yes response.

#### computeLL


The first group of commands returns the results of the retrieval request. In the present case, the result of the retrieval request is the slot matrix and value matrix for all chunks because partial matching is enabled (i.e. `mp=1`). The second group of commands computes the activation values associated with the result of the retrieval request. Finally, the last group of commands computes the log likelihood of the response. If the response is a retrieval failure (e.g. choice = -100), a seperate computation is performed. For all other cases, the marginal probability is computed. For example, if the response was `choice`= 3, prob is the sum of the probability of retrieving any chunk whose sum slot is 3 (i.e., 1+2, 2+1, 0+3, etc.). The log of prob is multiplied by the number of duplicates, which improves efficency by computing the activations only once.  
"""

# ╔═╡ aaa16e8f-81bc-4bb4-b7ef-149a69957111
md"""
Please reveal the cell below to see the Stan model.
"""

# ╔═╡ c7714787-bbf6-4a63-b984-30c0f5d66795
Model = "functions {

real exact_base_level(real d, real[] lags){
  real act;
  act = 0.0;
  for (i in 1:size(lags)){
    act += pow(lags[i], -d);
  }
  return log(act);
}

real base_level(real d, int k, real L, int N, real[] lags){
  real exact;
  real approx;
  real tk;
  real x1;
  real x2;
  exact = exact_base_level(d, lags);
  approx = 0.0;
  if (N > k) {
    tk = lags[k];
    x1 = (N-k)*(pow(L,1-d)-tk^(1-d));
    x2 = (1-d)*(L-tk);
    approx = x1/x2;
  }
  return log(exp(exact) + approx);
}

int isIn(real request_slot, row_vector chunk_slots){
  for(i in 1:num_elements(chunk_slots)){
    if(request_slot == chunk_slots[i]){
        return 1;
    }
  }
  return 0;
}

int get_slot_index(real request_slot, row_vector chunk_slots){
  for(i in 1:num_elements(chunk_slots)){
    if(request_slot == chunk_slots[i]){
        return i;
    }
  }
  return 0;
}

real mismatch(real delta, row_vector chunk_slots, row_vector chunk_values, row_vector request_slots,
    row_vector request_values){
  real p;
  int slot_index;
  p = 0;
  for(i in 1:num_elements(request_slots)){
    if(isIn(request_slots[i], chunk_slots) == 0){
      p -= delta;
      continue;
    }
    else{
      slot_index = get_slot_index(request_slots[i], chunk_slots);
      if(request_values[i] != chunk_values[slot_index]){
          p -= delta;
      }
    }
  }
  return p;
}

//Penalty when matches
real match_penalty(real delta, row_vector chunk_slots, row_vector chunk_values, row_vector request_slots,
    row_vector request_values){
  real p;
  int slot_index;
  p = 0;
  for(i in 1:num_elements(request_slots)){
    if(isIn(request_slots[i], chunk_slots) == 0){
      p -= delta;
      continue;
    }
    else{
      slot_index = get_slot_index(request_slots[i], chunk_slots);
      if(request_values[i] == chunk_values[slot_index]){
          p -= delta;
      }
    }
  }
  return p;
}

int count_values(row_vector chunk_values, real value){
  int cnt;
  cnt = 0;
  for(i in 1:cols(chunk_values)){
    if (chunk_values[i] == value){
      cnt +=1;
    }
  }
  return cnt;
}

real compute_denom(matrix memory_values, real imaginal_value){
  int N;
  int cnt;
  cnt = 0;
  N = rows(memory_values);
  for(c in 1:N){
    cnt += count_values(memory_values[c,:], imaginal_value);
  }
  return cnt;
}

real[] denom_spreading_activation(row_vector imaginal_values, matrix memory_values, int Nslots){
  real denoms[Nslots];
  for(i in 1:Nslots){
    denoms[i] = compute_denom(memory_values, imaginal_values[i]);
  }
  return denoms;
}

real compute_weights(row_vector chunk_values){
  return 1.0 / num_elements(chunk_values);
}

real spreading_activation(row_vector chunk_values, row_vector imaginal_values, real[] denoms,
  real gamma){
    real w;
    real r;
    real sa;
    real fan;
    real num;
    r = 0;
    sa = 0;
    w = compute_weights(chunk_values);
    for(i in 1:num_elements(imaginal_values)){
      num = count_values(chunk_values, imaginal_values[i]);
      fan = num/(denoms[i] + 1);
      if(fan == 0){
        r = 0;
      }else{
        r = gamma + log(fan);
      }
      sa += w * r;
    }
    return sa;
  }

real compute_activation(real blc, real d, real delta, real gamma, int k, real L, int N, real[] lags,
  int sa, int mp, int bll, row_vector chunk_slots, row_vector chunk_values, row_vector request_slots, row_vector request_values,
  row_vector imaginal_slots, row_vector imaginal_values, real[] denoms){
    real act;
    act = blc;
    if(bll == 1){
      act += base_level(d, k, L, N, lags);
    }
    if(mp == 1){
      act += mismatch(delta, chunk_slots, chunk_values, request_slots, request_values);
    }
    if(sa ==1){
      act += spreading_activation(chunk_values, imaginal_values, denoms, gamma);
    }
    return act;
}

real LNR_LL(row_vector mus, real sigma, real ter, real v, int c){
  real LL;
  LL = 0;
  for(i in 1:num_elements(mus)){
    if(i == c){
      LL += lognormal_lpdf(v - ter|mus[i], sigma);
  }else{
    LL += log(1  -lognormal_cdf(v - ter, mus[i], sigma));
  }
}
  return LL;
}

real compute_retrieval_prob(row_vector activations, real s, real tau, int n_chunks, int idx){
  real sigma;
  real prob;
  real v[n_chunks+1];
  sigma = s * sqrt(2);
  for(i in 1:n_chunks){
    v[i] = exp(activations[i] / sigma);
  }
  v[n_chunks+1] = exp(tau / sigma);
  prob = v[idx] / sum(v);
  return prob;
}

int all_values_match(row_vector request_slots, row_vector chuck_slots,
  row_vector request_values, row_vector chunk_values){
    int flag;
    for(s in 1:num_elements(request_slots)){
      flag = 0;
      for(m in 1:num_elements(chuck_slots)){
        if(request_slots[s] == chuck_slots[m] && request_values[s] == chunk_values[m]){
          flag = 1;
          break;
        }
      }
      if(flag == 0){
        return 0;
      }
    }
    return 1;
}

real marginal_retrieval_prob(row_vector activations, real s, real tau, matrix result_slots, matrix result_values,
  row_vector response_slots, row_vector response_values, int n_results) {
    int flag;
    real prob;
    int n_chunks;
    prob = 0;
    for(r in 1:n_results){
      flag = all_values_match(response_slots, result_slots[r,:], response_values, result_values[r,:]);
      if(flag == 1){
        prob += compute_retrieval_prob(activations, s, tau, n_results, r);
      }
    }
    return prob;
}

int get_chunk_index(matrix result_slots, matrix result_values,
  row_vector request_slots, row_vector request_values){
    int flag;
    for(r in 1:rows(result_slots)){
      flag = all_values_match(request_slots, result_slots[r,:], request_values, result_values[r,:]);
      if(flag == 1){
        return r;
      }
    }
    return -100;
}

matrix[,] request_chunks(row_vector request_slots, row_vector request_values,
  matrix memory_slots, matrix memory_values, int mp){
    matrix[rows(memory_slots),cols(memory_slots)] match_slots;
    matrix[rows(memory_slots),cols(memory_slots)] match_values;
    matrix[rows(memory_slots),cols(memory_slots)] match[1,3];
    int n_rows;
    int chunk_count;
    int flag;
    //if mismatch penalty is active, return the memory matrix
    //No subsetting required
    if(mp == 1){
      match[1,1] = memory_slots;
      match[1,2] = memory_values;
      match[1,3][1,1] = rows(memory_slots);
      return match;
    }

    chunk_count = 0;
    n_rows = rows(memory_values);
    for(r in 1:n_rows){
      flag = all_values_match(request_slots, memory_slots[r,:], request_values, memory_values[r,:]);
      if(flag == 1){
        chunk_count += 1;
        match_slots[chunk_count,:] = memory_slots[r,:];
        match_values[chunk_count,:] = memory_values[r,:];
      }
    }
    match[1,1] = match_slots;
    match[1,2] = match_values;
    match[1,3][1,1] = chunk_count;
    return match;
}

int to_int(real v){
  int i;
  i = 1;
  while (i < v){
    i+=1;
  }
  return i;
}

row_vector compute_all_activations(real blc, real d, real delta, real gamma, real tau, int k, real L, int N, real[] lags,
  int sa, int mp, int bll, matrix result_slots, matrix result_values, row_vector request_slots,
    row_vector request_values, int n_results, int n_chunks, row_vector imaginal_slots, row_vector imaginal_values, real[] denoms){
    row_vector[n_chunks+1] activations;
    //row_vector[n_results+1] activations;
    for(i in 1:n_results){
      activations[i] = compute_activation(blc, d, delta, gamma, k, L, N, lags, sa, mp, bll, result_slots[i,:],
        result_values[i,:], request_slots, request_values, imaginal_slots, imaginal_values, denoms);
    }
    activations[n_results+1] = tau;
    return activations;
  }


  real LLYes(real rt,real sigma,real ter, row_vector request_slots, row_vector request_values, matrix result_slots,matrix result_values,
      row_vector activations){
        int idx;
        // get index corresponding to retrieved chunk
        idx = get_chunk_index(result_slots, result_values, request_slots, request_values);
        // log normal race log likelihood of response idx at rt
        return LNR_LL(-activations, sigma, ter, rt, idx);
    }

  real LLNo(real rt, real sigma, real ter, row_vector stimulus_slots, row_vector stimulus_values,
      matrix result_slots, matrix result_values, row_vector activations){
      int idx;
      real LL[num_elements(activations)];
      int N;
      int j;
      N = num_elements(activations);
      // index of chunk mapping to yes response. This chunk could not have been retrieved for no response
      idx = get_chunk_index(result_slots, result_values, stimulus_slots, stimulus_values);
      j = 0;
      // compute the log likelihood of retrieving each possible chunk except idx
      for(i in 1:N){
        if(i != idx){
            j += 1;
            // log normal race log likelihood of response i at rt
            LL[j] = LNR_LL(-activations, sigma, ter, rt, i);
        }
      }
      return log_sum_exp(LL[1:j]);
  }

  real computeLL(real d, real blc, real delta, real gamma, int k, real L, int N, real sigma, real tau, real ter, int sa, int mp, int bll,
    int n_chunks, int n_slots, matrix memory_slots, matrix memory_values, row_vector stimulus_slots, row_vector stimulus_values, int resp,
    real rt){
    // real requestSlots[n_slots];
    // real requestValues[n_slots];
    row_vector[n_slots] imaginal_slots;
    row_vector[n_slots] imaginal_values;
    row_vector[n_chunks+1] activations; //n_chunks + threshold
    // slots: request_result[1,1]
    // values: request_result[1,2]
    // number of chunks: request_result[1,3][1,1]
    matrix[n_chunks,2] request_result[1,3];
    matrix[n_chunks,2] result_slots;
    matrix[n_chunks,2] result_values;
    real LLs[2];
    real lags[1];
    real denoms[n_slots];
    int n_results;

    lags[1] = 0;
    imaginal_slots = stimulus_slots;
    imaginal_values = stimulus_values;
    // pre-compute the denominator of fan values
    denoms = denom_spreading_activation(imaginal_values, memory_values, n_slots);

    if(resp==2){
      //Respond Yes

      // Return chunks that match retrieval request (stimulus_slots, stimulus_values). If mp = 1,
      // as in the current case, all chunk slots and values are returned
      request_result = request_chunks(stimulus_slots, stimulus_values, memory_slots, memory_values, mp);
      result_slots = request_result[1,1];
      result_values = request_result[1,2];
      n_results = to_int(request_result[1,3][1,1]);
      // compute activation for all chunks in the request result
      activations = compute_all_activations(blc, d, delta, gamma, tau, k, L, N, lags, sa, mp, bll, result_slots, result_values, stimulus_slots,
          stimulus_values, n_results, n_chunks, imaginal_slots, imaginal_values, denoms);
        // compute the log likelihood of a yes response
      return LLYes(rt, sigma, ter, stimulus_slots, stimulus_values, result_slots, result_values,
        activations[1:(n_results+1)]);
    }else{
      //Respond No

        // Return chunks that match retrieval request (stimulus_slots, stimulus_values). If mp = 1,
      // as in the current case, all chunk slots and values are returned
      request_result = request_chunks(stimulus_slots, stimulus_values, memory_slots, memory_values, mp);
      result_slots = request_result[1,1];
      result_values = request_result[1,2];
      n_results = to_int(request_result[1,3][1,1]);
      // compute activation for all chunks in the request result
      activations = compute_all_activations(blc, d, delta, gamma, tau, k, L, N, lags, sa, mp, bll, result_slots, result_values, stimulus_slots,
          stimulus_values, n_results, n_chunks, imaginal_slots, imaginal_values, denoms);
      // compute the log likelihood of a no response
      return LLNo(rt, sigma, ter, stimulus_slots, stimulus_values, result_slots, result_values,
        activations[1:(n_results+1)]);
    }
  }
}

data {
  int<lower=1> n_obs;
  int<lower=1> n_chunks;
  int<lower=1> n_slots;
  int<lower=0> resp[n_obs];
  real rts[n_obs];
  matrix[n_obs,2] stimuli_slots; //person,place
  matrix[n_obs,2] stimuli_values; //person,place
  // partial matching indicator
  int<lower=0,upper=1> mp;
  // base level learning indicator
  int<lower=0,upper=1> bll;
  // spreading activation indicator
  int<lower=0,upper=1> sa;
  // base level constant
  real blc;
  // retrieval threshold parameter
  real tau;
  // encoding and motor time 
  real ter;
  real s;
  matrix[n_chunks,2] memory_values; //person,place
  matrix[n_chunks,2] memory_slots; //person,place
}

parameters {
  real<lower=0> delta;
  real<lower=0> gamma;
}

model {
    real d;
    int k;
    real L;
    int N;
    real sigma;

    // Note that these variables are required by the functions are are not used in this model
    d = 0;
    k = 1;
    L = 0;
    N = 0;

    // standard deviation for activation noise
    sigma = s*pi()/sqrt(3);

    // prior distribution of mismatch penalty parameter
    delta ~ normal(.5, .25);
    // prior distribution of maximum association parameter
    gamma ~ normal(1.6, .8);

    for(trial in 1:n_obs){
      target += computeLL(d, blc, delta, gamma, k, L, N, sigma, tau, ter, sa, mp, bll, n_chunks, n_slots, memory_slots, memory_values,
         stimuli_slots[trial,:], stimuli_values[trial,:], resp[trial], rts[trial]);
    }
}
";

# ╔═╡ 8c4bacc2-c4ec-4445-aca7-6c3c18f764f6
md"""
## Estimate Parameters

Now that the priors, likelihood, and Turing model have been specified, we can now estimate the parameters. In the following code, we will run four MCMC chains with the NUTS sample for 2,000 iterations and omit the first 1,000 warmup samples.
"""

# ╔═╡ c8c79ae7-cc10-4d91-880d-55c388e61a19
begin
    n_chains = 4
    proj_dir = pwd()
    stanmodel = Stanmodel(
        Sample(save_warmup = false, num_warmup = 1000,
            num_samples = 1000, thin = 1), nchains = n_chains, name = "Fan_Model_2", model = Model,
        printsummary = false, output_format = :mcmcchains, random = CmdStan.Random(seed)
    )

    rc, chain, cnames = stan(stanmodel, stan_input, proj_dir)
    chain = replacenames(chain, Dict("gamma" => "γ", "delta" => "δ"))
end

# ╔═╡ db6571fc-d2ff-4216-b0a6-8b1a88f9183f
md"""
## Results

A summary of the parameter estimates can be found in the output above. The diagnostic plots below for $\gamma$ and $\delta$ indicate that the chains converged and low autocorrelation was acceptably low. The posterior distributions are centered near the data generating parameter values of $\gamma = 1.6$ and $\delta = .5$ indicating good recovery of the parameters. 
"""

# ╔═╡ 70d5e905-c063-4758-83ea-38486963aba3
let
    ch = group(chain, :γ)
    p1 = plot(ch, seriestype = (:traceplot), grid = false)
    p2 = plot(ch, seriestype = (:autocorplot), grid = false)
    p3 = plot(ch, seriestype = (:mixeddensity), grid = false)
    pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
end

# ╔═╡ e89e6689-bbe5-4fe4-8d8e-a9f8f68bacbc
let
    ch = group(chain, :δ)
    p1 = plot(ch, seriestype = (:traceplot), grid = false)
    p2 = plot(ch, seriestype = (:autocorplot), grid = false)
    p3 = plot(ch, seriestype = (:mixeddensity), grid = false)
    pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
end

# ╔═╡ 253b8bf6-67a3-4d44-9a31-406674f0b97a
md"""
### Posterior Predictive Distribution

The code block below plots the posterior predictive distributions for correct rts and incorrect rts. As expected, the density for incorrect rts is lower than correct rts, which reflects the fact that incorrect responses are less probable. In addition, incorrect RTs are about .100 seconds slower than correct RTs.
"""

# ╔═╡ 84c3fee7-302a-430d-992b-e4dd8f21d1b5
let
    sim(p) = simulate(stimuli, slots, parms, n_reps; p...)
    preds = posterior_predictive(x -> sim(x), chain, 1000, summarize)
    df = vcat(preds...)
    fan_effect = filter(x -> x.resp == :yes && x.trial == :target, df)
    df_data = DataFrame(data)
    filter!(x -> x.resp == :yes && x.trial == :target, df_data)
    groups = groupby(df_data, [:fanPlace, :fanPerson])
    data_means = combine(groups, :rt => mean).rt_mean
    title = [string("place: ", i, " ", "person: ", j) for i = 1:3 for j = 1:3]
    title = reshape(title, 1, 9)
    p4 = @df fan_effect histogram(
        :MeanRT,
        group = (:fanPlace, :fanPerson),
        ylabel = "Density",
        grid = false,
        norm = true,
        color = :grey,
        leg = false,
        xticks = [1.0, 1.3, 1.6, 1.9],
        title = title,
        layout = 9,
        xlims = (1.0, 2.0),
        ylims = (0, 8),
        bins = 15
    )
    vline!(p4, data_means', color = :darkred)
end

# ╔═╡ 533cb6fe-b615-4e4e-8153-9a10f2412b46
md"""
# References
Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. Cognitive psychology, 6(4), 451-474.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ACTRModels = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CmdStan = "593b3428-ca2f-500c-ae53-031589ec8ddd"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
ACTRModels = "~0.10.6"
CSV = "~0.10.4"
CmdStan = "~6.6.0"
DataFrames = "~1.3.4"
Distributions = "~0.25.62"
MCMCChains = "~5.3.1"
PlutoUI = "~0.7.39"
StatsPlots = "~0.14.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.ACTRModels]]
deps = ["ConcreteStructs", "Distributions", "Parameters", "Pkg", "PrettyTables", "Random", "Reexport", "SafeTestsets", "SequentialSamplingModels", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "54667a26ef188769599a1113fa10614d68783ba9"
uuid = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
version = "0.10.6"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CmdStan]]
deps = ["CSV", "DataFrames", "DelimitedFiles", "DocStringExtensions", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "6ed0a91233e7865924037b34ba17671ef1707ec5"
uuid = "593b3428-ca2f-500c-ae53-031589ec8ddd"
version = "6.6.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0ec161f87bf4ab164ff96dfacf4be8ffff2375fd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.62"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3a233eeeb2ca45842fe100e0413936834215abf5"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.4+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "57af5939800bce15980bddd2426912c4f83012d8"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8cb9b8fb081afd7728f5de25b9025bff97cb5c7a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "058d08594e91ba1d98dcc3669f9421a76824aa95"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.3"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "74d7fb54c306af241c5f9d4816b735cb4051e125"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.4.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "ec2e30596282d722f018ae784b7f44f3b88065e4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.6"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "3e32c8dbbbe1159a5057c80b8a463369a78dd8d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "0b727ac13565a2b665cc78db579e0093b869034e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.30.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SafeTestsets]]
deps = ["Test"]
git-tree-sha1 = "36ebc5622c82eb9324005cc75e7e2cc51181d181"
uuid = "1bc83da4-3b8d-516f-aca4-4fe02f6d838f"
version = "0.0.1"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.SequentialSamplingModels]]
deps = ["ConcreteStructs", "Distributions", "Interpolations", "KernelDensity", "Parameters", "PrettyTables", "Random"]
git-tree-sha1 = "d43eb5afe2f6be880d3bd79c9f72b964f12e99a5"
uuid = "0e71a2a6-2b30-4447-8742-d083a85e82d1"
version = "0.1.7"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "2bbd9f2e40afd197a1379aef05e0d85dba649951"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.7"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "271a7fea12d319f23d55b785c51f6876aadb9ac0"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "9abba8f8fb8458e9adf07c8a2377a070674a24f1"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.8"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─bee213aa-db64-4ae2-b65b-8ecbde45a0da
# ╟─c08fffe0-ecf8-11ec-05eb-15b906fa1db0
# ╟─831d4f96-65f1-4360-b218-912fd0ec3dce
# ╟─cf104abb-1d6b-4974-964d-85040a0415c2
# ╟─07eeccba-9e2d-4b9a-9f00-b38383cb8930
# ╟─63a1af39-3180-4db9-b223-b4942666b409
# ╟─ea8b58e1-d137-4c13-b5ae-ac9d7ca35c09
# ╠═b3808ebd-6969-4830-950e-2ec4dde2df2a
# ╟─2b4092e2-005c-4c38-9dc7-540d54dbaff2
# ╠═07bc0184-b1c2-4297-b60c-06451038c1f0
# ╟─e0485b53-47ab-488a-b5e4-d0df84f7295a
# ╠═717e576c-6900-4cd6-b8b2-fc8eb58f4b5d
# ╟─787aa115-592b-42e9-b29a-5634686d30d2
# ╠═dce50f6a-3dce-4009-8f20-20962c4727f8
# ╟─cae7c912-cf39-4433-8f05-c23e7f855f62
# ╟─aaa16e8f-81bc-4bb4-b7ef-149a69957111
# ╟─c7714787-bbf6-4a63-b984-30c0f5d66795
# ╟─8c4bacc2-c4ec-4445-aca7-6c3c18f764f6
# ╠═c8c79ae7-cc10-4d91-880d-55c388e61a19
# ╟─db6571fc-d2ff-4216-b0a6-8b1a88f9183f
# ╟─70d5e905-c063-4758-83ea-38486963aba3
# ╟─e89e6689-bbe5-4fe4-8d8e-a9f8f68bacbc
# ╟─253b8bf6-67a3-4d44-9a31-406674f0b97a
# ╟─84c3fee7-302a-430d-992b-e4dd8f21d1b5
# ╟─533cb6fe-b615-4e4e-8153-9a10f2412b46
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
