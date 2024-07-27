### A Pluto.jl notebook ###
# v0.19.44

using Markdown
using InteractiveUtils

# ╔═╡ 90feb1a2-fc4a-11ec-1fb3-bb5d1d9fd934
begin
    using StatsPlots, Revise, ACTRModels, Random
    using PlutoUI, DataFrames, CommonMark, NamedTupleTools
    using FFTDists, DifferentialEvolutionMCMC, MCMCChains
    using Distributions
    # seed random number generator
    Random.seed!(9851)
    TableOfContents()
end

# ╔═╡ b2505b48-dab3-4051-b008-9bf5bfbe87da
begin
    path_u5_1 = joinpath(pwd(), "../../Unit5/Semantic/Semantic_Model_Notebook.jl")

    nothing
end

# ╔═╡ 5fa6fcc2-0c09-4074-8f15-6e2ac17a354f
Markdown.parse(
    """
    # Introduction

    In this tutorial, we will use fast Fourier transform to develop a version of the [semantic model](./open?path=$path_u5_1) that predicts reaction time distributions. As will be illustrated below, the model predicts multimodal distributions when a variable number of memory retrievals can lead to a response through category chaining. The structure of the present model is identical to the Markov process version presented earlier. Some details of the model will be presented below. Please review the first version of the [model](./open?path=$path_u5_1) if you need to refamiliarize yourself with the details. 
    """
)

# ╔═╡ d64fbbaa-25f4-4aa7-95c6-d58e56e833a3
md"""
## Task

In the task, participants must verify whether one category is a member of another category by responding "yes" or "no". For example, a person might recieve a question such as "Is a bird a canary?" and must respond "yes" or "no" with the appropriate keys. 

# Semantic Model

According to the model, category relationships are represented as a semantic network in which nodes correspond to concepts and directed edges correspond to membership. The model can answer questions about category membership directly through a process called direct verification, or indirectly through a process called category chaining. For example, the model can answer the question "Is a bird an animal?" through direct verification by traversing from the bird node to the animal node using a single memory retrieval. By contrast, the question "is a canary an animal?" is answered indirectly through category chaining, which involves traversing from the canary node to the bird node and then from the bird node to the animal node using successive memory retrievals. 


## Declarative Memory

Declarative memory $M$ consists of 20 chunks with the slots, (i.e., domain), $Q = \{\textrm{object, attribute, value}\}$. The object slot refers to the entity being evaluated (e.g. canary); the attribute slot refers to the particular property on which the object is being evaluated (e.g. category, flies); and the value slot refers to the particular value of the attribute (e.g. bird, true). For example, chunk 20 is $c_{20}(\rm{object}) = \textrm{bird}$, $c_{20}(\textrm{attribute}) =\textrm{category}$, $c_{20}(\textrm{value}) = \textrm{animal}$, which indicates that the object bird belongs to the category animal. 
"""

# ╔═╡ 340b62c9-8c73-4bc0-a12b-5a018e18125d
md"""
### Activation

Memory activation for chunk $m$ is defined as 

$\begin{equation}
a_m = \textrm{blc} + \rho_m + \epsilon_m
\end{equation}$

where $\textrm{blc}$ is the base-level constant and $\rho_m$ is partial matching. The base-level constant simply scales memory activation up or down. Partial matching allows chunks to be retrieved as a function of dissimilarity to the retrieval request values, $r$. For simplicity, partial matching is defined as a weighted count of mismatching slot-values:

$\begin{equation}
\rho_m = -\delta  \sum_{q \in Q_r} I\left(r(q),  c_{m}(q)\right).
\end{equation}$

where $Q_r = \{\textrm{object},\textrm{attribute}\}$ is the set of slots in the retrieval request, the mismatch penalty parameter $\delta$ controls the steepness of the dissimilarity gradient, and $I$ is an indicator function: 

$$I(x,y) =
  \begin{cases}
    1      & x \neq y\\
    0  & x = y
  \end{cases}$$
"""

# ╔═╡ 9e14a221-5423-4e83-8978-2abac5ab9ae2
md"""
Thus, chunks that are more similar to the retrieval request are more likely to be retrieved.

## Perceptual-Motor Time

In the semantic model, a series of production rules are used to encode the words (e.g. canary, bird), initialize memory retrievals, and execute a response. As with previous models, a normal approximation will be used to represent the sum of these independent and serial processes. The number of perceptual motor processes depends on the number of category chains required to give a response. Thus, the the general expression of $\mu$ and $\sigma$ for perceptual-motor processing time is defined as:

$\begin{align}\mu_{pm,c} &= (9+c)\mu_{\text{cr}} + 2\mu_{a} + \mu_{\text{me}}\\\
\sigma_{pm,c} &= \sqrt{(9+c)\sigma_{\text{cr}}^2 + 2\sigma_{a}^2 + \sigma_{\text{me}}^2}
\end{align}$

where $c$ represents the number of category chains, pm indicates the sum of perceptual-motor processes, cr denotes conflict resolution, me denotes motor execution and a denotes visual attention. To make the notation more compact, we will represent these parameters with a single variable, $\gamma_c = \{\mu_{pm,c}, \sigma_{pm,c}\}$.

## State Space

The state space for the Semantic model is 

$S = \{s_{\rm ir}, s_{\rm cc1},   s_{\rm cc2},  s_{\rm yes},  s_{\rm no}\}$

where $s_{\rm ir}$ is the initial retrieval state, $s_{\rm cc1}$ is the first category chain retrieval state, $s_{\rm ccs}$ is the second category chain retrieval state, $s_{\rm yes}$ is the state in which the model responds "yes", and $s_{\rm no}$ is the state in which model responds "no".

## Mappings

As detailed in the first version of the semantic model, the model transitions between states based on the chunk that is retrieved at different points throughout the process. These mappings will be briefly reviewed here. 

The mapping between the results of the retrieval request and $s_{\rm yes}$ is given by:

$\begin{equation}
R_{\rm yes} = \{\mathbf{c_m} \in M: \forall {q\in Q}, p_i(q) = c_m(q) \}\\
\end{equation}$

where $\mathbf{p}_i = \{(\rm object, c_{s,imaginal}(\rm object)), (\rm attribute, category)\}$. The mapping between the result of the retrieval request and a category chain state is given by: 

$\begin{equation}
R_{\rm cc} = \{\mathbf{c_m}  \in M: \forall {q\in Q}, p_k(q) = c_m(q) \}\\
\end{equation}$

where $\mathbf{p}_k = \{(\rm object, c_{s,imaginal}(\rm object)), (\rm attribute, category), (\rm value, \neg c_{s,\rm imaginal}(\rm category)\}$ denotes the conditions for production rule $k$. The mapping between the result of the retrieval request and $s_{\rm no}$ is given by:

$\begin{equation}
R_{\rm no} = \{\mathbf{c_m}  \in M: \exists {q \in Q_{j}} \textrm{ s.t. } p_i(q) \neq c_m(q)\} \cup m'\\
\end{equation}$

where $Q_{j} = \{\rm object, attribute\}$. If the model transitions into absorbing states $s_{\rm yes}$ or $s_{\rm no}$, a response is produced and the model terminates. However, if the model transitions to state $s_{\rm cc1}$, a new retrieval request $\mathbf{r}_{\rm cc1}$ is formed by assigning the value of the value slot of retrieved chunk $\mathbf{c_{r1}} \in R_{\rm cc}$ to the value of the object slot in the new retrieval request: $\mathbf{r}_{\rm cc1} = \{\rm (object, c_{r1}(value)), (attribute, category)\}$. 
 In addition, $\mathbf{c}_{s, imaginal}$ is modified after each category chain $i$ as follows: $c_{s,\rm imaginal}(\rm object) = c_{ri}(value)$ .
 
For some stimuli considered here it is possible to transition to $s_{\rm cc2}$ where a second category chain is performed.  The mapping between the results of retrieval request $\mathbf{r_{cc1}}$ and state $s_{cc2}$ is $R_{\rm cc}$. In state $s_{\rm cc2}$, a new retrieval retrieval request $\mathbf{r_{cc2}} = \{\rm (object, c_{r2}(value)), (attribute,category) \}$ is formed from retrieved chunk $c_{r2} \in R_{\rm cc2}$. 
"""

# ╔═╡ 195280cd-1e61-4c3f-8802-460ba49caace
md"""
## Likelihood Function

The likelihood function for the Semantic model is divided into components for "yes" and "no" responses. Each component is a convolution of Normal random variables representing perceptual-motor processes and LNR random variables representing memory retrieval processes. The likelihood function for "yes" responses is divided into two functions based on the number of category chains required to reach the correct response. 

### Yes Responses

The likelihood of a "yes" response requiring zero category chains is given by:

$\begin{equation}
    f_{\rm yes,0}(rt) = \sum_{\mathbf{c}_m \in R_{\rm yes}} \left(f_{{\rm Normal}; \gamma_0} 
    * f_{{\rm LNR}; \mathbf{c}_m, \Theta}\right)(\textrm{rt}) 
\end{equation}$

where $f_{\rm Normal}$ is the PDF of the Normal distribution, $f_{\rm LNR}$ is the PDF of the Lognormal race model, and $*$ is the convolution operator. This function is applied for questions such as "Is a canary a bird?". The likelihood function for correctly responding "yes" with one category chain includes the convolution with an additional LNR random variable:

$\begin{equation}
    f_{\rm yes,1}(rt) = \sum_{\mathbf{c}_z \in R_{\rm yes}} \sum_{\mathbf{c}_m \in R_{\rm cc_1}} 
    \left(f_{{\rm Normal}; \gamma_1}  * 
    f_{{\rm LNR}; \mathbf{c}_m, \Theta} * f_{{\rm LNR}; \mathbf{c}_z,  \Theta}\right)(\textrm{rt}).
\end{equation}$

The function above is applied to questions such as "Is a canary an animal?".

### No Responses 

Unlike the likelihood function for a "yes" response, the number of category chains leading to a "no" response is unknown. For this reason, the likelihood function is a mixture based on the number of possible category chains leading to a "no" response, either 0, 1, or 2. The maximum number of category chains depends on the stimulus. Each component of this mixture is a mixture itself, based on mismatching chunks and retrieval failures. The likelihood for a "no" response a mixture based on category chains $c$:

$\begin{equation}
f_{\rm no}(rt) = \sum_{c=0}^2 f_{{\rm no},c}(rt).
\end{equation}$

Note that throughout we follow the convention that the sum of an empty set is zero. This means that if the maximum number of category chains 1 (e.g. "Is a canary a bird?"), the summation for $c=2$ drops out. 
The likelihood of responding "no" with zero category chains is defined as:
$\begin{equation}
f_{\rm no,0}(rt) = \sum_{\mathbf{c}_m \in R_{\rm no_0}} \left(f_{{\rm Normal}; \gamma_0} *  f_{{\rm LNR};\mathbf{c}_m,\Theta} \right)(\textrm{rt}).
\end{equation}$
The equation above marginalizes over all the possible chunks $ R_{\rm no_0}$ that leads to a "no response". The likelihood function for responding "no" using one category chain is defined as: 

$\begin{equation}
f_{\rm no,1}(rt)  =
 \sum_{c_k \in R_{\rm cc_1}} \sum_{c_z \in R_{\rm no_1}}  \left(f_{{\rm Normal}; \gamma_1} * f_{{\rm LNR}; \mathbf{c}_k,\Theta} * f_{{\rm LNR};\mathbf{c}_z,\Theta}\right) (\textrm{rt}). 
\end{equation}$

This equation differs from the first equation in that it consists of two retrieval processes: one for the initial retrieval and one for the first category chain. The likelihood function for responding "no" using two category chains is defined as: 

$\begin{equation}
f_{\rm no,2}(rt) = 
  \sum_{c_j \in R_{\rm cc_1}} \sum_{c_w \in R_{\rm cc_2}} \sum_{c_s \in R_{\rm no_2}}  
  \left(
  f_{{\rm Normal}; \gamma_2} * 
  f_{{\rm LNR};\mathbf{c}_j, \Theta} *
  f_{{\rm LNR};\mathbf{c}_w, \Theta} *
  f_{{\rm LNR};\mathbf{c}_s, \Theta} \right)(\textrm{rt})
\end{equation}$

In the equation above, there are three retrieval processes: one for the initial retrieval and two for the category chains. 
"""

# ╔═╡ 0fc1fb0b-4dea-44fd-bd9e-8ff7fa93e32d
md"""
# Generate Data

The `simulate` function generates multiple trials of simulated data for a given stimulus and returns a `NamedTuple` containing the stimulus, the rts for "yes" responses and the rts for "no" responses. `simulate` accepts the following inputs:

- fixed_parms: a `NamedTuple` of fixed parameters
- stimulus: a `NamedTuple` of stimulus slot-value pairs
- n_reps: the number of trials or repetitions for a given stimulus
- blc: the base-level constant parameter.
-  $\delta$: the mismatch penalty parameter

A model object and the stimulus is passed to `simulate_trial` where most of the heavy lifting is performed. `simulate_trial` begins by adding conflict resolution time and stimulus encoding time to the variable `rt`. The next stage of the stimulation is performed in a `while` loop, which continues to execute until a response is given. This allows the model to perform multiple category chains depending on the stimulus and which chunk is retrieved. In side the `while` loop, there are four states or conditions:

1. a retrieval failure, which leads to a "no" response
2. direct verify, which leads to a "yes" response when the chunk matches the retrieval request
3. chain category, which prompts the model "chain the category" by modifying the retrieval request and performing an additional memory retrieval
4. the state occurs when the chunk does not match a category chain or the retrievial request, resulting in a "no" response.
"""

# ╔═╡ c532e826-5066-4cbc-9c82-6f250e75021e
md"""
We will generate 10 repetitions for each of the four stimuli. The resulting data are formated as an array of `NamedTuples` where

- stimulus: a `NamedTuple` of stimulus properties

- yes_rts: a vector of reaction times for yes responses

- no_rts: a vector of reaction times for no responses
"""

# ╔═╡ 4b4de379-e596-46a2-9b55-2a6ff5e46ed3
md"""
## Define Log Likelihood Function

The functions that comprise the likelihood function are orginized in a similar way to the mathematical description of the likelihood function. The top-level function `loglike` iterates through the data and calls one of three functions that computes the likelihood based on the maximum number of category chains for a given stimulus. These functions are `zero_chains`, `one_chain`, and `two_chains`. Each of these functions call other functions based on whether the response is "yes" or "no" and the number of category chains up to the maximum. 


`loglike` requires the following arguments

- blc: the base level constant parameter
-  $\delta$: the mismatch penalty parameter
- fixed_parms: a `NamedTuple` of parameters that are fixed
- data: an array of data

`loglike` initializes the model and model components before looping through the data. Within the loop, the functions `zero_chains`, `one_chain`, or `two_chains` are called, depending on the stimulus. 
"""

# ╔═╡ 4dd0130c-1ad5-455c-910a-7571471649eb
md"""
The functions `zero_chains`, `one_chain`, and `two_chains` compute the likelihood of yes and no responses for stimuli that have a maximum of zero, one or two chains. Each function requires the following inputs:

- actr: an ACT-R model object
- stimulus: a `NamedTuple` with slot-value pars for object, attribute and value
- data: a `NamedTuple` containing a vector of reaction times for yes responses and a vector of reaction times for no responses.
"""

# ╔═╡ a7f8dc6f-f605-462b-acff-7f2dbaf3e241
md"""
For ease of exposition, we will only detail low level function, `zero_chain_no`. Similar functions can be found below. `zero_chain_no` corresponds to the following function introduced above:

$\begin{align*}
f_{\rm no,0}(rt) = \sum_{\mathbf{c}_m \in R_{\rm no_0}} \left(f_{{\rm Normal}; \gamma_0} *  f_{{\rm LNR};\mathbf{c}_m,\Theta} \right)(\textrm{rt}).
\end{align*}$

A question such as "Is a bird a canary" requires zero category chains because there are no intermediate categories between canary and bird. Thus, the set of chunks leading to a "no" response is the compliment of the set of chunks leading to a "yes" response (this set has only one element). At a high level, `zero_chain_no` performs two basic operations. First, it precomputes the activation values for each chunk and convolves the the distribution for perceptual-motor time, which is the same for all possible mappings in $R_{\rm no_0}$. Second, it marginalizes across all chunks whose index does *not* match the index of the chunk that maps to a yes response.  
"""

# ╔═╡ 4eec49e7-d9ba-4530-ac09-2d0beb9f0a9e
md"""
Reveal the cell below to see other functions for the log likelihood
"""

# ╔═╡ 4444a1d6-406f-4129-8ec3-145876d940a2
begin
    import Distributions: pdf, rand, logpdf

    struct LNRC{T1, T2, T3} <: ContinuousUnivariateDistribution
        μ::T1
        σ::T2
        ϕ::T3
        c::Int
    end

    Broadcast.broadcastable(x::LNRC) = Ref(x)

    LNRC(; μ, σ, ϕ, c) = LNRC(μ, σ, ϕ, c)

    function rand(dist::LNRC)
        (; μ, σ, ϕ, c) = dist
        x = @. rand(LogNormal(μ, σ)) + ϕ
        rt, resp = findmin(x)
        return rt
    end

    function rand_sim(dists)
        total_rt = 0.0
        resps = fill(0, length(dists))
        for (i, d) in enumerate(dists)
            (; μ, σ, ϕ) = d
            x = @. rand(LogNormal(μ, σ)) + ϕ
            rt, resp = findmin(x)
            total_rt += rt
            resps[i] = resp
        end
        return resps, total_rt
    end

    rand(dist::LNRC, N::Int) = [rand(dist) for i = 1:N]

    function logpdf(d::T, t::Float64) where {T <: LNRC}
        (; μ, σ, ϕ, c) = d
        LL = 0.0
        for (i, m) in enumerate(μ)
            if i == c
                LL += logpdf(LogNormal(m, σ), t - ϕ)
            else
                LL += logccdf(LogNormal(m, σ), t - ϕ)
            end
        end
        return LL
    end

    function logpdf(d::LNRC{T1, T2, Vector{T3}}, t::Float64) where {T1, T2, T3}
        (; μ, σ, ϕ, c) = d
        LL = 0.0
        for (i, m) in enumerate(μ)
            if i == c
                LL += logpdf(LogNormal(m, σ), t - ϕ[i])
            else
                LL += logccdf(LogNormal(m, σ), t - ϕ[i])
            end
        end
        return LL
    end

    function pdf(d::T, t::Float64) where {T <: LNRC}
        (; μ, σ, ϕ, c) = d
        density = 1.0
        for (i, m) in enumerate(μ)
            if i == c
                density *= pdf(LogNormal(m, σ), t - ϕ)
            else
                density *= (1 - cdf(LogNormal(m, σ), t - ϕ))
            end
        end
        return density
    end
end

# ╔═╡ 95037ee7-feaa-433b-a2c5-f17136272e4a
begin
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
end

# ╔═╡ 4fcce086-5c9c-4365-bf8f-abf5708ab372
begin
    # true value for base level constant
    blc = 1.5
    # true value for mismatch penalty
    δ = 1.0
    # fixed parameters
    fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    stimuli = get_stimuli()
    # repeat simulation 10 times per stimulus
    n_reps = 10
    data = map(s -> simulate(fixed_parms, s, n_reps; blc, δ), stimuli)
    first(data)
end

# ╔═╡ 80bae102-3727-4353-bc1b-a7b64ec448b1
begin
    function one_chain_yes(actr, stimulus, rts)
        probe = stimulus
        (; τ, s) = actr.parms
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
        (; τ, s) = actr.parms
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
        (; τ, s) = actr.parms
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
        (; τ, s) = actr.parms
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
end

# ╔═╡ cfb6f30b-ccd2-488b-9d1c-7b5d4908c5dd
begin
    function zero_chain_no(actr, stimulus, rts)
        (; τ, s) = actr.parms
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

    function zero_chain_yes(actr, stimulus, rts)
        (; τ, s) = actr.parms
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
end

# ╔═╡ 07f39cc2-6591-40d1-84c3-ff8b6fd44270
begin
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
end

# ╔═╡ b9d25ba5-ee4c-47f4-8bce-5d9dbad8b687
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

# ╔═╡ 44f8d6f0-b9ef-4148-989b-a3fc25480aff
md"""
# Plot Likelihood Function

In the following code blocks, the likelihood function is superimposed on simulated data to visually confirm that the likelihood is correctly specified. The plots are divided up by the number of category chains and whether a "yes" or "no" respose is given. In each case, there is a close correspondence between the simulated data and the likelihood function represented by the solid black line. The "no" distributions for 1 and 2 category chains are bimodal because there are multiple ways to respond "no".
"""

# ╔═╡ 3d14558e-6c17-4c0a-b81a-032ec846198a
# ╠═╡ disabled = true
#=╠═╡
let
	blc = 1.0
	δ = 1.0
	fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
	chunks = populate_memory()
	memory = Declarative(;memory=chunks)
	actr = ACTR(;declarative=memory, fixed_parms..., blc=blc, δ=δ, noise=false)
	stimuli = get_stimuli()
	sim_reps = 10_000
	stimulus = stimuli[1]
	sim_data = simulate(fixed_parms, stimulus, sim_reps; blc=blc, δ=δ)
	p_correct = length(sim_data.yes_rts) / sim_reps
	x = .8:.01:1.8
	exact_density_yes = map(i -> zero_chain_yes(actr, stimulus, i) |> exp, x)
	plot_yes = histogram(sim_data.yes_rts, norm=true, grid=false, xlabel="Reaction Time (seconds)", yaxis="Density", 
	    bins=75, leg=false, title="Yes, Zero Chains")
	plot_yes[1][1][:y] .*= p_correct
	plot!(x, exact_density_yes, linewidth=2, color=:black)

	exact_density_no = map(i -> zero_chain_no(actr, stimulus, i) |> exp, x)
	plot_no = histogram(sim_data.no_rts, norm=true, grid=false, xlabel="Reaction Time (seconds)", yaxis="Density", 
	    bins=75, leg=false, title="No, Zero Chains")
	plot_no[1][1][:y] .*= (1-p_correct)
	plot!(x, exact_density_no, linewidth=2, color=:black)
	plot(plot_yes, plot_no, layout=(2,1))
end
  ╠═╡ =#

# ╔═╡ 441c4a21-f4df-4f73-8399-f38ae1ae2918
let
    blc = 1.0
    δ = 1.0
    fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, fixed_parms..., blc = blc, δ = δ, noise = false)
    stimuli = get_stimuli()
    sim_reps = 10_000
    stimulus = stimuli[2]
    sim_data = simulate(fixed_parms, stimulus, sim_reps; blc = blc, δ = δ)
    p_correct = length(sim_data.yes_rts) / sim_reps
    x = 0.8:0.01:2.25
    exact_density_yes = map(i -> one_chain_yes(actr, stimulus, i) |> exp, x)
    plot_yes = histogram(sim_data.yes_rts, norm = true, grid = false,
        xlabel = "Reaction Time (seconds)", yaxis = "Density",
        leg = false, bins = 75, title = "Yes, One Chain")
    plot_yes[1][1][:y] .*= p_correct
    plot!(x, exact_density_yes, linewidth = 2, color = :black)

    exact_density_no = map(i -> one_chain_no(actr, stimulus, i) |> exp, x)
    plot_no = histogram(sim_data.no_rts, norm = true, grid = false,
        xlabel = "Reaction Time (seconds)", yaxis = "Density",
        leg = false, bins = 75, title = "No, One Chain")
    plot_no[1][1][:y] .*= (1 - p_correct)
    plot!(x, exact_density_no, linewidth = 2, color = :black)
    plot(plot_yes, plot_no, layout = (2, 1))
end

# ╔═╡ 786b69c7-bada-4ef6-97a8-cc1a70c68966
let
    blc = 1.0
    δ = 1.0
    fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
    chunks = populate_memory()
    memory = Declarative(; memory = chunks)
    actr = ACTR(; declarative = memory, fixed_parms..., blc = blc, δ = δ, noise = false)
    stimuli = get_stimuli()
    sim_reps = 10_000
    stimulus = stimuli[4]
    sim_data = simulate(fixed_parms, stimulus, sim_reps; blc = blc, δ = δ)
    x = 0.8:0.01:3.25

    exact_density_no = map(i -> two_chains_no(actr, stimulus, i) |> exp, x)
    plot_no = histogram(sim_data.no_rts, norm = true, grid = false,
        xlabel = "Reaction Time (seconds)", yaxis = "Density",
        bins = 100, leg = false, title = "No, Two Chains")
    plot!(x, exact_density_no, linewidth = 2, color = :black)
end

# ╔═╡ b2f13250-056f-4998-acb6-9160a7e67b10
md"""
## Define Model

The prior distributions and model are summarized as follows:

$\begin{align}
\rm blc \sim Normal(1.5,1)
\end{align}$

$\begin{align}
\rm \delta \sim Normal(1,.5)
\end{align}$


$\begin{align}
\textrm{rts}_{\textrm{yes}} \sim f_{\textrm{yes}}(\textrm{blc},\delta)
\end{align}$

$\begin{align}
\textrm{rts}_{\textrm{no}} \sim f_{\textrm{no}}(\textrm{blc},\delta)
\end{align}$

In computer code, the model is specified as follows:
"""

# ╔═╡ 9cc6b9dc-379c-40a0-af4d-0d53318a00df
begin
    # define prior distribution of parameters
    function sample_prior()
        blc = rand(Normal(1.5, 1))
        δ = rand(truncated(Normal(1, 0.5), 0.0, Inf))
        return [blc, δ]
    end

    function prior_loglike(blc, δ)
        LL = 0.0
        LL += logpdf(Normal(1.5, 1), blc)
        LL += logpdf(truncated(Normal(1, 0.5), 0.0, Inf), δ)
        return LL
    end

    # lower and upper bounds for each parameter
    bounds = ((-Inf, Inf), (eps(), Inf))
    names = (:blc, :δ)
end

# ╔═╡ 06f52cc9-63b6-4aad-b46c-cbf037989aa7
md"""
## Estimate Parameters

Now that the priors, likelihood and Turing model have been specified, we can now estimate the parameters. In the following code, we will run four MCMC chains with the NUTS sample for 2,000 iterations and omit the first 1,000 warmup samples. 
"""

# ╔═╡ 34bf98ae-7699-43cf-972d-ef1f7ab1cb13
begin
    # create model object
    de_model = DEModel(fixed_parms; sample_prior, prior_loglike, loglike, data, names)
    # create DE sampler object
    de = DE(; bounds, burnin = 1000, sample_prior, n_groups = 2, Np = 4)
    # total iterations
    n_iter = 2000
    chains = sample(de_model, de, MCMCThreads(), n_iter, progress = false)
end

# ╔═╡ 98e4ca33-dfef-4bfd-837a-a156571cd1bc
md"""
## Results

According to the summary above, $\hat{r} \approx 1$ for both parameters, which suggests that the chains converged. 
The trace plot below shows acceptable mixing among the chains.  In the density plots, the posterior distributions are centered near the data generating parameters of blc = 1.5 and $\delta = 1.$
"""

# ╔═╡ b7e35ff1-cc27-4b23-86a2-f8c6799448ae
let
    ch = group(chains, :blc)
    p1 = plot(ch, seriestype = (:traceplot), grid = false)
    p2 = plot(ch, seriestype = (:autocorplot), grid = false)
    p3 = plot(ch, seriestype = (:mixeddensity), grid = false)
    pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
end

# ╔═╡ e1bf7215-512c-407f-95c4-4f985ed1023b
let
    ch = group(chains, :δ)
    p1 = plot(ch, seriestype = (:traceplot), grid = false)
    p2 = plot(ch, seriestype = (:autocorplot), grid = false)
    p3 = plot(ch, seriestype = (:mixeddensity), grid = false)
    pcτ = plot(p1, p2, p3, layout = (3, 1), size = (600, 600))
end

# ╔═╡ 31c3f03d-b1ca-4af0-a84a-add078fb5c3b
md"""
### Posterior Predictive Distribution

The plot below shows the posterior predictive distribution of reaction times of "yes" and "no" responses for each of the four stimuli. Three patterns can be observed: (1) reaction times increase with the number of category chains required to answer the question (e.g. reaction times are longer for canary-animal compared to canary-bird), (2) the model cannot respond "yes" when the correct answer is "no", (3) the mixture of category chains is apparent on "no" responses for the questions canary-animal, bird-fish, canary-fish. 
"""

# ╔═╡ 12c2a1cf-5598-4088-8af9-21764d704608
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

# ╔═╡ 087084a3-fd37-4826-b8af-42ab7e4d111a
begin
    rt_preds(s) =
        posterior_predictive(x -> simulate(fixed_parms, s, n_reps; x...), chains, 1000)
    temp_preds = map(s -> rt_preds(s), stimuli)
    preds = merge.(temp_preds)
    grid_plot(preds, stimuli)
end

# ╔═╡ b3aba7ef-b498-40fd-b069-0acc3adc93de
md"""
# References

Weaver, R. (2008). Parameters, predictions, and evidence in computational modeling: A statistical view informed by ACT–R. Cognitive Science, 32(8), 1349-1375.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ACTRModels = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
CommonMark = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DifferentialEvolutionMCMC = "607db5a9-722a-4af8-9a06-1810c0fe385b"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FFTDists = "872ebf62-bb8d-4945-b244-254eba715075"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
NamedTupleTools = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
ACTRModels = "~0.13.0"
CommonMark = "~0.8.12"
DataFrames = "~1.6.1"
DifferentialEvolutionMCMC = "~0.7.8"
Distributions = "~0.25.109"
FFTDists = "~0.1.4"
MCMCChains = "~6.0.6"
NamedTupleTools = "~0.14.3"
PlutoUI = "~0.7.59"
Revise = "~3.5.15"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "cb5dedede5da515de6e055b1f8c3357129be81fd"

[[deps.ACTRModels]]
deps = ["ConcreteStructs", "Distributions", "PrettyTables", "Random", "SafeTestsets", "Test"]
git-tree-sha1 = "1d14ef780cf9b8162a92a933a202a8e5cbb0638b"
uuid = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
version = "0.13.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "b0489adc45a7c8cf0d8e2ddf764f89c1c3decebd"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "08e5fc6620a8d83534bf6149795054f1b1e8370a"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.2"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "b8fe8546d52ca154ac556809e10c75e6e7430ac8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.5"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonMark]]
deps = ["Crayons", "JSON", "PrecompileTools", "URIs"]
git-tree-sha1 = "532c4185d3c9037c0237546d817858b23cf9e071"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.8.12"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Dierckx]]
deps = ["Dierckx_jll"]
git-tree-sha1 = "d1ea9f433781bb6ff504f7d3cb70c4782c504a3a"
uuid = "39dd38d3-220a-591b-8e3c-4c3a8c710a94"
version = "0.5.3"

[[deps.Dierckx_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6596b96fe1caff3db36415eeb6e9d3b50bfe40ee"
uuid = "cd4c43a9-7502-52ba-aa6d-59fb2a88580b"
version = "0.1.0+0"

[[deps.DifferentialEvolutionMCMC]]
deps = ["AbstractMCMC", "ConcreteStructs", "Distributions", "LinearAlgebra", "MCMCChains", "Parameters", "ProgressMeter", "Random", "SafeTestsets", "StatsBase", "ThreadsX"]
git-tree-sha1 = "b3f78fc1a807aec28f6de42834d7228e38f1b1f6"
uuid = "607db5a9-722a-4af8-9a06-1810c0fe385b"
version = "0.7.8"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTDists]]
deps = ["Dierckx", "Distributions", "FFTW", "LambertW", "Random", "SafeTestsets", "Statistics", "StatsBase"]
git-tree-sha1 = "e8df8394b548e7021b90bac6c9bca90cd9868d7a"
uuid = "872ebf62-bb8d-4945-b244-254eba715075"
version = "0.1.4"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "3f74912a156096bd8fdbef211eff66ab446e7297"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "3e527447a45901ea392fe12120783ad6ec222803"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.6"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "182c478a179b267dd7a741b6f8f4c3e0803795d6"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.6+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14eb2b542e748570b56446f4c50fbfb2306ebc45"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "e7cbed5032c4c397a6ac23d1493f3289e01231c4"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.14"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "a6adc2dcfe4187c40dc7c2c9d2128e326360e90a"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.32"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "5b0d630f3020b82c0775a51d05895852f8506f50"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.4"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "eeaedcf337f33c039f9f3a209a8db992deefd7e9"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.4.8"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "d28056379864318172ff4b7958710cfddd709339"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.6"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8ba8b1840d3ab5b38e7c71c23c3193bb5cbc02b5"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.10"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "91a67b4d73842da90b526011fa85c5c4c9343fe0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.18"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a028ee3cb5641cccc4c24e90c36b0a4f7707bdf5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.14+0"

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
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "082f0c4b70c202c37784ce4bfbc33c9f437685bf"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.5"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

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
git-tree-sha1 = "80686d28ecb3ee7fb3ac5371cacaa0d673eb0d4a"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.1"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "02d31ad62838181c1a3a5fd23a1ce5914a643601"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.3"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "85ddd93ea15dcd8493400600e09104a9e94bb18d"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.15"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SafeTestsets]]
git-tree-sha1 = "81ec49d645af090901120a1542e67ecbbe044db3"
uuid = "1bc83da4-3b8d-516f-aca4-4fe02f6d838f"
version = "0.1.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["Accessors", "ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "SplittablesBase", "Transducers"]
git-tree-sha1 = "70bd8244f4834d46c3d68bd09e7792d8f571ef04"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.12"

[[deps.TranscodingStreams]]
git-tree-sha1 = "60df3f8126263c0d6b357b9a1017bb94f53e3582"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.0"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Accessors", "Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "5215a069867476fc8e3469602006b9670e68da23"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.82"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "dd260903fdabea27d9b6021689b3cd5401a57748"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.20.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "d9717ce3518dc68a99e6b96300813760d887a01d"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.1+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─b2505b48-dab3-4051-b008-9bf5bfbe87da
# ╟─90feb1a2-fc4a-11ec-1fb3-bb5d1d9fd934
# ╟─5fa6fcc2-0c09-4074-8f15-6e2ac17a354f
# ╟─d64fbbaa-25f4-4aa7-95c6-d58e56e833a3
# ╟─340b62c9-8c73-4bc0-a12b-5a018e18125d
# ╟─9e14a221-5423-4e83-8978-2abac5ab9ae2
# ╟─195280cd-1e61-4c3f-8802-460ba49caace
# ╟─0fc1fb0b-4dea-44fd-bd9e-8ff7fa93e32d
# ╟─c532e826-5066-4cbc-9c82-6f250e75021e
# ╟─95037ee7-feaa-433b-a2c5-f17136272e4a
# ╠═4fcce086-5c9c-4365-bf8f-abf5708ab372
# ╟─4b4de379-e596-46a2-9b55-2a6ff5e46ed3
# ╠═b9d25ba5-ee4c-47f4-8bce-5d9dbad8b687
# ╟─4dd0130c-1ad5-455c-910a-7571471649eb
# ╠═07f39cc2-6591-40d1-84c3-ff8b6fd44270
# ╟─a7f8dc6f-f605-462b-acff-7f2dbaf3e241
# ╠═cfb6f30b-ccd2-488b-9d1c-7b5d4908c5dd
# ╟─4eec49e7-d9ba-4530-ac09-2d0beb9f0a9e
# ╟─80bae102-3727-4353-bc1b-a7b64ec448b1
# ╟─4444a1d6-406f-4129-8ec3-145876d940a2
# ╟─44f8d6f0-b9ef-4148-989b-a3fc25480aff
# ╟─3d14558e-6c17-4c0a-b81a-032ec846198a
# ╟─441c4a21-f4df-4f73-8399-f38ae1ae2918
# ╟─786b69c7-bada-4ef6-97a8-cc1a70c68966
# ╟─b2f13250-056f-4998-acb6-9160a7e67b10
# ╠═9cc6b9dc-379c-40a0-af4d-0d53318a00df
# ╟─06f52cc9-63b6-4aad-b46c-cbf037989aa7
# ╠═34bf98ae-7699-43cf-972d-ef1f7ab1cb13
# ╟─98e4ca33-dfef-4bfd-837a-a156571cd1bc
# ╟─b7e35ff1-cc27-4b23-86a2-f8c6799448ae
# ╟─e1bf7215-512c-407f-95c4-4f985ed1023b
# ╟─31c3f03d-b1ca-4af0-a84a-add078fb5c3b
# ╟─12c2a1cf-5598-4088-8af9-21764d704608
# ╟─087084a3-fd37-4826-b8af-42ab7e4d111a
# ╟─b3aba7ef-b498-40fd-b069-0acc3adc93de
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
