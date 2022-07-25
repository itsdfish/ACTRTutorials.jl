### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 90feb1a2-fc4a-11ec-1fb3-bb5d1d9fd934
begin
	using StatsPlots, Revise, ACTRModels, Random
	using PlutoUI, DataFrames, CommonMark,NamedTupleTools
	using FFTDists, DifferentialEvolutionMCMC, MCMCChains
	# seed random number generator
	Random.seed!(9851);
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
	
	struct LNRC{T1,T2,T3} <: ContinuousUnivariateDistribution
	    μ::T1
	    σ::T2
	    ϕ::T3
	    c::Int
	end
	
	Broadcast.broadcastable(x::LNRC) = Ref(x)
	
	LNRC(;μ, σ, ϕ, c) = LNRC(μ, σ, ϕ, c)
	
	function rand(dist::LNRC)
	    @unpack μ,σ,ϕ,c = dist
	    x = @. rand(LogNormal(μ, σ)) + ϕ
	    rt,resp = findmin(x)
	    return rt
	end
	
	function rand_sim(dists)
	    total_rt = 0.0
	    resps = fill(0, length(dists))
	    for (i,d) in enumerate(dists)
	        @unpack μ,σ,ϕ = d 
	        x = @. rand(LogNormal(μ, σ)) + ϕ
	        rt,resp = findmin(x)
	        total_rt += rt
	        resps[i] = resp
	    end
	    return resps,total_rt
	end
	
	rand(dist::LNRC, N::Int) = [rand(dist) for i in 1:N]
	
	function logpdf(d::T, t::Float64) where {T <: LNRC}
	    @unpack μ,σ,ϕ,c = d
	    LL = 0.0
	    for (i,m) in enumerate(μ)
	        if i == c
	            LL += logpdf(LogNormal(m, σ), t - ϕ)
	        else
	            LL += logccdf(LogNormal(m, σ), t - ϕ)
	        end
	    end
	    return LL
	end
	
	function logpdf(d::LNRC{T1,T2,Vector{T3}}, t::Float64) where {T1,T2,T3}
	    @unpack μ,σ,ϕ,c = d
	    LL = 0.0
	    for (i,m) in enumerate(μ)
	        if i == c
	            LL += logpdf(LogNormal(m, σ), t - ϕ[i])
	        else
	            LL += logccdf(LogNormal(m, σ), t - ϕ[i])
	        end
	    end
	    return LL
	end
	
	function pdf(d::T, t::Float64) where {T <: LNRC}
	    @unpack μ,σ,ϕ,c = d
	    density = 1.0
	    for (i,m) in enumerate(μ)
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
	function populate_memory(act=0.0)
	    chunks = [
	        Chunk(object=:shark, attribute=:dangerous, value=:True, act=act),
	        Chunk(object=:shark, attribute=:locomotion, value=:swimming, act=act),
	        Chunk(object=:shark, attribute=:category, value=:fish, act=act),
	        Chunk(object=:salmon, attribute=:edible, value=:True, act=act),
	        Chunk(object=:salmon, attribute=:locomotion, value=:swimming, act=act),
	        Chunk(object=:salmon, attribute=:category, value=:fish, act=act),
	        Chunk(object=:fish, attribute=:breath, value=:gills, act=act),
	        Chunk(object=:fish, attribute=:locomotion, value=:swimming, act=act),
	        Chunk(object=:fish, attribute=:category, value=:animal, act=act),
	        Chunk(object=:animal, attribute=:moves, value=:True, act=act),
	        Chunk(object=:animal, attribute=:skin, value=:True, act=act),
	        Chunk(object=:canary, attribute=:color, value=:yellow, act=act),
	        Chunk(object=:canary, attribute=:sings, value=:True, act=act),
	        Chunk(object=:canary, attribute=:category, value=:bird, act=act),
	        Chunk(object=:ostritch, attribute=:flies, value=:False, act=act),
	        Chunk(object=:ostritch, attribute=:height, value=:tall, act=act),
	        Chunk(object=:ostritch, attribute=:category, value=:bird, act=act),
	        Chunk(object=:bird, attribute=:wings, value=:True, act=act),
	        Chunk(object=:bird, attribute=:locomotion, value=:flying, act=act),
	        Chunk(object=:bird, attribute=:category, value=:animal, act=act),
	    ]
	    return chunks
	end
	
	function simulate(fixed_parms, stimulus, n_reps; blc, δ)
	    # generate chunks 
	    chunks = populate_memory()
	    # add chunks to declarative memory
	    memory = Declarative(;memory=chunks)
	    # add declarative memory and parameters to ACT-R object
	    actr = ACTR(;declarative=memory, fixed_parms..., blc, δ)
	    # rts for yes responses
	    yes_rts = Float64[]
	    # rates for no respones
	    no_rts = Float64[]
	    for rep in 1:n_reps
	        # simulate a single trial
	        resp,rt = simulate_trial(actr, stimulus)
	        # save simulated data
	        resp == :yes ? push!(yes_rts, rt) : push!(no_rts, rt)
	    end
	    return (stimulus=stimulus, yes_rts = yes_rts, no_rts = no_rts)
	end
	
	function simulate_trial(actr, stimulus)
	    retrieving = true
	    probe = stimulus
	    response = :_
	    # add conflict resolution times
	    rt = mapreduce(_ -> process_time(.05), +, 1:7)
	    # add stimulus encoding times
	    rt += mapreduce(_ -> process_time(.085), +, 1:2)
	    while retrieving
	        # conflict resolution
	        rt += process_time(.05)
	        chunk = retrieve(actr; object=probe.object, attribute=:category)
	        rt += compute_RT(actr, chunk)
	        # retrieval failure, respond "no"
	        if isempty(chunk)
	            # add motor execution time
	            rt += process_time(.05) + process_time(.21)
	            retrieving = false
	            response = :no
	        # can respond "yes"
	        elseif direct_verify(chunk[1], probe)
	            # add motor execution time
	            rt += process_time(.05) + process_time(.21)
	            retrieving = false
	            response = :yes
	        # category chain
	        elseif chain_category(chunk[1], probe) 
	            probe = delete(probe, :object)
	            # update memory probe for category chain
	            probe = (object = chunk[1].slots.value, probe...)
	        else
	            response = :no
	            rt += process_time(.05) + process_time(.21)
	            retrieving = false
	        end
	    end
	    return response, rt
	end
	
	process_time(μ) = rand(Uniform(μ * (2 / 3), μ * (4 / 3)))
	
	function direct_verify(chunk, stim)
	    return match(chunk, object=stim.object,
	        value=stim.category, attribute=:category)
	end
	
	function chain_category(chunk, stim)
	    return match(chunk, ==, !=, ==, object=stim.object,
	        value=stim.category, attribute=:category)
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
	data = map(s -> simulate(fixed_parms, s, n_reps; blc, δ), stimuli);
	first(data)
end

# ╔═╡ 80bae102-3727-4353-bc1b-a7b64ec448b1
begin
	function one_chain_yes(actr, stimulus, rts)
		probe = stimulus
		@unpack τ,s = actr.parms
		chunks = actr.declarative.memory
		σ = s * π / sqrt(3)
		μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 10), visual=(μ = .085,N = 2))
		compute_activation!(actr; object=get_object(stimulus), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
		chain_idx = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
		chain1_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain_idx)
		
		probe = (object = get_chunk_value(chunks[chain_idx]), delete(probe, :object)...)
		compute_activation!(actr; object=get_object(probe), attribute=:category)
		yes_idx = find_index(actr, object=get_object(probe), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
		yes_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=yes_idx)
		model = Normal(μpm, σpm) + chain1_dist + yes_dist
		convolve!(model)
		LLs = logpdf.(model, rts)
		return sum(LLs)
	end
	
	function one_chain_no_branch1(actr, stimulus, rts)
		# respond no after initial retrieval
		probe = stimulus
		@unpack τ,s = actr.parms
		chunks = actr.declarative.memory
		σ = s * π / sqrt(3)
		μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 9), visual=(μ = .085,N = 2))
		compute_activation!(actr; object=get_object(stimulus), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
		chain_idx = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
		# Initialize likelihood
		n_resp = length(rts)
		likelihoods = fill(0.0, n_resp)
		Nc = length(chunks) + 1
		# Marginalize over all of the possible chunks that could have lead to the
		# observed response
		for i in 1:Nc
			# Exclude the chunk representing the stimulus because the response was "no"
			if i != chain_idx
				no_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
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
		@unpack τ,s = actr.parms
		chunks = actr.declarative.memory
		σ = s * π / sqrt(3)
		# perceptual motor time
		μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 10), visual=(μ = .085,N = 2))
		# compute activations
		compute_activation!(actr; object=get_object(stimulus), attribute=:category)
		# get activations
		μ = map(x -> x.act, chunks)
		# add retrieval threshold
		push!(μ, τ)
		# index to chain chunk
		chain_idx = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
		chain1_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain_idx)
		
		# change probe for second retrieval
		probe = (object = get_chunk_value(chunks[chain_idx]), delete(probe, :object)...)
		compute_activation!(actr; object=get_object(probe), attribute=:category)
		yes_idx = find_index(actr, object=get_object(probe), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
	
		# Initialize likelihood
		n_resp = length(rts)
		likelihoods = fill(0.0, n_resp)
		Nc = length(chunks) + 1
		# Marginalize over all of the possible chunks that could have lead to the
		# observed response
		for i in 1:Nc
			# Exclude the chunk representing the stimulus because the response was "no"
			if i != yes_idx
				no_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
				model = Normal(μpm, σpm) + chain1_dist + no_dist
				convolve!(model)
				likelihoods .+= pdf.(model, rts)
			end
		end
		return likelihoods
	end
	
	
	function two_chains_no_branch3(actr, stimulus, rts)
		probe = stimulus
		@unpack τ,s = actr.parms
		chunks = actr.declarative.memory
		σ = s * π / sqrt(3)
		# perceptual motor time
		μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 11), visual=(μ = .085,N = 2))
		# compute activations
		compute_activation!(actr; object=get_object(stimulus), attribute=:category)
		# get activations
		μ = map(x -> x.act, chunks)
		# add retrieval threshold
		push!(μ, τ)
		# index to chain chunk
		chain_idx1 = find_index(actr, ==, !=, ==, object=get_object(probe), value=get_category(probe), attribute=:category)
		chain1_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain_idx1)
		
		# change probe for second retrieval
		probe = (object = get_chunk_value(chunks[chain_idx1]), delete(probe, :object)...)
		compute_activation!(actr; object=get_object(probe), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
		chain2_idx = find_index(actr, object=get_object(probe), attribute=:category)
		chain2_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=chain2_idx)
	
		# change probe for third retrieval
		probe = (object = get_chunk_value(chunks[chain2_idx]), delete(probe, :object)...)
		compute_activation!(actr; object=get_object(probe), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
		chain3_idx = find_index(actr, object=get_object(probe), attribute=:category)
	
		# Initialize likelihood
		n_resp = length(rts)
		likelihoods = fill(0.0, n_resp)
		Nc = length(chunks) + 1
		# Marginalize over all of the possible chunks that could have lead to the
		# observed response
		for i in 1:Nc
			# Exclude the chunk representing the stimulus because the response was "no"
			if i != chain3_idx
				no_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
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
		yes = map(x->x.yes_rts, data) |> x->vcat(x...)
		no = map(x->x.no_rts, data) |> x->vcat(x...)
		return (yes=yes, no=no)
	end
	
end

# ╔═╡ cfb6f30b-ccd2-488b-9d1c-7b5d4908c5dd
begin
	function zero_chain_no(actr, stimulus, rts)
		@unpack τ,s = actr.parms
		chunks = actr.declarative.memory
		σ = s * π / sqrt(3)
		# normal approximation for perceptual-motor time
		μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 9), visual=(μ = .085,N = 2))
		# compute the activation for all chunks
		compute_activation!(actr; object=get_object(stimulus), attribute=:category)
		# extract the mean activation values
		μ = map(x -> x.act, chunks)
		# add retrieval threshold to mean activation values
		push!(μ, τ)
		# find the chunk index corresponding to a "yes" response
		yes_idx = find_index(actr, object=get_object(stimulus), value=get_category(stimulus))
		# Initialize likelihood
		n_resp = length(rts)
		# initialize likelihoods for each no response
		likelihoods = fill(0.0, n_resp)
		Nc = length(chunks) + 1
		# Marginalize over all of the possible chunks that could have lead to the
		# observed response
		for i in 1:Nc
			# Exclude the chunk representing the stimulus because the response was "no"
			if i != yes_idx
				# create Lognormal race distribution for chunk i
				retrieval_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=i)
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
		@unpack τ,s = actr.parms
		chunks = actr.declarative.memory
		σ = s * π / sqrt(3)
		μpm,σpm = convolve_normal(motor=(μ = .21,N = 1), cr=(μ = .05,N = 9), visual=(μ = .085,N = 2))
		compute_activation!(actr; object=get_object(stimulus), attribute=:category)
		μ = map(x -> x.act, chunks)
		push!(μ, τ)
		yes_idx = find_index(actr, object=get_object(stimulus), value=get_category(stimulus))
		retrieval_dist = LNRC(;μ=-μ, σ=σ, ϕ=0.0, c=yes_idx)
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
	memory = Declarative(;memory=chunks)
	# add declarative memory and parameters to ACT-R object
	actr = ACTR(;declarative=memory, fixed_parms..., blc, δ, noise=false)
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

# ╔═╡ 441c4a21-f4df-4f73-8399-f38ae1ae2918
let 
	blc = 1.0
	δ = 1.0
	fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
	chunks = populate_memory()
	memory = Declarative(;memory=chunks)
	actr = ACTR(;declarative=memory, fixed_parms..., blc=blc, δ=δ, noise=false)
	stimuli = get_stimuli()
	sim_reps = 10_000
	stimulus = stimuli[2]
	sim_data = simulate(fixed_parms, stimulus, sim_reps; blc=blc, δ=δ)
	p_correct = length(sim_data.yes_rts) / sim_reps
	x = .8:.01:2.25
	exact_density_yes = map(i -> one_chain_yes(actr, stimulus, i) |> exp, x)
	plot_yes = histogram(sim_data.yes_rts, norm=true, grid=false, xlabel="Reaction Time (seconds)", yaxis="Density", 
	    leg=false, bins=75, title="Yes, One Chain")
	plot_yes[1][1][:y] .*= p_correct
	plot!(x, exact_density_yes, linewidth=2, color=:black)
	
	exact_density_no = map(i -> one_chain_no(actr, stimulus, i) |> exp, x)
	plot_no = histogram(sim_data.no_rts, norm=true, grid=false, xlabel="Reaction Time (seconds)", yaxis="Density", 
	    leg=false, bins=75, title="No, One Chain")
	plot_no[1][1][:y] .*= (1-p_correct)
	plot!(x, exact_density_no, linewidth=2, color=:black)
	plot(plot_yes, plot_no, layout=(2,1))
end

# ╔═╡ 786b69c7-bada-4ef6-97a8-cc1a70c68966
let
	blc = 1.0
	δ = 1.0
	fixed_parms = (noise = true, τ = 0.0, s = 0.2, mmp = true)
	chunks = populate_memory()
	memory = Declarative(;memory=chunks)
	actr = ACTR(;declarative=memory, fixed_parms..., blc=blc, δ=δ, noise=false)
	stimuli = get_stimuli()
	sim_reps = 10_000
	stimulus = stimuli[4]
	sim_data = simulate(fixed_parms, stimulus, sim_reps; blc=blc, δ=δ)
	x = .8:.01:3.25
	
	exact_density_no = map(i -> two_chains_no(actr, stimulus, i) |> exp, x)
	plot_no = histogram(sim_data.no_rts, norm=true, grid=false, xlabel="Reaction Time (seconds)", yaxis="Density", 
	    bins=100, leg=false, title="No, Two Chains")
	plot!(x, exact_density_no, linewidth=2, color=:black)
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
	    δ = rand(truncated(Normal(1, .5), 0.0, Inf))
		return [blc,δ]
	end

	function prior_loglike(blc, δ)
	    LL = 0.0
	   	LL += logpdf(Normal(1.5, 1), blc)
	    LL += logpdf(truncated(Normal(1, .5), 0.0, Inf), δ)
	    return LL
	end
	
	# lower and upper bounds for each parameter
	bounds = ((-Inf,Inf),(eps(),Inf))
	names = (:blc,:δ)
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
	de = DE(;bounds, burnin=1000, sample_prior, n_groups=2, Np=4)
	# total iterations
	n_iter = 2000
	chains = sample(de_model, de, MCMCThreads(), n_iter, progress=false)
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
	p1 = plot(ch, seriestype=(:traceplot), grid=false)
	p2 = plot(ch, seriestype=(:autocorplot), grid=false)
	p3 = plot(ch, seriestype=(:mixeddensity), grid=false)
	pcτ = plot(p1, p2, p3, layout=(3,1), size=(600,600))
end

# ╔═╡ e1bf7215-512c-407f-95c4-4f985ed1023b
let
	ch = group(chains, :δ)
	p1 = plot(ch, seriestype=(:traceplot), grid=false)
	p2 = plot(ch, seriestype=(:autocorplot), grid=false)
	p3 = plot(ch, seriestype=(:mixeddensity), grid=false)
	pcτ = plot(p1, p2, p3, layout=(3,1), size=(600,600))
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
        prob_yes = length(pred.yes)/(length(pred.yes) + length(pred.no))
        object = stimulus.object
        category = stimulus.category
        hist = histogram(layout=(1,2), xlims=(0,2.5),  ylims=(0,3.5), title="$object-$category",
            grid=false, titlefont=font(12), xaxis=font(12), yaxis=font(12), xlabel="Yes RT", 
            xticks = 0:2, yticks=0:3)

        if !isempty(pred.yes)
            histogram!(hist, pred.yes, xlabel="Yes RT", norm=true, grid=false, color=:grey, leg=false, 
                size=(300,250), subplot=1)
            hist[1][1][:y] *= prob_yes
        end

        prob_no = 1 - prob_yes
        object = stimulus.object
        category = stimulus.category
        histogram!(hist, pred.no, xlabel="No RT", norm=true, grid=false, color=:grey, leg=false, 
            size=(300,250), subplot=2)
        hist[2][1][:y] *= prob_no
        push!(posterior_plots, hist)
    end
    return posterior_plot = plot(posterior_plots..., layout=(2,2), size=(800,600); kwargs...)
end

# ╔═╡ 087084a3-fd37-4826-b8af-42ab7e4d111a
begin
	rt_preds(s) = posterior_predictive(x -> simulate(fixed_parms, s, n_reps; x...), chains, 1000)
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
ACTRModels = "~0.10.6"
CommonMark = "~0.8.6"
DataFrames = "~1.3.4"
DifferentialEvolutionMCMC = "~0.7.2"
Distributions = "~0.25.64"
FFTDists = "~0.1.1"
MCMCChains = "~5.3.1"
NamedTupleTools = "~0.14.0"
PlutoUI = "~0.7.39"
Revise = "~3.3.3"
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
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

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
git-tree-sha1 = "2dd813e5f2f7eec2d1268c57cf2373d3ee91fcea"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.1"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "6d4fa04343a7fc9f9cb9cff9558929f3d2752717"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.0.9"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

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

[[deps.CommonMark]]
deps = ["Crayons", "JSON", "URIs"]
git-tree-sha1 = "4cd7063c9bdebdbd55ede1af70f3c2f48fab4215"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.8.6"

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
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
git-tree-sha1 = "a599cfb8b1909b0f97c5e1b923ab92e1c0406076"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.1"

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

[[deps.Dierckx]]
deps = ["Dierckx_jll"]
git-tree-sha1 = "633c119fcfddf61fb4c75d77ce3ebab552a44723"
uuid = "39dd38d3-220a-591b-8e3c-4c3a8c710a94"
version = "0.5.2"

[[deps.Dierckx_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6596b96fe1caff3db36415eeb6e9d3b50bfe40ee"
uuid = "cd4c43a9-7502-52ba-aa6d-59fb2a88580b"
version = "0.1.0+0"

[[deps.DifferentialEvolutionMCMC]]
deps = ["AbstractMCMC", "ConcreteStructs", "Distributions", "LinearAlgebra", "MCMCChains", "Parameters", "ProgressMeter", "Random", "SafeTestsets", "StatsBase"]
git-tree-sha1 = "46462b8b944e30f33e0ea57f1245fd42804872c3"
uuid = "607db5a9-722a-4af8-9a06-1810c0fe385b"
version = "0.7.2"

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
git-tree-sha1 = "d530092b57aef8b96b27694e51c575b09c7f0b2e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.64"

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

[[deps.FFTDists]]
deps = ["Dierckx", "Distributions", "FFTW", "LambertW", "Random", "SafeTestsets", "Statistics", "StatsBase"]
git-tree-sha1 = "487f521c71894563fe4af37eb84d291a6b7ea959"
uuid = "872ebf62-bb8d-4945-b244-254eba715075"
version = "0.1.1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

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

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "52617c41d2761cc05ed81fe779804d3b7f14fff7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.13"

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

[[deps.LambertW]]
git-tree-sha1 = "2d9f4009c486ef676646bca06419ac02061c088e"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.5"

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

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

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
git-tree-sha1 = "b8073fe6973dcfad5fec803dabc1d3a7f6c4ebc8"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.4.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "891d3b4e8f8415f53108b4918d0183e61e18015b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.0"

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

[[deps.NamedTupleTools]]
git-tree-sha1 = "befc30261949849408ac945a1ebb9fa5ec5e1fd5"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.0"

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
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

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
git-tree-sha1 = "9a36165cf84cff35851809a40a928e1103702013"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.16+0"

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
git-tree-sha1 = "ca433b9e2f5ca3a0ce6702a032fce95a3b6e1e48"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.14"

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
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "93e82cebd5b25eb33068570e3f63a86be16955be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.1"

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

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "4d4239e93531ac3e7ca7e339f15978d0b5149d03"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.3.3"

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
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "9f8a5dc5944dc7fbbe6eb4180660935653b0a9d9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

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
git-tree-sha1 = "48598584bacbebf7d30e20880438ed1d24b7c7d6"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.18"

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
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

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
