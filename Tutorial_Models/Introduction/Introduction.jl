### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c40c7ada-454a-4fa3-a71a-4d85e92fec03
begin
	using Distributions, Plots, Random, PlutoUI
	using CommonMark, HypertextLiteral, Optim
	# seed random number generator
	Random.seed!(2050)
	TableOfContents()
end

# ╔═╡ 5927a78e-6234-4b51-a1ce-9df2ca57e862
let
	using ACTRModels
	
	Random.seed!(87545)
	# create chunks of declarative knowledge
	chunks = [
	    Chunk(;name=:Bob, department=:accounting),
	    Chunk(;name=:Alice, department=:HR)
	    ]
	
	# initialize declarative memory
	declarative = Declarative(memory=chunks)
	
	# specify model parameters: partial matching, noise, mismatch penalty, activation noise
	Θ = (mmp=true, noise=true, δ=.5, s=.2)  
	
	# create an ACT-R object with activation noise and partial matching
	actr = ACTR(;declarative, Θ...)
	
	# compute activation for each chunk
	compute_activation!(actr; department=:accounting)
	# get mean activation
	μ = get_mean_activations(actr)
	# standard deviation 
	σ = Θ.s * pi / sqrt(3)
	# lognormal race distribution object
	dist = LNR(;μ=-μ, σ, ϕ=0.0)
	
	
	# index for accounting
	idx = find_index(actr; department=:accounting)
	# generate retrieval times
	rts = rand(dist, 10^5)
	# extract rts for accounting
	acc_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
	# probability of retrieving accounting
	p_acc = length(acc_rts)/length(rts)
	
	font_size = 12
	plots = plot(color=:grey, grid=false, legend=true,
	    bins = 100, xlabel="Mean Reaction Time", ylabel="", xlims=(.5,2), 
	    layout=(2,1), linewidth=2, xaxis=font(font_size), yaxis=font(font_size), legendfontsize=10)
	plot!(subplot=1, title="Old")
	means = [mean(acc_rts), mean(acc_rts)+.1]
	vline!([means[1]], color=:darkred, label="model", linewidth=2)
	vline!([means[2]], color=:black, label="data", linewidth=2)
	plot!(means, [2,2], linestyle=:dash, color=:grey, subplot=1, label="Difference", linewidth=2)
	
	# index for HR
	idx = find_index(actr; department=:HR)
	# extract rts for accounting
	hr_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
	plot!(subplot=2, title="New")
	means = [mean(hr_rts), mean(hr_rts)+.1]
	vline!([means[1]], color=:darkred, label="model", subplot=2, linewidth=2)
	vline!([means[2]], color=:black, label="data", subplot=2, linewidth=2)
	plot!(means, [2,2], linestyle=:dash, color=:grey, subplot=2, label="Difference", linewidth=2)
	
	
end

# ╔═╡ 54f6975f-c30b-4e1a-aba6-4e77b6a36ce6
begin
	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))
	
	details(x; summary="Show more") = @htl("""
		<details>
			<summary>$(summary)</summary>
			$(x)
		</details>
		""")
	nothing
end

# ╔═╡ 1055d000-35bf-11ec-2497-55c5e64198c4
 Markdown.parse("
# Introduction

In this introductory tutorial, we will demonstrate how to use various methods to express ACT-R models in analytic form. Analytic models are similar to computational or simulation based models in that they formally describe the mapping between stimulus inputs and behavioral outputs through a function that describes internal cognitive processes. However, analytic and computational models differ along several dimensions, including mathematical formulation, speed and ease of development, model run-time, mathematical tractability and the array of tools and techniques available for each approach. Each of these points of difference will be discussed in more detail in subsequent sections.  

## Background Knowledge 

This tutorial is geared toward readers who have at least some basic knowledge of ACT-R, statistical theory, and computer programming. Requsit background knowledge for ACT-R includes an understanding of the organization and function of the architecture and an understanding of the declarative memory equations. In terms, of statistical knowledge, we assume readers understand basic probability theory, such as joint and conditional probability, likelihood functions and generative models. For programming, we assume that readers can write a function and use  logical operators and loops to control the flow of a program. Readers who lack knowledge in one or more of these areas are encouraged to acquire a basic background before beginning the tutorial. Nonetheless, in the folder ACTRTutorials/Background_Turorials, we provide crashcourse tutorials for basic probability theory, Bayesian Inference, the Julia programming language, and software packages used throughout the tutorial. Additional sources of information for each topic can be found in the references section of each tutorial where applicable. 

- Julia Programming Language
- Stan Programming Language
- Notation
- MCMC Sampling
- ACT-R Models
- Differential Evolution MCMC
- Markov Processes

## Organization of Tutorial

The tutorial is organized around model structures and mathematical techniques, beginning with simple examples and gradually adding complexity. Typically, this will involve specifying a simple model with highly restrictive assumptions, then relaxing various assumptions to create more general and realistic models. The tutorial includes examples from the Lisp ACT-R tutorial, and variants thereof, as well as supplementary examples that provide scaffolding to advance from simple models to more complex models. In the remainder of this tutorial, we will distinguish between computational models and analytic models, compare their strengths and weaknesses, and in so doing, try to make a case that analytic models are a better approach for some applications of ACT-R. "
)

# ╔═╡ c0816507-5db1-4fe0-9dd1-f2f2d5471738
md"
# Computational Models
Computational models are computer simulations of complex systems, consisting of a set of mechanisms that interact according to simple rules to produce emergent behavior. The behavior that emerges from computational models is often complex, dynamic, and nonlinear. Computational models can be found throughout many scientific fields, such as physics, biology, chemistry, climatology, and psychology among others. Much like a statistical model, most computational models are composed of both deterministic and stochastic components, which produce a distribution of predictions. The distribution of predictions for computational models is generated through Monte Carlo simulation rather than mathematical analysis. In order to obtain stable predictions, the simulation is repeated thousands or tens of thousands of times. The resulting output can be compared to human data in order to evaluate the adequacy of the model. In a learning task, for example, reaction time within learning blocks might be averaged across many simulations and compared to the average human reaction times within learning blocks.

Perhaps the simplest demonstration of a computational model is a Monte Carlo simulation of a coin flip. Suppose we want to know the distribution of outcomes resulting from 10 coin flips where the probability of heads is $\theta = .5$. First, we will load the necessary packages for the demonstration:
"

# ╔═╡ f4c1b380-a922-4233-ba3e-4d6088e2e8ca
md"
In the following code block, we will define a function called `simulate!` that will simulate the coin flip process. Although, in principle, we could simulate the physical processes that govern the flip of the coin, we will simulate the coin flip at an abstract level that only concerns the statistical properties of the outcome. The inputs for `simulate!` are the following:

-  `θ`: the probability of heads
-  `n`: the number of trials
-  `sim_data`: an array that will hold the results, 

`sim_data[1]` holds the number of simulated outcomes in which zero heads were observed in 10 trials; `sim_data[2]` holds the number of simulated outcomes in which 1 head was observed in 10 trials and so on.
"

# ╔═╡ 9ae46dae-61c0-4c1d-9938-b0d029d42233
function simulate!(θ, n, sim_data)
    # number of successes
    h = 0
    # simulate n trials
    for t ∈ 1:n
        # increment h if successful
        h += rand() ≤ θ ? 1 : 0
    end
    # update distribution count vector
    sim_data[h + 1] += 1
    return nothing
end

# ╔═╡ f354da57-96f7-403c-999c-da0a48d168c2
md"
In the cell below, the a simulation of 10 coin flips is repeated 1,000 times and the results are plotted as a histogram. You can press the execute button ⊳ located at the bottom right of the cell below to run the simulation again.   
"

# ╔═╡ 61c395db-5b3d-4ab1-a827-e5307fce3eec
begin
	# the number of simulations
	n_sim = 1_000
	# true probability of heads
	θ = 0.5
	# number of trials per simulation
	n = 10
	# initialize counts (0-10 heads) at zero
	sim_data = fill(0, n+1)
	# execute the simulation n_sim times
	map(_ -> simulate!(θ, n, sim_data), 1:n_sim)
	nothing
end

# ╔═╡ 5080f6fd-fab5-4515-b186-52a517e34365
begin
	sim_probs =  sim_data/n_sim
data = fill(0, n + 1)
bar(
	0:n, 
	sim_probs, 
	xticks=0:n, 
	grid=false, 
	color=:grey, 
	alpha=.7, 
	xlabel="H=h heads", 
	ylabel="Probability",
    xaxis=font(12),
	yaxis=font(12),
	leg = false,
	ylims = (0,.3)
)
end

# ╔═╡ e3da61f9-0640-4b43-bceb-867459218c2f
md"
We see that the most probable outcomes tend to cluster around the midpoint of 5 whereas the outcomes toward the extremes become increasingly unlikely. It is possible to increase the precision by increasing the number of simulations and, by definition, the estimate will converge on the true distribution in the limit. However, the trade-off of greater precision is a greater computational burden. 

One important thing to note about the computational model is that it simulated the process one step at a time. This is in juxtaposition to representing the distribution concisely with an equation (although equations can appear in computational models). We did not exploit any theorems or structure of the problem to make simplifications. Moreover, the predictions are approximate rather than exact. 
"

# ╔═╡ 11a38efa-df32-4293-b29f-f5a11323c11e
md"
# Analytic Models

In contrast to computational models, analytic models are fully described mathematically in closed-form rather than a step-by-step simulation of a process. Analytic approaches often involve the use of theorems and rules to represent and simplify model. Perhaps the best way to explain analytic models is to continue with the coin flip example. The analytic expression for the series of independent and identically distributed coin flips is a binomial probability mass function:

$f(h; n, \theta) = {n\choose h} \theta^{h} (1-\theta)^{n-h}$

where:

-  $h \in \{0,1, \dots, n\}$: the number of heads
-  $n$: the number of trials
-  $\theta$: the probability of heads

Rather than simulating the coin flips, the analytic model uses two rules of probability theory to express the predictions in closed-form. According to the independence rule, independent events can be multiplied, as exemplified in the exponents. According to the addition rule, the probability of events that are mutually exclusive can be summed. The addition rule is exemplified in the term  ${n\choose h}= \frac{n!}{h!(n-h)!}$ (read as $n$ choose $h$), which  sums the number of mutually exclusive orderings of $h$ heads and and $n-h$ tails. Taken together, the analytic model provides a succinct description of a coin flip using two established rules from probability theory.

As a frame of reference, we will plot the second simulation against the true probabilities computed from the binomial probability mass function:
"

# ╔═╡ 3d52f0b9-5098-4e2f-a858-e4a916c22b61
begin
	bar(
		0:n, 
		sim_probs, 
		xticks = 0:n, 
		color = :grey, 
		alpha = 0.7, 
		grid = false, 
		label = "Simulation", 
		xlabel = "H=h heads", 
		ylabel = "Probability",
	    xaxis = font(12), 
		yaxis = font(12)
	)
	prob_mass = map(h -> pdf(Binomial(n, θ), h), 0:n)
	bar!(0:n, prob_mass, xticks=0:n, color=:darkred, alpha=0.7, label="Probability Mass")
end

# ╔═╡ 4d34e7a0-a9bd-4ddd-80df-cca58091c1ad
md"
# Computational vs Analytic Models

In this section, we will describe the strengths and weaknesses between computational and analytic approaches. Since many of these strengths and weaknesses involve trade-offs, it is important to carefully consider each approach before model development and choose the method that helps you best achieve your goals. Before we compare computational and analytic models, its worth mentioning that distinction between the two is not always clear cut. It is possible to develop models that combine aspects of both approaches. For example, in subsequent Chapters, we will specify Bayesian versions of ACT-R with analytically defined prior distributions and likelihood functions. However, the posterior distributions will be approximated through Monte Carlo simulation because the solutions are often intractable. Nonetheless, many of the trade-offs we discuss will be present to some degree.
"

# ╔═╡ d20f4e79-3760-4ca5-96bf-fbfc13602554
md"
$\begin{aligned}[]
\begin{array}{llll}
 & \textrm{computational} & \textrm{analytic} \\
\hline
\textrm{approach} & \textrm{simulate algorithm, process} & \textrm{mathematical abstraction} \\
\textrm{predictions} & \textrm{repeated simulation} & \textrm{evaluate equation} \\
  \textrm{data resolution}  & \textrm{average}  &  \textrm{individual responses}  \\
\textrm{analysis} & \textrm{simulate behavior}  & \textrm{proofs, formal analysis} \\
  \textrm{fit metric}  & \textrm{RMSE}, R^2    & \textrm{statistical,log likelihood}   \\
\textrm{fit procedure} & \textrm{grid search, algorithm} & \textrm{algorithm, MLE, Bayesian}\\
\textrm{execution speed} & \textrm{relatively slow} & \textrm{relatively fast} \\
\textrm{development} & \textrm{relatively easy} & \textrm{relatively difficult}\\
\end{array}
\end{aligned}$
"

# ╔═╡ d545a687-6097-4335-ab44-f4708c18cc38
md"
## Applying Models to Data

In this section, we will compare the different ways computational and analytic models are applied to data. 

### Computational Approach

ACT-R is typically applied to data according to the following steps:

1. Define a model in terms of production rules.
2. Generate hundreds or thousands of simulated data sets from the model
3. Select a statistic for comparing the data to the simulated data
4. Use $R^2$ or rmse to assess the fit of the model. 

As an example, suppose that you are building a model of recognition memory. A typical recognition memory experiment involves two phases: a study phase in which subjects learn a list of words, and a test phase in which subjects discriminate between the new words and words from the study list. One statistic a person might use is the mean reaction time for each of the four response categories: respond `old` on a target trial, respond `old` on a foil trial, respond `new` on a target trial, and respond `new` on a foil trial. The figure below illustrates this approach on target trials. The quality of the model is assessed using some sort of difference metric (i.e. rmse). 
"

# ╔═╡ 742b2209-5991-48f6-813a-3d2908496291
md"

### Limitations 

####  Averaging Artifacts

The use of averaged data may distort the results. For example, the average of two qualitatively different patterns may not be sensible.

#### Loss of Information

Some loss of information will occcur by averaging across subjects and/or trials. Additionally, the potentially informative shape of the reaction time distribution is lost in the process of averaging. 

#### Incommensurable DVs 

One challenge is synthesizing DVs with differing scales---in this case, reaction times and response probabilities. Reaction times have the domain $[0, \infty]$ whereas response probabilities have the domain $[0,1]$. Depending on the time units, RMSE will either weight reaction times more or less than response probabilities. One approach is to use normalized RMSE, however this approach implicitly assumes RT and accuracy should be weighted equally. These problems are ellaborated upon further below. 
"

# ╔═╡ e53cffd2-9bdc-4024-84b7-ca66d74e3697
md"
## Analytic Approach

### Likelihood Function
What we need is a method that takes into account the shape of the distribution and accuracy simultaneously. Likelihood functions are one such solution and are a key concept of the analytic approach. In the figure below, simulated RTs are paneled by old and new responses. Values from the likelihood function are superimposed on the histgram as an orange line. The likelihood function is an integral part of the analytic approach, which describes how likely a given response is under the model. Notice that the orange line is higher for some reaction times compared to others, which indicates some values are more likely. You can see how the likelihood function describes the simulated RTs under different parameters by adjusting the slider below. 
"

# ╔═╡ a3b54d77-450e-4c10-885a-dcc4ba193063
blc = @bind blc Slider(-1:.1:2, default=0, show_value=true)

# ╔═╡ d389e329-99b9-413e-8263-ca77ec582e6d
let
	
	font_size = 14
	# create chunks of declarative knowledge
	chunks = [
	    Chunk(;name=:Bob, department=:accounting),
	    Chunk(;name=:Alice, department=:HR)
	    ]
	
	# initialize declarative memory
	declarative = Declarative(memory=chunks)
	
	# specify model parameters: partial matching, noise, mismatch penalty, activation noise
	Θ = (;blc, mmp=true, noise=true, δ=.5, s=.2)  
	
	# create an ACT-R object with activation noise and partial matching
	actr = ACTR(;declarative, Θ...)
	
	# compute activation for each chunk
	compute_activation!(actr; department=:accounting)
	# get mean activation
	μ = get_mean_activations(actr)
	# standard deviation 
	σ = Θ.s * pi / sqrt(3)
	# lognormal race distribution object
	dist = LNR(;μ=-μ, σ, ϕ=0.0)
	
	
	# index for accounting
	idx = find_index(actr; department=:accounting)
	# generate retrieval times
	rts = rand(dist, 10^5)
	# extract rts for accounting
	acc_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
	# probability of retrieving accounting
	p_acc = length(acc_rts)/length(rts)
	# histogram of retrieval times
	hist = plot(layout=(2,1))
	histogram!(
		hist, 
		acc_rts, 
		xlims = (0,3),
		ylims = (0,5),
		color=:grey, 
		leg=false, 
		grid=false,
	    bins = 100, 
		norm=true, 
		xlabel="Reaction Time", 
		ylabel="Density",
	    title="Old", 
		linewidth=1.5, 
		xaxis=font(font_size), 
		yaxis=font(font_size), 
		legendfontsize=10
	)
	# weight histogram according to retrieval probability
	hist[1][1][:y] *= p_acc
	# collection of retrieval time values
	x = 0:.01:2.5
	# density for each x value
	dens = pdf.(dist, idx, x)
	# overlay PDF on histogram
	plot!(hist, x, dens, color=:darkorange, linewidth=2.5, xlims=(0,2.5))
	
	# index for accounting
	idx = find_index(actr; department=:HR)
	# extract rts for HR
	hr_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
	
	histogram!(
		hist, 
		hr_rts, 
		color=:grey, 
		leg=false, 
		grid=false,
	    bins = 100, 
		norm=true, 
		xlabel="Reaction Time", 
		ylabel="Density", 
		subplot=2,
	    title="New", 
		linewidth=1.5, 
		xaxis=font(font_size), 
		yaxis=font(font_size), 
		legendfontsize=10
	)
	# weight histogram according to retrieval probability
	hist[2][1][:y] *= (1 - p_acc)
	# density for each x value
	dens = pdf.(dist, idx, x)
	# overlay PDF on histogram
	plot!(hist, x, dens, color=:darkorange, linewidth=2.5, xlims=(0,2.5), subplot=2)
end

# ╔═╡ fa7a8997-0d57-43c6-93c7-54a8fede4692
md"
### Log Likelihood 

The figure below illustrates how a data are compared to the model using a likelihood function. In this example, each of four data points is represented as a dashed vertical line extending from the x-axis to the orange curve. The height of a dashed line indicates its likelihood under the model, which is known as a density. Greater values indicate better fit of the model to the data. For numerical reasons, the model fit is summarized as the sum of the log densities, which is called a log likelihood.
"

# ╔═╡ 4cfeb763-49d0-476d-b081-ff697f909da6
md"

Using the sliders below, you can adjust the base level constant `blc1` and activation noise `s1` to change the density of the data points (vertical dashed lines). The log likelihood LL is found by summing the log of the densities (i.e. height of vertical dashed lines) which can be computed seperately for old (o) and new (n) and summed together because the log likelihood is additive.

"

# ╔═╡ 94a21c75-f0c2-42e8-92e8-fd112adf3a00
blc1 = @bind blc1 Slider(0:.05:2, default=.4, show_value=true)

# ╔═╡ 697be78b-11cb-48d1-a846-dc690dfab5d7
s1 = @bind s1 Slider(0:.01:.3, default=.15, show_value=true)

# ╔═╡ a8bc022d-00f3-45ab-ac19-05d29731e93a
begin
	# create chunks of declarative knowledge
	chunks = [
	    Chunk(;name=:Bob, department=:accounting),
	    Chunk(;name=:Alice, department=:HR)
	    ]
	
	# initialize declarative memory
	declarative = Declarative(memory=chunks)
	
	# specify model parameters: partial matching, noise, mismatch penalty, activation noise
	Θ = (blc=blc1, mmp=true, noise=true, δ=.3, s=s1)  
	
	# create an ACT-R object with activation noise and partial matching
	actr = ACTR(;declarative, Θ...)
	
	# compute activation for each chunk
	compute_activation!(actr; department=:accounting)
	# get mean activation
	μ = get_mean_activations(actr)
	# standard deviation 
	σ = Θ.s * pi / sqrt(3)
	# lognormal race distribution object
	dist = LNR(;μ=-μ, σ, ϕ=0.0)
	
	
	# index for accounting
	idx = find_index(actr; department=:accounting)
	# generate retrieval times
	rts = rand(dist, 10^5)
	# extract rts for accounting
	acc_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
	# probability of retrieving accounting
	p_acc = length(acc_rts) / length(rts)
	# histogram of retrieval times
	hist = plot(layout=(2,1))
	x = 0:.01:3
	# density for each x value
	dens = pdf.(dist, idx, x)
	# overlay PDF on histogram
	font_size = 12
	
	plot!(
		hist, 
		x, 
		dens, 
		color=:darkorange, 
		linewidth = 2, 
		xlims = (0,1.5), 
		leg = false,
	    title = "Old", 
		grid = false, 
		ylims = (0,5), 
		xlabel="RT (seconds)", 
		ylabel="Density", 
		xaxis=font(font_size), 
		yaxis=font(font_size), 
		legendfontsize=10
	)
	
	# index for accounting
	idx = find_index(actr; department=:HR)
	# generate retrieval times
	# extract rts for accounting
	hr_rts = filter(x->x[1] == idx, rts) .|> x-> x[2]
	# density for each x value
	dens = pdf.(dist, idx, x)
	# overlay PDF on histogram
	plot!(
		hist, 
		x, 
		dens, 
		color=:darkorange, 
		linewidth=2, 
		subplot=2,
	    grid=false, 
		leg=false, 
		xlabel="RT (seconds)", 
		ylabel="Density",
	    title="New", 
		xaxis=font(font_size), 
		yaxis=font(font_size), 
		legendfontsize=10
	)
	
	# add density lines to correct distribution
	x_line1 = [.3,.6,.4,.7]
	density_max1 = pdf.(dist, 1, x_line1)
	density_min1 = fill(0.0, length(x_line1))
	# log likelihood correct
	LLo = logpdf.(dist, 1, x_line1) |> sum

	plot!(
		[x_line1';x_line1'],
		[density_min1';density_max1'], 
		color=:black, 
		subplot=1,
	    linestyle=:dash
	)
	
	
	# add density lines to incorrect distribution
	x_line2 = [.4]
	density_max2 = pdf.(dist, 2, x_line2)
	density_min2 = fill(0.0, length(x_line2))
	# log likelihood incorrect
	LLn = logpdf.(dist, 1, x_line2) |> sum
	plot!(
		[x_line2'; x_line2'], 
		[density_min2';density_max2'], 
		color=:black,
		subplot=2,
	    linestyle=:dash
	)

end

# ╔═╡ 9d683975-24b5-4078-9982-95bd82ed5526
cm"""
<div align="center">

 ``\rm LL_{old} =`` $(round(LLo, digits = 3))

``\rm LL_{new} =`` $(round(LLn, digits = 3))

``\rm LL = LL_{old} + LL_{new} =`` $(round(LLo + LLn, digits = 3))
</div>

"""

# ╔═╡ c2eb39ed-6f09-4f11-85ff-257b7dac45a6
md"
Can you find the parameters that maximize LL?
"

# ╔═╡ 55a1fceb-d4ca-40ea-ac61-bf250d079b9d
	hint(md"Try setting `blc1` near .70 and then adjust `s1`")

# ╔═╡ 00845607-4291-492a-a29d-9994bee25b78
md"

Additional detail on finding the maximum can be found below:
"

# ╔═╡ 5b9e6c29-9c4d-409b-9332-d963560f687d
let
	text = md"""

For those who are curious, the code for finding the parameters that maximize LL is given below. 
```julia
begin
using Optim

function f(x)
		Θ = (blc=x[1], mmp=true, noise=true, δ=.3, s=x[2])  
		# create an ACT-R object with activation noise and partial matching
		actr = ACTR(;declarative, Θ...)
		# compute activation for each chunk
		compute_activation!(actr; department=:accounting)
		# get mean activation
		μ = get_mean_activations(actr)
		# standard deviation 
		σ = Θ.s * pi / sqrt(3)
		# lognormal race distribution object
		dist = LNR(;μ=-μ, σ, ϕ=0.0)
		x_line1 = [.3,.6,.4,.7]
		# log likelihood correct
		LLc = logpdf.(dist, 1, x_line1) |> sum
		x_line2 = [.4]
		# log likelihood incorrect
		LLi = logpdf.(dist, 1, x_line2) |> sum
		return -(LLc + LLi)
end

results = optimize(f, [.6,.2], NelderMead())
Optim.minimizer(results)
end
```
	"""
	details(text)
end

# ╔═╡ 18875fed-49cf-42a1-bba0-f33d6dd91a91
md"

### Parameter Estimation

#### Computational Approaches

Parameters in ACT-R are typically estimated with grid search, with a search algorithm, or even by hand in some cases. Each method suffers from two limitations. First, prior information about plausible parameters is not formally incorporated in the estimation process (although some knowledge may be used to define lower and upper bounds on the parameters). This precludes the ability to incorporate information from previous experiments. Second, the resulting point estimates convey no information regarding uncertainty, which can complicate the interpretation of the model. 

#### Analytic Approaches

Broadly speaking, there are two approaches for parameter estimation for analytic models. One approach is to use an algorithm to find the maximum likelihood estimate---a set of parameters that maximizes the log likelihood of the data with respect to the model.  A second approach is Bayesian parameter estimation. We will focus a bit on this approach because it has many advantages over maximum likelihood estimation. In the Bayesian approach, uncertainty about parameter values is represented as distributions. A diffuse distribution indicates more uncertainty whereas distribution that is concentrated over a smaller range indicates greater certainty about those values. The Bayesian approach begins by specifying a prior distribution, which reflects prior knowledge about the model and data. As additional data are collected, the prior is updated to form what is called a posterior distribution. In essence, the posterior distribution functions like a compromise between the prior distribution and the information provided by the new data. 

The interactive plot below shows the relationship between the prior distribution and posterior distribution after 10 out of 20 coin flips were heads. The initial slider labeled `θp` controls the mean of the prior distribution, and the slider labeled `np` can be thought of as the certainty in the value of `θp` ( technically it is the number of prior observations). The prior is uniform with the initial values. The posterior distribution peaks around .5, which is what we would expect after observing 10 heads out of 20 trials. If `np` is increased, the posterior distribution becomes more concentrated around .5. If `θp` is changed, both the prior and posterior shift. What is important is (1) the ability to incorporate prior knowledge and (2) to decribe uncertainty in parameter estimates in terms of probability distributions. 

"

# ╔═╡ 2e64eb26-8b5e-4eba-a891-668da418c817
θp = @bind θp Slider(.01:.01:.99, default=.5, show_value = true)

# ╔═╡ 5ee3af0c-fe5b-45fa-8f2a-483f6d9c0c32
np = @bind np Slider(1:100, default=2, show_value = true)

# ╔═╡ 7647aecd-745d-4e29-a7e9-7af9a74f86f5
let
	n = 20
	h = 10
	θ = .5
	α = θp * np
	β = (1 - θp) * np
	θs = range(0, 1, length=150)
	y_prior = pdf.(Beta.(α, β), θs)
	
	plot(
		θs, 
		y_prior, 
		grid = false, 
		xlabel = "θ", 
		ylabel = "density", 
		linewidth = 2.5,
		xlims = (0, 1),
		ylims = (0,10),
		label = "prior",
		title = "Prior vs. Posterior Distribution of θ"
	)

	y_posterior = pdf.(Beta.(α + h, β + n - h), θs)
	plot!(
		θs, 
		y_posterior, 
		linewidth = 2.5,
		label = "posterior"
	)
end

# ╔═╡ ad5e7c5f-7b3b-41b4-b7c7-3c85931718f1
md"
Another important feature of Bayesian parameter estimation is the ability to incorporate uncertainty about parameters into predictions using the posterior predictive distribution. The posterior predictive distribution has two sources of variability: (1) irreducible noise in the model and (2) variability in the posterior distribution of the parameters. Non-Bayesian techniques do not include parametric uncertainty in their predictions, resulting in over confidence. 
"

# ╔═╡ 8e30281e-6535-49f2-b48b-b4e0dce50df1
let
	n = 20
	θ = .5
	α = θp * np
	β = (1 - θp) * np
	n_sim = 10000
	h_vals = 0:20
	pmf_1 = pdf.(Binomial(n, θ), h_vals)
	h1 = bar(
		h_vals,
		pmf_1, 
		xlims = (0, 20),
		ylims = (0, .3),
		norm = true, 
		grid = false,
		leg = false,
		title = "sampling distribution",
		xlabel = "H = h",
		ylabel = "density",
	)

	sd_sd = std(Binomial(n, θ))
	annotate!(h1, 16, .25, text("sd = $(round(sd_sd, digits=2))", :black, 12))

	pmf_2 = pdf.(BetaBinomial(n, α + θ * n, β + (1 - θ) * n), h_vals)
	
	h2 = bar(
		h_vals,
		pmf_2,
		xlims = (0, 20),
		ylims = (0, .3),
		norm = true, 
		grid = false,
		leg = false,
		title = "posterior predictive distribution",
		xlabel = "H = h",
		ylabel = "density",
	)
	
	sd_pd = std(BetaBinomial(n, α + θ * n, β + (1 - θ) * n))
	annotate!(h2, 16, .25, text("sd = $(round(sd_pd, digits=2))", :black, 12))
	
	plot(h1, h2, layout = (2,1))
end

# ╔═╡ f6643ef4-10ed-447f-81ea-6eda5b4b6cf8
md"
Each plot above shows the predictions for the number of heads in a binomial model. The top is the posterior predictive distribution, which is more diffuse due to uncertainty in the parameter $\theta$. The second panel shows the predictive distribution with $\theta$ fixed at its maximum likelihood estimate. Compared to the posterior predictive distribution, the distribution is less diffuse, indicating overconfidence. 

You can adjust the sliders to see the effect the prior distribution has on the posterior predictive distribution. The slider labeled θp is the mean of the prior distribution for $\theta$. A value of $.1$, for example, indicates that one tends to think heads in an unlikely outcome. The slider labeled `np` adjusts a parameter that represents prior confidence in the value `θp`. It can be interpreted as the number of past observations you have experienced. Notice that as you increase `np` with `θp = .5`, the standard deviation of the posterior distribution approaches that of the sampling distribution. What this means is that uncertainty about $\theta$ plays a smaller and smaller role because the prior distribution is highly concentrated on $\theta = .5$, which is the value assumed to be true in the sampling distribution.
"

# ╔═╡ a7cb024b-98a1-4d38-92b0-ac5ca1848ea5
md"
### Hierarchical Structures

Models developed within a statistical framework can be extended to accommodate hierarchical data structures. One common hierarchical structure involves multiple groups based on an experimental condition or demographic characteristic. A hierarchical model can simultaneously analyze the groups and individuals within each group. In addition to analyzing at multiple levels, hierarchical models can improve inference through sharing of information across individuals within a group. The rationale is that individuals within a group are more similar to each other than to individuals in other groups. Because each individual contributes to and is constrained by the group level distribution, individuals mutually inform each other. This has a desirable property of attenuating the influence of outliers through a process called shrinkage. 

Hierarchical models are beneficial because they fall somewhere between to extremes. At one extreme, the model is applied to each individual separately under the assumption that individuals are completely independent of each other. At the other extreme, data from individuals are aggregated under the assumption they are replicates of each other. Hierarchical models allow partial dependence between individuals and allow the data to influence the degree of dependence. 
"

# ╔═╡ 64f3b8ab-c1ee-4696-9ae1-4a868ae93719
md"

## Mathematical Analysis

One benefit of analytic models is that they are more amenable to various mathematical analyses. One application, for example, is the use of a classification system based on the structure and properties of a model. If a model belongs to a certain class, it might make a specific prediction or theoretical statement that is common across all of its members. In addition, models that belong to a certain class might be amenable to certain techniques or approximations that prove to be computationally efficient. Another use of mathematical analysis is showing that one model is (or is not) a transformation of another model that is seemingly different. 

In some cases, assessing the mathematical properties of a model can be more informative than typical model fitting techniques. One problem is that the predictions may rely on auxiliary assumptions that are not important to the underlying theory, but influence the predictions nonetheless. Another problem is that that model comparison techniques focus on relative fit to data. It is possible that none of the models are adequate, but one turns out to be `less bad`. Some models under consideration might be at odds with basic properties found in the human data. 

One example of the use of mathematical properties is Systems Factorial Technology Townsend,, & Nozawa,(1995). SFT is a system of methods and tests for distinguishing between basic properties of cognition, such as  architecture (parallel or serial), stopping rule (e.g. minimum or maximum processing time), stochastic dependence and workload capacity. Consider a situation in which an ACT-R model predicts several processes operate in parallel, but the data show support serial processing. This would provide strong evidence against the model, even if it performed better relative to other models. A new model with serial processes would need to be developed. 

## Tractability 
Roughly speaking, tractability refers to ease with which a model can be understood and applied. It is important to factor tractability into your modeling endeavor and weight it against other factors. The tractability of computational and analytic models varies on a case by case basis. There are cases in which the analytic version of a model is more tractable or can be made more tractable through the use of approximations. One example of increased tractability with the analytic approach is the hybrid approximation (Petrov, 2006) for base-level memory activation. The hybrid approximation more tractable mathematically (and computationally efficient) (Fisher, Houpt, Gunzlemann, 2018) while producing very similar results to the exact equation. Another example is the stick building task model from the Lisp tutorial (see the Utility Learning Chapter in this tutorial). In some ways, the analytic model is simpler because many of the production rules and their implementation details can be ignored. What matters more in this particular case is the timing of events in the data and the options form which the model can choose.  In other cases, however, seemingly simple computational models obscure complex statistical details. This tends to happen when the model is characterized by evolving latent processes that have a many-to-one maping to the same observable response. 

Its worth noting that some ACT-R models might be too complex to derive analytic solutions, at least without developing approximations (which can be time consuming in and of itself). In such cases, using a two stage development and validation process might be advisable. The first stage would involve the initial validation of the computational version of the model. If the first stage looks promising, it might be worthwhile to further test the model within an analytic framework. 
"

# ╔═╡ 7a0e48f9-f17b-4d59-85bd-2af60394e23c
md"

## Speed

One potential drawback with computational models is slow run time compared to analytic models. One reason that analytic models run faster is because they concisely summarize the behavior of the model. Another limitation with computational models is the inherent trade-off between precision and computational burden. Increasing the number of simulations improves precision at the expense of run time. If the model was simulated an infinite number of times, it would, by definition, approach the analytic solution, but, of course, the simulation would never terminate. This limitation can be circumvented if an analytic solution can be derived.  

In the following cell, we will run a benchmark to illustrate the relationship between the number of simulations and run time for the coin flip model. The benchmark compares the mean time to evaluate the likelihood of 100 data points for the analytic likelihood and the computational approximation with 1,000 simulated data points and 10,000 simulated data points. 
"

# ╔═╡ 297ee779-76d6-4227-8470-105714e766f3
let
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
	    map(x -> simulate!(.5, n, sim_data), 1:n_sim)
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


	n_reps = 10^4
	n = 10
	θ = .5
	
	data = rand(Binomial(10, .5), 100)
	
	time_analytic = @elapsed map(x -> analytic(data, n, θ), 1:n_reps)
	time_analytic /= n_reps
	time_computational = fill(0.0, 2)
	time_computational[1] = @elapsed map(x -> computational(data, θ, n, 1_000), 1:n_reps)
	time_computational[1] /= n_reps
	time_computational[2] = @elapsed map(x -> computational(data, θ, n, 10_000), 1:n_reps)
	time_computational[2] /= n_reps
	
	
	bar(["analytic"  "1k simulations"  "10k simulations"], [time_analytic  time_computational...], 
	     fillcolor=[:darkred :grey :darkgreen], fillrange=1e-5, alpha=.7,grid=false, xaxis=font(10), yaxis=font(10), 
	     leg=false, yscale=:log10, ylabel="Seconds (log10)", xrotation=20)
end

# ╔═╡ 04535867-baa9-4790-a126-b8e22bf02f1a
md"
In prior research, we have found that the increase in execution time can be even larger in some cases. With an ACT-R model of the fan effect, we found a speed up of 1-3 orders of magnitude depending on the amount of data and the number of simulations used for the computational model (Fisher, Houpt, Gunzelmann, 2020). Although timing benchmarks can be situation specific, in general we expect that analyic models will execute more quickly. 
"

# ╔═╡ 9bab4031-5145-49b5-8c65-72dad83c846f
md"

## Adaptive Design Optimization

A final benefit of using likelihood functions to describe a model is the ability to perform adaptive design optimization. Adaptive design optimization is a Bayesian procedure of iteratively changing design parameters of an experiment to either (1) maximize the discriminability between two models, or (2) to reduce the uncertainty in the posterior distributions of a single model. Adaptive design optimization is suitable for models whose predictions vary as a function of experimental design parameters. Some examples include, decision making models, which vary as a function of outcome distributions, psychophysical models, which transform physical units into psychological units, and memory models in which retrieval depends on stimulus properties, practice or timing. In Unit 12, we demonstrate how to select practice repetitions and test delays with adaptive design optimization to improve parameter estimation for an ACT-R memory model. 
"

# ╔═╡ 8f4964b7-67d9-488c-8a3b-11f9c4d545d2
md"
# References

Fisher, C. R., Houpt, J., & Gunzelmann, G. (2018). A Comparison of Approximations for Base-Level Activation in ACT-R. Computational Brain & Behavior, 1(3-4), 228-236.

Fisher, C. R., Houpt, J. W., & Gunzelmann, G. (2020). Developing memory-based models of ACT-R within a statistical framework. Journal of Mathematical Psychology, 98, #102416. (DOI: 10.1016/j.jmp.2020.102416)

Petrov, A. A. (2006). Computationally efficient approximation of the base-level learning equation in ACT-R. In Proceedings of the seventh international conference on cognitive modeling (pp. 391-392). Edizioni Goliardiche: Trieste, ITA.

Townsend, J. T., & Nozawa, G. (1995). Spatio-temporal properties of elementary perception: An investigation of parallel, serial, and coactive theories. Journal of mathematical psychology, 39(4), 321-359.
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ACTRModels = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
CommonMark = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
ACTRModels = "~0.10.11"
CommonMark = "~0.8.12"
Distributions = "~0.25.93"
HypertextLiteral = "~0.9.4"
Optim = "~1.7.5"
Plots = "~1.38.12"
PlutoUI = "~0.7.51"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "2ddb4b588f1b2bd897e031b6088fdf47026db304"

[[deps.ACTRModels]]
deps = ["ConcreteStructs", "Distributions", "Parameters", "Pkg", "PrettyTables", "Random", "Reexport", "SafeTestsets", "SequentialSamplingModels", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "1ed6d5235db0007fc4851c5b95b9f8c1019c3f22"
uuid = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
version = "0.10.11"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "917286faa2abb288796e75b88ca67edc016f3219"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.5"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
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
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonMark]]
deps = ["Crayons", "JSON", "PrecompileTools", "URIs"]
git-tree-sha1 = "532c4185d3c9037c0237546d817858b23cf9e071"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.8.12"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "5eeb2bd01e5065090ad591a205d8cad432ae6cb6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.93"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "3cce72ec679a5e8e6a84ff09dd03b721de420cfe"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.0.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "6604e18a0220650dbbea7854938768f15955dd8e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.20.0"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "efaac003187ccc71ace6c755b197284cd4811bfe"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4486ff47de4c18cb511a0da420efebb314556316"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.4+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "41f7dfb2b20e7e8bf64f6b6fae98f4d2df027b06"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "099e356f267354f46ba65087981a77da23a279b7"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.0"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

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
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

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
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "a89b11f0f354f06099e4001c151dffad7ebab015"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.5"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d03ef538114b38f89d66776f2d8fdc0280f90621"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.12"

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
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "6d7bb727e76147ba18eed998700998e17b8e4911"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.4"
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

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SafeTestsets]]
deps = ["Test"]
git-tree-sha1 = "36ebc5622c82eb9324005cc75e7e2cc51181d181"
uuid = "1bc83da4-3b8d-516f-aca4-4fe02f6d838f"
version = "0.0.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SequentialSamplingModels]]
deps = ["ConcreteStructs", "Distributions", "Interpolations", "KernelDensity", "PrettyTables", "Random"]
git-tree-sha1 = "cb3126b7eeb55d7b8e8f8793876932987f8293aa"
uuid = "0e71a2a6-2b30-4447-8742-d083a85e82d1"
version = "0.2.0"

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
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "8982b3607a212b070a5e46eea83eb62b4744ae12"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.25"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

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
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

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
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

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
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

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
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─c40c7ada-454a-4fa3-a71a-4d85e92fec03
# ╟─54f6975f-c30b-4e1a-aba6-4e77b6a36ce6
# ╟─1055d000-35bf-11ec-2497-55c5e64198c4
# ╟─c0816507-5db1-4fe0-9dd1-f2f2d5471738
# ╟─f4c1b380-a922-4233-ba3e-4d6088e2e8ca
# ╠═9ae46dae-61c0-4c1d-9938-b0d029d42233
# ╟─f354da57-96f7-403c-999c-da0a48d168c2
# ╠═61c395db-5b3d-4ab1-a827-e5307fce3eec
# ╟─5080f6fd-fab5-4515-b186-52a517e34365
# ╟─e3da61f9-0640-4b43-bceb-867459218c2f
# ╟─11a38efa-df32-4293-b29f-f5a11323c11e
# ╟─3d52f0b9-5098-4e2f-a858-e4a916c22b61
# ╟─4d34e7a0-a9bd-4ddd-80df-cca58091c1ad
# ╟─d20f4e79-3760-4ca5-96bf-fbfc13602554
# ╟─d545a687-6097-4335-ab44-f4708c18cc38
# ╟─5927a78e-6234-4b51-a1ce-9df2ca57e862
# ╟─742b2209-5991-48f6-813a-3d2908496291
# ╟─e53cffd2-9bdc-4024-84b7-ca66d74e3697
# ╟─d389e329-99b9-413e-8263-ca77ec582e6d
# ╟─a3b54d77-450e-4c10-885a-dcc4ba193063
# ╟─fa7a8997-0d57-43c6-93c7-54a8fede4692
# ╟─a8bc022d-00f3-45ab-ac19-05d29731e93a
# ╟─9d683975-24b5-4078-9982-95bd82ed5526
# ╟─4cfeb763-49d0-476d-b081-ff697f909da6
# ╟─94a21c75-f0c2-42e8-92e8-fd112adf3a00
# ╟─697be78b-11cb-48d1-a846-dc690dfab5d7
# ╟─c2eb39ed-6f09-4f11-85ff-257b7dac45a6
# ╟─55a1fceb-d4ca-40ea-ac61-bf250d079b9d
# ╟─00845607-4291-492a-a29d-9994bee25b78
# ╟─5b9e6c29-9c4d-409b-9332-d963560f687d
# ╟─18875fed-49cf-42a1-bba0-f33d6dd91a91
# ╟─7647aecd-745d-4e29-a7e9-7af9a74f86f5
# ╟─2e64eb26-8b5e-4eba-a891-668da418c817
# ╟─5ee3af0c-fe5b-45fa-8f2a-483f6d9c0c32
# ╟─ad5e7c5f-7b3b-41b4-b7c7-3c85931718f1
# ╟─8e30281e-6535-49f2-b48b-b4e0dce50df1
# ╟─f6643ef4-10ed-447f-81ea-6eda5b4b6cf8
# ╟─a7cb024b-98a1-4d38-92b0-ac5ca1848ea5
# ╟─64f3b8ab-c1ee-4696-9ae1-4a868ae93719
# ╟─7a0e48f9-f17b-4d59-85bd-2af60394e23c
# ╟─297ee779-76d6-4227-8470-105714e766f3
# ╟─04535867-baa9-4790-a126-b8e22bf02f1a
# ╟─9bab4031-5145-49b5-8c65-72dad83c846f
# ╟─8f4964b7-67d9-488c-8a3b-11f9c4d545d2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
