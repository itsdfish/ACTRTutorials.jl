### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 14666c46-31be-470a-b989-361e828260b4
begin
	using PlutoUI, Distributions, Plots, Random
	using QuadGK
	Random.seed!(454)
	TableOfContents()
end

# ╔═╡ 815269f0-6ecf-11ec-3c92-07db775981f9
md"
# Introduction

In this tutorial, we provide an overview of basic probability theory. This tutorial should be regarded not as a comprehensive source, but rather as a refresher. More extensive treatments can be found in the references as well as other sources. 

# Probability Theory

In probability theory, all events belong to a universal set, $\Omega$, also called a sample space. The events that comprise $\Omega$ are context-dependent. For example, consider the role of a standard six-sided die. In this context, $\Omega$ consists of all six outcomes of a single roll of a standard six-sided die: $\Omega = \{1,2,3,4,5,6\}$. It should be somewhat intuitive that the probability of any events in $\Omega$ is 1. The universal set $\Omega$ can be divided into sub-sets to describe a specific combination of events. For example, $B = \{1,3,5\}$ is the set of rolls resulting in an odd number. 

Throughout, we will use upper-case letters to represent random variables and lower-case letters to represents specific realizaations of a random variable. For example, $N$ a random variable representing the outcome of rolling a die, and $n$ is a specific value that $N$ can assume. Assuming that each outcome is equally likelily, the probability of observing $n$ is given by: 

$\Pr(N=n) = \frac{1}{|\Omega|} = \frac{1}{6}$

where $|\Omega|$ is the number of elements in $\Omega$.


## Joint Probability

Joint probability refers to the probability of two or more events occuring together. For example, what is the probability of rolling a 1 followed by a 2? In this case, the universal set is the set of all 36 pairs of outcomes: $\Omega = \{(1,1),(1,2) \dots (6,6)\}$. Note that each pair is order dependent. Now that we have defined the universal set, we can define the probability of rolling a 1 followed by a two:

$\Pr(N_1=1 \wedge N_2=2) = \frac{1}{6}\frac{1}{6} = \frac{1}{36} = \frac{1}{|\Omega|}$

where indices 1 and 2 refer to the first and second rolls of the die. The probability of each event is multiplied because the events are independent. One way to think about this is that there are 36 possible outcomes. In 1 out 6 outcomes a 1 will occur, and out of those outcomes, 1 in 6 of those outcomes, a 2 will occur. 

Consider a different case in which the order of rolling a 1 and a 2 does not matter. The probability of rolling a 1 and 2 in either order is the sum of the probability of each order. We can sum the probabilities because the events are mutually exclusive (they cannot occur together). Formally, the probability of rolling a 1 and a 2 in either order is given by:

$\begin{align}
\Pr((N_1=1 \wedge N_2=2) \vee (N_1=2 \wedge N_2=1)) = \\\Pr(N_1=1 \wedge N_2=2) + \Pr(N_1=2 \wedge N_2=1) = \\ \frac{1}{36} + \frac{1}{36} = \frac{1}{18}
\end{align}$
## Conditional Probability

A conditional probability provides a way to assign a probability based on additional information. This means that the probability of an event is evaluated with respect to a particular set rather than the entire sample space. A general expression for the probability $X=x$ given that $Y=y$

$\Pr(X=x \mid Y =y) = \frac{\Pr(X=x \wedge Y=y)}{\Pr(Y=y)}$

This means we take the joint probability of $X=x$ and $Y=y$ and divid it by the probability $Y=y$. For example, what is the probability that a die lands on 1 given that the outcome is less than 3? Let $N$ be a random variable that represents the number that the die lands on. The conditional probability is defined as:

$\Pr(N=1 \mid N < 3) = \frac{\Pr(N=1 \wedge N < 3)}{\Pr(N<3)}$

$\Pr(N=1 \mid N < 3) = \frac{\frac{1}{6}}{\frac{2}{6}} = \frac{1}{2}$

As we will discussed in more detail below, the event $N=1$ and $N < 3$ are dependent. The unconditional probability that $N = 1$ is $1/6$, but knowing $N<3$ changed the probability of $N=1$. Conditioning on the additional piece of information that $N < 3$, increased the probability because it decreased the sample space relative to the event $N=1$. In some cases, conditioning can reduce the probability compared to the unconditional probability. In both cases, the conditioning event provides information that allows us to define the probability more accurately. In the next section, we will discuss cases in which a conditioning event is not informative (i.e. does not change the probability).

### Independence and Dependence

Before conditioning on the even $N<3$ the $p(N=n) = \frac{1}{6}$. However, knowing that $N <3$ provides additional information, leading to a different probability. If the events X and Y are dependent, then 

$\Pr(X=x \mid Y =y) \neq \Pr(X=x)$

Otherwise, the events are independent:

$\Pr(X=x \mid Y =y) = \Pr(X=x)$

In the section above, we saw that when two events are independent, their joint probability can be calculated as the productive of the individual probabilities. This follows from the conditional probability:

$\Pr(X=x \mid Y =y) = \frac{\Pr(X=x \wedge Y=y)}{\Pr(Y=y)}$

Divide both sides by $P(Y=y)$

$\Pr(X=x \mid Y =y)\Pr(Y=y) = \Pr(X=x \wedge Y=y)$

Assume $\Pr(X=x \mid Y =y) = \Pr(X=x)$

By substitution

$\Pr(X=x)\Pr(Y=y) = \Pr(X=x \wedge Y=y)$
"

# ╔═╡ d3749ccf-20be-4a2f-910a-944859a997e6
md"
## Law of Total Probability

The law of total probability is a general law, but we will present it in terms of hypotheses and data because it it has implications for Bayes' theorem, which is presented in the next section. Suppose that we have a set of $k$ hypotheses $\mathcal{H} = \{H_1,\dots H_k\}$ and vector of observations $\mathbf{Y}= \{y_1,\dots, y_n\}$ According to the law of total probability, the probability of observing data $\mathbf{Y}$ can be represented as a function of each hypothesis $H_i$:

Before presenting the law of total probability, it will be helpful to provide an intuition for the concept. The figure below provides a visual illustration of the law of total probability . The universal set $\Omega$ is represented as the blue rectagle and the data $\mathbf{Y}$ is represented as the grey rectangle. $\Omega$ is split into three mutually exclusive hypotheses: $H_1, H_2$ and $H_3$. As the diagram shows, event $\mathbf{Y}$ is composed of the part of $\mathbf{Y}$ that intersects with $H_1$, plus the part of $\mathbf{Y}$ that intersects with $H_2$, plus the part of $\mathbf{Y}$ that intersects with $H_3$.
"

# ╔═╡ 9837348a-842c-421f-9ec2-664c766bc572
let
	url = "https://i.imgur.com/kPuyjPw.png"
	data = read(download(url))
	PlutoUI.Show(MIME"image/jpg"(), data)
end

# ╔═╡ 422eda59-3883-4e04-ad79-e991ef1bb4a8
md"
Formally, the law of total probability is stated as:

$\Pr(\mathbf{Y}) = \sum_i^N \Pr(\mathbf{Y} \wedge H_i=\textrm{true})$

In other words, the law of total probability states that we can break down $\mathbf{Y}$ into mutually exclusive and exhaustive sub-sets that can be summed to form $\Pr(\mathbf{Y})$. The law of total probability can be expressed equivalently as:

$\Pr(\mathbf{Y}) = \sum_i^N \Pr(\mathbf{Y} \mid H_i=\textrm{true}) \Pr(H_i=\textrm{true})$

This version follows from the relationship between conditional and join probabilities (i.e. multiply both sides of $\Pr(X=x \mid Y =y) = \frac{\Pr(X=x \wedge Y=y)}{\Pr(Y=y)}$ by $\Pr(Y=y)$.
"

# ╔═╡ 7bbe6b82-878d-4e79-b5ba-3909eaa525f4
md"
## Bayes' Theorem

Bayes' theorem describes the relation between a conditional probabilities $\Pr(A \mid B)$ and $\Pr(B \mid A)$. Although the mathematical relationship is quite simple, its implications are far-reaching and form the foundation of Bayesian statistics. Bayesian statistics is an approach to statistics based on representing beliefs as a probability distribution and updating the probability distribution as new information is encountered. As you might suspect, Bayes' theorem is particularly useful for hypothesis testing, model comparison, and parameter estimation. 

Continuing with our example about hypotheses and data, suppose we want to use Bayes' theorem to assess the probability of a hypothesis in light observed data. Our first step is to define the prior probability of each hypothesis $i$: $\Pr(H_i=\textrm{true})$. This reflects our beliefs about each hypothesis before taking into account $\mathbf{Y}$. The second step is to define the probability of $\mathbf{Y}$ under each hypothesis, which is sometimes called a likelihood. For example, $\Pr(\mathbf{Y} \mid H_i=\textrm{true})$ represents the probability of $Y$ assuming $H_i$ is true. Now that the priors and likelihoods are defined, we can use Bayes' theorem to compute to update the prior probability of each $H_i$ to form a posterior probability.Formally, ehe probability of hypothesis $i$ given $\mathbf{Y}$ is defined according to Bayes' theorem as:

$\Pr(H_i= \textrm{true} \mid \mathbf{Y}) = \frac{\Pr(\mathbf{Y} \mid H_i= \textrm{true}) \Pr(H_i= \textrm{true} )}{\Pr(\mathbf{Y})}$

The denominator $\Pr(\mathbf{Y})$ is the probability of the data regardless of what hypothesis might be true. $\Pr(Y)$ is not usually known directly, but can be computed using the total law of probability. Thus, we can rewrite Bayes theorem as:

$\Pr(H_i= \textrm{true} \mid \mathbf{Y}) = \frac{\Pr(\mathbf{Y} \mid H_i= \textrm{true}) \Pr(H_i= \textrm{true} )}{\sum_{j}^k\Pr(\mathbf{Y} \mid H_j= \textrm{true}) \Pr(H_j= \textrm{true} )}$

The figure below illustrates how conditioning in Bayes'theorem works: we can focus on the grey rectangle because we know that data $\mathbf{Y}$ was observed, rather than some other data set in the blue area of the figure above. The figure below splits the new sample space in grey into three parts: one for $H_1$, one for $H_2$ and one for $H_3$. According to the figure above, $H_i$ has the highest prior probability. However, the posterior probability of $H_3$ is greater than that of $H_1$ because the data are more consistent with $H_3$, as indicated by the higher degree of overlap. 

Bayes' theorem is useful because it tells us what we typically want to know: the probability distribution over a set of hypothesis in light of some observed data. The same logic can be applied to models such as ACT-R or the parameters of a model. More information about Bayesian inference can be found [here](Bayesian_Inference.ipynb). 

"

# ╔═╡ ffb3b928-9b2e-444a-a79d-5bc64ebcac11
md"
## Generative Models

As the name implies, a generative model generates simulated data from the model. Traditionally, ACT-R models are expressed as generative models because the output is simulated data which are compared to empirical data. We can express a generative model as a sampling statement. For example, suppose $k$ is a binomially distributed random variable (such as a coinflip or response probablity that is the same across trials). We can express $k$ as

$k \sim \textrm{Binomial}(n, \theta)$

where $\sim$ means distributed as, $n$ is the number of trials, and $\theta$ is a parameter for the probability of success. In the next code block, we will generate simulated data after loading the required packages.
"

# ╔═╡ 314c3674-89fa-4816-997f-916114237f06
begin
	n = 10
	θ = .3
	k = rand(Binomial(n, θ))
end

# ╔═╡ b66ef8ac-bf16-46aa-b14c-7c830fccae62
md"""
## PMF

A probability mass function (PMF) is a function that assigns probabilities to discrete data given some set of parameters $\theta$. Importantly, a PMF is a function of data with fixed parameters. In addition, a PMF is a probability function because the sum of probabilities across all possible data patterns is 1.

For example, the PMF for a binomially distributed randon variable is given by  

$f(k \mid \theta; n) = \Pr(K=k) = {n\choose k} \theta^k(1-\theta)^{n-k}$

where $k$ is the number of target events out of $n$ trials,  $\theta$ is probability of the target event, and ${n\choose k}$ is the binomial coefficient which represents the number of ways in which the event $k$ events can occur in $n$ trials. Note that $f$ is a function of $k$ and the symbol $\mid$ indicates that the function depends on the random variable $\theta$. We distinguish the fixed value $n$ from the random variable $\theta$ with the separator "$;$"

It is possible to calculate the PMF with the following code: 
"""

# ╔═╡ 2b1651cb-1eec-4a23-8c15-c239643ea5cc
pdf(Binomial(n, θ), k)

# ╔═╡ 1316792e-7af4-4269-93a5-fe358508fd24
md"
The following code plots the Binomial probability mass distribution as a function of $k$ with $\theta = .3$. As expected, the greatest probability mass is found at $k=2$, which is the expected value of random variable $K$. Note that this distribution sums to 1 because it is a probability distribution.
"

# ╔═╡ 0aa7d553-15c8-44f6-a79b-b29173724125
begin
	x = 0:n
	p = pdf.(Binomial(n, θ), x)
	bar(x, p, leg=false, grid=false, xlabel="k", ylabel="Probability", xaxis=font(12), yaxis=font(12), size=(600,400),
	    color=:grey)
end

# ╔═╡ 69eb4e92-3145-444d-bc91-628820b5d010
md"
Let's confirm that the binomial PMF is a probability distribution by summing the probabilities across k. What we should find is that

$\Pr(K=0) + \Pr(K=1) + \cdots + \Pr(K=10) = 1$

In the code below, we see that the sum across all values of $k$ is indeed 1 (with some float point error).
"

# ╔═╡ 0e83c830-1fde-4b80-9b74-e9a79cb6db6e
sum(pdf.(Binomial(n, θ), 0:10))

# ╔═╡ 56bb2a42-0216-481f-807b-7c8006ceadf8
md"
### Likelihood Function 

Whereas a PMF is a function of the data, a likelihood function is a function of a model parameter. The reason this distinction is important is because the likelihood function is not a probability function. The integral over the parameters is not necessarily 1. The likelihood function for a binomial random variable is given by:

$f(\theta; n, k) = {n\choose k} \theta^k(1-\theta)^{n-k}$

In the plot below, the likelihood of the data $k=3$ is plotted as a function of parameter $\theta$. As you might expect, the max of the likelihood is at $\theta=.2$ because 3 successes were observed in 10 trials. The value of $\theta$ corresponding to mode or max of the plot below is known as the maximum likelihood estimate: $\theta_{\textrm{mle}} = \frac{k}{n} = \frac{3}{10} = .3$.
"

# ╔═╡ d6ae1f70-5f2e-4691-911c-c030e61b62f3
begin
	θs = .0:.01:1
	dens = @. pdf(Binomial(n, θs), k)
	plot(θs, dens, leg=false, grid=false, xlabel="θ", ylabel="Likelihood", xaxis=font(14), yaxis=font(14), size=(600,400),
	    color=:darkblue, linewidth=2)
end

# ╔═╡ ea79096d-bbac-4abd-8488-50bf7aaade40
md"

We can confirm that the binomial likelihood function is not a probability function with the code below:
"

# ╔═╡ 2d1e6d92-1636-47c3-a746-5aac30145196
let
	integral,_ = quadgk(x->pdf(Binomial(n, x), k), 0, 1)
	integral
end

# ╔═╡ 1af5aabd-6b0f-4a47-847a-294c102a5986
md"
## PDF

The continous analogue to the PMF is the probability density function (PDF). As with a PMF, a PDF is a probability function is a function of the data with fixed parameters, and thus integrates to 1. Why are PDFs used for continuous random variables? The reason is that the probability of a specific value of a continuous random variable is vanishingly small. Consider the probability of $x = 1$. Does that mean exactly $1$ to an arbitrary decimal point? Or perhaps does it mean $1 + 10^{-40}$ or even $1 + 10^{-1000}$? Instead, we must think about the ratio of a probability divided by the width of an interval around $1$; hence, the term density. Rather that choosing a specific value value for the interval around $1$, we use calculus instead to compute the ratio as the width of the interval approaches zero. This gives us a probability density, or density for short.  Densities are non-negative.


The PDF can be generically defined as:


$f(y) = \frac{dF(y)}{dy}$

where $F(y)$ is the CDF as defined in the following section. 

Suppose $Y$ follows a Normal distribution: $Y \sim \textrm{Normal}(0,1)$. The PDF of a specific value of $Y$ value is written as:

$f(y \mid \mu=0,\sigma=1) = \frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-1}{2}(\frac{y-\mu}{\sigma})^2}$

Suppose, for example, $y=.5$. In this case, the PDF of this value can be calculated with the following code:
"

# ╔═╡ dc44c97d-4825-48cd-a4df-ce550c6e6581
let
	μ = 0.0
	σ = 1.0
	y = .5
	dens_y = pdf(Normal(μ, σ), y)
end

# ╔═╡ ee94d261-dd31-49ec-8bdf-6616bae4af01
md"
The figure below plots the density of the standard normal distribution as a function of data point $y$. The observed value of $y = .5$ defined above is represented as the dashed vertical line. Note that this distribution integrates to 1 and is thus a probability distribution.
"

# ╔═╡ db92f860-d815-43bc-9625-2f41b62d4af1
let
	x = -3.5:.01:3.5
	μ = 0.0
	σ = 1.0
	y = .5
	dens_y = pdf(Normal(μ, σ), y)
	dens = pdf.(Normal(μ, σ), x)
	plot(x, dens, leg=false, grid=false, xlabel="y", ylabel="Density", xaxis=font(14), yaxis=font(14), size=(600,400),
	    color=:grey, linewidth=2)
	plot!([y;y], [0;dens_y], color=:black, linestyle=:dash)
end

# ╔═╡ ade5f6a3-e5e7-46a3-9ae7-7a20b15100a3
md"
### Likelihood Function 
As with the Binimial distribution, it is possible to plot the likelihood of parameters of the normal distribution with data $y=.5$ fixed. Again this is not a probability distribution because it does not integrate to 1.
"

# ╔═╡ 29df83ab-cf6e-43df-98de-e19114514e8d
let 
	σ = 1.0
	y = .5
	μs = -3.5:.01:3.5
	x = μs
dens_μ = @. pdf(Normal(μs, σ), y)
plot(x, dens_μ, leg=false, grid=false, xlabel="μ", ylabel="Likelihood", xaxis=font(14), yaxis=font(14), size=(600,400),
    color=:darkblue, linewidth=2)
dens_y = pdf(Normal(y, σ), y)
plot!([y;y], [0;dens_y], color=:black, linestyle=:dash)
end

# ╔═╡ afa66a1c-5bb0-4332-a0b0-74a920a0cc9d
md"
A similar plot can be generated for the likelihood of $y=.5$ as a function of parameter $\sigma$:
"

# ╔═╡ 93d32bac-b7a1-40c5-88ee-6f8882ac90b1
let
	σs = 0:.01:20
	μ = 0.0
	y = .5
	dens_σ = @. pdf(Normal(μ, σs), y)
	plot(σs, dens_σ, leg=false, grid=false, xlabel="σ", ylabel="Likelihood", xaxis=font(14), yaxis=font(14), size=(600,400),
	    color=:darkblue, linewidth=2)
dens_y = pdf(Normal(μ, y), y)
plot!([y;y], [0;dens_y], color=:black, linestyle=:dash)
end

# ╔═╡ cf8e2ce4-c2f8-4c09-8a05-38d5d3b92779
md"
## CDF

The cummulative density function is the integral or sum of the PDF over some range. The CDF of the normal distribution is stated as follows:

$F(y) = \Pr(Y \leq y) = \int_{-\infty}^y f(x) dx$

The Normal CDF is defined as

$F(y) = \int_{-\infty}^y \frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-1}{2}\left(\frac{x-\mu}{\sigma}\right)^2} dx = \frac{1}{2}\left[1 + \rm erf \left(\frac{x-\mu}{\sigma\sqrt{2}}\right) \right]$

An example of a CDF can be found in the code block below:
"

# ╔═╡ a32b99c5-8076-4d1e-b0c8-736715e92b47
let 
	μ = 0.0
	σ = 1.0
	y = .5
	cdf_y = cdf(Normal(μ, σ), y)
end

# ╔═╡ 0e1609a9-08de-49fc-88e3-32f7e8a7ad2c
let
	x = -3.5:.01:3.5
	μ = 0.0
	σ = 1.0
	y = .5
	cdfs = cdf.(Normal(μ, σ), x)
	plot(x, cdfs, leg=false, grid=false, xlabel="y", ylabel="Pr(Y ≤ y)", xaxis=font(14), yaxis=font(14), size=(600,400),
	    color=:grey, linewidth=2)
	cdf_y = cdf(Normal(μ, σ), y)
	plot!([y;y], [0;cdf_y], color=:black, linestyle=:dash)
	plot!([-3.5;y], [cdf_y;cdf_y], color=:black, linestyle=:dash)
end

# ╔═╡ e95730eb-c132-431f-b3ae-5c0895df51ce
md"


## Likelihood of Data

A likelihood function calculates the likelihood of data given a set of parameters, $\Theta = \{\theta_1,\theta_2 \dots \theta_m\}$. Formally, this is written as

$\mathcal{L}(\Theta; \mathbf{Y}) = \prod_{i=1}^n f(y_i \mid \Theta)$

where $\mathbf{Y} = \{y_1,y_2,\dots y_n\}$ is a vector of data and $\prod$ is a product operator. In computing the likelihood with this function, we assume observations are independent. 

## Log Likelihood of Data

In practice, we often compute the log likelihood because it prevents numerical under-and-overflow. Rather than multiplying each likelihood, with log likelihoods we add each log likelihood instead. This is because of the product rule of logarithms which states $\log(x \cdot y) = \log(x) + \log(y)$. Here is the equation for the log likelihood of the data:

$\log\left(\mathcal{L}(\Theta; \mathbf{Y})\right) = \sum_{i=1}^n \log\left(f(y_i \mid \Theta) \right)$

The code block below shows how to compute the log likelihood of data that are assumed to be normally distributed. 
"

# ╔═╡ 20dede48-c2f3-4a73-989d-f7a51202a724
let
	n = 100
	y = rand(Normal(0, 1), n)
	LL = logpdf(Normal(0, 1), n)
end

# ╔═╡ 710c595d-5398-47ce-bef6-6e0ff80b99a8
md"
# References

[https://betanalpha.github.io/assets/case_studies/probability_theory.html](https://betanalpha.github.io/assets/case_studies/probability_theory.html)
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.62"
Plots = "~1.29.1"
PlutoUI = "~0.7.39"
QuadGK = "~2.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

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
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

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
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

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

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b316fd18f5bc025fedcb708332aecb3e13b9b453"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1e5490a51b4e9d07e8b04836f6008f46b48aaa87"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.3+0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "c6cf981474e7094ce044168d329274d797843467"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.6"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

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
git-tree-sha1 = "3411935b2904d5ad3917dee58c03f0d9e6ca5355"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.11"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

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
git-tree-sha1 = "9e42de869561d6bdf8602c57ec557d43538a92f0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "383a578bdf6e6721f480e749d503ebc8405a0b22"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.6"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
# ╟─14666c46-31be-470a-b989-361e828260b4
# ╟─815269f0-6ecf-11ec-3c92-07db775981f9
# ╟─d3749ccf-20be-4a2f-910a-944859a997e6
# ╟─9837348a-842c-421f-9ec2-664c766bc572
# ╟─422eda59-3883-4e04-ad79-e991ef1bb4a8
# ╟─7bbe6b82-878d-4e79-b5ba-3909eaa525f4
# ╟─ffb3b928-9b2e-444a-a79d-5bc64ebcac11
# ╠═314c3674-89fa-4816-997f-916114237f06
# ╟─b66ef8ac-bf16-46aa-b14c-7c830fccae62
# ╠═2b1651cb-1eec-4a23-8c15-c239643ea5cc
# ╟─1316792e-7af4-4269-93a5-fe358508fd24
# ╟─0aa7d553-15c8-44f6-a79b-b29173724125
# ╟─69eb4e92-3145-444d-bc91-628820b5d010
# ╠═0e83c830-1fde-4b80-9b74-e9a79cb6db6e
# ╟─56bb2a42-0216-481f-807b-7c8006ceadf8
# ╟─d6ae1f70-5f2e-4691-911c-c030e61b62f3
# ╟─ea79096d-bbac-4abd-8488-50bf7aaade40
# ╠═2d1e6d92-1636-47c3-a746-5aac30145196
# ╟─1af5aabd-6b0f-4a47-847a-294c102a5986
# ╠═dc44c97d-4825-48cd-a4df-ce550c6e6581
# ╟─ee94d261-dd31-49ec-8bdf-6616bae4af01
# ╟─db92f860-d815-43bc-9625-2f41b62d4af1
# ╟─ade5f6a3-e5e7-46a3-9ae7-7a20b15100a3
# ╟─29df83ab-cf6e-43df-98de-e19114514e8d
# ╟─afa66a1c-5bb0-4332-a0b0-74a920a0cc9d
# ╟─93d32bac-b7a1-40c5-88ee-6f8882ac90b1
# ╟─cf8e2ce4-c2f8-4c09-8a05-38d5d3b92779
# ╠═a32b99c5-8076-4d1e-b0c8-736715e92b47
# ╟─0e1609a9-08de-49fc-88e3-32f7e8a7ad2c
# ╟─e95730eb-c132-431f-b3ae-5c0895df51ce
# ╠═20dede48-c2f3-4a73-989d-f7a51202a724
# ╟─710c595d-5398-47ce-bef6-6e0ff80b99a8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
