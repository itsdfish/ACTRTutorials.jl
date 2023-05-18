### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ d87a7f68-13d5-41a9-a12d-6bffdad25809
begin

path_introduction = joinpath(pwd(), "../Tutorial_Models/Introduction/Introduction.jl")
	
path_u1_1 = joinpath(pwd(), "../Tutorial_Models/Unit1/Simple_Retrieval_1/Simple_Retrieval_Model_1_Notebook.jl")
	
path_u1_2 = joinpath(pwd(), "../Tutorial_Models/Unit1/Simple_Retrieval_2/Simple_Retrieval_Model_2_Notebook.jl")

path_u1_3 = joinpath(pwd(), "../Tutorial_Models/Unit1/Simple_Retrieval_3/Simple_Retrieval_Model_3_Notebook.jl")

path_u2_1 = joinpath(pwd(), "../Tutorial_Models/Unit2/Simple_RT_1/Simple_RT_Model_1_Notebook.jl")

path_u2_2 = joinpath(pwd(), "../Tutorial_Models/Unit2/Simple_RT_2/Simple_RT_Model_2_Notebook.jl")

path_u2_3 = joinpath(pwd(), "../Tutorial_Models/Unit2/Simple_RT_3/Simple_RT_Model_3_Notebook.jl")

path_u2_4 = joinpath(pwd(), "../Tutorial_Models/Unit2/Simple_RT_4/Simple_RT_Model_4_Notebook.jl")

path_u3_1 = joinpath(pwd(), "../Tutorial_Models/Unit3/Guessing_Mixture/Guessing_Mixture_Model_Notebook.jl")

path_u3_2 = joinpath(pwd(), "../Tutorial_Models/Unit3/Siegler_Choice/Siegler_Choice_Model_Notebook.jl")
	
path_u3_3 = joinpath(pwd(), "../Tutorial_Models/Unit3/Simple_Learning/Simple_Learning_Notebook.jl")

path_u3_4 = joinpath(pwd(), "../Tutorial_Models/Unit3/Serial_Recall_1/Serial_Recall_Model_1_Notebook.jl")

path_u3_5 = joinpath(pwd(), "../Tutorial_Models/Unit3/Serial_Recall_2/Serial_Recall_Model_2_Notebook.jl")

path_u4_1 = joinpath(pwd(), "../Tutorial_Models/Unit4/Count/Count_Model_Notebook.jl")

path_u4_2 = joinpath(pwd(), "../Tutorial_Models/Unit4/Addition/Addition_Model_Notebook.jl")

path_u5_1 = joinpath(pwd(), "../Tutorial_Models/Unit5/Semantic/Semantic_Model_Notebook.jl")

path_u5_2 = joinpath(pwd(), "../Tutorial_Models/Unit5/Simple_PVT/Simple_PVT_Model_Notebook.jl")

path_u6_1 = joinpath(pwd(), "../Tutorial_Models/Unit6/Paired/Paired_Model_Notebook.jl")

path_u6_2 = joinpath(pwd(), "../Tutorial_Models/Unit6/Semantic_FFT/Run_Semantic_FFT_Notebook.jl")
	
path_u6_3 = joinpath(pwd(), "../Tutorial_Models/Unit6/Siegler_RT/Siegler_RT_Model_Notebook.jl")

path_u7_1 = joinpath(pwd(), "../Tutorial_Models/Unit7/IBL/IBL_Model_Notebook.jl")

path_u7_2 = joinpath(pwd(), "../Tutorial_Models/Unit7/IBL_Inertia/IBL_Inertia_Model_Notebook.jl")

path_u8_1 = joinpath(pwd(), "../Tutorial_Models/Unit8/Grouped_Recall_1/Grouped_Recall_1_Notebook.jl")

path_u8_2 = joinpath(pwd(), "../Tutorial_Models/Unit8/Grouped_Recall_2/Grouped_Recall_2_Notebook.jl")
	
path_u9_1 = joinpath(pwd(), "../Tutorial_Models/Unit9/Fan_Model_1/Fan_Model_1_Notebook.jl")

path_u10_1 = joinpath(pwd(), "../Tutorial_Models/Unit10/Visual_Search/Visual_Search_Model_Notebook.jl")

path_ado = joinpath(pwd(), "../Tutorial_Models/Unit12/Optimize_Simple_Learning/Simple_Learning_Notebook.jl")

path_u13_3 = joinpath(pwd(), "../Tutorial_Models/Unit13/Fan_Model_Stan/Fan_Model_2_Notebook.jl")
	
path_julia = joinpath(pwd(), "../Background_Tutorials/julia_tutorial.jl")

path_stan = joinpath(pwd(), "../Background_Tutorials/stan_tutorial.jl")

path_mcmc = joinpath(pwd(), "../Background_Tutorials/mcmc_sampling.jl")

path_actr_models = joinpath(pwd(), "../Background_Tutorials/ACTRModels_Tutorial.jl")

path_convolutions = joinpath(pwd(), "../Background_Tutorials/Convolutions.jl")

path_DEMCMC = joinpath(pwd(), "../Background_Tutorials/Differential_Evolution_MCMC.jl")

path_notation = joinpath(pwd(), "../Background_Tutorials/Notation.jl")

path_probability = joinpath(pwd(), "../Background_Tutorials/Probability_Theory.jl")

path_tools = joinpath(pwd(), "../Background_Tutorials/Fundamental_Tools.jl")

path_bayesian = joinpath(pwd(), "../Background_Tutorials/bayesian_parameter_estimation.jl")

path_markov = joinpath(pwd(), "../Background_Tutorials/Markov_Process.jl")

path_lognormal = joinpath(pwd(), "../Background_Tutorials/Lognormal_Race_Process.jl")
nothing
end

# ╔═╡ e6912ca4-3638-11ec-3bb7-012c76824681
Markdown.parse("
# Table of contents


!!! warning
	Some tutorials have not been added, but will be added soon. 

## Tutorial Models

The model tutorials are organized based on concepts and increasing difficulty. We recommend completing model tutorials in the order listed below while referencing the background tutorials as needed. However, once you have completed units 1 and 2, it might be possible to skip around in some cases. 

* [Introduction](./open?path=$path_introduction)
    
    Explains the rationale behind using likelihood based approaches for cognitive architectures and outlines the tutorial. 
    
* Unit 1

    A series of choice probability of increasing complexity. 

    1. [Simple Retrieval Model 1](./open?path=$path_u1_1)
    2. [Simple Retrieval Model 2](./open?path=$path_u1_2)
    3. [Simple Retrieval Model 3](./open?path=$path_u1_3)

* Unit 2

    A series of joint choice reaction time models of increasing complexity. 

    1. [Simple RT Model 1](./open?path=$path_u2_1)
    2. [Simple RT Model 2](./open?path=$path_u2_2)
    3. [Simple RT Model 3](./open?path=$path_u2_3)
    4. [Simple RT Model 4](./open?path=$path_u2_4)

* Unit 3

    A series of choice probabilty models with mixture distributions.

    1. [Guessing Mixture](./open?path=$path_u3_1)
    2. [Siegler Choice](./open?path=$path_u3_2)
    3. [Simple Learning Model](./open?path=$path_u3_3)
    4. [Serial_Recall 1](./open?path=$path_u3_4)
    5. [Serial_Recall 2](./open?path=$path_u3_5)

    
*  Unit 4

    Introduces the concept of convolution with fast Fourier transform using the Count model as an example. The addition model is a practice exercise.

    1. [Count Model](./open?path=$path_u4_1)
    2. [Addition Model](./open?path=$path_u4_2)

* Unit 5
    1. [Semantic Model](./open?path=$path_u5_1)
    2. [Simple PVT Model](./open?path=$path_u5_2)

* Unit 6
    1. [Paired Associates Model](./open?path=$path_u6_1)
    2. [Semantic Model FFT](./open?path=$path_u6_2)
    3. [Siegler RT Model](./open?path=$path_u6_3)

* Unit 7

    Instance-based learning models
    1. [Instance Based Model](./open?path=$path_u7_1)
    2. [Instance Based Model Inertia](./open?path=$path_u7_2)

* Unit 8

    1. [Grouped Recall 1](./open?path=$path_u8_1)
    2. [Grouped Recall 2](./open?path=$path_u8_2)

* Unit 9
    
    Models of the fan effect using spreading activation.
    
    1. [Fan Model 1](./open?path=$path_u9_1)
    2. [Fan Model 2](../Tutorial_Models/Unit9/Fan_Model_2/Fan_Model_2.ipynb)

*  Unit 10

    A model of conjunctive visual search
    
    1. [Visual Search](./open?path=$path_u10_1)
    
* Unit 11

    Introduces likelihood approximation methods for performing Bayesian inference with ACT-R

    1. [Simple Retrieval Model 1 PDA](../Tutorial_Models/Unit11/Simple_Retrieval_PDA/Simple_Retrieval_Model_1_PDA.ipynb)
    2. [Simple RT Model 2 PDA](../Tutorial_Models/Unit11/Simple_RT_Model_2_PDA/Simple_RT_Model_2_PDA.ipynb)
    3. [Siegler PDA Practice](../Tutorial_Models/Unit11/Siegler_PDA/Siegler_Model_Choice_PDA.ipynb)
    
* Unit 12
    
    Introduces adaptive design optimization for iteratively changing experiment parameters to improve parameter estimation.
    1. [Optimize Simple Learning Model](./open?path=$path_ado)
    
* Unit 13
    
    Selected models implemented in the Stan programming language.
    1. [Simple RT Model 2](../Tutorial_Models/Unit13/Simple_RT_2_Stan/Simple_RT_Model_2.ipynb)
    2. [Siegler Choice Model](../Tutorial_Models/Unit13/Siegler_Model_Stan/Siegler_Choice_Model.ipynb)
    3. [Fan Model](./open?path=$path_u13_3)
"
	)

# ╔═╡ 464a18a4-90ee-4fcb-baa3-5d9f992c92c8
Markdown.parse(
"
## Background Tutorials

We provide a series of tutorials for using software and statistical and mathematical concepts.

### Software 

* [Julia Programming Language](./open?path=$path_julia)

    A 10-20 minute crash course on the Julia language.
 
* [Stan Programming Language](./open?path=$path_stan)
    
    A brief overview of the Stan probabalistic programming language

* [MCMC Sampling](./open?path=$path_mcmc)

  A crash course on MCMC sampling with Turing and diagnosing convergence of chains.
  

* [ACT-R Models](./open?path=$path_actr_models)

  A tutorial for simulating data from ACT-R models, defining likelihood functions, and using basic utilities.
  

* [Convolutions using FFT](./open?path=$path_convolutions)

  A tutorial showing how to define a likelihood function with fast Fourier transform.
  

* [Differential Evolution MCMC](./open?path=$path_DEMCMC)

  A tutorial on MCMC sampling with DEMCMC, which can used with simulation based models and models for which gradients cannot be computed.

### Math and Statistics 
* [Notation](./open?path=$path_notation)

    Mathematical notation for ACT-R models used throughout the tutorial
    

* [Probability Theory](./open?path=$path_probability)

    A basic tutorial on probability theory, which includes joint probabilities, marginalization, conditional probabilities, probability density functions and more.
    

* [Fundamental Tools](./open?path=$path_tools)

    A survey of composible statistical concepts and tools that can be used to build likelihood functions for ACT-R models. This includes, convolution for serial processes, minimimum processing time, maximum processing time and mixtures.
    

* [Bayesian Inference](./open?path=$path_bayesian)

    An introduction to Bayesian inference, including concepts such as prior distributions, likelihood functions, and posterior distributions. These concepts are illustrated with a simple Binomial model.
    

* [Markov Processes](./open?path=$path_markov)

    A brief tutorial on discrete time Makov processes. 

* [Lognormal Race Process](./open?path=$path_lognormal) 

    A tutorial on the Lognormal Race model, which can be used to develop likelihood functions of memory retrieval in ACT-R. "
	
)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─d87a7f68-13d5-41a9-a12d-6bffdad25809
# ╟─e6912ca4-3638-11ec-3bb7-012c76824681
# ╟─464a18a4-90ee-4fcb-baa3-5d9f992c92c8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
