### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 2bf04f4e-67df-45bc-a183-6b7b8c12cfe0
begin
	using ACTRModels, Random, PlutoUI, DataFrames
	Random.seed!(2125)
	TableOfContents()
end

# ╔═╡ 6c65c466-ffa6-4559-9668-85324ce39a2c
md"
# ACTRModels

In this tutorial, you will learn how to use the package ACTRModels.jl. Although ACTRModels.jl has the capability to develop discrete event simulations, as is typical in ACT-R, the primary goal is to provide functions and capabilities for developing likelihood functions.
"

# ╔═╡ 1500b6f4-bcdc-48a5-a3ee-a62fe581caeb
md"
## Documentation
Recall that documentation can be accessed by clicking on the function if the live documentation is open (see bottom right). Open the live documentation and click on the
function below to show the documentation:
"

# ╔═╡ 04476fd1-e79c-4cfe-bcda-ef3cb519683b
add_chunk!

# ╔═╡ 019eb1ae-1887-4ea8-aa34-329d1e7b7cda
md"
A list of available functions and objects can be found by clicking on the package name:
"

# ╔═╡ fac6d27f-1407-4eeb-9146-1f33da77dcbd
ACTRModels

# ╔═╡ 96a3b67d-e44e-4754-afdb-5dd0f1145459
md"
## Type Hierarchy

In ACTRModels.jl, objects are organized hierarchically: An ACTR object contains modules, and modules contain chunks along with other information. We will build an example from the bottom up, starting with chunks and moving to the full ACT-R model object. 

### Chunks

A chunk contains slot value pairs, activation values, and information about history of use. Click on the word `Chunk` below to see the documentation
"

# ╔═╡ fde6c44b-c2c6-4ffd-8401-5a6d09c5e1ce
Chunk

# ╔═╡ d952f0b9-a658-41cd-a9a6-94f4dcd9a110
md"
#### Default Chunk
If the constructor `Chunk` is called with no arguments, a chunk is created with default values no slot-value pairs. 
"

# ╔═╡ 4b931190-9ec7-4d5b-8add-ad6cf086c3e8
Chunk()

# ╔═╡ 709e7145-c17c-4c57-91b0-8e45b1813b39
md"
To improve performance, slot-value pairs are an immutable type by default. Immutable slot-value pairs is suitable for most models.
"

# ╔═╡ 2434876b-9410-42f1-bd57-04ce75ab715e
md"

#### Mutable slot-value pairs
To create a chunk with mutable slot-value pairs, simply pass `true` as a positional argument.
"

# ╔═╡ c8da2333-c7a3-4970-bc4a-2817f823ac72
chunk1 = Chunk(true)

# ╔═╡ 2d8e803d-25de-4ecc-8a7f-98c7d1fa3c56
md"

Now we can add and modify the slot-value pairs:
"

# ╔═╡ 8eea30b2-2eec-402a-ba39-7be7a5614900
begin
	chunk1.slots[:slot] = :value
	chunk1
end

# ╔═╡ 996d5d53-8061-4b78-8be3-0847290d53fe
md"
The following illustrates how to delete a slot by name:
"

# ╔═╡ 1e01779a-2d71-4f3e-bf30-4774d87eb212
delete!(chunk1.slots, :slot)

# ╔═╡ e9e4af3a-ac25-4281-be80-2a9d427ac5e2
md"
In cases where only the slot-values change, rather than the number of slot-values or their types, it is possible to wrap slot-values in an array as follows:
"

# ╔═╡ 3ef16fd3-5e12-4c5c-8f78-780f829b9f04
chunk2 = Chunk(state=[:sad])

# ╔═╡ 64899ca5-892e-4e8c-a058-c888d105e001
md"

Using this approach, it is possible to change the slot value by referencing the first element of the array:
"

# ╔═╡ 5f5a9016-d4ea-4bde-9ace-55b41ebd3dcd
chunk2.slots.state[1] = :happy

# ╔═╡ 140f75b2-0f76-4e27-a8ab-3d1308297367
md"
The is approach is more performant than using a dynamic chunk as shown in the first example. Note that unlike the dynmaic chunk the type cannot be changed when slot values are wrapped in an array. Because the type cannot be changed, the compiler can infer its type, thus generating more performant code. 
"

# ╔═╡ 0d3a23fd-c285-4a33-8fb0-5413d342df13
md"
### Creating specific chunks

In the following examples, we will create chunks with specific slot-value pairs and other properties. Let's begin by creating a chunk that represents Sigma the dog. 
"

# ╔═╡ 625a7c23-98ae-48d5-bb3e-1cd668d9dfad
chunk3 = Chunk(;animal=:dog, name=:Sigma)

# ╔═╡ 605cb2cf-2c47-417d-a4e2-5b2e959485be
md"
If you are using baselevel learning, you may want to pass a value for the time at which the chunk was created. The default value of zero is used if no value is passed
"

# ╔═╡ 80c5fdde-5baf-4373-afed-099ee2db33cf
chunk4 = Chunk(;animal=:dog, name=:Sigma, time_created=2.0)

# ╔═╡ 92451e4a-e8a6-4fc2-b4a3-a541beefc0a6
md"
You also may specify an chunk specific baselevel with the keyword `bl`:
"

# ╔═╡ 794ed9b2-7bb7-47fe-b4f5-a6a8ffc5a655
chunk5 = Chunk(;name=:Bob, department=:Accounting, bl=1.5)

# ╔═╡ 4143a295-92b2-4420-8477-6422f2c06186
md"
### Declarative Memory

The declarative memory module stores a vector of chunks. In the following example, we will create a vector of chunks and pass it to the `Declarative` constructor:
"

# ╔═╡ eda7f651-7dec-4d96-ad58-57e4c4173abb
begin
	chunks = [Chunk(;animal=:dog, name=:Sigma, bl=2.0), Chunk(;animal=:rat, name=:LordXenu, bl=1.5)]
	memory = Declarative(;memory = chunks)
end

# ╔═╡ 97cdd5b4-c104-4805-937b-9e21ca3e4de6
md"
### ACTR object

The `ACTR` object is a data structure that holds parameters and modules. Parameters are passed as keyword arguments. In typical applications, you will only use the declarative memory and imaginal modules because the processing time for the other modules can simply be added together. However, additional capabilities for other modules can be added as necessary. 

The documentation for the `ACTR` model object can be accessed by clicking `ACTR` in the cell below:
"

# ╔═╡ ddd5a5a2-53da-4a80-a553-e46d6c267e43
ACTR

# ╔═╡ 211031ed-658a-426f-94e0-f3113bfd1b41
md"
## Simple model
We will begin with a simple model by passing declarative memory object and parameters by keyword to the `ACTR` object. We will turn on activation noise and set it to 0.5:
"

# ╔═╡ df1fbc69-24bd-4ded-b0db-be04d60f38a7
actr = ACTR(;declarative=memory, noise=true, s=0.5);

# ╔═╡ 1cd35ea6-13f7-494c-8cc8-4281a1b9ba40
md"
### Retrieving a Chunk 

The function `retrieve` computes the activation of chunks based on the retrieval request and returns the chunk with the highest activation that exceeds the retrieval threshold. If no chunk has an activation above the retrieval threshold, an empty vector is returned. In the following example, we will attempt to retrieve a name for a rat. The retrieval request is the optional keyword argument that follows the `;`. Let's set the seed for the random number generator so that the results are reproducible:
"

# ╔═╡ 2ecbf025-f68c-4bd9-bad8-96137cf920b5
Random.seed!(5455)

# ╔═╡ c49ac25d-1c00-4313-87c7-e31f8da37dc7
begin
	retrieval_result = retrieve(actr; animal=:rat)
	retrieved_chunk = retrieval_result[1]
end

# ╔═╡ 67e89289-98c5-4f67-8c9f-8d4ce43cb5bc
md"
The retrieved chunk is an $(retrieved_chunk.slots.animal) named $(retrieved_chunk.slots.name) with an activation value of $(round(retrieved_chunk.act, digits=3))
"

# ╔═╡ 31db0f9e-23ea-439e-b71c-98f99c1d0abf
md"
## Retrieval Probability 
Next, we can compute the approximate retrieval probability of the retrieved chunk using the softmax rule. The second output is the retrieval failure probability. Note that when a value, such as the retrieval failure probability, is not used, it is costumary to assign its value to `_`, which is disregarded by the system.
"

# ╔═╡ 79e668dd-640f-45c8-bf15-c3878d7d3c1f
p1,_ = retrieval_prob(actr, retrieved_chunk; animal=:rat)

# ╔═╡ 04a457ad-418d-46ad-92ab-552c0e6208c0
md"
### Spreading Activation


We will build upon the previous example by including spreading activation from the imaginal buffer. Spreading activation is set to true in the `ACTR` object and the maximum association parameter is set to $\gamma = 1.6$. So that activation can spread to declarative memory, we will add a chunk to the imaginal buffer containing the slot-value pair (animal, rat).
"


# ╔═╡ 66578bdf-cb6d-4d95-b2fb-5ad7fc217af6
begin
	# create the chunks for Sigma and Lord Xenu
	chunks1 = [Chunk(;animal=:dog, name=:Sigma, bl=2.0), Chunk(;animal=:rat, name=:LordXenu, bl=1.5)]
	# create declarative memory object
	memory1 = Declarative(;memory = chunks1)
	# create a rat chunk
	rat_chunk = Chunk(;animal=:rat)
	# add rat chunk to imaginal buffer
	imaginal = Imaginal(;buffer=rat_chunk)
	# add all the components to the ACTR object
	actr1 = ACTR(;declarative=memory1, imaginal, noise=true, s=0.5, sa=true, γ=1.6);
end

# ╔═╡ 916bf149-fb82-43f3-a6f1-520e7c21f515
begin
	retrieval_result1 = retrieve(actr1; animal=:rat)
	retrieved_chunk1 = retrieval_result1[1]
end

# ╔═╡ 94b7c676-7fc7-4ffb-8aa1-558af8cb6e33
md"
The retrieved chunk is an $(retrieved_chunk1.slots.animal) named $(retrieved_chunk1.slots.name) with an activation value of $(round(retrieved_chunk1.act, digits=3)). As expected, activation is higher when spreading activation is enabled: $(round(retrieved_chunk1.act, digits=3)) vs. $(round(retrieved_chunk.act, digits=3)). As one might expect, the retrieval probability is also higher:
"

# ╔═╡ 62c1e6ed-01c8-4ca7-983d-fbc571d3f3ae
p2,_ = retrieval_prob(actr1, retrieved_chunk1; animal=:rat)

# ╔═╡ 18c5ace3-1aab-4aa7-9064-70ff2ae6cc4f
md"

## Retrieval Time

The function `compute_RT` is used to generate a sample from the retrieval time distribution. Before computing retrieval time, it is necesary to compute activation values with `compute_activation!` or `retrieve`, which calls `compute_activation!` on the retrieval set and returns the chunk with the highest activation. In the following example, we will retrieve a chunk and compute the retrieval time:
"

# ╔═╡ 65a6465f-bdd0-4d55-a628-5e9b53c11c09
t1 = compute_RT(actr, retrieved_chunk)

# ╔═╡ c9375abf-3fd2-41f7-810a-6c66ec80d447
t2 = compute_RT(actr1, retrieved_chunk1)

# ╔═╡ 141c26ad-bb6a-404d-879a-7cd755869a22
md"
As you might exect, the retrieval time was faster with spreading activation enabled than when it was disabled:  $(round(t2, digits=3)) seconds vs. $(round(t1, digits=3)) seconds.
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ACTRModels = "c095b0ea-a6ca-5cbd-afed-dbab2e976880"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
ACTRModels = "~0.10.6"
DataFrames = "~1.3.4"
PlutoUI = "~0.7.39"
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

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

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

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

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

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "c6cf981474e7094ce044168d329274d797843467"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.6"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

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

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "b4975062de00106132d0b01b5962c09f7db7d880"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.5"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "3411935b2904d5ad3917dee58c03f0d9e6ca5355"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

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

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[deps.SequentialSamplingModels]]
deps = ["ConcreteStructs", "Distributions", "Interpolations", "KernelDensity", "Parameters", "PrettyTables", "Random"]
git-tree-sha1 = "d43eb5afe2f6be880d3bd79c9f72b964f12e99a5"
uuid = "0e71a2a6-2b30-4447-8742-d083a85e82d1"
version = "0.1.7"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─2bf04f4e-67df-45bc-a183-6b7b8c12cfe0
# ╟─6c65c466-ffa6-4559-9668-85324ce39a2c
# ╟─1500b6f4-bcdc-48a5-a3ee-a62fe581caeb
# ╠═04476fd1-e79c-4cfe-bcda-ef3cb519683b
# ╟─019eb1ae-1887-4ea8-aa34-329d1e7b7cda
# ╠═fac6d27f-1407-4eeb-9146-1f33da77dcbd
# ╟─96a3b67d-e44e-4754-afdb-5dd0f1145459
# ╠═fde6c44b-c2c6-4ffd-8401-5a6d09c5e1ce
# ╟─d952f0b9-a658-41cd-a9a6-94f4dcd9a110
# ╠═4b931190-9ec7-4d5b-8add-ad6cf086c3e8
# ╟─709e7145-c17c-4c57-91b0-8e45b1813b39
# ╟─2434876b-9410-42f1-bd57-04ce75ab715e
# ╠═c8da2333-c7a3-4970-bc4a-2817f823ac72
# ╟─2d8e803d-25de-4ecc-8a7f-98c7d1fa3c56
# ╠═8eea30b2-2eec-402a-ba39-7be7a5614900
# ╟─996d5d53-8061-4b78-8be3-0847290d53fe
# ╠═1e01779a-2d71-4f3e-bf30-4774d87eb212
# ╟─e9e4af3a-ac25-4281-be80-2a9d427ac5e2
# ╠═3ef16fd3-5e12-4c5c-8f78-780f829b9f04
# ╟─64899ca5-892e-4e8c-a058-c888d105e001
# ╠═5f5a9016-d4ea-4bde-9ace-55b41ebd3dcd
# ╟─140f75b2-0f76-4e27-a8ab-3d1308297367
# ╟─0d3a23fd-c285-4a33-8fb0-5413d342df13
# ╠═625a7c23-98ae-48d5-bb3e-1cd668d9dfad
# ╟─605cb2cf-2c47-417d-a4e2-5b2e959485be
# ╠═80c5fdde-5baf-4373-afed-099ee2db33cf
# ╟─92451e4a-e8a6-4fc2-b4a3-a541beefc0a6
# ╠═794ed9b2-7bb7-47fe-b4f5-a6a8ffc5a655
# ╟─4143a295-92b2-4420-8477-6422f2c06186
# ╠═eda7f651-7dec-4d96-ad58-57e4c4173abb
# ╟─97cdd5b4-c104-4805-937b-9e21ca3e4de6
# ╠═ddd5a5a2-53da-4a80-a553-e46d6c267e43
# ╟─211031ed-658a-426f-94e0-f3113bfd1b41
# ╠═df1fbc69-24bd-4ded-b0db-be04d60f38a7
# ╟─1cd35ea6-13f7-494c-8cc8-4281a1b9ba40
# ╠═2ecbf025-f68c-4bd9-bad8-96137cf920b5
# ╠═c49ac25d-1c00-4313-87c7-e31f8da37dc7
# ╟─67e89289-98c5-4f67-8c9f-8d4ce43cb5bc
# ╟─31db0f9e-23ea-439e-b71c-98f99c1d0abf
# ╠═79e668dd-640f-45c8-bf15-c3878d7d3c1f
# ╟─04a457ad-418d-46ad-92ab-552c0e6208c0
# ╠═66578bdf-cb6d-4d95-b2fb-5ad7fc217af6
# ╠═916bf149-fb82-43f3-a6f1-520e7c21f515
# ╟─94b7c676-7fc7-4ffb-8aa1-558af8cb6e33
# ╠═62c1e6ed-01c8-4ca7-983d-fbc571d3f3ae
# ╟─18c5ace3-1aab-4aa7-9064-70ff2ae6cc4f
# ╠═65a6465f-bdd0-4d55-a628-5e9b53c11c09
# ╠═c9375abf-3fd2-41f7-810a-6c66ec80d447
# ╟─141c26ad-bb6a-404d-879a-7cd755869a22
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
