### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ f0b52870-6ecd-11ec-2565-213cacf93546
begin
	using PlutoUI
	TableOfContents()
end

# ╔═╡ f0b5288e-6ecd-11ec-2b6a-27ff56a3cd0d
md"
# Notation 

In this background tutorial, we introduce the notational conventions used throughout the model tutorials.

## Declarative Memory

In ACT-R, declarative memory represents knowledge of factual information, such as a dog is an animal. We denote $M = [\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_{N_m}]$ as a set of relevant facts in memory, called chunks.

## Chunks

In ACT-R a chunk is a basic element of declarative memory, defined as a collection of slot-value pairs. We denote a chunk $m$ as 

$\begin{align}
\mathbf{c}_m = \left\{\left(s_i, v_i\right)\right\}_{i\in \mathcal{I}_m}
\end{align}$


 $s_i$ denotes a slot (e.g. name), $v_i$ denotes a value (e.g. Jennifer), and $\mathcal{I}_m$ denotes the index set for all slot value pairs in chunk $m$. The set of slots in chunk $m$ is denoted as $Q_m = \left\{s_i
\right\}_{i\in \mathcal{I}_m}$

As an example, consider a chunk that represents your co-worker Bob who works in accounting. This chunk includes a slot for your co-worker's name and department:

$\begin{align}
\mathbf{c}_m = \{\rm (name,Bob), (department, accounting)\}
\end{align}$

In this example, the value of the name slot is Bob and the value of the department slot is accounting. The set of slots in chunk $m$ is 

$Q_m = \left\{\textrm{name}, \textrm{department}
\right\}$

In some cases, we need to explicitly indicate where the chunk is currently held, which we do by adding another subscript, $\mathbf{c}_{m,b},$ where $b \in \left\{\rm{retrieval,imaginal,\ldots}\right\}$.

We can also treat a chunk as a function that maps from slots to values, with any slots that are not included in the collection mapping to null or the empty set,

$\begin{align}
c_m(s) = \begin{cases}
v & \textrm{if } (s, v) \in \mathbf{c}_m\\
\emptyset & \textrm{otherwise} 
\end{cases}
\end{align}$

Returning to our example with Bob, suppose we want to know the value associated with name. This can be accomplished as follows:

$c_m(\textrm{name}) = \textrm{Bob}$

"

# ╔═╡ 14ae3925-11a6-4583-a9f7-521dc4ebe3dc
md"
## Retrieval Requests and Production Rule Conditions

Following the notation above, retrieval requests will be specified as

$\begin{align}
\mathbf{r}_j = \left\{\left(s_i, v_i\right)\right\}_{i\in \mathcal{I}_j}
\end{align}$

and production conditions will be specifed as

$\begin{align}
\mathbf{p}_k = \left\{\left(s_i, v_i\right)\right\}_{i\in \mathcal{I}_k}
\end{align}$


## Activation

The probability and speed with which a chunk is retrieved is a monotonically increasing function of activation. Broadly speaking, activation consists of a deterministic component and a stochastic component. The deterministic component may consist of several terms that represent ACT-R's declarative memory mechanisms. Some of these terms can be excluded when simplifying assumptions can be justified. For completeness, we present the full activation equation for chunk $m$:

$\begin{align}
a_m = \underbrace{ \textrm{blc} + \textrm{bll}_m + \rho_m  + S_m }_{\rm deterministic} + \underbrace{ \epsilon_m}_{\rm stochastic}
\end{align}$

The components are defined as:

-  blc: base level constant

-  $\textrm{bll}_m$: base-level learning

-  $\rho_m$: mismatch penalty

-  $S_m$: spreading activation

-  $\epsilon_m$: normally distributed noise

It is often ncessary to treat the deterministic and stochastic components seperately when expressing ACT-R models. In such cases, we designate $\mu_m$ as the deterministic component to make the notation more concise. In the present case, would could define it as:

$\begin{align}
E[a_m] = \mu_m = \textrm{blc} + \textrm{bll}_m + \rho_m  + S_m 
\end{align}$

## Response Mapping

A response mapping identifies a set of conditions that produce a specific response from the model. In ACT-R, response mappings are based on buffer conditions specified in a production rule, such as the result of a memory retrieval. We will represent response mappings with the set $R$ For readers who are unfamiliar with set theory, we will define sets with set builder notation. Elements of a set can be enumerated as so:

$\begin{align}
X = \{1,2,3,4\}
\end{align}$

Alternatively, the set $X$ can be defined with a rule:

$\begin{align}
X = \{\forall x \in  \mathbb{Z} : x > 0, x < 5\}
\end{align}$

The first part of the rule $\forall x \in \mathbb{Z}$ specifies all elements $x$ in the set of all integers $\mathbb{Z}$ (note that sometimes $\forall$  is omitted). The symbol specifies constraints on $x$ and $:$ reads such that. The last part of the rule specifies two constraints: $x$ is greater than zero and $x$ is less than 5. Turning back to response mappings, consider the following set of conditions that maps to a yes:

$\begin{align}
R_{\rm yes} = \{\mathbf{c}_m \in M : \forall q \in Q_j, c_m(q) = p_j(q)\}
\end{align}$

where $\mathbf{p}_j$ is a collection of production rule conditions, and $Q_j$ is the collection of slots in $\mathbf{p}_j$. In words, the response mapping says: the set of all chunks in declarative memory whose slot-value pairs match the slot-pairs in the production rule conditions. Now consider the complementary set of conditions that maps to a no response:

$\begin{align}
R_{\rm no} = \{\mathbf{c}_m \in M : \exists q \in Q_k \textrm{ s. t. } c_m(q) \neq p_k(q)\}
\end{align}$

In the set above, the symbol $\exists q$ reads there is at least one q .... Taken together, this reads: the set of all chunks in declarative memory whose slot-value pairs does not match at least one slot value-pair in the production rule conditions.
"

# ╔═╡ f2c57b78-e609-4d4e-9629-08e1d434ec13
md"
### Example
As a specific example, suppose that a person is asked whether mathematical statements are true. For simplicity, we will assume that declarative memory consists of the following addition facts:

$\begin{align}
M = \{\{(n_1,1),(n_2,1),(\textrm{sum},2)\},\{(n_1,1),(n_2,2),(\textrm{sum},3)\}, \{(n_1,1),(n_1,3),(\textrm{sum},4)\},\\\{(n_1,2),(n_2,1),(\textrm{sum},3)\},  \{(n_1,2),(n_2,2),(\textrm{sum},4)\} ,\{(n_1,2),(n_2,3),(\textrm{sum},5)\},\\\{(n_1,3),(n_2,1),(\textrm{sum},4)\}, \{(n_1,3),(n_2,2),(\textrm{sum},5)\}, \{(n_1,3),(n_2,3),(\textrm{sum},6)\}\}
\end{align}$

In other words, $M$ consists of all permutations of the sum of two numbers ranging from 1 to 3. Suppose that a person is asked whether $2 + 2 > 3$ and that it is possible to retrieve the wrong chunk (i.e. partial matching is enabled). Let's assume that a person responds yes if the sum of the retrieved chunk is greater than 3. The production rule for responding yes contains a slot called num, which contains the comparison number 3. In its simplest form, the production rule is defined as:

$\mathbf{p}_{\textrm{yes}} = \{(\textrm{val},3) \}$

The response set is all possible chunks for which the sum is greater than the comparison value of three. Formally, this is stated as follows: 

$R_{\rm yes} = \{\mathbf{c}_m \in M : c_m(\textrm{sum}) > p_{\textrm{yes}}(\textrm{num})\}$

We can also enumerate all the chunks in the the response set:
$\begin{align}
R_{\rm yes} = \{ \{(n_1,1),(n_1,3),(\textrm{sum},4)\},  \{(n_1,2),(n_2,2),(\textrm{sum},4)\} ,\{(n_1,2),(n_2,3),(\textrm{sum},5)\},\\\{(n_1,3),(n_2,1),(\textrm{sum},4)\}, \{(n_1,3),(n_2,2),(\textrm{sum},5)\}, \{(n_1,3),(n_2,3),(\textrm{sum},6)\}\}
\end{align}$

"


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.51"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "dcebd3174a85b0f68c71e8431fe1914ebcbe8db2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─f0b52870-6ecd-11ec-2565-213cacf93546
# ╟─f0b5288e-6ecd-11ec-2b6a-27ff56a3cd0d
# ╟─14ae3925-11a6-4583-a9f7-521dc4ebe3dc
# ╟─f2c57b78-e609-4d4e-9629-08e1d434ec13
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
