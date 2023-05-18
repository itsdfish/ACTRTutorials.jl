### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ 43366b1a-541e-11ec-0545-15db49af78b7
begin
	using LinearAlgebra,PlutoUI
	TableOfContents()
end

# ╔═╡ 513f6e52-b64f-4d55-97b1-3dfcafe31d73
using Distributions

# ╔═╡ 8c35371c-e3a1-4204-8b7d-45edd52ebc1b
md"

# Julia

In this tutorial, you will learn the basics of programming in the Julia language. Julia is a high level scientific programming language with similar syntax to R, Python and Matlab. Unlike R, Python and Matlab, native Julia code is fast without relying on C or C++, usually clocking in within a factor of 2 of optimized C code. This allows one to write readable code with few trade-offs between flexibility and speed.

## Paradigm

Multiple dispatch is the primary design paradigm on which Julia is based. In multiple dispatch, different versions of the same function called methods are selected at runtime based on the types of argument inputs. [Multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch) can emulate functional and object oriented patterns with less boiler plate code, and is often more flexible because the inheritance hierarchy can be circumvented. Julia relies primarily on just-in-time (JIT) complication: code is compiled upon execution. This results in an initial cost for the first execution which is generally small, but can be noticible for large packages, such as plotting packages. Developers anticipate that compilation times will improve in future versions. 
"

# ╔═╡ cd684ca4-6483-4407-84f9-24a7757c096f
md"
# Pluto

The environment you are currently using is called Pluto. Pluto is an interactive notebook that supports Julia, html, markdown and LaTeX equations, making it ideal for tutorials. It is important to note two ways Julia functions differently in a Pluto notebook. First, multi-line commands must be wrapped in a `begin` block in Pluto. This is not the case for Julia in general. Second, Pluto notebooks are reactive much like an Excel spreadsheet. This means that all references to a variable are changed when it is changed. A variable cannot be re-declared however. If you type `x = .2` into a new cell (created by clicking the plus icon, which appears upon hovering over an existing cell), Pluto will return an error indicating multiple definitions for `x`. Instread, the value of variables can be changed via a slider (hover over cell to see slider code). If you move the slider, the value of `x1` will update. 
"

# ╔═╡ 352f3570-5344-4629-8ed1-bed5c36c25a7
x1 = @bind x1 Slider(-100:100, default=0)

# ╔═╡ cda2f5d7-1e04-46c2-830b-60e55841bfb3
"x1 = $x1"

# ╔═╡ 5bb33a0d-dabb-4988-97f6-539458c9a639
md"
## Help

There are two methods for accessing documentation in a Pluto notebook. One method involves typing `? function_name` in a cell, which will open the live documentation located in the bottom left hand corner. The other method is to expand the live documentation and click in a cell. Live documentation will automatically update based on content in the cell. Try typing `? sum` into the cell below:
"

# ╔═╡ d1e1ad29-088a-4315-9f65-4c32417ec868


# ╔═╡ 775f90c2-222c-43b2-a62c-e3485ba0c628
md"
## Adding Packages

Pluto notebooks automatically downloads, installs and adds packages to the version tracker when the command `using PackageName` is issued. This ensures that Pluto notebook has same package version when it is shared with other people. Outside of Pluto notebooks, you can add a pacakge with the command `] add PackageName`, which will switch to package mode and add the specified package. Here is an example of adding a package:
"

# ╔═╡ 4ec22c90-2256-4f0c-967f-c8869c9abca1
md"

## Updating Packages

In Pluto, packages are updated via a menu that is opened by clicking on the checkmark next to the package name. Upon opening, the menu shows the version number and an up arrow icon, which will update the package. When a package is updated, a copy of the notebook with the previous package versions is created in case the update contains breaking changes. Outside of Pluto, the command `] update` will update all packages.
"

# ╔═╡ 203307dd-1e36-4c32-9349-c45a5d5e55b0
md"
## Using Packages 

Now that the package `Distributions` has been added to the current session, we can access methods and objects defined in the package. Methods that are exported from the package into the current session, which can be called directly in most cases. In the following example, we will generate an object for a standard normal distribution:
"

# ╔═╡ f7bd7a61-36e5-4692-8f79-40a256f4a6a1
Normal()

# ╔═╡ 7b60f7db-56dd-4e30-87bc-0b202c504a74
md"
Name collisions occur when two packages export methods with the same name, in which case a warning message will appear and you must qualify the method call with the package name: 
"

# ╔═╡ 558ef5a3-9ead-4de4-b164-be783d951f5b
Distributions.Normal()

# ╔═╡ 69c9540f-b1bb-4840-b8a9-7acaffa1c1f5
md"
Name collisions are rare, and when they do occur, you can optionally assign an allias to the package name with `constant my_pkg = my_package` to make the syntax more terse. In future releases, you will be able to specify `using my_package as my_pkg`, similar to Python.

The following example illustrates the nested scopes of `Main`, `Distributions` and the constructor `Normal`.
"

# ╔═╡ 9a0f0a84-1d5b-4715-b041-42cd6940987f
md"
# Type System

Julia has a type system which works in tandem with multiple dispatch to create a powerful abstraction. In what follows, we will provide examples of primative and custom types. Later in the tutorial, we will go through some examples to showcase the flexiblity of multiple dispatch, which can help you understand code in the tutorial and develop your own models if you so choose. 

## Primative types
Primative types in Julia include numbers, strings, symbols and functions. These types are arranged hierarchically. For example:
"

# ╔═╡ c15068d4-afbc-44b1-b092-01a3f45456a6
Integer <: Number

# ╔═╡ c1e51d62-ba32-43c0-b44a-3c7e820579fe
md"
which indicates that an integer is a sub-type of a number. We also see that a `Float64` is a sub-type of `Number`:
"

# ╔═╡ 01e83efb-045b-431e-95f5-02f4ad78a164
Float64 <: Number

# ╔═╡ 2235469f-83f7-4bad-bd0d-4a5ced52d96a
md"
As expected, a `Complex` type is not a sub-type of `Real`:
"

# ╔═╡ 0dcaef7c-947b-4034-89c6-bd55b1dbe457
Float64 <: Number

# ╔═╡ 9d032dc2-fcaf-49e7-b991-ef3a69e3c665
md"
The function `typeof` returns the type of a variable. For instance:
"

# ╔═╡ a3736648-5890-4a2b-9211-bb11ca462e6c
begin
	x = 0.3
	string("x is a ", typeof(x))
end

# ╔═╡ c22c2af3-9abe-4266-b103-105650223045
md"
## Unicode Characters

It is possible to use unicode as variable names in Julia with `\unicode + tab` where tab is the tab button on your keyboard. For example, θ is specified as `\theta + tab`:
"

# ╔═╡ d6fa8f6c-9b79-464f-88c5-29c13a43c09f
θ = .3

# ╔═╡ 5d6f1cb7-21f9-4886-b666-3900e84c2845
md" Unicode characters can be helpful because they make code look more similar to mathematical formulas."

# ╔═╡ fa8ccde6-22b7-4afc-a08f-ca07833ee865
md"

# Custom Types

Custom types in Julia are composites of primative types and other custom types. Custom types are immutable by default, but can be set to mutable with the mutable keyword. Unlike simple variables, it is generally advicable to declare the type of fields in the object.

### Immutable 

 The following example demonstrates how to declare an immutable type.
"

# ╔═╡ 14b6c9a0-ac08-4abf-a477-219632348310
struct MyImmutable
	x::Int
end

# ╔═╡ 406dfacf-f33d-4e91-a6a7-e60df4839def
md" An instance of `MyImmutabable` is created as follows:"

# ╔═╡ 5f9e8a03-488e-4cb9-a1a8-26e6883c0f64
my_immutable = MyImmutable(1)

# ╔═╡ 544f84c3-739c-41f6-b43c-1a272f73f85a
md"
The following code demonstrates that the value of an immutable object cannot be changed:
"

# ╔═╡ cfb9249b-9313-4807-8503-fbd4e01aab48
my_immutable.x = 2

# ╔═╡ 60d8c7b9-af8b-4d83-a2d0-2c3d873eff6c
md"
### Mutable

Mutable types allow the value but not type of the fields to change. Let's create a mutable type:
"

# ╔═╡ 001f99c3-424e-43f2-b2f3-90a49003b38a
mutable struct MyMutable
    x::Int
end

# ╔═╡ ff531449-dc5f-444f-b256-7db310da7d01
md"
Now that we have specified the type `MyMutable` we can create an instance:
"

# ╔═╡ 8db9a982-4cd1-49b1-94f9-700c99f00e79
md"
The value of `x` in `my_mutable` can be changed by accessing the field `x` with `.`:
"

# ╔═╡ ef4f1b15-0454-4a78-82c5-9caf3aa3f872
md"
This example also shows that it is possible to modify the contents of a container (a variable that holds multiple values) in Pluto. However, we cannot redeclare the container itself:
"

# ╔═╡ 90f6ea01-8282-4d46-8a08-c2525d1ef15e
my_mutable.x = 2

# ╔═╡ 9f927455-1d6d-47d1-91a7-6003ea59d366
md"
Although we can change the value of a mutable type because it is a container, we cannot assign a value to `x` for which the type is not an `Int`:
"

# ╔═╡ b78132b7-5944-4c86-91dd-aa1fcc99b9f4
my_mutable.x = "hello"

# ╔═╡ ffcff15c-c6a1-4dcc-a425-c10c285854ee
md"
Suppose that we wanted to change the type of `x` in a mutable type. By default, a field without an explicit type annotation is designated as `Any`, which will accept any type. We could also be explicit by using `x::Any`. In the following cell, we will create a mutable type that allows any variable type in the `x` field:
"

# ╔═╡ 9b4cd942-fd4b-4095-ac5d-a5800f2cd5d5
mutable struct Anything
    x
end

# ╔═╡ b51fe7d1-9d3b-432a-adc0-573d657756aa
md"
Let's create an instance of `Anything` in which the `x` field is an `Int`:
"

# ╔═╡ cbb8d9ce-bd5d-4ca9-9edc-188e4b2393da
anything = Anything(1)

# ╔═╡ 17ed7e45-ac79-4aac-be14-4611455d9747
md" In this example using the type `Anything`, we can change the type in addition to the value:"

# ╔═╡ 2a42120f-884d-4bfe-a691-105ee10ff2ce
anything.x = "hello"

# ╔═╡ 2aa517d3-4f3a-42da-8b3a-a34decc8f3e0
md"
As you can see, declarying the type to be Any or, equivalently, ommitting the type declaration altogether, allows the type of the field to be anything and can be changed arbitrarily. This flexibility comes at the cost of performance. Generally, this level of flexibility is not needed. Using parametric types is typically a viable approach for making objects more flexible with minimal performance costs. 
"

# ╔═╡ 23bc5cbe-535d-4942-81ad-b397d205eaf8
md"

### Parametric Types

A parametric type (sometimes called a generic type) allows object fields to assume a variety of types, which can optinally be restricted to a specified sub-type. However, once an object is initialized, the type cannot change. Here is an example in which the field x must be a subtype of number:
"

# ╔═╡ d04b2bd8-2c56-453b-82d7-066219d2156e
mutable struct MyParametric{T <:Number}
    x::T
end

# ╔═╡ fe0c8f58-b4b6-4ab3-8f55-7c05e5aa97b7
md"

In the following cells, we have an example of types in which `x` is a `Float64` and `Int`. The type information can be viewed by clicking the ▶ next to the output.
"

# ╔═╡ 340fb788-3502-474e-a2ef-f2cbd19d8fa4
my_parametric_float = MyParametric(.3)

# ╔═╡ c7b17819-b35c-4a9e-861e-6e16b853ad53
my_parametric_int = MyParametric(100)

# ╔═╡ d2578db2-ffc8-4d37-a333-354f08bbf811
md"
Of course, we cannot assign a type `MyParametric` that is not a sub-type of `Number`:
"

# ╔═╡ 132d7bd7-671b-4256-9a76-8453071d7788
my_parametric_string = MyParametric("hello")

# ╔═╡ fa8c2b80-3885-4035-a3f0-d641e07261be
md"
In practice, you may define many fields with different type restrictions. In the following example, name must be a string, x is a 1 dimensional array in which elements of type T1 must be a subtype of Number, and fields m an z must be the same type, but no restrictions are imposed on that type. Here is an example:
"

# ╔═╡ 364a303c-7491-4e75-a8cc-7a736b6a8265
mutable struct MyType{T1<:Number,T2}
    name::String
    x::Array{T1,1}
    m::T2
    z::T2
end

# ╔═╡ aec8d779-65d0-4b55-a869-30778a87dd0b
md" An instance can be created as follows:"

# ╔═╡ 9e909bbf-6c2e-416f-b499-dbc39d2a45b7
MyType("hello", [1,2,3], .4, .5)

# ╔═╡ 7c3108ab-9778-40e7-ae21-3a02e58cdf3f
md"
## Built-in Data Structures

Julia provides a variety of built-in structures for organizing data, including arrays and dictionaries.
"

# ╔═╡ 410772e1-015b-46c9-8746-37b20b18b5f0
md"

### Arrays

An array is a numerically indexed container of values. In the following example, a one dimensional array of random numbers is generated:
"

# ╔═╡ 04da2ede-fd5d-4575-92e6-c529e1ab900a
my_array = rand(5)

# ╔═╡ d180db94-1657-46cc-a15e-f214152b49d6
md"
Much like Matlab and R, but in contrast to Python, Julia uses 1 based indexing (although custom [indexing](https://github.com/JuliaArrays/OffsetArrays.jl) can be used if desired). Square brackets are used to index specific values of an array:
"

# ╔═╡ 33167674-0e58-4492-8ff1-7401e268c88e
my_array[1]

# ╔═╡ 09f2b339-8dfd-45ae-bbf9-f9b1ccba8f39
md" The keyword end will return the last element:"

# ╔═╡ 5c249a3d-bb89-453e-ac96-5ff91ea881df
my_array[end]

# ╔═╡ d9694ad9-7467-4f0e-8f3b-771c4cdcce7e
md" An array of indices can be used to return more than 1 element of an array:"

# ╔═╡ 0cc514c1-9bf1-473d-bb92-1369a4863675
sub_array = my_array[[1,2]]

# ╔═╡ 59c4f8eb-9b37-4461-9324-cd83c8b50ce4
md" By default, Julia creates a copy when multiple values are indexed as illustrated here:"

# ╔═╡ 9794d991-fc63-49df-bafd-fa3812795c7d
begin
	# change the first element
	sub_array[1] = 0.0
	# my_array does not change because sub_array is a copy
	my_array
end

# ╔═╡ 2f5be4df-24e0-414d-9264-f658beb0c90e
my_view = @view my_array[[1,2]]

# ╔═╡ 00393d8b-0b3e-453d-b4d6-d9a26e5ace41
md" It is possible to return a view or reference of a sub_array with the `@view` macro:"

# ╔═╡ ff7fd34f-3107-448c-b450-5c6e9c71b22e
begin
	# change the first element
	my_view[1] = 0.0
	# my_array now has 0.0 as the first element
	my_array
end

# ╔═╡ 7d031965-6a4a-4d8a-826f-bc92ce4734bd
md" It is possible to create arrays with an arbitrary number of dimensions in Julia. The following example generates a 3 dimensional array containing uniform random numbers:"

# ╔═╡ 13b2509b-bf31-4747-8f44-8754fe8e3f71
three_d_array = rand(5, 2, 3)

# ╔═╡ b0d4f161-5461-41e0-974c-88caeff02fdc
md" In addition, you can create an array of arrays in which each array has different dimensions  The following code generates an array containing a 1X1 array, a 2X2 array and a 3X3 array with a list comprehension:"

# ╔═╡ b075d869-0850-472a-8f6f-29e0907cb082
weird_array = [rand(i,i) for i in 1:3]

# ╔═╡ 29667264-6aea-4694-a336-3214c28520f2
md" The second array can be accessed as follows:"

# ╔═╡ f98231c1-b575-443d-b99b-2579a642c829
weird_array[2]

# ╔═╡ 6127da88-a63a-4582-a4de-2661d7a71f0f
md" Use push! to add elements to an array without creating a copy:"

# ╔═╡ bff2e4bd-8a55-4745-a9ae-c89bcbdfdae9
push!(my_array, .99)

# ╔═╡ 5e4450cc-e657-4609-8417-049e973ae4ef
md" Note that in Julia appending `!` to a function name does not change the way it works, but serves as a convention to express that the function modifies the input." 

# ╔═╡ b5f634e3-194a-4de1-8c93-5f8c8c2e4e59
md"Use the elipse `...` to push multiple elements into a list:"

# ╔═╡ c8727f82-2a14-47a6-8656-42180d1fc4e9
push!(my_array, rand(3)...)

# ╔═╡ 9bafc6cf-8889-4a68-9fc7-a6705b5cbb4f
md"

### Dictionaries

Dictionaries are unordered and mutable key-value pairs. Dictionaries allow you to associate a unique quantity called a key with a quantity called a value using the arrow operator "=>". Here is an example of a dictionary:
"

# ╔═╡ c4023896-a26b-4809-97f7-d085af96df5a
my_dictionary = Dict(:a=>1 ,:b=>2)

# ╔═╡ 2b6584b6-a432-40eb-8186-8c9f33b25cef
md" Values are accessed with the key as follows:"

# ╔═╡ de93684a-9d46-4999-b078-25b56a9661e8
my_dictionary[:a]

# ╔═╡ 75fa9237-4499-4653-b803-f5d957fe24e8
md"Because dictionaries are mutable, it is possible to change the value of an entry:"

# ╔═╡ 6f703da2-ee26-4a2a-a0be-f16e2f982fb2
my_dictionary[:b] = 100

# ╔═╡ 73e9fda3-b117-467e-b8d7-837b9fc1af31
md"
## Named Tuples
`Dictionaries` are quite convienent and flexible because they are mutable. However, they are not fast as other data structures. When performance is important, `NamedTuples` can be a good option to consider. `NamedTuples` are immutable ordered keyword value pairs. Here are some examples showing how to acess elements and how they can be used as function inputs:
"

# ╔═╡ e0327e7a-25b0-456f-b8b6-0d8c472b9ede
named_tuple = (x =1,y=3,z=4)

# ╔═╡ 7502dcba-45ae-4526-8276-041992599145
md"Unlike dictionaries, named tuples are ordered. As a result, elements can be accessed by numeric index:"

# ╔═╡ a2b234a2-2af2-4c4e-bf32-1cfb576a6865
named_tuple[1]

# ╔═╡ 18483020-a3fc-42ea-86e9-3b317edfe1ec
md"Alternatively, elements can be accessed by `key` or name:"

# ╔═╡ bff68ed1-0d32-49ab-83f4-715f827c3ae2
named_tuple.x

# ╔═╡ 02d2ace5-368b-40a2-83ca-478c718ac372
named_tuple[:x]

# ╔═╡ 851589c2-3751-4e3e-86f2-b85227aae210
md" The values and keys of a `NamedTuple` are provided with the functions `values` and `keys`:"

# ╔═╡ 1d081da4-4c8e-4592-bd35-06a7574e31ce
values(named_tuple)

# ╔═╡ 02bfbcaf-847b-4ad4-a187-106374d645c5
keys(named_tuple)

# ╔═╡ 2bed12e0-99f1-4a38-af10-37493417214e
md"
# Control Flow

Julia provides basic control flow similar to other languages

## Comparison

The following blocks of code demonstrate how to compare values of quantities

"

# ╔═╡ 3b15c2f0-ae3f-4155-99d7-1a8bcd6d284d
5 > 1

# ╔═╡ 17b027c4-d6ea-4003-9fa5-880576bb5489
1 ≤ 1 # alternatively 1 <= 1

# ╔═╡ e517f621-e79c-49d0-a2e0-b1874f11be76
1 == 2

# ╔═╡ e65aec82-fbdb-46b4-b489-a13c36b71dc5
1 ≠ 2 # alernatively, 1 != 2

# ╔═╡ 03d262ad-7e3f-4d28-b12b-893b13a1cd3d
isapprox(0.0, 0.0000000001, atol=1e-8)

# ╔═╡ 4981e112-bc2e-45aa-9bb4-cf13806a3fb9
md"

## if

`If` statements are used to conditionally execute code. Here is a simple if statement:
"

# ╔═╡ 6f68fc3d-f3ee-4c29-a960-3b3393aa65e1
begin
	num = 3
	if num < 3
	    string("$num is < 3")
	end
end

# ╔═╡ 18a8be44-503d-4a47-9f71-18e993e9c5c6
md"In this case, the string is not returned because the condition `num < 3` fails. However, if you change `num=3` to `num=2` in the cell above, the string will return"

# ╔═╡ 1ec03ec5-d2f0-4ea4-81d0-1324cf45b093
md"
## if-else

An `if-else` evaluates one of two sets of code, depending on whether the specified condition is true or not. Here is an example:
"

# ╔═╡ 40a19dc4-4deb-4411-8fa0-5d2aec531f57
begin
	num1 = 3
	if num1 < 3
	    string("$num1 is < 3")
	else
	    string("$num1 is not < 3")
	end
end

# ╔═╡ d4a69932-bc26-4018-aa7c-7d118732cf4c
md"
## elseif
The keyword `elseif` is used to specify multiple conditions. Consider the following example:
"

# ╔═╡ c87e14e6-8ebf-4abd-b4f0-4377be224b74
begin
	num3 = 3
	if num3 == 1
	    string("$num equals 1")
	elseif num3 == 2
	    string("$num equals 2")
	elseif num3 == 3
	    string("$num is too large")
	end
end

# ╔═╡ cd7b0046-e9dd-4e31-80b2-1c225376b89b
md" Julia provides special syntax for short if-else statements. For example:"

# ╔═╡ 15dd4226-0227-4e19-9cd8-d8800a8ddeea
begin
	num4 = 3
	num3 > 2 ? 1 : 2
end

# ╔═╡ 9b73ae11-b4f3-4a02-8b8c-721a4f0c4a56
md" As expected, the if statement above returns 1."

# ╔═╡ f5dea31b-136e-4fde-8384-efb559fbbeab
md"
## Boolean Logic

The following examples show how to use logical operators and, or, and, negation. `&&` are `||` short circuit versions of and and or operators, respectively. Short circuit operators are computationally efficient because they execute as few components as possible in order to evalute whether the composite statement is true. In the first example, isodd(3) is not evaluated because 4 < 4 is false, which means the entire satement is false. 
"

# ╔═╡ b1523e96-d769-4914-b147-42e14e2ba9a9
(4 < 4) && isodd(3)

# ╔═╡ 44a60a00-4a68-49c7-b68c-794c1a0357e0
(4 < 4) || isodd(3)

# ╔═╡ c571c1a6-4457-4abc-8bba-e3fc920655ac
(4 < 4) || !isodd(3)

# ╔═╡ 1f70e46c-71d1-4ecf-a5d8-1f14916b8bc8
md"
## For loops

Julia provides a rich syntax for using for loops to repeat operations. Unlike R, Matlab and Python, loops and vectorized code are similarly fast in Julia, allowing you to choose whichever style is most convienent and transparent for your purposes. Julia provides various ways to iterate over objects, similar to Python.

### Index

In the following example, `i` is incremented from 1 to length of `iter` and values of `iter1` are accessed with `i`. The values are collected in the array called `vals`.
"

# ╔═╡ 21c08857-6a80-46a6-9c17-8408ae727666
vals = []

# ╔═╡ 343fa00c-56a2-4f7a-b6c0-a36100bfe574
begin
	iter1 = 4:8
	for i in 1:length(iter1)
	    push!(vals, iter1[i])
	end
	vals
end

# ╔═╡ 36be7744-ccc4-4d2d-88ff-f2fb371f9e14
md"
Alternatively, it is possible to loop directly over the values of `iter1`.
"

# ╔═╡ 1e8958fb-a781-494d-9ffe-bcb1e31111b8
begin
	# empty vals
	empty!(vals)
	for v in iter1
	    push!(vals, v)
	end
	vals
end

# ╔═╡ ce782e13-78b8-4721-a1fc-fd5b366811b0
md"
### Enumerate

The iterator `enumerate` is used to ierate over index and value simultaneously. In the example below, click ▶ to see the output line by line.
"

# ╔═╡ 3ef703a1-c27e-4fd9-bff5-640dda63f928
begin
	empty!(vals)
	for (i,v) in enumerate(iter1)
	    push!(vals, string("i: $i v: $v"))
	end
	vals
end

# ╔═╡ ff2e0e47-15f2-4fc3-bdfd-432f9298f1ec
md"

### Zip

You can iterate over multiple containers of the same length with the iterator `zip`.

"

# ╔═╡ 7629d3a5-18b7-495e-beea-72548a84d0c4
begin
	iter2 = 14:20
	zipped = zip(iter1, iter2)
	empty!(vals)
	for (i,j) in zipped
	    push!(vals, string("i: ",i, " j: ",j))
	end
	vals
end

# ╔═╡ 5a98a896-2818-44bb-a7a5-2d749551a5a3
md"
### Nested Loops

The following code block illustrates how to write nested loops. The first value from `iter1` is selected, and all elements from `iter2` are iterated one by one. Next, the second value from `iter1` is selected and the process repeats. 
"

# ╔═╡ a6b386c8-1575-458b-a2f7-c8f9ce11811c
begin
	empty!(vals)
	for i in iter1
	    for j in iter2
	        push!(vals, (i,j))
	    end
	end
	vals
end

# ╔═╡ ccaa90e0-45f7-490b-b7f6-6a60f216a8b3
begin
	empty!(vals)
	for i in iter1, j in iter2
	        push!(vals, (i,j))
	end
	vals
end

# ╔═╡ d830a9b5-4379-4b7c-80ff-3a2681e092cf
md"
### List Comprehensions

List comprehensions provide a streamlined syntax for loop operations and automatically populations and returns an array much like a function could.
"

# ╔═╡ 8dc3c788-0ddc-45d1-be50-9e11bb2ce4ff
[(i,j) for (i,j) in zipped]

# ╔═╡ bc6c8c37-f0d1-4fe2-a450-1c8e111935a8
begin
	dict1 = Dict(:x=>1,:y=>3)
	empty!(vals)
	for (k,v) in dict1
	    push!(vals, string("key: ",k," value: ",v))
	end
	vals
end

# ╔═╡ e751fecb-5565-4358-9ab0-3c984f90f04e
md"
## While loops

A while loop is a block that begins with the keyword `while` and terminates with the keyword `end`. A while loop will continue to execute until the condition is no longer satisfied.
"

# ╔═╡ b1cd0299-0e09-4bb8-a236-0f9a1eb49949
begin
	v1 = 0
	while v1 < 10
	    v1 += 1
	    println("v: $v1")
	end
	v1
end

# ╔═╡ 6ad73cea-11a2-4d5b-bcd9-4f28284c1747
md"

# Functions
Julia provides two ways to specify a function.
## One-liners

In cases in which functions are short, using the syntax for a one-liner can be convienent. Consider the following example: 
"

# ╔═╡ 1283687a-b1a1-45b2-88d2-4363a9126617
silly_sum(a, b) = a + b

# ╔═╡ ff7ca66a-042b-44eb-8193-55be3b2e09f5
md"Let's pass the values 1 and 2 to `silly_sum`:"

# ╔═╡ 260976f1-4f53-4ba3-8652-f339ce7c16fe
silly_sum(1, 2)

# ╔═╡ 88bae89b-6d9d-4223-937a-3f50eb8a264f
md"
## Multi-line Functions

When functions contain many operations, multi-line function syntax is often more readable. Multi-line functions begin with the `function` keyword, followed by a function name and arguments, and end with the `end` keyword. The body of the function is placed in the middle. Here is an example:
"

# ╔═╡ 9dc8ff6a-cc4b-4506-ae35-41191444576f
function my_function(a, b)
    string("You entered: a = $a, b = $b")
end

# ╔═╡ 51735be0-2ef5-4eed-a8cd-c31b9ce2cc0f
my_function(1, "car")

# ╔═╡ a717bba7-b7e5-4e70-9d5d-1e8bb02f5109
md" 

## Arguments
As with many other languages, arguments for a function can be specified by position or keyword. 

### Positional
As shown above, positional arguments do not require keywords, but require the inputs to be in the correct order

### Keyword

Keyword arguments allow arguments to be specified by keyword with no order restrictions:
"

# ╔═╡ 7efb5994-e99c-46bd-8f7f-262aec7b41e5
keyword_function(;a, b) =  string("You entered: a = $a, b = $b")

# ╔═╡ a550a09e-330f-438d-9f80-673a8f010eac
md"A function with keywords is called as follows:"

# ╔═╡ 9f75af0e-4372-484d-9a76-a63541259325
keyword_function(b=1, a=40)

# ╔═╡ ace5dd6e-e1f4-474d-86ab-0bd5cb02df55
md"Notice that the arguments were not entered in order specified in the function definition"

# ╔═╡ df5dcd49-479c-41ea-8ccc-97942c1988b7
md"
#### Implicit keywords

When passing a variable as a keyword argument, this can be done explictly or implicitly if the variable name matches the keyword. The following is an example of explicit keyword passing:
"

# ╔═╡ d83e79ae-3b6e-4408-90e8-159e99e225a3
begin
	a = 1
	b = 2
	keyword_function(a=a, b=b)
end

# ╔═╡ 4b268e73-434d-4766-b01f-18eef22a0330
keyword_function(;b, a)

# ╔═╡ 1e1fff30-5721-46a6-a28a-bcc180e6f44a
md"
### Variable positional
Use the elipse $\dots$ after the last positional argument to make it variable length. All of these arguments for b are collected into a Tuple.
"

# ╔═╡ 50a085ad-ffc9-4215-a2cc-25de82c5a6d2
function vararg_function(a, b...)
    string("You entered: a = $a, b = $b")
end

# ╔═╡ b3d2e27e-42ce-444f-bfca-fabddfb30507
md"In the following example, we will pass five arguments to `vararg_function` to show that the last three arguments are grouped with `b`:"

# ╔═╡ 2a864107-e944-40b1-9e15-6f133c42df4a
vararg_function(1, 1, 2, 3, 4)

# ╔═╡ 3ef0ac78-c6b3-485a-9593-22d1e6a36f28
md"
### Variable keyword

Just as it is possible to have a variable length positional argument, it is possible to have a variable length keyword argument. The variable keyword argument must occur at the end of the argument list with `...`.
"

# ╔═╡ 9470bc90-ef5b-474c-a855-dbe4dd20468d
function kwvarg_function(;a, kwargs...)
    string("You entered: a = $a, kwargs = $kwargs")
end

# ╔═╡ 94f1b251-b8e8-4b24-a9eb-7b41545c3ea8
md" In the example below, keywords `c` and `b` are collected as pairs because they were not declared in the function `kwvarg_function."

# ╔═╡ 034dfb39-a8a4-410b-8677-9f2c8ae92f0e
kwvarg_function(b=2 ,c="dog", a= 1)

# ╔═╡ 85fb7761-4afa-46fc-9dcc-cd47c3bfe89f
md"
## Function Chaining

When multiple operations need to be performed, it is possible to use compound functions. Compound functions can be called in one of two ways. The first way involves wrapping successive functions around previous functions: `f_n(...f_2(f_1())...)`. In this example, function 1 is executed first, and the result of function 1 is the input into function 2, whose output becomes the input for function 3 and so forth. The downside to wrapping functions inside each other is that it quickly becomes difficult to read and debug. Chaining is a useful alternative: `f_1 |> f_2 |> ... |> f_n`. Moving from left to right, in the example below, the result of modulo 10 of x + 1 is passed to `sqrt` and the result of `sqrt` is passed to `exp`. 
"

# ╔═╡ 1b5db27a-15d8-49ce-b292-48fc5a1c86f4
begin
	x2 = 3
	mod(10, x2 + 1) |> sqrt |> exp
end

# ╔═╡ eae8315b-c40e-4ee4-8858-762a9bbbef3f
md"

# Broadcasting

In Julia, an array can be broadcast (or applied) across any function with the `.` operator. Consider the following function:
"

# ╔═╡ 1169509b-d1ed-4fc7-b344-37f211605b1f
function some_function(x)
    if x > 1
     x += 100.0
    elseif x < 0 
        x = sqrt(abs(x))
    end
    return x
end

# ╔═╡ b5d11f00-f07c-44dd-9df7-54ef24cff4a8
md"
In order to apply `some_function` to every element in the vector x, simply add `.` between the function name and the first parenthesis. 
"

# ╔═╡ bb183b21-3536-4ec9-8608-827f5a5b1313
begin
	x3 = [-1,0.0,2]
	some_function.(x3)
end

# ╔═╡ f98cfd18-32ff-4ee2-b39d-aa4c9a2e07a7
md"
Here are two more examples of broadcasting:
"

# ╔═╡ 148dc0dc-be01-471e-bd02-384313ef4b4f
x3 .> 0

# ╔═╡ cbc4c535-9cf8-4637-b694-d92f021a6fee
begin
	y = [1,2,3] 
	z = [1 3]
	y .+ z
end

# ╔═╡ aecfb3da-0ad8-4008-b447-0a844bce0a6c
md"
## in-place operations 

Julia provides a way to perform in place operations that mutate an existing array rather than creating a new copy. In the following example, elements in `my_array,` created above, are set to 4 without creating a new copy using the command `.=`"

# ╔═╡ 314d78f2-70d5-40f1-a52f-1256c1c95a76
my_array .= 4.0

# ╔═╡ 32c10052-f36f-4c9f-9b64-132772eff239
md"
Of course, inplace mutation of an array cannot be performed on an array that has not already been defined:
"

# ╔═╡ d9953f04-8a28-4add-bf35-c6056d55d155
undefined_array .= 4.0

# ╔═╡ 5655a7ae-286a-4408-88cd-97fc7a910fea
md"
# Multiple Dispatch

Multiple dispatch allows different versions of a function evoked at runtime depending on the type of the inputs. Suppose we have a dog type and a human type that move, but in different ways. We can also use abstract types to define default behavior. 
"

# ╔═╡ 0c1d02bd-3906-48f3-93b4-dffbd494ca0c
begin
	abstract type Animal end
	
	struct Chinchilla <: Animal
	    name::String
	end
	
	struct Human <: Animal
	    name::String
	end
	
	struct SomeAnimal <: Animal
	  name::String  
	end
end

# ╔═╡ 299c52c7-66cb-4786-85d4-89e3e2d617cb
begin
	dog = Chinchilla("Chinchi")
	human = Human("Sam")
	some_animal = SomeAnimal("Bonkers")
end

# ╔═╡ a34e683c-b26f-4a83-90b3-3ff04cfdf179
begin
		move(dog::Chinchilla) = string(dog.name, " runs on four legs")
		move(human::Human) = string(human.name, " runs on two legs")
		move(animal::Animal) = string(animal.name, " has default movement")
		move(dog)
end

# ╔═╡ 2892fd8a-6bec-47f6-85cd-9a5918375484
move(human)

# ╔═╡ f56b02db-5834-401f-960c-ce630d33ff6a
move(some_animal)

# ╔═╡ e7487c62-e0f2-474a-8e2e-ac22b6a18721
md"

## Interactions among objects

One of the convenient features of multiple dispatch is that it easy to (1) create default behavior, (2) create specialized behavior, (3) add new methods and types, and (4) implement behavior for arbitrary combinations of interacting objects. This capability provides an elegant solution to the [expression problem](https://en.wikipedia.org/wiki/Expression_problem) in which new types and methods must be added without type casting, modifying existing code, or code duplication. This is particularly useful when developing simulations in which all combinations of objects interact with each other. 

The example below show how to define interaction methods for animals and trainers. The code defines a `Dog` and `Cat` type, and a `GoodTrianer` and `BadTrainer` type, which are subtypes of the abstract `Trainer` type. The code at the bottom of the code block iterates through all four possible combinations of animal-trainer interactions.

### Animal Types

In the cell below, we will declare two sub-types of `Animal`: a `Dog` type and a `Cat` type.
"

# ╔═╡ 16650eb1-e624-4aa3-9da5-661a7cb13bc7
begin
	
	struct Dog <: Animal
	
	end
	
	struct Cat <: Animal
	
	end
end

# ╔═╡ d3c22202-3731-4bef-aeef-e63a09453b76
md" 

## Trainer Types

Now we will define an abstract type called `Trainer` and define sub-types `GoodTrainer` and `BadTrainer`.
"

# ╔═╡ b1512c6b-195d-4fcf-bc26-2daf8bbff92a
begin
	abstract type Trainer end
	
	struct GoodTrainer <: Trainer
	
	end
	
	struct BadTrainer <: Trainer 
	
	end
end

# ╔═╡ 241aeae2-8a4f-442d-83a0-6d7085e40c39
md"
### Interaction  Methods
Here we define various interaction methods. The first method defined default behavior and is sometimes called a fallback method. This method is invoked when more specific methods do not exist. The other methods define various combinations of specific and generic behavior for animals and trainers. Each method out puts the input types ans well as the method of dispatch.
"

# ╔═╡ 16227ac9-c5ce-4c01-8979-b33f37c43450
begin
	function interact(animal::Animal, trainer::Trainer) 
		return string(
			typeof(animal),
			" interacts with ", typeof(trainer), 
			". Dispatch: generic default"
		)
	end
	
	function interact(animal::Animal, trainer::GoodTrainer)
		return string(
			typeof(animal), " interacts with ", 
			typeof(trainer), 
			". Dispatch: Good trainer, generic animal"
		)
	end
		
	function interact(animal::Dog, trainer::BadTrainer) 
		return string(
			typeof(animal), " interacts with ", 
			typeof(trainer), 
			". Dispatch: bad trainer and dog"
		)
	end
		
	function interact(animal::Cat, trainer::BadTrainer)
		return string(
			typeof(animal), " interacts with ", 
			typeof(trainer), 
			". Dispatch: bad trainer and cat"
		)
	end
end

# ╔═╡ 969fc1cd-c022-4689-83f3-f7fc2525e466
md"
### Add new Animal

Adding a new type is simple. Let's add a type for a rat.
"

# ╔═╡ a58d0b14-130f-4f76-9aeb-a8a1b3b6db7f
struct Rat <: Animal

end

# ╔═╡ 56c8fb8d-4289-4dc0-b1ac-24be7e8098e1
md"
As demonstrated below, the new rat type automatically works with the generic function. No modification is required.
"

# ╔═╡ b6c87d82-ce12-41a4-b8a7-eb27b2d9c498
md"
### Add new method
If the generic method does not impliment the appropriate behavior,  a specific method for the rat type can be easily added. Copy and paste the following code into the cell below and the output to `interact` will update to reflect the changes.
"

# ╔═╡ 9814617c-91b0-429b-8f7a-b51cd0535590
interact(animal::Rat, trainer::Trainer) = string(typeof(animal), " interacts with ", typeof(trainer), ". Dispatch: rat and any trainer")

# ╔═╡ 8532da17-e951-43a2-9daf-6746875cbe6f
begin
	animals = [Cat(), Dog()]
	trainers = [GoodTrainer(), BadTrainer()]
	empty!(vals)
	for trainer in trainers
	    for animal in animals 
	        result = interact(animal, trainer)
			push!(vals, result)
	    end 
	end
	vals
end

# ╔═╡ 648f8b22-335f-4141-9e23-3963b1d6847d
interact(Rat(), BadTrainer())

# ╔═╡ 5a6d1250-0279-4bbf-959a-be26095f0db0
md"
Now lets execute the new method
"

# ╔═╡ d4f50ca8-5e9b-41d7-bb34-ff607e636478
interact(Rat(), BadTrainer())

# ╔═╡ fbea4d41-1079-424c-a730-23002474d14d
my_mutable = MyMutable(1)

# ╔═╡ 6c58a4e9-e636-4281-aa47-a232abcac8a9
my_mutable = MyMutable(3)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.62"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "dae24c494f1947755dffc309d3fccc93cf0772af"

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
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

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
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

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

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

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

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

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
# ╟─43366b1a-541e-11ec-0545-15db49af78b7
# ╟─8c35371c-e3a1-4204-8b7d-45edd52ebc1b
# ╟─cd684ca4-6483-4407-84f9-24a7757c096f
# ╟─cda2f5d7-1e04-46c2-830b-60e55841bfb3
# ╟─352f3570-5344-4629-8ed1-bed5c36c25a7
# ╟─5bb33a0d-dabb-4988-97f6-539458c9a639
# ╠═d1e1ad29-088a-4315-9f65-4c32417ec868
# ╟─775f90c2-222c-43b2-a62c-e3485ba0c628
# ╠═513f6e52-b64f-4d55-97b1-3dfcafe31d73
# ╟─4ec22c90-2256-4f0c-967f-c8869c9abca1
# ╟─203307dd-1e36-4c32-9349-c45a5d5e55b0
# ╠═f7bd7a61-36e5-4692-8f79-40a256f4a6a1
# ╟─7b60f7db-56dd-4e30-87bc-0b202c504a74
# ╠═558ef5a3-9ead-4de4-b164-be783d951f5b
# ╟─69c9540f-b1bb-4840-b8a9-7acaffa1c1f5
# ╟─9a0f0a84-1d5b-4715-b041-42cd6940987f
# ╠═c15068d4-afbc-44b1-b092-01a3f45456a6
# ╟─c1e51d62-ba32-43c0-b44a-3c7e820579fe
# ╠═01e83efb-045b-431e-95f5-02f4ad78a164
# ╟─2235469f-83f7-4bad-bd0d-4a5ced52d96a
# ╠═0dcaef7c-947b-4034-89c6-bd55b1dbe457
# ╟─9d032dc2-fcaf-49e7-b991-ef3a69e3c665
# ╠═a3736648-5890-4a2b-9211-bb11ca462e6c
# ╟─c22c2af3-9abe-4266-b103-105650223045
# ╠═d6fa8f6c-9b79-464f-88c5-29c13a43c09f
# ╟─5d6f1cb7-21f9-4886-b666-3900e84c2845
# ╟─fa8ccde6-22b7-4afc-a08f-ca07833ee865
# ╠═14b6c9a0-ac08-4abf-a477-219632348310
# ╟─406dfacf-f33d-4e91-a6a7-e60df4839def
# ╠═5f9e8a03-488e-4cb9-a1a8-26e6883c0f64
# ╟─544f84c3-739c-41f6-b43c-1a272f73f85a
# ╠═cfb9249b-9313-4807-8503-fbd4e01aab48
# ╟─60d8c7b9-af8b-4d83-a2d0-2c3d873eff6c
# ╠═001f99c3-424e-43f2-b2f3-90a49003b38a
# ╟─ff531449-dc5f-444f-b256-7db310da7d01
# ╠═fbea4d41-1079-424c-a730-23002474d14d
# ╟─8db9a982-4cd1-49b1-94f9-700c99f00e79
# ╠═90f6ea01-8282-4d46-8a08-c2525d1ef15e
# ╟─ef4f1b15-0454-4a78-82c5-9caf3aa3f872
# ╠═6c58a4e9-e636-4281-aa47-a232abcac8a9
# ╟─9f927455-1d6d-47d1-91a7-6003ea59d366
# ╟─b78132b7-5944-4c86-91dd-aa1fcc99b9f4
# ╟─ffcff15c-c6a1-4dcc-a425-c10c285854ee
# ╠═9b4cd942-fd4b-4095-ac5d-a5800f2cd5d5
# ╟─b51fe7d1-9d3b-432a-adc0-573d657756aa
# ╠═cbb8d9ce-bd5d-4ca9-9edc-188e4b2393da
# ╟─17ed7e45-ac79-4aac-be14-4611455d9747
# ╠═2a42120f-884d-4bfe-a691-105ee10ff2ce
# ╟─2aa517d3-4f3a-42da-8b3a-a34decc8f3e0
# ╟─23bc5cbe-535d-4942-81ad-b397d205eaf8
# ╠═d04b2bd8-2c56-453b-82d7-066219d2156e
# ╟─fe0c8f58-b4b6-4ab3-8f55-7c05e5aa97b7
# ╠═340fb788-3502-474e-a2ef-f2cbd19d8fa4
# ╠═c7b17819-b35c-4a9e-861e-6e16b853ad53
# ╟─d2578db2-ffc8-4d37-a333-354f08bbf811
# ╠═132d7bd7-671b-4256-9a76-8453071d7788
# ╟─fa8c2b80-3885-4035-a3f0-d641e07261be
# ╠═364a303c-7491-4e75-a8cc-7a736b6a8265
# ╟─aec8d779-65d0-4b55-a869-30778a87dd0b
# ╠═9e909bbf-6c2e-416f-b499-dbc39d2a45b7
# ╟─7c3108ab-9778-40e7-ae21-3a02e58cdf3f
# ╟─410772e1-015b-46c9-8746-37b20b18b5f0
# ╠═04da2ede-fd5d-4575-92e6-c529e1ab900a
# ╟─d180db94-1657-46cc-a15e-f214152b49d6
# ╠═33167674-0e58-4492-8ff1-7401e268c88e
# ╟─09f2b339-8dfd-45ae-bbf9-f9b1ccba8f39
# ╠═5c249a3d-bb89-453e-ac96-5ff91ea881df
# ╟─d9694ad9-7467-4f0e-8f3b-771c4cdcce7e
# ╠═0cc514c1-9bf1-473d-bb92-1369a4863675
# ╟─59c4f8eb-9b37-4461-9324-cd83c8b50ce4
# ╠═9794d991-fc63-49df-bafd-fa3812795c7d
# ╠═2f5be4df-24e0-414d-9264-f658beb0c90e
# ╟─00393d8b-0b3e-453d-b4d6-d9a26e5ace41
# ╠═ff7fd34f-3107-448c-b450-5c6e9c71b22e
# ╟─7d031965-6a4a-4d8a-826f-bc92ce4734bd
# ╠═13b2509b-bf31-4747-8f44-8754fe8e3f71
# ╟─b0d4f161-5461-41e0-974c-88caeff02fdc
# ╠═b075d869-0850-472a-8f6f-29e0907cb082
# ╟─29667264-6aea-4694-a336-3214c28520f2
# ╠═f98231c1-b575-443d-b99b-2579a642c829
# ╟─6127da88-a63a-4582-a4de-2661d7a71f0f
# ╠═bff2e4bd-8a55-4745-a9ae-c89bcbdfdae9
# ╟─5e4450cc-e657-4609-8417-049e973ae4ef
# ╟─b5f634e3-194a-4de1-8c93-5f8c8c2e4e59
# ╠═c8727f82-2a14-47a6-8656-42180d1fc4e9
# ╟─9bafc6cf-8889-4a68-9fc7-a6705b5cbb4f
# ╠═c4023896-a26b-4809-97f7-d085af96df5a
# ╟─2b6584b6-a432-40eb-8186-8c9f33b25cef
# ╠═de93684a-9d46-4999-b078-25b56a9661e8
# ╟─75fa9237-4499-4653-b803-f5d957fe24e8
# ╠═6f703da2-ee26-4a2a-a0be-f16e2f982fb2
# ╟─73e9fda3-b117-467e-b8d7-837b9fc1af31
# ╠═e0327e7a-25b0-456f-b8b6-0d8c472b9ede
# ╟─7502dcba-45ae-4526-8276-041992599145
# ╠═a2b234a2-2af2-4c4e-bf32-1cfb576a6865
# ╟─18483020-a3fc-42ea-86e9-3b317edfe1ec
# ╠═bff68ed1-0d32-49ab-83f4-715f827c3ae2
# ╠═02d2ace5-368b-40a2-83ca-478c718ac372
# ╟─851589c2-3751-4e3e-86f2-b85227aae210
# ╠═1d081da4-4c8e-4592-bd35-06a7574e31ce
# ╠═02bfbcaf-847b-4ad4-a187-106374d645c5
# ╟─2bed12e0-99f1-4a38-af10-37493417214e
# ╠═3b15c2f0-ae3f-4155-99d7-1a8bcd6d284d
# ╠═17b027c4-d6ea-4003-9fa5-880576bb5489
# ╠═e517f621-e79c-49d0-a2e0-b1874f11be76
# ╠═e65aec82-fbdb-46b4-b489-a13c36b71dc5
# ╠═03d262ad-7e3f-4d28-b12b-893b13a1cd3d
# ╟─4981e112-bc2e-45aa-9bb4-cf13806a3fb9
# ╠═6f68fc3d-f3ee-4c29-a960-3b3393aa65e1
# ╟─18a8be44-503d-4a47-9f71-18e993e9c5c6
# ╟─1ec03ec5-d2f0-4ea4-81d0-1324cf45b093
# ╠═40a19dc4-4deb-4411-8fa0-5d2aec531f57
# ╟─d4a69932-bc26-4018-aa7c-7d118732cf4c
# ╠═c87e14e6-8ebf-4abd-b4f0-4377be224b74
# ╟─cd7b0046-e9dd-4e31-80b2-1c225376b89b
# ╠═15dd4226-0227-4e19-9cd8-d8800a8ddeea
# ╟─9b73ae11-b4f3-4a02-8b8c-721a4f0c4a56
# ╟─f5dea31b-136e-4fde-8384-efb559fbbeab
# ╠═b1523e96-d769-4914-b147-42e14e2ba9a9
# ╠═44a60a00-4a68-49c7-b68c-794c1a0357e0
# ╠═c571c1a6-4457-4abc-8bba-e3fc920655ac
# ╟─1f70e46c-71d1-4ecf-a5d8-1f14916b8bc8
# ╠═21c08857-6a80-46a6-9c17-8408ae727666
# ╠═343fa00c-56a2-4f7a-b6c0-a36100bfe574
# ╟─36be7744-ccc4-4d2d-88ff-f2fb371f9e14
# ╠═1e8958fb-a781-494d-9ffe-bcb1e31111b8
# ╟─ce782e13-78b8-4721-a1fc-fd5b366811b0
# ╠═3ef703a1-c27e-4fd9-bff5-640dda63f928
# ╟─ff2e0e47-15f2-4fc3-bdfd-432f9298f1ec
# ╠═7629d3a5-18b7-495e-beea-72548a84d0c4
# ╟─5a98a896-2818-44bb-a7a5-2d749551a5a3
# ╠═a6b386c8-1575-458b-a2f7-c8f9ce11811c
# ╠═ccaa90e0-45f7-490b-b7f6-6a60f216a8b3
# ╟─d830a9b5-4379-4b7c-80ff-3a2681e092cf
# ╠═8dc3c788-0ddc-45d1-be50-9e11bb2ce4ff
# ╠═bc6c8c37-f0d1-4fe2-a450-1c8e111935a8
# ╟─e751fecb-5565-4358-9ab0-3c984f90f04e
# ╠═b1cd0299-0e09-4bb8-a236-0f9a1eb49949
# ╟─6ad73cea-11a2-4d5b-bcd9-4f28284c1747
# ╠═1283687a-b1a1-45b2-88d2-4363a9126617
# ╟─ff7ca66a-042b-44eb-8193-55be3b2e09f5
# ╠═260976f1-4f53-4ba3-8652-f339ce7c16fe
# ╟─88bae89b-6d9d-4223-937a-3f50eb8a264f
# ╠═9dc8ff6a-cc4b-4506-ae35-41191444576f
# ╠═51735be0-2ef5-4eed-a8cd-c31b9ce2cc0f
# ╟─a717bba7-b7e5-4e70-9d5d-1e8bb02f5109
# ╠═7efb5994-e99c-46bd-8f7f-262aec7b41e5
# ╟─a550a09e-330f-438d-9f80-673a8f010eac
# ╠═9f75af0e-4372-484d-9a76-a63541259325
# ╟─ace5dd6e-e1f4-474d-86ab-0bd5cb02df55
# ╟─df5dcd49-479c-41ea-8ccc-97942c1988b7
# ╠═d83e79ae-3b6e-4408-90e8-159e99e225a3
# ╠═4b268e73-434d-4766-b01f-18eef22a0330
# ╟─1e1fff30-5721-46a6-a28a-bcc180e6f44a
# ╠═50a085ad-ffc9-4215-a2cc-25de82c5a6d2
# ╟─b3d2e27e-42ce-444f-bfca-fabddfb30507
# ╠═2a864107-e944-40b1-9e15-6f133c42df4a
# ╟─3ef0ac78-c6b3-485a-9593-22d1e6a36f28
# ╠═9470bc90-ef5b-474c-a855-dbe4dd20468d
# ╟─94f1b251-b8e8-4b24-a9eb-7b41545c3ea8
# ╠═034dfb39-a8a4-410b-8677-9f2c8ae92f0e
# ╟─85fb7761-4afa-46fc-9dcc-cd47c3bfe89f
# ╠═1b5db27a-15d8-49ce-b292-48fc5a1c86f4
# ╟─eae8315b-c40e-4ee4-8858-762a9bbbef3f
# ╟─1169509b-d1ed-4fc7-b344-37f211605b1f
# ╟─b5d11f00-f07c-44dd-9df7-54ef24cff4a8
# ╠═bb183b21-3536-4ec9-8608-827f5a5b1313
# ╟─f98cfd18-32ff-4ee2-b39d-aa4c9a2e07a7
# ╠═148dc0dc-be01-471e-bd02-384313ef4b4f
# ╠═cbc4c535-9cf8-4637-b694-d92f021a6fee
# ╟─aecfb3da-0ad8-4008-b447-0a844bce0a6c
# ╠═314d78f2-70d5-40f1-a52f-1256c1c95a76
# ╟─32c10052-f36f-4c9f-9b64-132772eff239
# ╠═d9953f04-8a28-4add-bf35-c6056d55d155
# ╟─5655a7ae-286a-4408-88cd-97fc7a910fea
# ╠═0c1d02bd-3906-48f3-93b4-dffbd494ca0c
# ╠═a34e683c-b26f-4a83-90b3-3ff04cfdf179
# ╠═299c52c7-66cb-4786-85d4-89e3e2d617cb
# ╠═2892fd8a-6bec-47f6-85cd-9a5918375484
# ╠═f56b02db-5834-401f-960c-ce630d33ff6a
# ╟─e7487c62-e0f2-474a-8e2e-ac22b6a18721
# ╠═16650eb1-e624-4aa3-9da5-661a7cb13bc7
# ╟─d3c22202-3731-4bef-aeef-e63a09453b76
# ╠═b1512c6b-195d-4fcf-bc26-2daf8bbff92a
# ╟─241aeae2-8a4f-442d-83a0-6d7085e40c39
# ╠═16227ac9-c5ce-4c01-8979-b33f37c43450
# ╠═8532da17-e951-43a2-9daf-6746875cbe6f
# ╟─969fc1cd-c022-4689-83f3-f7fc2525e466
# ╠═a58d0b14-130f-4f76-9aeb-a8a1b3b6db7f
# ╟─56c8fb8d-4289-4dc0-b1ac-24be7e8098e1
# ╠═648f8b22-335f-4141-9e23-3963b1d6847d
# ╟─b6c87d82-ce12-41a4-b8a7-eb27b2d9c498
# ╠═9814617c-91b0-429b-8f7a-b51cd0535590
# ╟─5a6d1250-0279-4bbf-959a-be26095f0db0
# ╟─d4f50ca8-5e9b-41d7-bb34-ff607e636478
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
