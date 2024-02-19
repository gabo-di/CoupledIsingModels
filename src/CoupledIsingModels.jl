module CoupledIsingModels
# ---- imports ----
using Random
using StaticArrays
using LinearAlgebra

# ---- includes ----
include("abstracttypes.jl")
include("utils.jl")
include("spinglassmodel.jl")
include("latticemodel.jl")
include("updatingalgorithms.jl")

# ---- exports ----
export SpinGlassIsingModel,
       LittleAlgorithm,
       MetropolisHastingsAlgorithm,
       GlauberAlgorithm,
       updateIsingModel!


end
