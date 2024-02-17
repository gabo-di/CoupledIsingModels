module CoupledIsingModels
# ---- imports ----
using Random
using StaticArrays
using LinearAlgebra

# ---- includes ----
include("abstracttypes.jl")
include("spinglassmodel.jl")
include("utils.jl")

# ---- exports ----
export SpinGlassIsingModel,
       LittleUpdate!,
       MetropolisHastingsUpdate!,
       GlauberUpdate!

end
