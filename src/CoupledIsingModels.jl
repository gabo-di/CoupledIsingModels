module CoupledIsingModels
# ---- imports ----
using Random
using StaticArrays
using LinearAlgebra
using SparseArrays

# ---- includes ----
include("abstracttypes.jl")
include("utils.jl")
include("spinglassmodel.jl")
include("latticemodel.jl")
include("updatingalgorithms.jl")
include("boundarykind.jl")

# ---- exports ----
export SpinGlassIsingModel,
       LittleAlgorithm,
       MetropolisHastingsAlgorithm,
       GlauberAlgorithm,
       updateIsingModel!,
       PeriodicBoundary,
       makeNearestNeighboursConnections,
       LatticeIsingModel
       


end
