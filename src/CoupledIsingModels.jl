module CoupledIsingModels
# ---- imports ----
using Random
using StaticArrays
using LinearAlgebra
using SparseArrays
using Statistics

# ---- includes ----
include("abstracttypes.jl")
include("utils.jl")
include("updatingalgorithms.jl")
include("boundarykind.jl")
include("fields.jl")
include("meanalgorithms.jl")
include("spinglassmodel.jl")
include("latticemodel.jl")

# ---- exports ----
export SpinGlassIsingModel,
       LatticeIsingModel,

       makeNearestNeighboursConnections,
       PeriodicBoundary,

       LittleAlgorithm,
       MetropolisHastingsAlgorithm,
       GlauberAlgorithm,
       CheckerboardMetropolisAlgorithm,
       CheckerboardGlauberAlgorithm,

       updateIsingModel!,

       MagnetizationField,
       LocalEnergyField,
       InteractionEnergyField,
       MeanFields,

       TemporalAverageAlgorithm,
       calculateFieldsAverage!
       


end
