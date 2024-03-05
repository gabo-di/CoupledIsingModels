# ----- structs with the computation type
struct SingleCPU <: AbstractComputation
end

struct ThreadsCPU <: AbstractComputation
end

struct SingleGPU <: AbstractComputation
end


# ----- General update

function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta, alg::AbstractUpdateIsingModel) where {T, N, M}
    rng = Random.default_rng()
    updateIsingModel!(ising, T(Beta), alg, rng)
end


# ----- Simple Methods

struct MetropolisHastingsAlgorithm <: AbstractSerialUpdate
    #cache
end

function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta::T, alg::MetropolisHastingsAlgorithm, rng::Random.AbstractRNG) where {T, N, M}
    MetropolisHastingsUpdate!(ising, Beta, rng)
end

struct LittleAlgorithm <: AbstractParallelUpdate
    #cache
end

function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta::T, alg::LittleAlgorithm, rng::Random.AbstractRNG) where {T, N, M}
    LittleUpdate!(ising, Beta, rng)
end

struct GlauberAlgorithm <: AbstractSerialUpdate
    #cache
end

function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta::T, alg::GlauberAlgorithm, rng::Random.AbstractRNG) where {T, N, M}
    GlauberUpdate!(ising, Beta, rng)
end


# ----- Checkerboard Methods

function makeCheckerboardIndeces(shp::NTuple{1,Int})
    i_o = collect(1:2:shp[1])
    i_e = collect(2:2:shp[1])
    return i_o, i_e 
end

function makeCheckerboardIndeces(shp::NTuple{2,Int})
    sze = prod(shp) 
    i_o = Int[]
    sizehint!(i_o, div(sze,2))
    i_e = Int[]
    sizehint!(i_e, div(sze,2))

    arr = reshape(collect(1:sze), shp)

    for i_2 in axes(arr,2)
        for i_1 in axes(arr,1)
            if mod(i_2 + i_1,2)==0
                push!(i_o, arr[i_1,i_2])
            else
                push!(i_e, arr[i_1,i_2])
            end
        end
    end
    return i_o, i_e
end

struct CheckerboardMetropolisAlgorithm{T, C<:AbstractComputation} <: AbstractSerialUpdate
    sze::Int
    i_o::AbstractArray{Int,1}
    i_e::AbstractArray{Int,1}
    cmp::C
end

function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta::T, alg::CheckerboardMetropolisAlgorithm{T, C}, rng::Random.AbstractRNG) where {T, N, M, C}
    CheckerboardMetropolisUpdate!(ising, Beta, rng, alg)
end


struct CheckerboardGlauberAlgorithm{T, C<:AbstractComputation} <: AbstractSerialUpdate
    sze::Int
    i_o::AbstractArray{Int,1}
    i_e::AbstractArray{Int,1}
    cmp::C
end

function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta::T, alg::CheckerboardGlauberAlgorithm{T, C}, rng::Random.AbstractRNG) where {T, N, M, C}
    CheckerboardGlauberUpdate!(ising, Beta, rng, alg)
end
