function updateIsingModel!(ising::AbstractIsingModel{T, N, M}, Beta, alg::AbstractUpdateIsingModel) where {T, N, M}
    rng = Random.default_rng()
    updateIsingModel!(ising, T(Beta), alg, rng)
end

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

