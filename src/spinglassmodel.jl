"""
    Spin Glass Ising Model with a simple spin chain
    the Hamiltonian is given by 
    H(s_1, s_0) = - ( H + J' * s_0)' * s_1 =  - ( H' * s_1 + s_0' * J * s_1 )
    where in general s_1 is at time t=1 and s_0 is at time t=0
"""
struct SpinGlassIsingModel{T, M} <: AbstractSpinGlassIsingModel{T, 1, M}
    # possible spin values
    _s::SVector{M,T}
    # spin array
    s::AbstractArray{T, 1}
    # size of the spin array
    sze::Int
    # the external field acting on each spin, has same size as spin array
    H::AbstractArray{T, 1}
    # the interaction matrix between spins, has shape = (sze, sze)
    J::AbstractArray{T, 2}
end



# ----- How to declare a Spin Glass Ising Model -----

"""
    # number type, in general a kind of Float 
    T
    # possible spin values
    _s::SVector{M,T}
    # size of the spin array
    sze::int  
    # random number generator
    rng::AbstractRNG
    
    
    J and H are filled with zeros of type T 
    s is filled with random values drawn from _s
"""
function SpinGlassIsingModel(::Type{T}, _s::SVector{M, T}, sze::Int, rng::Random.AbstractRNG) where{M, T}
    SpinGlassIsingModel{T, M}(_s, rand(rng, _s, sze), sze, zeros(T, sze), zeros(T, sze, sze))
end

function SpinGlassIsingModel(::Type{T}, _s::AbstractArray{T,1}, sze::Int, rng::Random.AbstractRNG) where{T}
    SpinGlassIsingModel(T, SVector(_s...), sze, rng)
end

function SpinGlassIsingModel(::Type{T}, _s::NTuple{M, T}, sze::Int, rng::Random.AbstractRNG) where{M, T}
    SpinGlassIsingModel(T, SVector(_s), sze, rng)
end

"""
    # number type 
    T
    # size of the spin array
    sze::int  
    
    _s is assumed to be (T(-1), T(1))
    J and H are filled with zeros of type T 
    s is filled with random values drawn from _s
"""
function SpinGlassIsingModel(::Type{T}, sze::Int, rng=Random.default_rng()) where {T}
    SpinGlassIsingModel(T, SVector(T(-1),T(1)), sze, rng)
end



# ----- How to update a Spin Glass Ising Model -----

"""
    Little parallel update
    See:
    Collective Properties of Neural Networks: A Statistical Physics Approach - P. Peretto
    section 5
"""
function LittleUpdate!(ising::SpinGlassIsingModel{T,2}, Beta::T, rng::AbstractRNG) where {T}
    h = (ising.H + ising.J'*ising.s) * Beta * (ising._s[2] - ising._s[1])
    r = rand(rng, T, ising.sze)
    ising.s .= ifelse.( sigmoid.(h)>=r, ising._s[2], ising._s[1] )
    return nothing
end


function LittleUpdate!(ising::SpinGlassIsingModel{T,M}, Beta::T, rng::AbstractRNG) where {T, M}
    h = (ising.H + ising.J'*ising.s)
    @inbounds Threads.@threads for i in 1:ising.sze
        r = rand(rng, T) 
        y = cumsum(map(x->exp(h[i] * Beta * x), ising._s))
        idx = findfirst( x -> x/y[end] >= r, y)
        ising.s[i] = ising._s[idx]
    end
    return nothing
end


"""
    Metropolis - Hastings update
"""
function MetropolisHastingsUpdate!(ising::SpinGlassIsingModel{T, M}, Beta::T, rng::AbstractRNG) where{T, M}
    i = rand(rng, 1:ising.sze)
    h = ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])
    r = rand(rng, T)
    new_s = rand(rng, ising._s)
    ising.s[i] = ifelse(exp( Beta * h * (new_s - ising.s[i]) ) >= r, new_s, ising.s[i])
    return nothing
end

function MetropolisHastingsUpdate!(ising::SpinGlassIsingModel{T, 2}, Beta::T, rng::AbstractRNG) where{T}
    i = rand(rng, 1:ising.sze)
    s = sum(ising._s) - ising.s[i] # (s_1  - s_i) + (s_-1 - s_i) + s_i -> flip s_i
    h = (ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])) *
            Beta * (s - ising.s[i])
    r = rand(rng, T)
    ising.s[i] = ifelse(exp( h ) >= r, s, ising.s[i])
    return nothing
end


"""
    Glauber update
"""
function GlauberUpdate!(ising::SpinGlassIsingModel{T, 2}, Beta::T, rng::AbstractRNG) where {T}
    i = rand(rng, 1:ising.sze)
    h = (ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])) *
            Beta * (ising._s[2] - ising._s[1])
    r = rand(rng, T)
    ising.s[i] = ifelse(sigmoid( h ) >= r, ising._s[2], ising._s[1])
    return nothing
end

function GlauberUpdate!(ising::SpinGlassIsingModel{T, M}, Beta::T, rng::AbstractRNG) where{T, M}
    i = rand(rng, 1:ising.sze)
    h = ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])
    r = rand(rng, T) 
    y = cumsum(map(x->exp(h * Beta * x), ising._s))
    idx = findfirst( x -> x/y[end] >= r, y)
    ising.s[i] = ising._s[idx]
    return nothing
end



# ----- How to calculate the fields of Spin Glass Ising Model -----

"""
    instant magnetization
"""
function magnetization(ising::SpinGlassIsingModel{T,M}, Beta::T, field::MagnetizationField{T},
    upd_alg::AbstractUpdateIsingModel) where {T,M}
    return ising.s 
end

"""
    instant energy
    for Serial Update methods
"""
function energy(ising::SpinGlassIsingModel{T,M}, Beta::T, field::EnergyField{T},
    upd_alg::AbstractSerialUpdate) where {T,M}
    return - ( ising.H + ising.J' * ising.s)' * ising.s 
end

"""
    instant energy
    for Parallel Update
    TODO
    see eq 38 of P. Peretto 1984 "Collective Properties of Neural Networks: A Statistical Physicis Approach"
"""
function energy(ising::SpinGlassIsingModel{T,M}, Beta::T, field::EnergyField{T},
    upd_alg::AbstractParallelUpdate) where {T,M}
    return - ( ising.H + ising.J' * ising.s)' * ising.s 
end
