"""
    Lattice Ising Model with a simple spin chain
    the Hamiltonian is given by 
    H(s_1, s_0) = - ( H + J' * s_0)' * s_1 =  - ( H' * s_1 + s_0' * J * s_1 )
    where in general s_1 is at time t=1 and s_0 is at time t=0
"""
struct LatticeIsingModel{T, N, M} <: AbstractLatticeIsingModel{T, N, M}
    # possible spin values
    _s::SVector{M,T}
    # spin array
    s::AbstractArray{T, 1}
    # shape of the spin array
    shp::NTuple{N,Int}
    # size of the spin array
    sze::Int
    # the external field acting on each spin, has same size as spin array
    H::AbstractArray{T, 1}
    # the interaction matrix between spins, has shape = (sze, sze)
    J::AbstractArray{T, 2}
    # boundaries condition
    boundaries::NTuple{N, AbstractBoundaryKind}
end



# ----- How to declare a Lattice Ising Model -----

"""
    # number type, in general a kind of Float 
    T
    # possible spin values
    _s::SVector{M,T}
    # shape of the spin array
    shp::NTuple{N, Int}  
    # random number generator
    rng::AbstractRNG
    
    J and H are filled with zeros of type T 
    s is filled with random values drawn from _s
"""
function LatticeIsingModel(::Type{T}, _s::SVector{M, T}, shp::NTuple{N, Int}, rng::Random.AbstractRNG) where{M, N, T}
    sze = prod(shp)
    LatticeIsingModel{T, N, M}(
        _s, 
        rand(rng, _s, sze),
        shp,
        sze,
        zeros(T, sze),
        spzeros(T, sze, sze),
        ntuple(i->PeriodicBoundary(shp[i]), N)
        )
end

function LatticeIsingModel(::Type{T}, _s::AbstractArray{T,1}, sze::Int, rng::Random.AbstractRNG) where{T}
    LatticeIsingModel(T, SVector(_s...), (sze,), rng)
end

function LatticeIsingModel(::Type{T}, _s::NTuple{M, T}, sze::Int, rng::Random.AbstractRNG) where{M, T}
    LatticeIsingModel(T, SVector(_s), (sze,), rng)
end

function LatticeIsingModel(::Type{T}, _s::AbstractArray{T,1}, shp::NTuple{N,Int}, rng::Random.AbstractRNG) where{T, N}
    LatticeIsingModel(T, SVector(_s...), shp, rng)
end

function LatticeIsingModel(::Type{T}, _s::NTuple{M, T}, shp::NTuple{N,Int}, rng::Random.AbstractRNG) where{M, T, N}
    LatticeIsingModel(T, SVector(_s), shp, rng)
end

"""
    # number type 
    T
    # shape of the spin array
    shp::NTuple{N, Int}  
    
    _s is assumed to be (T(-1), T(1))
    J and H are filled with zeros of type T 
    s is filled with random values drawn from _s
"""
function LatticeIsingModel(::Type{T}, sze::Int, rng=Random.default_rng()) where {T}
    LatticeIsingModel(T, SVector(T(-1),T(1)), sze, rng)
end

function LatticeIsingModel(::Type{T}, shp::NTuple{N,Int}, rng=Random.default_rng()) where {T, N}
    LatticeIsingModel(T, SVector(T(-1),T(1)), shp, rng)
end



# ----- How to update a Lattice Ising Model -----

"""
    Little parallel update
    See:
    Collective Properties of Neural Networks: A Statistical Physics Approach - P. Peretto
    section 5
"""
function LittleUpdate!(ising::LatticeIsingModel{T,N,2}, Beta::T, rng::AbstractRNG) where {T,N}
    h = (ising.H + ising.J'*ising.s) * Beta * (ising._s[2] - ising._s[1])
    r = rand(rng, T, ising.sze)
    ising.s .= ifelse.( sigmoid.(h).>=r, ising._s[2], ising._s[1] )
    return nothing
end


function LittleUpdate!(ising::LatticeIsingModel{T,N,M}, Beta::T, rng::AbstractRNG) where {T, N, M}
    h = (ising.H + ising.J'*ising.s)
    @inbounds for i in 1:ising.sze
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
function MetropolisHastingsUpdate!(ising::LatticeIsingModel{T,N,M}, Beta::T, rng::AbstractRNG) where{T, N, M}
    i = rand(rng, 1:ising.sze)
    h = ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])
    r = rand(rng, T)
    new_s = rand(rng, ising._s)
    ising.s[i] = ifelse(exp( Beta * h * (new_s - ising.s[i]) ) >= r, new_s, ising.s[i])
    return nothing
end

function MetropolisHastingsUpdate!(ising::LatticeIsingModel{T,N,2}, Beta::T, rng::AbstractRNG) where{T, N}
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
function GlauberUpdate!(ising::LatticeIsingModel{T,N,2}, Beta::T, rng::AbstractRNG) where {T, N}
    i = rand(rng, 1:ising.sze)
    h = (ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])) *
            Beta * (ising._s[2] - ising._s[1])
    r = rand(rng, T)
    ising.s[i] = ifelse(sigmoid( h ) >= r, ising._s[2], ising._s[1])
    return nothing
end

function GlauberUpdate!(ising::LatticeIsingModel{T, N, M}, Beta::T, rng::AbstractRNG) where{T, N, M}
    i = rand(rng, 1:ising.sze)
    h = ising.H[i] + LinearAlgebra.dot(ising.J[:,i], ising.s[:])
    r = rand(rng, T) 
    y = cumsum(map(x->exp(h * Beta * x), ising._s))
    idx = findfirst( x -> x/y[end] >= r, y)
    ising.s[i] = ising._s[idx]
    return nothing
end


"""
    CheckerboardMetropolisAlgorithm
    it is defined for Lattice Ising models
    right now considers that in each dimension the ising model has an even size
"""
function CheckerboardMetropolisAlgorithm(ising::LatticeIsingModel{T,N,M}) where {T,N,M}
    sze = ising.sze
    for i in ising.shp
        if mod(i,2)!=0
            @error "Need even grid size, got $i"
        end
    end

    i_o, i_e = makeCheckerboardIndeces(ising.shp)

    return CheckerboardMetropolisAlgorithm{T}(sze, i_o, i_e)
end

"""
    Metropolis Checkerboard update
"""
function CheckerboardMetropolisUpdate!(ising::LatticeIsingModel{T, N, 2}, Beta::T, rng::AbstractRNG, alg::CheckerboardMetropolisAlgorithm{T}) where{T, N}
    i_o = alg.i_o
    i_e = alg.i_e
    r = rand(rng, T, ising.sze)

    # odd indeces
    h = ising.H + ising.J' * ising.s
    @inbounds for i in i_o
        s = sum(ising._s) - ising.s[i] # (s_1  - s_i) + (s_-1 - s_i) + s_i -> flip s_i
        hh = h[i] * Beta * (s - ising.s[i])
        ising.s[i] = ifelse(exp( hh ) >= r[i], s, ising.s[i])
    end
    # even indeces
    h = ising.H + ising.J' * ising.s
    @inbounds for i in i_e
        s = sum(ising._s) - ising.s[i] # (s_1  - s_i) + (s_-1 - s_i) + s_i -> flip s_i
        hh = h[i] * Beta * (s - ising.s[i])
        ising.s[i] = ifelse(exp( hh ) >= r[i], s, ising.s[i])
    end
    return nothing
end

function CheckerboardMetropolisUpdate!(ising::LatticeIsingModel{T, N, M}, Beta::T, rng::AbstractRNG, alg::CheckerboardMetropolisAlgorithm{T}) where{T, N, M}
    i_o = alg.i_o
    i_e = alg.i_e
    r = rand(rng, T, ising.sze)
    new_s = rand(rng, ising._s, ising.sze)

    # odd indeces
    h = ising.H + ising.J' * ising.s
    @inbounds for i in i_o
        ising.s[i] = ifelse(exp( Beta * h[i] * (new_s[i] - ising.s[i]) ) >= r[i], new_s[i], ising.s[i])
    end
    # even indeces
    h = ising.H + ising.J' * ising.s
    @inbounds for i in i_e
        ising.s[i] = ifelse(exp( Beta * h[i] * (new_s[i] - ising.s[i]) ) >= r[i], new_s[i], ising.s[i])
    end
    return nothing
end


"""
    CheckerboardGlauberAlgorithm
    it is defined for Lattice Ising models
    right now considers that in each dimension the ising model has an even size
"""
function CheckerboardGlauberAlgorithm(ising::LatticeIsingModel{T,N,M}) where {T,N,M}
    sze = ising.sze
    for i in ising.shp
        if mod(i,2)!=0
            @error "Need even grid size, got $i"
        end
    end

    i_o, i_e = makeCheckerboardIndeces(ising.shp)

    return CheckerboardGlauberAlgorithm{T}(sze, i_o, i_e)
end

"""
    Glauber Checkerboard update
"""
function CheckerboardGlauberUpdate!(ising::LatticeIsingModel{T, N, 2}, Beta::T, rng::AbstractRNG, alg::CheckerboardGlauberAlgorithm{T}) where{T, N}
    i_o = alg.i_o
    i_e = alg.i_e
    r = rand(rng, T, ising.sze)

    # odd indeces
    h = (ising.H + ising.J' * ising.s) * Beta * (ising._s[2] - ising._s[1])
    @inbounds for i in i_o
        ising.s[i] = ifelse(sigmoid( h[i] ) >= r[i], ising._s[2], ising._s[1])
    end
    # even indeces
    h = (ising.H + ising.J' * ising.s) * Beta * (ising._s[2] - ising._s[1])
    @inbounds for i in i_e
        ising.s[i] = ifelse(sigmoid( h[i] ) >= r[i], ising._s[2], ising._s[1])
    end
    return nothing
end

function CheckerboardGlauberUpdate!(ising::LatticeIsingModel{T, N, M}, Beta::T, rng::AbstractRNG, alg::CheckerboardGlauberAlgorithm{T}) where{T, N, M}
    i_o = alg.i_o
    i_e = alg.i_e
    r = rand(rng, T, ising.sze)

    # odd indeces
    h = ising.H + ising.J' * ising.s
    @inbounds for i in i_o
        y = cumsum(map(x->exp(h[i] * Beta * x), ising._s))
        idx = findfirst( x -> x/y[end] >= r[i], y)
        ising.s[i] = ising._s[idx]
    end
    # even indeces
    h = ising.H + ising.J' * ising.s
    @inbounds for i in i_e
        y = cumsum(map(x->exp(h[i] * Beta * x), ising._s))
        idx = findfirst( x -> x/y[end] >= r[i], y)
        ising.s[i] = ising._s[idx]
    end
    return nothing
end


# ----- How to calculate the fields of Lattice Ising Model -----

"""
    instant magnetization
"""
function magnetization(ising::LatticeIsingModel{T, N, M}, Beta::T, field::MagnetizationField{T},
    upd_alg::AbstractUpdateIsingModel) where {T,N,M}
    return ising.s 
end

"""
    instant energy
    for Serial Update methods
"""
function energy(ising::LatticeIsingModel{T, N, M}, Beta::T, field::EnergyField{T},
    upd_alg::AbstractSerialUpdate) where {T,N,M}
    return - ( ising.H + ising.J' * ising.s)' * ising.s 
end

"""
    instant energy
    for Parallel Update
    TODO
    see eq 38 of P. Peretto 1984 "Collective Properties of Neural Networks: A Statistical Physicis Approach"
"""
function energy(ising::LatticeIsingModel{T, N, M}, Beta::T, field::EnergyField{T},
    upd_alg::AbstractParallelUpdate) where {T,N,M}
    return - ( ising.H + ising.J' * ising.s)' * ising.s 
end

