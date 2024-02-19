"""
    Lattice Ising Model with a simple spin chain
    the Hamiltonian is given by 
    H(s_1, s_0) = - ( H + J * s_0)' * s_1 =  - ( H' * s_1 + s_0' * J' * s_1 )
    where in genera s_1 is at time t=1 and s_0 is at time t=0
"""
struct LatticeIsingModel{T, N, M} <: AbstractLatticeIsingModel{T, N, M}
    # possible spin values
    _s::SVector{M,T}
    # spin array
    s::AbstractArray{T, 1}
    # size of the spin array
    sze::NTuple{N, Int}
    # the external field acting on each spin, has same size as spin array
    H::AbstractArray{T, 2}
    # the interaction matrix between spins, has shape = (sze, sze)
    J::AbstractArray{T, 2}
end

"""
    # number type, in general a kind of Float 
    T
    # possible spin values
    _s::SVector{M,T}
    # size of the spin array
    sze::int  
    
    J and H are filled with zeros of type T 
    s is filled with random values drawn from _s
"""
# function SpinGlassIsingModel(::Type{T}, _s::SVector{M, T}, sze::Int, rng::Random.AbstractRNG) where{M, T}
#     SpinGlassIsingModel{T, M}(_s, rand(rng, _s, sze), sze, zeros(T, sze), zeros(T, sze, sze))
# end
#
# function SpinGlassIsingModel(::Type{T}, _s::AbstractArray{T,1}, sze::Int, rng::Random.AbstractRNG) where{T}
#     SpinGlassIsingModel(T, SVector(_s...), sze, rng)
# end
#
# function SpinGlassIsingModel(::Type{T}, _s::NTuple{M, T}, sze::Int, rng::Random.AbstractRNG) where{M, T}
#     SpinGlassIsingModel(T, SVector(_s), sze, rng)
# end
