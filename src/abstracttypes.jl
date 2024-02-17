"""
    The AbstractIsingModel has a 
    T type of the arguments, in general T <: Number
    N the dimension of the spin array, so N is an integer number
    M enumerates the different spin values, so M is an integer number.
    M = 2 means two values for the spin, -T(1) and T(1) for example.
    M = 3 means three values for the spin, -T(1), T(0), T(1) for example.
"""
abstract type AbstractIsingModel{T, N, M} end


"""
    Abstract type for Spin Glass Ising models, the parameters are similar to
    AbstractIsingModel{T, N, M}
    in general we expect to have N = 1 and M = 2
"""
abstract type AbstractSplinGlassIsingModel{T, N, M} <: AbstractIsingModel{T, N, M} end


"""
    Abstract type for Lattice Ising models, the parameters are similar to
    AbstractIsingModel{T, N, M}
    in general we expect to have M = 2
"""
abstract type AbstractLatticeIsingModel{T, N, M} <: AbstractIsingModel{T, N, M} end

