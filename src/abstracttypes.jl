# ----- Ising Models -----
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






# ----- Updating Algorihtms -----


"""
    Abstract type for algorithms that update Ising  Models
"""
abstract type AbstractUpdateIsingModel end

"""
    Abstract type for parallel updating
"""
abstract type AbstractParallelUpdate <: AbstractUpdateIsingModel end

"""
    Abstract type for serial updating
"""
abstract type AbstractSerialUpdate <: AbstractUpdateIsingModel end



# ----- Kind of Boundary -----
"""
    Abstract type for Kind of Boundary 
"""
abstract type AbstractBoundaryKind end

"""
    Abstract Periodic Boundary
"""
abstract type AbstractPeriodicBoundary <: AbstractBoundaryKind end

