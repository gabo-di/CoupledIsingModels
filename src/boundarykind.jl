"""
    Periodic Boundary Kind
"""
struct PeriodicBoundary <: AbstractPeriodicBoundary
    # period size
    n::Int
end

"""
    # distance to consider nearest neighbours, (NOTE is exact distance) 
    d::Int
    # boundary condition for each dimension N
    boundaries::NTuple{N, AbstractBoundaryKind}
    # shape of the spin array
    shp::NTuple{N, Int}  
"""
function makeNearestNeighboursConnections(d::Int,
    boundaries::NTuple{1, AbstractBoundaryKind},
    shp::NTuple{1,Int}) 

    sze = shp[1] 
    js = Int[]
    sizehint!(js, 2*sze)
    is = Int[]
    sizehint!(is, 2*sze)

    arr = collect(1:sze)

    boundary = boundaries[1]
    for i in eachindex(arr)
        j = findNearestNeighbours(i, d, boundary)
        append!(js, j)
        append!(is, repeat([i], length(j)))
    end
    return is, js
end

function makeNearestNeighboursConnections(d::Int,
    boundaries::NTuple{2, AbstractBoundaryKind},
    shp::NTuple{2,Int}) 

    sze = prod(shp) 
    js = Int[]
    sizehint!(js, 4*sze)
    is = Int[]
    sizehint!(is, 4*sze)

    arr = reshape(collect(1:sze), shp)

    for i_2 in axes(arr,2)
        for i_1 in axes(arr,1)
            i = arr[i_1,i_2]
            j_1 = findNearestNeighbours(i_1, d, boundaries[1])
            for _j in j_1
                j = arr[_j, i_2]
                push!(js, j)
                push!(is, i)
            end
            j_2 = findNearestNeighbours(i_2, d, boundaries[2])
            for _j in j_2
                j = arr[i_1, _j]
                push!(js, j)
                push!(is, i)
            end
        end
    end
    return is, js
end

function findNearestNeighbours(i::Int, d::Int,
    boundary::PeriodicBoundary)
    j_right = mod1(i+d, boundary.n)
    j_left = mod1(i-d, boundary.n)
    return [j_right, j_left]
end
