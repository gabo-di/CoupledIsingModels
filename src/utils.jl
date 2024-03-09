"""
    The sigmoid function
using StaticArrays: length_val
    note that 
    σ(-Inf) ~ 0
    σ(Inf) ~ 1
"""
function sigmoid(x)
    1/(1 + exp(-x))
end

"""
    TAP equation for magnetization
        m_i = tanh(β H_i + β J_ij*m_j - β² m_i (J_ij)^2*(1 - m_j^2)  ) )
    see for example https://www.nature.com/articles/s41467-021-20890-5
    equation (20)
"""
function _TAPEquation(m, Beta, J, H)
    tanh.(Beta*(H + J*m - Beta*(J.^2)*(1 .- m.^2 ))) 
end

"""
    Magnetization in Lattice Ising Model in 1D with 
    first neighbours interactions 
    see for example https://www.thphys.uni-heidelberg.de/~wolschin/statsem20_3s.pdf
    equation 3.27
"""
function _magnetization1DLatticeModel(Beta, J, H)
    exp(2*Beta*J)*sinh(Beta*H)/sqrt(exp(4*Beta*J)*sinh(Beta*H)^2+1) 
end

"""
   Spontaneous Magnetization in Lattice Ising Model in 2D with 
   first neighbours interactions 
   see for example https://en.wikipedia.org/wiki/Ising_model#Two_dimensions
   equation in Onsager's formula for spontaneous magnetization 
"""
function _magnetization2DLatticeModel(Beta, J, H)
    (1 - (sinh(2*Beta*J))^(-4))^(1/8)
end
    
"""
    Generates all spin combinations for a given chain 
    of size sze and posible spins _spins
"""
function _generateSpinCombinations(sze)
    n_c = 2^sze - 1

    return [digits(i, base=2, pad=sze) for i in 0:n_c]
end


"""
    Generates the partition function and probabilities for a given ising model
    consider that size needs to be at most 7
"""
function _generatePartitionFunction(ising, Beta::T) where T
    max_size = 8
    if ising.sze > max_size
        return @error "the size is bigger than $max_size" 
    end

    spins = _generateSpinCombinations(ising.sze)
    probs = zeros(T, length(spins))
    delta_spin = ising._s[2] - ising._s[1]
    
    for (i,spin_) in enumerate(spins)
        spin = delta_spin .* spin_ .+ ising._s[1]
        e = -(ising.H + ising.J' * spin)' * spin
        probs[i] = exp(-Beta*e)
    end
    pF = sum(probs) 
    return pF, probs ./ pF, spins
end


"""
    Calculates the probability distribution for a given spin, uses 
    the results from_generateSpinCombinations 
"""
function probDist(i::Int, probs, spins)
    p = zeros(eltype(probs), 2)
    for (k, spin) in enumerate(spins)
        if spin[i] == 0
            p[1] += probs[k]
        else
            p[2] += probs[k]
        end
    end
    return p
end

"""
    Calculates the probability distribution for a given set of spins, uses 
    the results from_generateSpinCombinations 
    TODO optimize this function
"""
function probDist(i::Array{Int,1}, probs, spins)
    c = _generateSpinCombinations(length(i)) 
    n = length(c)

    p = zeros(eltype(probs), n)
    for (k, spin) in enumerate(spins)
        spin_example = [spin[ii] for ii in i] 
        for j in 1:n
            if isapprox(spin_example, c[j])
                p[j] += probs[k]
                continue
            end
        end
    end
    # in case we are using a non complete set of spins we need to re normalize
    return p ./ sum(p)
end

function probDist(i::NTuple{N, Int}, probs, spins) where N
    probDist([i...], probs, spins)
end

"""
    Calculates the Mutual Information for a given set of spins, uses 
    the results from probDist 
"""
function mutualInformation(i::NTuple{2,Int}, probs, spins)
    p = probDist(i, probs, spins)

    # the marginal probabilities
    p_1 = p[1] + p[3]
    p_2 = p[1] + p[2]
    y = [p_1 * p_2, (1-p_1)*p_2, p_1*(1-p_2), (1-p_1)*(1-p_2)] 

    s = entropy(p)
    s_ = crossentropy(p, y) 
    return -s + s_
end

function mutualInformation(i::Tuple{NTuple{M,Int},NTuple{N,Int}}, probs, spins) where {M,N}
    ii = [i[1]..., i[2]...]
    p = probDist(ii, probs, spins)
    new_spins = _generateSpinCombinations(length(ii))

    # the marginal probabilities
    p_1 = probDist(ntuple(x -> x, M), p, new_spins)
    p_2 = probDist(ntuple(x -> x + M, N), p, new_spins)

    y = zero(p)
    for (k, spin) in enumerate(new_spins)
        i_1 = bin2dec([spin[j] for j in 1:M]) + 1
        i_2 = bin2dec([spin[j] for j in M+1:M+N]) + 1
        y[k] = p_1[i_1] * p_2[i_2]
    end

    s = entropy(p)
    s_ = crossentropy(p, y) 
    return -s + s_
end

"""
    Calculates the entropy of a mass distribution
"""
function entropy(p::AbstractArray{T,1}) where T
    mapreduce(x -> x>0 ? -x*log(x) : 0, +, p)
end

"""
    Calculates the cross entropy of two mass distributions
"""
function crossentropy(p::AbstractArray{T,1}, y::AbstractArray{T,1}) where T
    if length(p) != length(y)
        @error "probability vectors with different lengths"
    end
    mapreduce(x -> y[x]>0 ? -p[x]*log(y[x]) : 0, +, eachindex(y))
end


"""
    Makes a base 10 number from binary representation
"""
function bin2dec(digi::Array{Int,1})
    sum(digi[k]*2^(k-1) for k=1:length(digi))
end



