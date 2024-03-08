"""
    The sigmoid function
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
function _generateSpinCombinations(sze, _spins)
    n_c = 2^sze - 1
    delta_spin = _spins[2] - _spins[1]

    collect( [delta_spin .* digits(i, base=2, pad=sze) .+ _spins[1] for i in 0:n_c])
end


"""
    Generates the partition function and probabilities for a given ising model
    consider that size needs to be at most 7
"""
function _generatePartitionFunction(ising, Beta::T) where T
    max_size = 7
    if ising.sze > max_size
        return @error "the size is bigger than $max_size" 
    end

    spins = _generateSpinCombinations(ising.sze, ising._s)
    probs = zeros(T, length(spins))
    
    for (i,spin) in enumerate(spins)
        e = -(ising.H + ising.J' * spin)' * spin
        probs[i] = exp(-Beta*e)
    end
    pF = sum(probs) 
    return pF, probs ./ pF, spins
end


"""
    Calculates the probability distribution for a given spin, uses 
    the results from _generatePartitionFunction
"""
function probDist(i::Int, probs, spins, ising)
    p = zeros(eltype(probs), 2)
    for (k, spin) in enumerate(spins)
        if isapprox(spin[i], ising._s[1])
            p[1] += probs[k]
        else
            p[2] += probs[k]
        end
    end
    return p
end

"""
    Calculates the probability distribution for a given set of spins, uses 
    the results from _generatePartitionFunction
"""
function probDist(i::Array{Int,1}, probs, spins, ising)
    c = _generateSpinCombinations(length(i), ising._s) 
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

"""
    Calculates the Mutual Information for a given set of spins, uses 
    the results from probDist 
"""
function mutualInformation(i::Array{Int,1}, probs, spins, ising)
    if length(i)!=2
        @error "Function only works for 2 spins"
    end
    p = probDist(i, probs, spins, ising)

    # the marginal probabilities
    p_1 = p[1] + p[3]
    p_2 = p[1] + p[2]
    y = [p_1 * p_2, (1-p_1)*p_2, p_1*(1-p_2), (1-p_1)*(1-p_2)] 

    s = mapreduce(x -> x>0 ? -x*log(x) : 0, +, p)
    s_ = mapreduce(x -> p[x]>0 ? -p[x]*log(y[x]) : 0, +, 1:4)
    return -s + s_
end


