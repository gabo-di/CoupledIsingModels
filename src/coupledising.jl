# -----  Probability Stuff  -----
"""
    Coupled Ising Model Probability Distribution
"""
struct CIMProbability{T} <: AbstractProbability
    p_1::Array{AbstractArray{T,2},1}
    path::Array{AbstractArray{Int,1},1}
end

function CIMProbability(::Type{T}, sze::Int) where{T}
    CIMProbability{T}(
        [zeros(T, 4, sze), zeros(T, 4, sze)],
        [zeros(Int, sze), zeros(Int, sze)]
    )
end

"""
    forward and backward jump ratio
"""
struct KRatio{T} <: AbstractProbability
    k_ratio::Array{AbstractArray{T,2},1}
end

function KRatio(::Type{T}, sze::Int) where{T}
    KRatio{T}( [zeros(T,4,sze), zeros(T,4,sze)])
end

# -----  Entropy Stuff  -----
"""
    Mutual Information in Coupled Ising Models
"""
struct CIMMutualInformation{T} <: AbstractEntropy 
    sh::Array{AbstractArray{T,1},1}
end

function CIMMutualInformation(::Type{T}, sze::Int) where{T}
    CIMMutualInformation{T}([zeros(T,sze), zeros(T,sze)])
end

function calcEntropy!(cim_Probability::CIMProbability{T}, cim_kRatio::KRatio{T},
    cim_mutualInformation::CIMMutualInformation{T}, itt::Int,
    sze::Int, upd_alg::AbstractUpdateIsingModel ) where{T}
    spins = _generateSpinCombinations(2)
    p_12_C = cim_Probability.p_1[itt]
    @inbounds Threads.@threads for i in 1:sze
        cim_mutualInformation.sh[itt][i] = mutualInformation((1,2), p_12_C[:,i], spins)
    end
end

"""
    Entropy Production 
"""
struct CIMEntropyProduction{T} <: AbstractEntropy
    sh::Array{AbstractArray{T,1},1}
end

function CIMEntropyProduction(::Type{T}, sze::Int) where{T}
    CIMEntropyProduction{T}([zeros(T,sze), zeros(T,sze)])
end

function calcEntropy!(cim_Probability::CIMProbability{T}, cim_kRatio::KRatio{T},
    cim_entropyProduction::CIMEntropyProduction{T}, itt::Int,
    sze::Int, upd_alg::AbstractUpdateIsingModel ) where{T}

    itt_f = mod1(itt+1,length(cim_Probability.p_1))

    p_12_C = cim_Probability.p_1[itt]
    p_21_C = cim_Probability.p_1[itt_f]
    k_ratio_1 = cim_kRatio.k_ratio[itt]
    path_1 = cim_Probability.path[itt]


    @inbounds Threads.@threads for i in 1:sze
        path_f = path_1[i]
        p_1 = probDist(Val(2), p_12_C)
        p_0 = probDist(Val(1), p_21_C)
        s_aux_0 = div(path_f-1,2) + 1
        s_aux_1 = rem(path_f-1,2) + 1
        path_b = (s_aux_1-1)*2 + s_aux_0
        if path_b != path_f 
            cim_entropyProduction.sh[itt][i] = log(k_ratio_1[path_f]*p_0[s_aux_0]/(k_ratio_1[path_b] * p_1[s_aux_1]) )
        else
            cim_entropyProduction.sh[itt][i] = log(p_0[s_aux_0]/(p_1[s_aux_1]))
        end
    end
end

"""
    Entropy Production Rate
"""
struct CIMEntropyProductionRate{T} <: AbstractEntropy
    sh::Array{AbstractArray{T,1},1}
end

function CIMEntropyProductionRate(::Type{T}, sze::Int) where{T}
    CIMEntropyProduction{T}([zeros(T,sze), zeros(T,sze)])
end

function calcEntropy!(cim_Probability::CIMProbability{T}, cim_kRatio::KRatio{T},
    cim_entropyProductionRate::CIMEntropyProductionRate{T}, itt::Int,
    sze::Int, upd_alg::AbstractUpdateIsingModel ) where{T}

    itt_f = mod1(itt+1,length(cim_Probability.p_1))

    p_12_C = cim_Probability.p_1[itt]
    p_21_C = cim_Probability.p_1[itt_f]
    k_ratio_1 = cim_kRatio.k_ratio[itt]


    @inbounds Threads.@threads for i in 1:sze
        sh = T(0)
        p_1 = probDist(Val(2), p_12_C)
        p_0 = probDist(Val(1), p_21_C)
        for path_f in 1:4
            s_aux_0 = div(path_f-1,2) + 1
            s_aux_1 = rem(path_f-1,2) + 1
            path_b = (s_aux_1-1)*2 + s_aux_0
            j_ = (k_ratio_1[path_f]*p_0[s_aux_0] - k_ratio_1[path_b] * p_1[s_aux_1])
            if path_b != path_f 
                sh += j_ * log(k_ratio_1[path_f]*p_0[s_aux_0]/(k_ratio_1[path_b] * p_1[s_aux_1]) )
            else
                sh += j_ * log(p_0[s_aux_0]/(p_1[s_aux_1]))
            end
        end
        cim_entropyProductionRate.sh[itt][i] = sh
    end
end
# -----  Update Coupled Ising Models  -----

function coupledIsingProbabilitiesUpdate!(cim_Probability::CIMProbability{T},  cim_kRatio::KRatio{T}, 
    Beta::T, h::Array{T,1}, cim::Array{LatticeIsingModel{T,N,2},1}, itt::Int,
    alg::CheckerboardGlauberAlgorithm{T,ThreadsCPU},
    rng::AbstractRNG) where{T,N}

    idxs = alg.idxs
    idx = alg.itt[1]

    itt_f = mod1(itt+1,length(cim))
    ising_1 = cim[itt]
    ising_2 = cim[itt_f]
    h_12 = h[itt]
    p_12_C = cim_Probability.p_1[itt]
    p_21_C = cim_Probability.p_1[itt_f]
    k_ratio = cim_kRatio.k_ratio[itt]
    path = cim_Probability.path[itt]

    r = rand(rng, T, ising_1.sze)

    # note that Glauber Algorithm has the same jump rates for each spin value
    # so we use reduced probabilities for simplicity 
    # we do not need to consider each 4-possible initial probability
    # nor each 8-possible outcome probability 
    

    # half of updates
    δ_s = ising_1._s[2] - ising_1._s[1]
    h = (ising_1.H + ising_1.J' * ising_1.s) * Beta 

    @inbounds Threads.@threads for i in idxs[idx] 
        pp_f = sigmoid( (h[i] + h_12 * (sum(ising_2._s) - ising_2.s[i]) * Beta) * δ_s)
        h[i] += h_12 * (ising_2.s[i]) * Beta  
        pp_c = sigmoid(h[i] * δ_s)
        p_21_c = probDist(Val(2), p_21_C[:,i]) 

        if ising_2.s[i] == ising_2._s[1]
            p_12_C[1,i] = p_21_c[1] * (1 - pp_c) # 0 0 
            p_12_C[2,i] = p_21_c[2] * (1 - pp_f) # 1 0 
            p_12_C[3,i] = p_21_c[1] * pp_c       # 0 1
            p_12_C[4,i] = p_21_c[2] * pp_f       # 1 1
        else
            p_12_C[4,i] = p_21_c[2] * pp_c       # 1 1
            p_12_C[2,i] = p_21_c[2] * (1 - pp_c) # 1 0 
            p_12_C[3,i] = p_21_c[1] * pp_f       # 0 1
            p_12_C[1,i] = p_21_c[1] * (1 - pp_f) # 0 0 
        end

        s_aux_1 = ifelse( pp_c >= r[i], 2, 1)
        s_aux_0 = ifelse( ising_1.s[i] == ising_1._s[2], 2, 1)
        #                             1->1  1->2  2->1 2->2
        k_ratio[:,1] = [1-pp_c, pp_c, 1-pp_c, pp_c]
        path[i] = [(s_aux_0-1)*2 + s_aux_1]
        ising_1.s[i] = ising_1._s[s_aux_1]
    end
end

function coupledIsingProbabilitiesUpdate!(p_12_c::AbstractArray{T,2}, p_12_C::AbstractArray{T,2}, p_21_c::AbstractArray{T,2},
    k_ratio::AbstractArray{T,1}, k_path::AbstractArray{T,1},
    Beta::T, h_12::T, ising_1::LatticeIsingModel{T,N,2}, ising_2::LatticeIsingModel{T,N,2},
    alg::CheckerboardGlauberAlgorithm{T,ThreadsCPU},
    rng::AbstractRNG) where{T,N}
    idxs = alg.idxs
    idx = alg.itt[1]
    r = rand(rng, T, ising_1.sze)

    # note that Glauber Algorithm has the same jump rates for each spin value
    # so we use reduced probabilities for simplicity 
    # we do not need to consider each 4-possible initial probability
    # nor each 8-possible outcome probability 


    # half of updates
    δ_s = ising_1._s[2] - ising_1._s[1]
    h = (ising_1.H + ising_1.J' * ising_1.s) * Beta 

    @inbounds Threads.@threads for i in idxs[idx] 
        pp_f = sigmoid( (h[i] + h_12 * (sum(ising_2._s) - ising_2.s[i]) * Beta) * δ_s)
        h[i] += h_12 * (ising_2.s[i]) * Beta  
        pp_c = sigmoid(h[i] * δ_s)

        if ising_2.s[i] == ising_2._s[1]
            p_12_C[1,i] = p_21_c[1,i] * (1 - pp_c) # 0 0 
            p_12_C[2,i] = p_21_c[2,i] * (1 - pp_f) # 1 0 
            p_12_C[3,i] = p_21_c[1,i] * pp_c       # 0 1
            p_12_C[4,i] = p_21_c[2,i] * pp_f       # 1 1
        else
            p_12_C[4,i] = p_21_c[2,i] * pp_c       # 1 1
            p_12_C[2,i] = p_21_c[2,i] * (1 - pp_c) # 1 0 
            p_12_C[3,i] = p_21_c[1,i] * pp_f       # 0 1
            p_12_C[1,i] = p_21_c[1,i] * (1 - pp_f) # 0 0 
        end

        s_aux = ifelse( pp_c >= r[i], 2, 1)
        k_ratio[i] =  exp(h[i] * (ising_1._s[s_aux] - ising_1.s[i]) )
        k_path[i] = s_aux
        ising_1.s[i] = s_aux
        # reduced considers system forgets the agent's state
        p_12_c[1,i] =  p_12_C[1,i] + p_12_C[2,i] # 0
        p_12_c[2,i] =  p_12_C[3,i] + p_12_C[4,i] # 1
    end
end

function coupledIsingUpdate!(cim_Probability::CIMProbability{T},
    cim_kRatio::KRatio{T},
    entropies::Array{AbstractEntropy,1},
    Beta::T, h_12::T, h_21::T,
    ising_1::LatticeIsingModel{T,N,2}, ising_2::LatticeIsingModel{T,N,2},
    upd_alg::CheckerboardGlauberAlgorithm{T,ThreadsCPU}, 
    rng::AbstractRNG) where{T,N}

    p_12_C = cim_Probability.p_12_C
    p_21_C = cim_Probability.p_21_C
    p_12_c = cim_Probability.p_12_c
    p_21_c = cim_Probability.p_21_c
    k_ratio_1 = cim_kRatio.k_ratio_1
    k_path_1 = cim_kRatio.k_path_1
    k_ratio_2 = cim_kRatio.k_ratio_2
    k_path_2 = cim_kRatio.k_path_2


    for _ in 1:length(upd_alg.idxs)
        # change the set for the current iteration
        upd_alg.itt[1] = mod1(upd_alg.itt[1] + 1, length(upd_alg.idxs))

        # half of system 1
        coupledIsingProbabilitiesUpdate!(p_12_c, p_12_C, p_21_c, k_ratio_1, k_path_1, Beta, h_12, ising_1, ising_2, upd_alg, rng) 

        # half of system 2
        coupledIsingProbabilitiesUpdate!(p_21_c, p_21_C, p_12_c, k_ratio_2, k_path_2, Beta, h_21, ising_2, ising_1, upd_alg, rng) 
    end

    for entr in entropies
        calcEntropy!(cim_Probability, cim_kRatio, entr, ising_1.sze, upd_alg)
    end
end
