function coupledIsingProbabilities!(p_12_c::AbstractArray{T,2}, p_12_C::AbstractArray{T,2}, p_21_c::AbstractArray{T,2},
    Beta::T, h_12::T, ising_1::LatticeIsingModel{T,N,2}, ising_2::LatticeIsingModel{T,N,2},
    alg::CheckerboardGlauberAlgorithm{T,ThreadsCPU}, rng::AbstractRNG) where{T,N}
    idxs = alg.idxs
    idx = alg.itt[1]
    r = rand(rng, T, ising_1.sze)


    # half of updates
    h = (ising_1.H + ising_1.J' * ising_1.s) * Beta * (ising_1._s[2] - ising_1._s[1])

    @inbounds Threads.@threads for i in idxs[idx] 
        pp_f = sigmoid( h[i] + h_12 * (sum(ising_2._s) - ising_2.s[i]) * Beta * (ising_1._s[2] - ising_1._s[1]))
        pp_c = sigmoid( h[i] + h_12 * (ising_2.s[i]) * Beta * (ising_1._s[2] - ising_1._s[1]))

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

        ising_1.s[i] = ifelse( pp_c >= r[i], ising_1._s[2], ising_1._s[1])
        p_12_c[1,i] = 1-pp_c
        p_12_c[2,i] = pp_c
    end
end

function coupledIsingMutualInformation!(in_12::AbstractArray{T,1}, in_21::AbstractArray{T,1},
    p_12_C::AbstractArray{T,2}, p_21_C::AbstractArray{T,2}, p_12_c::AbstractArray{T,2}, p_21_c::AbstractArray{T,2},
    Beta::T, h_12::T, h_21::T, ising_1::LatticeIsingModel{T,N,2}, ising_2::LatticeIsingModel{T,N,2},
    alg::CheckerboardGlauberAlgorithm{T,ThreadsCPU}, rng::AbstractRNG) where{T,N}


    spins = _generateSpinCombinations(2)

    for _ in 1:length(alg.idxs)
        # change the set for the current iteration
        alg.itt[1] = mod1(alg.itt[1] + 1, length(alg.idxs))

        # half of system 1
        coupledIsingProbabilities!(p_12_c, p_12_C, p_21_c, Beta, h_12, ising_1, ising_2, alg, rng) 

        # half of system 2
        coupledIsingProbabilities!(p_21_c, p_21_C, p_12_c, Beta, h_21, ising_2, ising_1, alg, rng) 
    end

    @inbounds Threads.@threads for i in 1:ising_1.sze
        in_12[i] = mutualInformation((1,2), p_12_C[:,i], spins)
        in_21[i] = mutualInformation((1,2), p_21_C[:,i], spins)
    end
end
