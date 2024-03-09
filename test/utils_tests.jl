using CoupledIsingModels
using Test
using StableRNGs

@testset "helper functions" begin
    atol = 1e-5

    # sigmoid function
    @test isapprox(CoupledIsingModels.sigmoid(0.0), 1/2; atol=atol)
    @test isapprox(CoupledIsingModels.sigmoid(Inf), 1, atol=atol)
    @test isapprox(CoupledIsingModels.sigmoid(-Inf), 0; atol=atol)
end

@testset "Mutual Information" begin
    @testset "Basic Stuff" begin
        seed = 42
        rng = StableRNG(seed)
        T = Float32

        spins = CoupledIsingModels._generateSpinCombinations(2)
        @test isapprox(spins, [[0,0], [1,0], [0,1], [1,1]])
        

        probs = rand(rng, 2^2)
        probs = probs ./ sum(probs)
        p = CoupledIsingModels.probDist(2, probs, spins)
        @test isapprox(p, [sum(probs[1:2]), sum(probs[3:4])])


        probs = rand(rng, 2^3)
        probs = probs ./ sum(probs)
        spins = CoupledIsingModels._generateSpinCombinations(3)
        p = CoupledIsingModels.probDist([1,2], probs, spins)
        @test isapprox(p,
            [probs[1]+probs[5], probs[2]+probs[6], probs[3]+probs[7], probs[4]+probs[8]])

        probs = ones(T,2^3) .* T(1/2^3)
        s_in = CoupledIsingModels.mutualInformation((1,2), probs, spins)
        @test isapprox(s_in, T(0))

        probs = T[0.25, 0, 0, 0, 0.25, 0, 0, 0.5]
        s_in = CoupledIsingModels.mutualInformation((1,2), probs, spins)
        @test isapprox(s_in, -log(T(0.5)))

        probs = T[0.5, 0, 0, 0, 0.5, 0, 0, 0]
        s_in = CoupledIsingModels.mutualInformation((1,2), probs, spins)
        @test isapprox(s_in, T(0))


        probs = rand(rng, 2^3)
        probs = probs ./ sum(probs)
        s_in_1 = CoupledIsingModels.mutualInformation(((1,),(2,)), probs, spins)
        s_in_2 = CoupledIsingModels.mutualInformation((1,2), probs, spins)
        @test isapprox(s_in_1, s_in_2)

        probs = T[0.25, 0, 0, 0, 0.25, 0, 0, 0.5]
        s_in = CoupledIsingModels.mutualInformation(((1,3), (2,)), probs, spins)
        @test isapprox(s_in, -log(T(0.5)))
        s_in = CoupledIsingModels.mutualInformation(((3,2), (1,)), probs, spins)
        @test isapprox(s_in, -log(T(0.5)))
        s_in = CoupledIsingModels.mutualInformation(((1,2), (3,)), probs, spins)
        @test isapprox(s_in, -T(0.25*log(0.5)+0.5*log(0.75)+0.25*log(1.5)))
    end

    @testset "Is additive?" begin
        seed = 42
        rng = StableRNG(seed)
        T = Float32
        
        # --- 2 interacting spins
        # this matrix flips only the first spin
        tr_1 = zeros(T, 4, 4)
        tr_1[1,2] = 0.7
        tr_1[2,1] = 0.3
        tr_1[1,1] = 1 - tr_1[2,1]
        tr_1[2,2] = 1 - tr_1[1,2]
        tr_1[3,4] = 0.3
        tr_1[4,3] = 0.7
        tr_1[3,3] = 1 - tr_1[4,3]
        tr_1[4,4] = 1 - tr_1[3,4]

        # this matrix flips only the second spin
        tr_2 = zeros(T, 4, 4)
        tr_2[1,3] = 0.3
        tr_2[3,1] = 0.7
        tr_2[1,1] = 1 - tr_2[3,1]
        tr_2[3,3] = 1 - tr_2[1,3]
        tr_2[2,4] = 0.7
        tr_2[4,2] = 0.3
        tr_2[2,2] = 1 - tr_2[4,2]
        tr_2[4,4] = 1 - tr_2[2,4]

        # start from a uniform distribution
        p = T[1, 0, 0, 0]
        # make the flips
        probs = tr_2 * tr_1 * p
        # the same with another independent system
        probs = reshape([probs[i]*probs[j] for i=1:4, j=1:4], :)

        # compute the mutual information
        spins = CoupledIsingModels._generateSpinCombinations(4)
        s_in_1 = CoupledIsingModels.mutualInformation((1,2), probs, spins) 
        s_in_2 = CoupledIsingModels.mutualInformation((3,4), probs, spins) 
        s_in_t = CoupledIsingModels.mutualInformation(((1,3),(2,4)), probs, spins) 
        @test isapprox(s_in_t, s_in_1+s_in_2)

            
        # start from random distribution
        p_1 = rand(rng, T, 4)
        p_1 = p_1 ./ sum(p_1)
        p_2 = T[1, 0, 0, 0]
        # make the flips
        probs_1 = tr_2 * tr_1 * p_1
        probs_2 = tr_2 * tr_1 * p_2
        # the same with another independent system
        probs = reshape([probs_1[i]*probs_2[j] for i=1:4, j=1:4], :)

        # compute the mutual information
        spins = CoupledIsingModels._generateSpinCombinations(4)
        s_in_1 = CoupledIsingModels.mutualInformation((1,2), probs, spins) 
        s_in_2 = CoupledIsingModels.mutualInformation((3,4), probs, spins) 
        s_in_t = CoupledIsingModels.mutualInformation(((1,3),(2,4)), probs, spins) 
        @test isapprox(s_in_t, s_in_1+s_in_2)

    end
end
