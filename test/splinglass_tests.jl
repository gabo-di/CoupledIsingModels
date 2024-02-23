using CoupledIsingModels
using Test
using StableRNGs
using LinearAlgebra

@testset "Spin Glass Update Algorithms" begin
    rng = StableRNG(42)
    sze = 10;
    T = Float64;
    Beta = T(5);

    algorithm_list=Dict(
        :MetropolisHastings => MetropolisHastingsAlgorithm(),
        :Glauber => GlauberAlgorithm(),
        :Little => LittleAlgorithm()
    )
    ising = SpinGlassIsingModel(T, sze, rng);
    ising.J .= 1/sze .* (ones(T,sze,sze) - LinearAlgebra.I);
    ising.H .= ones(T, sze);
    n = 100_000;
    y = zeros(T,sze);

    @testset for algorithm_name in keys(algorithm_list)
        algorithm = algorithm_list[algorithm_name]
        y .= 0
        ising.s .= rand(rng,ising._s,ising.sze)
        for i in 1:n
            updateIsingModel!(ising, Beta, algorithm, rng)
            y .+= ising.s
        end
        y .= y./n
        a =  tanh.(Beta*(ising.H + ising.J*y)) 
        a = CoupledIsingModels._TAPEquation(y, Beta, ising.J, ising.H)
        @test isapprox(y,a; atol=1e-3) 
    end
end

