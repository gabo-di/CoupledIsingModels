using CoupledIsingModels
using Test
using StableRNGs
using LinearAlgebra
using SparseArrays
using Statistics


@testset "Boundary Conditions in Lattice Models" begin
    @testset "Periodic Boundary Condition" begin
        @testset "1 D" begin
            d = 1
            n = 4
            shp = (n,)
            boundaries = (PeriodicBoundary(n),)
            is, js = makeNearestNeighboursConnections(d, boundaries, shp)
            
            _is = [1, 1, 2, 2, 3, 3, 4, 4]
            _js = [2, 4, 3, 1, 4, 2, 1, 3]

            @test isequal(is, _is)
            @test isequal(js, _js)
        end
        @testset "2 D" begin
            d = 1
            n = 4
            shp = (n,n)
            boundaries = (PeriodicBoundary(n), PeriodicBoundary(n))
            is, js = makeNearestNeighboursConnections(d, boundaries, shp)
            
            _is = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                   8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14
                   , 14, 14, 15, 15, 15, 15, 16, 16, 16, 16] 

            _js = [2, 4, 5, 13, 3, 1, 6, 14, 4, 2, 7, 15, 1, 3, 8, 16, 6, 8, 9, 1, 7, 5, 10, 2, 8, 6, 11,
                   3, 5, 7, 12, 4, 10, 12, 13, 5, 11, 9, 14, 6, 12, 10, 15, 7, 9, 11, 16, 8, 14, 16, 1, 9, 15,
                   13, 2, 10, 16, 14, 3, 11, 13, 15, 4, 12] 

            @test isequal(is,_is)
            @test isequal(js,_js)
        end
    end
end


@testset "Lattice Model Update Algorithms" begin
    @testset "1 D" begin
        rng = StableRNG(42)
        sze = 10;
        T = Float64;
        Beta = T(1);
        d = 1;

        ising = LatticeIsingModel(T, sze, rng);
        is, js = makeNearestNeighboursConnections(d, ising.boundaries, ising.shp);
        vs = ones(T, length(is));
        ising.J .= sparse(is, js, vs);
        ising.H .= ones(T, sze);
        n = 100_000;
        y = zeros(T,sze);

        algorithm_list=Dict(
            :MetropolisHastings => MetropolisHastingsAlgorithm(),
            :Glauber => GlauberAlgorithm(),
            :MetropolisCheckerboard => CheckerboardMetropolisAlgorithm(ising),
            :GlauberCheckerboard => CheckerboardGlauberAlgorithm(ising)
        )

        @testset for algorithm_name in keys(algorithm_list)
            algorithm = algorithm_list[algorithm_name]
            y .= 0
            # ising.s .= rand(rng,ising._s,ising.sze)
            ising.s .= ones(T, ising.sze)
            for i in 1:n
                updateIsingModel!(ising, Beta, algorithm, rng)
                y .+= ising.s
            end
            y .= y./n
            a = CoupledIsingModels._magnetization1DLatticeModel(Beta, 1, 1)
            @test isapprox(mean(y),a; atol=1e-3) 
        end
    end
    @testset "2 D" begin
        rng = StableRNG(42)
        shp = (10,10);
        T = Float64;
        Beta = T(1);
        d = 1;

        sze = prod(shp)
        ising = LatticeIsingModel(T, shp, rng);
        is, js = makeNearestNeighboursConnections(d, ising.boundaries, ising.shp);
        vs = ones(T, length(is));
        ising.J .= sparse(is, js, vs);

        algorithm_list=Dict(
            :MetropolisHastings => MetropolisHastingsAlgorithm(),
            :Glauber => GlauberAlgorithm(),
            :MetropolisCheckerboard => CheckerboardMetropolisAlgorithm(ising),
            :GlauberCheckerboard => CheckerboardGlauberAlgorithm(ising)
        )

        n = 1_000_000;
        nn = 1000
        y = zeros(T,sze);
        @testset for algorithm_name in keys(algorithm_list)
            algorithm = algorithm_list[algorithm_name]
            y .= 0
            # ising.s .= rand(rng,ising._s,ising.sze)
            ising.s .= ones(T,ising.sze)
            for i in 1:n
                updateIsingModel!(ising, Beta, algorithm, rng)
                if i>nn
                    y .+= ising.s
                end
            end
            y .= y./(n-nn)
            a = CoupledIsingModels._magnetization2DLatticeModel(Beta, 1, 0)
            algorithm_name = string(algorithm_name)
            @test isapprox(mean(y),a; atol=1e-3) 
        end
    end
end

