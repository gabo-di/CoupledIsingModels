using CoupledIsingModels
using Test
using StableRNGs
using LinearAlgebra
using SparseArrays
using Statistics


@testset "Energy Fields" begin
    @testset "Local Energy Field" begin
        rng = StableRNG(42)
        sze = 2;
        T = Float64;
        Beta = T(1);
        shp = (sze,sze);
        d = 1;

        # Declare variables
        ising = LatticeIsingModel(T, shp, rng);
        is, js = makeNearestNeighboursConnections(d, ising.boundaries, ising.shp);
        vs = ones(T, length(is));
        ising.J .= sparse(is, js, T(1).*vs);
        alg_upd = CheckerboardGlauberAlgorithm(ising);
        ising.s .= fill(ising._s[1], ising.sze)

        # Fields
        e_0 = LocalEnergyField(T, ising.sze) 
        e_0.field .= CoupledIsingModels.calcField(ising, Beta, e_0, alg_upd)  
        @test isapprox(e_0.field, repeat(T[-4], ising.sze))
    end
    @testset "Interaction Energy Field" begin
        rng = StableRNG(42)
        sze = 2;
        T = Float64;
        Beta = T(1);
        shp = (sze,sze);
        d = 1;

        # Declare variables
        ising = LatticeIsingModel(T, shp, rng);
        is, js = makeNearestNeighboursConnections(d, ising.boundaries, ising.shp);
        vs = ones(T, length(is));
        ising.J .= sparse(is, js, T(1).*vs);
        alg_upd = CheckerboardGlauberAlgorithm(ising);
        ising.s .= fill(ising._s[1], ising.sze)

        # Fields
        e_01 = InteractionEnergyField(T, ising.sze) 
        e_01.field .= CoupledIsingModels.calcField(ising, Beta, e_01, alg_upd)  
        @test isapprox(e_01.field, -ising.J)
    end
end

