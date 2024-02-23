using CoupledIsingModels
using Test
using StableRNGs
using LinearAlgebra
using SparseArrays
using Statistics

@testset "Temporal Average Algorithm" begin
    seed = 42
    rng = StableRNG(seed)
    sze = 10;
    T = Float64;
    Beta = T(1);
    d = 1;
    n = 100_000;

    ising = LatticeIsingModel(T, sze, rng);
    is, js = makeNearestNeighboursConnections(d, ising.boundaries, ising.shp);
    vs = ones(T, length(is));
    ising.J .= sparse(is, js, vs);
    ising.H .= ones(T, sze);

    upd_alg = GlauberAlgorithm()
    avg_alg = TemporalAverageAlgorithm(n) 
    fields = [MagnetizationField(T,sze)]
    mean_fields = MeanFields(fields, upd_alg, avg_alg)
    
    ising.s .= ones(T, ising.sze)    
    rng = StableRNG(seed)
    calculateFieldsAverage!(ising, Beta, mean_fields, rng)

    y = zeros(T,sze);
    ising.s .= ones(T, ising.sze)
    rng = StableRNG(seed)
    for i in 1:n
        updateIsingModel!(ising, Beta, upd_alg, rng)
        y .+= ising.s
    end
    y .= y./n
    a = CoupledIsingModels._magnetization1DLatticeModel(Beta, 1, 1)
    @test isapprox(mean(y),a; atol=1e-3) 
    @test isapprox(mean(mean_fields.fields[1].field),a; atol=1e-3) 
    @test isapprox(mean_fields.fields[1].field, y; atol=1e-8) 
end
