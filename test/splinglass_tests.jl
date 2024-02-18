using CoupledIsingModels
using Test
using StableRNGs
using LinearAlgebra

@testset "Metropolis Hasting Update" begin
    rng = StableRNG(42)
    sze = 10;
    T = Float64;
    Beta = T(5);

    ising = SpinGlassIsingModel(T, sze, rng);
    ising.J .= 1/sze .* (ones(T,sze,sze) - LinearAlgebra.I);
    ising.H .= ones(T, sze);

    y = zeros(T,sze);
    n = 100_000;
    for i in 1:n
        MetropolisHastingsUpdate!(ising, Beta, rng)
        y .+= ising.s
    end
    y .= y./n
    a =  tanh.(Beta*(ising.H + ising.J*y)) 
    @test isapprox(y,a; atol=1e-3)
end

@testset "Little Update" begin
    rng = StableRNG(42)
    sze = 10;
    T = Float64;
    Beta = T(5);

    ising = SpinGlassIsingModel(T, sze, rng);
    ising.J .= 1/sze .* (ones(T,sze,sze) - LinearAlgebra.I);
    ising.H .= ones(T, sze);

    y = zeros(T,sze);
    n = 100_000;
    for i in 1:n
        LittleUpdate!(ising, Beta, rng)
        y .+= ising.s
    end
    y .= y./n
    a =  tanh.(Beta*(ising.H + ising.J*y)) 
    @test isapprox(y,a; atol=1e-3)
end


@testset "Glauber Update" begin
    rng = StableRNG(42)
    sze = 10;
    T = Float64;
    Beta = T(5);

    ising = SpinGlassIsingModel(T, sze, rng);
    ising.J .= 1/sze .* (ones(T,sze,sze) - LinearAlgebra.I);
    ising.H .= ones(T, sze);

    y = zeros(T,sze);
    n = 100_000;
    for i in 1:n
        GlauberUpdate!(ising, Beta, rng)
        y .+= ising.s
    end
    y .= y./n
    a =  tanh.(Beta*(ising.H + ising.J*y)) 
    @test isapprox(y,a; atol=1e-3)
end
