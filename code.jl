using StaticArrays: LinearAlgebra
using Revise
using Debugger
using Infiltrator



using Random
using StaticArrays
using LinearAlgebra

includet("src/abstracttypes.jl")
includet("src/utils.jl")
includet("src/spinglassmodel.jl")

function main()
    sze = 10;
    T = Float64;
    rng = Random.default_rng();
    Beta = T(5);

    ising = SpinGlassIsingModel(T, sze, rng);
    ising.J .= 1/sze .* (ones(T,sze,sze) - LinearAlgebra.I);
    ising.H .= ones(T, sze);

    y = zeros(T,sze);
    n = 100_000;
    for i in 1:n
        # GlauberUpdate!(ising, Beta, rng)
        LittleUpdate!(ising, Beta, rng)
        # MetropolisHastingsUpdate!(ising, Beta, rng)
        y .+= ising.s
    end
    y .= y./n
    a =  tanh.(Beta*(ising.H + ising.J*y)) 
    println(isapprox(y,a; atol=1e-3), " isapprox ")
    println(max(abs.(y-a)...), " l-inf ")
    println(LinearAlgebra.norm(y-a), " l-2 norm ")
end




