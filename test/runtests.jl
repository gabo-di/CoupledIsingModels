using SafeTestsets

@safetestset "Utils" begin include("utils_tests.jl") end

@safetestset "Splin Glass Models" begin include("splinglass_tests.jl") end

@safetestset "Lattice Ising Models" begin include("latticemodels_tests.jl") end
