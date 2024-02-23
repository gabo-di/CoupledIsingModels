using SafeTestsets

@safetestset "Utils" begin include("utils_tests.jl") end

@safetestset "Spin Glass Models" begin include("spinglass_tests.jl") end

@safetestset "Lattice Ising Models" begin include("latticemodels_tests.jl") end

@safetestset "Mean Algorithms" begin include("meanalgorithms_tests.jl") end
