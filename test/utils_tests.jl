using CoupledIsingModels
using Tests

@testset "helper functions" begin
    atol = 1e-5

    # sigmoid function
    @test isapprox(sigmoid(0.0), 1/2; atol=atol)
    @test isapprox(sigmoid(Inf), 1, atol=atol)
    @test isapprox(sigmoid(-Inf), 0; atol=atol)
end
