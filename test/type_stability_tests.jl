# Tests for type stability
@testset "PSDesc type stability" begin
    x = CTPS(1.0, 1, 2, 3)
    desc = x.desc

    # Check that exp_to_idx is a concrete type
    @test isconcretetype(typeof(desc.exp_to_idx))
    
    # Check Dict type parameters
    @test keytype(desc.exp_to_idx) != Any
    # * Changed to Int32 based on ctps.jl line 136 - Kelly
    @test valtype(desc.exp_to_idx) == Int32 # Int
end

@testset "CTPS type stability" begin
    nv = 3
    order = 4
    
    # Float64 CTPS
    x_float = CTPS(Float64, nv, order)
    @test eltype(x_float.c) == Float64
    @test isconcretetype(typeof(x_float))
    
    # ComplexF64 CTPS
    x_complex = CTPS(ComplexF64, nv, order)
    @test eltype(x_complex.c) == ComplexF64
    @test isconcretetype(typeof(x_complex))
end

@testset "Multiplication type stability" begin
    nv = 2
    order = 3
    
    x1 = CTPS(Float64, nv, order)
    x2 = CTPS(Float64, nv, order)
    
    x1.c[1] = 1.0
    x2.c[1] = 2.0
    
    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)
    
    # Test type stability of mul! operation
    result = CTPS(Float64, nv, order)
    @inferred PolySeries.mul!(result, x1, x2)
end

@testset "Schedule type stability" begin
    x = CTPS(1.0, 2, 3)
    desc = x.desc

    # Check new 2D k-map schedule types
    sched = desc.mul[1]
    @test isa(sched.k_local, Matrix{Int32})
    @test isa(sched.i_start, Int32)
    @test isa(sched.j_start, Int32)
    @test isa(sched.k_start, Int32)
    @test isa(sched.Ni, Int32)
    @test isa(sched.Nj, Int32)
end
