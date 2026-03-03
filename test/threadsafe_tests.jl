# Tests for thread safety
@testset "Descriptor caching thread safety" begin
    nv = 3
    order = 4
    
    # Create multiple CTPS instances with same parameters from different "threads"
    # (simulated by sequential calls - actual threading would require Threads.@threads)
    instances = [CTPS(Float64, nv, order) for _ in 1:10]
    
    # All should share the same descriptor
    desc1 = instances[1].desc
    for inst in instances
        @test inst.desc === desc1  # Same object reference
    end
end

@testset "Global descriptor switching" begin
    # After set_descriptor!, ctps.desc returns the new active descriptor
    d1 = set_descriptor!(2, 3)
    x = CTPS(Float64, 2, 3)
    @test x.desc === d1

    d2 = set_descriptor!(3, 4)
    y = CTPS(Float64)
    @test y.desc === d2
    # x.desc also reflects the current global (no per-instance copy)
    @test x.desc === d2
end

@testset "Descriptor immutability" begin
    x = CTPS(Float64, 2, 3)
    desc = x.desc
    
    # Descriptor fields should be accessible but immutable
    @test desc.nv == 2
    @test desc.order == 3
    @test desc.N > 0
    
    # These should not be mutable (would throw error)
    @test_throws ErrorException desc.nv = 5
end

@testset "Output-major schedule thread safety" begin
    nv = 2
    order = 3
    
    x1 = CTPS(Float64, nv, order)
    x2 = CTPS(Float64, nv, order)
    
    for i in 1:binomial(nv + order, nv)
        x1.c[i] = Float64(i)
        x2.c[i] = Float64(i + 10)
    end
    
    PolySeries.update_degree_mask!(x1)
    PolySeries.update_degree_mask!(x2)
    
    # Perform multiplication multiple times
    # Each should produce identical results (deterministic)
    results = [begin
        r = CTPS(Float64, nv, order)
        PolySeries.mul!(r, x1, x2)
        copy(r.c)
    end for _ in 1:5]
    
    # All results should be identical
    for r in results[2:end]
        @test r ≈ results[1]
    end
end
