# Accuracy tests for CTPS composition (compose / compose!)
#
# Strategy:
#   1. Coefficient equality  — compose result matches a reference polynomial
#      built by direct arithmetic (x+y)*(x-y), etc.
#   2. Degree-mask correctness  — degree_mask of result equals that of the
#      reference after composition.
#   3. Pointwise evaluation  — h(a,b) == f(g1(a,b), g2(a,b)) at several
#      numerical points, verifying the callable and compose are consistent.
#   4. Edge cases:
#        • f is the zero polynomial
#        • f is a constant (no substitution variables appear)
#        • identity substitution map
#        • higher-order (nv=3) composition
#        • in-place compose! vs allocating compose
#   5. Degree-mask guard  — result has no garbage from undef coefficient buffers
#      (the bug fixed in build: calling compose on a polynomial whose buffer was
#      allocated with undef must only accumulate terms in active degrees).

# ─── helpers ──────────────────────────────────────────────────────────────────

_coeff2c(p::CTPS, e1, e2) = element(p, [e1 + e2, e1, e2])

# ─── Basic correctness: (x*y) ∘ (x+y, x-y) = (x+y)*(x-y) = x²-y² ───────────

@testset "compose: x*y substituted by (x+y, x-y)" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    z   = x * y                  # z(x,y) = xy  — only degree-2 active
    h   = compose(z, [x+y, x-y]) # h(x,y) = (x+y)(x-y) = x²-y²
    ref = x*x - y*y

    @test h.degree_mask[] == ref.degree_mask[]
    # Compare only the active coefficient range (inactive positions use undef buffers)
    desc = h.desc
    (s, e) = PolySeries.active_range_bounds(desc, h.degree_mask[])
    @test isapprox(h.c[s:e], ref.c[s:e]; atol=1e-14)

    # Pointwise check
    for (a, b) in ((0.3, 0.1), (0.5, -0.2), (-0.1, 0.4), (0.0, 0.7))
        @test h(a, b) ≈ a^2 - b^2
    end
end

# ─── Degree-mask guard: undef-buffer polynomial ─────────────────────────────
# mul! (used to build x*y) only zeros the output degree band — all other
# positions are uninitialized garbage.  compose must not read those positions
# and must set the degree_mask on the result reflecting only active degrees.

@testset "compose: degree_mask guard (undef-buffer polynomial)" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    # x*y has an undef-allocated buffer; only degree-2 is initialized.
    p = x * y
    @test p.degree_mask[] == UInt64(1 << 2)   # only bit-2 set

    h = compose(p, [x, y])     # identity map — result must equal p exactly

    # degree_mask must match: only degree-2 active
    @test h.degree_mask[] == p.degree_mask[]

    # Active degree-2 coefficients must match exactly
    desc = p.desc
    d2_start = desc.off[3]
    d2_end   = d2_start + desc.Nd[3] - 1
    @test h.c[d2_start:d2_end] ≈ p.c[d2_start:d2_end]

    # Callable evaluation respects the degree_mask (no garbage from other degrees)
    for (a, b) in ((0.3, 0.1), (0.5, -0.2), (-0.1, 0.4))
        @test h(a, b) ≈ a * b   rtol=1e-12
    end
end

# ─── Identity map: compose(f, [x, y]) == f ────────────────────────────────────

@testset "compose: identity substitution" begin
    set_descriptor!(2, 3)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    f = 3.0 + 2*x - y + x*x + x*y - 2*y*y + x*x*y
    h = compose(f, [x, y])

    @test h.degree_mask[] == f.degree_mask[]
    desc = f.desc
    (s, e) = PolySeries.active_range_bounds(desc, f.degree_mask[])
    @test h.c[s:e] ≈ f.c[s:e]
end

# ─── Zero polynomial fast path ────────────────────────────────────────────────

@testset "compose: zero polynomial" begin
    set_descriptor!(2, 3)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    fz = CTPS(Float64)          # zero CTPS
    hz = compose(fz, [x+y, x-y])

    # Only the degree_mask is guaranteed: inactive positions use undef buffers.
    @test hz.degree_mask[] == UInt64(0)
    @test hz(0.3, 0.1) == 0.0
    @test hz(0.0, 0.0) == 0.0
end

# ─── Constant polynomial ──────────────────────────────────────────────────────

@testset "compose: constant polynomial" begin
    set_descriptor!(2, 3)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    fc = CTPS(7.5)              # f(x,y) = 7.5 for all x,y
    hc = compose(fc, [x+y, x-y])

    @test hc.degree_mask[] == UInt64(1)   # only degree-0
    @test hc.c[1] ≈ 7.5
    @test hc(0.3, 0.1) ≈ 7.5
end

# ─── Nonlinear substitution: p(x,y) = x³+2xy  ∘  (2x+y, x-y) ──────────────
# Use linear maps so that max_degree(p ∘ g) = max_degree(p) and the result
# is exactly representable without any order-truncation mismatch.

@testset "compose: nonlinear substitution (linear map)" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    p  = x^3 + 2*x*y               # p(x,y) = x³ + 2xy
    g1 = 2*x + y                   # g₁(x,y) = 2x + y  (linear → no truncation)
    g2 = x - y                     # g₂(x,y) = x - y
    h  = compose(p, [g1, g2])       # h(x,y) = p(2x+y, x-y)

    @test h.degree_mask[] != UInt64(0)

    # Pointwise: h(a,b) == p(2a+b, a-b) = (2a+b)³ + 2(2a+b)(a-b)
    # Linear map → no Taylor truncation; values agree exactly.
    for (a, b) in ((0.1, 0.2), (0.4, -0.3), (0.0, 0.5), (0.3, 0.1))
        @test h(a, b) ≈ p(2a + b, a - b)   rtol=1e-12
    end
end

# ─── Nonlinear substitution with truncation ───────────────────────────────────
# p = x² + y  ∘  (x+y², x-y): g1 degree 2, p degree 2 → max needed degree=4,
# exactly representable at order=4.

@testset "compose: nonlinear substitution (quadratic map, exact at order 4)" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    p  = x^2 + y                   # p(x,y) = x² + y
    g1 = x + y^2                   # g₁(x,y) = x + y²  (degree 2)
    g2 = x - y                     # g₂(x,y) = x - y
    h  = compose(p, [g1, g2])       # h = (x+y²)² + (x-y) = x²+2xy²+y⁴+x-y

    # (x+y²)² = x²+2xy²+y⁴ and (x-y) are both degree ≤ 4 → exactly representable
    for (a, b) in ((0.1, 0.2), (0.4, -0.3), (0.0, 0.5), (0.3, 0.1))
        @test h(a, b) ≈ (a + b^2)^2 + (a - b)   rtol=1e-12
    end
end

# ─── 3-variable composition ───────────────────────────────────────────────────

@testset "compose: 3-variable map" begin
    set_descriptor!(3, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)
    z = CTPS(0.0, 3)

    f  = sin(x) * cos(y) + z^2

    # Rotation-like map
    g  = [x+y, x-y, 2*z]
    h  = compose(f, g)

    for (a, b, c) in ((0.1, 0.2, 0.3), (0.5, -0.1, 0.4), (-0.3, 0.3, -0.2))
        @test h(a, b, c) ≈ f(a+b, a-b, 2c)   rtol=1e-10
    end
end

# ─── In-place compose! vs allocating compose ─────────────────────────────────

@testset "compose!: in-place equals allocating" begin
    set_descriptor!(2, 3)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    f  = x^2 * y + x - y
    g  = [x + 1.0, y * 2.0]

    r1 = compose(f, g)

    r2 = CTPS(Float64)
    compose!(r2, f, g)

    @test r1.degree_mask[] == r2.degree_mask[]
    desc = f.desc
    (s, e) = PolySeries.active_range_bounds(desc, r1.degree_mask[])
    @test r1.c[s:e] ≈ r2.c[s:e]
end

# ─── Full polynomial ∘ shifted identity ───────────────────────────────────────
# Substituting x → x+a should shift the expansion center.

@testset "compose: shift expansion center" begin
    set_descriptor!(2, 4)
    x = CTPS(0.0, 1)
    y = CTPS(0.0, 2)

    f  = x^2 + x*y + y^2       # f(x,y) expanded around (0,0)

    # Shift: compose with (x+1, y+0) → expansion around (-1, 0)
    shift_x = x + 1.0
    h = compose(f, [shift_x, y])

    # h(x,y) = (x+1)² + (x+1)y + y² = x²+2x+1 + xy+y + y²
    for (a, b) in ((0.3, 0.1), (-0.5, 0.2), (0.0, 0.4))
        @test h(a, b) ≈ f(a + 1.0, b)   rtol=1e-12
    end
end
