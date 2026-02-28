# ==============================================================================
# Auto-differentiation with TPSA
#
# TPSA provides two complementary differentiation mechanisms:
#
#   1. Built-in TPS derivatives (fast, exact, zero overhead)
#      – CTPS coefficients *are* the Taylor-map derivatives.
#      – element(ctps, [i,j,...]) returns the exact partial derivative ∂^n f/∂x^i∂y^j...
#
#   2. AD through CTPS coefficients (coefficient-level sensitivity)
#      – Useful when you want d/dθ of a *map coefficient* with respect to a
#        physical parameter θ (e.g., magnet strength, drift length).
#      – Two supported frameworks:
#          a) ForwardDiff.jl   — works out of the box; ZERO code changes needed.
#          b) Enzyme.jl        — requires EnzymeRules (see section 3 below).
#
# Dependencies (install once):
#   julia> using Pkg; Pkg.add(["ForwardDiff", "BenchmarkTools", "Enzyme"])
# ==============================================================================

using TPSA
using BenchmarkTools

# ------------------------------------------------------------------------------
# Section 1: ForwardDiff through CTPS coefficients
#
# Because CTPS{T} is fully generic in T, ForwardDiff's Dual numbers propagate
# through every arithmetic and math operation with no special handling.
# ------------------------------------------------------------------------------

println("="^70)
println("AD Through TPSA Map Coefficients — ForwardDiff")
println("="^70)

using ForwardDiff

set_descriptor!(1, 8)

# ── Example 1: exp(k·x) ────────────────────────────────────────────────────
# exp(k·x) = 1 + k·x + k²/2·x² + k³/6·x³ + ...
# c₂ (linear coeff) = k           →  dc₂/dk = 1
# c₃ (quadratic coeff) = k²/2     →  dc₃/dk = k
# c₄ (cubic coeff) = k³/6         →  dc₄/dk = k²/2

function coeff_exp(k)
    x = CTPS(zero(k), 1)
    r = exp(k * x)
    return r.c[2]     # linear coefficient = k
end

function coeff_exp_cubic(k)
    x = CTPS(zero(k), 1)
    r = exp(k * x)
    return r.c[4]     # cubic coefficient = k³/6
end

k0 = 2.0

println("\n--- exp(k·x) ---")
g_linear = ForwardDiff.derivative(coeff_exp, k0)
println("  dc₂/dk at k=$k0: got $g_linear  expected 1.0")

g_cubic = ForwardDiff.derivative(coeff_exp_cubic, k0)
println("  dc₄/dk at k=$k0: got $g_cubic   expected $(k0^2/2)")

# ── Example 2: The Jacobian of multiple map coefficients w.r.t. multiple params
# sin(k·x): c₄ = -k³/6 → dc₄/dk = -k²/2
function coeff_sin_cubic(k)
    x = CTPS(zero(k), 1)
    r = sin(k * x)
    return r.c[4]    # = -k³/6
end

println("\n--- sin(k·x) ---")
g_sin = ForwardDiff.derivative(coeff_sin_cubic, k0)
println("  dc₄/dk at k=$k0: got $g_sin  expected $(-k0^2/2)")

# ── Example 3: Gradient w.r.t. 2 parameters simultaneously (using gradient)
# f(a, b) = coefficient of x² in (a + b·x)^3
# = coefficient of x² in a³ + 3a²bx + 3ab²x² + b³x³
# = 3ab²
# ∇ = [3b², 6ab]

function coeff_poly(params)
    a, b = params[1], params[2]
    x = CTPS(zero(a), 1)
    r = (a + b * x)^3
    return r.c[3]    # x² coefficient = 3ab²
end

params0 = [2.0, 3.0]
grad = ForwardDiff.gradient(coeff_poly, params0)
a0, b0 = params0
println("\n--- (a + b·x)³ quad coeff gradient ---")
println("  ∇(3a·b²) at a=$a0, b=$b0: got $grad  expected $([ 3b0^2, 6a0*b0 ])")

# ── Example 4: Physics case — thin sextupole map
# A thin sextupole of integrated strength K₂L adds a kick proportional to x².
# The output is  x_out = x - K₂L/6 · x²  (normalized coordinates).
# The quadratic coefficient of x_out is  c₃ = -K₂L/6
# → dc₃/d(K₂L) = -1/6

function sext_coeff(K2L)
    x = CTPS(zero(K2L), 1)
    r = x - (K2L / 6) * x^2
    return r.c[3]    # = -K2L/6
end

K2L0 = 3.0
g_sext = ForwardDiff.derivative(sext_coeff, K2L0)
println("\n--- Thin sextupole: dc₃/d(K₂L) ---")
println("  at K₂L=$K2L0: got $g_sext  expected $(-1/6)")

# ── Benchmark ForwardDiff vs finite differences ─────────────────────────────
println("\n--- Benchmark ---")
h = 1e-7

println("ForwardDiff.derivative(coeff_exp, k0):")
b1 = @benchmark ForwardDiff.derivative($coeff_exp, $k0)
display(b1); println()

println("Finite difference ((f(k+h)-f(k))/h):")
b2 = @benchmark (coeff_exp($k0 + $h) - coeff_exp($k0)) / $h
display(b2); println()

# ------------------------------------------------------------------------------
# Section 2: Enzyme.jl — current status and required setup
# ------------------------------------------------------------------------------

println("="^70)
println("Enzyme.jl — AD Through TPSA Map Coefficients")
println("="^70)

using Enzyme
import Enzyme: EnzymeRules

# ── Required: mark non-differentiable TPSA internals as inactive ─────────────
#
# CTPS stores `TPSADesc` (the combinatorial descriptor) alongside the active
# coefficient vector `c::Vector{Float64}`. Enzyme must be told that all
# descriptor-related types are non-differentiable (they are pure index tables).
#
# Without these rules, Enzyme raises EnzymeRuntimeActivityError because it
# sees a constant (`TPSADesc`) being stored into a struct that also carries
# active data (`c`).
#
# Additionally, the internal `DescPool` (thread-local pre-allocated scratch
# buffers) causes Enzyme to conflate constant pool storage with active working
# memory. The `inactive_type` rules below partially address the descriptor
# issue; full Enzyme support would additionally require custom forward/reverse
# rules (EnzymeRules.forward / EnzymeRules.reverse) for mul!, exp!, sin!, etc.
# to properly handle the pool.
#
# Current state: Enzyme Forward mode works correctly for simple arithmetic
# (add, subtract, scale) but produces incorrect results for pool-backed math
# functions (exp, sin, cos, sqrt, log, inv) without custom rules.
# Tracking issue: https://github.com/EnzymeAD/Enzyme.jl/issues (runtime activity)

EnzymeRules.inactive_type(::Type{<:TPSA.TPSADesc})      = true
EnzymeRules.inactive_type(::Type{<:TPSA.DescPool})      = true
EnzymeRules.inactive_type(::Type{<:TPSA.PolyMap})       = true
EnzymeRules.inactive_type(::Type{<:TPSA.MulSchedule2D}) = true
EnzymeRules.inactive_type(::Type{<:TPSA.CompPlan})      = true

println("""
Status of Enzyme + TPSA:
  ✓  Arithmetic (add, scale, mul)  — works with inactive_type rules above
  ✗  Pool-backed math (exp, sin, cos, sqrt, log, inv) — needs custom EnzymeRules
  ✓  ForwardDiff (recommended)     — works out of the box for all operations
""")

# ── Enzyme demo: simple arithmetic (no pool) ─────────────────────────────────
# Multiplication of two CTPS expanded around zero: just a loop over index pairs.
# This path does not use the DescPool and works correctly with Enzyme today.

set_descriptor!(1, 4)   # lower order for simpler example

function linear_combo(k)
    x = CTPS(zero(k), 1)
    r = k * x + k^2      # r.c[1] = k², r.c[2] = k
    return r.c[2]         # = k  →  d/dk = 1
end

g_enz = autodiff(set_runtime_activity(Forward), linear_combo, Duplicated(3.0, 1.0))
fd_g  = ForwardDiff.derivative(linear_combo, 3.0)
println("Simple arithmetic (k·x + k²), dc₂/dk at k=3.0:")
println("  Enzyme (runtime_activity):  $(g_enz[1])")
println("  ForwardDiff:                $fd_g")
println("  Expected:                   1.0")

println()
println("="^70)
println("Recommendation")
println("="^70)
println("""
For differentiating TPSA map coefficients w.r.t. physical parameters:

  • Use ForwardDiff.jl — zero setup, correct for all TPSA operations.
    Works because CTPS{T} is fully T-generic; dual numbers propagate
    through arithmetic, power series loops, and math functions unchanged.

  • Enzyme.jl — planned for future support via custom EnzymeRules.
    The pool pattern (DescPool) and constant struct fields (TPSADesc)
    require explicit forward/reverse rules before Enzyme can be used
    for the full set of TPSA operations.
""")
