"""
Investigation of mul! bottleneck in TPSA.jl vs GTPSA
Checks: SIMD, memory, schedule size, access patterns
"""

using BenchmarkTools
using Printf
using InteractiveUtils   # for @code_native / @code_llvm

include(joinpath(@__DIR__, "..", "src", "TPSA.jl"))
using .TPSA

# -------------------------------------------------------------------
# Helper: create a DENSE CTPS (all degrees filled with random values)
# so the degree_mask has all bits set and no inner-loop pairs are skipped
# -------------------------------------------------------------------
function dense_ctps(nv, order)
    TPSA.clear_descriptor!()
    TPSA.set_descriptor!(nv, order)
    desc = TPSA.get_descriptor()
    c = rand(Float64, desc.N)
    mask = (UInt64(1) << min(order+1, 63)) - UInt64(1)   # all degree bits set
    return CTPS{Float64}(c, desc, Ref(mask))
end

# -------------------------------------------------------------------
# 1. Schedule statistics: how big are the k_local matrices?
# -------------------------------------------------------------------
function schedule_stats(desc)
    total_entries = 0
    for s in desc.mul
        total_entries += length(s.k_local)
    end
    N = desc.N
    println("  N (coefficients)     = $N")
    println("  Total k_local entries = $(total_entries)  ($(round(total_entries*4/1024^2, digits=2)) MB for k_local)")
    println("  c storage per CTPS   = $(round(N*8/1024, digits=1)) KB")
end

# -------------------------------------------------------------------
# 2. Dense mul! micro-benchmark
# -------------------------------------------------------------------
function bench_mul_dense(nv, order; samples=100)
    a = dense_ctps(nv, order)
    b = dense_ctps(nv, order)
    r = dense_ctps(nv, order)
    println("\n=== nv=$nv  order=$order ===")
    schedule_stats(a.desc)
    desc = a.desc
    total_fma = sum(length(s.k_local) for s in desc.mul)
    println("  Total FMAs: $total_fma")
    t = @benchmark TPSA.mul!($r, $a, $b) samples=samples evals=1
    med_ms = median(t).time / 1e6
    min_ms = minimum(t).time / 1e6
    fma_per_ns = total_fma / (median(t).time)
    println("  mul! median = $(round(med_ms, digits=4)) ms  min=$(round(min_ms, digits=4)) ms")
    @printf("  FMA throughput: %.2f GFlop/s\n", fma_per_ns)
    return a, b, r
end

# -------------------------------------------------------------------
# 3. SIMD check via LLVM IR
# -------------------------------------------------------------------
function check_simd(a, b, r)
    println("\n--- LLVM IR for mul! (searching for vector ops) ---")
    buf = IOBuffer()
    code_llvm(buf, TPSA.mul!, (typeof(r), typeof(a), typeof(b)); optimize=true)
    ir = String(take!(buf))
    has_fmul_vec = occursin("<2 x double>", ir) || occursin("<4 x double>", ir)
    has_gather   = occursin("gather", lowercase(ir))
    has_llvm_vec = occursin("llvm.fmuladd.v", ir) || occursin("llvm.fma.v", ir)
    println("  Vectorized fmul (2x/4x doubles): $has_fmul_vec")
    println("  Gather instructions: $has_gather")
    println("  llvm.fmuladd/fma vector: $has_llvm_vec")
end

# -------------------------------------------------------------------
# Run investigation
# -------------------------------------------------------------------

println("="^65)
println(" MUL! BOTTLENECK INVESTIGATION (2D k-map schedule)")
println("="^65)

# Warm up
bench_mul_dense(2, 6, samples=400)

bench_mul_dense(4, 8, samples=200)
bench_mul_dense(6, 8, samples=100)
bench_mul_dense(6, 10, samples=80)
bench_mul_dense(6, 12, samples=40)

# SIMD check
println()
TPSA.clear_descriptor!()
TPSA.set_descriptor!(4, 6)
a_s = CTPS(1.0, 1); b_s = CTPS(1.0, 2); r_s = CTPS(0.0)
check_simd(a_s, b_s, r_s)

