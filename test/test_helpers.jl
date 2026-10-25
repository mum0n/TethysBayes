# ==============================================================================
# TethysBayes Test Suite: Shared Test Helpers and Fixtures
# ==============================================================================

using Test
using TethysBayes

using DynamicPPL
# `AbstractMCMC` is referenced explicitly in test_models.jl. A project's direct
# dependencies are not automatically in scope in Julia, so it must be imported here
# rather than relied upon transitively.
using AbstractMCMC
using Distributions
using LinearAlgebra
using DataFrames
using CategoricalArrays
using Turing
using Random
using SparseArrays
using Clustering
using LogExpFunctions
using Graphs
using StatsModels

# ==============================================================================
# Project-local scratch space
#
# Test artifacts (saved bundles, rendered plots, exported HTML) are written inside the
# repository rather than to the OS temp directory, so a test run leaves nothing behind
# outside the project. Use `scratch_dir()` / `scratch_path()` in place of `mktempdir()`
# and `tempdir()`.
# ==============================================================================
const PROJECT_SCRATCH_ROOT = let root = normpath(joinpath(@__DIR__, "..", "work", "test_scratch"))
    mkpath(root)
    root
end

"""
    scratch_dir(prefix="TethysBayes") -> String

A unique directory inside `work/test_scratch`, for test artifacts.
"""
scratch_dir(prefix::AbstractString="TethysBayes") = mktempdir(PROJECT_SCRATCH_ROOT; prefix=prefix)

"""
    scratch_path(name; prefix="TethysBayes") -> String

A unique path inside `work/test_scratch` for a single test artifact.
"""
function scratch_path(name::AbstractString; prefix::AbstractString="TethysBayes")
    d = mktempdir(PROJECT_SCRATCH_ROOT; prefix=prefix)
    return joinpath(d, name)
end

# `Plots`, `StatsPlots` and `ColorSchemes` are *weak* dependencies of TethysBayes (they back the
# `TethysBayesPlotsExt` package extension) and are therefore not present in the main project
# environment. Only the `:persistence` test segment exercises the plotting subsystem, so
# these are imported softly: every other segment stays runnable without the optional
# plotting stack installed. See `test/Project.toml` for the full plotting test env.
const HAS_PLOTTING = let
    ok = true
    for mod in (:Plots, :StatsPlots, :ColorSchemes)
        try
            @eval using $(mod)
        catch
            ok = false
        end
    end
    ok
end

if !HAS_PLOTTING
    @info "Optional plotting stack (Plots/StatsPlots/ColorSchemes) not available; " *
          "the `:persistence` segment will skip its plotting testset."
end

const TethysBayes_Likelihood = TethysBayes.TethysBayes_Likelihood

"""
    create_chain_adj_matrix(n::Int)

Helper function for creating a simple 1D chain spatial adjacency matrix of size `n × n`.
"""
function create_chain_adj_matrix(n::Int)
    W = spzeros(Int, n, n)
    for i in 1:(n - 1)
        W[i, i + 1] = 1
        W[i + 1, i] = 1
    end
    return W
end

"""
    stable_logdiffexp(a, b)

Helper function for stable numerical calculation of log(exp(a) - exp(b)).
"""
stable_logdiffexp(a, b) = a + LogExpFunctions.log1mexp(b - a)

"""
    mock_M_config(N_obs, N_areas, N_time, model_arch="univariate")

Mock model configuration object for isolated unit testing of component primitives.
"""
function mock_M_config(N_obs, N_areas, N_time, model_arch="univariate")
    return (
        data = DataFrame(
            y = rand(N_obs),
            s_idx = repeat(1:N_areas, inner=N_time)[1:N_obs],
            t_idx = repeat(1:N_time, outer=N_areas)[1:N_obs],
            grouping_covariate = repeat(1:N_areas, inner=N_obs ÷ N_areas)[1:N_obs]
        ),
        model_arch = model_arch,
        technical = Dict(
            :component_levels => Dict(),
            :component_indices => Dict()
        )
    )
end

"""
    mock_spec(key, hyper_obj=NamedTuple(), params=Dict(), structure=:any)

Mock component specification NamedTuple.
"""
function mock_spec(key, hyper_obj=NamedTuple(), params=Dict(), structure=:any)
    return (
        key = Symbol(key),
        structure = structure,
        var = string(key),
        hyper = hyper_obj,
        params = params
    )
end

"""
    mock_chain(param_names_and_values::Dict, n_samples=10)

Mock chain dictionary mapping parameter symbols to sample matrices.
"""
function mock_chain(param_names_and_values::Dict, n_samples=10)
    mock_params = Dict{Symbol, Matrix{Float64}}()
    for (name, val) in param_names_and_values
        if val isa Vector
            mock_params[Symbol(name)] = reshape(val, 1, :)
        elseif val isa Matrix
            mock_params[Symbol(name)] = val
        else
            mock_params[Symbol(name)] = fill(Float64(val), 1, n_samples)
        end
    end
    return mock_params
end

# ==============================================================================
# Shared marginalized-component reconstruction check
#
# All 20 marginalized components were tested with a copy-pasted four-line tail:
#
#     eff = TethysBayes.get_effects(comp, chain, spec, M, nothing)
#     @test length(eff.structured) == 1
#     @test size(eff.structured[1]) == (5, 3)
#     @test !all(iszero, eff.structured[1])
#
# Identical every time, and the third line is the important one. `get_effects` used to bail
# to a ZERO matrix for 51 site/parameter combinations behind a single `@warn`, and a zero
# matrix is *finite* -- so the only assertion that catches that class of defect is a
# NONZERO check. Copy-pasted, one block out of twenty eventually loses it.
# ==============================================================================

"""
    check_marginalized_reconstruction(component, chain, spec, M;
                                     n_latent=5, n_draws=3) -> eff

Assert that a marginalized component's `get_effects` returns one outcome block of the
expected shape carrying a **nonzero** field, and return the effect so the caller can make
component-specific assertions on top.
"""
function check_marginalized_reconstruction(
    component, chain, spec, M; n_latent::Integer=5, n_draws::Integer=3
)
    eff = TethysBayes.get_effects(component, chain, spec, M, nothing)
    @test length(eff.structured) == 1
    @test size(eff.structured[1]) == (n_latent, n_draws)
    # Nonzero is the load-bearing assertion here -- see the note above.
    @test !all(iszero, eff.structured[1])
    return eff
end
