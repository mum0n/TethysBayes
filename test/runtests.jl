# ==============================================================================
# TethysBayes Comprehensive Test Suite Orchestrator
# ==============================================================================

import Pkg

# Only activate the package project when this file is executed directly
# (`julia --project=. test/runtests.jl ...`). Under `Pkg.test()` the active
# environment is already the dedicated `test/Project.toml`, and re-activating the
# parent project there would drop the test-only dependencies (notably the optional
# plotting stack) and make them unresolvable.
const RUNNING_UNDER_PKG_TEST =
    basename(dirname(Base.active_project())) == "test"
if !RUNNING_UNDER_PKG_TEST
    Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
    Pkg.instantiate(; io = devnull)
end

# Usage:
#   julia --project=. test/runtests.jl                # Run every segment
#   julia --project=. test/runtests.jl core           # Run only that segment
#   julia --project=. test/runtests.jl core par eiv   # Run several
#
# Segments: core, likelihoods, partitioning, components, derivatives, eiv,
#           pipeline, par, models, multinomial, persistence, nested
#
# Under `Pkg.test()` the whole suite runs; pass segments as test_args:
#   Pkg.test(test_args = ["core", "par"])
#
# A single file can also be run directly:
#   julia --project=. test/test_core_formula.jl
# ==============================================================================

include(joinpath(@__DIR__, "test_helpers.jl"))

const AVAILABLE_SEGMENTS = Dict{Symbol, @NamedTuple{file::String, desc::String}}(
    :core => (
        file = "test_core_formula.jl",
        desc = "Core Formula Parsing, ParamRegistry & Manifolds"
    ),
    :likelihoods => (
        file = "test_likelihoods.jl",
        desc = "Likelihood Engine Taxonomy & Distributions"
    ),
    :partitioning => (
        file = "test_partitioning.jl",
        desc = "Spatial Partitioning, Graph Engine & Polygon Topology"
    ),
    :components => (
        file = "test_components.jl",
        desc = "ComponentModel Interface, NNGP, MCAR"
    ),
    :derivatives => (
        file = "test_derivatives.jl",
        desc = "Surface Derivatives, Topographic Metrics & Bessel BPI"
    ),
    :eiv => (
        file = "test_eiv.jl",
        desc = "Errors-in-Variables (EIV) Covariate Measurement Error Priors"
    ),
    :pipeline => (
        file = "test_pipeline.jl",
        desc = "Modular DAG Pipeline Orchestrator & Cross-Mesh Resharding"
    ),
    :par => (
        file = "test_par.jl",
        desc = "Population Attributable Risk (PAR / PAF) Engine"
    ),
    :models => (
        file = "test_models.jl",
        desc = "Model Instantiation, Smoke Tests & Complex Inference"
    ),
    :reconstruction => (
        file = "test_reconstruction.jl",
        desc = "Reconstruction, Multivariate & Likelihood-Plumbing Regressions"
    ),
    :multinomial => (
        file = "test_multinomial.jl",
        desc = "Multinomial, Categorical & Dirichlet Formulations"
    ),
    :persistence => (
        file = "test_data_persistence_plots.jl",
        desc = "DuckDB Analytics, JLD2 Bundles, GeoJSON & Plot Validation"
    ),
    :nested => (
        file = "test_nested.jl",
        desc = "Nested Multi-Fidelity Models & Sub-Model Linkage Architecture"
    )
)

"""
    parse_requested_segments(args::Vector{String}) -> Vector{Symbol}

Resolves CLI segment arguments to segment keys. Each segment is addressed by its own
name only; there is no alias table, so `julia test/runtests.jl bpi` is an error naming
the available segments rather than a silent guess at what the author meant.
"""
function parse_requested_segments(args::Vector{String})::Vector{Symbol}
    if isempty(args)
        return collect(keys(AVAILABLE_SEGMENTS))
    end

    selected = Symbol[]
    for a in args
        key = Symbol(lowercase(strip(a)))
        if haskey(AVAILABLE_SEGMENTS, key)
            push!(selected, key)
        else
            avail = sort(string.(collect(keys(AVAILABLE_SEGMENTS))))
            @warn "Unrecognized test segment: '$a'. Available segments: $(join(avail, ", "))"
        end
    end

    return isempty(selected) ? sort!(collect(keys(AVAILABLE_SEGMENTS))) : unique(selected)
end

requested_segments = parse_requested_segments(ARGS)

@testset "TethysBayes Comprehensive Test Suite" begin
    for seg_key in requested_segments
        info = AVAILABLE_SEGMENTS[seg_key]
        println("\n" * "="^80)
        println(">>> Running Segment: $(info.desc) ($(info.file))")
        println("="^80)
        include(joinpath(@__DIR__, info.file))
    end
end
