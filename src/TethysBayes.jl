"""
    module TethysBayes

Bayesian spatio-temporal modeling and inference suite.
Formerly known as `TethysBayes`.
"""
module TethysBayes

    # `Reexport` pulls another package's names into this module's namespace and
    # re-exports them, so `using TethysBayes` also brings in that package's API. Turing is
    # re-exported because TethysBayes is a framework built on it and users need `@model`,
    # `NUTS` and friends; `Random`, `Distributions` and `DataFrames` are re-exported for
    # the same convenience. Everything else is imported privately and, where the public
    # API needs it, re-exported explicitly further down.
    using Reexport

    @reexport using Random 
    @reexport using Distributions
    @reexport using Turing
    @reexport using DataFrames
    
    using 
        AbstractGPs, 
        AbstractMCMC,
        DynamicPPL,
        AdvancedVI,
        GLM,
        StatsBase,
        Statistics, 
        ADTypes,
        KernelAbstractions,
        Bijectors,
        CategoricalArrays,
        Clustering,
        Dates,
        DelaunayTriangulation,
        DimensionalData,
        DuckDB,
        LogExpFunctions,
        Distances,
        FFTW,
        FillArrays,
        FlexiChains,
        Graphs,
        HypothesisTests,
        Interpolations,
        JLD2,
        KernelFunctions,
        LibGEOS,
        LinearAlgebra,
        NamedArrays,
        NearestNeighbors,
        Optim,
        Optimisers,
        OrderedCollections,
        PDMats,
        Printf,
        PosteriorStats,
        PrecompileTools,
        SHA,
        SparseArrays,
        SpecialFunctions,
        StaticArrays,
        StatsModels,
        ForwardDiff,
        ReverseDiff,
        Enzyme,     
        Wavelets, 
        WaveletsExt, 
        NNlib, 
        CoordRefSystems, 
        Unitful

    # Spatial/temporal partitioning engine. This lives in the
    # Tethys.Partitioning module; its public API is imported explicitly so every
    # internal call site and the `export` block further down keep working.
    # Underscore-prefixed internals of Partitioning (`_compute_scaling_factor`,
    # `nodes`, `_safe_mean`, `ensure_connected!`, ...) are deliberately NOT
    # imported: `model.jl` owns `_compute_scaling_factor(evals::Vector{Float64},
    # rank_deficiency::Int)`, consumed by the smoothness components, and merging
    # the two same-named method tables would silently change dispatch.
    using Tethys.Partitioning:
        assign_spatial_units,
        assign_spatial_units_inferred,
        assign_time_units,
        assign_spatiotemporal_units,
        discretize_data,
        spatial_block_cv,
        spatial_weights_matrix,
        spatial_knn_graph,
        spatial_radius_graph,
        scaling_factor_bym2,
        expand_hull,
        build_hex_mesh_planar,
        map_point_to_units,
        lonlat_to_xy_km,
        xy_km_to_lonlat,
        libgeos_lattice_adjacency_matrix,
        get_polygon_area,
        get_coords_from_geom,
        estimate_local_kde_with_extrapolation,
        _detect_xy_columns,
        _detect_time_column,
        STANDARD_SPATIAL_COORDINATE_PAIRS,
        STANDARD_TEMPORAL_CANDIDATES

    # Unified visualization and mapping engine
    using TethysPlots
    @reexport using TethysPlots

    # Export thin wrappers for essential symbols from demoted packages.
    export LatLon, Mercator, Cartesian
    export @u_str, ustrip

    srcdir = @__DIR__  # TethysBayes/src
    rootdir = dirname(srcdir) # TethysBayes
    docsdir = joinpath(rootdir, "docs") # TethysBayes/docs

    # Bayesian-specific diagnostic plotting stubs for TethysBayesPlotsExt extension
    function TethysBayes_plots end
    function _TethysBayes_plots_impl end
    function model_results_plots end
    function par_credible_interval_plot end
    function par_forest_plot end
    function plot_ppc end
    function plot_prior_vs_posterior end
    function plot_marginal_effects end
    function plot_spatial_residuals end
    function plot_spatiotemporal_facets end

    # Core framework files 
    include( "definitions.jl")  # must be first
    
    # order is important:
    include( "data.jl")
    include( "derivatives.jl")
    include( "hierarchical.jl") 
    include( "input_output.jl")
    include( "leaflet.jl")
    include( "likelihoods.jl")
    include( "model.jl")
    include( "parameters.jl")
    # `partitioning.jl` was relocated into Tethys.Partitioning.
    include( "pipeline.jl")
    include( "reconstruction.jl")  
    include( "par.jl")

    # component definitions
    #
    # `sort` matters: every component file performs top-level registration into the
    # `COMPONENT_TYPE_REGISTRY` / `COMPONENT_CONSTRUCTORS` / `MODEL_TO_STRUCTURE_MAP`
    # globals, so include order determines the order those registries are populated.
    # `readdir` returns entries in filesystem order, which is not specified to be
    # alphabetical; sorting keeps precompilation output reproducible across machines.
    components_dir = joinpath(srcdir, "components")

    for f in sort(readdir(components_dir))
        if endswith(f, ".jl")
            include(joinpath(components_dir, f))
        end
    end

    # work in progress: 
    # include( joinpath(docsdir, "hierarchical_workflow", "hierarchical_workflow.jl") )


    # User-facing API exports
    export 
        @tbm,
        @TethysBayes,
        TethysBayes_config,
        model_results_comprehensive,
        convert_to_chains,
        reconstruct,
        get_optimal_sampler,
        precompute_step_sizes,
        predict,
        show_model,
        TethysBayes_surface_derivatives,
        compute_topographic_metrics,
        generate_rff_params,
        build_structure_template,
        TethysBayes_pipeline,
        compute_network_transfer_matrix,
        reshard_spatial_field,
        summarize_sample_matrix,
        PipelineResult,
        PipelineTierSpec,
        TethysBayes_cv_orchestrator,
        TethysBayes_plots,
        plot_spatial_surface,
        TethysBayes_sample,
        save_plots,
        NESTED_COUPLING_MODES,
        STANDARD_SPATIAL_COORDINATE_PAIRS,
        STANDARD_TEMPORAL_CANDIDATES,
        STANDARD_SPATIAL_UNIT_CANDIDATES,
        STANDARD_SEASONAL_CANDIDATES,
        STANDARD_GROUP_CANDIDATES,

        assign_spatial_units_inferred, 
        plot_kde_simple,
        assign_spatial_units, 
        assign_time_units, 
        assign_spatiotemporal_units,
        discretize_data, 
        TethysBayes_data, 
        expand_hull, 
        generate_mock_hierarchical_datasets,
        map_to_units,
        spatial_block_cv, 
        spatial_weights_matrix, 
        spatial_knn_graph,
        spatial_radius_graph, 
        scaling_factor_bym2,
        load_open_bathymetry,
        extract_hydrodynamic_dataset,
        build_hex_mesh_planar,
        map_point_to_units,
        lonlat_to_xy_km,
        xy_km_to_lonlat,

        ParamRegistry, 
        ParamDescriptor, 
        build_param_registry,
        calibrate_param_registry, 
        get_samples, 
        get_param_samples,
        get_descriptors_by_role,
        get_effects,

        # Model Architecture taxonomy
        AbstractModelArchitecture,
        UnivariateArchitecture,
        MultivariateArchitecture,
        MultifidelityArchitecture,
 
        create_theme, 
        choropleth, 
        timeseries_ci, 
        spatial_graph_plot,
        render_paths!, 
        map_point_occupancy, 
        save_plot, 
        model_results_plots,
        plot_hsi_choropleth, 
        plot_diffusion_map, 
        plot_residence_time_map,
        plot_advection_arrows, 
        plot_velocity_field, 
        plot_hexagonal_field,
        plot_hydrodynamic_stratification,
        plot_hydrodynamic_diffusion,
        plot_hydrodynamic_section,
        householder_to_eigenvector, 
        eigenvector_to_householder,

        # Leaflet Interactive HTML System exports
        LeafletMap, 
        save_html, 
        utm_to_lonlat, 
        lonlat_to_utm,
        leaflet_choropleth, 
        leaflet_spatial_map, 
        leaflet_spatial_graph,
        leaflet_hsi_map, 
        leaflet_diffusion_map, 
        leaflet_residence_time_map,
        leaflet_advection_arrows, 
        leaflet_velocity_field,
        leaflet_spacetime_map,

        # Input / Output & Persistence exports
        save_TethysBayes_model,
        load_TethysBayes_model,
        TethysBayes_SCHEMA_VERSION,

        save_TethysBayes_results, 
        load_TethysBayes_results, 
        query_duckdb,   
        export_posterior_samples_to_duckdb, 
        import_posterior_samples_from_duckdb,
        append_posterior_samples, 
        extend_sampling,
        save_TethysBayes_bundle, 
        load_TethysBayes_bundle,
        export_spatial_results_to_geojson, 
        extract_posterior_priors,
        extract_prior_posterior,
        PriorPosteriorBundle,
        save_model_ensemble, 
        bma_weighted_predictions, 
        save_out_of_sample_predictions,
        export_results_to_csv, 
        compact_duckdb,

        # Hierarchical Pipeline & Provenance exports
        compute_data_traits, 
        init_pipeline_manifest!, 
        write_tier_table!, 
        read_tier_table, 
        has_tier_table,
        get_manifest_entry, 
        update_manifest_entry!, 
        is_tier_up_to_date,
        check_pipeline_status,
        
        par_from_posterior, 
        summarize_par_effects, 
        par_counterfactual,
        export_par_to_table,
        par_credible_interval_plot, 
        par_forest_plot  


    # ---------------------------------------------------------------------------
    # Runtime namespace for generated model code
    #
    # The model body produced by `TethysBayes_text_assembler` used to be `Core.eval`'d in
    # `TethysBayes` itself, so it resolved names -- `Normal`, `filldist`, `MvNormal`, `I`,
    # `logistic`, ... -- only because `TethysBayes` happened to re-export Turing, Distributions
    # and LinearAlgebra. That made the generated code's dependencies implicit and
    # load-bearing: dropping a `using` from this module, or a future change to what we
    # re-export, would silently change the code the generator emits.
    #
    # Generated code is now evaluated in `_GeneratedModelRuntime`, which declares exactly
    # the imports it needs. Anything else it references has to be bound in explicitly
    # below, so a missing dependency surfaces here rather than as a confusing
    # UndefVarError at model-definition time.
    # ---------------------------------------------------------------------------
    module _GeneratedModelRuntime
        using Turing
        using DynamicPPL
        using Distributions
        using LinearAlgebra
        using Statistics
        using Random
        # `logistic` / `logit`: re-exported by LogExpFunctions, which is where the
        # generated body's bare `logistic(...)` calls come from.
        using LogExpFunctions
        # `ifft`/`fft` for the Fourier-basis components, `CategoricalArray` for
        # group-indexed effects.
        using FFTW
        using CategoricalArrays
        # Cubic interpolation for the basis / spline components.
        using Interpolations
        # `nzrange` for the `dag` component's generated neighbour loop. `SparseArrays`
        # exports it and it is defined in TethysBayes, but the runtime module does not inherit
        # TethysBayes's imports, so the unqualified name did not resolve there and `dag` failed at
        # `rand(m)`. Same root cause as a missing entry in `_GENERATED_CODE_HELPERS` below.
        using SparseArrays
    end

    # TethysBayes's own helpers, reachable from the generated body. Bound as values once every
    # include has run, so the module needs no relative-path import.
    #
    # This list is the complete set of package functions the generated model body calls.
    # Adding a name here is a deliberate act: if a component starts calling a new
    # helper, the model fails to build with that name in the error until it is added.
    const _GENERATED_CODE_HELPERS = Symbol[
        :_model_float_type,
        :TethysBayes_Likelihood,
        :evaluate_kernel_matrix,
        :evaluate_cross_kernel_matrix,
        :_sparse_gp_lambda_diag,
        :_zero_null_modes!,
        :anisotropic_matern_spectral_density,
        :householder_to_eigenvector,
        :ar1_statespace,
        :ar2_statespace,
        :_adaptivesmooth_log_marginal_likelihood,
        :_ar1_log_marginal_likelihood,
        :_ar2_log_marginal_likelihood,
        :_barycentric_log_marginal_likelihood,
        :_bcgn_log_marginal_likelihood,
        :_besag_log_marginal_likelihood,
        :_bspline_log_marginal_likelihood,
        :_bym2_log_marginal_likelihood,
        :_cyclic_log_marginal_likelihood,
        :_gp_log_marginal_likelihood,
        :_icar_log_marginal_likelihood,
        :_iid_log_marginal_likelihood,
        :_leroux_log_marginal_likelihood,
        :_moran_log_marginal_likelihood,
        :_pspline_log_marginal_likelihood,
        :_rff_log_marginal_likelihood,
        :_rw1_log_marginal_likelihood,
        :_rw2_log_marginal_likelihood,
        :_sar_log_marginal_likelihood,
        :_spde_log_marginal_likelihood,
        :_tps_log_marginal_likelihood,
        # --- added for the components that could not be sampled at all ---
        # Each of these is a TethysBayes helper called from a generated model body. They were
        # missing from this list, so `fft`, `wavelet` and `hyperbolic` all failed at
        # `rand(m)` with `UndefVarError: <helper> not defined in
        # _GeneratedModelRuntime` -- i.e. these three components could not be sampled at all.
        # The error names the missing binding, which is the intended behaviour of the
        # allowlist: add the name here rather than widening the runtime module.
        :TethysBayes_fourier_basis,                      # `fft`
        :TethysBayes_tensor_product_wavelet_basis,       # `wavelet`
        :_evaluate_hyperbolic_kernel_matrix,      # `hyperbolic`
    ]
    for _helper in _GENERATED_CODE_HELPERS
        isdefined(@__MODULE__, _helper) || error(
            "generated-code helper `$_helper` is not defined; it is declared in " *
            "_GENERATED_CODE_HELPERS but missing from TethysBayes")
        Core.eval(_GeneratedModelRuntime, Expr(:const, _helper, getfield(@__MODULE__, _helper)))
    end

    # ---------------------------------------------------------------------------
    # Precompilation workload
    #
    # TethysBayes's hot path is runtime string codegen (`TethysBayes_text_assembler` -> `Core.eval`),
    # so almost nothing in the model pipeline is compiled until a user first fits a
    # model. This workload walks the representative path once at precompile time:
    # formula parse -> config -> codegen -> model instantiation -> density evaluation ->
    # chain conversion. It is deliberately tiny and uses prior sampling rather than
    # NUTS, because the goal is to compile the machinery, not to spend time tuning.
    #
    # If anything here throws, precompilation of the whole package fails, so the
    # workload must stay minimal and dependency-free.
    # ---------------------------------------------------------------------------
    PrecompileTools.@setup_workload begin
        _wl_df = DataFrames.DataFrame(
            y = [3.0, 5.0, 4.0, 6.0, 2.0, 7.0],
            x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            s_idx = [1, 2, 3, 4, 5, 6],
        )
        _wl_W = spatial_knn_graph([(Float64(i), 0.0) for i in 1:6], 2)[2]

        _wl_m = TethysBayes_core(
            "likelihood(y, family=poisson) ~ intercept() + fixed(x) + random(s_idx, model=bym2)",
            _wl_df, TethysBayes; W = _wl_W, verbose = false)

        # Exercises model evaluation, the world-age trampoline and chain construction.
        _wl_chn = sample(_wl_m, Turing.Prior(), 3; progress = false, check_model = false)
        convert_to_chains(_wl_chn)
    end

end # module TethysBayes

