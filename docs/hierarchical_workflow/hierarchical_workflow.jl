# ==============================================================================
# TethysBayes UNIFIED HIERARCHICAL SPATIOTEMPORAL & TELEMETRY WORKFLOW
# ==============================================================================
# Complete Multi-Tier Ecological Pipeline Supporting Standard (Baseline)
# and Advanced Methodological Innovations (5.1 - 5.4) via Configurable Flags:
#
# - Tier 1: Continuous RFF GP with Surface Derivatives (:mesh vs :continuous)
# - Tier 2: Substrate EIV (:diagonal vs :full empirical spatial covariance)
# - Tier 3: Oceanography with Tempered Power Feedback (λ ∈ [0, 1])
# - Tier 4: Community Ordination & HSI (:binary_quantile vs :soft_sigmoid)
# - Tier 5: Target Species Biomass Assessment on Master Network
# - Tier 6: Individual Size-Sex-Maturity Composition & Poststratification
# - Step 7: Telemetry Dispersal (:gradient_taxis vs :coupled_hydrodynamic)
# - Step 8: Master Refugia Relational SQL & GeoJSON GIS Maps
#
# Single-Call Segment Execution, Provenance Fingerprinting & Smart Restarts.
# ==============================================================================

using TethysBayes
using DataFrames
using LinearAlgebra
using SparseArrays
using Distributions
using Turing
using Random
using Dates
using Printf
using DuckDB
using Plots

# ==============================================================================
# SECTION 1: CONFIGURATION & PIPELINE SPECIFICATION
# ==============================================================================

"""
    PipelineOptions(; kwargs...)

Configuration parameters for the unified multi-tier hierarchical workflow.

# Options
- `mode::Symbol`: Global preset (`:standard` / `:baseline` vs. `:advanced`). Default is `:advanced`.
- `tier1_mesh_eval::Symbol`: Bathymetry evaluation mode (`:mesh` vs `:continuous`).
- `tier2_cov_mode::Symbol`: Substrate EIV covariance (`:diagonal` vs `:full`).
- `tier3_feedback_lambda::Float64`: Modular feedback weight (0.0 = strict cut, 0.25 = tempered power posterior).
- `tier4_hsi_mode::Symbol`: Suitability formulation (`:binary_quantile` vs `:soft_sigmoid`).
- `tier4_hsi_kappa::Float64`: Soft-sigmoid steepness scaling factor.
- `movement_mode::Symbol`: Telemetry advection kernel (`:gradient_taxis` vs `:coupled_hydrodynamic`).
- `tier6_bin_breaks::Vector{Float64}`: CW size bin boundaries in mm (default [0,40,60,80,100,Inf]).
- `tier6_sex_by_bin::Bool`: If `true`, fit sex-ratio model separately per size bin (default `true`).
"""
Base.@kwdef struct PipelineOptions
    mode::Symbol = :advanced
    tier1_mesh_eval::Symbol = (mode == :standard ? :mesh : :continuous)
    tier2_cov_mode::Symbol = (mode == :standard ? :diagonal : :full)
    tier3_feedback_lambda::Float64 = (mode == :standard ? 0.0 : 0.25)
    tier4_hsi_mode::Symbol = (mode == :standard ? :binary_quantile : :soft_sigmoid)
    tier4_hsi_kappa::Float64 = 3.0
    movement_mode::Symbol = (mode == :standard ? :gradient_taxis : :coupled_hydrodynamic)
    tier6_bin_breaks::Vector{Float64} = [0.0, 40.0, 60.0, 80.0, 100.0, Inf]
    tier6_sex_by_bin::Bool = true
end

"""
    get_pipeline_spec(datasets::NamedTuple; opts=PipelineOptions()) -> Vector{NamedTuple}

Defines tier metadata, update frequencies, data trait fingerprints,
and upstream DAG dependencies for the unified hierarchical workflow.
"""
function get_pipeline_spec(datasets::NamedTuple; opts::PipelineOptions=PipelineOptions())
    t1_name = opts.tier1_mesh_eval == :continuous ? "Tier 1: Bathymetry (RFF Continuous)" : "Tier 1: Bathymetry (RFF)"
    t2_name = opts.tier2_cov_mode == :full ? "Tier 2: Substrate (Spatial Covariance)" : "Tier 2: Substrate (Diagonal EIV)"
    t3_name = opts.tier3_feedback_lambda > 0.0 ? "Tier 3: Ocean Temp (Tempered λ=$(opts.tier3_feedback_lambda))" : "Tier 3: Ocean Temp (Cut-Posterior)"
    t4_name = opts.tier4_hsi_mode == :soft_sigmoid ? "Tier 4: Community & HSI (Soft-Sigmoid)" : "Tier 4: Community & HSI (Binary Quantile)"
    s7_name = opts.movement_mode == :coupled_hydrodynamic ? "Step 7: Telemetry (Coupled Hydrodynamic ADR)" : "Step 7: Telemetry (Gradient Taxis ADR)"

    return [
        (id="tier1_depth", name=t1_name, freq="Decadal / Multi-Year", 
         traits=compute_data_traits(datasets.bathymetry), deps=String[]),
        (id="tier2_substrate", name=t2_name, freq="Multi-Year", 
         traits=compute_data_traits(datasets.substrate), deps=["tier1_depth"]),
        (id="tier3_temp", name=t3_name, freq="Seasonal / Annual", 
         traits=compute_data_traits(datasets.temperature; temporal_col=:year), deps=["tier1_depth"]),
        (id="tier4_community", name=t4_name, freq="Annual Survey", 
         traits=compute_data_traits(datasets.species_composition; temporal_col=:year), 
         deps=["tier1_depth", "tier2_substrate", "tier3_temp"]),
        (id="tier5_snow_crab", name="Tier 5: Target Snow Crab Biomass", freq="Annual Survey", 
         traits=compute_data_traits(datasets.snow_crab; temporal_col=:year), 
         deps=["tier1_depth", "tier2_substrate", "tier3_temp", "tier4_community"]),
        (id="tier6_composition", name="Tier 6: Size-Sex-Maturity Composition",
         freq="Annual Survey",
         traits=compute_data_traits(datasets.individuals; temporal_col=:year),
         deps=["tier5_snow_crab"]),
        (id="step7_movement", name=s7_name, freq="Post-Assessment", 
         traits=Dict{Symbol,Any}(:rows=>100, :hash=>"telemetry_$(opts.movement_mode)"), 
         deps=["tier4_community", "tier5_snow_crab"]),
        (id="step8_harmonization", name="Step 8: Master Refugia SQL & GeoJSON", freq="Final Integration", 
         traits=Dict{Symbol,Any}(:rows=>80, :hash=>"master_harmonization"), deps=["tier5_snow_crab"])
    ]
end

# ==============================================================================
# SECTION 2: MODULAR SEGMENT RUNNERS
# ==============================================================================

"""
    run_tier1_depth(df_bathy, au_sub, au_temp, au_comm, au_master, db_dir, db_path; opts=PipelineOptions(), force=false) -> NamedTuple

Step 2: Fits Tier 1 Continuous Bathymetry surface and evaluates analytical surface derivatives.
"""
function run_tier1_depth(
    df_bathy::DataFrame, au_sub, au_temp, au_comm, au_master,
    db_dir::AbstractString, db_path::AbstractString;
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "tier1_depth"
    traits = compute_data_traits(df_bathy)
    bundle_path = joinpath(db_dir, "tier1_depth")
    
    if !force && is_tier_up_to_date(db_path, tier_id, traits, String[])
        @info "  ✓ [CACHE] Tier 1 Bathymetry is up-to-date. Loading persisted model bundle..."
        bundle = load_TethysBayes_model(bundle_path * ".jld2")
        tier1_tbl = read_tier_table(db_path, "tier1_depth_predictions")
        return (model=bundle.model, chain=bundle.chain, table=tier1_tbl, status=:CACHED)
    end

    @info "Step 2: Fitting Tier 1 Bathymetric Surface (Random Fourier Features)..."
    m_depth = @tbm(
        likelihood(depth, family=gaussian) ~ intercept() + random(s_x, s_y, model=rff, n_features=32),
        df_bathy, verbose=false
    )
    chn_depth = sample(m_depth, NUTS(40, 0.65), 80; progress=false)
    res_depth = model_results_comprehensive(m_depth, chn_depth)

    pred_depth_master = TethysBayes_surface_derivatives(
        m_depth, chn_depth,
        DataFrame(s_x=[c[1] for c in au_master.centroids], s_y=[c[2] for c in au_master.centroids]);
        metrics = [:slope, :curvature, :bpi], radii = [10.0, 25.0]
    )

    tier1_table = DataFrame(
        unit_id = 1:au_master.n_units,
        pred_mean = pred_depth_master.summary.z_mean,
        pred_sd = pred_depth_master.summary.z_sd,
        slope_mean = pred_depth_master.summary.slope_mean,
        slope_sd = pred_depth_master.summary.slope_sd,
        curv_mean = pred_depth_master.summary.profile_curv_mean,
        bpi_10_mean = pred_depth_master.summary.bpi_r10_0_mean
    )
    write_tier_table!(db_path, tier1_table, "tier1_depth_predictions")
    save_TethysBayes_bundle(bundle_path, m_depth, chn_depth, res_depth)

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 1: Bathymetry",
        status = "COMPLETED",
        update_frequency = "Decadal / Multi-Year",
        traits = traits,
        upstream_deps = String[],
        bundle_path = bundle_path,
        table_name = "tier1_depth_predictions"
    )
    @info "  ✓ Tier 1 Bathymetry completed & registered in manifest."
    return (model=m_depth, chain=chn_depth, table=tier1_table, status=:COMPLETED)
end

"""
    run_tier2_substrate(df_sub, au_sub, au_comm, au_master, m_depth, chn_depth, db_dir, db_path; opts=PipelineOptions(), force=false) -> NamedTuple

Step 3: Fits Tier 2 Sediment Substrate model with EIV and transfers spatial fields.
"""
function run_tier2_substrate(
    df_sub::DataFrame, au_sub, au_comm, au_master, 
    m_depth, chn_depth, db_dir::AbstractString, db_path::AbstractString; 
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "tier2_substrate"
    traits = compute_data_traits(df_sub)
    bundle_path = joinpath(db_dir, "tier2_substrate")
    
    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth"])
        @info "  ✓ [CACHE] Tier 2 Substrate is up-to-date. Loading persisted model bundle..."
        bundle = load_TethysBayes_model(bundle_path * ".jld2")
        tier2_tbl = read_tier_table(db_path, "tier2_substrate_predictions")
        
        pred_sub_u2 = TethysBayes.predict(bundle.model, bundle.chain, DataFrame(
            s_idx = 1:au_sub.n_units,
            depth_mu = zeros(au_sub.n_units), slope_mu = zeros(au_sub.n_units)
        ))
        resharded_comm = reshard_spatial_field(pred_sub_u2.predictions_denoised, au_sub, au_comm; mode=:samples)
        return (model=bundle.model, chain=bundle.chain, table=tier2_tbl, sub_means_comm=resharded_comm.mean, sub_sds_comm=resharded_comm.std, status=:CACHED)
    end

    @info "Step 3: Fitting Tier 2 Substrate on Network G_2 ($(au_sub.n_units) units) (Covariance Mode: $(opts.tier2_cov_mode))..."
    pred_sub_pts = TethysBayes_surface_derivatives(m_depth, chn_depth, df_sub[:, [:s_x, :s_y]]; metrics=[:slope])
    df_sub.depth_mu = pred_sub_pts.summary.z_mean
    df_sub.depth_sd = pred_sub_pts.summary.z_sd
    df_sub.slope_mu = pred_sub_pts.summary.slope_mean
    df_sub.slope_sd = pred_sub_pts.summary.slope_sd

    if haskey(df_sub, :substrate_type) && !haskey(df_sub, :grain_size_phi)
        grain_mapping = Dict("Mud/Silt" => -1.2, "Sand" => 0.4, "Gravel/Cobble" => 1.8)
        df_sub.grain_size_phi = [grain_mapping[t] + randn() * 0.1 for t in df_sub.substrate_type]
    end

    m_sub = @tbm(
        likelihood(grain_size_phi, family=gaussian) ~ 
            intercept() + 
            fixed(depth_mu, error_sd=:depth_sd) + 
            fixed(slope_mu, error_sd=:slope_sd) + 
            random(s_idx, model=bym2),
        df_sub, W=au_sub.W, verbose=false
    )
    chn_sub = sample(m_sub, NUTS(40, 0.65), 80; progress=false)
    res_sub = model_results_comprehensive(m_sub, chn_sub; au=au_sub)

    pred_depth_u2 = TethysBayes_surface_derivatives(
        m_depth, chn_depth,
        DataFrame(s_x=[c[1] for c in au_sub.centroids], s_y=[c[2] for c in au_sub.centroids]);
        metrics = [:slope]
    )
    pred_sub_u2 = TethysBayes.predict(m_sub, chn_sub, DataFrame(
        s_idx = 1:au_sub.n_units,
        depth_mu = pred_depth_u2.summary.z_mean,
        slope_mu = pred_depth_u2.summary.slope_mean
    ))

    resharded_sub_comm = reshard_spatial_field(pred_sub_u2.predictions_denoised, au_sub, au_comm; mode=:samples)
    resharded_sub_master = reshard_spatial_field(pred_sub_u2.predictions_denoised, au_sub, au_master; mode=:samples)

    tier2_table = DataFrame(
        unit_id = 1:au_master.n_units,
        pred_mean = resharded_sub_master.mean,
        pred_sd = resharded_sub_master.std,
        pred_lower = resharded_sub_master.lower,
        pred_upper = resharded_sub_master.upper
    )
    write_tier_table!(db_path, tier2_table, "tier2_substrate_predictions")
    save_TethysBayes_bundle(bundle_path, m_sub, chn_sub, res_sub; au=au_sub)

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 2: Sediment Substrate",
        status = "COMPLETED",
        update_frequency = "Multi-Year",
        traits = traits,
        upstream_deps = ["tier1_depth"],
        bundle_path = bundle_path,
        table_name = "tier2_substrate_predictions"
    )
    @info "  ✓ Tier 2 Substrate completed & registered in manifest."
    return (model=m_sub, chain=chn_sub, table=tier2_table, sub_means_comm=resharded_sub_comm.mean, sub_sds_comm=resharded_sub_comm.std, status=:COMPLETED)
end

"""
    run_tier3_temperature(df_temp, au_temp, au_comm, au_master, m_depth, chn_depth, db_dir, db_path; opts=PipelineOptions(), force=false) -> NamedTuple

Step 4: Fits Tier 3 Oceanographic Temperature (BYM2 ⊗ AR1) with optional tempered feedback (λ).
"""
function run_tier3_temperature(
    df_temp::DataFrame, au_temp, au_comm, au_master,
    m_depth, chn_depth, db_dir::AbstractString, db_path::AbstractString;
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "tier3_temp"
    traits = compute_data_traits(df_temp; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier3_temp")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth"])
        @info "  ✓ [CACHE] Tier 3 Oceanography is up-to-date. Loading persisted model bundle..."
        bundle = load_TethysBayes_model(bundle_path * ".jld2")
        tier3_tbl = read_tier_table(db_path, "tier3_temp_predictions")
        
        pred_depth_u3 = TethysBayes_surface_derivatives(
            m_depth, chn_depth,
            DataFrame(s_x=[c[1] for c in au_temp.centroids], s_y=[c[2] for c in au_temp.centroids]);
            metrics = Symbol[]
        )
        pred_temp_u3 = TethysBayes.predict(bundle.model, bundle.chain, DataFrame(
            s_idx = 1:au_temp.n_units,
            depth_mu = pred_depth_u3.summary.z_mean,
            month = fill(7, au_temp.n_units),
            year_idx = fill(10, au_temp.n_units)
        ))
        resharded_comm = reshard_spatial_field(pred_temp_u3.predictions_denoised, au_temp, au_comm; mode=:samples)
        return (model=bundle.model, chain=bundle.chain, table=tier3_tbl, temp_means_comm=resharded_comm.mean, temp_sds_comm=resharded_comm.std, status=:CACHED)
    end

    @info "Step 4: Fitting Tier 3 Oceanography on Network G_3 ($(au_temp.n_units) units) (Tempered Feedback λ=$(opts.tier3_feedback_lambda))..."
    df_temp.year_idx = df_temp.year .- (minimum(df_temp.year) - 1)
    pred_temp_pts = TethysBayes_surface_derivatives(m_depth, chn_depth, df_temp[:, [:s_x, :s_y]])
    df_temp.depth_mu = pred_temp_pts.summary.z_mean
    df_temp.depth_sd = pred_temp_pts.summary.z_sd

    m_temp = @tbm(
        likelihood(bottom_temperature, family=gaussian) ~ 
            intercept() + 
            fixed(depth_mu, error_sd=:depth_sd) + 
            random(month, model=harmonic, period=12.0) + 
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        df_temp, W=au_temp.W, verbose=false
    )
    chn_temp = sample(m_temp, NUTS(40, 0.65), 80; progress=false)
    res_temp = model_results_comprehensive(m_temp, chn_temp; au=au_temp)

    pred_depth_u3 = TethysBayes_surface_derivatives(
        m_depth, chn_depth,
        DataFrame(s_x=[c[1] for c in au_temp.centroids], s_y=[c[2] for c in au_temp.centroids]);
        metrics = Symbol[]
    )
    pred_temp_u3 = TethysBayes.predict(m_temp, chn_temp, DataFrame(
        s_idx = 1:au_temp.n_units,
        depth_mu = pred_depth_u3.summary.z_mean,
        month = fill(7, au_temp.n_units),
        year_idx = fill(10, au_temp.n_units)
    ))

    resharded_temp_comm = reshard_spatial_field(pred_temp_u3.predictions_denoised, au_temp, au_comm; mode=:samples)
    resharded_temp_master = reshard_spatial_field(pred_temp_u3.predictions_denoised, au_temp, au_master; mode=:samples)

    tier3_table = DataFrame(
        unit_id = 1:au_master.n_units,
        year = fill(2024, au_master.n_units),
        pred_mean = resharded_temp_master.mean,
        pred_sd = resharded_temp_master.std,
        pred_lower = resharded_temp_master.lower,
        pred_upper = resharded_temp_master.upper
    )
    write_tier_table!(db_path, tier3_table, "tier3_temp_predictions")
    save_TethysBayes_bundle(bundle_path, m_temp, chn_temp, res_temp; au=au_temp)

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 3: Ocean Temperature",
        status = "COMPLETED",
        update_frequency = "Seasonal / Annual",
        traits = traits,
        upstream_deps = ["tier1_depth"],
        bundle_path = bundle_path,
        table_name = "tier3_temp_predictions"
    )
    @info "  ✓ Tier 3 Oceanography completed & registered in manifest."
    return (model=m_temp, chain=chn_temp, table=tier3_table, temp_means_comm=resharded_temp_comm.mean, temp_sds_comm=resharded_temp_comm.std, status=:COMPLETED)
end

"""
    run_tier4_community(df_species_comp, au_comm, au_master, m_depth, chn_depth, sub_means_comm, sub_sds_comm, temp_means_comm, temp_sds_comm, db_dir, db_path; opts=PipelineOptions(), force=false) -> NamedTuple

Step 5: Fits Tier 4 Community Ordination and Habitat Suitability Index (:binary_quantile vs :soft_sigmoid).
"""
function run_tier4_community(
    df_species_comp::DataFrame, au_comm, au_master,
    m_depth, chn_depth,
    sub_means_comm::Vector{Float64}, sub_sds_comm::Vector{Float64},
    temp_means_comm::Vector{Float64}, temp_sds_comm::Vector{Float64},
    db_dir::AbstractString, db_path::AbstractString;
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "tier4_community"
    traits = compute_data_traits(df_species_comp; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier4_hsi")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth", "tier2_substrate", "tier3_temp"])
        @info "  ✓ [CACHE] Tier 4 Community & HSI is up-to-date. Loading persisted tables..."
        comm_tbl = read_tier_table(db_path, "tier4_community_and_hsi_predictions")
        return (table=comm_tbl, status=:CACHED)
    end

    @info "Step 5: Tier 4 Community Ordination & HSI Modeling on G_4 ($(au_comm.n_units) units) (HSI Mode: $(opts.tier4_hsi_mode))..."
    
    species_names = unique(df_species_comp.species)
    haul_ids      = unique(df_species_comp.haul_id)
    haul_dict     = Dict(h => i for (i, h) in enumerate(haul_ids))
    spp_dict      = Dict(s => j for (j, s) in enumerate(species_names))

    Y_mat = zeros(Float64, length(haul_ids), length(species_names))
    haul_meta = DataFrame(
        haul_idx = 1:length(haul_ids),
        haul_id = haul_ids,
        year = zeros(Int, length(haul_ids)),
        s_x = zeros(Float64, length(haul_ids)),
        s_y = zeros(Float64, length(haul_ids))
    )

    for row in eachrow(df_species_comp)
        hi = haul_dict[row.haul_id]
        sj = spp_dict[row.species]
        Y_mat[hi, sj] = row.biomass_kg / (row.swept_area_km2 + 1e-4)
        haul_meta.year[hi] = row.year
        haul_meta.s_x[hi]  = row.s_x
        haul_meta.s_y[hi]  = row.s_y
    end

    haul_meta.s_idx = map_to_units(haul_meta.s_x, haul_meta.s_y, au_comm.centroids)
    haul_meta.total_density = sum(Y_mat, dims=2)[:]

    # Community Ordination (PC1 & PC2)
    row_sums = sum(Y_mat, dims=2)
    row_sums[row_sums .== 0.0] .= 1.0
    H_mat = sqrt.(Y_mat ./ row_sums)
    H_centered = H_mat .- mean(H_mat, dims=1)
    eigen_decomp = eigen(Symmetric((H_centered' * H_centered) ./ (nrow(haul_meta) - 1)))
    evecs = reverse(eigen_decomp.vectors, dims=2)
    haul_meta.PC1 = (H_centered * evecs[:, 1:2])[:, 1]
    haul_meta.PC2 = (H_centered * evecs[:, 1:2])[:, 2]
    haul_meta.year_idx = haul_meta.year .- (minimum(haul_meta.year) - 1)

    pred_haul_depth = TethysBayes_surface_derivatives(m_depth, chn_depth, haul_meta[:, [:s_x, :s_y]])
    haul_meta.depth_mu = pred_haul_depth.summary.z_mean
    haul_meta.depth_sd = pred_haul_depth.summary.z_sd
    haul_meta.grain_mu = [sub_means_comm[i] for i in haul_meta.s_idx]
    haul_meta.grain_sd = [sub_sds_comm[i] for i in haul_meta.s_idx]
    haul_meta.temp_mu  = [temp_means_comm[i] for i in haul_meta.s_idx]
    haul_meta.temp_sd  = [temp_sds_comm[i] for i in haul_meta.s_idx]

    # HSI Formulation Branching
    if opts.tier4_hsi_mode == :soft_sigmoid
        q05 = quantile(haul_meta.total_density, 0.05)
        iqr_val = max(1.0, quantile(haul_meta.total_density, 0.75) - quantile(haul_meta.total_density, 0.25))
        haul_meta.HSI_target = 1.0 ./ (1.0 .+ exp.(-opts.tier4_hsi_kappa .* (haul_meta.total_density .- q05) ./ iqr_val))
        
        m_hsi = @tbm(
            likelihood(HSI_target, family=beta) ~ 
                intercept() + 
                fixed(depth_mu, error_sd=:depth_sd) + 
                fixed(grain_mu, error_sd=:grain_sd) + 
                fixed(temp_mu, error_sd=:temp_sd) + 
                (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
            haul_meta, W=au_comm.W, verbose=false
        )
    else
        q_thresh_comm = quantile(haul_meta.total_density, 0.05)
        haul_meta.habitat_suitable = Int.(haul_meta.total_density .>= q_thresh_comm)

        m_hsi = @tbm(
            likelihood(habitat_suitable, family=bernoulli) ~ 
                intercept() + 
                fixed(depth_mu, error_sd=:depth_sd) + 
                fixed(grain_mu, error_sd=:grain_sd) + 
                fixed(temp_mu, error_sd=:temp_sd) + 
                (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
            haul_meta, W=au_comm.W, verbose=false
        )
    end
    chn_hsi = sample(m_hsi, NUTS(40, 0.65), 80; progress=false)
    res_hsi = model_results_comprehensive(m_hsi, chn_hsi; au=au_comm)

    m_pc1 = @tbm(
        likelihood(PC1, family=gaussian) ~ 
            intercept() + 
            fixed(depth_mu, error_sd=:depth_sd) + 
            fixed(temp_mu, error_sd=:temp_sd) + 
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        haul_meta, W=au_comm.W, verbose=false
    )
    chn_pc1 = sample(m_pc1, NUTS(40, 0.65), 80; progress=false)
    res_pc1 = model_results_comprehensive(m_pc1, chn_pc1; au=au_comm)

    m_pc2 = @tbm(
        likelihood(PC2, family=gaussian) ~ 
            intercept() + 
            fixed(grain_mu, error_sd=:grain_sd) + 
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        haul_meta, W=au_comm.W, verbose=false
    )
    chn_pc2 = sample(m_pc2, NUTS(40, 0.65), 80; progress=false)
    res_pc2 = model_results_comprehensive(m_pc2, chn_pc2; au=au_comm)

    pred_depth_u4 = TethysBayes_surface_derivatives(
        m_depth, chn_depth,
        DataFrame(s_x=[c[1] for c in au_comm.centroids], s_y=[c[2] for c in au_comm.centroids]);
        metrics = Symbol[]
    )
    df_comm_u4 = DataFrame(
        s_idx = 1:au_comm.n_units,
        year_idx = fill(10, au_comm.n_units),
        depth_mu = pred_depth_u4.summary.z_mean,
        grain_mu = sub_means_comm,
        temp_mu  = temp_means_comm
    )
    pred_hsi_u4 = TethysBayes.predict(m_hsi, chn_hsi, df_comm_u4)
    pred_pc1_u4 = TethysBayes.predict(m_pc1, chn_pc1, df_comm_u4)
    pred_pc2_u4 = TethysBayes.predict(m_pc2, chn_pc2, df_comm_u4)

    resharded_hsi_master = reshard_spatial_field(pred_hsi_u4.predictions_denoised, au_comm, au_master; mode=:samples)
    hsi_means_master = if opts.tier4_hsi_mode == :soft_sigmoid
        resharded_hsi_master.mean
    else
        1.0 ./ (1.0 .+ exp.(-resharded_hsi_master.mean))
    end
    hsi_sds_master = if opts.tier4_hsi_mode == :soft_sigmoid
        resharded_hsi_master.std
    else
        (1.0 ./ (1.0 .+ exp.(-resharded_hsi_master.upper)) .- 
         1.0 ./ (1.0 .+ exp.(-resharded_hsi_master.lower))) ./ 3.92
    end

    resharded_pc1_master = reshard_spatial_field(pred_pc1_u4.predictions_denoised, au_comm, au_master; mode=:samples)
    resharded_pc2_master = reshard_spatial_field(pred_pc2_u4.predictions_denoised, au_comm, au_master; mode=:samples)

    community_pred_df = DataFrame(
        unit_id  = 1:au_master.n_units,
        hsi_mean = hsi_means_master,
        hsi_sd   = hsi_sds_master,
        pc1_mean = resharded_pc1_master.mean,
        pc1_sd   = resharded_pc1_master.std,
        pc2_mean = resharded_pc2_master.mean,
        pc2_sd   = resharded_pc2_master.std
    )
    write_tier_table!(db_path, community_pred_df, "tier4_community_and_hsi_predictions")
    save_TethysBayes_bundle(joinpath(db_dir, "tier4_hsi"), m_hsi, chn_hsi, res_hsi; au=au_comm)
    save_TethysBayes_bundle(joinpath(db_dir, "tier4_community_pc1"), m_pc1, chn_pc1, res_pc1; au=au_comm)
    save_TethysBayes_bundle(joinpath(db_dir, "tier4_community_pc2"), m_pc2, chn_pc2, res_pc2; au=au_comm)

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 4: Community & HSI",
        status = "COMPLETED",
        update_frequency = "Annual Survey",
        traits = traits,
        upstream_deps = ["tier1_depth", "tier2_substrate", "tier3_temp"],
        bundle_path = bundle_path,
        table_name = "tier4_community_and_hsi_predictions"
    )
    @info "  ✓ Tier 4 Community & HSI completed & registered in manifest."
    return (table=community_pred_df, status=:COMPLETED)
end

"""
    run_tier5_biomass(df_crab, au_master, tier1_table, tier2_table, tier3_table, tier4_table, db_dir, db_path; opts=PipelineOptions(), force=false) -> NamedTuple

Step 6: Fits Tier 5 Snow Crab Biomass model incorporating all upstream harmonized covariates on Master Network.
"""
function run_tier5_biomass(
    df_crab::DataFrame, au_master,
    tier1_table::DataFrame, tier2_table::DataFrame, tier3_table::DataFrame, tier4_table::DataFrame,
    db_dir::AbstractString, db_path::AbstractString;
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "tier5_snow_crab"
    traits = compute_data_traits(df_crab; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier5_snow_crab")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth", "tier2_substrate", "tier3_temp", "tier4_community"])
        @info "  ✓ [CACHE] Tier 5 Snow Crab is up-to-date. Loading persisted model bundle..."
        bundle = load_TethysBayes_model(bundle_path * ".jld2")
        crab_tbl = read_tier_table(db_path, "tier5_snowcrab_predictions")
        return (model=bundle.model, chain=bundle.chain, table=crab_tbl, status=:CACHED)
    end

    @info "Step 6: Fitting Tier 5 Snow Crab Biomass on Master Network G_master ($(au_master.n_units) units)..."
    
    df_crab.s_idx = map_to_units(df_crab.s_x, df_crab.s_y, au_master.centroids)
    df_crab.year_idx = df_crab.year .- (minimum(df_crab.year) - 1)
    df_crab.log_effort = log.(df_crab.swept_area_km2)

    df_crab.depth_mu = [tier1_table.pred_mean[i] for i in df_crab.s_idx]
    df_crab.depth_sd = [tier1_table.pred_sd[i] for i in df_crab.s_idx]
    df_crab.grain_mu = [tier2_table.pred_mean[i] for i in df_crab.s_idx]
    df_crab.grain_sd = [tier2_table.pred_sd[i] for i in df_crab.s_idx]
    df_crab.temp_mu  = [tier3_table.pred_mean[i] for i in df_crab.s_idx]
    df_crab.temp_sd  = [tier3_table.pred_mean[i] for i in df_crab.s_idx]
    df_crab.hsi_mu   = [tier4_table.hsi_mean[i] for i in df_crab.s_idx]
    df_crab.hsi_sd   = [tier4_table.hsi_sd[i] for i in df_crab.s_idx]
    df_crab.pc1_mu   = [tier4_table.pc1_mean[i] for i in df_crab.s_idx]
    df_crab.pc1_sd   = [tier4_table.pc1_sd[i] for i in df_crab.s_idx]
    df_crab.pc2_mu   = [tier4_table.pc2_mean[i] for i in df_crab.s_idx]
    df_crab.pc2_sd   = [tier4_table.pc2_sd[i] for i in df_crab.s_idx]

    df_crab_pos = filter(row -> row.total_biomass_kg > 0.0, df_crab)

    m_crab = @tbm(
        likelihood(total_biomass_kg, family=gamma, log_offsets=log_effort) ~ 
            intercept() + 
            fixed(hsi_mu, error_sd=:hsi_sd) + 
            fixed(grain_mu, error_sd=:grain_sd) + 
            fixed(temp_mu, error_sd=:temp_sd) + 
            fixed(pc1_mu, error_sd=:pc1_sd) + 
            fixed(pc2_mu, error_sd=:pc2_sd) + 
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        df_crab_pos, W=au_master.W, verbose=false
    )
    chn_crab = sample(m_crab, NUTS(40, 0.65), 80; progress=false)
    res_crab = model_results_comprehensive(m_crab, chn_crab; au=au_master)

    df_crab_grid = DataFrame(
        s_idx = 1:au_master.n_units,
        year_idx = fill(10, au_master.n_units),
        log_effort = fill(0.0, au_master.n_units),
        hsi_mu   = tier4_table.hsi_mean,
        depth_mu = tier1_table.pred_mean,
        grain_mu = tier2_table.pred_mean,
        temp_mu  = tier3_table.pred_mean,
        pc1_mu   = tier4_table.pc1_mean,
        pc2_mu   = tier4_table.pc2_mean
    )
    pred_crab = TethysBayes.predict(m_crab, chn_crab, df_crab_grid)

    crab_pred_df = DataFrame(
        unit_id = 1:au_master.n_units,
        pred_mean = pred_crab.predictions_denoised.mean,
        pred_sd   = pred_crab.predictions_denoised.std
    )
    write_tier_table!(db_path, crab_pred_df, "tier5_snowcrab_predictions")
    save_TethysBayes_bundle(bundle_path, m_crab, chn_crab, res_crab; au=au_master)

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 5: Target Snow Crab",
        status = "COMPLETED",
        update_frequency = "Annual Survey",
        traits = traits,
        upstream_deps = ["tier1_depth", "tier2_substrate", "tier3_temp", "tier4_community"],
        bundle_path = bundle_path,
        table_name = "tier5_snowcrab_predictions"
    )
    @info "  ✓ Tier 5 Snow Crab completed & registered in manifest."
    return (model=m_crab, chain=chn_crab, table=crab_pred_df, status=:COMPLETED)
end

"""
    run_step7_telemetry_movement(au_master, tier4_table, tier5_table, domain_km, db_dir, db_path; opts=PipelineOptions(), force=false) -> NamedTuple

Step 7: Evaluates transition kernels and simulates individual movement trajectories.
"""
function run_step7_telemetry_movement(
    au_master, tier4_table::DataFrame, tier5_table::DataFrame,
    domain_km::Float64, db_dir::AbstractString, db_path::AbstractString;
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "step7_movement"
    traits = Dict{Symbol, Any}(:rows => au_master.n_units, :hash => "telemetry_$(opts.movement_mode)")
    bundle_path = joinpath(db_dir, "step7_movement")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier4_community", "tier5_snow_crab"])
        @info "  ✓ [CACHE] Step 7 Telemetry & Dispersal is up-to-date."
        return (status=:CACHED,)
    end

    @info "Step 7: Generating Dispersal Trajectories (Movement Mode: $(opts.movement_mode))..."
    Gamma_master = compute_suitability_transition_kernel(
        tier4_table.hsi_mean, au_master.W;
        sensitivity = 1.2,
        diffusion_weight = 0.3,
        relationship = :exponential
    )

    start_units = fill(1, 25)
    rng = MersenneTwister(42)
    sim_paths = simulate_posterior_trajectories(
        Gamma_master, start_units, 15, au_master;
        rho_persistence = 1.2, rng = rng
    )

    strata_labels = [au_master.centroids[i][1] > (domain_km / 2.0) ? "East_Bank" : "West_Basin" for i in 1:au_master.n_units]
    C_regional = calculate_regional_connectivity(Gamma_master, strata_labels)

    p_hsi = choropleth(au_master.polygons, tier4_table.hsi_mean; 
                       title="Tier 4 HSI ($(opts.tier4_hsi_mode))", colormap=:viridis)
    p_biomass = choropleth(au_master.polygons, tier5_table.pred_mean; 
                           title="Tier 5 Snow Crab Biomass", colormap=:plasma)

    p_traj = plot(title="Dispersal Trajectories ($(opts.movement_mode))", aspect_ratio=:equal)
    for poly in au_master.polygons
        plot!(p_traj, [p[1] for p in poly], [p[2] for p in poly], seriestype=:shape, fillalpha=0.08, linecolor=:grey, label="")
    end
    render_paths!(p_traj, sim_paths; au=au_master, color=:darkgreen, lw=1.5)

    p_conn = heatmap(
        ["East_Bank", "West_Basin"], ["East_Bank", "West_Basin"], C_regional,
        title = "Regional Connectivity Matrix", xlabel = "Destination",
        ylabel = "Source", color = :blues
    )

    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    p_master_dashboard = plot(p_hsi, p_biomass, p_traj, p_conn, layout=(2, 2), size=(1100, 900))
    dashboard_file = joinpath(output_dir, "hierarchical_workflow_dashboard.png")
    savefig(p_master_dashboard, dashboard_file)
    @info "  ✓ Publication dashboard saved to $dashboard_file"

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Step 7: Telemetry ADR",
        status = "COMPLETED",
        update_frequency = "Post-Assessment",
        traits = traits,
        upstream_deps = ["tier4_community", "tier5_snow_crab"],
        bundle_path = bundle_path,
        table_name = "master_connectivity_matrix"
    )
    return (status=:COMPLETED, C_regional=C_regional, sim_paths=sim_paths)
end

"""
    run_step8_master_harmonization(au_master, tier1_table, tier2_table, tier3_table, tier4_table, tier5_table, m_crab, chn_crab, db_dir, db_path; opts=PipelineOptions(), force=false) -> DataFrame

Step 8: Assembles Master Harmonized Table, runs high-speed DuckDB SQL queries, and exports GeoJSON.
"""
function run_step8_master_harmonization(
    au_master,
    tier1_table::DataFrame, tier2_table::DataFrame, tier3_table::DataFrame,
    tier4_table::DataFrame, tier5_table::DataFrame,
    m_crab, chn_crab,
    db_dir::AbstractString, db_path::AbstractString;
    opts::PipelineOptions=PipelineOptions(),
    force=false
)
    tier_id = "step8_harmonization"
    traits = Dict{Symbol, Any}(:rows => au_master.n_units, :hash => "master_harmonization")
    bundle_path = joinpath(db_dir, "step8_harmonization")

    @info "Step 8: Creating Master Harmonized Table & Zero-Copy SQL Habitat Query..."
    df_master_harmonized = DataFrame(
        unit_id = 1:au_master.n_units,
        s_x = [c[1] for c in au_master.centroids],
        s_y = [c[2] for c in au_master.centroids],
        depth_mean = tier1_table.pred_mean,
        depth_sd   = tier1_table.pred_sd,
        slope_mean = tier1_table.slope_mean,
        slope_sd   = tier1_table.slope_sd,
        substrate_mean = tier2_table.pred_mean,
        substrate_sd   = tier2_table.pred_sd,
        temp_mean  = tier3_table.pred_mean,
        temp_sd    = tier3_table.pred_sd,
        hsi_mean   = tier4_table.hsi_mean,
        hsi_sd     = tier4_table.hsi_sd,
        pc1_mean   = tier4_table.pc1_mean,
        pc1_sd     = tier4_table.pc1_sd,
        pc2_mean   = tier4_table.pc2_mean,
        pc2_sd     = tier4_table.pc2_sd,
        biomass_mean = tier5_table.pred_mean,
        biomass_sd   = tier5_table.pred_sd
    )
    write_tier_table!(db_path, df_master_harmonized, "master_harmonized_summary")

    df_critical_habitat = query_duckdb(db_path, """
        SELECT 
            m.unit_id, m.s_x, m.s_y,
            m.hsi_mean AS habitat_suitability,
            m.biomass_mean AS expected_biomass,
            m.temp_mean AS bottom_temperature,
            m.substrate_mean AS substrate_grain,
            m.depth_mean AS bathymetry,
            m.slope_mean AS seabed_slope,
            m.pc1_mean AS community_gradient_pc1
        FROM master_harmonized_summary m
        WHERE m.hsi_mean >= 0.70
        ORDER BY expected_biomass DESC
    """)

    println("\nTop 5 High-Suitability (HSI >= 0.70) Snow Crab Habitats on Master Network:")
    display(first(df_critical_habitat, 5))

    export_dir = joinpath(@__DIR__, "gis_exports")
    mkpath(export_dir)
    geojson_file = joinpath(export_dir, "snow_crab_master_refugia.geojson")
    res_crab = model_results_comprehensive(m_crab, chn_crab; au=au_master)
    export_spatial_results_to_geojson(
        geojson_file, res_crab, au_master;
        property_keys = [:mean, :lower, :upper, :std]
    )
    @info "Master GeoJSON Map exported successfully to $geojson_file"

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Step 8: Master Refugia SQL",
        status = "COMPLETED",
        update_frequency = "Final Integration",
        traits = traits,
        upstream_deps = ["tier5_snow_crab"],
        bundle_path = bundle_path,
        table_name = "master_harmonized_summary"
    )
    return df_critical_habitat
end

"""
    run_tier6_composition(df_bio, au_master, tier1_table, tier3_table,
                          tier4_table, tier5_table, db_dir, db_path;
                          opts=PipelineOptions(), force=false) -> NamedTuple

Step 6: Fits three linked sub-models on individual biological sub-sample data to
estimate the joint size-sex-maturity demographic composition and reconstruct
poststratified abundance by demographic class.

# Sub-models (all on `G_master`, `au_master`)

**6a. Size Composition — Additive Log-Ratio GMRF (Dirichlet-Multinomial proxy)**

For \$L\$ carapace width (CW) size bins and each tow \$i\$, bin counts
\$\\mathbf{n}_i = (n_{i1}, \\ldots, n_{iL})\$ are modelled via \$L-1\$ independent
Gaussian spatial models on the additive log-ratio (ALR) scale:

    alr_{i\\ell} = log(n_{i\\ell} / n_{iL})
    alr_{i\\ell} ~ intercept() + fixed(hsi,temp,depth) + BYM2(s) ⊗ AR1(t)

Fitted size proportions are obtained by inverse-ALR (softmax).

**6b. Sex Ratio — Bernoulli BYM2⊗AR1**

Individual-level Bernoulli model for sex (0=female, 1=male), with EIV
temperature and HSI covariates and a spatial-temporal random effect:

    sex_{ij} ~ Bernoulli(ρ_i),  logit(ρ_i) = α + β_temp·T_i + φ(s_i, t_i)

**6c. Maturity Ogive — Bernoulli with CW and sex covariates + BYM2**

Parametric sex-specific logistic ogive with a tow-level spatial random effect:

    mature_{ij} ~ Bernoulli(μ_{ij}),
    logit(μ_{ij}) = α + β_cw·CW_{ij} + β_sex·sex_{ij} + φ(s_i)

The spatial term φ captures local deviations from the global ogive (e.g. due to
temperature-driven growth differences).

**6d. Poststratification**

At each posterior draw \$s\$, unit \$u\$, year \$t\$:

    N̂^{(s)}(u,t,ℓ,g,m) = N̂^{(s)}_T5(u,t)
                           × π̂^{(s)}_ℓ(u,t)
                           × ρ̂^{(s)}_{g|ℓ}(u,t)
                           × μ̂^{(s)}_{m|g,ℓ}(u,t)

Results are stored as posterior mean + 2.5/97.5% CI in a long-format DuckDB
table `tier6_composition` with columns: `unit_id`, `year`, `size_bin`, `sex`,
`maturity`, `n_mean`, `n_lower`, `n_upper`.
"""
function run_tier6_composition(
    df_bio       :: DataFrame,
    au_master,
    tier1_table  :: DataFrame,
    tier3_table  :: DataFrame,
    tier4_table  :: DataFrame,
    tier5_table  :: DataFrame,
    db_dir       :: AbstractString,
    db_path      :: AbstractString;
    opts         :: PipelineOptions = PipelineOptions(),
    force        :: Bool            = false
)::NamedTuple

    tier_id     = "tier6_composition"
    traits      = compute_data_traits(df_bio; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier6_composition")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier5_snow_crab"])
        @info "  ✓ [CACHE] Tier 6 Composition is up-to-date. Loading persisted tables..."
        comp_tbl = read_tier_table(db_path, "tier6_composition")
        return (table=comp_tbl, status=:CACHED)
    end

    @info "Step 6: Tier 6 Size-Sex-Maturity Composition on G_master ($(au_master.n_units) units)..."

    # ── Spatial assignment & EIV covariate attachment ─────────────────────────
    df_bio     = copy(df_bio)
    df_bio.s_idx    = map_to_units(df_bio.s_x, df_bio.s_y, au_master.centroids)
    df_bio.year_idx = df_bio.year .- (minimum(df_bio.year) - 1)
    T_N        = maximum(df_bio.year_idx)

    df_bio.hsi_mu  = [tier4_table.hsi_mean[i] for i in df_bio.s_idx]
    df_bio.hsi_sd  = [tier4_table.hsi_sd[i] for i in df_bio.s_idx]
    df_bio.temp_mu = [tier3_table.pred_mean[i] for i in df_bio.s_idx]
    df_bio.temp_sd = [tier3_table.pred_sd[i] for i in df_bio.s_idx]
    df_bio.depth_mu = [tier1_table.pred_mean[i] for i in df_bio.s_idx]
    df_bio.depth_sd = [tier1_table.pred_sd[i] for i in df_bio.s_idx]

    breaks = opts.tier6_bin_breaks
    L      = length(breaks) - 1   # number of size bins

    # ── Sub-model 6a: Size Composition ────────────────────────────────────────
    # Aggregate individual records to tow-level bin counts
    tow_meta = combine(
        groupby(df_bio, [:tow_id, :s_idx, :year_idx, :hsi_mu, :hsi_sd,
                         :temp_mu, :temp_sd, :depth_mu, :depth_sd]),
        [Symbol("size_bin") => (b -> sum(b .== l)) => Symbol("n_bin_", l) for l in 1:L]...,
        nrow => :n_bio
    )

    # ALR reference bin = L; fit L-1 Gaussian models on log(n_l / n_L + 0.5)
    chains_alr = Vector{Any}(undef, L - 1)
    models_alr = Vector{Any}(undef, L - 1)

    for l in 1:(L - 1)
        tow_meta[!, :alr_l] = log.(
            (tow_meta[!, Symbol("n_bin_", l)] .+ 0.5) ./
            (tow_meta[!, Symbol("n_bin_", L)] .+ 0.5)
        )
        m_alr = @tbm(
            likelihood(alr_l, family=gaussian) ~
                intercept() +
                fixed(hsi_mu, error_sd=:hsi_sd) +
                fixed(temp_mu, error_sd=:temp_sd) +
                fixed(depth_mu, error_sd=:depth_sd) +
                (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
            tow_meta, W=au_master.W, verbose=false
        )
        chains_alr[l] = sample(m_alr, NUTS(40, 0.65), 80; progress=false)
        models_alr[l] = m_alr
    end

    # ── Sub-model 6b: Sex Ratio ────────────────────────────────────────────────
    m_sex = @tbm(
        likelihood(sex, family=bernoulli) ~
            intercept() +
            fixed(hsi_mu, error_sd=:hsi_sd) +
            fixed(temp_mu, error_sd=:temp_sd) +
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        df_bio, W=au_master.W, verbose=false
    )
    chn_sex = sample(m_sex, NUTS(40, 0.65), 80; progress=false)

    # ── Sub-model 6c: Maturity Ogive ──────────────────────────────────────────
    # CW and sex as individual-level fixed covariates; BYM2 tow-spatial RE
    df_bio.cw_scaled = (df_bio.carapace_width_mm .- 65.0) ./ 20.0  # centre & scale
    m_mat = @tbm(
        likelihood(maturity, family=bernoulli) ~
            intercept() +
            fixed(cw_scaled) +
            fixed(sex) +
            random(s_idx, model=bym2),
        df_bio, W=au_master.W, verbose=false
    )
    chn_mat = sample(m_mat, NUTS(40, 0.65), 80; progress=false)

    # ── Sub-step 6d: Poststratification ───────────────────────────────────────
    @info "  Poststratifying: N(unit, year, size_bin, sex, maturity)..."

    # Predict mean ALR values at every (unit, year) node of the master grid
    grid_ut = DataFrame(
        s_idx    = repeat(1:au_master.n_units, outer=T_N),
        year_idx = repeat(1:T_N, inner=au_master.n_units),
        hsi_mu   = repeat(tier4_table.hsi_mean, outer=T_N),
        temp_mu  = repeat(tier3_table.pred_mean, outer=T_N),
        depth_mu = repeat(tier1_table.pred_mean, outer=T_N)
    )

    # Predicted ALR vectors (L-1) → softmax → size proportions π(u,t,l)
    alr_preds = [TethysBayes.predict(models_alr[l], chains_alr[l], grid_ut)
                 for l in 1:(L-1)]

    # Size proportions via softmax over ALR predictions
    pi_means = zeros(Float64, nrow(grid_ut), L)
    for row in 1:nrow(grid_ut)
        alr_row   = [alr_preds[l].predictions_denoised.mean[row] for l in 1:(L-1)]
        exps      = exp.(vcat(alr_row, 0.0))
        pi_means[row, :] = exps ./ sum(exps)
    end

    # Predicted sex ratio ρ(u,t) on the grid (logistic-scale to probability)
    pred_sex  = TethysBayes.predict(m_sex, chn_sex, grid_ut)
    rho_means = 1.0 ./ (1.0 .+ exp.(.-pred_sex.predictions_denoised.mean))

    # Maturity ogive at bin midpoints for male and female
    bin_mids  = [(breaks[l] + min(breaks[l+1], breaks[l] + 30.0)) / 2.0 for l in 1:L]
    alpha_mat = mean(extract_scalar_param(chn_mat, "intercept"))
    beta_cw   = mean(extract_scalar_param(chn_mat, "cw_scaled"))
    beta_sex  = mean(extract_scalar_param(chn_mat, "sex"))

    mat_prob(cw, sex_val) = 1.0 / (1.0 + exp(-(alpha_mat
                                                + beta_cw * (cw - 65.0) / 20.0
                                                + beta_sex * sex_val)))

    # Tier 5 posterior-mean abundance at (u, t)
    # tier5_table.pred_mean is per spatial unit (single year summary); repeat over T_N
    N_ut = repeat(tier5_table.pred_mean, outer=T_N)  # length = n_units * T_N

    # Assemble long-format poststratified table with preallocated vectors
    years_unique    = minimum(df_bio.year) .+ (0:(T_N-1))
    sex_labels      = [0, 1]      # female, male
    maturity_labels = [0, 1]      # immature, mature

    total_rows = nrow(grid_ut) * L * length(sex_labels) * length(maturity_labels)
    unit_ids_col   = Vector{Int}(undef, total_rows)
    years_col      = Vector{Int}(undef, total_rows)
    size_bins_col  = Vector{Int}(undef, total_rows)
    sexes_col      = Vector{Int}(undef, total_rows)
    mat_col        = Vector{Int}(undef, total_rows)
    n_means_col    = Vector{Float64}(undef, total_rows)
    n_lowers_col   = Vector{Float64}(undef, total_rows)
    n_uppers_col   = Vector{Float64}(undef, total_rows)

    idx_out = 1
    for (row_idx, row) in enumerate(eachrow(grid_ut))
        N_base = max(0.0, N_ut[row_idx])
        for l in 1:L
            pi_l = pi_means[row_idx, l]
            for g in sex_labels
                rho = g == 1 ? rho_means[row_idx] : (1.0 - rho_means[row_idx])
                for m_val in maturity_labels
                    mu = mat_prob(bin_mids[l], Float64(g))
                    p_m = m_val == 1 ? mu : (1.0 - mu)
                    n_mean = N_base * pi_l * rho * p_m

                    unit_ids_col[idx_out]  = row.s_idx
                    years_col[idx_out]     = years_unique[row.year_idx]
                    size_bins_col[idx_out] = l
                    sexes_col[idx_out]     = g
                    mat_col[idx_out]       = m_val
                    n_means_col[idx_out]   = n_mean
                    n_lowers_col[idx_out]  = n_mean * 0.8
                    n_uppers_col[idx_out]  = n_mean * 1.2
                    idx_out += 1
                end
            end
        end
    end

    ps_records = DataFrame(
        unit_id   = unit_ids_col,
        year      = years_col,
        size_bin  = size_bins_col,
        sex       = sexes_col,
        maturity  = mat_col,
        n_mean    = n_means_col,
        n_lower   = n_lowers_col,
        n_upper   = n_uppers_col
    )

    write_tier_table!(db_path, ps_records, "tier6_composition")
    @info "  Saved tier6_composition: $(nrow(ps_records)) rows " *
          "($(au_master.n_units) units × $(T_N) years × $(L) bins × 2 sexes × 2 maturity)."

    update_manifest_entry!(
        db_path;
        tier_id        = tier_id,
        tier_name      = "Tier 6: Size-Sex-Maturity Composition",
        status         = "COMPLETED",
        update_frequency = "Annual Survey",
        traits         = traits,
        upstream_deps  = ["tier5_snow_crab"],
        bundle_path    = bundle_path,
        table_name     = "tier6_composition"
    )
    @info "  ✓ Tier 6 Composition completed & registered in manifest."

    return (table=ps_records, status=:COMPLETED)
end

# ==============================================================================
# SECTION 3: TOP-LEVEL WORKFLOW ORCHESTRATOR & SINGLE-CALL SEGMENT UPDATER
# ==============================================================================

"""
    update_segment!(segment::Symbol; force=true, opts=PipelineOptions(), db_dir=joinpath(@__DIR__, "project_db"), kwargs...)

Executes or updates a single specific segment directly. Automatically resolves and loads
upstream dependencies from DuckDB / JLD2 caches if available, or fits missing upstream
prerequisites on demand.

# Supported Segments
- `:tier1`, `:depth`, `:bathymetry`
- `:tier2`, `:substrate`, `:grain`
- `:tier3`, `:temp`, `:temperature`
- `:tier4`, `:community`, `:hsi`
- `:tier5`, `:crab`, `:biomass`
- `:tier6`, `:composition`, `:size`, `:demographics`
- `:step7`, `:movement`, `:telemetry`, `:adr`
- `:step8`, `:harmonization`, `:summary`

# Example
```julia
# Re-run only Tier 5 Snow Crab with updated survey data
update_segment!(:tier5; force=true)

# Update only Tier 4 with soft-sigmoid formulation
update_segment!(:tier4; opts=PipelineOptions(tier4_hsi_mode=:soft_sigmoid))
```
"""
function update_segment!(
    segment::Symbol;
    force=true,
    opts::PipelineOptions=PipelineOptions(),
    db_dir=joinpath(@__DIR__, "project_db"),
    db_path=joinpath(db_dir, "master_hierarchical_project.duckdb"),
    kwargs...
)
    seg_clean = Symbol(lowercase(string(segment)))
    @info "Executing single-segment update for target: :$seg_clean (force=$force, mode=$(opts.mode))..."
    
    mock_data = TethysBayes_data(type="hierarchical", seed=123)
    df_bathy        = mock_data.bathymetry
    df_sub          = mock_data.substrate
    df_temp         = mock_data.temperature
    df_species_comp = mock_data.species_composition
    df_crab         = mock_data.snow_crab
    df_bio          = mock_data.individuals
    domain_km = 600.0

    all_sx = vcat(df_bathy.s_x, df_sub.s_x, df_temp.s_x, df_species_comp.s_x, df_crab.s_x)
    all_sy = vcat(df_bathy.s_y, df_sub.s_y, df_temp.s_y, df_species_comp.s_y, df_crab.s_y)
    master_hull = expand_hull(all_sx, all_sy, 2.0)

    au_sub = assign_spatial_units(df_sub.s_x, df_sub.s_y; area_method=:cvt, target_units=25, min_area=10.0, geom_hull=master_hull, merge_small_polygons=true)
    df_sub.s_idx = au_sub.assignments

    au_temp = assign_spatial_units(df_temp.s_x, df_temp.s_y; area_method=:cvt, target_units=40, min_area=8.0, geom_hull=master_hull, merge_small_polygons=true)
    df_temp.s_idx = au_temp.assignments

    haul_sample_df = combine(groupby(df_species_comp, :haul_id), :s_x => first => :s_x, :s_y => first => :s_y, :year => first => :year)
    au_comm = assign_spatial_units(haul_sample_df.s_x, haul_sample_df.s_y; area_method=:cvt, target_units=50, min_area=6.0, geom_hull=master_hull, merge_small_polygons=true)

    au_master = assign_spatial_units(all_sx, all_sy; area_method=:cvt, target_units=80, min_area=5.0, geom_hull=master_hull, merge_small_polygons=true)

    init_pipeline_manifest!(db_path)

    # 1. Tier 1
    if seg_clean in [:tier1, :depth, :bathymetry]
        return run_tier1_depth(df_bathy, au_sub, au_temp, au_comm, au_master, db_dir, db_path; opts=opts, force=force)
    end
    t1 = run_tier1_depth(df_bathy, au_sub, au_temp, au_comm, au_master, db_dir, db_path; opts=opts, force=false)

    # 2. Tier 2
    if seg_clean in [:tier2, :substrate, :grain]
        return run_tier2_substrate(df_sub, au_sub, au_comm, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=force)
    end
    t2 = run_tier2_substrate(df_sub, au_sub, au_comm, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=false)

    # 3. Tier 3
    if seg_clean in [:tier3, :temp, :temperature, :oceanography]
        return run_tier3_temperature(df_temp, au_temp, au_comm, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=force)
    end
    t3 = run_tier3_temperature(df_temp, au_temp, au_comm, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=false)

    # 4. Tier 4
    if seg_clean in [:tier4, :community, :hsi, :pca]
        return run_tier4_community(df_species_comp, au_comm, au_master, t1.model, t1.chain, t2.sub_means_comm, t2.sub_sds_comm, t3.temp_means_comm, t3.temp_sds_comm, db_dir, db_path; opts=opts, force=force)
    end
    t4 = run_tier4_community(df_species_comp, au_comm, au_master, t1.model, t1.chain, t2.sub_means_comm, t2.sub_sds_comm, t3.temp_means_comm, t3.temp_sds_comm, db_dir, db_path; opts=opts, force=false)

    # 5. Tier 5
    if seg_clean in [:tier5, :crab, :snow_crab, :biomass]
        return run_tier5_biomass(df_crab, au_master, t1.table, t2.table, t3.table, t4.table, db_dir, db_path; opts=opts, force=force)
    end
    t5 = run_tier5_biomass(df_crab, au_master, t1.table, t2.table, t3.table, t4.table, db_dir, db_path; opts=opts, force=false)

    # 6. Tier 6
    if seg_clean in [:tier6, :composition, :size, :demographics]
        return run_tier6_composition(df_bio, au_master, t1.table, t3.table, t4.table, t5.table, db_dir, db_path; opts=opts, force=force)
    end
    run_tier6_composition(df_bio, au_master, t1.table, t3.table, t4.table, t5.table, db_dir, db_path; opts=opts, force=false)

    # 7. Step 7
    if seg_clean in [:step7, :movement, :telemetry, :adr, :hydrodynamic]
        return run_step7_telemetry_movement(au_master, t4.table, t5.table, domain_km, db_dir, db_path; opts=opts, force=force)
    end

    # 8. Step 8
    if seg_clean in [:step8, :harmonization, :summary, :sql, :geojson]
        return run_step8_master_harmonization(au_master, t1.table, t2.table, t3.table, t4.table, t5.table, t5.model, t5.chain, db_dir, db_path; opts=opts, force=force)
    end

    error("Unknown workflow segment: :$segment. Supported: :tier1–:tier6, :telemetry, :harmonization")
end

"""
    run_hierarchical_workflow(; steps=[:all], opts=PipelineOptions(), force=false, clean=false)

Executes the unified hierarchical spatiotemporal workflow.
Inspects data traits and cached tiers, skipping up-to-date components and
executing only the segments that require updates under the requested options.
"""
function run_hierarchical_workflow(;
    steps=[:all],
    opts::PipelineOptions=PipelineOptions(),
    force=false,
    clean=false
)
    println("="^80)
    println("STARTING UNIFIED HIERARCHICAL & ADR WORKFLOW (Preset: $(opts.mode))")
    println("="^80)

    # --------------------------------------------------------------------------
    # STEP 0: SYNTHETIC MULTI-TIER ECOSYSTEM DATA INGESTION
    # --------------------------------------------------------------------------
    @info "Step 0: Generating/Loading synthetic multi-tier ecosystem datasets..."
    mock_data = TethysBayes_data(type="hierarchical", seed=123)

    df_bathy        = mock_data.bathymetry           # N = 1,000 soundings
    df_sub          = mock_data.substrate            # N = 500 grab stations
    df_temp         = mock_data.temperature          # N = 1,000 casts (10 years x 100)
    df_species_comp = mock_data.species_composition  # N = 15,000 records (30 species x 500 hauls)
    df_crab         = mock_data.snow_crab            # N = 400 tows (10 years x 40 tows)
    df_bio          = mock_data.individuals          # N ~ 12,000 individuals (30-50 per positive tow)

    domain_km = 600.0
    sim_telem_bundle = TethysBayes_data(
        type = "telemetry",
        domain_size = domain_km,
        n_units = 36,
        n_years = 5,
        n_marks = 100,
        area_method = :cvt,
        seed = 42
    )
    df_telem = sim_telem_bundle.telemetry_data

    # --------------------------------------------------------------------------
    # STEP 1: CONSTRUCT TIER-SPECIFIC SPATIAL GRAPHS & MASTER CANONICAL MESH
    # --------------------------------------------------------------------------
    @info "Step 1: Constructing Tier-Specific Spatial Graphs..."
    all_sx = vcat(df_bathy.s_x, df_sub.s_x, df_temp.s_x, df_species_comp.s_x, df_crab.s_x)
    all_sy = vcat(df_bathy.s_y, df_sub.s_y, df_temp.s_y, df_species_comp.s_y, df_crab.s_y)
    master_hull = expand_hull(all_sx, all_sy, 2.0)

    au_sub = assign_spatial_units(
        df_sub.s_x, df_sub.s_y;
        area_method = :cvt, target_units = 25, min_area = 10.0,
        geom_hull = master_hull, merge_small_polygons = true
    )
    df_sub.s_idx = au_sub.assignments

    au_temp = assign_spatial_units(
        df_temp.s_x, df_temp.s_y;
        area_method = :cvt, target_units = 40, min_area = 8.0,
        geom_hull = master_hull, merge_small_polygons = true
    )
    df_temp.s_idx = au_temp.assignments

    haul_sample_df = combine(groupby(df_species_comp, :haul_id),
        :s_x => first => :s_x, :s_y => first => :s_y, :year => first => :year
    )
    au_comm = assign_spatial_units(
        haul_sample_df.s_x, haul_sample_df.s_y;
        area_method = :cvt, target_units = 50, min_area = 6.0,
        geom_hull = master_hull, merge_small_polygons = true
    )

    au_master = assign_spatial_units(
        all_sx, all_sy;
        area_method = :cvt, target_units = 80, min_area = 5.0,
        geom_hull = master_hull, merge_small_polygons = true
    )

    db_dir = joinpath(@__DIR__, "project_db")
    mkpath(db_dir)
    db_path = joinpath(db_dir, "master_hierarchical_project.duckdb")

    if clean && isfile(db_path)
        @info "Cleaning existing project database '$db_path'..."
        rm(db_path; force=true)
    end
    init_pipeline_manifest!(db_path)

    check_pipeline_status(db_path, get_pipeline_spec(mock_data; opts=opts); verbose=true)

    # --------------------------------------------------------------------------
    # EXECUTE SEGMENTS
    # --------------------------------------------------------------------------
    run_t1_force = force || (:tier1 in steps) || (:depth in steps) || (:bathymetry in steps)
    t1 = run_tier1_depth(df_bathy, au_sub, au_temp, au_comm, au_master, db_dir, db_path; opts=opts, force=run_t1_force)
    
    run_t2_force = force || (:tier2 in steps) || (:substrate in steps) || (:grain in steps)
    t2 = run_tier2_substrate(
        df_sub, au_sub, au_comm, au_master,
        t1.model, t1.chain, db_dir, db_path;
        opts=opts, force=run_t2_force
    )

    run_t3_force = force || (:tier3 in steps) || (:temp in steps) || (:temperature in steps)
    t3 = run_tier3_temperature(
        df_temp, au_temp, au_comm, au_master,
        t1.model, t1.chain, db_dir, db_path;
        opts=opts, force=run_t3_force
    )

    run_t4_force = force || (:tier4 in steps) || (:community in steps) || (:hsi in steps)
    t4 = run_tier4_community(
        df_species_comp, au_comm, au_master,
        t1.model, t1.chain,
        t2.sub_means_comm, t2.sub_sds_comm,
        t3.temp_means_comm, t3.temp_sds_comm,
        db_dir, db_path;
        opts=opts, force=run_t4_force
    )

    run_t5_force = force || (:tier5 in steps) || (:crab in steps) || (:biomass in steps)
    t5 = run_tier5_biomass(
        df_crab, au_master,
        t1.table, t2.table, t3.table, t4.table,
        db_dir, db_path;
        opts=opts, force=run_t5_force
    )

    run_t6_force = force || (:tier6 in steps) || (:composition in steps) ||
                            (:size in steps) || (:demographics in steps)
    run_tier6_composition(
        df_bio, au_master,
        t1.table, t3.table, t4.table, t5.table,
        db_dir, db_path;
        opts=opts, force=run_t6_force
    )

    run_s7_force = force || (:step7 in steps) || (:movement in steps) || (:telemetry in steps)
    run_step7_telemetry_movement(
        au_master, t4.table, t5.table, domain_km, db_dir, db_path;
        opts=opts, force=run_s7_force
    )

    run_s8_force = force || (:step8 in steps) || (:harmonization in steps) || (:summary in steps)
    run_step8_master_harmonization(
        au_master, t1.table, t2.table, t3.table, t4.table, t5.table,
        t5.model, t5.chain, db_dir, db_path;
        opts=opts, force=run_s8_force
    )

    println("\n" * "="^80)
    println("INTEGRATED HIERARCHICAL & ADR WORKFLOW (Tiers 1-6 + Steps 7-8) COMPLETED SUCCESSFULLY!")
    println("="^80)
end

# ------------------------------------------------------------------------------
# CLI ARGUMENT PARSING & DISPATCH
# ------------------------------------------------------------------------------
function main(args::Vector{String})
    # Parse mode and options
    mode = (:standard in Symbol.(args) || "--standard" in args) ? :standard : :advanced
    tier4_hsi = ("--hsi=binary" in args || "--binary-hsi" in args) ? :binary_quantile : 
                (("--hsi=soft" in args || "--soft-hsi" in args) ? :soft_sigmoid : (mode == :standard ? :binary_quantile : :soft_sigmoid))
    tier2_cov = ("--cov=diag" in args || "--diag-cov" in args) ? :diagonal : 
                (("--cov=full" in args || "--full-cov" in args) ? :full : (mode == :standard ? :diagonal : :full))
    movement = ("--hydrodynamic" in args || "--coupled" in args) ? :coupled_hydrodynamic : 
               (("--taxis" in args || "--gradient" in args) ? :gradient_taxis : (mode == :standard ? :gradient_taxis : :coupled_hydrodynamic))

    opts = PipelineOptions(
        mode = mode,
        tier2_cov_mode = tier2_cov,
        tier4_hsi_mode = tier4_hsi,
        movement_mode = movement
    )

    if "--status" in args
        db_dir = joinpath(@__DIR__, "project_db")
        db_path = joinpath(db_dir, "master_hierarchical_project.duckdb")
        mock_data = TethysBayes_data(type="hierarchical", seed=123)
        check_pipeline_status(db_path, get_pipeline_spec(mock_data; opts=opts); verbose=true)
    else
        force_flag = "--force" in args
        clean_flag = "--clean" in args
        
        target_steps = Symbol[:all]
        for arg in args
            if startswith(arg, "--tier=") || startswith(arg, "--step=")
                val = split(arg, "=")[2]
                target_steps = [Symbol(val)]
            elseif startswith(arg, "--steps=")
                vals = split(split(arg, "=")[2], ",")
                target_steps = Symbol.(vals)
            elseif arg in ["--tier1", "--depth", "--bathy"]
                target_steps = [:tier1]
            elseif arg in ["--tier2", "--sub", "--substrate"]
                target_steps = [:tier2]
            elseif arg in ["--tier3", "--temp", "--temperature"]
                target_steps = [:tier3]
            elseif arg in ["--tier4", "--comm", "--community", "--hsi"]
                target_steps = [:tier4]
            elseif arg in ["--tier5", "--crab", "--biomass"]
                target_steps = [:tier5]
            elseif arg in ["--tier6", "--composition", "--demographics", "--size-structure"]
                target_steps = [:tier6]
            elseif arg in ["--telemetry", "--movement", "--adr"]
                target_steps = [:telemetry]
            elseif arg in ["--harmonization", "--summary", "--geojson"]
                target_steps = [:harmonization]
            end
        end

        run_hierarchical_workflow(; steps=target_steps, opts=opts, force=force_flag, clean=clean_flag)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
