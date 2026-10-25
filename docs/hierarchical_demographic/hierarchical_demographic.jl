# ==============================================================================
# TethysBayes HIERARCHICAL DEMOGRAPHIC & STAGE-STRUCTURED WORKFLOW (APPROACH B)
# ==============================================================================
# Class-Expanded Factorial & Stage-Specific Hurdle Pipeline:
#
#   P/A ~ size + sex + maturity + environment + spatial/temporal GMRFs
#
# Complements Approach A (Conditional Decomposition) by directly modeling:
# - Stage-specific habitat suitability indices: HSI_c(s, t) = P(Encounter_c)
# - Stage-specific positive density: D_c(s, t) = p_c · μ_c
# - Ontogenetic migration & stage-specific advection vectors: v_c = γ_c ∇HSI_c
# - Bias accumulation diagnostics & sampling zero trade-offs
# ==============================================================================

using TethysBayes
using DataFrames
import DataFrames: combine, groupby, select, transform, nrow, ncol, leftjoin, innerjoin
using LinearAlgebra
using SparseArrays
using Distributions
using Turing
using Random
using Dates
using DuckDB
using Plots

_sample(m, args...; kwargs...) = Base.invokelatest(sample, m, args...; kwargs...)
_predict(m, args...; kwargs...) = Base.invokelatest(TethysBayes.predict, m, args...; kwargs...)




# ==============================================================================
# SECTION 1: CONFIGURATION & PIPELINE SPECIFICATION
# ==============================================================================

"""
    DemographicPipelineOptions(; kwargs...)

Configuration parameters for the class-expanded demographic workflow (Approach B).

# Options
- `class_scheme::Symbol`: Demographic resolution (`:functional_stages` [4 stages:
  Juvenile, Adolescent Male, Mature Female, Commercial Male] vs. `:fine_bins`
  [5 CW bins × 2 sexes × 2 maturity = 20 classes]). Default is `:functional_stages`.
- `tier1_mesh_eval::Symbol`: Bathymetry evaluation mode (`:mesh` vs `:continuous`).
- `tier2_cov_mode::Symbol`: Substrate EIV covariance (`:diagonal` vs `:full`).
- `tier3_feedback_lambda::Float64`: Modular feedback weight (0.0 = cut-posterior,
  0.25 = tempered power posterior).
- `hurdle_positive_family::Symbol`: Distribution for positive stage catch (`:gamma`
  vs. `:lognormal`).
- `movement_mode::Symbol`: Telemetry advection kernel (`:stage_specific_taxis` vs.
  `:coupled_hydrodynamic`).
- `tier6_bin_breaks::Vector{Float64}`: Carapace width break points in mm.
"""
Base.@kwdef struct DemographicPipelineOptions
    class_scheme::Symbol = :functional_stages
    ordination_mode::Symbol = :joint_eigen # :joint_eigen vs :two_step_householder
    n_ordination_factors::Int = 2
    tier1_mesh_eval::Symbol = :continuous
    tier2_cov_mode::Symbol = :full
    tier3_feedback_lambda::Float64 = 0.25
    hurdle_positive_family::Symbol = :gamma
    movement_mode::Symbol = :stage_specific_taxis
    tier6_bin_breaks::Vector{Float64} = [0.0, 40.0, 60.0, 80.0, 100.0, Inf]
end


"""
    get_demographic_pipeline_spec(datasets::NamedTuple;
                                  opts=DemographicPipelineOptions()) -> Vector{NamedTuple}

Defines tier metadata, update frequencies, data trait fingerprints, and upstream DAG
dependencies for the class-expanded demographic workflow.
"""
function get_demographic_pipeline_spec(
    datasets::NamedTuple;
    opts::DemographicPipelineOptions = DemographicPipelineOptions()
)::Vector{NamedTuple}
    t1_name = opts.tier1_mesh_eval == :continuous ?
        "Tier 1: Bathymetry (RFF Continuous)" : "Tier 1: Bathymetry (RFF)"
    t2_name = opts.tier2_cov_mode == :full ?
        "Tier 2: Substrate (Spatial Covariance)" : "Tier 2: Substrate (Diagonal EIV)"
    t3_name = opts.tier3_feedback_lambda > 0.0 ?
        "Tier 3: Ocean Temp (Tempered λ=$(opts.tier3_feedback_lambda))" :
        "Tier 3: Ocean Temp (Cut-Posterior)"
    t4c_name = "Tier 4C: Community Ordination (Householder Eigen)"
    t4d_name = "Tier 4D: Stage-Structured Encounter & HSI ($(opts.class_scheme))"
    t5d_name = "Tier 5D: Stage Density & Hurdle Positive Biomass ($(opts.hurdle_positive_family))"
    s6d_name = "Step 6D: Stage-Specific Dispersal & Ontogenetic Migration ($(opts.movement_mode))"

    return [
        (id="tier1_depth", name=t1_name, freq="Decadal / Multi-Year",
         traits=compute_data_traits(datasets.bathymetry), deps=String[]),
        (id="tier2_substrate", name=t2_name, freq="Multi-Year",
         traits=compute_data_traits(datasets.substrate), deps=["tier1_depth"]),
        (id="tier3_temp", name=t3_name, freq="Seasonal / Annual",
         traits=compute_data_traits(datasets.temperature; temporal_col=:year),
         deps=["tier1_depth"]),
        (id="tier4c_community", name=t4c_name, freq="Annual Survey",
         traits=compute_data_traits(datasets.species_composition; temporal_col=:year),
         deps=["tier1_depth", "tier2_substrate", "tier3_temp"]),
        (id="tier4d_demographic_hsi", name=t4d_name, freq="Annual Survey",
         traits=compute_data_traits(datasets.individuals; temporal_col=:year),
         deps=["tier1_depth", "tier2_substrate", "tier3_temp", "tier4c_community"]),
        (id="tier5d_demographic_density", name=t5d_name, freq="Annual Survey",
         traits=compute_data_traits(datasets.snow_crab; temporal_col=:year),
         deps=["tier1_depth", "tier2_substrate", "tier3_temp", "tier4c_community", "tier4d_demographic_hsi"]),
        (id="step6d_demographic_movement", name=s6d_name, freq="Post-Assessment",
         traits=Dict{Symbol,Any}(:rows=>80, :hash=>"demog_movement_$(opts.movement_mode)"),
         deps=["tier4d_demographic_hsi", "tier5d_demographic_density"]),
        (id="step7d_demographic_harmonization", name="Step 7D: Demographic Refugia & GIS",
         freq="Final Integration",
         traits=Dict{Symbol,Any}(:rows=>80, :hash=>"demog_harmonization"),
         deps=["tier5d_demographic_density"])
    ]
end

# ==============================================================================
# SECTION 2: DATA PREPARATION & RECTANGULAR TOW EXPANSION
# ==============================================================================

"""
    expand_tows_to_demographic_classes(df_crab::DataFrame, df_bio::DataFrame;
                                       scheme::Symbol = :functional_stages,
                                       breaks::Vector{Float64} = [0.0, 40.0, 60.0, 80.0, 100.0, Inf]) -> DataFrame

Expands tow-level catch records into a balanced, rectangular table with one row per
tow and demographic class.

# Functional Stage Classification (`scheme = :functional_stages`)
1. `:immature_juvenile`: Carapace width < 60 mm, immature (male or female).
2. `:adolescent_male`: Male, Carapace width 60–94 mm, immature / adolescent.
3. `:mature_female`: Female, mature (spawning stock & egg-bearing potential).
4. `:commercial_male`: Male, Carapace width ≥ 95 mm, mature (commercial legal size).

# Fine Bins Classification (`scheme = :fine_bins`)
Creates `L × 2 × 2` classes for all size bin × sex × maturity combinations.

# Arguments
- `df_crab::DataFrame`: Tow metadata and aggregate catch (Tier 5 table).
- `df_bio::DataFrame`: Individual biological sub-sample measurements (Tier 6 table).
- `scheme::Symbol`: Classification scheme (`:functional_stages` or `:fine_bins`).
- `breaks::Vector{Float64}`: Bin break points for carapace width.

# Returns
- `DataFrame`: Balanced rectangular dataset containing columns:
  `:tow_id`, `:year`, `:month`, `:s_x`, `:s_y`, `:swept_area_km2`,
  `:stage_id`, `:stage_label`, `:present`, `:sample_count`, `:biomass_kg`.
"""
function expand_tows_to_demographic_classes(
    df_crab::DataFrame,
    df_bio::DataFrame;
    scheme::Symbol = :functional_stages,
    breaks::Vector{Float64} = [0.0, 40.0, 60.0, 80.0, 100.0, Inf]
)::DataFrame

    L = length(breaks) - 1
    tows_unique = unique(df_crab.tow_id)
    n_tows = length(tows_unique)

    # Classify individual biological records
    df_bio_classified = copy(df_bio)
    if scheme == :functional_stages
        stages = Symbol[:immature_juvenile, :adolescent_male, :mature_female, :commercial_male]
        stage_labels = Dict(
            :immature_juvenile => "Immature Juvenile (<60mm)",
            :adolescent_male   => "Adolescent Male (60-94mm)",
            :mature_female     => "Mature Female (Spawning Stock)",
            :commercial_male   => "Commercial Male (≥95mm Mature)"
        )

        function _assign_stage(cw, sex, mat)::Symbol
            if cw < 60.0 && mat == 0
                return :immature_juvenile
            elseif sex == 1 && cw >= 95.0 && mat == 1
                return :commercial_male
            elseif sex == 1 && mat == 0
                return :adolescent_male
            elseif sex == 0 && mat == 1
                return :mature_female
            elseif sex == 0
                return :immature_juvenile
            else
                return :adolescent_male
            end
        end

        df_bio_classified[!, :class_id] = [
            _assign_stage(r.carapace_width_mm, r.sex, r.maturity)
            for r in eachrow(df_bio_classified)
        ]
    else # :fine_bins
        stages = Symbol[Symbol("bin_", l, "_sex_", g, "_mat_", m)
                        for l in 1:L for g in 0:1 for m in 0:1]
        stage_labels = Dict(s => string(s) for s in stages)

        df_bio_classified[!, :class_id] = [
            Symbol("bin_", r.size_bin, "_sex_", r.sex, "_mat_", r.maturity)
            for r in eachrow(df_bio_classified)
        ]
    end

    # Aggregate individual counts and biomass proportions by tow and class
    bio_counts = DataFrames.combine(
        groupby(df_bio_classified, [:tow_id, :class_id]),

        nrow => :n_sampled,
        :carapace_width_mm => mean => :mean_cw
    )

    tow_dict = Dict(r.tow_id => r for r in eachrow(df_crab))
    bio_map = Dict((r.tow_id, r.class_id) => r.n_sampled for r in eachrow(bio_counts))

    n_stages = length(stages)
    total_rows = n_tows * n_stages

    # Pre-allocate column vectors for high-speed construction
    tow_ids_col      = Vector{String}(undef, total_rows)
    years_col        = Vector{Int}(undef, total_rows)
    months_col       = Vector{Int}(undef, total_rows)
    sx_col           = Vector{Float64}(undef, total_rows)
    sy_col           = Vector{Float64}(undef, total_rows)
    swept_col        = Vector{Float64}(undef, total_rows)
    stage_id_col     = Vector{Symbol}(undef, total_rows)
    stage_label_col  = Vector{String}(undef, total_rows)
    present_col      = Vector{Int}(undef, total_rows)
    sample_count_col = Vector{Int}(undef, total_rows)
    biomass_kg_col   = Vector{Float64}(undef, total_rows)

    idx = 1
    for tow_row in eachrow(df_crab)
        tid = tow_row.tow_id
        tot_bio_kg = Float64(tow_row.total_biomass_kg)
        tot_cnt = Int(tow_row.total_count)

        # Count total sampled crabs for this tow across all stages
        tow_bio_samples = [get(bio_map, (tid, st), 0) for st in stages]
        tow_sample_sum = sum(tow_bio_samples)

        for (st_idx, st) in enumerate(stages)
            cnt_sampled = tow_bio_samples[st_idx]
            is_present = (cnt_sampled > 0) ? 1 : 0

            # Apportion aggregate tow biomass proportional to sub-sample representation
            prop = tow_sample_sum > 0 ? (Float64(cnt_sampled) / tow_sample_sum) : 0.0
            stage_bio = round(tot_bio_kg * prop, digits=3)

            tow_ids_col[idx]      = tid
            years_col[idx]        = tow_row.year
            months_col[idx]       = tow_row.month
            sx_col[idx]           = tow_row.s_x
            sy_col[idx]           = tow_row.s_y
            swept_col[idx]        = tow_row.swept_area_km2
            stage_id_col[idx]     = st
            stage_label_col[idx]  = stage_labels[st]
            present_col[idx]      = is_present
            sample_count_col[idx] = cnt_sampled
            biomass_kg_col[idx]   = stage_bio
            idx += 1
        end
    end

    return DataFrame(
        tow_id         = tow_ids_col,
        year           = years_col,
        month          = months_col,
        s_x            = sx_col,
        s_y            = sy_col,
        swept_area_km2 = swept_col,
        stage_id       = stage_id_col,
        stage_label    = stage_label_col,
        present        = present_col,
        sample_count   = sample_count_col,
        biomass_kg     = biomass_kg_col
    )
end

# ==============================================================================
# SECTION 3: MODULAR SEGMENT RUNNERS
# ==============================================================================

"""
    run_tier1_depth(df_bathy, au_sub, au_temp, au_master, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 2: Fits Tier 1 Continuous Bathymetry surface and evaluates analytical derivatives.
"""
function run_tier1_depth(
    df_bathy::DataFrame, au_sub, au_temp, au_master,
    db_dir::AbstractString, db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
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
        likelihood(depth, family=gaussian) ~
            intercept() + random(s_x, s_y, model=rff, n_features=32),
        df_bathy, verbose=false
    )
    chn_depth = _sample(m_depth, NUTS(40, 0.65), 80; progress=false)
    res_depth = model_results_comprehensive(m_depth, chn_depth)


    pred_depth_master = TethysBayes_surface_derivatives(
        m_depth, chn_depth,
        DataFrame(s_x=[c[1] for c in au_master.centroids], s_y=[c[2] for c in au_master.centroids]);
        metrics = [:slope, :curvature, :bpi], radii = [10.0, 25.0]
    )

    tier1_table = DataFrame(
        unit_id     = 1:au_master.n_units,
        pred_mean   = pred_depth_master.summary.z_mean,
        pred_sd     = pred_depth_master.summary.z_sd,
        slope_mean  = pred_depth_master.summary.slope_mean,
        slope_sd    = pred_depth_master.summary.slope_sd,
        curv_mean   = pred_depth_master.summary.profile_curv_mean,
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
    run_tier2_substrate(df_sub, au_sub, au_master, m_depth, chn_depth, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 3: Fits Tier 2 Sediment Substrate model with EIV and transfers spatial fields.
"""
function run_tier2_substrate(
    df_sub::DataFrame, au_sub, au_master,
    m_depth, chn_depth, db_dir::AbstractString, db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
    tier_id = "tier2_substrate"
    traits = compute_data_traits(df_sub)
    bundle_path = joinpath(db_dir, "tier2_substrate")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth"])
        @info "  ✓ [CACHE] Tier 2 Substrate is up-to-date. Loading persisted model bundle..."
        bundle = load_TethysBayes_model(bundle_path * ".jld2")
        tier2_tbl = read_tier_table(db_path, "tier2_substrate_predictions")
        return (model=bundle.model, chain=bundle.chain, table=tier2_tbl, status=:CACHED)
    end

    @info "Step 3: Fitting Tier 2 Substrate on Network G_2 ($(au_sub.n_units) units)..."
    pred_sub_pts = TethysBayes_surface_derivatives(m_depth, chn_depth, df_sub[:, [:s_x, :s_y]]; metrics=[:slope])
    df_sub.depth_mu = pred_sub_pts.summary.z_mean
    df_sub.depth_sd = pred_sub_pts.summary.z_sd
    df_sub.slope_mu = pred_sub_pts.summary.slope_mean
    df_sub.slope_sd = pred_sub_pts.summary.slope_sd

    m_sub = @tbm(
        likelihood(grain_size_phi, family=gaussian) ~
            intercept() +
            fixed(depth_mu, error_sd=:depth_sd) +
            fixed(slope_mu, error_sd=:slope_sd) +
            random(s_idx, model=bym2),
        df_sub, W=au_sub.W, verbose=false
    )
    chn_sub = _sample(m_sub, NUTS(40, 0.65), 80; progress=false)
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

    resharded_sub_master = reshard_spatial_field(
        pred_sub_u2.predictions_denoised, au_sub, au_master; mode=:samples
    )

    tier2_table = DataFrame(
        unit_id    = 1:au_master.n_units,
        pred_mean  = resharded_sub_master.mean,
        pred_sd    = resharded_sub_master.std,
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
    return (model=m_sub, chain=chn_sub, table=tier2_table, status=:COMPLETED)
end

"""
    run_tier3_temperature(df_temp, au_temp, au_master, m_depth, chn_depth, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 4: Fits Tier 3 Oceanographic Temperature (BYM2 ⊗ AR1) with harmonic seasonality.
"""
function run_tier3_temperature(
    df_temp::DataFrame, au_temp, au_master,
    m_depth, chn_depth, db_dir::AbstractString, db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
    tier_id = "tier3_temp"
    traits = compute_data_traits(df_temp; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier3_temp")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth"])
        @info "  ✓ [CACHE] Tier 3 Oceanography is up-to-date. Loading persisted model bundle..."
        bundle = load_TethysBayes_model(bundle_path * ".jld2")
        tier3_tbl = read_tier_table(db_path, "tier3_temp_predictions")
        return (model=bundle.model, chain=bundle.chain, table=tier3_tbl, status=:CACHED)
    end

    @info "Step 4: Fitting Tier 3 Oceanography on Network G_3 ($(au_temp.n_units) units)..."
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
    chn_temp = _sample(m_temp, NUTS(40, 0.65), 80; progress=false)
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

    resharded_temp_master = reshard_spatial_field(
        pred_temp_u3.predictions_denoised, au_temp, au_master; mode=:samples
    )

    tier3_table = DataFrame(
        unit_id    = 1:au_master.n_units,
        year       = fill(2024, au_master.n_units),
        pred_mean  = resharded_temp_master.mean,
        pred_sd    = resharded_temp_master.std,
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
    return (model=m_temp, chain=chn_temp, table=tier3_table, status=:COMPLETED)
end


"""
    run_tier4c_community(df_species_comp::DataFrame, au_comm, au_master, tier1_table::DataFrame, tier2_table::DataFrame, tier3_table::DataFrame, db_dir::AbstractString, db_path::AbstractString; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 4C: Fits Multi-Species Community Ordination via Householder Reflection Transformation
(`eigen()` in TethysBayes) on Community Graph G_comm, extracting dominant community axes (PC1, PC2)
and resharding to G_master.

# Mathematical Formulation
1. **Hellinger Transformation of Trawl Catches**:
   \$H_{ik} = \\sqrt{Y_{ik} / \\sum_{j=1}^K Y_{ij}}\$
2. **Orthonormal Loadings via Householder Reflections**:
   \$U = \\prod_{i=1}^k \\left( \\mathbf{I} - 2 \\frac{\\mathbf{v}_i \\mathbf{v}_i^\\top}{\\Vert\\mathbf{v}_i\\Vert^2} \\right) \\begin{bmatrix} \\mathbf{I}_{k \\times k} \\\\ \\mathbf{0}_{(p-k) \\times k} \\end{bmatrix}\$
   where Householder reflector vectors \$\\mathbf{v}_i\$ are computed via `eigenvector_to_householder(U)`
   and reconstructed via `householder_to_eigenvector(v_mat, p, k)`.
3. **Spatial GMRF Mapping**:
   Models spatial ordination scores \$(\\text{PC1}, \\text{PC2})\$ with depth, substrate, and
   temperature EIV covariates on \$G_{\\text{comm}}\$, resharding posterior sample distributions to \$G_{\\text{master}}\$.
"""
function run_tier4c_community(
    df_species_comp::DataFrame,
    au_comm,
    au_master,
    tier1_table::DataFrame,
    tier2_table::DataFrame,
    tier3_table::DataFrame,
    db_dir::AbstractString,
    db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
    tier_id = "tier4c_community"
    traits = compute_data_traits(df_species_comp; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier4c_community")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth", "tier2_substrate", "tier3_temp"])
        @info "  ✓ [CACHE] Tier 4C Community Ordination (Householder Eigen) is up-to-date. Loading persisted tables..."
        comm_tbl = read_tier_table(db_path, "tier4c_community_predictions")
        return (table=comm_tbl, status=:CACHED)
    end

    @info "Step 4C: Multi-Species Community Ordination via Householder Transform on G_comm ($(au_comm.n_units) units)..."

    species_names = unique(df_species_comp.species)
    haul_ids      = unique(df_species_comp.haul_id)
    haul_dict     = Dict(h => i for (i, h) in enumerate(haul_ids))
    spp_dict      = Dict(s => j for (j, s) in enumerate(species_names))

    Y_mat = zeros(Float64, length(haul_ids), length(species_names))
    haul_meta = DataFrame(
        haul_idx = 1:length(haul_ids),
        haul_id  = haul_ids,
        year     = zeros(Int, length(haul_ids)),
        s_x      = zeros(Float64, length(haul_ids)),
        s_y      = zeros(Float64, length(haul_ids))
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
    haul_meta.year_idx = haul_meta.year .- (minimum(haul_meta.year) - 1)
    T_N = maximum(haul_meta.year_idx)

    # 1. Hellinger Transformation & Centering
    row_sums = sum(Y_mat, dims=2)
    row_sums[row_sums .== 0.0] .= 1.0
    H_mat = sqrt.(Y_mat ./ row_sums)
    H_centered = H_mat .- mean(H_mat, dims=1)

    # 2. Householder Eigenvector Parameterization
    N_hauls, K_spp = size(H_centered)
    n_factors = min(2, K_spp - 1)
    cov_H = Symmetric((H_centered' * H_centered) ./ max(1, N_hauls - 1))
    raw_eig = LinearAlgebra.eigen(cov_H)

    # Sort eigenvalues/vectors descending
    sort_idx = sortperm(raw_eig.values, rev=true)
    U_full = raw_eig.vectors[:, sort_idx]
    U_k = U_full[:, 1:n_factors]

    # Convert orthonormal eigenvectors to Householder reflector vectors
    v_mat = eigenvector_to_householder(U_k)
    # Reconstruct orthonormal loadings via Householder reflections
    U_householder = householder_to_eigenvector(v_mat, K_spp, n_factors)

    # Compute Ordination Factor Scores
    scores = H_centered * U_householder
    haul_meta.PC1 = scores[:, 1]
    haul_meta.PC2 = scores[:, 2]

    # Attach Upstream EIV Covariates from Tiers 1-3 mapped to G_comm
    pred_haul_depth = map_to_units(haul_meta.s_x, haul_meta.s_y, [c for c in au_master.centroids])
    haul_meta.depth_mu = [tier1_table.pred_mean[i] for i in pred_haul_depth]
    haul_meta.depth_sd = [tier1_table.pred_sd[i] for i in pred_haul_depth]
    haul_meta.grain_mu = [tier2_table.pred_mean[i] for i in pred_haul_depth]
    haul_meta.grain_sd = [tier2_table.pred_sd[i] for i in pred_haul_depth]
    haul_meta.temp_mu  = [tier3_table.pred_mean[i] for i in pred_haul_depth]
    haul_meta.temp_sd  = [tier3_table.pred_sd[i] for i in pred_haul_depth]

    # 3. Fit Spatial GMRF on G_comm for PC1 and PC2
    m_pc1 = @tbm(
        likelihood(PC1, family=gaussian) ~
            intercept() +
            fixed(depth_mu, error_sd=:depth_sd) +
            fixed(grain_mu, error_sd=:grain_sd) +
            fixed(temp_mu, error_sd=:temp_sd) +
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        haul_meta, W=au_comm.W, verbose=false
    )
    chn_pc1 = _sample(m_pc1, NUTS(40, 0.65), 80; progress=false)
    res_pc1 = model_results_comprehensive(m_pc1, chn_pc1; au=au_comm)

    m_pc2 = @tbm(
        likelihood(PC2, family=gaussian) ~
            intercept() +
            fixed(depth_mu, error_sd=:depth_sd) +
            fixed(grain_mu, error_sd=:grain_sd) +
            fixed(temp_mu, error_sd=:temp_sd) +
            (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
        haul_meta, W=au_comm.W, verbose=false
    )
    chn_pc2 = _sample(m_pc2, NUTS(40, 0.65), 80; progress=false)
    res_pc2 = model_results_comprehensive(m_pc2, chn_pc2; au=au_comm)


    # 4. Predict and Reshard to G_master
    grid_comm = DataFrame(
        s_idx    = repeat(1:au_comm.n_units, outer=T_N),
        year_idx = repeat(1:T_N, inner=au_comm.n_units)
    )
    grid_comm.depth_mu = repeat([tier1_table.pred_mean[min(i, nrow(tier1_table))] for i in 1:au_comm.n_units], outer=T_N)
    grid_comm.depth_sd = repeat([tier1_table.pred_sd[min(i, nrow(tier1_table))] for i in 1:au_comm.n_units], outer=T_N)
    grid_comm.grain_mu = repeat([tier2_table.pred_mean[min(i, nrow(tier2_table))] for i in 1:au_comm.n_units], outer=T_N)
    grid_comm.grain_sd = repeat([tier2_table.pred_sd[min(i, nrow(tier2_table))] for i in 1:au_comm.n_units], outer=T_N)
    grid_comm.temp_mu  = repeat([tier3_table.pred_mean[min(i, nrow(tier3_table))] for i in 1:au_comm.n_units], outer=T_N)
    grid_comm.temp_sd  = repeat([tier3_table.pred_sd[min(i, nrow(tier3_table))] for i in 1:au_comm.n_units], outer=T_N)

    pred_pc1_comm = TethysBayes.predict(m_pc1, chn_pc1, grid_comm)
    pred_pc2_comm = TethysBayes.predict(m_pc2, chn_pc2, grid_comm)

    pc1_samples_comm = pred_pc1_comm.predictions_denoised.samples[1:au_comm.n_units, :]
    pc2_samples_comm = pred_pc2_comm.predictions_denoised.samples[1:au_comm.n_units, :]

    resharded_pc1 = reshard_spatial_field(pc1_samples_comm, au_comm, au_master; mode=:samples)
    resharded_pc2 = reshard_spatial_field(pc2_samples_comm, au_comm, au_master; mode=:samples)

    tier4c_table = DataFrame(
        unit_id    = 1:au_master.n_units,
        year       = fill(2024, au_master.n_units),
        pc1_mean   = resharded_pc1.mean,
        pc1_sd     = resharded_pc1.std,
        pc1_lower  = resharded_pc1.lower,
        pc1_upper  = resharded_pc1.upper,
        pc2_mean   = resharded_pc2.mean,
        pc2_sd     = resharded_pc2.std,
        pc2_lower  = resharded_pc2.lower,
        pc2_upper  = resharded_pc2.upper
    )

    write_tier_table!(db_path, tier4c_table, "tier4c_community_predictions")
    save_TethysBayes_bundle(joinpath(db_dir, "tier4c_pc1"), m_pc1, chn_pc1, res_pc1; au=au_comm)
    save_TethysBayes_bundle(joinpath(db_dir, "tier4c_pc2"), m_pc2, chn_pc2, res_pc2; au=au_comm)

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 4C: Community Ordination (Householder Eigen)",
        status = "COMPLETED",
        update_frequency = "Annual Survey",
        traits = traits,
        upstream_deps = ["tier1_depth", "tier2_substrate", "tier3_temp"],
        bundle_path = bundle_path,
        table_name = "tier4c_community_predictions"
    )
    @info "  ✓ Tier 4C Community Ordination (Householder Eigen) completed & registered in manifest."
    return (model_pc1=m_pc1, model_pc2=m_pc2, table=tier4c_table, v_mat=v_mat, status=:COMPLETED)
end

"""
    run_tier4d_demographic_hsi(df_expanded, au_master, tier1_table, tier2_table, tier3_table, tier4c_table, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 5D: Fits Stage-Structured Presence/Absence Models (Bernoulli BYM2 ⊗ AR1) for each
demographic life stage on Master Network G_master, incorporating community ordination axes (PC1, PC2)
and estimating stage-specific HSI fields.

# Mathematical Formulation
For each demographic class \$c \\in \\mathcal{C}\$:
```math
y_{i, c} \\sim \\operatorname{Bernoulli}(p_{i, c})
```
```math
\\operatorname{logit}(p_{i, c}) = \\alpha_c + \\beta_c^{\\text{depth}} \\tilde{z}_i + \\beta_c^{\\text{temp}} \\tilde{T}_i + \\beta_c^{\\text{grain}} \\tilde{g}_i + \\beta_c^{\\text{pc1}} \\widetilde{\\text{PC1}}_i + \\beta_c^{\\text{pc2}} \\widetilde{\\text{PC2}}_i + \\phi_c(s_i) + \\psi_c(s_i, t_i)
```
"""
function run_tier4d_demographic_hsi(
    df_expanded::DataFrame,
    au_master,
    tier1_table::DataFrame,
    tier2_table::DataFrame,
    tier3_table::DataFrame,
    tier4c_table::DataFrame,
    db_dir::AbstractString,
    db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
    tier_id = "tier4d_demographic_hsi"
    traits = compute_data_traits(df_expanded; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier4d_demographic_hsi")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth", "tier2_substrate", "tier3_temp", "tier4c_community"])
        @info "  ✓ [CACHE] Tier 4D Demographic HSI is up-to-date. Loading persisted tables..."
        hsi_tbl = read_tier_table(db_path, "tier4d_demographic_hsi_predictions")
        return (table=hsi_tbl, status=:CACHED)
    end

    @info "Step 5D: Fitting Stage-Structured Encounter & HSI Models on G_master ($(au_master.n_units) units)..."

    # Attach spatial unit index and upstream EIV covariates
    df_exp = copy(df_expanded)
    df_exp.s_idx    = map_to_units(df_exp.s_x, df_exp.s_y, au_master.centroids)
    df_exp.year_idx = df_exp.year .- (minimum(df_exp.year) - 1)
    T_N = maximum(df_exp.year_idx)

    df_exp.depth_mu = [tier1_table.pred_mean[i] for i in df_exp.s_idx]
    df_exp.depth_sd = [tier1_table.pred_sd[i] for i in df_exp.s_idx]
    df_exp.grain_mu = [tier2_table.pred_mean[i] for i in df_exp.s_idx]
    df_exp.grain_sd = [tier2_table.pred_sd[i] for i in df_exp.s_idx]
    df_exp.temp_mu  = [tier3_table.pred_mean[i] for i in df_exp.s_idx]
    df_exp.temp_sd  = [tier3_table.pred_sd[i] for i in df_exp.s_idx]
    df_exp.pc1_mu   = [tier4c_table.pc1_mean[i] for i in df_exp.s_idx]
    df_exp.pc1_sd   = [tier4c_table.pc1_sd[i] for i in df_exp.s_idx]
    df_exp.pc2_mu   = [tier4c_table.pc2_mean[i] for i in df_exp.s_idx]
    df_exp.pc2_sd   = [tier4c_table.pc2_sd[i] for i in df_exp.s_idx]

    stages_unique = unique(df_exp.stage_id)
    grid_master = DataFrame(
        s_idx    = repeat(1:au_master.n_units, outer=T_N),
        year_idx = repeat(1:T_N, inner=au_master.n_units),
        depth_mu = repeat(tier1_table.pred_mean, outer=T_N),
        depth_sd = repeat(tier1_table.pred_sd, outer=T_N),
        grain_mu = repeat(tier2_table.pred_mean, outer=T_N),
        grain_sd = repeat(tier2_table.pred_sd, outer=T_N),
        temp_mu  = repeat(tier3_table.pred_mean, outer=T_N),
        temp_sd  = repeat(tier3_table.pred_sd, outer=T_N),
        pc1_mu   = repeat(tier4c_table.pc1_mean, outer=T_N),
        pc1_sd   = repeat(tier4c_table.pc1_sd, outer=T_N),
        pc2_mu   = repeat(tier4c_table.pc2_mean, outer=T_N),
        pc2_sd   = repeat(tier4c_table.pc2_sd, outer=T_N)
    )

    hsi_results_df = DataFrame(
        unit_id  = grid_master.s_idx,
        year     = minimum(df_exp.year) .+ (grid_master.year_idx .- 1)
    )

    spp_cols = [:Gadus_morhua, :Melanogrammus_aeglefinus, :Pollachius_virens, :Sebastes_fasciatus]
    for sp in spp_cols
        if hasproperty(df_exp, sp)
            grid_master[!, sp] = fill(0.0, nrow(grid_master))
        end
    end

    models_dict = Dict{Symbol, Any}()
    chains_dict = Dict{Symbol, Any}()

    for st in stages_unique
        @info "  Fitting Encounter / HSI model for stage: :$st (Ordination: $(opts.ordination_mode))..."
        df_st = filter(r -> r.stage_id == st, df_exp)

        m_st = if opts.ordination_mode == :joint_eigen
            @tbm(
                likelihood(present, family=bernoulli) ~
                    intercept() +
                    fixed(depth_mu, error_sd=:depth_sd) +
                    fixed(grain_mu, error_sd=:grain_sd) +
                    fixed(temp_mu, error_sd=:temp_sd) +
                    eigen(Gadus_morhua, Melanogrammus_aeglefinus, Pollachius_virens, Sebastes_fasciatus, n_factors=2) +
                    (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
                df_st, W=au_master.W, verbose=false
            )
        else
            @tbm(
                likelihood(present, family=bernoulli) ~
                    intercept() +
                    fixed(depth_mu, error_sd=:depth_sd) +
                    fixed(grain_mu, error_sd=:grain_sd) +
                    fixed(temp_mu, error_sd=:temp_sd) +
                    fixed(pc1_mu, error_sd=:pc1_sd) +
                    fixed(pc2_mu, error_sd=:pc2_sd) +
                    (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
                df_st, W=au_master.W, verbose=false
            )
        end

        chn_st = _sample(m_st, NUTS(40, 0.65), 80; progress=false)
        res_st = model_results_comprehensive(m_st, chn_st; au=au_master)


        pred_st = TethysBayes.predict(m_st, chn_st, grid_master)
        # Transform latent logit prediction to encounter probability (HSI) scale
        p_means = 1.0 ./ (1.0 .+ exp.(.-pred_st.predictions_denoised.mean))
        p_sds   = pred_st.predictions_denoised.std .* p_means .* (1.0 .- p_means)

        hsi_results_df[!, Symbol("hsi_", st, "_mean")] = p_means
        hsi_results_df[!, Symbol("hsi_", st, "_sd")]   = p_sds

        models_dict[st] = m_st
        chains_dict[st] = chn_st
        save_TethysBayes_bundle(joinpath(db_dir, "tier4d_hsi_$(st)"), m_st, chn_st, res_st; au=au_master)
    end


    write_tier_table!(db_path, hsi_results_df, "tier4d_demographic_hsi_predictions")

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 4D: Stage-Structured HSI",
        status = "COMPLETED",
        update_frequency = "Annual Survey",
        traits = traits,
        upstream_deps = ["tier1_depth", "tier2_substrate", "tier3_temp", "tier4c_community"],
        bundle_path = bundle_path,
        table_name = "tier4d_demographic_hsi_predictions"
    )
    @info "  ✓ Tier 4D Demographic HSI completed & registered in manifest."
    return (models=models_dict, chains=chains_dict, table=hsi_results_df, status=:COMPLETED)
end

"""
    run_tier5d_demographic_density(df_expanded, au_master, tier1_table, tier2_table, tier3_table, tier4c_table, tier4d_table, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 6D: Fits Stage-Structured Positive Catch & Density Hurdle Models on Master Network.
Evaluates expected demographic density and aggregate sum-to-total divergence diagnostics.
"""
function run_tier5d_demographic_density(
    df_expanded::DataFrame,
    au_master,
    tier1_table::DataFrame,
    tier2_table::DataFrame,
    tier3_table::DataFrame,
    tier4c_table::DataFrame,
    tier4d_table::DataFrame,
    db_dir::AbstractString,
    db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
    tier_id = "tier5d_demographic_density"
    traits = compute_data_traits(df_expanded; temporal_col=:year)
    bundle_path = joinpath(db_dir, "tier5d_demographic_density")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier1_depth", "tier2_substrate", "tier3_temp", "tier4c_community", "tier4d_demographic_hsi"])
        @info "  ✓ [CACHE] Tier 5D Demographic Density is up-to-date. Loading persisted tables..."
        dens_tbl = read_tier_table(db_path, "tier5d_demographic_density_predictions")
        return (table=dens_tbl, status=:CACHED)
    end

    @info "Step 6D: Fitting Stage-Structured Density Models on G_master ($(au_master.n_units) units)..."

    df_exp = copy(df_expanded)
    df_exp.s_idx      = map_to_units(df_exp.s_x, df_exp.s_y, au_master.centroids)
    df_exp.year_idx   = df_exp.year .- (minimum(df_exp.year) - 1)
    df_exp.log_effort = log.(max.(0.01, df_exp.swept_area_km2))
    T_N = maximum(df_exp.year_idx)

    df_exp.depth_mu = [tier1_table.pred_mean[i] for i in df_exp.s_idx]
    df_exp.depth_sd = [tier1_table.pred_sd[i] for i in df_exp.s_idx]
    df_exp.grain_mu = [tier2_table.pred_mean[i] for i in df_exp.s_idx]
    df_exp.grain_sd = [tier2_table.pred_sd[i] for i in df_exp.s_idx]
    df_exp.temp_mu  = [tier3_table.pred_mean[i] for i in df_exp.s_idx]
    df_exp.temp_sd  = [tier3_table.pred_sd[i] for i in df_exp.s_idx]
    df_exp.pc1_mu   = [tier4c_table.pc1_mean[i] for i in df_exp.s_idx]
    df_exp.pc1_sd   = [tier4c_table.pc1_sd[i] for i in df_exp.s_idx]
    df_exp.pc2_mu   = [tier4c_table.pc2_mean[i] for i in df_exp.s_idx]
    df_exp.pc2_sd   = [tier4c_table.pc2_sd[i] for i in df_exp.s_idx]

    stages_unique = unique(df_exp.stage_id)
    grid_master = DataFrame(
        s_idx      = repeat(1:au_master.n_units, outer=T_N),
        year_idx   = repeat(1:T_N, inner=au_master.n_units),
        log_effort = fill(0.0, au_master.n_units * T_N),
        depth_mu   = repeat(tier1_table.pred_mean, outer=T_N),
        depth_sd   = repeat(tier1_table.pred_sd, outer=T_N),
        grain_mu   = repeat(tier2_table.pred_mean, outer=T_N),
        grain_sd   = repeat(tier2_table.pred_sd, outer=T_N),
        temp_mu    = repeat(tier3_table.pred_mean, outer=T_N),
        temp_sd    = repeat(tier3_table.pred_sd, outer=T_N),
        pc1_mu     = repeat(tier4c_table.pc1_mean, outer=T_N),
        pc1_sd     = repeat(tier4c_table.pc1_sd, outer=T_N),
        pc2_mu     = repeat(tier4c_table.pc2_mean, outer=T_N),
        pc2_sd     = repeat(tier4c_table.pc2_sd, outer=T_N)
    )

    density_results_df = DataFrame(
        unit_id = grid_master.s_idx,
        year    = minimum(df_exp.year) .+ (grid_master.year_idx .- 1)
    )

    spp_cols = [:Gadus_morhua, :Melanogrammus_aeglefinus, :Pollachius_virens, :Sebastes_fasciatus]
    for sp in spp_cols
        if hasproperty(df_exp, sp)
            grid_master[!, sp] = fill(0.0, nrow(grid_master))
        end
    end

    models_pos = Dict{Symbol, Any}()
    chains_pos = Dict{Symbol, Any}()

    for st in stages_unique
        @info "  Fitting positive catch model for stage: :$st (Ordination: $(opts.ordination_mode))..."
        df_pos = filter(r -> r.stage_id == st && r.biomass_kg > 0.0, df_exp)

        # Attach stage-specific HSI from Tier 4D
        hsi_col = Symbol("hsi_", st, "_mean")
        hsi_sd_col = Symbol("hsi_", st, "_sd")
        if hasproperty(tier4d_table, hsi_col)
            grid_master[!, :hsi_stage_mu] = tier4d_table[!, hsi_col]
            grid_master[!, :hsi_stage_sd] = tier4d_table[!, hsi_sd_col]
            df_pos[!, :hsi_stage_mu]      = [tier4d_table[i, hsi_col] for i in df_pos.s_idx]
            df_pos[!, :hsi_stage_sd]      = [tier4d_table[i, hsi_sd_col] for i in df_pos.s_idx]
        else
            grid_master[!, :hsi_stage_mu] = fill(0.5, nrow(grid_master))
            grid_master[!, :hsi_stage_sd] = fill(0.1, nrow(grid_master))
            df_pos[!, :hsi_stage_mu]      = fill(0.5, nrow(df_pos))
            df_pos[!, :hsi_stage_sd]      = fill(0.1, nrow(df_pos))
        end

        m_pos = if opts.ordination_mode == :joint_eigen
            @tbm(
                likelihood(biomass_kg, family=gamma, log_offsets=log_effort) ~
                    intercept() +
                    fixed(hsi_stage_mu, error_sd=:hsi_stage_sd) +
                    fixed(depth_mu, error_sd=:depth_sd) +
                    fixed(temp_mu, error_sd=:temp_sd) +
                    eigen(Gadus_morhua, Melanogrammus_aeglefinus, Pollachius_virens, Sebastes_fasciatus, n_factors=2) +
                    (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
                df_pos, W=au_master.W, verbose=false
            )
        else
            @tbm(
                likelihood(biomass_kg, family=gamma, log_offsets=log_effort) ~
                    intercept() +
                    fixed(hsi_stage_mu, error_sd=:hsi_stage_sd) +
                    fixed(depth_mu, error_sd=:depth_sd) +
                    fixed(temp_mu, error_sd=:temp_sd) +
                    fixed(pc1_mu, error_sd=:pc1_sd) +
                    fixed(pc2_mu, error_sd=:pc2_sd) +
                    (random(s_idx, model=bym2) ⊗ random(year_idx, model=ar1)),
                df_pos, W=au_master.W, verbose=false
            )
        end

        chn_pos = _sample(m_pos, NUTS(40, 0.65), 80; progress=false)
        res_pos = model_results_comprehensive(m_pos, chn_pos; au=au_master)


        pred_pos = TethysBayes.predict(m_pos, chn_pos, grid_master)
        mu_pos = pred_pos.predictions_denoised.mean

        # Hurdle expected stage density = p(present) × μ(positive)
        p_enc = tier4d_table[!, Symbol("hsi_", st, "_mean")]
        expected_dens = p_enc .* mu_pos

        density_results_df[!, Symbol("dens_", st, "_mean")] = expected_dens
        density_results_df[!, Symbol("pos_mu_", st)]        = mu_pos


        models_pos[st] = m_pos
        chains_pos[st] = chn_pos
        save_TethysBayes_bundle(joinpath(db_dir, "tier5d_dens_$(st)"), m_pos, chn_pos, res_pos; au=au_master)
    end

    # Aggregate total biomass density across all stages (emergent sum)
    stage_dens_cols = [Symbol("dens_", st, "_mean") for st in stages_unique]
    density_results_df[!, :total_emergent_biomass] = sum(
        eachcol(density_results_df[:, stage_dens_cols])
    )

    write_tier_table!(db_path, density_results_df, "tier5d_demographic_density_predictions")

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Tier 5D: Stage Density & Hurdle",
        status = "COMPLETED",
        update_frequency = "Annual Survey",
        traits = traits,
        upstream_deps = ["tier1_depth", "tier2_substrate", "tier3_temp", "tier4c_community", "tier4d_demographic_hsi"],
        bundle_path = bundle_path,
        table_name = "tier5d_demographic_density_predictions"
    )
    @info "  ✓ Tier 5D Demographic Density completed & registered in manifest."
    return (models=models_pos, chains=chains_pos, table=density_results_df, status=:COMPLETED)
end

"""
    run_step6d_demographic_movement(au_master, tier4d_table, domain_km, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> NamedTuple

Step 7D: Computes life-stage-specific transition probability kernels Γ_c(∇HSI_c, W)
and simulates distinct ontogenetic migration trajectories.
"""
function run_step6d_demographic_movement(
    au_master,
    tier4d_table::DataFrame,
    domain_km::Float64,
    db_dir::AbstractString,
    db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::NamedTuple
    tier_id = "step6d_demographic_movement"
    traits = Dict{Symbol, Any}(:rows => au_master.n_units, :hash => "demog_movement_$(opts.movement_mode)")
    bundle_path = joinpath(db_dir, "step6d_demographic_movement")

    if !force && is_tier_up_to_date(db_path, tier_id, traits, ["tier4d_demographic_hsi", "tier5d_demographic_density"])
        @info "  ✓ [CACHE] Step 6D Demographic Movement is up-to-date."
        return (kernels=Dict{Symbol, Matrix{Float64}}(), status=:CACHED)
    end

    @info "Step 7D: Computing Stage-Specific ADR Transition Kernels & Ontogenetic Paths..."

    df_hsi_latest = filter(r -> r.year == 2024, tier4d_table)
    hsi_cols = filter(c -> startswith(string(c), "hsi_") && endswith(string(c), "_mean"), names(df_hsi_latest))

    kernels = Dict{Symbol, Matrix{Float64}}()
    trajectories = Dict{Symbol, Any}()

    for col in hsi_cols
        st_name = Symbol(replace(replace(string(col), "hsi_" => ""), "_mean" => ""))
        hsi_vec = df_hsi_latest[!, col]

        # Construct directional stage-specific Markov transition kernel
        gamma_st = compute_suitability_transition_kernel(
            hsi_vec, au_master.W;
            sensitivity = 1.5,
            diffusion_weight = (st_name == :commercial_male ? 0.4 : 0.15),
            relationship = :exponential
        )
        kernels[st_name] = gamma_st

        # Forward simulate 15 individual dispersal paths per stage
        start_units = rand(1:au_master.n_units, 15)
        sim_paths = simulate_posterior_trajectories(
            gamma_st, start_units, 20, au_master;
            rho_persistence = (st_name == :commercial_male ? 1.4 : 0.8)
        )
        trajectories[st_name] = sim_paths
    end

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Step 6D: Stage-Specific Dispersal",
        status = "COMPLETED",
        update_frequency = "Post-Assessment",
        traits = traits,
        upstream_deps = ["tier4d_demographic_hsi", "tier5d_demographic_density"],
        bundle_path = bundle_path,
        table_name = "demographic_movement_kernels"
    )
    @info "  ✓ Step 6D Demographic Dispersal completed & registered in manifest."
    return (kernels=kernels, trajectories=trajectories, status=:COMPLETED)
end

"""
    run_step7d_demographic_harmonization(au_master, tier1_table, tier2_table, tier3_table, tier4c_table, tier4d_table, tier5d_table, db_dir, db_path; opts=DemographicPipelineOptions(), force=false) -> DataFrame

Step 8D: Assembles the Master Demographic Refugia Summary, queries stage-specific critical
habitats, and exports RFC 7946 GeoJSON GIS maps.
"""
function run_step7d_demographic_harmonization(
    au_master,
    tier1_table::DataFrame,
    tier2_table::DataFrame,
    tier3_table::DataFrame,
    tier4c_table::DataFrame,
    tier4d_table::DataFrame,
    tier5d_table::DataFrame,
    db_dir::AbstractString,
    db_path::AbstractString;
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false
)::DataFrame
    tier_id = "step7d_demographic_harmonization"
    traits = Dict{Symbol, Any}(:rows => au_master.n_units, :hash => "demog_harmonization")
    bundle_path = joinpath(db_dir, "step7d_demographic_harmonization")

    @info "Step 8D: Building Master Demographic Summary & Zero-Copy DuckDB SQL Queries..."

    df_master = DataFrame(
        unit_id    = 1:au_master.n_units,
        s_x        = [c[1] for c in au_master.centroids],
        s_y        = [c[2] for c in au_master.centroids],
        depth_mean = tier1_table.pred_mean,
        slope_mean = tier1_table.slope_mean,
        grain_mean = tier2_table.pred_mean,
        temp_mean  = tier3_table.pred_mean,
        pc1_mean   = tier4c_table.pc1_mean,
        pc2_mean   = tier4c_table.pc2_mean
    )

    # Join stage HSI and Density columns (take single year slice 2024)
    df_hsi_slice = filter(r -> r.year == 2024, tier4d_table)
    df_dens_slice = filter(r -> r.year == 2024, tier5d_table)

    for col in names(df_hsi_slice)
        if startswith(string(col), "hsi_")
            df_master[!, col] = df_hsi_slice[!, col]
        end
    end
    for col in names(df_dens_slice)
        if startswith(string(col), "dens_") || col == :total_emergent_biomass
            df_master[!, col] = df_dens_slice[!, col]
        end
    end

    write_tier_table!(db_path, df_master, "master_demographic_summary")

    # SQL Query: Identify Essential Juvenile Nursery Refugia (High Juvenile HSI & Shallow Depth)
    sql_nursery = if hasproperty(df_master, :hsi_immature_juvenile_mean)
        """
        SELECT unit_id, s_x, s_y,
               hsi_immature_juvenile_mean AS juvenile_hsi,
               dens_immature_juvenile_mean AS juvenile_density,
               depth_mean, temp_mean, grain_mean, pc1_mean, pc2_mean
        FROM master_demographic_summary
        WHERE hsi_immature_juvenile_mean >= 0.60
        ORDER BY juvenile_density DESC
        """
    else
        """
        SELECT unit_id, s_x, s_y, total_emergent_biomass, depth_mean, temp_mean
        FROM master_demographic_summary
        ORDER BY total_emergent_biomass DESC
        """
    end

    df_nursery = query_duckdb(db_path, sql_nursery)
    println("\nTop 5 Identified Juvenile Nursery / Spawning Refugia (DuckDB SQL):")
    display(first(df_nursery, 5))

    update_manifest_entry!(
        db_path;
        tier_id = tier_id,
        tier_name = "Step 7D: Demographic Refugia & GIS",
        status = "COMPLETED",
        update_frequency = "Final Integration",
        traits = traits,
        upstream_deps = ["tier5d_demographic_density"],
        bundle_path = bundle_path,
        table_name = "master_demographic_summary"
    )
    return df_nursery
end

# ==============================================================================
# SECTION 4: WORKFLOW ORCHESTRATOR & SINGLE-CALL SEGMENT UPDATER
# ==============================================================================

"""
    update_demographic_segment!(segment::Symbol; force=true, opts=DemographicPipelineOptions(), db_dir=joinpath(@__DIR__, "project_db"), kwargs...)

Directly fits or updates a single specific demographic segment. Automatically loads
upstream dependencies from DuckDB / JLD2 caches or fits missing prerequisites on demand.

# Supported Segments
- `:tier1`, `:depth`, `:bathymetry`
- `:tier2`, `:substrate`, `:grain`
- `:tier3`, `:temp`, `:temperature`
- `:tier4c`, `:community`, `:pca`, `:ordination`
- `:tier4d`, `:hsi`, `:encounter`, `:demographic_hsi`
- `:tier5d`, `:density`, `:hurdle`, `:demographic_density`
- `:step6d`, `:movement`, `:dispersal`, `:migration`
- `:step7d`, `:harmonization`, `:summary`, `:refugia`
"""
function update_demographic_segment!(
    segment::Symbol;
    force::Bool = true,
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    db_dir::AbstractString = joinpath(@__DIR__, "project_db"),
    db_path::AbstractString = joinpath(db_dir, "demographic_hierarchical_project.duckdb"),
    kwargs...
)
    seg_clean = Symbol(lowercase(string(segment)))
    @info "Executing single demographic segment update: :$seg_clean (scheme=$(opts.class_scheme))..."

    mock_data = TethysBayes_data(type="hierarchical", seed=123)
    df_bathy = mock_data.bathymetry
    df_sub   = mock_data.substrate
    df_temp  = mock_data.temperature
    df_comm  = mock_data.species_composition
    df_crab  = mock_data.snow_crab
    df_bio   = mock_data.individuals
    domain_km = 600.0

    df_expanded = expand_tows_to_demographic_classes(
        df_crab, df_bio;
        scheme = opts.class_scheme,
        breaks = opts.tier6_bin_breaks
    )

    all_sx = vcat(df_bathy.s_x, df_sub.s_x, df_temp.s_x, df_comm.s_x, df_crab.s_x)
    all_sy = vcat(df_bathy.s_y, df_sub.s_y, df_temp.s_y, df_comm.s_y, df_crab.s_y)
    master_hull = expand_hull(all_sx, all_sy, 2.0)

    au_sub = assign_spatial_units(df_sub.s_x, df_sub.s_y; area_method=:cvt, target_units=25, min_area=10.0, geom_hull=master_hull, merge_small_polygons=true)
    df_sub.s_idx = au_sub.assignments

    au_temp = assign_spatial_units(df_temp.s_x, df_temp.s_y; area_method=:cvt, target_units=40, min_area=8.0, geom_hull=master_hull, merge_small_polygons=true)
    df_temp.s_idx = au_temp.assignments

    au_comm = assign_spatial_units(df_comm.s_x, df_comm.s_y; area_method=:cvt, target_units=50, min_area=6.0, geom_hull=master_hull, merge_small_polygons=true)
    df_comm.s_idx = au_comm.assignments

    au_master = assign_spatial_units(all_sx, all_sy; area_method=:cvt, target_units=80, min_area=5.0, geom_hull=master_hull, merge_small_polygons=true)

    init_pipeline_manifest!(db_path)

    # 1. Tier 1
    if seg_clean in [:tier1, :depth, :bathymetry]
        return run_tier1_depth(df_bathy, au_sub, au_temp, au_master, db_dir, db_path; opts=opts, force=force)
    end
    t1 = run_tier1_depth(df_bathy, au_sub, au_temp, au_master, db_dir, db_path; opts=opts, force=false)

    # 2. Tier 2
    if seg_clean in [:tier2, :substrate, :grain]
        return run_tier2_substrate(df_sub, au_sub, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=force)
    end
    t2 = run_tier2_substrate(df_sub, au_sub, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=false)

    # 3. Tier 3
    if seg_clean in [:tier3, :temp, :temperature]
        return run_tier3_temperature(df_temp, au_temp, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=force)
    end
    t3 = run_tier3_temperature(df_temp, au_temp, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=false)

    # 4. Tier 4C (Community Ordination via Householder Eigen)
    if seg_clean in [:tier4c, :community, :pca, :ordination]
        return run_tier4c_community(df_comm, au_comm, au_master, t1.table, t2.table, t3.table, db_dir, db_path; opts=opts, force=force)
    end
    t4c = run_tier4c_community(df_comm, au_comm, au_master, t1.table, t2.table, t3.table, db_dir, db_path; opts=opts, force=false)

    # 5. Tier 4D
    if seg_clean in [:tier4d, :hsi, :encounter, :demographic_hsi]
        return run_tier4d_demographic_hsi(df_expanded, au_master, t1.table, t2.table, t3.table, t4c.table, db_dir, db_path; opts=opts, force=force)
    end
    t4d = run_tier4d_demographic_hsi(df_expanded, au_master, t1.table, t2.table, t3.table, t4c.table, db_dir, db_path; opts=opts, force=false)

    # 6. Tier 5D
    if seg_clean in [:tier5d, :density, :hurdle, :demographic_density]
        return run_tier5d_demographic_density(df_expanded, au_master, t1.table, t2.table, t3.table, t4c.table, t4d.table, db_dir, db_path; opts=opts, force=force)
    end
    t5d = run_tier5d_demographic_density(df_expanded, au_master, t1.table, t2.table, t3.table, t4c.table, t4d.table, db_dir, db_path; opts=opts, force=false)

    # 7. Step 6D
    if seg_clean in [:step6d, :movement, :dispersal, :migration]
        return run_step6d_demographic_movement(au_master, t4d.table, domain_km, db_dir, db_path; opts=opts, force=force)
    end

    # 8. Step 7D
    if seg_clean in [:step7d, :harmonization, :summary, :refugia]
        return run_step7d_demographic_harmonization(au_master, t1.table, t2.table, t3.table, t4c.table, t4d.table, t5d.table, db_dir, db_path; opts=opts, force=force)
    end

    error("Unknown demographic segment: :$segment.")
end

"""
    run_hierarchical_demographic(; steps=[:all], opts=DemographicPipelineOptions(), force=false, clean=false)

Executes the full class-expanded demographic workflow (Approach B) including multi-species Householder ordination.
"""
function run_hierarchical_demographic(;
    steps = [:all],
    opts::DemographicPipelineOptions = DemographicPipelineOptions(),
    force::Bool = false,
    clean::Bool = false
)
    println("="^80)
    println("STARTING DEMOGRAPHIC CLASS-EXPANDED WORKFLOW (APPROACH B: $(opts.class_scheme))")
    println("="^80)

    mock_data = TethysBayes_data(type="hierarchical", seed=123)
    df_bathy = mock_data.bathymetry
    df_sub   = mock_data.substrate
    df_temp  = mock_data.temperature
    df_comm  = mock_data.species_composition
    df_crab  = mock_data.snow_crab
    df_bio   = mock_data.individuals
    domain_km = 600.0

    @info "Expanding tow records to demographic life stages ($(opts.class_scheme))..."
    df_expanded = expand_tows_to_demographic_classes(
        df_crab, df_bio;
        scheme = opts.class_scheme,
        breaks = opts.tier6_bin_breaks
    )

    # Pivot and Hellinger-transform species composition to hauls for joint eigen modeling
    species_names = Symbol.(unique(df_comm.species))
    haul_ids = unique(df_comm.haul_id)
    haul_spp_df = DataFrame(haul_id = haul_ids)
    for sp in species_names
        haul_spp_df[!, sp] = zeros(Float64, length(haul_ids))
    end
    h_dict = Dict(h => i for (i, h) in enumerate(haul_ids))

    for row in eachrow(df_comm)
        hi = h_dict[row.haul_id]
        sp = Symbol(row.species)
        haul_spp_df[hi, sp] = row.biomass_kg / (row.swept_area_km2 + 1e-4)
    end

    spp_mat = Matrix(haul_spp_df[:, species_names])
    row_sums = sum(spp_mat, dims=2)
    row_sums[row_sums .== 0.0] .= 1.0
    spp_hellinger = sqrt.(spp_mat ./ row_sums)
    for (j, sp) in enumerate(species_names)
        haul_spp_df[!, sp] = spp_hellinger[:, j]
    end

    df_expanded[!, :haul_id] = df_expanded.tow_id
    df_expanded = DataFrames.leftjoin(df_expanded, haul_spp_df, on=:haul_id)
    for sp in species_names
        df_expanded[!, sp] = [ismissing(x) || isnothing(x) || !isfinite(x) ? 0.0 : Float64(x) for x in df_expanded[!, sp]]
    end


    all_sx = vcat(df_bathy.s_x, df_sub.s_x, df_temp.s_x, df_comm.s_x, df_crab.s_x)
    all_sy = vcat(df_bathy.s_y, df_sub.s_y, df_temp.s_y, df_comm.s_y, df_crab.s_y)
    master_hull = expand_hull(all_sx, all_sy, 2.0)

    au_sub = assign_spatial_units(df_sub.s_x, df_sub.s_y; area_method=:cvt, target_units=25, min_area=10.0, geom_hull=master_hull, merge_small_polygons=true)
    df_sub.s_idx = au_sub.assignments

    au_temp = assign_spatial_units(df_temp.s_x, df_temp.s_y; area_method=:cvt, target_units=40, min_area=8.0, geom_hull=master_hull, merge_small_polygons=true)
    df_temp.s_idx = au_temp.assignments

    au_comm = assign_spatial_units(df_comm.s_x, df_comm.s_y; area_method=:cvt, target_units=50, min_area=6.0, geom_hull=master_hull, merge_small_polygons=true)
    df_comm.s_idx = au_comm.assignments

    au_master = assign_spatial_units(all_sx, all_sy; area_method=:cvt, target_units=80, min_area=5.0, geom_hull=master_hull, merge_small_polygons=true)

    db_dir = joinpath(@__DIR__, "project_db")
    mkpath(db_dir)
    db_path = joinpath(db_dir, "demographic_hierarchical_project.duckdb")

    if clean && isfile(db_path)
        rm(db_path; force=true)
    end
    init_pipeline_manifest!(db_path)

    check_pipeline_status(db_path, get_demographic_pipeline_spec(mock_data; opts=opts); verbose=true)

    # 1. Tier 1
    t1_force = force || (:tier1 in steps) || (:depth in steps)
    t1 = run_tier1_depth(df_bathy, au_sub, au_temp, au_master, db_dir, db_path; opts=opts, force=t1_force)

    # 2. Tier 2
    t2_force = force || (:tier2 in steps) || (:substrate in steps)
    t2 = run_tier2_substrate(df_sub, au_sub, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=t2_force)

    # 3. Tier 3
    t3_force = force || (:tier3 in steps) || (:temp in steps)
    t3 = run_tier3_temperature(df_temp, au_temp, au_master, t1.model, t1.chain, db_dir, db_path; opts=opts, force=t3_force)

    # 4. Tier 4C (Community Ordination via Householder Eigen)
    t4c_force = force || (:tier4c in steps) || (:community in steps) || (:pca in steps) || (:ordination in steps)
    t4c = run_tier4c_community(df_comm, au_comm, au_master, t1.table, t2.table, t3.table, db_dir, db_path; opts=opts, force=t4c_force)

    # 5. Tier 4D
    t4d_force = force || (:tier4d in steps) || (:hsi in steps)
    t4d = run_tier4d_demographic_hsi(df_expanded, au_master, t1.table, t2.table, t3.table, t4c.table, db_dir, db_path; opts=opts, force=t4d_force)

    # 6. Tier 5D
    t5d_force = force || (:tier5d in steps) || (:density in steps)
    t5d = run_tier5d_demographic_density(df_expanded, au_master, t1.table, t2.table, t3.table, t4c.table, t4d.table, db_dir, db_path; opts=opts, force=t5d_force)

    # 7. Step 6D
    s6d_force = force || (:step6d in steps) || (:movement in steps)
    run_step6d_demographic_movement(au_master, t4d.table, domain_km, db_dir, db_path; opts=opts, force=s6d_force)

    # 8. Step 7D
    s7d_force = force || (:step7d in steps) || (:harmonization in steps)
    run_step7d_demographic_harmonization(au_master, t1.table, t2.table, t3.table, t4c.table, t4d.table, t5d.table, db_dir, db_path; opts=opts, force=s7d_force)

    println("\n" * "="^80)
    println("DEMOGRAPHIC CLASS-EXPANDED WORKFLOW (APPROACH B) COMPLETED SUCCESSFULLY!")
    println("="^80)
end

# ------------------------------------------------------------------------------
# CLI DISPATCH
# ------------------------------------------------------------------------------
function main(args::Vector{String})
    scheme = ("--fine-bins" in args || "--fine" in args) ? :fine_bins : :functional_stages
    ord_mode = ("--two-step" in args) ? :two_step_householder : :joint_eigen
    opts = DemographicPipelineOptions(
        class_scheme = scheme,
        ordination_mode = ord_mode
    )


    if "--status" in args
        db_dir = joinpath(@__DIR__, "project_db")
        db_path = joinpath(db_dir, "demographic_hierarchical_project.duckdb")
        mock_data = TethysBayes_data(type="hierarchical", seed=123)
        check_pipeline_status(db_path, get_demographic_pipeline_spec(mock_data; opts=opts); verbose=true)
    else
        force_flag = "--force" in args
        clean_flag = "--clean" in args
        target_steps = Symbol[:all]

        for arg in args
            if startswith(arg, "--tier=") || startswith(arg, "--step=")
                val = split(arg, "=")[2]
                target_steps = [Symbol(val)]
            elseif arg in ["--tier1", "--depth"]
                target_steps = [:tier1]
            elseif arg in ["--tier2", "--sub"]
                target_steps = [:tier2]
            elseif arg in ["--tier3", "--temp"]
                target_steps = [:tier3]
            elseif arg in ["--tier4c", "--community", "--pca", "--ordination"]
                target_steps = [:tier4c]
            elseif arg in ["--tier4d", "--hsi"]
                target_steps = [:tier4d]
            elseif arg in ["--tier5d", "--density"]
                target_steps = [:tier5d]
            elseif arg in ["--movement", "--step6d"]
                target_steps = [:step6d]
            elseif arg in ["--harmonization", "--step7d"]
                target_steps = [:step7d]
            end
        end

        run_hierarchical_demographic(; steps=target_steps, opts=opts, force=force_flag, clean=clean_flag)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
