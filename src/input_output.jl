"""
    input_output.jl

Serialization, analytical persistence (JLD2 and DuckDB), GIS export, and model ensembling
for Bayesian Spatio-Temporal Models (TethysBayes).

Version: v1.0.0
"""

using Dates, Printf

# ==============================================================================
# PREDICTIVE-SD RECOVERY HELPERS
#
# The DuckDB predictions table persists a per-observation posterior predictive SD.
# Databases written before that column existed (and any result object whose
# `predictions.denoised` carries no `std`) only have credible-interval bounds, from
# which a SD can be recovered by inverting the symmetric 95% width. The helpers below
# are the single implementation of that recovery, shared by the writer and the reader
# so the two can never disagree about which SD was actually stored.
# ==============================================================================

# 97.5th standard-normal quantile, i.e. the ±z half-width of a two-sided 95%
# interval. Defined once so the writer and reader cannot drift to different levels.
const Z975 = 1.959963984540054

# Midpoint drift, as a fraction of the interval HALF-WIDTH, above which a credible
# interval is considered too asymmetric for the width inversion to be trustworthy.
#
# Normalised by the half-width rather than by `|mean|`. The old denominator was the mean,
# which is the wrong yardstick whenever the mean is small next to the interval -- the
# ordinary situation for a near-zero rate with a wide credible interval, or one that
# straddles zero. Measured: for a prediction of 0.001 in an interval of half-width 1.0
# whose midpoint is 0.03 away (a 3% drift, a perfectly good interval), the old measure
# reported 30.0 -- 600x this tolerance -- purely because the mean was small.
#
# It is worth being precise that the old measure is NOT scale-dependent in the ordinary
# sense: rescaling the problem leaves `|mid-mu|/|mu|` unchanged. The defect is a wrong
# denominator, which only bites in the near-zero-mean case.
#
# The half-width form is directly interpretable: 0 is a perfectly centred interval, 1
# means the mean sits one half-width from the midpoint.
const INTERVAL_ASYMMETRY_TOL = 0.05

_as_float(v) = v isa Real ? Float64(v) : NaN

"""
    _interval_asymmetry(lower, upper, mean) -> Float64

Largest displacement of an interval's midpoint from the corresponding prediction mean,
**as a fraction of that interval's half-width**. A large value means the recovered SD is
biased, because `sd = (upper - lower) / (2 Z975)` assumes a symmetric interval.

Returns `0.0` when the bounds are absent, mismatched, or not all finite, so an unusable
interval is not reported as a maximally asymmetric one.
"""
function _interval_asymmetry(lower, upper, mean::AbstractVector)
    n = length(mean)
    if lower === nothing || upper === nothing ||
            length(lower) != n || length(upper) != n
        return 0.0
    end
    worst = 0.0
    @inbounds for i in 1:n
        lo = _as_float(lower[i])
        hi = _as_float(upper[i])
        mu = _as_float(mean[i])
        if isfinite(lo) && isfinite(hi) && isfinite(mu) && hi > lo
            mid = (lo + hi) / 2
            half = (hi - lo) / 2
            rel = abs(mid - mu) / max(half, eps(Float64))
            rel > worst && (worst = rel)
        end
    end
    return worst
end

"""
    _warn_if_interval_asymmetric(asym, context) -> Bool

Emit the interval-asymmetry diagnostic, or an informational note when the drift is within
tolerance. Returns `true` when it warned.

All three `_recover_pred_sd` call sites route through this, so a width-derived SD is
reported wherever it is produced rather than only in the BMA path -- previously only one of
the three sites checked, so a biased SD could enter the stored predictions table and the
bundle loader without a word. `context` names the source for the reader.
"""
function _warn_if_interval_asymmetric(asym::Real, context::AbstractString)
    if asym > INTERVAL_ASYMMETRY_TOL
        @warn "Predictive SD for $(context) was derived from a credible interval whose " *
              "midpoint is up to $(round(asym * 100; digits=1))% of a half-width from the " *
              "mean, so the symmetric ±$(Z975)σ width inversion is biased. Re-fit the " *
              "model to populate `pred_sd` before relying on `bma_pred_sd`."
        return true
    end
    @info "Predictive SD for $(context) was derived from the 95% credible interval " *
          "width (±$(Z975)σ). Interval midpoint drift is $(round(asym * 100; digits=1))% " *
          "of a half-width, within tolerance. Treat the resulting `bma_pred_sd` as " *
          "approximate."
    return false
end

"""
    _recover_pred_sd(sd, lower, upper, mean) -> (sd, source, asymmetry)

Resolve the per-observation predictive SD from whatever is actually available,
preferring the model's own `std` and falling back to the 95% interval width
`sd = (upper - lower) / (2 * Z975)`.

The choice is made per observation, so a single unusable entry cannot discard a
whole model's real predictive SD. Returns the resolved vector together with a
provenance tag:

- `:model`          every observation came from a genuine `std`.
- `:interval_width` at least one observation came from the width inversion.
- `:missing`        neither was usable; the returned vector is all `NaN`.

The third element is the worst interval asymmetry (see `_interval_asymmetry`), which is
only meaningful when `source === :interval_width`. It is returned rather than warned about
here so that each caller can name its own source, and so that a caller cannot forget to
check: destructuring into two variables still works, but the number is always available.

Non-numeric entries (`missing`, `nothing`) are treated as unusable rather than
raising, so a partially-populated column degrades instead of failing.
"""
function _recover_pred_sd(sd, lower, upper, mean::AbstractVector)
    n = length(mean)
    out = fill(NaN, n)
    n_from_sd = 0
    n_from_iv = 0

    if sd !== nothing && length(sd) == n
        @inbounds for i in 1:n
            v = _as_float(sd[i])
            if isfinite(v)
                out[i] = v
                n_from_sd += 1
            end
        end
    end

    if lower !== nothing && upper !== nothing &&
            length(lower) == n && length(upper) == n
        @inbounds for i in 1:n
            lo = _as_float(lower[i])
            hi = _as_float(upper[i])
            if !isfinite(out[i]) && isfinite(lo) && isfinite(hi) && hi > lo
                out[i] = (hi - lo) / (2 * Z975)
                n_from_iv += 1
            end
        end
    end

    source = if n_from_sd == n
        :model
    elseif n_from_iv > 0
        :interval_width
    else
        :missing
    end
    return out, source, _interval_asymmetry(lower, upper, mean)
end

# ==============================================================================
# SECTION 1: JLD2 MODEL STATE PERSISTENCE
# ==============================================================================

"""
    TethysBayes_SCHEMA_VERSION

Version of the on-disk bundle layout written by [`save_TethysBayes_model`](@ref).

This is deliberately separate from the package version. It changes only when the shape
of a saved bundle changes, so a reader can tell "written by an older TethysBayes" (same layout,
possibly missing fields) from "written by a newer TethysBayes whose layout I do not understand".

A bundle records the version that wrote it; [`load_TethysBayes_model`](@ref) refuses a bundle
newer than this, and accepts an older one with a warning, rather than failing later with
an unrelated error about a missing field.
"""
const TethysBayes_SCHEMA_VERSION = 2

"""
    save_TethysBayes_model(filepath::AbstractString, model::DynamicPPL.Model; 
                    chain=nothing, au=nothing, metadata::Dict=Dict(), compress::Bool=true)

Saves the complete state of a `TethysBayes` model `m` and optional posterior `chain` to a JLD2 file.

# Arguments
- `filepath::AbstractString`: Path to output file (should end with `.jld2` or `.TethysBayes`).
- `model::DynamicPPL.Model`: Instantiated `TethysBayes` Turing model.
- `chain`: (Optional) MCMC chain object (`FlexiChain` or `MCMCChains.Chains`).
- `au`: (Optional) Areal units NamedTuple from `assign_spatial_units`.
- `metadata::Dict`: (Optional) User-defined metadata dictionary (e.g. project name, notes, author).
- `compress::Bool`: Whether to compress data on disk (default: `true`).

# Example
```julia
save_TethysBayes_model("output/my_model.jld2", m; chain=chn, au=st_data.au_spatial)
```
"""
function save_TethysBayes_model(
    filepath::AbstractString, 
    model::DynamicPPL.Model; 
    chain=nothing, 
    au=nothing, 
    metadata::Dict=Dict(), 
    compress::Bool=true
)
    # Ensure parent directory exists
    dir = dirname(filepath)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    if !endswith(filepath, ".jld2") && !endswith(filepath, ".TethysBayes")
        filepath = filepath * ".jld2"
    end

    M = model.args.M
    spec_registry = model.args.spec_registry

    # Package clean serializable configuration without redundant handles
    meta_info = merge(Dict(
        "created_at" => string(now()),
        "schema_version" => TethysBayes_SCHEMA_VERSION,
        "formula" => string(get(M, :formula, "")),
        "model_arch" => string(get(M, :model_arch, "univariate")),
        "family" => string(get(M, :family, :gaussian)),
        "n_obs" => size(get(M, :data, DataFrame()), 1),
        "has_chain" => !isnothing(chain),
        "has_au" => !isnothing(au)
    ), Dict(string(k) => v for (k, v) in metadata))

    # Strip runtime function handles from spec_registry for clean JLD2 serialization.
    # What was dropped is recorded rather than discarded silently: a function that turns
    # out to have mattered is then visible in the bundle instead of showing up later as
    # an unexplained missing field.
    clean_spec_reg = Dict{Symbol, Any}()
    dropped = String[]
    for (k, v) in pairs(spec_registry)
        if v isa Function
            push!(dropped, string(k))
            continue
        elseif v isa Dict
            sub_dict = Dict{Symbol, Any}()
            for (sk, sv) in pairs(v)
                if sv isa Function
                    push!(dropped, "$(k).$(sk)")
                else
                    sub_dict[sk] = sv
                end
            end
            clean_spec_reg[k] = sub_dict
        else
            clean_spec_reg[k] = v
        end
    end
    meta_info["dropped_spec_registry_keys"] = join(sort(dropped), ", ")

    # Extract W matrix if present
    W_mat = haskey(M, :W) ? M.W : (haskey(M, :technical) && haskey(M.technical, :W) ?
      M.technical[:W] : nothing)

    JLD2.jldsave(filepath; compress=compress,
        formula = string(M.formula),
        data = M.data,
        generated_model_code = get(M, :generated_model_code, ""),
        W = W_mat,
        spec_registry = clean_spec_reg,
        chain = chain,
        au = au,
        schema_version = TethysBayes_SCHEMA_VERSION,
        metadata = meta_info
    )

    @info "TethysBayes model successfully saved to '$filepath' (schema v$TethysBayes_SCHEMA_VERSION)."
    return filepath
end

"""
    load_TethysBayes_model(filepath::AbstractString; calling_module::Module=Main)

Loads a saved `TethysBayes` model from a JLD2 file and re-instantiates a live, callable
Turing `@model` object ready for sampling, prediction, or diagnostic post-processing.

# Returns
A `NamedTuple` containing:
- `model`: Instantiated, callable `DynamicPPL.Model`.
- `chain`: MCMC chain object (or `nothing` if not saved).
- `au`: Areal units object (or `nothing`).
- `metadata`: Saved metadata dictionary.

# Example
```julia
bundle = load_TethysBayes_model("output/my_model.jld2")
m = bundle.model
chn = bundle.chain
```
"""
function load_TethysBayes_model(filepath::AbstractString; calling_module::Module=Main)
    if !isfile(filepath)
        if isfile(filepath * ".jld2")
            filepath = filepath * ".jld2"
        else
            error("TethysBayes model file not found: '$filepath'")
        end
    end

    f = JLD2.jldopen(filepath, "r")
    formula_str = f["formula"]
    data_df = f["data"]
    W_mat = haskey(f, "W") ? f["W"] : nothing
    chain = haskey(f, "chain") ? f["chain"] : nothing
    au = haskey(f, "au") ? f["au"] : nothing
    meta = haskey(f, "metadata") ? f["metadata"] : Dict()
    # Bundles written before schema versioning (v1) carry no top-level marker; they are
    # layout-compatible, so treat a missing marker as v1 rather than as an error.
    schema = haskey(f, "schema_version") ? Int(f["schema_version"]) :
        Int(get(meta, "schema_version", 1))
    close(f)

    if schema > TethysBayes_SCHEMA_VERSION
        error(
            "'$filepath' was written with TethysBayes bundle schema v$schema, but this version " *
            "of TethysBayes only understands up to v$TethysBayes_SCHEMA_VERSION. Upgrade TethysBayes to read it.")
    elseif schema < TethysBayes_SCHEMA_VERSION
        @warn "Bundle '$filepath' uses schema v$schema, older than the current v$TethysBayes_SCHEMA_VERSION. Loading may lose fields added since."
    end
    dropped = get(meta, "dropped_spec_registry_keys", "")
    if !isempty(dropped)
        @warn "Bundle '$filepath' omits non-serializable spec_registry entries: $dropped"
    end

    # Re-instantiate the live Turing Model using TethysBayes_core
    kwargs = Dict{Symbol, Any}()
    if !isnothing(W_mat)
        kwargs[:W] = W_mat
    end
    if !isnothing(au)
        kwargs[:au] = au
    end
    kwargs[:verbose] = false

    # Reconstruct live model
    live_model = TethysBayes_core(formula_str, data_df, calling_module; kwargs...)

    @info "TethysBayes model successfully loaded and instantiated from '$filepath' (schema v$schema)."
    return (
        model = live_model,
        chain = chain,
        au = au,
        metadata = meta
    )
end

# ==============================================================================
# SECTION 2: DUCKDB RESULTS & POSTERIOR PERSISTENCE
# ==============================================================================

"""
    save_TethysBayes_results(duckdb_path::AbstractString, res::NamedTuple; 
                      model=nothing, chain=nothing, au=nothing,
                      table_prefix::String="", overwrite::Bool=true)

Persists post-processing results (`res` from `model_results_comprehensive`) into an analytical
DuckDB database without data redundancy.

Stores normalized relational tables:
- `<prefix>model_metadata`: Model formula, family, creation timestamp, observation count.
- `<prefix>metrics`: Summary metrics (RMSE, Pearson r, ESS, Rhat, WAIC, time).
- `<prefix>parameter_stats`: Parameter posterior means, medians, stds, credible intervals.
- `<prefix>predictions`: Denoised observation-level predictions, intervals, and residuals.
- `<prefix>spatial_geometries`: (Optional) Polygon boundaries in standard WKT format.
- `<prefix>plot_data_<key>`: Tidy dataframes for all diagnostic plots.
- `<prefix>posterior_samples`: (Optional) Raw posterior parameter draws if `chain` is passed.

# Example
```julia
save_TethysBayes_results("output/results.duckdb", res; model=m, chain=chn, au=data.au)
```
"""
function save_TethysBayes_results(
    duckdb_path::AbstractString, 
    res::NamedTuple; 
    model=nothing, 
    chain=nothing, 
    au=nothing, 
    table_prefix::String="", 
    overwrite::Bool=true
)
    # Ensure directory exists
    dir = dirname(duckdb_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    if !endswith(duckdb_path, ".duckdb") && !endswith(duckdb_path, ".db")
        duckdb_path = duckdb_path * ".duckdb"
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)

    pfx = isempty(table_prefix) ? "" : (endswith(table_prefix, "_") ? table_prefix :
      table_prefix * "_")

    try
        # 1. Save Model Metadata Table
        formula_str = !isnothing(model) && hasproperty(model, :args) &&
          hasproperty(model.args, :M) ? string(model.args.M.formula) : "unknown"
        family_str = !isnothing(model) && hasproperty(model, :args) &&
          hasproperty(model.args, :M) ? string(get(model.args.M, :family, :gaussian)) :
          "unknown"
        
        df_meta = DataFrame(
            property = ["saved_at", "formula", "family", "TethysBayes_version"],
            value = [string(now()), formula_str, family_str, "1.0.0"]
        )
        _write_df_to_duckdb(con, df_meta, "$(pfx)model_metadata", overwrite)

        # 2. Save Metrics Table
        if hasproperty(res, :metrics)
            m_keys = String[]
            m_vals = Float64[]
            for (k, v) in pairs(res.metrics)
                if v isa Number
                    push!(m_keys, string(k))
                    push!(m_vals, Float64(v))
                elseif v isa AbstractVector{<:Number}
                    for (idx, sub_v) in enumerate(v)
                        push!(m_keys, "$(k)_$idx")
                        push!(m_vals, Float64(sub_v))
                    end
                end
            end
            df_metrics = DataFrame(metric = m_keys, value = m_vals)
            _write_df_to_duckdb(con, df_metrics, "$(pfx)metrics", overwrite)
        end

        # 3. Save Parameter Posteriors Table
        df_params = if hasproperty(res, :parameters) && res.parameters isa DataFrame &&
          !isempty(res.parameters)
            res.parameters
        elseif !isnothing(chain)
            try
                _compute_direct_parameter_summary(chain, model)
            catch
                DataFrame()
            end
        else
            DataFrame()
        end
        if !isempty(df_params)
            _write_df_to_duckdb(con, df_params, "$(pfx)parameter_stats", overwrite)
        end

        # 4. Save Denoised Predictions Table
        preds = hasproperty(res, :predictions) && hasproperty(res.predictions, :denoised) ?
          res.predictions.denoised : nothing
        if !isnothing(preds) && preds isa NamedTuple && hasproperty(preds, :mean)
            N = length(preds.mean)
            pred_lower = hasproperty(preds, :lower) ? preds.lower : fill(NaN, N)
            pred_upper = hasproperty(preds, :upper) ? preds.upper : fill(NaN, N)
            # `pred_sd` is the within-model posterior predictive SD required by
            # `bma_weighted_predictions` to apply the law of total variance.
            pred_sd_col, sd_source, pred_asym = _recover_pred_sd(
                hasproperty(preds, :std) ? preds.std : nothing,
                pred_lower, pred_upper, preds.mean)
            # Report it here too, not only in the BMA path: a width-derived SD is an
            # approximation, and this is where it first enters the stored predictions.
            if sd_source == :interval_width
                _warn_if_interval_asymmetric(pred_asym, "the predictions table $(pfx)predictions")
            end
            df_preds = DataFrame(
                obs_id = 1:N,
                pred_mean = preds.mean,
                pred_sd = pred_sd_col,
                # Record where the SD came from. Without this, a width-derived
                # approximation is indistinguishable from the model's real predictive
                # SD once it has been round-tripped through this table.
                pred_sd_source = fill(string(sd_source), N),
                pred_lower = pred_lower,
                pred_upper = pred_upper
            )
            y_obs_vec = if hasproperty(res.predictions, :observed) &&
              !isnothing(res.predictions.observed)
                res.predictions.observed
            elseif !isnothing(model) && hasproperty(model, :args) && hasproperty(model.args,
              :M) && hasproperty(model.args.M, :y_obs)
                Array(model.args.M.y_obs)
            else
                nothing
            end
            if !isnothing(y_obs_vec) && length(y_obs_vec) == N
                df_preds.y_obs = y_obs_vec
                df_preds.residual = y_obs_vec .- preds.mean
            end
            _write_df_to_duckdb(con, df_preds, "$(pfx)predictions", overwrite)
        end

        # 5. Save Spatial Geometries in WKT format (if au is available)
        if !isnothing(au) && hasproperty(au, :polygons) && hasproperty(au, :centroids)
            polys = au.polygons
            cents = au.centroids
            S = length(cents)
            wkt_vec = String[_polygon_to_wkt(polys[i]) for i in 1:min(S, length(polys))]
            cx_vec = Float64[cents[i][1] for i in 1:S]
            cy_vec = Float64[cents[i][2] for i in 1:S]
            area_vec = (hasproperty(au, :areas) && length(au.areas) == S) ? Float64.(au.areas) : fill(NaN, S)
            pt_cnt = (hasproperty(au, :point_counts) && length(au.point_counts) == S) ? Int.(au.point_counts) : fill(0, S)

            df_geom = DataFrame(
                unit_id = 1:S,
                centroid_x = cx_vec,
                centroid_y = cy_vec,
                area = area_vec,
                point_count = pt_cnt,
                wkt = wkt_vec
            )
            _write_df_to_duckdb(con, df_geom, "$(pfx)spatial_geometries", overwrite)
        end

        # 6. Save Plots Data Tables
        if hasproperty(res, :plots_data) && !isnothing(res.plots_data)
            for (p_key, p_df) in pairs(res.plots_data)
                if p_df isa DataFrame && !isempty(p_df)
                    tbl_name = "$(pfx)plot_data_$(p_key)"
                    _write_df_to_duckdb(con, p_df, tbl_name, overwrite)
                end
            end
        end

        # 7. Save Posterior Samples Table (if chain is passed)
        if !isnothing(chain)
            df_samples = _chain_to_tidy_df(chain, model)
            _write_df_to_duckdb(con, df_samples, "$(pfx)posterior_samples", overwrite)
        end

        # Explicitly flush and checkpoint storage blocks to disk with compression
        DuckDB.query(con, "CHECKPOINT;")
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end

    @info "TethysBayes analytical results successfully saved to DuckDB database '$duckdb_path'."
    return duckdb_path
end

"""
    load_TethysBayes_results(duckdb_path::AbstractString; table_prefix::String="")::NamedTuple

Loads previously saved model results from a DuckDB database into a structured NamedTuple
matching `model_results_comprehensive`.
"""
function load_TethysBayes_results(duckdb_path::AbstractString; table_prefix::String="")::NamedTuple
    if !isfile(duckdb_path)
        if isfile(duckdb_path * ".duckdb")
            duckdb_path = duckdb_path * ".duckdb"
        else
            error("DuckDB database file not found: '$duckdb_path'")
        end
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)

    pfx = isempty(table_prefix) ? "" : (endswith(table_prefix, "_") ? table_prefix :
      table_prefix * "_")

    metrics_nt = (;)
    parameters_df = DataFrame()
    plots_data_dict = Dict{Symbol, DataFrame}()
    preds_dict = Dict{Symbol, Any}()

    try
        # 1. Load Metrics
        try
            df_m = DataFrame(DuckDB.query(con, "SELECT * FROM $(pfx)metrics"))
            m_dict = Dict{Symbol, Float64}()
            for row in eachrow(df_m)
                m_dict[Symbol(row.metric)] = Float64(row.value)
            end
            metrics_nt = NamedTuple(m_dict)
        catch
        end

        # 2. Load Parameter Stats
        try
            parameters_df = DataFrame(DuckDB.query(con, "SELECT * FROM $(pfx)parameter_stats"))
        catch
        end

        # 3. Load Predictions
        try
            df_p = DataFrame(DuckDB.query(con, "SELECT * FROM $(pfx)predictions"))
            lower = hasproperty(df_p, :pred_lower) ? df_p.pred_lower : fill(NaN, nrow(df_p))
            upper = hasproperty(df_p, :pred_upper) ? df_p.pred_upper : fill(NaN, nrow(df_p))
            # Re-derive rather than trusting a stored value blindly: a `pred_sd` column
            # may be absent (older database) or only partially populated.
            sd, sd_source, sd_asym = _recover_pred_sd(
                hasproperty(df_p, :pred_sd) ? df_p.pred_sd : nothing,
                lower, upper, df_p.pred_mean)
            # This is a bare `catch` block, so an exception here would vanish. A biased
            # width-derived SD is worth surfacing at load time too -- this is the third
            # of the three recovery sites, and until now it checked nothing.
            if sd_source == :interval_width
                _warn_if_interval_asymmetric(sd_asym, "the stored table $(pfx)predictions")
            end
            preds_dict[:denoised] = (
                mean = df_p.pred_mean,
                std = sd,
                std_source = string(sd_source),
                lower = lower,
                upper = upper
            )
        catch
        end

        # 4. Load Plot Data Tables
        tbl_names = DataFrame(DuckDB.query(con, "SELECT table_name FROM
          information_schema.tables WHERE table_schema='main'"))
        plot_prefix = "$(pfx)plot_data_"
        for row in eachrow(tbl_names)
            t_name = string(row.table_name)
            if startswith(t_name, plot_prefix)
                clean_k = Symbol(replace(t_name, plot_prefix => ""))
                plots_data_dict[clean_k] = DataFrame(DuckDB.query(con, "SELECT * FROM $(t_name)"))
            end
        end

    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end

    predictions_nt = if haskey(preds_dict, :denoised)
        (denoised = preds_dict[:denoised],)
    else
        (;)
    end

    return (
        metrics = metrics_nt,
        parameters = parameters_df,
        predictions = predictions_nt,
        plots_data = NamedTuple(plots_data_dict)
    )
end

"""
    query_duckdb(duckdb_path::AbstractString, sql_query::AbstractString)::DataFrame

Executes an arbitrary SQL query against a TethysBayes DuckDB database
and returns the result as a DataFrame.

# Example
```julia
df_high_risk = query_duckdb("output/results.duckdb", 
    "SELECT * FROM TethysBayes_plot_data_latent_field_spatial WHERE latent_field_mean > 1.5 ORDER BY latent_field_mean DESC")
```
"""
function query_duckdb(duckdb_path::AbstractString, sql_query::AbstractString)::DataFrame
    if !isfile(duckdb_path) && isfile(duckdb_path * ".duckdb")
        duckdb_path = duckdb_path * ".duckdb"
    end
    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    local df
    try
        df = DataFrame(DuckDB.query(con, sql_query))
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end
    return df
end

# Helper to write DataFrame into DuckDB table
function _write_df_to_duckdb(con::Any, df::DataFrame, table_name::String, overwrite::Bool)
    if isempty(df)
        return
    end
    df_clean = copy(df)
    for col in names(df_clean)
        if eltype(df_clean[!, col]) <: Symbol
            df_clean[!, col] = string.(df_clean[!, col])
        end
    end
    temp_view = "temp_view_$(rand(10000:99999))"
    DuckDB.register_data_frame(con, df_clean, temp_view)
    create_stmt = overwrite ? "CREATE OR REPLACE TABLE $(table_name) AS SELECT * FROM
      $(temp_view)" :
                              "CREATE TABLE IF NOT EXISTS $(table_name) AS SELECT * FROM
                                $(temp_view)"
    DuckDB.query(con, create_stmt)
    DuckDB.unregister_data_frame(con, temp_view)
end

# Helper to convert polygon coordinates to Well-Known Text (WKT)
function _polygon_to_wkt(poly)
    pts = if poly isa AbstractVector
        if !isempty(poly) && poly[1] isa Tuple
            poly
        elseif !isempty(poly) && poly[1] isa AbstractVector
            [(p[1], p[2]) for p in poly]
        else
            return "POLYGON EMPTY"
        end
    else
        return "POLYGON EMPTY"
    end

    if isempty(pts)
        return "POLYGON EMPTY"
    end
    # Ensure closed ring
    closed_pts = copy(pts)
    if closed_pts[1] != closed_pts[end]
        push!(closed_pts, closed_pts[1])
    end

    coord_strs = ["$(p[1]) $(p[2])" for p in closed_pts]
    return "POLYGON((" * join(coord_strs, ", ") * "))"
end

# ==============================================================================
# SECTION 3: SPATIAL GIS & GEOJSON EXPORT
# ==============================================================================

"""
    export_spatial_results_to_geojson(geojson_path::AbstractString, res::NamedTuple,
                                      au::NamedTuple; property_keys=nothing)

Exports spatial model estimates and polygon geometries to a standard RFC 7946 GeoJSON file
for direct visualization in GIS tools (QGIS, ArcGIS, Mapbox, Leaflet, Kepler.gl).

# Arguments
- `geojson_path::AbstractString`: Output `.geojson` file path.
- `res::NamedTuple`: Result from `model_results_comprehensive`.
- `au::NamedTuple`: Areal units object with `polygons` and `centroids`.
- `property_keys`: (Optional) Vector of column symbols to attach as GeoJSON properties.
"""
function export_spatial_results_to_geojson(
    geojson_path::AbstractString, 
    res::NamedTuple, 
    au::NamedTuple; 
    property_keys = nothing
)
    polys = au.polygons
    cents = au.centroids
    S = length(cents)

    # Extract spatial DataFrame from plots_data or pstats
    df_spatial = if hasproperty(res, :plots_data) && hasproperty(res.plots_data, :latent_field_spatial)
        res.plots_data.latent_field_spatial
    else
        DataFrame(unit_id = 1:S)
    end

    props_to_include = if isnothing(property_keys)
        names(df_spatial)
    else
        string.(property_keys)
    end

    features = String[]
    for i in 1:min(S, length(polys))
        p = polys[i]
        if isempty(p)
            continue
        end
        closed_p = copy(p)
        if closed_p[1] != closed_p[end]
            push!(closed_p, closed_p[1])
        end

        coord_json = "[" * join(["[$(pt[1]), $(pt[2])]" for pt in closed_p], ", ") * "]"
        
        # Build properties JSON
        prop_pairs = String[]
        if nrow(df_spatial) >= i
            row = df_spatial[i, :]
            for col in props_to_include
                if hasproperty(row, Symbol(col))
                    val = row[Symbol(col)]
                    if val isa Number
                        push!(prop_pairs, "\"$col\": $(val)")
                    else
                        push!(prop_pairs, "\"$col\": \"$(val)\"")
                    end
                end
            end
        end
        props_json = "{" * join(prop_pairs, ", ") * "}"

        feature_json = """{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [$coord_json]
            },
            "properties": $props_json
        }"""
        push!(features, feature_json)
    end

    geojson_str = """{
        "type": "FeatureCollection",
        "features": [
            $(join(features, ",\n"))
        ]
    }"""

    dir = dirname(geojson_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    write(geojson_path, geojson_str)

    @info "Spatial model results successfully exported to GeoJSON: '$geojson_path'."
    return geojson_path
end

# ==============================================================================
# SECTION 4: SEQUENTIAL BAYESIAN PRIOR EXTRACTION
# ==============================================================================

"""
    extract_posterior_priors(source; parameter_names=nothing, prior_family=:normal)

Extracts posterior distributions from previous model runs (`res` or a DuckDB database)
and constructs fitted prior distributions (`Normal(mean, std)` or `truncated(Normal(...))`)
ready for sequential Bayesian updating in subsequent `TethysBayes` models.

# Example
```julia
# Extract priors from stage-1 run
priors = extract_posterior_priors("stage1_results.duckdb")

# Use as informative priors in stage-2 model
m2 = @tbm(
    likelihood(y) ~ intercept(prior=priors[:intercept]) +
                    fixed(elev, prior=priors[:beta_elev]),
    df2
)
```
"""
function extract_posterior_priors(
    source::Union{AbstractString, NamedTuple}; 
    parameter_names = nothing, 
    prior_family::Symbol = :normal
)
    df_stats = if source isa AbstractString
        query_duckdb(source, "SELECT * FROM parameter_stats")
    elseif hasproperty(source, :parameters)
        source.parameters
    else
        error("Source must be a DuckDB path or a model results NamedTuple with parameters.")
    end

    if isempty(df_stats)
        @warn "No parameter statistics found in source."
        return Dict{Symbol, Any}()
    end

    p_col = hasproperty(df_stats, :parameters) ? :parameters : (hasproperty(df_stats,
      :parameter) ? :parameter : names(df_stats)[1])
    m_col = :mean
    s_col = :std

    priors_dict = Dict{Symbol, Any}()

    for row in eachrow(df_stats)
        p_name = Symbol(row[p_col])
        if !isnothing(parameter_names) && !(p_name in parameter_names)
            continue
        end

        mu = Float64(row[m_col])
        sigma = max(Float64(row[s_col]), 1e-4) # ensure non-zero variance

        if prior_family == :normal
            priors_dict[p_name] = Normal(mu, sigma)
        elseif prior_family == :truncated_normal
            priors_dict[p_name] = truncated(Normal(mu, sigma), 0.0, Inf)
        else
            priors_dict[p_name] = Normal(mu, sigma)
        end
    end

    @info "Extracted $(length(priors_dict)) informative prior distributions."
    return priors_dict
end

# ==============================================================================
# SECTION 5: MULTI-MODEL ENSEMBLING & BAYESIAN MODEL AVERAGING (BMA)
# ==============================================================================

"""
    save_model_ensemble(duckdb_path::AbstractString, ensemble_dict::Dict; overwrite::Bool=true)

Persists a collection of candidate models into a unified DuckDB database.

Constructs a central `models_registry` table comparing:
- `rmse`, `r_pearson`, `waic`
- `delta_waic`: \\Delta WAIC_k = WAIC_k - min_j WAIC_j
- `waic_weight`: Akaike/WAIC model weights
"""
function save_model_ensemble(
    duckdb_path::AbstractString, 
    ensemble_dict::Dict; 
    overwrite::Bool=true
)
    # Save individual models with table prefixes
    waic_vals = Float64[]
    rmse_vals = Float64[]
    m_names = String[]

    for (m_sym, bundle) in ensemble_dict
        pfx = string(m_sym) * "_"
        m_obj = hasproperty(bundle, :model) ? bundle.model : nothing
        c_obj = hasproperty(bundle, :chain) ? bundle.chain : nothing
        r_obj = hasproperty(bundle, :results) ? bundle.results : bundle

        save_TethysBayes_results(duckdb_path, r_obj; model=m_obj, chain=c_obj, table_prefix=pfx,
          overwrite=overwrite)

        w_val = hasproperty(r_obj.metrics, :waic) ? Float64(r_obj.metrics.waic) : NaN
        rm_val = hasproperty(r_obj.metrics, :rmse) && r_obj.metrics.rmse isa Number ?
          Float64(r_obj.metrics.rmse) : NaN

        push!(m_names, string(m_sym))
        push!(waic_vals, w_val)
        push!(rmse_vals, rm_val)
    end

    # Compute WAIC Weights
    valid_waic = filter(!isnan, waic_vals)
    min_w = isempty(valid_waic) ? 0.0 : minimum(valid_waic)
    delta_w = [isnan(w) ? NaN : w - min_w for w in waic_vals]
    exp_w = [isnan(d) ? 0.0 : exp(-0.5 * d) for d in delta_w]
    sum_exp = sum(exp_w)
    weights = sum_exp > 0 ? exp_w ./ sum_exp : fill(1.0 / length(m_names), length(m_names))

    df_registry = DataFrame(
        model_name = m_names,
        rmse = rmse_vals,
        waic = waic_vals,
        delta_waic = delta_w,
        bma_weight = weights
    )

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        _write_df_to_duckdb(con, df_registry, "models_registry", overwrite)
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end

    @info "Model ensemble saved with $(length(m_names)) models in '$duckdb_path'."
    return df_registry
end

"""
    bma_weighted_predictions(duckdb_path::AbstractString) -> DataFrame

Computes Bayesian Model Averaged (BMA) predictions across all candidate models
registered in the DuckDB database using their normalized WAIC weights.

The BMA predictive variance follows the law of total variance,
`Var = Σ_k w_k [ σ²_k + (μ_k - μ̄)² ]`, combining each model's own predictive
variance (`pred_sd`, or the 95% interval width for databases written before that
column existed) with the between-model spread of the means.

Returns a DataFrame with `(obs_id, bma_pred_mean, bma_pred_sd)`.
"""
function bma_weighted_predictions(duckdb_path::AbstractString)::DataFrame
    df_reg = query_duckdb(
        duckdb_path, "SELECT * FROM models_registry WHERE NOT isnan(bma_weight)"
    )
    if isempty(df_reg)
        error("No models with valid BMA weights found in '$duckdb_path'.")
    end

    # Fetch predictions for each model
    preds_list = DataFrame[]
    weights = Float64[]
    model_names = String[]

    for row in eachrow(df_reg)
        m_name = string(row.model_name)
        w = Float64(row.bma_weight)
        # `SELECT *` so that databases written before the `pred_sd` column existed still load.
        df_p = query_duckdb(duckdb_path, "SELECT * FROM $(m_name)_predictions ORDER BY obs_id")
        push!(preds_list, df_p)
        push!(weights, w)
        push!(model_names, m_name)
    end

    N = nrow(preds_list[1])
    bma_mean = zeros(Float64, N)
    bma_var = zeros(Float64, N)

    for (df_p, w) in zip(preds_list, weights)
        bma_mean .+= w .* df_p.pred_mean
    end

    # Law of total variance:
    #   Var(y) = Σ_k w_k [ σ²_k + (μ_k - μ̄)² ]
    # The first term is the *within-model* predictive variance and the second is the
    # *between-model* spread. Omitting the first term (as this function previously did)
    # collapses the BMA SD to zero whenever the candidate models agree on the mean,
    # even if every one of them is wildly uncertain.
    for (df_p, w, m_name) in zip(preds_list, weights, model_names)
        lower = hasproperty(df_p, :pred_lower) ? df_p.pred_lower : nothing
        upper = hasproperty(df_p, :pred_upper) ? df_p.pred_upper : nothing
        sd_k, sd_source, asym = _recover_pred_sd(
            hasproperty(df_p, :pred_sd) ? df_p.pred_sd : nothing,
            lower, upper, df_p.pred_mean)

        if sd_source == :interval_width
            # This database predates the `pred_sd` column, or the model produced no
            # `std`. Say so: the value is an approximation, and it is only trustworthy
            # when the interval is close to symmetric. The measure itself comes from
            # `_recover_pred_sd`, so all three recovery sites agree on what "asymmetric"
            # means.
            _warn_if_interval_asymmetric(asym, "model '$(m_name)'")
        elseif sd_source == :missing
            @warn "Model '$(m_name)' has no usable predictive SD (no `pred_sd` and no " *
                  "valid interval bounds); it contributes only its between-model spread " *
                  "to `bma_pred_sd`."
            sd_k = zeros(Float64, N)
        end

        bma_var .+= w .* (sd_k .^ 2 .+ (df_p.pred_mean .- bma_mean) .^ 2)
    end

    return DataFrame(
        obs_id = preds_list[1].obs_id,
        bma_pred_mean = bma_mean,
        bma_pred_sd = sqrt.(bma_var)
    )
end

# ==============================================================================
# SECTION 6: OUT-OF-SAMPLE PREDICTIONS, ZERO-COPY EXPORT & COMPACTION
# ==============================================================================

"""
    save_out_of_sample_predictions(duckdb_path::AbstractString, pred_df::DataFrame; 
                                   table_name::String="out_of_sample_predictions",
                                   overwrite::Bool=true)

Saves out-of-sample predictions (e.g. from `predict(model, chain, new_data)`) into DuckDB.
"""
function save_out_of_sample_predictions(
    duckdb_path::AbstractString, 
    pred_df::DataFrame; 
    table_name::String="out_of_sample_predictions", 
    overwrite::Bool=true
)
    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        _write_df_to_duckdb(con, pred_df, table_name, overwrite)
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end
    @info "Out-of-sample predictions saved to table '$table_name' in '$duckdb_path'."
end


"""
    export_results_to_csv(duckdb_path::AbstractString, table_name::AbstractString,
                          output_csv_path::AbstractString)

Exports a DuckDB table to a CSV file.
"""
function export_results_to_csv(
    duckdb_path::AbstractString, 
    table_name::AbstractString, 
    output_csv_path::AbstractString
)
    dir = dirname(output_csv_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        clean_path = replace(output_csv_path, "\\" => "/")
        DuckDB.query(con, "COPY (SELECT * FROM $(table_name)) TO '$(clean_path)' (HEADER,
          DELIMITER ',')")
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end
    @info "Table '$table_name' exported to CSV: '$output_csv_path'."
    return output_csv_path
end

"""
    compact_duckdb(duckdb_path::AbstractString)

Performs `VACUUM; ANALYZE;` on the DuckDB database to reclaim unallocated disk space
and optimize index query statistics.
"""
function compact_duckdb(duckdb_path::AbstractString)
    if !isfile(duckdb_path) && isfile(duckdb_path * ".duckdb")
        duckdb_path = duckdb_path * ".duckdb"
    end
    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        DuckDB.query(con, "VACUUM")
        DuckDB.query(con, "ANALYZE")
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end
    @info "DuckDB database '$duckdb_path' compacted and optimized."
end

# ==============================================================================
# SECTION 7: POSTERIOR SAMPLES EXPORT & MULTI-MODEL INTEGRATION
# ==============================================================================

"""
    export_posterior_samples_to_duckdb(duckdb_path::AbstractString, chain, model=nothing;
                                       table_name::String="TethysBayes_posterior_samples", 
                                       format::Symbol=:tidy, overwrite::Bool=true)

Exports raw MCMC posterior draws into DuckDB for high-performance SQL querying.
"""
function export_posterior_samples_to_duckdb(
    duckdb_path::AbstractString, 
    chain, 
    model=nothing;
    table_name::String="TethysBayes_posterior_samples", 
    format::Symbol=:tidy, 
    overwrite::Bool=true
)
    if isnothing(chain)
        return
    end

    # Convert chain into tidy or wide DataFrame
    df_samples = if format == :wide
        DataFrame(chain)
    else
        _chain_to_tidy_df(chain, model)
    end

    db = DuckDB.DB(duckdb_path)
    con = DuckDB.connect(db)
    try
        _write_df_to_duckdb(con, df_samples, table_name, overwrite)
    finally
        DuckDB.disconnect(con)
        DuckDB.close(db)
    end

    @info "Posterior samples exported to DuckDB table '$table_name' in '$duckdb_path'."
end

function _chain_to_tidy_df(chain, model=nothing)
    if chain isa DataFrame
        return chain
    end
    p_names = _get_clean_chain_param_names(chain)
    if isempty(p_names) && !isnothing(model) && hasproperty(model.args, :M)
        p_names = build_param_registry(model.args.M).names
    end
    
    n_chains_val = _get_chain_n_chains(chain)
    
    rows = []
    for pname in p_names
        try
            samples_mat = extract_param_matrix(chain, pname)
            dim = size(samples_mat, 2)
            n_total = size(samples_mat, 1)
            n_iter_per_chain = n_chains_val >= 1 ? max(1, n_total ÷ n_chains_val) : n_total
            
            for d in 1:dim
                v = samples_mat[:, d]
                param_label = dim == 1 ? string(pname) : "$(pname)[$d]"
                for idx in 1:n_total
                    c = n_chains_val > 1 ? min(n_chains_val, (idx - 1) ÷ n_iter_per_chain + 1) : 1
                    it = n_chains_val > 1 ? ((idx - 1) % n_iter_per_chain + 1) : idx
                    push!(rows, (
                        iteration = it,
                        chain = c,
                        parameter = param_label,
                        value = Float64(v[idx])
                    ))
                end
            end
        catch
        end
    end
    return DataFrame(rows)
end

"""
    import_posterior_samples_from_duckdb(duckdb_path::AbstractString; 
                                         table_name::String="TethysBayes_posterior_samples")::DataFrame

Reads posterior samples stored in DuckDB into a DataFrame.
"""
function import_posterior_samples_from_duckdb(
    duckdb_path::AbstractString; 
    table_name::String="TethysBayes_posterior_samples"
)::DataFrame
    return query_duckdb(duckdb_path, "SELECT * FROM $(table_name)")
end

# ==============================================================================
# SECTION 8: CHAIN EXTENSION & RESUMING SAMPLING
# ==============================================================================

"""
    append_posterior_samples(chain1, chain2)

Concatenates two MCMC chains across sampling iterations.
Supports `FlexiChain`, `VNChain`, `MCMCChains.Chains`, and `DataFrame`.
"""
function append_posterior_samples(chain1, chain2)
    if isnothing(chain1)
        return chain2
    end
    if isnothing(chain2)
        return chain1
    end

    try
        return vcat(chain1, chain2)
    catch
        try
            return [chain1; chain2]
        catch
            return chain2
        end
    end
end

"""
    extend_sampling(model::DynamicPPL.Model, prev_chain, n_additional_samples::Int; 
                    sampler=NUTS(), kwargs...)

Draws `n_additional_samples` from `model` and concatenates them with `prev_chain`.
"""
function extend_sampling(
    model::DynamicPPL.Model, 
    prev_chain, 
    n_additional_samples::Int; 
    sampler=NUTS(), 
    kwargs...
)
    new_chain = sample(model, sampler, n_additional_samples; kwargs...)
    combined = append_posterior_samples(prev_chain, new_chain)
    @info "Successfully extended chain by $n_additional_samples iterations."
    return combined
end

# ==============================================================================
# SECTION 9: UNIFIED MODEL & RESULTS BUNDLE
# ==============================================================================

"""
    save_TethysBayes_bundle(base_path::AbstractString, model::DynamicPPL.Model, chain, res::NamedTuple; 
                     au=nothing, metadata::Dict=Dict(), compress::Bool=true)

Unified one-line persister saving:
1. `<base_path>.jld2`: Live Turing Model state `m`, `M`, and posterior `chain`.
2. `<base_path>.duckdb`: Relational analytical database containing `res`, metrics, and plot tables.
"""
function save_TethysBayes_bundle(
    base_path::AbstractString, 
    model::DynamicPPL.Model, 
    chain, 
    res::NamedTuple; 
    au=nothing, 
    metadata::Dict=Dict(), 
    compress::Bool=true
)
    clean_base = replace(base_path, r"\.(jld2|duckdb|db|TethysBayes)$" => "")
    jld2_file = clean_base * ".jld2"
    duckdb_file = clean_base * ".duckdb"

    save_TethysBayes_model(jld2_file, model; chain=chain, au=au, metadata=metadata, compress=compress)
    save_TethysBayes_results(duckdb_file, res; model=model, chain=chain, au=au, overwrite=true)

    @info "Complete TethysBayes bundle saved to:\n  - Model:   '$jld2_file'\n  - Results: '$duckdb_file'"
    return (model_file = jld2_file, results_file = duckdb_file)
end

"""
    load_TethysBayes_bundle(base_path::AbstractString; calling_module::Module=Main)

Loads a complete TethysBayes model bundle (`<base_path>.jld2` and `<base_path>.duckdb`).
"""
function load_TethysBayes_bundle(base_path::AbstractString; calling_module::Module=Main)
    clean_base = replace(base_path, r"\.(jld2|duckdb|db|TethysBayes)$" => "")
    jld2_file = clean_base * ".jld2"
    duckdb_file = clean_base * ".duckdb"

    m_data = load_TethysBayes_model(jld2_file; calling_module=calling_module)
    res_data = isfile(duckdb_file) ? load_TethysBayes_results(duckdb_file) : (;)

    return (
        model = m_data.model,
        chain = m_data.chain,
        results = res_data,
        au = m_data.au,
        metadata = m_data.metadata
    )
end

"""
    PriorPosteriorBundle

Structured container encapsulating both prior and posterior parameter distributions,
MCMC chains, and model provenance.

# Fields
- `model`: Underlying `DynamicPPL.Model`.
- `prior_chain`: MCMC draws sampled from the prior distribution.
- `posterior_chain`: MCMC draws sampled from the posterior distribution.
- `prior_parameters`: DataFrame of summary statistics for prior parameters.
- `posterior_parameters`: DataFrame of summary statistics for posterior parameters.
- `au`: Spatial analytical units or mesh metadata, if available.
- `metadata`: Provenance metadata dictionary.

# Indexing
- `bundle[:prior, :parameters]` -> returns prior summary DataFrame.
- `bundle[:prior, :parameters, 1:1000]` -> returns specified rows of prior summary DataFrame.
- `bundle[:posterior, :parameters]` -> returns posterior summary DataFrame.
- `bundle[:posterior, :parameters, 1:1000]` -> returns specified rows of posterior summary DataFrame.
- `bundle[:prior, :chain]` -> returns prior MCMC draws.
- `bundle[:posterior, :chain]` -> returns posterior MCMC draws.
- `bundle[:prior]` -> `(parameters = bundle.prior_parameters, chain = bundle.prior_chain)`
- `bundle[:posterior]` -> `(parameters = bundle.posterior_parameters, chain = bundle.posterior_chain)`
"""
struct PriorPosteriorBundle
    model::Any
    prior_chain::Any
    posterior_chain::Any
    prior_parameters::DataFrame
    posterior_parameters::DataFrame
    au::Any
    metadata::Dict
end

function Base.getindex(b::PriorPosteriorBundle, dist_type::Symbol)
    d = Symbol(lowercase(string(dist_type)))
    if d in (:prior, :priors)
        return (parameters = b.prior_parameters, chain = b.prior_chain)
    elseif d in (:posterior, :posteriors)
        return (parameters = b.posterior_parameters, chain = b.posterior_chain)
    else
        error("Invalid distribution key ':$dist_type'. Expected :prior or :posterior.")
    end
end

function Base.getindex(b::PriorPosteriorBundle, dist_type::Symbol, target::Symbol)
    d = Symbol(lowercase(string(dist_type)))
    t = Symbol(lowercase(string(target)))

    if d in (:prior, :priors)
        if t in (:param, :params, :parameters)
            return b.prior_parameters
        elseif t in (:chain, :chains, :draws)
            return b.prior_chain
        end
    elseif d in (:posterior, :posteriors)
        if t in (:param, :params, :parameters)
            return b.posterior_parameters
        elseif t in (:chain, :chains, :draws)
            return b.posterior_chain
        end
    end
    error("Invalid query: [:$dist_type, :$target]. Expected (:prior|:posterior, :parameters|:chain).")
end

function Base.getindex(b::PriorPosteriorBundle, dist_type::Symbol, target::Symbol, rows)
    obj = b[dist_type, target]
    if obj isa DataFrame
        max_r = nrow(obj)
        valid_rows = if rows isa AbstractRange
            intersect(rows, 1:max_r)
        elseif rows isa Integer
            clamp(rows, 1, max_r)
        else
            filter(r -> 1 <= r <= max_r, rows)
        end
        return obj[valid_rows, :]
    elseif obj isa AbstractMatrix
        return obj[rows, :]
    elseif obj isa AbstractArray && ndims(obj) >= 3
        # 3D chain containers (e.g. `MCMCChains.Chains` is `[iterations, params, chains]`,
        # `FlexiChains.VNChain` is `[iterations, chains]`) index on the leading dimension.
        # MCMCChains is an optional (undeclared) dependency, so dispatch structurally
        # rather than on its concrete type to avoid an `UndefVarError` at runtime.
        return obj[rows, :, :]
    else
        return obj[rows]
    end
end

function Base.show(io::IO, b::PriorPosteriorBundle)
    n_p = nrow(b.posterior_parameters)
    n_prior_p = nrow(b.prior_parameters)
    print(io, "PriorPosteriorBundle(\n")
    print(io, "  Prior Parameters:     $n_prior_p parameters\n")
    print(io, "  Posterior Parameters: $n_p parameters\n")
    print(io, "  Prior Chain:          $(!isnothing(b.prior_chain) ? string(typeof(b.prior_chain)) : "none")\n")
    print(io, "  Posterior Chain:      $(!isnothing(b.posterior_chain) ? string(typeof(b.posterior_chain)) : "none")\n")
    print(io, ")")
end

"""
    extract_prior_posterior(bundle; n_prior=1000, seed=42, calling_module=Main)
    extract_prior_posterior(model::DynamicPPL.Model, chain; n_prior=1000, seed=42, au=nothing, metadata=Dict())

Extracts prior and posterior chains and parameter summaries from a saved bundle,
model results, or model and chain.

# Mathematical Background
Given an observation likelihood \$\\mathcal{L}(\\mathbf{y} \\mid \\boldsymbol{\\theta})\$
and joint prior \$\\pi(\\boldsymbol{\\theta})\$, Bayesian inference computes the posterior:
\$\\pi(\\boldsymbol{\\theta} \\mid \\mathbf{y}) = \\frac{\\mathcal{L}(\\mathbf{y} \\mid \\boldsymbol{\\theta}) \\pi(\\boldsymbol{\\theta})}{\\int \\mathcal{L}(\\mathbf{y} \\mid \\boldsymbol{\\theta}) \\pi(\\boldsymbol{\\theta}) d\\boldsymbol{\\theta}}\$
This function extracts both the unconditioned prior realizations \$\\boldsymbol{\\theta}^{(s)} \\sim \\pi(\\boldsymbol{\\theta})\$
and the conditioned posterior realizations \$\\boldsymbol{\\theta}^{(s)} \\sim \\pi(\\boldsymbol{\\theta} \\mid \\mathbf{y})\$,
enabling direct prior-vs-posterior shrinkage and update diagnostics.

# Arguments
- `bundle`: A bundle NamedTuple returned by `save_TethysBayes_bundle` or `load_TethysBayes_bundle`,
  a filepath string pointing to a saved `.jld2` or `.duckdb` bundle, or a model results NamedTuple.
- `n_prior`: Number of prior samples to generate if prior draws were not pre-saved (default: 1000).
- `seed`: Random seed for prior sampling reproducibility (default: 42).

# Returns
A `PriorPosteriorBundle` supporting:
```julia
prior_posterior = extract_prior_posterior(bn_bundle)
display(prior_posterior[:prior, :parameters, 1:1000])
display(prior_posterior[:posterior, :parameters])
```
"""
function extract_prior_posterior(
    bundle;
    n_prior::Int=1000,
    seed::Int=42,
    calling_module::Module=Main
)
    model = nothing
    post_chain = nothing
    post_params = DataFrame()
    au = nothing
    metadata = Dict{String, Any}()

    if bundle isa AbstractString
        loaded = load_TethysBayes_bundle(bundle; calling_module=calling_module)
        model = loaded.model
        post_chain = loaded.chain
        au = loaded.au
        metadata = loaded.metadata
        if hasproperty(loaded, :results) && hasproperty(loaded.results, :parameters) &&
           loaded.results.parameters isa DataFrame && !isempty(loaded.results.parameters)
            post_params = loaded.results.parameters
        end
    elseif bundle isa NamedTuple && haskey(bundle, :model_file)
        loaded = load_TethysBayes_bundle(bundle.model_file; calling_module=calling_module)
        model = loaded.model
        post_chain = loaded.chain
        au = loaded.au
        metadata = loaded.metadata
        if hasproperty(loaded, :results) && hasproperty(loaded.results, :parameters) &&
           loaded.results.parameters isa DataFrame && !isempty(loaded.results.parameters)
            post_params = loaded.results.parameters
        end
    elseif bundle isa NamedTuple && haskey(bundle, :model) && haskey(bundle, :chain)
        model = bundle.model
        post_chain = bundle.chain
        au = get(bundle, :au, nothing)
        metadata = get(bundle, :metadata, Dict{String, Any}())
        if haskey(bundle, :results) && hasproperty(bundle.results, :parameters) &&
           bundle.results.parameters isa DataFrame && !isempty(bundle.results.parameters)
            post_params = bundle.results.parameters
        elseif haskey(bundle, :parameters) && bundle.parameters isa DataFrame && !isempty(bundle.parameters)
            post_params = bundle.parameters
        end
    elseif bundle isa NamedTuple && haskey(bundle, :parameters) && haskey(bundle, :effects)
        # return from model_results_comprehensive
        model = get(bundle, :model, nothing)
        post_chain = get(bundle, :chain, nothing)
        post_params = bundle.parameters
        au = get(bundle, :au, nothing)
    else
        error("Unrecognized bundle format passed to extract_prior_posterior. Expected filepath, NamedTuple from save_TethysBayes_bundle, or model results.")
    end

    if isnothing(model)
        error("Model object could not be resolved from bundle. Ensure the bundle contains the DynamicPPL model.")
    end

    # Sample prior draws
    rng = Random.MersenneTwister(seed)
    prior_chain = try
        Base.invokelatest(sample, rng, model, Prior(), n_prior; progress=false)
    catch e
        @warn "Prior sampling failed: $e. Returning empty prior chain."
        nothing
    end

    prior_params = if !isnothing(prior_chain)
        try
            _compute_direct_parameter_summary(prior_chain, model)
        catch e
            DataFrame()
        end
    else
        DataFrame()
    end

    if isempty(post_params) && !isnothing(post_chain)
        post_params = try
            _compute_direct_parameter_summary(post_chain, model)
        catch e
            DataFrame()
        end
    end

    return PriorPosteriorBundle(
        model,
        prior_chain,
        post_chain,
        prior_params,
        post_params,
        au,
        metadata
    )
end

function extract_prior_posterior(
    model::DynamicPPL.Model,
    chain;
    n_prior::Int=1000,
    seed::Int=42,
    au=nothing,
    metadata=Dict{String, Any}()
)
    rng = Random.MersenneTwister(seed)
    prior_chain = try
        Base.invokelatest(sample, rng, model, Prior(), n_prior; progress=false)
    catch e
        @warn "Prior sampling failed: $e. Returning empty prior chain."
        nothing
    end

    prior_params = if !isnothing(prior_chain)
        try
            _compute_direct_parameter_summary(prior_chain, model)
        catch e
            DataFrame()
        end
    else
        DataFrame()
    end

    post_params = try
        _compute_direct_parameter_summary(chain, model)
    catch e
        DataFrame()
    end

    return PriorPosteriorBundle(
        model,
        prior_chain,
        chain,
        prior_params,
        post_params,
        au,
        metadata
    )
end

