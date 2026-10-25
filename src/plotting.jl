"""
    plotting.jl

Centralized visualization library for Bayesian Spatio-Temporal Models (TethysBayes),
including PPC checks, parameter diagnostics, and model effect visualizations.
Generic spatial, graph, and hydrodynamic visualization primitives are provided
by TethysPlots.

Version: v1.1.0
"""

using TethysPlots: _poly_xy, _centroids_xy, _extract_polygons, _extract_centroids,
                   _build_spatial_domain_grid, _reconstruct_spatial_surface,
                   _extract_graph_edges_and_degrees, reconstruct_spatial_surface,
                   build_spatial_domain_grid, extract_graph_edges_and_degrees


"""
    _resolve_temporal_coordinates(M, input_data, tm_len::Int, comp_key=nothing)

Robustly infers and aligns temporal coordinates (raw per-observation time vector,
sorted unique time units, and axis label) for plotting.
Prevents observation index fallback (e.g. 1:560) when time units range across 1:10 or
  calendar years.
"""
function _resolve_temporal_coordinates(M, input_data, tm_len::Int, comp_key=nothing)
    # 0. Check domain_values or t_values directly inside M
    if !isnothing(M) && hasproperty(M, :domain_values) && !isnothing(comp_key)
        d_dict = M.domain_values
        if d_dict isa AbstractDict && haskey(d_dict, Symbol(comp_key))
            d_vals = d_dict[Symbol(comp_key)]
            if length(d_vals) == tm_len
                t_lbl = (Symbol(comp_key) == :t_idx) ? "Time Index" : (Symbol(comp_key) == :year ? "Year" : string(comp_key))
                return (raw_t = d_vals, unique_t = sort(unique(d_vals)), label = t_lbl)
            end
        end
    end

    if !isnothing(M) && hasproperty(M, :t_values) && !isnothing(M.t_values)
        if length(M.t_values) == tm_len
            t_lbl = haskey(M, :t_idx_var) ? (M.t_idx_var == :year ? "Year" : string(M.t_idx_var)) : "Year"
            return (raw_t = M.t_values, unique_t = sort(unique(M.t_values)), label = t_lbl)
        end
    end

    # 1. Search in input_data or M.data if available
    df_to_check = if !isnothing(input_data) && input_data isa DataFrame
        input_data
    elseif !isnothing(M) && hasproperty(M, :data) && !isnothing(M.data) && M.data isa DataFrame
        M.data
    else
        nothing
    end

    if !isnothing(df_to_check)
        candidates = Symbol[]
        if !isnothing(comp_key)
            push!(candidates, Symbol(comp_key))
        end
        if !isnothing(M) && haskey(M, :t_idx_var) && !isnothing(M.t_idx_var)
            push!(candidates, Symbol(M.t_idx_var))
        end
        append!(candidates, [:year, :time, :date, :month, :t_idx, :t, :day, :period])

        for cand in unique(candidates)
            if hasproperty(df_to_check, cand)
                col_data = df_to_check[!, cand]
                u_vals = sort(unique(col_data))
                t_lbl = (cand == :t_idx) ? "Time Index" : (cand == :year ? "Year" : string(cand))
                if length(u_vals) == tm_len
                    return (raw_t = u_vals, unique_t = u_vals, label = t_lbl)
                elseif length(col_data) == tm_len
                    return (raw_t = col_data, unique_t = u_vals, label = t_lbl)
                elseif tm_len > 1 && eltype(col_data) <: Number
                    grid_vals = collect(range(minimum(col_data), maximum(col_data), length=tm_len))
                    return (raw_t = grid_vals, unique_t = grid_vals, label = t_lbl)
                end
            end
        end
    end

    # 2. Check M.t_idx
    if !isnothing(M) && haskey(M, :t_idx) && !isnothing(M.t_idx) && !isempty(M.t_idx)
        t_vec = M.t_idx
        if length(t_vec) == tm_len
            unique_t = sort(unique(t_vec))
            t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Time Index"
            return (raw_t = t_vec, unique_t = unique_t, label = t_lbl)
        elseif length(unique(t_vec)) == tm_len
            unique_t = sort(unique(t_vec))
            t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Time Index"
            return (raw_t = unique_t, unique_t = unique_t, label = t_lbl)
        end
    end

    # 3. Check M.t_values with interpolation if length matches tm_len
    if !isnothing(M) && hasproperty(M, :t_values) && !isnothing(M.t_values) && tm_len > 1
        tv = M.t_values
        grid_vals = collect(range(minimum(tv), maximum(tv), length=tm_len))
        t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Year"
        return (raw_t = grid_vals, unique_t = grid_vals, label = t_lbl)
    end

    # 4. Check M.t_N (number of time steps)
    if !isnothing(M) && haskey(M, :t_N) && M.t_N > 0
        t_N = M.t_N
        unique_t = collect(1:t_N)
        t_lbl = haskey(M, :t_idx_var) ? string(M.t_idx_var) : "Time Index"
        if tm_len == t_N
            return (raw_t = unique_t, unique_t = unique_t, label = t_lbl)
        elseif haskey(M, :s_N) && M.s_N > 0 && tm_len == M.s_N * t_N
            raw_t = [(i - 1) % t_N + 1 for i in 1:tm_len]
            return (raw_t = raw_t, unique_t = unique_t, label = t_lbl)
        elseif tm_len % t_N == 0
            raw_t = [(i - 1) % t_N + 1 for i in 1:tm_len]
            return (raw_t = raw_t, unique_t = unique_t, label = t_lbl)
        end
    end

    # 5. Fallback: integer sequence
    unique_t = collect(1:tm_len)
    return (raw_t = unique_t, unique_t = unique_t, label = !isnothing(comp_key) ? string(comp_key) : "Time Index")
end


"""
    _detect_spatial_coords(data, M=NamedTuple(), au=nothing)

Safely discovers and extracts a two-column `Matrix{Float64}` of 2D spatial
coordinates from the input dataset, model metadata `M`, or spatial partitioning `au`.
Returns `nothing` if 2D coordinates cannot be identified.
"""
function _detect_spatial_coords(data, M=NamedTuple(), au=nothing)
    # 1. Direct DataFrame columns check
    df_candidate = if !isnothing(data) && data isa DataFrame
        data
    elseif haskey(M, :data) && !isnothing(M.data) && M.data isa DataFrame
        M.data
    else
        nothing
    end

    if !isnothing(df_candidate)
        # 1. Check component positional args first if present (matching formula variables)
        if haskey(M, :components)
            for spec in M.components
                vars = get(spec.params, :positional_args, Symbol[])
                if length(vars) == 2 && all(hasproperty(df_candidate, Symbol(v)) for v in vars)
                    v1, v2 = Symbol(vars[1]), Symbol(vars[2])
                    # If given as (northing/lat, easting/lon), re-orient to (x, y)
                    if (v1 in [:plat, :lat, :latitude, :northing]) &&
                       (v2 in [:plon, :lon, :longitude, :easting])
                        return Matrix{Float64}(df_candidate[!, [v2, v1]])
                    else
                        return Matrix{Float64}(df_candidate[!, [v1, v2]])
                    end
                end
            end
        end

        # 2. Check detected coordinate column pairs
        detected = try
            _detect_xy_columns(df_candidate)
        catch
            nothing
        end
        if !isnothing(detected)
            return Matrix{Float64}(df_candidate[!, [detected[1], detected[2]]])
        end
    end

    # 2. Check centroids or points from au
    if !isnothing(au) && (au isa NamedTuple || au isa AbstractDict)
        if haskey(au, :centroids) && !isnothing(au[:centroids])
            xs, ys = _centroids_xy(au[:centroids])
            if !isempty(xs)
                return [xs ys]
            end
        elseif haskey(au, :pts) && !isnothing(au[:pts])
            xs, ys = _centroids_xy(au[:pts])
            if !isempty(xs)
                return [xs ys]
            end
        end
    end

    # 3. Check M coordinates field
    if haskey(M, :coords) && !isnothing(M.coords)
        if M.coords isa AbstractMatrix && size(M.coords, 2) >= 2
            return Matrix{Float64}(M.coords[:, 1:2])
        end
    end

    if haskey(M, :s_coord_tuple) && !isnothing(M.s_coord_tuple)
        xs, ys = _centroids_xy(M.s_coord_tuple)
        if !isempty(xs)
            return [xs ys]
        end
    end

    return nothing
end


# -----------------------------------------------------------------------------
# Section 5: High-Level Model Diagnostics & Reconstruction Visualization
# -----------------------------------------------------------------------------

"""
    bstm_plots(model_obj::DynamicPPL.Model, chain, res, M; au=nothing, data=nothing, outcome=1)

Primary diagnostic and effect visualization engine for the `bstm` framework.
Takes model results and generates standard diagnostic plots:
- Posterior Predictive Check (PPC)
- Fixed Effects coefficients
- Conditional / marginal covariate effects
- Component-wise spatial, temporal, seasonal, smooth, and mixed effects
- Spatiotemporal interaction grids
"""
function _bstm_plots_impl(model_obj, chain, res, M; au=nothing, data=nothing, outcome=1)
    plots = Dict{Symbol, Any}()
    plots_data = Dict{Symbol, Any}()
    
    effects = get(res, :effects, nothing)
    arch_obj = get(res, :arch, UnivariateArchitecture())
    is_mv = arch_obj isa MultivariateArchitecture
    pred_denoised = hasproperty(res, :predictions) && hasproperty(res.predictions,
        :denoised) ? res.predictions.denoised : nothing
    
    input_data = !isnothing(data) ? data : get(M, :data, nothing)
    y_obs = if hasproperty(res, :predictions) && hasproperty(res.predictions,
        :observed) && !isnothing(res.predictions.observed)
        res.predictions.observed
    elseif haskey(M, :y_obs) && !isnothing(M.y_obs)
        M.y_obs
    elseif !isnothing(input_data)
        if haskey(M, :outcomes) && !isempty(M.outcomes) && hasproperty(input_data,
            Symbol(M.outcomes[1]))
            Matrix(input_data[!, Symbol.(M.outcomes)])
        elseif hasproperty(input_data, :y)
            input_data.y
        elseif hasproperty(input_data, :y_rate)
            input_data.y_rate
        else
            nothing
        end
    else
        nothing
    end

    function _extract_outcome_vec(y_raw, k_idx)
        if isnothing(y_raw)
            return nothing
        elseif y_raw isa AbstractVector{<:AbstractVector}
            return (k_idx <= length(y_raw)) ? vec(collect(Float64, y_raw[k_idx])) : nothing
        elseif y_raw isa AbstractMatrix
            return (k_idx <= size(y_raw, 2)) ? vec(collect(Float64, y_raw[:, k_idx])) : nothing
        elseif y_raw isa AbstractVector
            return vec(collect(Float64, y_raw))
        else
            return try vec(collect(Float64, y_raw)) catch; nothing end
        end
    end

    polygons = if !isnothing(au) && (au isa NamedTuple || au isa Dict)
        get(au, :polygons, nothing)
    else
        nothing
    end
    
    centroids = if !isnothing(au) && (au isa NamedTuple || au isa Dict)
        get(au, :centroids, nothing)
    else
        nothing
    end

    coords_detected = _detect_spatial_coords(input_data, M, au)

    # --- 1. Posterior Predictive Check ---
    if !isnothing(pred_denoised) && !isnothing(y_obs)
        pred_summary = is_mv ? (outcome <= length(pred_denoised) ? pred_denoised[outcome] : nothing) : pred_denoised
        y_o = _extract_outcome_vec(y_obs, outcome)
        if !isnothing(pred_summary) && hasproperty(pred_summary, :mean) && !isnothing(y_o)
            y_p = vec(collect(Float64, pred_summary.mean))
            if length(y_p) == length(y_o)
                clean_mask = .!isnan.(y_p) .& .!isnan.(y_o)
                clean_p, clean_o = y_p[clean_mask], y_o[clean_mask]
                
                if !isempty(clean_p) && !isempty(clean_o)
                    min_val = min(minimum(clean_p), minimum(clean_o))
                    max_val = max(maximum(clean_p), maximum(clean_o))
                    pad = max((max_val - min_val) * 0.05, 1e-6)
                    lims = (min_val - pad, max_val + pad)

                    # Plot with Credible Intervals if available
                    has_intervals = hasproperty(pred_summary,
                        :lower) && hasproperty(pred_summary, :upper)
                    
                    p_ppc = if has_intervals
                        y_low = vec(collect(Float64, pred_summary.lower))[clean_mask]
                        y_high = vec(collect(Float64, pred_summary.upper))[clean_mask]
                        err_low = max.(clean_p .- y_low, 0.0)
                        err_high = max.(y_high .- clean_p, 0.0)
                        Plots.scatter(
                            clean_o, clean_p, yerror=(err_low, err_high),
                            title="Posterior Predictive Check (PPC)",
                            xlabel="Observed Value (y_obs)",
                            ylabel="Predicted Expected Value (ŷ)",
                            alpha=0.7, markersize=3.5, color=:dodgerblue,
                            markerstrokecolor=:dodgerblue, markerstrokewidth=0.5,
                            xlims=lims, ylims=lims,
                            legend=:topleft, label="Predictions (95% CI)"
                        )
                    else
                        Plots.scatter(
                            clean_o, clean_p,
                            title="Posterior Predictive Check (PPC)",
                            xlabel="Observed Value (y_obs)",
                            ylabel="Predicted Expected Value (ŷ)",
                            alpha=0.7, markersize=3.5, color=:dodgerblue,
                            markerstrokewidth=0,
                            xlims=lims, ylims=lims,
                            legend=:topleft, label="Predictions"
                        )
                    end
                    
                    Plots.plot!(p_ppc, [min_val - pad, max_val + pad], [min_val - pad,
                        max_val + pad],
                                color=:crimson, ls=:dash, lw=2, label="Identity (1:1)")
                    
                    plots[:posterior_predictive_check] = p_ppc
                    plots_data[:posterior_predictive_check] = (observed = y_o, predicted = y_p)
                end
            end
        end
    end

    # --- 2. Helper function for spatial maps (handles continuous surface & discrete units) ---
    function _render_spatial_effect(field_data, title_str, polys, cents; coords=nothing, kwargs...)
        if isnothing(field_data) || !hasproperty(field_data, :mean)
            return nothing
        end 
        s_mean = vec(collect(Float64, field_data.mean))
        if isempty(s_mean) || all(iszero, s_mean)
            return nothing
        end

        coords_to_use = !isnothing(coords) ? coords : coords_detected
        if !isnothing(coords_to_use) && coords_to_use isa Tuple && length(coords_to_use) == 2
            coords_to_use = [coords_to_use[1] coords_to_use[2]]
        end

        # Continuous spatial process: render smooth random surface over entire domain
        if !isnothing(coords_to_use) && coords_to_use isa AbstractMatrix &&
           size(coords_to_use, 1) == length(s_mean)
            return plot_spatial_surface(
                coords_to_use, s_mean;
                title=title_str,
                polygons=polys,
                kwargs...
            )
        end

        n_units = !isnothing(polys) ? length(polys) : (!isnothing(cents) ? length(cents) : (!isnothing(coords_to_use) ? size(coords_to_use, 1) : 0))
        if n_units == 0 && isnothing(coords_to_use)
            return nothing
        end

        unit_vals = if length(s_mean) == n_units
            s_mean
        elseif length(s_mean) > n_units && haskey(M, :s_idx)
            s_idx_vec = M.s_idx
            [begin
                vals = s_mean[s_idx_vec .== i]
                isempty(vals) ? 0.0 : mean(vals)
            end for i in 1:n_units]
        elseif length(s_mean) > n_units && !isnothing(input_data)
            det_s = _detect_spatial_unit_column(input_data; allow_nothing=true)
            if !isnothing(det_s) && hasproperty(input_data, det_s)
                raw_s = input_data[!, det_s]
                u_units = unique(raw_s)
                s_map = Dict(u => i for (i, u) in enumerate(u_units))
                s_idx_vec = [get(s_map, u, 1) for u in raw_s]
                [begin
                    vals = s_mean[s_idx_vec .== i]
                    isempty(vals) ? 0.0 : mean(vals)
                end for i in 1:n_units]
            else
                s_mean[1:min(length(s_mean), max(1, n_units))]
            end
        else
            s_mean[1:min(length(s_mean), max(1, n_units))]
        end

        if !isnothing(polys) && length(polys) == length(unit_vals)
            return choropleth(polys, unit_vals; title=title_str, kwargs...)
        elseif !isnothing(coords_to_use) && coords_to_use isa AbstractMatrix &&
               size(coords_to_use, 1) == length(unit_vals)
            return plot_spatial_surface(
                coords_to_use, unit_vals;
                title=title_str,
                polygons=polys,
                kwargs...
            )
        elseif !isnothing(cents)
            xs, ys = _centroids_xy(cents)
            return plot_spatial_surface(
                [xs ys], unit_vals;
                title=title_str,
                polygons=polys,
                kwargs...
            )
        end
        return nothing
    end

    if (!isnothing(polygons) || !isnothing(centroids) || !isnothing(coords_detected)) && !isnothing(y_obs)
        y_o = _extract_outcome_vec(y_obs, outcome)
        n_units = !isnothing(polygons) ? length(polygons) : (!isnothing(centroids) ? length(centroids) : (!isnothing(coords_detected) ? size(coords_detected, 1) : 0))
        s_idx_vec = if haskey(M, :s_idx)
            M.s_idx
        elseif !isnothing(input_data)
            det_s = _detect_spatial_unit_column(input_data; allow_nothing=true)
            if !isnothing(det_s) && hasproperty(input_data, det_s)
                raw_s = input_data[!, det_s]
                u_units = unique(raw_s)
                s_map = Dict(u => i for (i, u) in enumerate(u_units))
                [get(s_map, u, 1) for u in raw_s]
            elseif !isnothing(y_o)
                1:length(y_o)
            else
                1:n_units
            end
        elseif !isnothing(y_o)
            1:length(y_o)
        else
            1:n_units
        end

        if !isnothing(y_o) && length(y_o) >= n_units && n_units > 0
            family_str = haskey(M, :likelihood_specs) && length(M.likelihood_specs) >= outcome ? string(get(M.likelihood_specs[outcome], :family, "gaussian")) : "gaussian"
            has_count_offset = family_str in ["poisson", "negbin"] && haskey(M,
                :log_offsets) && !isnothing(M.log_offsets) && !all(iszero, M.log_offsets)
            log_off_vec = if has_count_offset
                vec(M.log_offsets isa AbstractVector ? M.log_offsets : M.log_offsets[:, outcome])
            else
                nothing
            end

            y_unit_obs = if !isnothing(log_off_vec)
                [sum(y_o[s_idx_vec .== i]) / max(sum(exp.(log_off_vec[s_idx_vec .== i])), 1e-12) for i in 1:n_units]
            elseif length(y_o) == n_units
                y_o
            else
                [mean(y_o[s_idx_vec .== i]) for i in 1:n_units]
            end
            obs_title = !isnothing(log_off_vec) ? "Observed Standardized Rate (SIR)" : "Observed Mean by Spatial Unit"
            p_obs_map = _render_spatial_effect((mean=y_unit_obs,), obs_title, polygons, centroids; coords=coords_detected)
            if !isnothing(p_obs_map)
                plots[:spatial_observed] = p_obs_map
                plots_data[:spatial_observed] = (values=y_unit_obs,
                    geometry=isnothing(polygons) ? centroids : polygons,
                    coordinates=coords_detected)
            end

            if !isnothing(pred_denoised)
                pred_summary = is_mv ? pred_denoised[outcome] : pred_denoised
                if hasproperty(pred_summary, :mean)
                    y_p = vec(pred_summary.mean)
                    y_unit_fit = if !isnothing(log_off_vec)
                        [sum(y_p[s_idx_vec .== i]) / max(sum(exp.(log_off_vec[s_idx_vec .== i])), 1e-12) for i in 1:n_units]
                    elseif length(y_p) == n_units
                        y_p
                    else
                        [mean(y_p[s_idx_vec .== i]) for i in 1:n_units]
                    end
                    fit_title = !isnothing(log_off_vec) ? "Fitted Relative Risk (RR)" : "Fitted Mean by Spatial Unit"
                    p_fit_map = _render_spatial_effect((mean=y_unit_fit, ), fit_title,
                        polygons, centroids; coords=coords_detected)
                    if !isnothing(p_fit_map)
                        plots[:spatial_fitted] = p_fit_map
                        plots_data[:spatial_fitted] = (values=y_unit_fit,
                            geometry=isnothing(polygons) ? centroids : polygons,
                            coordinates=coords_detected)
                    end

                    y_unit_res = y_unit_obs .- y_unit_fit
                    p_res_map = _render_spatial_effect((mean=y_unit_res, ),
                        "Spatial Residuals (Observed - Fitted)", polygons, centroids; coords=coords_detected)
                    if !isnothing(p_res_map)
                        plots[:spatial_residuals] = p_res_map
                        plots_data[:spatial_residuals] = (values=y_unit_res,
                            geometry=isnothing(polygons) ? centroids : polygons,
                            coordinates=coords_detected)
                    end
                end
            end
        end
    end

    # --- 3. Fixed Effects Plots ---
    if !isnothing(effects) && hasproperty(effects, :fixed) && !isnothing(effects.fixed)
        fe_summary = is_mv ? effects.fixed[outcome] : effects.fixed
        if hasproperty(fe_summary, :mean) && !all(iszero, fe_summary.mean) 
            fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
            if !isempty(fm)
                coef_names = haskey(M,
                    :Xfixed_names) ? string.(M.Xfixed_names) : ["Beta_$i" for i in 1:length(fm)]
                p_forest = Plots.scatter(fm, 1:length(fm), xerror=(fm .- fl, fu .- fm),
                    yticks=(1:length(fm), coef_names), title="Fixed Effects Coefficients",
                    xlabel="Estimate (95% CI)", markersize=5, color=:royalblue, legend=false)
                Plots.vline!(p_forest, [0], color=:crimson, ls=:dash, lw=1.5)
                plots[:fixed_effects] = p_forest
                plots_data[:fixed_effects] = (names=coef_names, mean=fm, lower=fl, upper=fu)
            end
        end
    end

    # --- 4. Conditional Effects Plots ---
    conditional_plots = Dict{Symbol, Any}()
    conditional_plots_data = Dict{Symbol, Any}()
    all_covariates = String[]
    if haskey(M, :Xfixed_names)
        append!(all_covariates, string.(M.Xfixed_names))
    end
    if haskey(M, :components)
        for spec in M.components
            if spec.structure == :smooth
                vars = get(spec.params, :positional_args, [])
                append!(all_covariates, string.(vars))
            end
        end
    end
    all_covariates = unique(Symbol.(all_covariates))

    if haskey(M, :data) && !isnothing(M.data)
        for cov_sym in all_covariates
            if !hasproperty(M.data, cov_sym)
                continue
            end

            if eltype(M.data[!, cov_sym]) <: Number # Continuous covariate
                cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, cov_sym)
                if !isnothing(cond_preds_res)
                    cond_preds, cov_range = cond_preds_res
                    cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                    
                    cm, cl, cu = vec(cond_summary.mean), vec(cond_summary.lower), vec(cond_summary.upper)
                    p_cond = timeseries_ci(cov_range, cm, cl, cu; color=:blue,
                        title="Conditional Effect: $(cov_sym)", xlabel=string(cov_sym),
                        ylabel="Expected Response")
                    plot_key = Symbol("conditional_$(cov_sym)")
                    conditional_plots[plot_key] = p_cond
                    conditional_plots_data[plot_key] = (covariate_values=cov_range, mean=cm,
                        lower=cl, upper=cu)
                end
            else # Categorical covariate
                cond_preds_res = _generate_conditional_predictions(model_obj, chain, M, cov_sym)
                if !isnothing(cond_preds_res)
                    cond_preds, cov_levels = cond_preds_res
                    cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                    
                    cm, cl, cu = vec(cond_summary.mean), vec(cond_summary.lower), vec(cond_summary.upper)
                    p_cond = Plots.bar(string.(cov_levels), cm, yerror=(cm .- cl, cu .- cm),
                        title="Conditional Effect: $(cov_sym)", xlabel=string(cov_sym),
                        ylabel="Expected Response", color=:blue, legend=false)
                    plot_key = Symbol("conditional_$(cov_sym)")
                    conditional_plots[plot_key] = p_cond
                    conditional_plots_data[plot_key] = (covariate_levels=string.(cov_levels), mean=cm, lower=cl, upper=cu)
                end
            end
        end
    end
    if !isempty(conditional_plots)
        plots[:conditional_effects] = conditional_plots
        plots_data[:conditional_effects] = conditional_plots_data
    end

    # --- 5. Component-wise Plotting ---
    smooth_effects_plots = Dict{Symbol, Any}()
    smooth_effects_plots_data = Dict{Symbol, Any}()
    if !isnothing(effects)
        comp_keys = haskey(M, :components) ? [s.key for s in M.components] : [k for k in keys(effects) if !(k in [:fixed, :intercept, :mixed_effects, :st_interaction])]
        for key in comp_keys
            spec_idx = haskey(M, :components) ? findfirst(s -> s.key == key,
                M.components) : nothing
            spec = !isnothing(spec_idx) ? M.components[spec_idx] : nothing
            if !isnothing(spec) && spec.component_obj isa Mixed
                continue
            end
            if !haskey(effects, key)
                continue
            end
            
            component_effects = effects[key]
            main_effect_summary = if hasproperty(component_effects, :noisy)
                is_mv ? component_effects.noisy[outcome] : component_effects.noisy
            elseif hasproperty(component_effects, :structured)
                is_mv ? component_effects.structured[outcome] : component_effects.structured
            else
                continue
            end

            if isnothing(main_effect_summary) || !hasproperty(main_effect_summary,
                :mean) || isempty(main_effect_summary.mean)
                continue
            end

            is_continuous_comp = !isnothing(spec) && (
                spec.component_obj isa RFF ||
                spec.component_obj isa SpectralGP ||
                spec.component_obj isa WaveletGP ||
                spec.component_obj isa SPDE ||
                spec.component_obj isa TPS ||
                spec.component_obj isa GP ||
                spec.component_obj isa Nystrom ||
                (spec.structure in [:smooth, :spatial] && haskey(spec.params, :positional_args) &&
                 length(spec.params[:positional_args]) == 2 &&
                 any(c -> occursin(string(c), lowercase(string(spec.params[:positional_args]))),
                     ["x", "y", "lon", "lat", "coord", "east", "north"]))
            )

            struct_type = if !isnothing(spec)
                is_continuous_comp ? :spatial : spec.structure
            elseif key in [:s_idx, :spatial, :space] ||
                   key in STANDARD_SPATIAL_UNIT_CANDIDATES ||
                   hasproperty(component_effects, :unstructured)
                :spatial
            elseif key in [:t_idx, :temporal, :t] ||
                   key in STANDARD_TEMPORAL_CANDIDATES
                :temporal
            elseif key in [:season, :u_idx, :seasonal]
                :seasonal
            else
                :smooth
            end

            if struct_type == :spatial
                plot_key = Symbol("spatial_$(key)")
                p = _render_spatial_effect(main_effect_summary, "Spatial Effect: $key",
                    polygons, centroids; coords=coords_detected)
                if !isnothing(p)
                    plots[plot_key] = p
                    plots[:spatial] = p
                    plots[Symbol("smooth_$(key)")] = p
                    if !isnothing(coords_detected) &&
                       size(coords_detected, 1) == length(main_effect_summary.mean)
                        recon_p = _reconstruct_spatial_surface(
                            coords_detected, vec(main_effect_summary.mean)
                        )
                        plots_data[plot_key] = (
                            grid_x = recon_p.grid_x,
                            grid_y = recon_p.grid_y,
                            surface = recon_p.surface,
                            values = vec(main_effect_summary.mean),
                            coordinates = coords_detected
                        )
                    else
                        plots_data[plot_key] = (
                            values = vec(main_effect_summary.mean),
                            geometry = isnothing(polygons) ? centroids : polygons
                        )
                    end
                    plots_data[:spatial] = plots_data[plot_key]
                end

                if hasproperty(component_effects, :structured)
                    struct_summary = is_mv ? component_effects.structured[outcome] : component_effects.structured
                    p_struct = _render_spatial_effect(struct_summary,
                        "Structured Spatial Effect: $key", polygons, centroids; coords=coords_detected)
                    if !isnothing(p_struct)
                        plot_key_struct = Symbol("structured_$(key)")
                        plots[plot_key_struct] = p_struct
                        if !isnothing(coords_detected) &&
                           size(coords_detected, 1) == length(struct_summary.mean)
                            recon_p = _reconstruct_spatial_surface(
                                coords_detected, vec(struct_summary.mean)
                            )
                            plots_data[plot_key_struct] = (
                                grid_x = recon_p.grid_x,
                                grid_y = recon_p.grid_y,
                                surface = recon_p.surface,
                                values = vec(struct_summary.mean),
                                coordinates = coords_detected
                            )
                        else
                            plots_data[plot_key_struct] = (
                                values = vec(struct_summary.mean),
                                geometry = isnothing(polygons) ? centroids : polygons
                            )
                        end
                    end
                end
                if hasproperty(component_effects, :unstructured)
                    unstruct_summary = is_mv ? component_effects.unstructured[outcome] : component_effects.unstructured
                    p_unstruct = _render_spatial_effect(unstruct_summary,
                        "Unstructured Spatial Effect: $key", polygons, centroids; coords=coords_detected)
                    if !isnothing(p_unstruct)
                        plot_key_unstruct = Symbol("unstructured_$(key)")
                        plots[plot_key_unstruct] = p_unstruct
                        if !isnothing(coords_detected) &&
                           size(coords_detected, 1) == length(unstruct_summary.mean)
                            recon_p = _reconstruct_spatial_surface(
                                coords_detected, vec(unstruct_summary.mean)
                            )
                            plots_data[plot_key_unstruct] = (
                                grid_x = recon_p.grid_x,
                                grid_y = recon_p.grid_y,
                                surface = recon_p.surface,
                                values = vec(unstruct_summary.mean),
                                coordinates = coords_detected
                            )
                        else
                            plots_data[plot_key_unstruct] = (
                                values = vec(unstruct_summary.mean),
                                geometry = isnothing(polygons) ? centroids : polygons
                            )
                        end
                    end
                end

            elseif struct_type == :temporal
                plot_key = Symbol("temporal_$(key)")
                tm, tl, tu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                
                t_info = _resolve_temporal_coordinates(M, input_data, length(tm), key)
                raw_t = t_info.raw_t
                unique_t = t_info.unique_t
                t_label = t_info.label

                if length(unique_t) < length(tm) && length(unique_t) > 1
                    t_mean = [mean(tm[raw_t .== t]) for t in unique_t]
                    t_low = [mean(tl[raw_t .== t]) for t in unique_t]
                    t_up = [mean(tu[raw_t .== t]) for t in unique_t]
                    p_temp = timeseries_ci(unique_t, t_mean, t_low, t_up; color=:royalblue,
                        title="Temporal Trend: $key", xlabel=t_label)
                    plots[plot_key] = p_temp
                    plots[:temporal] = p_temp
                    plots_data[plot_key] = (time=unique_t, mean=t_mean, lower=t_low, upper=t_up)
                    plots_data[:temporal] = plots_data[plot_key]
                else
                    p_order = sortperm(raw_t)
                    p_temp = timeseries_ci(raw_t[p_order], tm[p_order], tl[p_order],
                        tu[p_order]; color=:royalblue, title="Temporal Trend: $key",
                        xlabel=t_label)
                    plots[plot_key] = p_temp
                    plots[:temporal] = p_temp
                    plots_data[plot_key] = (time=raw_t[p_order], mean=tm[p_order],
                        lower=tl[p_order], upper=tu[p_order])
                    plots_data[:temporal] = plots_data[plot_key]
                end

            elseif struct_type == :seasonal
                plot_key = Symbol("seasonal_$(key)")
                um, ul, uu = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)
                u_info = _resolve_temporal_coordinates(M, input_data, length(um), key)
                raw_u = u_info.raw_t
                unique_u = u_info.unique_t
                u_label = "Period"

                if length(unique_u) < length(um) && length(unique_u) > 1
                    u_mean = [mean(um[raw_u .== u]) for u in unique_u]
                    u_low = [mean(ul[raw_u .== u]) for u in unique_u]
                    u_up = [mean(uu[raw_u .== u]) for u in unique_u]
                    p_seas = timeseries_ci(unique_u, u_mean, u_low, u_up; color=:forestgreen,
                        title="Seasonal Component: $key", xlabel=u_label)
                    plots[plot_key] = p_seas
                    plots[:seasonal] = p_seas
                    plots_data[plot_key] = (period=unique_u, mean=u_mean, lower=u_low, upper=u_up)
                    plots_data[:seasonal] = plots_data[plot_key]
                else
                    p_order = sortperm(raw_u)
                    p_seas = timeseries_ci(raw_u[p_order], um[p_order], ul[p_order],
                        uu[p_order]; color=:forestgreen, title="Seasonal Component: $key",
                        xlabel=u_label)
                    plots[plot_key] = p_seas
                    plots[:seasonal] = p_seas
                    plots_data[plot_key] = (period=raw_u[p_order], mean=um[p_order],
                        lower=ul[p_order], upper=uu[p_order])
                    plots_data[:seasonal] = plots_data[plot_key]
                end

            elseif struct_type == :smooth
                vars = !isnothing(spec) ? get(spec.params, :positional_args,
                    [string(key)]) : [string(key)]
                cov_df = !isnothing(input_data)&&
                    input_data isa DataFrame ? input_data : (haskey(M, :data)&&
                    !isnothing(M.data) && M.data isa DataFrame ? M.data : nothing)

                if length(vars) == 1 # 1D smooth
                    var_sym = Symbol(vars[1])
                    cov_data = nothing
                    if !isnothing(cov_df) && hasproperty(cov_df, var_sym)
                        cov_data = cov_df[!, var_sym]
                    elseif !isnothing(M) && hasproperty(M, :domain_values) && haskey(M.domain_values, var_sym)
                        cov_data = M.domain_values[var_sym]
                    elseif !isnothing(M) && hasproperty(M, Symbol("$(var_sym)_values"))
                        cov_data = getproperty(M, Symbol("$(var_sym)_values"))
                    end

                    sm, sl, su = vec(main_effect_summary.mean), vec(main_effect_summary.lower), vec(main_effect_summary.upper)

                    if !isnothing(cov_data) && !isempty(cov_data)
                        if length(cov_data) == length(sm)
                            unique_cov = sort(unique(cov_data))
                            
                            # If there are duplicate covariate values (e.g. repeated space-time observations)
                            if length(unique_cov) < length(sm) && length(unique_cov) > 1
                                sm_mean = [mean(sm[cov_data .== x]) for x in unique_cov]
                                sm_low  = [mean(sl[cov_data .== x]) for x in unique_cov]
                                sm_up   = [mean(su[cov_data .== x]) for x in unique_cov]
                                p_sm = timeseries_ci(unique_cov, sm_mean, sm_low, sm_up;
                                    color=:darkorange, title="Smooth Effect: $var_sym",
                                    xlabel=string(var_sym))
                                smooth_effects_plots[var_sym] = p_sm
                                smooth_effects_plots_data[var_sym] = (covariate_values=unique_cov,
                                    mean=sm_mean, lower=sm_low, upper=sm_up)
                            else
                                p_order = sortperm(cov_data)
                                p_sm = timeseries_ci(cov_data[p_order], sm[p_order],
                                    sl[p_order], su[p_order]; color=:darkorange,
                                    title="Smooth Effect: $var_sym", xlabel=string(var_sym))
                                smooth_effects_plots[var_sym] = p_sm
                                smooth_effects_plots_data[var_sym] = (covariate_values=cov_data[p_order],
                                    mean=sm[p_order], lower=sl[p_order], upper=su[p_order])
                            end
                        else
                            # sm represents smooth curve across knots or discretized bins
                            u_cov = sort(unique(cov_data))
                            cov_grid = if length(u_cov) == length(sm)
                                u_cov
                            elseif length(cov_data) > 1
                                collect(range(minimum(cov_data), maximum(cov_data), length=length(sm)))
                            else
                                collect(1:length(sm))
                            end
                            p_sm = timeseries_ci(cov_grid, sm, sl, su;
                                color=:darkorange, title="Smooth Effect: $var_sym",
                                xlabel=string(var_sym))
                            smooth_effects_plots[var_sym] = p_sm
                            smooth_effects_plots_data[var_sym] = (covariate_values=cov_grid,
                                mean=sm, lower=sl, upper=su)
                        end
                    else
                        p_sm = timeseries_ci(1:length(sm), sm, sl, su;
                            color=:darkorange, title="Smooth Effect: $var_sym", xlabel=string(var_sym))
                        smooth_effects_plots[var_sym] = p_sm
                        smooth_effects_plots_data[var_sym] = (covariate_values=1:length(sm),
                            mean=sm, lower=sl, upper=su)
                    end
                    plots[Symbol("smooth_$(var_sym)")] = p_sm
                    if !haskey(plots, :smooth)
                        plots[:smooth] = p_sm
                    end
                elseif length(vars) == 2 # 2D smooth (interaction or continuous spatial)
                    var1_sym, var2_sym = Symbol(vars[1]), Symbol(vars[2])
                    plot_key = Symbol("$(var1_sym)_$(var2_sym)")
                    if !isnothing(cov_df) && hasproperty(cov_df,
                        var1_sym) && hasproperty(cov_df, var2_sym)
                        cond_preds_res = !isnothing(model_obj) && !isnothing(chain) ?
                            _generate_conditional_predictions(model_obj, chain, M,
                                var1_sym, second_cov=var2_sym) : nothing
                        if !isnothing(cond_preds_res)
                            cond_preds, range1, range2 = cond_preds_res
                            cond_summary = is_mv ? cond_preds[outcome] : cond_preds
                            grid_mean = reshape(vec(cond_summary.mean), length(range1), length(range2))
                            
                            p_2d = Plots.heatmap(range1, range2, grid_mean', title="2D Smooth Effect: $(var1_sym) & $(var2_sym)", xlabel=string(var1_sym), ylabel=string(var2_sym), c=:viridis, aspect_ratio=:equal)
                            smooth_effects_plots[plot_key] = p_2d
                            smooth_effects_plots_data[plot_key] = (x=range1, y=range2, z=grid_mean)
                            plots[Symbol("smooth_$(plot_key)")] = p_2d
                        elseif length(main_effect_summary.mean) == nrow(cov_df)
                            c_2d = Matrix{Float64}(cov_df[!, [var1_sym, var2_sym]])
                            p_2d = plot_spatial_surface(
                                c_2d, vec(main_effect_summary.mean);
                                title="2D Smooth Effect: $(var1_sym) & $(var2_sym)",
                                colorbar_title=string(key)
                            )
                            smooth_effects_plots[plot_key] = p_2d
                            recon_2d = _reconstruct_spatial_surface(c_2d, vec(main_effect_summary.mean))
                            smooth_effects_plots_data[plot_key] = (x=recon_2d.grid_x, y=recon_2d.grid_y, z=recon_2d.surface)
                            plots[Symbol("smooth_$(plot_key)")] = p_2d
                        end
                    end
                end
            end
        end
    end
    if !isempty(smooth_effects_plots)
        plots[:smooth_effects] = smooth_effects_plots
        plots_data[:smooth_effects] = smooth_effects_plots_data
    end

    # --- 6. Spatiotemporal Interaction Effects ---
    if !isnothing(effects) && hasproperty(effects,
        :st_interaction) && !isnothing(effects.st_interaction)
        st_summary = is_mv ? effects.st_interaction[outcome] : effects.st_interaction
        if hasproperty(st_summary, :mean) && !all(iszero, st_summary.mean)
            if haskey(M, :s_N) && haskey(M, :t_N)
                st_mean_grid = reshape(vec(st_summary.mean), M.s_N, M.t_N)
                p_st_heatmap = Plots.heatmap(1:M.t_N, 1:M.s_N, st_mean_grid,
                    title="Spatiotemporal Interaction", xlabel="Time Index",
                    ylabel="Spatial Unit Index", c=:viridis, legend=false)
                plots[:st_interaction_heatmap] = p_st_heatmap
                plots_data[:st_interaction_heatmap] = (time_idx=1:M.t_N, space_idx=1:M.s_N,
                    mean_effect=st_mean_grid)
            end
        end
    end

    # --- 6b. Spatiotemporal Trajectory Curves & Spacetime Map Slice ---
    if !isnothing(pred_denoised)
        pred_summary = is_mv ? pred_denoised[outcome] : pred_denoised
        if hasproperty(pred_summary, :mean)
            y_p_all = vec(pred_summary.mean)
            N_all = length(y_p_all)

            s_vec_st = if haskey(M, :s_idx) && length(M.s_idx) == N_all
                M.s_idx
            elseif !isnothing(input_data) && hasproperty(input_data,
                :s_idx) && length(input_data.s_idx) == N_all
                input_data.s_idx
            elseif !isnothing(input_data)
                det_s = _detect_spatial_unit_column(input_data; allow_nothing=true)
                if !isnothing(det_s) && length(input_data[!, det_s]) == N_all
                    input_data[!, det_s]
                elseif haskey(M, :s_N) && haskey(M, :t_N) && N_all == M.s_N * M.t_N
                    repeat(1:M.s_N, inner=M.t_N)
                else
                    nothing
                end
            elseif haskey(M, :s_N) && haskey(M, :t_N) && N_all == M.s_N * M.t_N
                repeat(1:M.s_N, inner=M.t_N)
            else
                nothing
            end

            t_info_st = _resolve_temporal_coordinates(M, input_data, N_all, haskey(M,
                :t_idx_var) ? M.t_idx_var : nothing)
            t_vec_st = t_info_st.raw_t
            unique_t = t_info_st.unique_t
            t_label = t_info_st.label

            if !isnothing(s_vec_st) && length(unique_t) > 1
                s_vec = s_vec_st
                t_vec = t_vec_st
                unique_s = sort(unique(s_vec))

                if length(unique_s) > 1
                    # 1. Multi-unit Timeseries
                    p_temporal_curves = Plots.plot(title="Temporal Fitted Trajectories across Units", xlabel=t_label, ylabel="Fitted Value", legend=false)
                    for u in unique_s
                        mask_u = s_vec .== u
                        if any(mask_u)
                            t_u = t_vec[mask_u]
                            p_order = sortperm(t_u)
                            Plots.plot!(p_temporal_curves, t_u[p_order],
                                y_p_all[mask_u][p_order], color=:royalblue, alpha=0.35, lw=1.2)
                        end
                    end
                    plots[:temporal_trajectories] = p_temporal_curves
                    plots_data[:temporal_trajectories] = (spatial_units=unique_s,
                        time_points=unique_t)

                    if !haskey(plots, :temporal)
                        plots[:temporal] = p_temporal_curves
                        plots_data[:temporal] = (spatial_units=unique_s, time_points=unique_t)
                    end
                end

                # 2. Spacetime Predictions Maps across ALL time slices
                n_units = !isnothing(polygons) ? length(polygons) : (!isnothing(centroids) ? length(centroids) : length(unique_s))
                
                valid_y_p = filter(!isnan, y_p_all)
                clims_global = !isempty(valid_y_p) ? (minimum(valid_y_p), maximum(valid_y_p)) : nothing

                st_slice_plots = Plots.Plot[]
                st_slice_data = Dict{Any, Vector{Float64}}()

                for t_val in unique_t
                    mask_t = t_vec .== t_val
                    s_t = s_vec[mask_t]
                    y_p_t = y_p_all[mask_t]

                    unit_preds_t = zeros(Float64, n_units)
                    for i in 1:n_units
                        match_idx = findall(s_t .== i)
                        if !isempty(match_idx)
                            unit_preds_t[i] = mean(y_p_t[match_idx])
                        else
                            match_all = findall(s_vec .== i)
                            unit_preds_t[i] = !isempty(match_all) ? mean(y_p_all[match_all]) : 0.0
                        end
                    end
                    st_slice_data[t_val] = unit_preds_t

                    title_slice = "$t_label = $t_val"
                    p_slice = if !isnothing(clims_global)
                        _render_spatial_effect((mean=unit_preds_t, ), title_slice, polygons,
                            centroids; clims=clims_global)
                    else
                        _render_spatial_effect((mean=unit_preds_t, ), title_slice, polygons,
                            centroids)
                    end
                    if !isnothing(p_slice)
                        push!(st_slice_plots, p_slice)
                    end
                end

                if !isempty(st_slice_plots)
                    T = length(st_slice_plots)
                    if T == 1
                        plots[:spacetime_predictions] = st_slice_plots[1]
                    else
                        ncols = min(4, T)
                        nrows = ceil(Int, T / ncols)
                        p_all_st = Plots.plot(st_slice_plots..., layout=(nrows, ncols),
                            size=(360 * ncols, 320 * nrows),
                            plot_title="Spacetime Predictions Across Time Slices")
                        plots[:spacetime_predictions] = p_all_st
                        
                        slice_dict = Dict{Symbol, Any}()
                        for (idx, t_val) in enumerate(unique_t)
                            if idx <= length(st_slice_plots)
                                slice_dict[Symbol("t_$(t_val)")] = st_slice_plots[idx]
                            end
                        end
                        plots[:spacetime_slices] = NamedTuple(slice_dict)
                    end
                    plots_data[:spacetime_predictions] = (time_points=unique_t,
                        predictions_by_time=st_slice_data,
                        geometry=isnothing(polygons) ? centroids : polygons)
                end
            end
        end
    end

    # --- 7. Spatially Varying Coefficients (SVC) Plots ---
    if !isnothing(effects) && haskey(M, :components)
        for spec in M.components
            if spec.structure == :svc
                key = spec.key
                if !haskey(effects, key) || (isnothing(polygons) && isnothing(centroids))
                    continue
                end
                
                svc_effect_summary = is_mv ? effects[key].structured[outcome] : effects[key].structured
                if !isnothing(svc_effect_summary) && hasproperty(svc_effect_summary, :mean)
                    plot_key = Symbol("svc_$(key)")
                    p_svc = _render_spatial_effect(svc_effect_summary, "SVC Effect: $(key)",
                        polygons, centroids)
                    if !isnothing(p_svc)
                        plots[plot_key] = p_svc
                        plots_data[plot_key] = (values=vec(svc_effect_summary.mean), geometry=isnothing(polygons) ? centroids : polygons)
                    end
                end
            end
        end
    end

    # --- 8. Hierarchical Effects (Mixed Effects) Plots ---
    if !isnothing(effects) && hasproperty(effects,
        :mixed_effects) && !isnothing(effects.mixed_effects)
        mixed_plots = Dict{Symbol, Any}()
        mixed_plots_data = Dict{Symbol, Any}()
        for (key, effect_summary) in pairs(effects.mixed_effects)
            grouping_covariate = Symbol(effect_summary.grouping_covariate)
            group_levels = (haskey(M, :data) && !isnothing(M.data) && hasproperty(M.data,
                grouping_covariate)) ? unique(M.data[!, grouping_covariate]) : nothing

            summaries_to_plot = is_mv ? effect_summary.summaries[outcome] : effect_summary.summaries

            for (term_name, summary) in pairs(summaries_to_plot)
                if hasproperty(summary, :mean) && !all(iszero, summary.mean)
                    means = vec(summary.mean)
                    lowers = vec(summary.lower)
                    uppers = vec(summary.upper)
                    n_levels = length(means) 
                    
                    y_ticks_labels = isnothing(group_levels) || length(group_levels) != n_levels ? ["Level $i" for i in 1:n_levels] : string.(group_levels)
                    p_title = "Mixed Effect: $(term_name) | $(grouping_covariate)"
                    
                    p_forest = Plots.scatter(means, 1:n_levels, xerror=(means .- lowers,
                        uppers .- means), yticks=(1:n_levels, y_ticks_labels), title=p_title,
                        xlabel="Effect Size", markersize=4, color=:black, legend=false,
                        yflip=true)
                    Plots.vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                    
                    plot_key = Symbol("$(key)_$(term_name)")
                    mixed_plots[plot_key] = p_forest
                    mixed_plots_data[plot_key] = (group_levels=y_ticks_labels, mean=means,
                        lower=lowers, upper=uppers)
                end
            end
        end
        if !isempty(mixed_plots)
            plots[:mixed_effects] = mixed_plots
            plots_data[:mixed_effects] = mixed_plots_data
        end
    end


    return (plots=NamedTuple(plots), plots_data=NamedTuple(plots_data))
end

"""
    save_plots(plots_obj, save_dir::AbstractString; prefix::String="", fmt::String="png",
      dpi::Integer=150)

Saves all plots in a `plots` NamedTuple/Dict or `bstm_plots` result to `save_dir`.
Automatically saves `Plots.Plot` objects as `fmt` (default: `"png"`) and `LeafletMap` objects as `.html`.
"""
function save_plots(plots_obj, save_dir::AbstractString; prefix::String="", fmt::String="png",
    dpi::Integer=150)
    if !isdir(save_dir)
        mkpath(save_dir)
    end
    
    plots_to_save = hasproperty(plots_obj, :plots) ? plots_obj.plots : plots_obj
    saved_paths = String[]
    clean_fmt = lowercase(string(lstrip(fmt, '.')))

    for (k, p) in pairs(plots_to_save)
        if p isa LeafletMap
            fn = isempty(prefix) ? "$(k).html" : "$(prefix)_$(k).html"
            out_path = joinpath(save_dir, fn)
            save_html(p, out_path)
            push!(saved_paths, out_path)
        elseif p isa Plots.Plot
            fn = isempty(prefix) ? "$(k).$(clean_fmt)" : "$(prefix)_$(k).$(clean_fmt)"
            out_path = joinpath(save_dir, fn)
            save_plot(p, out_path; fmt=nothing, dpi=dpi)
            push!(saved_paths, out_path)
        elseif p isa NamedTuple || p isa Dict
            for (sub_k, sub_p) in pairs(p)
                if sub_p isa LeafletMap
                    fn = isempty(prefix) ? "$(k)_$(sub_k).html" : "$(prefix)_$(k)_$(sub_k).html"
                    out_path = joinpath(save_dir, fn)
                    save_html(sub_p, out_path)
                    push!(saved_paths, out_path)
                elseif sub_p isa Plots.Plot
                    fn = isempty(prefix) ? "$(k)_$(sub_k).$(clean_fmt)" : "$(prefix)_$(k)_$(sub_k).$(clean_fmt)"
                    out_path = joinpath(save_dir, fn)
                    save_plot(sub_p, out_path; fmt=nothing, dpi=dpi)
                    push!(saved_paths, out_path)
                end
            end
        end
    end
    @info "Saved $(length(saved_paths)) plot(s) to '$save_dir'."
    return saved_paths
end

"""
    bstm_plots(res::NamedTuple; au=nothing, data=nothing, outcome=1, save_dir=nothing,
      save_prefix="", fmt="png", dpi=150)
    bstm_plots(res::NamedTuple, data::Any; au=nothing, outcome=1, save_dir=nothing,
      save_prefix="", fmt="png", dpi=150)
    bstm_plots(model::DynamicPPL.Model, chain, res; au=nothing, data=nothing, outcome=1,
      save_dir=nothing, save_prefix="", fmt="png", dpi=150)
    bstm_plots(model::DynamicPPL.Model, chain; au=nothing, data=nothing, outcome=1,
      alpha=0.05, save_dir=nothing, save_prefix="", fmt="png", dpi=150)

Generates diagnostic, spatial, temporal, fixed, and conditional effect plots from model results.
Optionally saves all generated figures to `save_dir`.
"""
function bstm_plots(res::NamedTuple; au=nothing, data=nothing, outcome=1, save_dir=nothing,
    save_prefix="", fmt="png", dpi=150)
    model = get(res, :model, nothing)
    chain = get(res, :chain, nothing)
    M = hasproperty(res, :M) && !isnothing(res.M) ? res.M :
        (!isnothing(model) && hasproperty(model, :args) && hasproperty(model.args, :M) ? model.args.M : NamedTuple())
    au_to_use = !isnothing(au) ? au : get(res, :au, nothing)
    data_to_use = !isnothing(data) ? data :
        (hasproperty(res, :data) && !isnothing(res.data) ? res.data :
        (!isnothing(model) && hasproperty(model, :args) && hasproperty(model.args, :M) ? get(model.args.M, :data, nothing) : nothing))
    
    plot_res = _bstm_plots_impl(model, chain, res, M; au=au_to_use, data=data_to_use,
        outcome=outcome)
    
    if !isnothing(save_dir)
        save_plots(plot_res.plots, save_dir; prefix=save_prefix, fmt=fmt, dpi=dpi)
    end
    
    return plot_res
end

function bstm_plots(res::NamedTuple, data::Any; au=nothing, outcome=1, save_dir=nothing,
    save_prefix="", fmt="png", dpi=150)
    return bstm_plots(res; data=data, au=au, outcome=outcome, save_dir=save_dir,
        save_prefix=save_prefix, fmt=fmt, dpi=dpi)
end

function bstm_plots(model::DynamicPPL.Model, chain, res; au=nothing, data=nothing, outcome=1,
    save_dir=nothing, save_prefix="", fmt="png", dpi=150)
    M = model.args.M
    au_to_use = !isnothing(au) ? au : (hasproperty(res, :au) ? res.au : nothing)
    data_to_use = !isnothing(data) ? data : get(M, :data, nothing)
    
    plot_res = _bstm_plots_impl(model, chain, res, M; au=au_to_use, data=data_to_use,
        outcome=outcome)
    
    if !isnothing(save_dir)
        save_plots(plot_res.plots, save_dir; prefix=save_prefix, fmt=fmt, dpi=dpi)
    end
    
    return plot_res
end

function bstm_plots(model::DynamicPPL.Model, chain; au=nothing, data=nothing, outcome=1,
    alpha=0.05, save_dir=nothing, save_prefix="", fmt="png", dpi=150)
    res = model_results_comprehensive(model, chain; data=data, alpha=alpha)
    return bstm_plots(model, chain, res; au=au, data=data, outcome=outcome, save_dir=save_dir,
        save_prefix=save_prefix, fmt=fmt, dpi=dpi)
end

function _TethysBayes_plots_impl(model_obj, chain, res, M; au=nothing, data=nothing, outcome=1)
    return _bstm_plots_impl(model_obj, chain, res, M; au=au, data=data, outcome=outcome)
end

function TethysBayes_plots(args...; kwargs...)
    return bstm_plots(args...; kwargs...)
end

"""
    model_results_plots(res)

Convenience utility to display all plots generated by `bstm_plots`.
"""
function model_results_plots(res)
    plots_obj = hasproperty(res, :plots) ? res.plots : res
    if isempty(plots_obj)
        println("No plots found in the results object.") 
        return
    end

    println("--- Displaying Generated Plots ---")
    for (plot_name, plot_obj) in pairs(plots_obj)
        if plot_obj isa Dict || plot_obj isa NamedTuple
            for (sub_name, sub_plot) in pairs(plot_obj)
                println("--- Plot: $plot_name -> $sub_name ---")
                display(sub_plot)
            end
        else
            println("--- Plot: $plot_name ---")
            display(plot_obj)
        end
    end
    println("--- End of Plots ---")
end


# -----------------------------------------------------------------------------
# Section 10: Population Attributable Risk Plots
# -----------------------------------------------------------------------------

"""
    par_credible_interval_plot(par_result::NamedTuple; title="PAF Estimate")
"""
function par_credible_interval_plot(par_result::NamedTuple; title="PAF Estimate")
    y_pos = 1
    label_text = "$(par_result.covariate) ($(par_result.family), ref=$(par_result.reference_population))"
    p = Plots.plot(title=title, xlabel="Population Attributable Fraction (PAF)", ylabel="", legend=:topright, size=(700, 300))
    Plots.plot!(p, [par_result.paf_lower, par_result.paf_upper], [y_pos, y_pos], linewidth=3, color=:steelblue, label="95% CI")
    Plots.scatter!(p, [par_result.paf_mean], [y_pos], markersize=8, color=:darkblue, label="Posterior Mean", markerstrokewidth=0)
    Plots.yticks!(p, [y_pos], [label_text])
    return p
end

"""
    par_forest_plot(par_results::Union{Dict, NamedTuple}; title="Forest Plot: PAF by Risk Factor")
"""
function par_forest_plot(par_results::Union{Dict, NamedTuple}; title="Forest Plot: PAF by Risk Factor")
    results_dict = par_results isa NamedTuple ? Dict(String(k) => v for (k, v) in pairs(par_results)) : par_results
    valid_results = Dict(k => v for (k, v) in results_dict if v isa NamedTuple && !isnan(v.paf_mean))
    isempty(valid_results) && return nothing
    cov_names = sort(collect(keys(valid_results)))
    n_cov = length(cov_names)
    means = [valid_results[c].paf_mean for c in cov_names]
    lowers = [valid_results[c].paf_lower for c in cov_names]
    uppers = [valid_results[c].paf_upper for c in cov_names]
    p = Plots.plot(title=title, xlabel="Population Attributable Fraction (PAF)", ylabel="", legend=false, size=(700, 300 + 50 * n_cov))
    for (i, cov) in enumerate(reverse(cov_names))
        y = n_cov + 1 - i
        Plots.plot!(p, [lowers[n_cov + 1 - i], uppers[n_cov + 1 - i]], [y, y], linewidth=2, color=:steelblue, label="")
        Plots.scatter!(p, [means[n_cov + 1 - i]], [y], markersize=7, color=:darkblue, label="", markerstrokewidth=0)
    end
    Plots.yticks!(p, 1:n_cov, reverse(cov_names))
    return p
end


# -----------------------------------------------------------------------------
# Section 11: Bayesian Diagnostics & Advanced Spatiotemporal Plots
# -----------------------------------------------------------------------------

"""
    plot_ppc(y_observed::AbstractVector, y_rep_matrix::AbstractMatrix; num_draws::Int=50, title="Posterior Predictive Check", kwargs...)
Overlays the density of the observed data against draws from the posterior predictive distribution.
"""
function plot_ppc(y_observed::AbstractVector, y_rep_matrix::AbstractMatrix; num_draws::Int=50, title="Posterior Predictive Check", kwargs...)
    p = Plots.plot(title=title, xlabel="Value", ylabel="Density", legend=true; kwargs...)
    n_samples = size(y_rep_matrix, 2)
    draws = min(n_samples, num_draws)
    indices = sample(1:n_samples, draws, replace=false)
    for i in indices
        StatsPlots.density!(p, y_rep_matrix[:, i], color=:lightblue, alpha=0.3, linewidth=1, label= i == indices[1] ? "y_rep" : "")
    end
    StatsPlots.density!(p, y_observed, color=:black, linewidth=2, label="Observed y")
    return p
end

"""
    plot_prior_vs_posterior(prior_dist::Distribution, posterior_samples::AbstractVector; title="Prior vs Posterior", param_name="Parameter", kwargs...)
Visualizes how much the posterior distribution shifted from the prior.
"""
function plot_prior_vs_posterior(prior_dist::Distribution, posterior_samples::AbstractVector; title="Prior vs Posterior", param_name="Parameter", kwargs...)
    p = Plots.plot(title=title, xlabel=param_name, ylabel="Density"; kwargs...)
    StatsPlots.density!(p, posterior_samples, color=:steelblue, linewidth=2, label="Posterior", fill=(0, 0.3, :steelblue))
    Plots.plot!(p, prior_dist, color=:darkred, linewidth=2, label="Prior", linestyle=:dash)
    return p
end

"""
    plot_marginal_effects(x_range::AbstractVector, mean_pred::AbstractVector, lower_pred::AbstractVector, upper_pred::AbstractVector; covariate_name="Covariate", outcome_name="Response", title="Marginal Effect", kwargs...)
Plots the predicted outcome given a covariate, with 95% credible intervals.
"""
function plot_marginal_effects(x_range::AbstractVector, mean_pred::AbstractVector, lower_pred::AbstractVector, upper_pred::AbstractVector; covariate_name="Covariate", outcome_name="Response", title="Marginal Effect", kwargs...)
    p = Plots.plot(title=title, xlabel=covariate_name, ylabel=outcome_name, legend=false; kwargs...)
    Plots.plot!(p, x_range, mean_pred, ribbon=(mean_pred .- lower_pred, upper_pred .- mean_pred), fillalpha=0.3, color=:royalblue, linewidth=2)
    return p
end

"""
    plot_spatial_residuals(polygons::AbstractVector, observed::AbstractVector, predicted::AbstractVector; title="Spatial Residuals", mode=:static, kwargs...)
Choropleth mapping of the residuals (observed - predicted) using a diverging colormap.
"""
function plot_spatial_residuals(polygons::AbstractVector, observed::AbstractVector, predicted::AbstractVector; title="Spatial Residuals", mode=:static, kwargs...)
    residuals = observed .- predicted
    if mode == :leaflet
        return leaflet_choropleth(polygons, residuals; title=title, colorscheme=:RdBu_11, kwargs...)
    else
        return choropleth(polygons, residuals; title=title, colorscheme=:RdBu_11, kwargs...)
    end
end

"""
    plot_spatiotemporal_facets(polygons::AbstractVector, values_matrix::AbstractMatrix, time_labels::AbstractVector; title="Spatiotemporal Evolution", colorscheme=:viridis, kwargs...)
Generates a grid of small multiples for spatial evolution over time.
"""
function plot_spatiotemporal_facets(polygons::AbstractVector, values_matrix::AbstractMatrix, time_labels::AbstractVector; title="Spatiotemporal Evolution", colorscheme=:viridis, kwargs...)
    n_times = length(time_labels)
    plots = Plots.Plot[]
    v_min = minimum(skipmissing(values_matrix))
    v_max = maximum(skipmissing(values_matrix))
    for t in 1:n_times
        vals = values_matrix[:, t]
        # Note: colorbar is only rendered on the last subplot to reduce clutter
        p_t = choropleth(polygons, vals; title=string(time_labels[t]), colorscheme=colorscheme, clims=(v_min, v_max), colorbar=(t==n_times), kwargs...)
        push!(plots, p_t)
    end
    r = ceil(Int, sqrt(n_times))
    c = ceil(Int, n_times / r)
    layout = Plots.grid(r, c)
    return Plots.plot(plots..., layout=layout, plot_title=title, size=(r*300, c*300))
end


