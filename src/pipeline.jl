"""
    pipeline.jl

Declarative Modular DAG Pipeline Orchestrator (`TethysBayes_pipeline`), Multi-Tier
Spatial Network Resharding, Uncertainty Propagation (EIV), and Relational
Database Harmonization for Bayesian Spatio-Temporal Models.

# Mathematical Foundation & Multi-Tier Resharding
In hierarchical ecological workflows (e.g. Bathymetry -> Substrate -> Oceanography
-> Multi-Species -> Focal Biomass), different tiers represent physical or biological
processes operating at distinct observational scales, data densities, and geometries:
- **Tier 1 (Continuous Bathymetry)**: Point soundings modeled via continuous `RFF` or `SpectralGP`.
- **Tier 2 (Sediment Properties)**: Point core grabs discretized onto a coarse graph \$G_2 = (V_2, W_2)\$.
- **Tier 3 (Hydrography)**: Space-time CTD ocean casts on regular grid \$G_3 = (V_3, W_3)\$.
- **Tier 4 (Multispecies Assemblage)**: Trawl surveys on stratified stratum mesh \$G_4 = (V_4, W_4)\$.
- **Tier 5 (Target Species / Focal Assessment)**: Management area or fine master mesh \$G_{\\text{master}}\$.

### Information Transfer & Network Resharding:
1. **Continuous-to-Discrete Evaluation**:
   For continuous models (\$f(\\mathbf{s})\$), the posterior mean and variance are evaluated
   directly at the target network's coordinates \$\\mathbf{s}_j \\in V_{\\text{dest}}\$:
   \$\\hat{z}(\\mathbf{s}_j) = \\mathbb{E}[f(\\mathbf{s}_j) \\mid \\mathcal{D}], \\quad \\hat{\\sigma}^2(\\mathbf{s}_j) = \\text{Var}(f(\\mathbf{s}_j) \\mid \\mathcal{D})\$
2. **Discrete Mesh-to-Mesh Transfer (Area / Distance Weighting)**:
   For fields defined on areal units \$V_{\\text{src}} = \\{A_1, \\dots, A_m\\}\$ transferred to
   \$V_{\\text{dest}} = \\{B_1, \\dots, B_n\\}\$, the transfer matrix \$P \\in \\mathbb{R}^{n \\times m}\$ satisfies:
   \$P_{j, i} = \\frac{\\text{Area}(B_j \\cap A_i)}{\\text{Area}(B_j)}, \\quad \\sum_i P_{j, i} = 1\$
   \$\\mathbf{u}_{\\text{dest}} = P \\mathbf{u}_{\\text{src}}, \\quad \\text{Var}(\\mathbf{u}_{\\text{dest}}) = P \\text{Cov}(\\mathbf{u}_{\\text{src}}) P^\\top\$
3. **Master Network Harmonization**:
   Upon pipeline completion, all upstream physical, sediment, oceanographic, and community
   covariates are resharded and joined onto \$G_{\\text{master}}\$, populating DuckDB relational
   tables and exporting GIS GeoJSON files.

Version: v1.0.0
"""

# ==============================================================================
# SECTION 1: SPATIAL NETWORK RESHARDING & TRANSFER OPERATORS
# ==============================================================================

"""
    compute_network_transfer_matrix(au_src::NamedTuple, au_dest::NamedTuple)

Computes the spatial interpolation / resharding matrix \$P \\in \\mathbb{R}^{n_{\\text{dest}} \\times n_{\\text{src}}}\$
between two areal unit spatial partitions.

Uses polygon geometric area-overlap weighting via `LibGEOS.intersection` when polygons
are available, with automated coordinate alignment, AABB envelope pre-filtering, and
row-stochastic normalization. Falls back to k-nearest inverse-distance-weighted (IDW)
centroid mapping if geometries are absent.
"""
function compute_network_transfer_matrix(
    au_src::NamedTuple,
    au_dest::NamedTuple;
    method::Symbol = :area_weighted,
    kwargs...
)
    _extract_cents(au) = if hasproperty(au, :centroids) && !isnothing(au.centroids)
        au.centroids
    elseif hasproperty(au, :centroids_lonlat) && !isnothing(au.centroids_lonlat)
        au.centroids_lonlat
    elseif hasproperty(au, :centroids_km) && !isnothing(au.centroids_km)
        au.centroids_km
    else
        error("Areal unit partition has no centroids field.")
    end

    src_cents = _extract_cents(au_src)
    dest_cents = _extract_cents(au_dest)
    n_src = length(src_cents)
    n_dest = length(dest_cents)

    if n_src == n_dest && src_cents == dest_cents
        return spdiagm(0 => ones(Float64, n_dest))
    end

    # Determine coordinate scale (geographic degrees vs projected planar meters/km)
    is_geo_coords(pts) = !isempty(pts) && all(abs(c[1]) <= 180.5 && abs(c[2]) <= 90.5 for c in pts)
    src_is_geo = is_geo_coords(src_cents)

    resolve_au_geom(au, ref_geo) = begin
        if ref_geo && hasproperty(au, :polygons_lonlat) && !isempty(au.polygons_lonlat)
            cents = hasproperty(au, :centroids_lonlat) ? au.centroids_lonlat : _extract_cents(au)
            return (polygons = au.polygons_lonlat, centroids = cents)
        elseif !ref_geo && hasproperty(au, :polygons_km) && !isempty(au.polygons_km)
            cents = hasproperty(au, :centroids_km) ? au.centroids_km : _extract_cents(au)
            return (polygons = au.polygons_km, centroids = cents)
        elseif hasproperty(au, :polygons) && !isempty(au.polygons)
            cents = _extract_cents(au)
            return (polygons = au.polygons, centroids = cents)
        else
            return (polygons = Vector{Vector{Tuple{Float64, Float64}}}(), centroids = _extract_cents(au))
        end
    end

    geom_src = resolve_au_geom(au_src, src_is_geo)
    geom_dest = resolve_au_geom(au_dest, src_is_geo)

    has_polys = !isempty(geom_src.polygons) && !isempty(geom_dest.polygons) &&
                length(geom_src.polygons) == n_src && length(geom_dest.polygons) == n_dest

    if has_polys && method != :centroid_idw && method != :idw
        try
            # Helper to create valid closed LibGEOS Polygon
            function _to_libgeos_valid_polygon(poly_coords)
                if isempty(poly_coords) || length(poly_coords) < 3
                    return nothing
                end
                pts = [[Float64(pt[1]), Float64(pt[2])] for pt in poly_coords if !isnan(pt[1]) && !isnan(pt[2])]
                if length(pts) < 3
                    return nothing
                end
                if pts[1] != pts[end]
                    push!(pts, pts[1])
                end
                if length(pts) < 4
                    return nothing
                end
                try
                    lg_p = LibGEOS.Polygon([pts])
                    if !LibGEOS.isValid(lg_p)
                        lg_p = LibGEOS.buffer(lg_p, 0.0)
                    end
                    return lg_p
                catch
                    return nothing
                end
            end

            src_lg_polys = [_to_libgeos_valid_polygon(p) for p in geom_src.polygons]
            dest_lg_polys = [_to_libgeos_valid_polygon(p) for p in geom_dest.polygons]

            # Pre-compute Axis-Aligned Bounding Boxes (AABB) for fast candidate filtering
            src_bboxes = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, n_src)
            for i in 1:n_src
                p = geom_src.polygons[i]
                xs = [pt[1] for pt in p if !isnan(pt[1])]
                ys = [pt[2] for pt in p if !isnan(pt[2])]
                if isempty(xs) || isempty(ys)
                    src_bboxes[i] = (0.0, 0.0, 0.0, 0.0)
                else
                    src_bboxes[i] = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
                end
            end

            P = spzeros(Float64, n_dest, n_src)
            all_valid = true

            for j in 1:n_dest
                dest_p = dest_lg_polys[j]
                if dest_p === nothing
                    all_valid = false
                    break
                end
                dest_area = get_polygon_area(geom_dest.polygons[j])
                if dest_area <= 1e-9
                    continue
                end

                dp = geom_dest.polygons[j]
                d_xs = [pt[1] for pt in dp if !isnan(pt[1])]
                d_ys = [pt[2] for pt in dp if !isnan(pt[2])]
                if isempty(d_xs) || isempty(d_ys)
                    continue
                end
                d_min_x, d_max_x = minimum(d_xs), maximum(d_xs)
                d_min_y, d_max_y = minimum(d_ys), maximum(d_ys)

                total_overlap = 0.0
                for i in 1:n_src
                    src_p = src_lg_polys[i]
                    src_p === nothing && continue

                    # AABB Disjoint Rejection Test
                    s_min_x, s_max_x, s_min_y, s_max_y = src_bboxes[i]
                    if d_min_x > s_max_x || d_max_x < s_min_x ||
                       d_min_y > s_max_y || d_max_y < s_min_y
                        continue
                    end

                    if LibGEOS.intersects(dest_p, src_p)
                        inter_geom = LibGEOS.intersection(dest_p, src_p)
                        if !LibGEOS.isEmpty(inter_geom)
                            inter_area = get_polygon_area(get_coords_from_geom(inter_geom))
                            if inter_area > 1e-9
                                w = inter_area / dest_area
                                P[j, i] = w
                                total_overlap += w
                            end
                        end
                    end
                end

                # Normalize row if overlapping area is detected
                if total_overlap > 1e-6
                    P[j, :] ./= total_overlap
                else
                    # Fallback for peripheral destination cells outside source envelope:
                    # Assign full mass to nearest source centroid
                    c_j = geom_dest.centroids[j]
                    dists = [hypot(c_j[1] - geom_src.centroids[i][1], c_j[2] - geom_src.centroids[i][2]) for i in 1:n_src]
                    nearest_i = argmin(dists)
                    P[j, nearest_i] = 1.0
                end
            end

            if all_valid && all(sum(P, dims=2) .> 0.99)
                return P
            end
        catch err
            # Fall back to centroid IDW if geometric intersection encounters unexpected errors
        end
    end

    # 2. Centroid Inverse Distance Weighting (k-d Tree Fallback)
    src_coords = hcat([[c[1], c[2]] for c in src_cents]...)
    dest_coords = hcat([[c[1], c[2]] for c in dest_cents]...)

    kdtree = KDTree(src_coords)
    k_nn = min(4, n_src)
    idxs, dists = knn(kdtree, dest_coords, k_nn, true)

    P = spzeros(Float64, n_dest, n_src)
    for j in 1:n_dest
        cur_idxs = idxs[j]
        cur_dists = dists[j]

        if cur_dists[1] <= 1e-6
            P[j, cur_idxs[1]] = 1.0
        else
            weights = 1.0 ./ (cur_dists .^ 2)
            weights ./= sum(weights)
            for (idx, w) in zip(cur_idxs, weights)
                P[j, idx] = w
            end
        end
    end

    return P
end

"""
    summarize_sample_matrix(samples::AbstractMatrix{<:Real}; alpha::Real=0.05)

Computes posterior empirical summary statistics (mean, median, std, lower/upper quantiles)
across Monte Carlo sample columns for each spatial unit row.
"""
function summarize_sample_matrix(samples::AbstractMatrix{<:Real}; alpha::Real=0.05)
    n_units, n_samples = size(samples)
    means = vec(Statistics.mean(samples, dims=2))
    stds = vec(Statistics.std(samples, dims=2))
    medians = [quantile(samples[i, :], 0.5) for i in 1:n_units]
    lowers = [quantile(samples[i, :], alpha / 2.0) for i in 1:n_units]
    uppers = [quantile(samples[i, :], 1.0 - alpha / 2.0) for i in 1:n_units]

    return (
        mean = means,
        median = medians,
        std = stds,
        lower = lowers,
        upper = uppers,
        samples = samples
    )
end

"""
    reshard_spatial_field(values::AbstractArray, au_src::NamedTuple, au_dest::NamedTuple)
    reshard_spatial_field(vectors::Tuple, au_src::NamedTuple, au_dest::NamedTuple)
    reshard_spatial_field(nt::NamedTuple, au_src::NamedTuple, au_dest::NamedTuple)
    reshard_spatial_field(summary::NamedTuple, au_src::NamedTuple, au_dest::NamedTuple; mode::Symbol=:samples, alpha::Real=0.05)

Reshards a spatial field from source network `au_src` to destination network `au_dest`.

# Mathematical Formulation
For intensive physical and environmental variables, the resharding operator evaluates:
```math
\\mathbf{u}_{\\text{dest}} = P \\mathbf{u}_{\\text{src}}
```
where \$P \\in \\mathbb{R}^{n_{\\text{dest}} \\times n_{\\text{src}}}\$ is the area-overlap
transfer matrix computed by `compute_network_transfer_matrix`.

# Multi-Dimensional Support:
- 1D vector: \$S_{\\text{src}} \\to S_{\\text{dest}}\$
- 2D matrix: \$S_{\\text{src}} \\times N_z \\to S_{\\text{dest}} \\times N_z\$
- 3D array: \$S_{\\text{src}} \\times N_z \\times N_t \\to S_{\\text{dest}} \\times N_z \\times N_t\$
- Vector tuples: e.g. `(u, v)` or `(u, v, w)` resharded simultaneously
- Field bundles: `NamedTuple` of fields resharded in a unified call
"""
function reshard_spatial_field(
    P::AbstractMatrix{<:Real},
    values::AbstractArray{<:Real}
)
    nd = ndims(values)
    if nd == 1
        return Vector{Float64}(P * values)
    elseif nd == 2
        return Matrix{Float64}(P * values)
    elseif nd == 3
        s1, s2, s3 = size(values)
        flat = reshape(values, s1, s2 * s3)
        res_flat = Matrix{Float64}(P * flat)
        return reshape(res_flat, size(P, 1), s2, s3)
    else
        error("Unsupported array dimensionality ($nd) for reshard_spatial_field.")
    end
end

function reshard_spatial_field(
    P::AbstractMatrix{<:Real},
    vectors::Tuple{Vararg{AbstractArray{<:Real}}}
)
    return map(arr -> reshard_spatial_field(P, arr), vectors)
end

function reshard_spatial_field(
    P::AbstractMatrix{<:Real},
    nt_or_summary::NamedTuple;
    mode::Symbol = :samples,
    alpha::Real = 0.05
)
    is_summary = hasproperty(nt_or_summary, :mean) && hasproperty(nt_or_summary, :std)

    if is_summary
        if mode in [:samples, :full_mc, :matrix] && hasproperty(nt_or_summary, :samples) &&
           !isnothing(nt_or_summary.samples) && nt_or_summary.samples isa AbstractMatrix
            U_dest = Matrix{Float64}(P * nt_or_summary.samples)
            return summarize_sample_matrix(U_dest; alpha=alpha)
        end

        mean_resharded = Vector{Float64}(P * nt_or_summary.mean)
        # Propagate the SD through the SAME aggregation that produced the mean.
        #
        # The mean is a weighted average of fine cells, so its variance is
        #     Var(P * x) = diag(P * Cov(x) * P')
        # and using only the marginal variances gives the diagonal approximation
        #     Var ~ diag(P * Diagonal(std.^2) * P')
        #
        # The previous code computed `sqrt.(P * (std.^2))`, which is wrong twice over: it
        # omits the trailing P' (so cross-covariances are ignored entirely) and it treats
        # the aggregation as a SUM rather than an average. For a coarse cell averaging k
        # independent fine cells the truth is sd/sqrt(k), and the old formula returned sd --
        # overstating by exactly sqrt(k). Measured: k=4 gave 1.0 where the truth is 0.5, a
        # 100% overstatement; k=100 would be 10x. See `scripts/_verify_ar1_reshard.jl`.
        #
        # The cross-covariance term is genuinely absent, so this remains an approximation
        # for a spatially correlated field -- exact under independence. The `:samples` path
        # above reshards actual draws and is exact either way.
        sd_resharded = Vector{Float64}(sqrt.(diag(
            Matrix{Float64}(P) * Diagonal(Float64.(nt_or_summary.std) .^ 2) * Matrix{Float64}(P)')))
        z_crit = quantile(Normal(0.0, 1.0), 1.0 - alpha / 2.0)
        lower_resharded = hasproperty(nt_or_summary, :lower) ?
            Vector{Float64}(P * nt_or_summary.lower) : (mean_resharded .- z_crit .* sd_resharded)
        upper_resharded = hasproperty(nt_or_summary, :upper) ?
            Vector{Float64}(P * nt_or_summary.upper) : (mean_resharded .+ z_crit .* sd_resharded)
        median_resharded = hasproperty(nt_or_summary, :median) ?
            Vector{Float64}(P * nt_or_summary.median) : mean_resharded

        return (
            mean = mean_resharded,
            median = median_resharded,
            std = sd_resharded,
            lower = lower_resharded,
            upper = upper_resharded,
            samples = nothing
        )
    else
        res_pairs = Pair{Symbol, Any}[]
        n_src_expected = size(P, 2)
        for (k, v) in pairs(nt_or_summary)
            if v isa AbstractArray{<:Real} && size(v, 1) == n_src_expected
                push!(res_pairs, k => reshard_spatial_field(P, v))
            elseif v isa Tuple && all(x -> x isa AbstractArray{<:Real} && size(x, 1) == n_src_expected, v)
                push!(res_pairs, k => map(x -> reshard_spatial_field(P, x), v))
            else
                push!(res_pairs, k => v)
            end
        end
        return NamedTuple(res_pairs)
    end
end

function reshard_spatial_field(
    values::AbstractArray{<:Real}, au_src::NamedTuple, au_dest::NamedTuple
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    return reshard_spatial_field(P, values)
end

function reshard_spatial_field(
    vectors::Tuple{Vararg{AbstractArray{<:Real}}},
    au_src::NamedTuple,
    au_dest::NamedTuple
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    return reshard_spatial_field(P, vectors)
end

function reshard_spatial_field(
    nt_or_summary::NamedTuple,
    au_src::NamedTuple,
    au_dest::NamedTuple;
    mode::Symbol = :samples,
    alpha::Real = 0.05
)
    P = compute_network_transfer_matrix(au_src, au_dest)
    return reshard_spatial_field(P, nt_or_summary; mode=mode, alpha=alpha)
end

# ==============================================================================
# SECTION 2: DECLARATIVE PIPELINE ORCHESTRATOR ENGINE
# ==============================================================================

"""
    PipelineTierSpec

Specification for a single hierarchical modeling tier in `TethysBayes_pipeline`.
"""
struct PipelineTierSpec
    name::Symbol
    formula::String
    data::DataFrame
    au::Union{NamedTuple, Nothing}
    W::Union{AbstractMatrix, Nothing}
    sampler::Any
    n_samples::Int
    derivatives::Vector{Symbol}
    radii::Vector{Float64}
    eiv_inputs::Vector{Symbol}
    options::Dict{Symbol, Any}
end

function PipelineTierSpec(
    name::Symbol, conf::NamedTuple
)
    formula = string(get(conf, :formula, ""))
    data = get(conf, :data, DataFrame())
    au = get(conf, :au, nothing)
    W = get(conf, :W, nothing)
    sampler = get(conf, :sampler, NUTS(20, 0.65))
    n_samples = get(conf, :n_samples, 50)
    derivatives = get(conf, :derivatives, Symbol[])
    radii = get(conf, :radii, Float64[10.0, 20.0])
    eiv_inputs = get(conf, :eiv_inputs, Symbol[])
    
    opts = Dict{Symbol, Any}()
    for (k, v) in pairs(conf)
        if !(k in [:formula, :data, :au, :W, :sampler, :n_samples, :derivatives, :radii, :eiv_inputs])
            opts[k] = v
        end
    end

    return PipelineTierSpec(
        name, formula, data, au, W, sampler, n_samples, derivatives, radii, eiv_inputs, opts
    )
end

"""
    PipelineResult

Complete execution artifacts, multi-tier models, chains, relational DuckDB database,
and harmonized master datasets produced by `TethysBayes_pipeline`.
"""
struct PipelineResult
    tier_names::Vector{Symbol}
    models::Dict{Symbol, DynamicPPL.Model}
    chains::Dict{Symbol, Any}
    results::Dict{Symbol, Any}
    derivatives::Dict{Symbol, Any}
    tier_networks::Dict{Symbol, Union{NamedTuple, Nothing}}
    master_au::NamedTuple
    master_summary::DataFrame
    duckdb_path::String
end

# ==============================================================================
# SECTION 3: PUBLIC API (`TethysBayes_pipeline`)
# ==============================================================================

"""
    TethysBayes_pipeline(tiers...; master_au=nothing, duckdb_path=":memory:", jld2_path=nothing, geojson_path=nothing, verbose=true)

Executes a multi-tier Bayesian Spatio-Temporal Model (TethysBayes) hierarchical pipeline
with automated network resharding, Errors-in-Variables (EIV) uncertainty propagation,
DuckDB relational persistence, and master network harmonization.

# Arguments
- `tiers...`: Pairs of `:tier_name => (formula=..., data=..., [au=...], [W=...], ...)`
- `master_au::NamedTuple`: The master spatial network / partition for final harmonization.
  If `nothing`, defaults to the network of the final tier.
- `duckdb_path::String`: File path to DuckDB database (or `":memory:"`).
- `jld2_path::Union{String, Nothing}`: Optional file path to persist complete pipeline state.
- `geojson_path::Union{String, Nothing}`: Optional file path to export harmonized master GIS GeoJSON.
- `verbose::Bool`: Whether to print progress diagnostics (default: `true`).

# Returns
A `PipelineResult` object containing all models, chains, derivatives, DuckDB tables, and
the unified master summary DataFrame.
"""
function TethysBayes_pipeline(
    tiers::Pair{Symbol, <:NamedTuple}...;
    master_au::Union{NamedTuple, Nothing}=nothing,
    duckdb_path::AbstractString=":memory:",
    jld2_path::Union{AbstractString, Nothing}=nothing,
    geojson_path::Union{AbstractString, Nothing}=nothing,
    verbose::Bool=true
)
    tier_specs = [PipelineTierSpec(p.first, p.second) for p in tiers]
    n_tiers = length(tier_specs)

    if n_tiers == 0
        error("TethysBayes_pipeline requires at least one modeling tier.")
    end

    models = Dict{Symbol, DynamicPPL.Model}()
    chains = Dict{Symbol, Any}()
    results = Dict{Symbol, Any}()
    derivatives = Dict{Symbol, Any}()
    tier_networks = Dict{Symbol, Union{NamedTuple, Nothing}}()
    tier_predictions = Dict{Symbol, DataFrame}()
    tier_unit_predictions = Dict{Symbol, Any}()

    if verbose
        println("\n==============================================================================")
        println("  TethysBayes HIERARCHICAL MODULAR DAG PIPELINE ORCHESTRATOR")
        println("  Tiers: $(join([string(t.name) for t in tier_specs], " -> "))")
        println("  DuckDB: $(duckdb_path)")
        println("==============================================================================\n")
    end

    # --- Sequential Execution Across Tiers ---
    for (t_idx, spec) in enumerate(tier_specs)
        t_name = spec.name
        if verbose
            println("--- [Tier $(t_idx)/$(n_tiers)]: $(uppercase(string(t_name))) ---")
            println("  Formula: $(spec.formula)")
        end

        cur_df = copy(spec.data)
        cur_au = spec.au
        cur_W = spec.W

        # Auto-resolve network if W not explicitly supplied but au is available
        if isnothing(cur_W) && !isnothing(cur_au) && hasproperty(cur_au, :W)
            cur_W = cur_au.W
        end

        # 1. Propagate Upstream Covariates (EIV & Predictions)
        for (prev_name, prev_pred_df) in tier_predictions
            # Check if any column or tier name is referenced in formula
            val_col = Symbol("$(prev_name)_mean")
            sd_col = Symbol("$(prev_name)_sd")

            if hasproperty(prev_pred_df, val_col) && (
                occursin(string(prev_name), spec.formula) || 
                occursin(string(val_col), spec.formula)
            )
                # Map upstream prediction to current tier dataframe
                prev_spec = tier_specs[findfirst(s -> s.name == prev_name, tier_specs)]
                prev_au = tier_networks[prev_name]

                # Match coordinates or spatial units
                cur_xy = try _detect_xy_columns(cur_df) catch; nothing end
                prev_xy = try _detect_xy_columns(prev_pred_df) catch; nothing end

                if !isnothing(cur_xy) && !isnothing(prev_xy)
                    # Coordinate-based nearest neighbor / interpolation mapping
                    kdt = KDTree(hcat(prev_pred_df[!, prev_xy[1]], prev_pred_df[!, prev_xy[2]])')
                    query_pts = hcat(cur_df[!, cur_xy[1]], cur_df[!, cur_xy[2]])'
                    nns, _ = knn(kdt, query_pts, 1)
                    nn_idxs = [n[1] for n in nns]
                    cur_df[!, prev_name] = prev_pred_df[!, val_col][nn_idxs]
                    cur_df[!, Symbol("$(prev_name)_se")] = prev_pred_df[!, sd_col][nn_idxs]
                elseif hasproperty(cur_df, :s_idx) && !isnothing(cur_au) && !isnothing(prev_au)
                    # Mesh-to-mesh resharding transfer
                    if haskey(tier_unit_predictions, prev_name)
                        resharded = reshard_spatial_field(
                            tier_unit_predictions[prev_name], prev_au, cur_au; mode=:samples
                        )
                        cur_df[!, prev_name] = resharded.mean[cur_df.s_idx]
                        cur_df[!, Symbol("$(prev_name)_se")] = resharded.std[cur_df.s_idx]
                    else
                        P_trans = compute_network_transfer_matrix(prev_au, cur_au)
                        u_means = [mean(prev_pred_df[!, val_col][prev_pred_df.s_idx .== i]) for i in 1:prev_au.n_units]
                        u_sds   = [mean(prev_pred_df[!, sd_col][prev_pred_df.s_idx .== i]) for i in 1:prev_au.n_units]
                        unit_means = Vector{Float64}(P_trans * u_means)
                        unit_sds = Vector{Float64}(sqrt.(P_trans * (u_sds .^ 2)))
                        cur_df[!, prev_name] = unit_means[cur_df.s_idx]
                        cur_df[!, Symbol("$(prev_name)_se")] = unit_sds[cur_df.s_idx]
                    end
                end
            end
        end



        # 2. Fit Model & Sample Posterior
        
        # wrap in trampoline
        
        function _sample_tier_model(m, sampler, n_samples)
            return Base.invokelatest(sample, m, sampler, n_samples; progress=false)
        end
 
        m_tier = if !isnothing(cur_W)
            TethysBayes_core(spec.formula, cur_df, @__MODULE__; W=cur_W, verbose=false)
        else
            TethysBayes_core(spec.formula, cur_df, @__MODULE__; verbose=false)
        end

         
        chn_tier = _sample_tier_model(m_tier, spec.sampler, spec.n_samples)
        
        res_tier = model_results_comprehensive(m_tier, chn_tier; alpha=0.05)
 
        models[t_name] = m_tier
        chains[t_name] = chn_tier
        results[t_name] = res_tier
        tier_networks[t_name] = cur_au

        # 3. Compute Surface Derivatives if requested
        if !isempty(spec.derivatives)
            cur_xy = try _detect_xy_columns(cur_df) catch; nothing end
            coords_query = if !isnothing(cur_xy)
                cur_df[:, [cur_xy[1], cur_xy[2]]]
            elseif !isnothing(cur_au)
                DataFrame(s_x = [c[1] for c in cur_au.centroids], s_y = [c[2] for c in cur_au.centroids])
            else
                nothing
            end

            if !isnothing(coords_query)
                deriv_res = TethysBayes_surface_derivatives(
                    m_tier, chn_tier, coords_query;
                    metrics=spec.derivatives,
                    radii=spec.radii,
                    return_samples=true
                )
                derivatives[t_name] = deriv_res
            end
        end

        # 4. Generate Predictions DataFrame for Downstream Sharing
        pred_df = copy(cur_df)
        pred_res = predict(m_tier, chn_tier, cur_df)
        pred_df[!, Symbol("$(t_name)_mean")] = pred_res.predictions_denoised.mean
        pred_df[!, Symbol("$(t_name)_sd")] = pred_res.predictions_denoised.std
        tier_predictions[t_name] = pred_df

        # Unit-level predictions on the tier's spatial network
        if !isnothing(cur_au)
            df_units = DataFrame(
                s_idx = 1:cur_au.n_units,
                s_x = [c[1] for c in cur_au.centroids],
                s_y = [c[2] for c in cur_au.centroids]
            )
            for col in names(cur_df)
                if !(col in ["s_idx", "s_x", "s_y", string(t_name), "y_obs"])
                    if eltype(cur_df[!, col]) <: Number
                        df_units[!, col] = fill(mean(skipmissing(cur_df[!, col])), cur_au.n_units)
                    else
                        df_units[!, col] = fill(first(cur_df[!, col]), cur_au.n_units)
                    end
                end
            end
            try
                pred_res_u = predict(m_tier, chn_tier, df_units)
                tier_unit_predictions[t_name] = pred_res_u.predictions_denoised
            catch
                u_m = [mean(pred_df[!, Symbol("$(t_name)_mean")][pred_df.s_idx .== i]) for i in 1:cur_au.n_units]
                u_s = [mean(pred_df[!, Symbol("$(t_name)_sd")][pred_df.s_idx .== i]) for i in 1:cur_au.n_units]
                tier_unit_predictions[t_name] = (mean=u_m, std=u_s, samples=nothing)
            end
        end

        # 5. Persist Tier Results to DuckDB Table
        table_name = "tier_$(t_idx)_$(t_name)"
        if duckdb_path != ":memory:" && !isempty(duckdb_path)
            export_posterior_samples_to_duckdb(
                duckdb_path, chn_tier, m_tier;
                table_name = table_name, overwrite = true
            )
            if verbose
                println("  Saved to DuckDB table: $(table_name)")
            end
        end
    end

    # --- Final Master Network Harmonization ---
    final_master_au = !isnothing(master_au) ? master_au : tier_networks[tier_specs[end].name]
    if isnothing(final_master_au)
        # Construct fallback master mesh from bounding coordinates
        all_x = Float64[]
        all_y = Float64[]
        for s in tier_specs
            s_xy = try _detect_xy_columns(s.data) catch; nothing end
            if !isnothing(s_xy)
                append!(all_x, s.data[!, s_xy[1]])
                append!(all_y, s.data[!, s_xy[2]])
            end
        end
        final_master_au = assign_spatial_units(all_x, all_y; target_units=30)
    end

    n_master = final_master_au.n_units
    master_df = DataFrame(
        unit_id = 1:n_master,
        s_x = [c[1] for c in final_master_au.centroids],
        s_y = [c[2] for c in final_master_au.centroids]
    )

    # Add WKT geometry column if polygons exist
    if hasproperty(final_master_au, :polygons)
        wkt_strings = String[]
        for poly in final_master_au.polygons
            poly_closed = copy(poly)
            if !isempty(poly) && poly[1] != poly[end]
                push!(poly_closed, poly[1])
            end
            coords_str = join(["$(pt[1]) $(pt[2])" for pt in poly_closed], ", ")
            push!(wkt_strings, "POLYGON(($(coords_str)))")
        end
        master_df[!, :geom_wkt] = wkt_strings
    end

    # Reshard / evaluate all tiers onto master network
    for (t_name, pred_df) in tier_predictions
        src_au = tier_networks[t_name]
        val_col = Symbol("$(t_name)_mean")
        sd_col = Symbol("$(t_name)_sd")

        if !isnothing(src_au) && haskey(tier_unit_predictions, t_name)
            resharded = reshard_spatial_field(
                tier_unit_predictions[t_name], src_au, final_master_au; mode=:samples
            )
            master_df[!, val_col] = resharded.mean
            master_df[!, sd_col] = resharded.std
        elseif !isnothing(src_au)
            P_master = compute_network_transfer_matrix(src_au, final_master_au)
            u_means = [mean(pred_df[!, val_col][pred_df.s_idx .== i]) for i in 1:src_au.n_units]
            u_sds   = [mean(pred_df[!, sd_col][pred_df.s_idx .== i]) for i in 1:src_au.n_units]
            master_df[!, val_col] = Vector{Float64}(P_master * u_means)
            master_df[!, sd_col] = Vector{Float64}(sqrt.(P_master * (u_sds .^ 2)))
        else
            pred_xy = try _detect_xy_columns(pred_df) catch; nothing end
            if !isnothing(pred_xy)
                kdt = KDTree(hcat(pred_df[!, pred_xy[1]], pred_df[!, pred_xy[2]])')
                q_pts = hcat(master_df.s_x, master_df.s_y)'
                nns, _ = knn(kdt, q_pts, 1)
                idxs = [n[1] for n in nns]
                master_df[!, val_col] = pred_df[!, val_col][idxs]
                master_df[!, sd_col] = pred_df[!, sd_col][idxs]
            end
        end
    end

    # Save harmonized master summary to DuckDB
    if duckdb_path != ":memory:" && !isempty(duckdb_path)
        db = DuckDB.DB(duckdb_path)
        con = DuckDB.connect(db)
        try
            _write_df_to_duckdb(con, master_df, "master_harmonized_summary", true)
        finally
            DuckDB.disconnect(con)
            
            
        end
    end

    # Export GeoJSON if requested
    if !isnothing(geojson_path)
        export_spatial_results_to_geojson(geojson_path, master_df, final_master_au)
    end

    # Save JLD2 bundle if requested
    if !isnothing(jld2_path)
        JLD2.jldsave(jld2_path;
            models=models, chains=chains, results=results,
            derivatives=derivatives, master_summary=master_df, master_au=final_master_au
        )
    end

    if verbose
        println("\n==============================================================================")
        println("  PIPELINE EXECUTION COMPLETE")
        println("  Master Network Units: $(n_master)")
        println("  Harmonized Summary Table: 'master_harmonized_summary'")
        println("==============================================================================\n")
    end

    return PipelineResult(
        [s.name for s in tier_specs],
        models,
        chains,
        results,
        derivatives,
        tier_networks,
        final_master_au,
        master_df,
        duckdb_path
    )
end
