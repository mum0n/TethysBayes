"""
    data.jl

Monolithic synthetic data generator and built-in spatiotemporal benchmark datasets
for Bayesian Spatio-Temporal Models (TethysBayes).

Version: v1.0.0
"""

"""
    TethysBayes_data(type="scottish_lip"; kwargs...)

Consolidated synthetic and benchmark dataset generator for TethysBayes models.

# Supported Dataset Types (`type`):
- `"scottish_lip"` (Default): Scottish Lip Cancer spatiotemporal dataset (primary & nested)
  enriched
  with comprehensive covariates (`y`, `y_rate`, `y_bin`, `y_gauss`, `y_pois`, `ordinal_y`,
  `y_cat1..3`, `counts`, `t_idx`, `group`, `group_id`, `grouping_covariate`, `cell_area`, `effort`,
  `removal`, `removal_total`, `proxy_val`, `predator_pop`, `recruitment`, `habitat`,
  `species_1..3`, `age_1..3`, `class_1..4`).
- `"ordinal"`: Non-proportional odds ordinal dataset.
- `"sim"` / `"spatiotemporal"`: Standardized spatiotemporal dataset.
- `"lgcp_regular"`: Regular grid Log-Gaussian Cox Process synthetic data.
- `"lgcp_irregular"`: Irregular grid Log-Gaussian Cox Process synthetic data.
- `"advanced"`: Multi-response (Gaussian, Poisson, Multinomial, Proxy) spatiotemporal dataset.
- `"logistic"`: Logistic population dynamics synthetic dataset.
- `"delay_difference"`: Delay-difference multivariate population dynamics dataset.
- `"glv"` / `"generalized_lotka_volterra"`: Generalized Lotka-Volterra dynamics dataset.
- `"lotka_volterra"`: Lotka-Volterra prey-predator dynamics dataset.
- `"leslie_logistic"`: Leslie-Logistic population dataset.
- `"logistic_spatial_k"`: Logistic growth with spatially varying carrying capacity K.
- `"logistic_spatial_r"`: Logistic growth with spatially varying growth rate r.
- `"leslie_matrix"`: Multivariate Leslie matrix dynamics dataset.
- `"hierarchical"` / `"marine_ecosystem"` / `"multi_tier"`: 6-tier marine ecological
  pipeline bundle returning a NamedTuple:
  `(bathymetry, substrate, temperature, species_composition, snow_crab, individuals)`.
  The `individuals` table contains one row per measured animal from the Tier 6 biological
  sub-sampling program (carapace width, sex, maturity).
- `"bathymetry"`: Tier 1 continuous bathymetric soundings dataset.
- `"substrate"`: Tier 2 benthic sediment grab stations dataset.
- `"temperature"` / `"ctd"`: Tier 3 hydrographic CTD casts across 10 years.
- `"species_composition"` / `"community"` / `"trawl"`: Tier 4 multi-species survey haul records across 30 species.
- `"snow_crab"` / `"crab"`: Tier 5 target species survey tows and demographic biomass records.
- `"biosampling"` / `"individuals"` / `"tier6_biosampling"`: Tier 6 individual biological
  sampling records: `tow_id`, `year`, `month`, `s_x`, `s_y`, `carapace_width_mm`,
  `size_bin` (Int, 1–L), `sex` (0=female, 1=male), `maturity` (0=immature, 1=mature).

# Common Keyword Arguments:
- `s_N::Int = 10`: Number of spatial units.
- `t_N::Int = 5`: Number of temporal units.
- `n_years::Int = 10`: Number of years for Scottish Lip Cancer dataset.
- `spatial_expansion::Float64 = 1.5`: Spatial expansion factor for nested Scottish Lip dataset.
- `temporal_expansion::Float64 = 1.5`: Temporal expansion factor for nested Scottish Lip dataset.
- `rndseed::Int = 42`: Random seed.
- `seed::Union{Int, Nothing} = nothing`: Alternative random seed keyword.
- `recreate::Bool = false`: Force recreation of cached datasets.
- `grid_side::Int = 15`: Grid side length for LGCP datasets.
- `n_species::Int = 3`: Number of species for multi-species models.
- `n_age_classes::Int = 3`: Number of age classes for Leslie models.
- `n_classes::Int = 4`: Number of classes for generalized Leslie models.
- `n_obs::Int = 500`: Total observations for ordinal data.
- `n_groups::Int = 10`: Group count for random intercepts.
- `n_obs_per_st_unit::Int = 1`: Observations per space-time unit.
- `n_obs_per_unit::Int = 10`: Observations per unit for Dirichlet-Multinomial.
- `n_units::Int = 25`: Spatial units for Dirichlet-Multinomial.
- `n_categories::Int = 3`: Category count for multinomial models.
- `use_effort::Bool = false`: Include effort covariate.
- `use_removal::Bool = false`: Include removal covariate.
"""
function TethysBayes_data(
    type_pos::Union{String, Symbol, Nothing} = nothing;
    type::Union{String, Symbol, Nothing} = nothing,
    s_N::Int = 10,
    t_N::Int = 5,
    n_years::Int = 10,
    spatial_expansion::Float64 = 1.5,
    temporal_expansion::Float64 = 1.5,
    rndseed::Int = 42,
    seed::Union{Int, Nothing} = nothing,
    recreate::Bool = false,
    grid_side::Int = 15,
    n_species::Int = 3,
    n_age_classes::Int = 3,
    n_classes::Int = 4,
    n_obs::Int = 500,
    n_groups::Int = 10,
    n_obs_per_st_unit::Int = 1,
    n_obs_per_unit::Int = 10,
    n_units::Int = 25,
    n_categories::Int = 3,
    domain_size::Float64 = 600.0,
    n_marks::Int = 100,
    area_method::Symbol = :cvt,
    use_effort::Bool = false,
    use_removal::Bool = false,
    kwargs...
)
    actual_seed = seed !== nothing ? seed : rndseed
    resolved_type = if type !== nothing
        type
    elseif type_pos !== nothing
        type_pos
    else
        "scottish_lip"
    end

    type_str = lowercase(string(resolved_type))

    if type_str in ["hierarchical", "hierarchical_workflow", "marine_ecosystem", "multi_tier"]
        return generate_mock_hierarchical_datasets(seed=actual_seed)

    elseif type_str in ["bathymetry", "tier1_bathymetry"]
        return generate_mock_hierarchical_datasets(seed=actual_seed).bathymetry

    elseif type_str in ["substrate", "tier2_substrate"]
        return generate_mock_hierarchical_datasets(seed=actual_seed).substrate

    elseif type_str in ["temperature", "tier3_temperature", "ctd"]
        return generate_mock_hierarchical_datasets(seed=actual_seed).temperature

    elseif type_str in ["species_composition", "tier4_species", "community", "trawl"]
        return generate_mock_hierarchical_datasets(seed=actual_seed).species_composition

    elseif type_str in ["snow_crab", "tier5_snow_crab", "crab"]
        return generate_mock_hierarchical_datasets(seed=actual_seed).snow_crab

    elseif type_str in ["biosampling", "individuals", "tier6_biosampling"]
        return generate_mock_hierarchical_datasets(seed=actual_seed).individuals

    elseif type_str in ["scottish_lip_neighbours" ]
 
        neighbor_list = [
            [5, 9, 11, 19], [7, 10], [6, 12], [18, 20, 28], [1, 11, 12, 13, 19],
            [3, 8], [2, 10, 13, 16, 17], [6], [1, 11, 17, 19, 23, 29], [2, 7, 16, 22],
            [1, 5, 9, 12], [3, 5, 11], [5, 7, 17, 19], [31, 32, 35], [25, 29, 50],
            [7, 10, 17, 21, 22, 29], [7, 9, 13, 16, 19, 29], [4, 20, 28, 33, 55, 56],
            [1, 5, 9, 13, 17], [4, 18, 55], [16, 29, 50], [10, 16],
            [9, 29, 34, 36, 37, 39], [27, 30, 31, 44, 47, 48, 55, 56], [15, 26, 29],
            [26, 29, 42, 43], [24, 31, 32, 55], [4, 18, 33, 45],
            [9, 15, 16, 17, 21, 23, 25, 26, 34, 43, 50], [24, 38, 42, 44, 45, 56],
            [14, 24, 27, 32, 35, 46, 47], [14, 27, 31, 35], [18, 28, 45, 56],
            [23, 29, 39, 40, 42, 43, 51, 52, 54], [14, 31, 32, 37, 46],
            [23, 37, 39, 41], [23, 35, 36, 41, 46], [30, 42, 44, 49, 51, 54],
            [23, 34, 36, 40, 41], [34, 39, 41, 49, 52], [36, 37, 39, 40, 46, 49, 53],
            [26, 30, 34, 38, 43, 51], [26, 29, 34, 42], [24, 30, 38, 48, 49],
            [28, 30, 33, 56], [31, 35, 37, 41, 47, 53], [24, 31, 46, 48, 49, 53],
            [24, 44, 47, 49], [38, 40, 41, 44, 47, 48, 52, 53, 54], [15, 21, 29],
            [34, 38, 42, 54], [34, 40, 49, 54], [41, 46, 47, 49], [34, 38, 49, 51, 52],
            [18, 20, 24, 27, 56], [18, 24, 30, 33, 45, 55]
        ]

        n_districts = length(neighbor_list)

        W_raw = spzeros(Int, n_districts, n_districts)
        for i in 1:n_districts
            for nb in neighbor_list[i]
                W_raw[i, nb] = 1
            end
        end
        W = sparse(Symmetric(Matrix(W_raw + W_raw')) .> 0) 

        return W

    elseif type_str in ["scottish_lip_au" ]

        cache_path = "data/scottish_lip_au_cache.jld2"

        if isfile(cache_path) && !recreate
            try
                println("Loading cached dataset from: ", cache_path)
                data_bundle = JLD2.load(cache_path)
                au = data_bundle["au"]
                return au
            catch e
                println("Failed to load cache ($e); regenerating dataset...")
            end
        end

        W = TethysBayes_data("scottish_lip_neighbours")

        au = assign_spatial_units_inferred(W)

        if !isdir("data")
            mkdir("data")
        end

        JLD2.save(cache_path, "au", au )
        println("Dataset successfully cached at: ", cache_path)

        return au

    elseif type_str in ["scottish_lip" ]
 
        println("Generating new spatiotemporal dataset...")
        Random.seed!(actual_seed)

        W  = TethysBayes_data("scottish_lip_neighbours")
        au = TethysBayes_data("scottish_lip_au")
        p_centroids = au.centroids

        y_orig = [
            9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
            16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
            10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0
        ]
        E_orig = [
            1.4, 8.7, 3.0, 2.5, 4.3, 2.4, 8.1, 2.3, 2.0, 6.6, 4.4, 1.8, 1.1, 3.3,
            7.8, 4.6, 1.1, 4.2, 5.5, 4.4, 10.5, 22.7, 8.8, 5.6, 15.5, 12.5, 6.0,
            9.0, 14.4, 10.2, 4.8, 2.9, 7.0, 8.5, 12.3, 10.1, 12.7, 9.4, 7.2, 5.3,
            18.8, 15.8, 4.3, 14.6, 50.7, 8.2, 5.6, 9.3, 88.7, 19.6, 3.4, 3.6, 5.7,
            7.0, 4.2, 1.8
        ]
        x_orig = [
            16, 16, 10, 24, 10, 24, 10, 7, 7, 16, 7, 16, 10, 24, 7, 16, 10, 7, 7,
            10, 7, 16, 10, 7, 1, 1, 7, 7, 10, 10, 7, 24, 10, 7, 7, 0, 10, 1, 16,
            0, 1, 16, 16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1, 16, 10
        ]

        n_districts = length(y_orig)
        data_sl = DataFrame()
        for i in 1:n_districts
            log_off = log.(fill(E_orig[i], n_years))
            innov = cumsum(randn(n_years) .* 0.1)
            y_p = floor.(Int, abs.(fill(y_orig[i], n_years) .+ (innov .* 4.0)))

            d_df = DataFrame(
                district = i,
                year = 1:n_years,
                t_idx = 1:n_years,
                y = y_p,
                log_offsets = log_off,
                cov1 = fill(Float64(x_orig[i]) / 10.0, n_years)
            )
            d_df.y_rate = d_df.y ./ exp.(d_df.log_offsets)
            append!(data_sl, d_df)
        end

        n_total = nrow(data_sl)

        data_sl.y_bin = [v > mean(data_sl.y_rate) ? 1 : 0 for v in data_sl.y_rate]

        data_sl.cov2 = 0.5 .* data_sl.cov1 .+ randn(n_total)
        data_sl.cov3 = randn(n_total) .* (data_sl.y_rate .^ 2)
        data_sl.cov4 = randn(n_total) .* log.(data_sl.y_rate .+ 1.0)
        data_sl.cov5 = randn(n_total) .* exp.(data_sl.y_rate) .* 2.0
        data_sl.cov6 = randn(n_total)

        data_sl.day = rand(1:365, n_total)
        data_sl.month = Int.(round.(data_sl.day ./ 365 .* 12)) .+ 1

        data_sl.f1 = rand(["A", "B"], n_total)
        data_sl.s_idx = data_sl.district
        data_sl.s_x = [c[1] for c in p_centroids[data_sl.s_idx]]
        data_sl.s_y = [c[2] for c in p_centroids[data_sl.s_idx]]

        reg_indices = mod1.(1:n_total, 4)
        reg_levels = ["North", "South", "East", "West"]
        reg = reg_levels[reg_indices]
        data_sl.region = categorical(reg)

        data_sl.group = categorical(data_sl.district)
        data_sl.group_id = categorical(data_sl.district)
        data_sl.grouping_covariate = categorical(data_sl.region)

        # Comprehensive additional response types and covariates for component testing
        data_sl.y_gauss = Float64.(data_sl.y_rate) .+ randn(n_total) .* 0.1
        data_sl.y_pois = data_sl.y
        data_sl.counts = data_sl.y
        data_sl.cell_area = fill(1.0, n_total)

        data_sl.ordinal_y = [v == 0 ? 1 : (v < 5 ? 2 : 3) for v in data_sl.y]

        eta_m = hcat(data_sl.cov1, data_sl.cov2, zeros(n_total))
        y_mult = zeros(Int, n_total, 3)
        for i in 1:n_total
            p = NNlib.softmax(eta_m[i, :])
            y_mult[i, :] = rand(Multinomial(20, p))
        end
        data_sl.y_cat1 = y_mult[:, 1]
        data_sl.y_cat2 = y_mult[:, 2]
        data_sl.y_cat3 = y_mult[:, 3]

        data_sl.effort = rand(n_total) .* 0.5 .+ 0.1
        data_sl.removal = rand(n_total) .* 2.0
        data_sl.removal_total = data_sl.removal
        data_sl.proxy_val = data_sl.y_gauss .+ randn(n_total) .* 0.3
        data_sl.predator_pop = rand(n_total) .* 5.0 .+ 1.0
        data_sl.recruitment = rand(n_total) .* 10.0 .+ 2.0
        data_sl.habitat = rand(n_total)
        data_sl.species_1 = rand(n_total) .* 10.0
        data_sl.species_2 = rand(n_total) .* 10.0
        data_sl.species_3 = rand(n_total) .* 10.0
        data_sl.age_1 = rand(n_total) .* 10.0
        data_sl.age_2 = rand(n_total) .* 10.0
        data_sl.age_3 = rand(n_total) .* 10.0
        data_sl.class_1 = rand(n_total) .* 10.0
        data_sl.class_2 = rand(n_total) .* 10.0
        data_sl.class_3 = rand(n_total) .* 10.0
        data_sl.class_4 = rand(n_total) .* 10.0

        au = merge(au, (
            s_idx = data_sl.s_idx,
            s_x = [c[1] for c in p_centroids[data_sl.s_idx]],
            s_y = [c[2] for c in p_centroids[data_sl.s_idx]],
            s_vals = collect(1:n_districts)
        ))

        # Nested dataset construction
        px = [c[1] for c in p_centroids]
        py = [c[2] for c in p_centroids]
        x_min, x_max = minimum(px), maximum(px)
        y_min, y_max = minimum(py), maximum(py)
        x_rng, y_rng = x_max - x_min, y_max - y_min

        s_buff = (spatial_expansion - 1.0) / 2.0
        nx_min, nx_max = x_min - s_buff * x_rng, x_max + s_buff * x_rng
        ny_min, ny_max = y_min - s_buff * y_rng, y_max + s_buff * y_rng

        nt_max = Int(round(n_years * temporal_expansion))
        
        scottish_lip = (data=data_sl, au=au)
 
        return scottish_lip

    elseif type_str in ["nested" ]
        # Generate a new "nested" dataset within the domain of the primary dataset
        # using scottish lip cancer as a starting basis .. this is to allow nested() examples

        println("Generating new spatiotemporal dataset...")
        Random.seed!(actual_seed)

        W  = TethysBayes_data("scottish_lip_neighbours")
        au = TethysBayes_data("scottish_lip_au")
        p_centroids = au.centroids

        n_districts = size(W, 2)
   
        # "Nested" dataset construction ( within scottish lip data domain)
        px = [c[1] for c in p_centroids]
        py = [c[2] for c in p_centroids]
        x_min, x_max = minimum(px), maximum(px)
        y_min, y_max = minimum(py), maximum(py)
        x_rng, y_rng = x_max - x_min, y_max - y_min

        s_buff = (spatial_expansion - 1.0) / 2.0
        nx_min, nx_max = x_min - s_buff * x_rng, x_max + s_buff * x_rng
        ny_min, ny_max = y_min - s_buff * y_rng, y_max + s_buff * y_rng

        nt_max = Int(round(n_years * temporal_expansion))

        n_obs_nested = Int(round(n_total * spatial_expansion * temporal_expansion))

        sx_nested = rand(Uniform(nx_min, nx_max), n_obs_nested)
        sy_nested = rand(Uniform(ny_min, ny_max), n_obs_nested)
        time_nested = rand(1:nt_max, n_obs_nested)

        au = assign_spatial_units(sx_nested, sy_nested; target_units=100)

        slnested = DataFrame(
            s_x = sx_nested,
            s_y = sy_nested,
            year = time_nested,
            t_idx = time_nested,
            district = au.s_idx,
            s_idx = au.s_idx
        )

        s_lat_n = cumsum(randn(length(au.centroids))) .* 0.3
        t_lat_n = sin.(collect(1:nt_max) .* (2π/nt_max))

        eta_n = [
            1.5 + s_lat_n[slnested.district[i]] + t_lat_n[slnested.year[i]]
            for i in 1:n_obs_nested
        ]

        slnested.y = [rand(Poisson(exp(v))) for v in eta_n]
        slnested.y_rate = exp.(eta_n) .+ randn(n_obs_nested) .* 0.2
        slnested.y_bin = [v > mean(slnested.y_rate) ? 1 : 0 for v in slnested.y_rate]

        slnested.cov1 = 0.6 .* eta_n .+ randn(n_obs_nested)
        slnested.cov2 = randn(n_obs_nested) .* exp.(slnested.y_rate)
        slnested.cov3 = randn(n_obs_nested)
        slnested.ncov1 = slnested.cov1
        slnested.ncov2 = slnested.cov2
        slnested.ncov3 = slnested.cov3
        slnested.group = categorical(slnested.district)
        slnested.group_id = categorical(slnested.district)
        slnested.month = mod1.(slnested.year, 12)
        slnested.day = rand(1:365, n_obs_nested)
 
        return ( data=slnested, au=au )

    elseif type_str == "ordinal"
        Random.seed!(actual_seed)
        cov1 = randn(n_obs)
        cov2 = rand(n_obs) .* 2
        cov3 = rand(-2:2, n_obs)
        group_id = rand(1:n_groups, n_obs)

        alpha_1 = -1.0
        alpha_2 = 1.5
        beta_cov2 = 0.5
        beta_cov3 = -0.4
        beta_cov1_cat1 = 1.2
        beta_cov1_cat2 = -0.8

        sigma_random_intercept = 0.7
        random_intercepts = rand(Normal(0, sigma_random_intercept), n_groups)

        ordinal_y = Vector{Int}(undef, n_obs)
        for i in 1:n_obs
            eta_prop = (beta_cov2 * cov2[i]) + (beta_cov3 * cov3[i]) +
                       random_intercepts[group_id[i]]
            linear_pred_1 = alpha_1 - (eta_prop + cov1[i] * beta_cov1_cat1)
            linear_pred_2 = alpha_2 - (eta_prop + cov1[i] * beta_cov1_cat2)

            cum_prob_1 = cdf(Normal(), linear_pred_1)
            cum_prob_2 = cdf(Normal(), linear_pred_2)

            prob_1 = cum_prob_1
            prob_2 = max(0.0, cum_prob_2 - cum_prob_1)
            prob_3 = max(0.0, 1.0 - cum_prob_2)

            probs = [prob_1, prob_2, prob_3]
            probs ./= sum(probs)
            ordinal_y[i] = rand(Categorical(probs))
        end

        return DataFrame(
            ordinal_y = ordinal_y,
            cov1 = cov1,
            cov2 = cov2,
            cov3 = cov3,
            group_id = categorical(group_id)
        )

    elseif type_str in ["sim", "spatiotemporal"]
        s_coords = [(Float64(mod(i-1, 5)), Float64(div(i-1, 5))) for i in 1:s_N]
        s_x = [pt[1] for pt in s_coords]
        s_y = [pt[2] for pt in s_coords]
        _, W = spatial_knn_graph(s_coords, min(4, max(1, s_N - 1)))

        t_idx = 1:t_N
        phi_s = rand(MvNormal(zeros(s_N), I))
        phi_t = sin.(2 * pi .* (t_idx./t_N)) .+ rand(Normal(0,0.1), t_N)
        phi_st = rand(MvNormal(zeros(s_N*t_N), I))

        s_cov = rand(Normal(0,1), s_N)
        t_cov = rand(Normal(0,1), t_N)

        n_total = s_N * t_N
        s_v = vcat([fill(i, t_N) for i in 1:s_N]...)
        t_v = vcat([t_idx for i in 1:s_N]...)

        eta = (
            0.5 .* phi_s[s_v] .+
            0.8 .* phi_t[t_v] .+
            0.2 .* s_cov[s_v] .+
            0.3 .* t_cov[t_v]
        )

        y1 = eta .+ rand(Normal(0, 0.25), n_total)
        y2 = [rand(Poisson(exp(v))) for v in eta]

        y_binary = Vector{Int}(undef, n_total)
        mean_eta = mean(eta)
        for i in 1:n_total
            y_binary[i] = eta[i] > mean_eta ? 1 : 0
        end

        eta_mult = hcat(eta, -eta, sin.(eta))
        y_mult = zeros(Int, n_total, 3)
        for i in 1:n_total
            p = NNlib.softmax(eta_mult[i, :])
            y_mult[i, :] = rand(Multinomial(20, p))
        end

        proxy_y = rand(Normal(0,1), n_total)

        return DataFrame(
            s_idx = s_v,
            t_idx = t_v,
            s_x = s_x[s_v],
            s_y = s_y[s_v],
            year = t_v,
            month = mod1.(t_v, 12),
            y_gauss = y1,
            y_pois = y2,
            y_bin = y_binary,
            y_cat1 = y_mult[:, 1],
            y_cat2 = y_mult[:, 2],
            y_cat3 = y_mult[:, 3],
            proxy_val = proxy_y
        )

    elseif type_str in ["lgcp_regular", "lgcp"]
        Random.seed!(actual_seed)
        s_N_eff = grid_side^2
        x_coords = repeat(1:grid_side, inner=grid_side)
        y_coords = repeat(1:grid_side, outer=grid_side)

        Z_true = [2.0 * sin(x/3.0) * cos(y/3.0) for (x, y) in zip(x_coords, y_coords)]
        y_counts = [rand(Poisson(exp(z))) for z in Z_true]

        df = DataFrame(
            s_idx = 1:s_N_eff,
            s_x = Float64.(x_coords),
            s_y = Float64.(y_coords),
            counts = y_counts
        )
        W = libgeos_lattice_adjacency_matrix(grid_side, grid_side)
        return df, W, s_N_eff

    elseif type_str == "lgcp_irregular"
        Random.seed!(actual_seed)
        s_N_eff = grid_side^2
        x_coords = repeat(1:grid_side, inner=grid_side)
        y_coords = repeat(1:grid_side, outer=grid_side)

        areas = [
            1.0 + 2.0 * exp(-((x - 5)^2 + (y - 5)^2) / 10.0)
            for (x, y) in zip(x_coords, y_coords)
        ]
        Z_true = fill(log(5.0), s_N_eff)
        y_counts = [rand(Poisson(exp(z) * a)) for (z, a) in zip(Z_true, areas)]

        df = DataFrame(
            s_idx = 1:s_N_eff,
            counts = y_counts,
            cell_area = areas
        )
        W = libgeos_lattice_adjacency_matrix(grid_side, grid_side)
        return df, W, s_N_eff, areas

    elseif type_str == "advanced"
        Random.seed!(actual_seed)
        n_s = 30
        n_t = 12
        n_obs_tot = n_s * n_t

        unique_coords = [(rand()*10, rand()*10) for _ in 1:n_s]
        s_coords = repeat(unique_coords, inner=n_t)
        s_x = [c[1] for c in s_coords]
        s_y = [c[2] for c in s_coords]
        t_v = repeat(collect(1:n_t), outer=n_s)
        s_idx = repeat(collect(1:n_s), inner=n_t)

        latent_field = [sin(x/2) * cos(y/2) + 0.1*t for (x, y, t) in zip(s_x, s_y, t_v)]

        y1 = latent_field .+ randn(n_obs_tot) .* 0.2
        y2 = [rand(Poisson(exp(v))) for v in latent_field]

        eta_mult = hcat(latent_field, -0.5 .* latent_field, zeros(n_obs_tot))
        y_mult = zeros(Int, n_obs_tot, 3)
        for i in 1:n_obs_tot
            p = NNlib.softmax(eta_mult[i, :])
            y_mult[i, :] = rand(Multinomial(20, p))
        end

        proxy_y = latent_field .* 0.8 .+ randn(n_obs_tot) .* 0.5

        return DataFrame(
            s_idx = s_idx,
            s_x = s_x,
            s_y = s_y,
            year = t_v,
            t_idx = t_v,
            month = mod1.(t_v, 12),
            cov1 = randn(n_obs_tot),
            cov2 = rand(n_obs_tot),
            cov3 = rand(n_obs_tot),
            region = categorical(mod1.(s_idx, 5)),
            recruitment = rand(n_obs_tot),
            y_gauss = y1,
            y_pois = y2,
            y_cat1 = y_mult[:, 1],
            y_cat2 = y_mult[:, 2],
            y_cat3 = y_mult[:, 3],
            proxy_val = proxy_y
        )

    elseif type_str == "logistic"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        r_true = 0.5
        K_true = 100.0
        q_true = 0.01

        effort_sim = use_effort ? rand(s_N, t_N) .* 0.5 .+ 0.1 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)

        initial_pop = rand(s_N) * 10.0 .+ 5.0
        y_sim = zeros(s_N, t_N)
        y_sim[:, 1] = initial_pop

        for t in 2:t_N
            for s in 1:s_N
                N_prev = y_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_true / grid_areas[s]
                growth = r_true * D_prev * (1.0 - D_prev / K_density)

                exploitation = 0.0
                if use_effort
                    exploitation += q_true * effort_sim[s, t] * N_prev
                end
                if use_removal
                    exploitation += removal_sim[s, t]
                end

                y_sim[s, t] = max(
                    0.0, N_prev + growth * grid_areas[s] - exploitation + randn() * 2.0
                )
            end
        end

        df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
        if use_effort
            df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
        end
        if use_removal
            df.removal = repeat(vec(removal_sim'), inner=n_obs_per_st_unit)
        end

        return df, W, grid_areas

    elseif type_str == "delay_difference"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        r_true = 0.6
        K_true = 150.0
        M_nat_true = 0.2
        sigma_rec_true = 0.2
        sigma_pop_true = 0.1

        population_sim = zeros(s_N, t_N)
        recruitment_sim = zeros(s_N, t_N)

        q_true = 0.01
        effort_sim = use_effort ? rand(s_N, t_N) .* 10.0 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)

        initial_pop = rand(s_N) .* 20.0 .+ 10.0
        population_sim[:, 1] = initial_pop
        recruitment_sim[:, 1] = initial_pop .* 0.2

        for t in 2:t_N
            for s in 1:s_N
                N_prev = population_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_true / grid_areas[s]

                mean_rec = r_true * D_prev * (1.0 - D_prev / K_density) * grid_areas[s]
                recruitment_sim[s, t] = exp(log(mean_rec + 1e-6) + randn() * sigma_rec_true)

                C_prev = 0.0
                if use_effort
                    C_prev += q_true * effort_sim[s, t-1] * N_prev
                end
                if use_removal
                    C_prev += removal_sim[s, t-1]
                end

                N_survived = (N_prev - C_prev) * exp(-M_nat_true)
                population_sim[s, t] = max(
                    0.0, N_survived + recruitment_sim[s, t] + randn() * sigma_pop_true
                )
            end
        end

        df.y = repeat(vec(population_sim'), inner=n_obs_per_st_unit)
        df.recruitment = repeat(vec(recruitment_sim'), inner=n_obs_per_st_unit)
        if use_effort
            df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
        end
        if use_removal
            df.removal = repeat(vec(removal_sim'), inner=n_obs_per_st_unit)
        end

        return df, W, grid_areas

    elseif type_str in ["glv", "generalized_lotka_volterra"]
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        r_true = [0.5, 0.6, 0.7]
        K_true = [100.0, 120.0, 150.0]
        alpha_true = [1.0 0.5 0.2; 0.3 1.0 0.6; 0.1 0.4 1.0]
        sigma_process_true = [0.1, 0.1, 0.1]

        pop_sim = zeros(s_N, t_N, n_species)
        initial_total_pop = rand(s_N) .* 30.0 .+ 10.0
        for s in 1:s_N
            pop_sim[s, 1, :] = initial_total_pop[s] .* NNlib.softmax(randn(n_species))
        end

        for t in 2:t_N
            for s in 1:s_N
                N_prev = pop_sim[s, t-1, :]
                D_prev = N_prev ./ grid_areas[s]
                K_density = K_true ./ grid_areas[s]

                N_intermediate = zeros(n_species)
                for i in 1:n_species
                    interaction_sum_density = dot(alpha_true[i, :], D_prev)
                    growth_density = r_true[i] * D_prev[i] *
                                     (1.0 - interaction_sum_density / K_density[i])
                    N_intermediate[i] = N_prev[i] + growth_density * grid_areas[s]
                end

                pop_sim[s, t, :] = max.(
                    0.0, N_intermediate .+ randn(n_species) .* sigma_process_true
                )
            end
        end

        for a in 1:n_species
            species_col_name = Symbol("species_$(a)")
            species_data_flat = vec(pop_sim[:, :, a]')
            df[!, species_col_name] = repeat(species_data_flat, inner=n_obs_per_st_unit)
        end
        df.y = df.species_1

        return df, W, grid_areas, n_species

    elseif type_str == "lotka_volterra"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        prey_sim = zeros(s_N, t_N)
        predator_sim = zeros(s_N, t_N)

        initial_prey = rand(s_N) * 10.0 .+ 5.0
        initial_predator = rand(s_N) * 2.0 .+ 1.0
        prey_sim[:, 1] = initial_prey
        predator_sim[:, 1] = initial_predator

        alpha_true = 0.5
        beta_true = 0.01
        gamma_true = 0.005
        delta_true = 0.2

        for t in 2:t_N
            for s in 1:s_N
                N_prey_prev = prey_sim[s, t-1]
                N_pred_prev = predator_sim[s, t-1]

                d_prey = alpha_true * N_prey_prev - beta_true * N_prey_prev * N_pred_prev
                d_pred = gamma_true * N_prey_prev * N_pred_prev - delta_true * N_pred_prev

                prey_sim[s, t] = max(0.0, N_prey_prev + d_prey + randn() * 0.5)
                predator_sim[s, t] = max(0.0, N_pred_prev + d_pred + randn() * 0.1)
            end
        end

        df.y = repeat(vec(prey_sim'), inner=n_obs_per_st_unit)
        df.predator_pop = repeat(vec(predator_sim'), inner=n_obs_per_st_unit)
        return df, W, grid_areas

    elseif type_str == "leslie_logistic"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        y_sim = zeros(s_N, t_N)
        initial_pop = rand(s_N) * 10.0 .+ 5.0
        y_sim[:, 1] = initial_pop

        K_true = 100.0
        survival_true = fill(0.8, n_age_classes - 1)
        fecundity_true = [0.0, 1.5, 2.0]

        L_true = zeros(n_age_classes, n_age_classes)
        for i in 1:(n_age_classes - 1)
            L_true[i+1, i] = survival_true[i]
        end
        L_true[1, :] = fecundity_true

        r_leslie_true = log(maximum(abs.(eigen(L_true).values)))

        for t in 2:t_N
            for s in 1:s_N
                N_prev = y_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_true / grid_areas[s]

                growth = r_leslie_true * D_prev * (1.0 - D_prev / K_density)
                y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] + randn() * 2.0)
            end
        end

        df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
        return df, W, grid_areas, n_age_classes

    elseif type_str == "logistic_spatial_k"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        s_coords = unique(df[!, [:s_idx, :s_x]])
        sort!(s_coords, :s_idx)
        K_spatial_true = 50.0 .+ 150.0 * (s_coords.s_x ./ maximum(s_coords.s_x))

        y_sim = zeros(s_N, t_N)
        initial_pop = rand(s_N) * 20.0 .+ 10.0
        y_sim[:, 1] = initial_pop
        r_true = 0.6

        for t in 2:t_N
            for s in 1:s_N
                N_prev = y_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_spatial_true[s] / grid_areas[s]
                growth = r_true * D_prev * (1.0 - D_prev / K_density)
                y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] + randn() * 2.5)
            end
        end
        df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
        return df, W, grid_areas

    elseif type_str == "logistic_spatial_r"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        s_coords = unique(df[!, [:s_idx, :s_y]])
        sort!(s_coords, :s_idx)
        r_spatial_true = 0.2 .+ 0.8 * (s_coords.s_y ./ maximum(s_coords.s_y))

        y_sim = zeros(s_N, t_N)
        initial_pop = rand(s_N) * 20.0 .+ 10.0
        y_sim[:, 1] = initial_pop
        K_true = 100.0

        for t in 2:t_N
            for s in 1:s_N
                N_prev = y_sim[s, t-1]
                D_prev = N_prev / grid_areas[s]
                K_density = K_true / grid_areas[s]
                growth = r_spatial_true[s] * D_prev * (1.0 - D_prev / K_density)
                y_sim[s, t] = max(0.0, N_prev + growth * grid_areas[s] + randn() * 2.5)
            end
        end
        df.y = repeat(vec(y_sim'), inner=n_obs_per_st_unit)
        return df, W, grid_areas

    elseif type_str == "leslie_matrix"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )
        survival_true = [0.5, 0.8]
        fecundity_true = [0.0, 1.5, 3.0]

        L_true = zeros(n_age_classes, n_age_classes)
        for i in 1:(n_age_classes - 1)
            L_true[i+1, i] = survival_true[i]
        end
        L_true[1, :] = fecundity_true

        q_true = fill(0.005, n_age_classes)
        effort_sim = use_effort ? rand(s_N, t_N) .* 10.0 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N, n_age_classes) .* 2.0 :
                      zeros(s_N, t_N, n_age_classes)

        pop_sim = zeros(s_N, t_N, n_age_classes)
        initial_total_pop = rand(s_N) * 50.0 .+ 20.0
        for s in 1:s_N
            pop_sim[s, 1, :] = initial_total_pop[s] .* NNlib.softmax(randn(n_age_classes))
        end

        for t in 2:t_N
            for s in 1:s_N
                N_prev = pop_sim[s, t-1, :]
                C_prev = zeros(n_age_classes)
                if use_effort
                    C_prev .+= q_true .* effort_sim[s, t-1] .* N_prev
                end
                if use_removal
                    C_prev .+= removal_sim[s, t-1, :]
                end
                N_after_removal = max.(0.0, N_prev - C_prev)

                N_projected = L_true * N_after_removal
                pop_sim[s, t, :] = max.(0.0, N_projected .+ randn(n_age_classes) .* 0.5)
            end
        end

        for a in 1:n_age_classes
            age_col_name = Symbol("age_$(a)")
            age_data_flat = vec(pop_sim[:, :, a]')
            df[!, age_col_name] = repeat(age_data_flat, inner=n_obs_per_st_unit)
        end
        df.y = df.age_1

        if use_effort
            df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
        end
        if use_removal
            tot_rem = vec(sum(removal_sim, dims=3)[:, :, 1]')
            df.removal_total = repeat(tot_rem, inner=n_obs_per_st_unit)
        end

        return df, W, grid_areas, n_age_classes

    elseif type_str == "dirichlet_multinomial"
        Random.seed!(actual_seed)
        centroids = rand(n_units, 2) .* 10.0
        points_for_partition = vcat(
            [centroids[i,:]' .+ randn(n_obs_per_unit, 2) for i in 1:n_units]...
        )

        au = assign_spatial_units(
            points_for_partition[:, 1], points_for_partition[:, 2];
            target_units=n_units, area_method=:kvt
        )
        W = au.W

        s_coords = vcat([centroids[i,:]' .+ randn(n_obs_per_unit, 2) for i in 1:n_units]...)
        s_idx = vcat([fill(i, n_obs_per_unit) for i in 1:n_units]...)

        dist_matrix = pairwise(Euclidean(), centroids, dims=1)
        alpha_fields = zeros(n_units, n_categories)
        for k in 1:n_categories
            length_scale = rand(1.5:0.1:3.0)
            sigma_f = rand(0.8:0.1:1.2)
            K = sigma_f^2 .* exp.(-0.5 .* (dist_matrix ./ length_scale).^2) + I * 1e-6
            alpha_fields[:, k] = rand(MvNormal(zeros(n_units), K))
        end

        total_counts_per_obs = rand(80:150, n_units * n_obs_per_unit)
        category_counts = zeros(Int, n_units * n_obs_per_unit, n_categories)

        for i in 1:(n_units * n_obs_per_unit)
            unit_idx = s_idx[i]
            alphas = exp.(alpha_fields[unit_idx, :])
            proportions = rand(Dirichlet(alphas))
            category_counts[i, :] = rand(Multinomial(total_counts_per_obs[i], proportions))
        end

        df = DataFrame(
            s_x = s_coords[:, 1],
            s_y = s_coords[:, 2],
            s_idx = s_idx
        )
        for k in 1:n_categories
            df[!, Symbol("cat_", k)] = category_counts[:, k]
        end

        return df, W

    elseif type_str == "generalized_leslie_matrix"
        df, W, grid_areas = create_base_st_data(
            s_N=s_N, t_N=t_N, n_obs_per_st_unit=n_obs_per_st_unit, seed=actual_seed
        )

        A_true = zeros(n_classes, n_classes)
        if n_classes == 4
            A_true = [
                0.0  0.1  1.5  2.0;
                0.5  0.2  0.0  0.0;
                0.0  0.6  0.3  0.0;
                0.0  0.0  0.7  0.4
            ]
        else
            for i in 1:n_classes
                A_true[1, i] = rand() * 0.5
                if i > 1
                    A_true[i, i-1] = rand(0.4:0.1:0.8)
                end
                A_true[i, i] = rand(0.1:0.1:0.4)
            end
        end

        s_coords = unique(df[!, [:s_idx, :s_x]])
        sort!(s_coords, :s_idx)
        K_spatial_true = 50.0 .+ 150.0 * (s_coords.s_x ./ maximum(s_coords.s_x))

        q_true = fill(0.01, n_classes)
        effort_sim = use_effort ? rand(s_N, t_N) .* 5.0 : zeros(s_N, t_N)
        removal_sim = use_removal ? rand(s_N, t_N, n_classes) .* 1.0 :
                      zeros(s_N, t_N, n_classes)

        pop_sim = zeros(s_N, t_N, n_classes)
        initial_total_pop = rand(s_N) .* 40.0 .+ 10.0
        for s in 1:s_N
            pop_sim[s, 1, :] = initial_total_pop[s] .* NNlib.softmax(randn(n_classes))
        end

        for t in 2:t_N
            for s in 1:s_N
                N_prev = pop_sim[s, t-1, :]
                C_prev = zeros(n_classes)
                if use_effort
                    C_prev .+= q_true .* effort_sim[s, t-1] .* N_prev
                end
                if use_removal
                    C_prev .+= removal_sim[s, t-1, :]
                end
                N_after_removal = max.(0.0, N_prev - C_prev)

                L_effective = copy(A_true)
                total_pop_prev = sum(N_after_removal)
                K_s = K_spatial_true[s]
                dd_factor = max(0.0, 1.0 - total_pop_prev / K_s)
                L_effective[1, :] .*= dd_factor

                N_projected = L_effective * N_after_removal
                pop_sim[s, t, :] = max.(0.0, N_projected .+ randn(n_classes) .* 0.5)
            end
        end

        for a in 1:n_classes
            class_col_name = Symbol("class_$(a)")
            class_data_flat = vec(pop_sim[:, :, a]')
            df[!, class_col_name] = repeat(class_data_flat, inner=n_obs_per_st_unit)
            if use_removal
                rem_flat = vec(removal_sim[:, :, a]')
                df[!, Symbol("removal_class_$(a)")] = repeat(rem_flat, inner=n_obs_per_st_unit)
            end
        end
        df.y = df.class_1

        if use_effort
            df.effort = repeat(vec(effort_sim'), inner=n_obs_per_st_unit)
        end
        if use_removal
            tot_rem = vec(sum(removal_sim, dims=3)[:, :, 1]')
            df.removal_total = repeat(tot_rem, inner=n_obs_per_st_unit)
        end

        return df, W, grid_areas, n_classes

    else
        throw(ArgumentError("Unknown dataset type '$type'. Supported types are: " *
            "scottish_lip, hierarchical, bathymetry, substrate, temperature, " *
            "species_composition, snow_crab, ordinal, sim, lgcp_regular, " *
            "lgcp_irregular, advanced, logistic, delay_difference, glv, lotka_volterra, " *
            "leslie_logistic, logistic_spatial_k, logistic_spatial_r, leslie_matrix, " *
            "dirichlet_multinomial, generalized_leslie_matrix."))
    end
end

# ==============================================================================
# Internal Helper Functions & Multi-Tier Ecosystem Generator
# ==============================================================================

function create_base_st_data(;
    s_N::Int=10, t_N::Int=5, n_obs_per_st_unit::Int=1, seed::Int=123
)
    Random.seed!(seed)
    s_x = rand(s_N) * 10.0
    s_y = rand(s_N) * 10.0
    W = spzeros(Bool, s_N, s_N)
    for i in 1:s_N
        if i > 1
            W[i, i-1] = true
        end
        if i < s_N
            W[i, i+1] = true
        end
    end
    W = max.(W, W')
    grid_areas = rand(s_N) * 5.0 .+ 1.0
    s_idx_flat = repeat(1:s_N, inner=t_N * n_obs_per_st_unit)
    t_idx_flat = repeat(repeat(1:t_N, inner=n_obs_per_st_unit), s_N)
    s_x_flat = repeat(s_x, inner=t_N * n_obs_per_st_unit)
    s_y_flat = repeat(s_y, inner=t_N * n_obs_per_st_unit)
    df = DataFrame(
        s_idx=s_idx_flat,
        year=t_idx_flat,
        t_idx=t_idx_flat,
        s_x=s_x_flat,
        s_y=s_y_flat,
        grid_area_col=repeat(grid_areas, inner=t_N * n_obs_per_st_unit)
    )
    return df, W, grid_areas
end




"""
    generate_mock_hierarchical_datasets(; seed=42, N_bathy=1000, N_sub=500,
        N_temp_per_year=100, N_hauls_per_year=50, N_crab_per_year=40) -> NamedTuple

Generates a complete multi-tier synthetic marine ecological dataset bundle with independent
sampling geometries across all 6 tiers:
1. `bathymetry`: N = 1,000 continuous bathymetric soundings with depth variations.
2. `substrate`: N = 500 benthic grab stations with log grain size measurements.
3. `temperature`: N = 1,000 hydrographic CTD casts across 10 years and multiple seasons.
4. `species_composition`: N = 15,000 trawl haul records across 30 marine fish & invertebrate species.
5. `snow_crab`: N = 400 survey tows capturing target species biomass and demographics.
6. `individuals`: Biological sub-sample records (30–50 per positive tow) with columns
   `tow_id`, `year`, `month`, `s_x`, `s_y`, `carapace_width_mm`, `size_bin` (1–5),
   `sex` (0=female, 1=male), `maturity` (0=immature, 1=mature). CW is log-normal with
   a depth/temperature-driven spatial mean mimicking known snow crab growth patterns.
   Sex is a logistic function of CW (male L50 > female L50). Maturity follows sex-specific
   logistic ogives (male L50=65 mm, female L50=45 mm).

# Outputs:
- NamedTuple: `(bathymetry, substrate, temperature, species_composition, snow_crab, individuals)`
"""
function generate_mock_hierarchical_datasets(;
    seed::Int=42,
    N_bathy::Int=1000,
    N_sub::Int=500,
    N_temp_per_year::Int=100,
    N_hauls_per_year::Int=50,
    N_crab_per_year::Int=40
)
    rng = MersenneTwister(seed)

    # --------------------------------------------------------------------------
    # TRUE LATENT PHYSICAL FIELDS (Generative Ground Truth)
    # --------------------------------------------------------------------------
    # True Bathymetry Surface: Shelf basin with banks and submarine canyons
    true_depth(x, y) = 180.0 - 1.2 * x + 0.8 * y + 
                       35.0 * sin(x / 12.0) * cos(y / 12.0) + 
                       15.0 * sin(x / 25.0)

    # True Bathymetric Gradient & Slope Magnitude
    true_slope(x, y) = begin
        dz_dx = -1.2 + (35.0/12.0) * cos(x / 12.0) * cos(y / 12.0) + (15.0/25.0) * cos(x / 25.0)
        dz_dy = 0.8 - (35.0/12.0) * sin(x / 12.0) * sin(y / 12.0)
        sqrt(dz_dx^2 + dz_dy^2)
    end

    # True Substrate Log Grain Size (mm) (Coarse on banks/slopes, fine in deep basins)
    true_grain(x, y) = 1.8 - 0.008 * true_depth(x, y) + 
                       0.15 * true_slope(x, y) + 
                       0.4 * sin(x / 15.0)

    # True Bottom Temperature (°C)
    true_temperature(x, y, month, year) = begin
        z = true_depth(x, y)
        year_idx = year - 2014
        # Thermal stratification + seasonal harmonic lag + interannual warming trend
        5.5 - 0.012 * z + 
        3.2 * sin(2π * (month - 3.0) / 12.0) + 
        0.18 * year_idx + 
        0.5 * sin(x / 20.0) * cos(y / 20.0)
    end

    # --------------------------------------------------------------------------
    # 1. TIER 1: BATHYMETRIC SOUNDINGS
    # --------------------------------------------------------------------------
    bx = rand(rng, Uniform(5.0, 95.0), N_bathy)
    by = rand(rng, Uniform(5.0, 95.0), N_bathy)
    b_depth = [true_depth(x, y) + randn(rng) * 2.5 for (x, y) in zip(bx, by)]

    df_bathy = DataFrame(
        sounding_id = 1:N_bathy,
        s_x = bx,
        s_y = by,
        depth = b_depth
    )

    # --------------------------------------------------------------------------
    # 2. TIER 2: BENTHIC SUBSTRATE SAMPLES
    # --------------------------------------------------------------------------
    sub_x = rand(rng, Uniform(5.0, 95.0), N_sub)
    sub_y = rand(rng, Uniform(5.0, 95.0), N_sub)
    sub_grain = [true_grain(x, y) + randn(rng) * 0.35 for (x, y) in zip(sub_x, sub_y)]
    sub_class = [g < -0.5 ? "Mud/Silt" : (g > 1.0 ? "Gravel/Cobble" : "Sand") for g in sub_grain]

    df_substrate = DataFrame(
        station_id = 1:N_sub,
        s_x = sub_x,
        s_y = sub_y,
        grain_size_phi = sub_grain,
        log_grain_size = sub_grain,
        grain = sub_grain,
        substrate_type = sub_class
    )

    # --------------------------------------------------------------------------
    # 3. TIER 3: SPATIOTEMPORAL BOTTOM TEMPERATURE (10 Years)
    # --------------------------------------------------------------------------
    years = 2015:2024
    temp_records = DataFrame()

    for yr in years
        tx = rand(rng, Uniform(5.0, 95.0), N_temp_per_year)
        ty = rand(rng, Uniform(5.0, 95.0), N_temp_per_year)
        months = rand(rng, [3, 4, 5, 6, 7, 8, 9, 10, 11], N_temp_per_year)
        t_vals = [true_temperature(x, y, m, yr) + randn(rng) * 0.45 for (x, y, m) in zip(tx, ty, months)]

        df_yr = DataFrame(
            cast_id = string("CTD_", yr, "_", lpad.(1:N_temp_per_year, 3, "0")),
            year = fill(yr, N_temp_per_year),
            month = months,
            s_x = tx,
            s_y = ty,
            bottom_temperature = t_vals
        )
        append!(temp_records, df_yr)
    end
    df_temperature = temp_records

    # --------------------------------------------------------------------------
    # 4. TIER 4: MULTI-SPECIES COMPOSITION (30 Species)
    # --------------------------------------------------------------------------
    species_list = [
        "Gadus_morhua", "Melanogrammus_aeglefinus", "Pollachius_virens", "Sebastes_fasciatus",
        "Hippoglossus_hippoglossus", "Reinhardtius_hippoglossoides", "Glyptocephalus_cynoglossus",
        "Hippoglossoides_platessoides", "Limanda_ferruginea", "Pseudopleuronectes_americanus",
        "Merluccius_bilinearis", "Urophycis_tenuis", "Anarhichas_lupus", "Amblyraja_radiata",
        "Malacoraja_senta", "Squalus_acanthias", "Clupea_harengus", "Scomber_scombrus",
        "Mallotus_villosus", "Ammodytes_dubius", "Illex_illecebrosus", "Pandalus_borealis",
        "Homarus_americanus", "Placopecten_magellanicus", "Strongylocentrotus_droebachiensis",
        "Ophiura_sarsii", "Pagurus_acadienus", "Cancer_irroratus", "Hyas_coarctatus", "Lithodes_maja"
    ]
    K_species = length(species_list)

    species_opt_temp = rand(rng, Uniform(1.5, 9.0), K_species)
    species_opt_depth = rand(rng, Uniform(60.0, 260.0), K_species)
    species_opt_grain = rand(rng, Uniform(-1.5, 2.0), K_species)
    species_base_density = rand(rng, Uniform(5.0, 80.0), K_species)

    species_records = DataFrame()

    for yr in years
        hx = rand(rng, Uniform(5.0, 95.0), N_hauls_per_year)
        hy = rand(rng, Uniform(5.0, 95.0), N_hauls_per_year)
        months = rand(rng, [6, 7, 8, 9, 10], N_hauls_per_year)
        efforts = rand(rng, Uniform(0.75, 1.35), N_hauls_per_year)

        for h in 1:N_hauls_per_year
            x, y, m, eff = hx[h], hy[h], months[h], efforts[h]
            z_loc = true_depth(x, y)
            g_loc = true_grain(x, y)
            t_loc = true_temperature(x, y, m, yr)
            haul_tag = string("HAUL_", yr, "_", lpad(h, 3, "0"))

            for k in 1:K_species
                t_suit = exp(-0.5 * ((t_loc - species_opt_temp[k]) / 2.2)^2)
                z_suit = exp(-0.5 * ((z_loc - species_opt_depth[k]) / 45.0)^2)
                g_suit = exp(-0.5 * ((g_loc - species_opt_grain[k]) / 1.2)^2)

                lambda_bio = species_base_density[k] * t_suit * z_suit * g_suit * eff

                p_presence = 1.0 - exp(-0.15 * lambda_bio)
                if rand(rng) < p_presence
                    catch_kg = rand(rng, Gamma(2.5, max(0.1, lambda_bio) / 2.5))
                    catch_cnt = rand(rng, Poisson(max(1.0, lambda_bio * 3.0)))
                else
                    catch_kg = 0.0
                    catch_cnt = 0
                end

                push!(species_records, (
                    haul_id = haul_tag,
                    year = yr,
                    month = m,
                    s_x = x,
                    s_y = y,
                    swept_area_km2 = eff,
                    species = species_list[k],
                    biomass_kg = round(catch_kg, digits=2),
                    count = catch_cnt
                ))
            end
        end
    end
    df_species_comp = species_records

    # --------------------------------------------------------------------------
    # 5. TIER 5: SNOW CRAB ABUNDANCE & DEMOGRAPHICS (10 Years)
    # --------------------------------------------------------------------------
    crab_records = DataFrame()

    for yr in years
        cx = rand(rng, Uniform(5.0, 95.0), N_crab_per_year)
        cy = rand(rng, Uniform(5.0, 95.0), N_crab_per_year)
        months = rand(rng, [6, 7, 8, 9], N_crab_per_year)
        efforts = rand(rng, Uniform(0.8, 1.2), N_crab_per_year)

        for c in 1:N_crab_per_year
            x, y, m, eff = cx[c], cy[c], months[c], efforts[c]
            z_loc = true_depth(x, y)
            g_loc = true_grain(x, y)
            t_loc = true_temperature(x, y, m, yr)
            tow_tag = string("CRAB_", yr, "_", lpad(c, 3, "0"))

            t_niche = exp(-0.5 * ((t_loc - 2.5) / 1.8)^2)
            z_niche = exp(-0.5 * ((z_loc - 140.0) / 40.0)^2)
            g_niche = exp(-0.5 * ((g_loc - (-0.8)) / 0.9)^2)

            base_abundance = 120.0 * t_niche * z_niche * g_niche * eff

            p_pres = 1.0 - exp(-0.08 * base_abundance)
            if rand(rng) < p_pres
                total_kg = rand(rng, Gamma(3.0, max(0.5, base_abundance) / 3.0))
                mature_male_kg   = round(total_kg * rand(rng, Uniform(0.40, 0.65)), digits=2)
                immature_male_kg = round(total_kg * rand(rng, Uniform(0.15, 0.35)), digits=2)
                female_kg        = round(max(0.0, total_kg - mature_male_kg - immature_male_kg), digits=2)
                total_count      = round(Int, total_kg * rand(rng, Uniform(1.8, 3.2)))
            else
                total_kg = 0.0
                mature_male_kg = 0.0
                immature_male_kg = 0.0
                female_kg = 0.0
                total_count = 0
            end

            push!(crab_records, (
                tow_id = tow_tag,
                year = yr,
                month = m,
                s_x = x,
                s_y = y,
                swept_area_km2 = eff,
                total_biomass_kg = round(total_kg, digits=2),
                mature_male_kg = mature_male_kg,
                immature_male_kg = immature_male_kg,
                female_kg = female_kg,
                total_count = total_count
            ))
        end
    end
    df_snow_crab = crab_records

    # --------------------------------------------------------------------------
    # 6. TIER 6: INDIVIDUAL BIOLOGICAL SAMPLING (Carapace Width, Sex, Maturity)
    # --------------------------------------------------------------------------
    # Size bins: [0, 40), [40, 60), [60, 80), [80, 100), [100, Inf) mm CW  (bins 1–5)
    cw_breaks = [0.0, 40.0, 60.0, 80.0, 100.0, Inf]

    # Logistic ogive helper: P(mature | CW, sex) with sex-specific L50
    #   male  L50 = 65 mm, slope = 0.12 /mm
    #   female L50 = 45 mm, slope = 0.15 /mm
    function p_mature(cw::Float64, sex::Int)::Float64
        L50    = sex == 1 ? 65.0 : 45.0
        slope  = sex == 1 ? 0.12 : 0.15
        return 1.0 / (1.0 + exp(-slope * (cw - L50)))
    end

    # Sex probability: P(male) = logistic(-0.3 + 0.006 * CW)
    # Larger individuals are slightly more likely to be male (growth dimorphism).
    p_male(cw::Float64) = 1.0 / (1.0 + exp(-(-0.3 + 0.006 * cw)))

    bio_records = DataFrame()

    for row in eachrow(df_snow_crab)
        row.total_count == 0 && continue

        x, y, m, yr = row.s_x, row.s_y, row.month, row.year
        z_loc = true_depth(x, y)
        t_loc = true_temperature(x, y, m, yr)

        # Mean CW (mm): larger animals in colder, deeper water.
        # Typical snow crab CW range: 35–130 mm; mean ≈ 60 + depth/4 - 2*temp
        mu_cw_log  = log(max(10.0, 60.0 + 0.1 * z_loc - 2.0 * t_loc))
        sigma_cw_log = 0.25  # log-scale SD ≈ 28% CV on CW

        n_bio = rand(rng, 30:50)
        n_bio = min(n_bio, row.total_count)

        for _ in 1:n_bio
            cw  = exp(rand(rng, Normal(mu_cw_log, sigma_cw_log)))
            cw  = clamp(cw, 5.0, 160.0)
            sex = Int(rand(rng) < p_male(cw))
            mat = Int(rand(rng) < p_mature(cw, sex))
            # Bin assignment (searchsortedfirst locates the first break > cw)
            bin = min(searchsortedfirst(cw_breaks, cw) - 1, length(cw_breaks) - 1)

            push!(bio_records, (
                tow_id             = row.tow_id,
                year               = yr,
                month              = m,
                s_x                = x,
                s_y                = y,
                carapace_width_mm  = round(cw, digits=1),
                size_bin           = bin,
                sex                = sex,
                maturity           = mat
            ))
        end
    end
    df_individuals = bio_records

    return (
        bathymetry         = df_bathy,
        substrate          = df_substrate,
        temperature        = df_temperature,
        species_composition = df_species_comp,
        snow_crab          = df_snow_crab,
        individuals        = df_individuals
    )
end


# =============================================================================
# SECTION: OPEN BATHYMETRY & HYDRODYNAMIC DATA INGESTION
# =============================================================================

"""
    load_open_bathymetry(;
        source = :synthetic,
        bbox = (-68.0, -57.0, 42.0, 48.0),
        grid_resolution = (60, 50),
        seed = 42,
        crs = nothing
    ) -> NamedTuple

Ingest or synthesize high-resolution open-sourced bathymetry data for coastal
shelf environments (e.g., Scotian Shelf, Cabot Strait, and Gulf of St. Lawrence).

# Mathematical & Physical Foundation
Seafloor elevation ``z_{\\text{bottom}}(\\mathbf{s})`` satisfies:
```math
z_{\\text{bottom}}(\\mathbf{s}) \\le 0, \\quad H(\\mathbf{s}) = -z_{\\text{bottom}}(\\mathbf{s})
```
where ``H(\\mathbf{s})`` is the water-column depth in meters.
Topographic slope is evaluated as:
```math
\\text{Slope}(\\mathbf{s}) = \\arctan\\left( \\sqrt{ \\left(\\frac{\\partial z}{\\partial x}\\right)^2 + \\left(\\frac{\\partial z}{\\partial y}\\right)^2 } \\right)
```

# Arguments
- `source`: Bathymetric data source. Supported options:
  - `:synthetic` (default): Realistic regional shelf synthesis featuring coastal
    shallows (0–40 m), offshore banks (30–70 m), basins (150–250 m), submarine
    canyons (e.g. The Gully), shelf break (200 m), and continental slope (up to 2500 m).
  - `filepath::AbstractString`: Path to a local `.nc`, `.tif`, `.csv`, `.duckdb`, or `.jld2` file.
- `bbox::Tuple{Real, Real, Real, Real}`: Geographic bounding box `(min_lon, max_lon, min_lat, max_lat)`.
- `grid_resolution::Tuple{Int, Int}`: Regular grid dimensions `(nx, ny)`. Default: `(60, 50)`.
- `seed::Int`: Random seed for synthetic bathymetric perturbations.
- `crs`: Optional coordinate reference system identifier.

# Returns
A `NamedTuple` containing:
- `lons::Vector{Float64}`: 1D vector of grid longitudes (length `nx`).
- `lats::Vector{Float64}`: 1D vector of grid latitudes (length `ny`).
- `depth::Matrix{Float64}`: 2D matrix of water column depths in meters (positive down, size `nx × ny`).
- `elevation::Matrix{Float64}`: 2D matrix of seafloor elevation in meters (negative below sea level, size `nx × ny`).
- `slope::Matrix{Float64}`: 2D matrix of topographic seabed slope in degrees.
- `is_land::BitMatrix`: Boolean mask indicating terrestrial units (`elevation >= 0.0`).
- `centroids::Vector{Tuple{Float64, Float64}}`: Centroids for each marine cell.
- `polygons::Vector{Vector{Tuple{Float64, Float64}}}`: Closed bounding polygon vertex rings for each cell.
- `depth_vec::Vector{Float64}`: 1D vector of marine unit depths matching `centroids`.
- `bbox::Tuple{Float64, Float64, Float64, Float64}`: Bounding box.
"""
function load_open_bathymetry(;
    source::Union{Symbol, AbstractString} = :synthetic,
    bbox::Union{Nothing, Tuple{<:Real, <:Real, <:Real, <:Real}} = nothing,
    lon_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    lat_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    grid_resolution::Union{Nothing, Tuple{Int, Int}} = nothing,
    resolution_deg::Union{Nothing, Real} = nothing,
    seed::Int = 42,
    crs = nothing
)::NamedTuple
    actual_bbox = if bbox !== nothing
        Float64.(bbox)
    elseif lon_range !== nothing && lat_range !== nothing
        (Float64(lon_range[1]), Float64(lon_range[2]), Float64(lat_range[1]), Float64(lat_range[2]))
    else
        (-68.0, -57.0, 42.0, 48.0)
    end
    min_lon, max_lon, min_lat, max_lat = actual_bbox

    actual_res = if grid_resolution !== nothing
        grid_resolution
    elseif resolution_deg !== nothing
        nx_calc = max(4, round(Int, (max_lon - min_lon) / Float64(resolution_deg)))
        ny_calc = max(4, round(Int, (max_lat - min_lat) / Float64(resolution_deg)))
        (nx_calc, ny_calc)
    else
        (60, 50)
    end
    nx, ny = actual_res

    @assert nx >= 4 && ny >= 4 "Grid resolution must be at least (4, 4)"
    @assert min_lon < max_lon "Bounding box min_lon must be less than max_lon"
    @assert min_lat < max_lat "Bounding box min_lat must be less than max_lat"

    lons = collect(range(min_lon, max_lon, length=nx))
    lats = collect(range(min_lat, max_lat, length=ny))
    dx = (max_lon - min_lon) / max(1, nx - 1)
    dy = (max_lat - min_lat) / max(1, ny - 1)

    elev = zeros(Float64, nx, ny)
    loaded_from_file = false

    # Check for file-based ingestion
    if source isa AbstractString && isfile(source)
        ext = lowercase(splitext(source)[2])
        try
            if ext == ".csv"
                df = CSV.read(source, DataFrame)
                lon_col = filter(c -> occursin("lon", lowercase(string(c))), names(df))
                lat_col = filter(c -> occursin("lat", lowercase(string(c))), names(df))
                z_col   = filter(c -> occursin("elev", lowercase(string(c))) ||
                                      occursin("depth", lowercase(string(c))) ||
                                      occursin("z", lowercase(string(c))), names(df))
                if !isempty(lon_col) && !isempty(lat_col) && !isempty(z_col)
                    pts = [Float64.([df[i, first(lon_col)], df[i, first(lat_col)]]) for i in 1:nrow(df)]
                    tree = KDTree(hcat(pts...))
                    is_depth = occursin("depth", lowercase(string(first(z_col))))
                    for j in 1:ny, i in 1:nx
                        idx, _ = knn(tree, [lons[i], lats[j]], 1)
                        val = Float64(df[first(idx), first(z_col)])
                        elev[i, j] = is_depth ? -abs(val) : val
                    end
                    loaded_from_file = true
                end
            elseif ext == ".jld2"
                d = JLD2.load(source)
                key = haskey(d, "elevation") ? "elevation" : (haskey(d, "bathymetry") ? "bathymetry" : nothing)
                if key !== nothing
                    raw_elev = d[key]
                    if size(raw_elev) == (nx, ny)
                        elev .= Float64.(raw_elev)
                        loaded_from_file = true
                    end
                end
            end
        catch err
            @warn "Failed to parse open bathymetry file '$(source)': $(err). Falling back to synthetic shelf model."
        end
    end

    if !loaded_from_file
        # Realistic continental shelf synthetic bathymetry
        rng = MersenneTwister(seed)
        for j in 1:ny
            y_norm = (lats[j] - min_lat) / (max_lat - min_lat)
            for i in 1:nx
                x_norm = (lons[i] - min_lon) / (max_lon - min_lon)

                # Distance from shelf-edge line (roughly southwest to northeast)
                # Shelf edge runs from (0.0, 0.35) to (1.0, 0.70)
                shelf_edge_y = 0.35 + 0.35 * x_norm
                dist_to_slope = y_norm - shelf_edge_y

                base_elev = if dist_to_slope < -0.05
                    # Continental Slope and Abyss (deep ocean)
                    slope_t = clamp((-dist_to_slope - 0.05) / 0.35, 0.0, 1.0)
                    -200.0 - 2200.0 * (slope_t ^ 1.8)
                else
                    # Continental Shelf Platform: depth typically 50m to 220m
                    shelf_t = clamp(dist_to_slope / 0.6, 0.0, 1.0)
                    # Outer shelf banks (shallow offshore features)
                    bank_signal = 55.0 * sin(3.0 * π * x_norm) * cos(2.5 * π * y_norm)
                    # Central shelf basins / troughs
                    basin_signal = -80.0 * exp(-((x_norm - 0.5)^2 + (y_norm - 0.55)^2) / 0.04)
                    # Coastal shallowing
                    coastal_rise = 110.0 * (shelf_t ^ 1.2)
                    -170.0 + bank_signal + basin_signal + coastal_rise
                end

                # Submarine Canyon cut (e.g., The Gully at x ≈ 0.65)
                canyon_dist = abs(x_norm - 0.65)
                if canyon_dist < 0.08 && dist_to_slope < 0.15
                    canyon_depth = 450.0 * (1.0 - canyon_dist / 0.08) * max(0.0, 0.15 - dist_to_slope) / 0.15
                    base_elev -= canyon_depth
                end

                # Micro-topographic soundings roughness
                roughness = 4.0 * (rand(rng) - 0.5)
                elev[i, j] = min(5.0, base_elev + roughness)
            end
        end
    end

    # Water column depth: H = max(0.0, -elevation)
    depth = zeros(Float64, nx, ny)
    is_land = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        if elev[i, j] >= 0.0
            is_land[i, j] = true
            depth[i, j] = 0.0
        else
            depth[i, j] = -elev[i, j]
        end
    end

    # Calculate topographic slope: arctan(sqrt(dz_dx^2 + dz_dy^2))
    slope = zeros(Float64, nx, ny)
    for j in 1:ny
        lat_rad = deg2rad(lats[j])
        dx_m = dx * 111320.0 * cos(lat_rad)
        dy_m = dy * 110540.0
        for i in 1:nx
            dz_dx = if i == 1
                (elev[2, j] - elev[1, j]) / dx_m
            elseif i == nx
                (elev[nx, j] - elev[nx - 1, j]) / dx_m
            else
                (elev[i + 1, j] - elev[i - 1, j]) / (2.0 * dx_m)
            end

            dz_dy = if j == 1
                (elev[i, 2] - elev[i, 1]) / dy_m
            elseif j == ny
                (elev[i, ny] - elev[i, ny - 1]) / dy_m
            else
                (elev[i, j + 1] - elev[i, j - 1]) / (2.0 * dy_m)
            end

            grad_mag = sqrt(dz_dx^2 + dz_dy^2)
            slope[i, j] = rad2deg(atan(grad_mag))
        end
    end

    # Build regular grid polygon cells for seamless LibGEOS geometric resharding
    centroids = Tuple{Float64, Float64}[]
    polygons = Vector{Vector{Tuple{Float64, Float64}}}()
    depth_vec = Float64[]

    half_dx = dx / 2.0
    half_dy = dy / 2.0

    for j in 1:ny, i in 1:nx
        cx = lons[i]
        cy = lats[j]
        # Closed counter-clockwise rectangular bounding box
        poly = [
            (cx - half_dx, cy - half_dy),
            (cx + half_dx, cy - half_dy),
            (cx + half_dx, cy + half_dy),
            (cx - half_dx, cy + half_dy),
            (cx - half_dx, cy - half_dy)
        ]
        push!(centroids, (cx, cy))
        push!(polygons, poly)
        push!(depth_vec, depth[i, j])
    end

    au = (
        centroids = centroids,
        centroids_lonlat = centroids,
        polygons = polygons,
        polygons_lonlat = polygons
    )

    return (
        lons = lons,
        lats = lats,
        grid_lon = [c[1] for c in centroids],
        grid_lat = [c[2] for c in centroids],
        depth = depth,
        depths = depth_vec,
        depth_vec = depth_vec,
        elevation = elev,
        slope = slope,
        slopes = vec(slope),
        is_land = is_land,
        centroids = centroids,
        polygons = polygons,
        au = au,
        bbox = actual_bbox,
        crs = crs
    )
end

"""
    extract_hydrodynamic_dataset(
        hydro_input::Any = nothing;
        bathymetry_data::Union{Nothing, NamedTuple} = nothing,
        bbox::Tuple{<:Real, <:Real, <:Real, <:Real} = (-68.0, -57.0, 42.0, 48.0),
        grid_resolution::Tuple{Int, Int} = (60, 50),
        depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0, -150.0, -250.0],
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = nothing,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = nothing
    ) -> NamedTuple

Extract, compute, and format 3D/4D ocean hydrodynamic fields at specific depths and times.

# Physical Formulations
- **Potential Density** (UNESCO Linear Equation of State):
  ```math
  \\rho(T, S) = \\rho_0 \\left[ 1 - \\alpha (T - T_0) + \\beta (S - S_0) \\right]
  ```
  with ``\\rho_0 = 1025.0\\text{ kg/m}^3``, ``\\alpha = 2.0 \\times 10^{-4}\\text{ K}^{-1}``,
  ``\\beta = 7.6 \\times 10^{-4}\\text{ PSU}^{-1}``.
- **Salinity Stratification & Brunt-Väisälä Frequency**:
  ```math
  N^2 = -\\frac{g}{\\rho_0} \\frac{\\partial \\rho}{\\partial z} \\approx g \\left( \\alpha \\frac{\\partial T}{\\partial z} - \\beta \\frac{\\partial S}{\\partial z} \\right)
  ```
- **Turbulent Eddy Diffusivity & Viscosity**:
  ```math
  \\kappa_v(z) = \\kappa_{\\text{surf}} e^{z / h_{\\text{mix}}} + \\frac{\\kappa_{\\text{bkg}}}{1 + 10 Ri} + \\kappa_{\\text{bbl}} e^{-(H + z) / h_{\\text{bbl}}}
  ```
  where ``Ri = N^2 / [(\\partial u / \\partial z)^2 + (\\partial v / \\partial z)^2]``.

# Arguments
- `hydro_input`: Ocean circulation model instance, NamedTuple, Dict, or `nothing`.
- `bathymetry_data`: Optional bathymetric dataset from `load_open_bathymetry`.
- `bbox`: Geographic spatial domain `(min_lon, max_lon, min_lat, max_lat)`.
- `grid_resolution`: Grid dimensions `(nx, ny)`.
- `depths`: Vertical depth coordinate levels (m, negative below surface).
- `depth`: Target continuous depth for 2D slice extraction.
- `depth_level`: Target vertical level index (1 = surface).
- `time_seconds`: Simulation time in seconds.
- `time_index`: Snapshot time index.

# Returns
A `NamedTuple` containing:
- `lons`, `lats`, `depths`: Coordinate axes.
- `temperature`, `salinity`, `density`: 3D scalar fields (`nx × ny × nz`).
- `stratification_N2`, `salinity_gradient`: 3D stratification diagnostics (`nx × ny × nz`).
- `u`, `v`, `w`, `speed`: 3D velocity fields (`nx × ny × nz`).
- `diffusivity_v`, `viscosity_v`: 3D turbulent mixing fields (`nx × ny × nz`).
- `elevation`: 2D sea surface height (m).
- `bathymetry`: 2D seafloor elevation (m).
- `slice_2d`: NamedTuple containing 2D horizontal slices of all fields at the requested `depth`.
- `centroids`, `polygons`: Areal unit representation for LibGEOS resharding.
"""
function extract_hydrodynamic_dataset(
    hydro_input::Any = nothing;
    bathymetry_data::Union{Nothing, NamedTuple} = nothing,
    bbox::Tuple{<:Real, <:Real, <:Real, <:Real} = (-68.0, -57.0, 42.0, 48.0),
    grid_resolution::Tuple{Int, Int} = (60, 50),
    depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0, -150.0, -250.0],
    depth_levels::Union{Nothing, AbstractVector{<:Real}} = nothing,
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = nothing,
    time_seconds::Union{Nothing, Real} = nothing,
    times::Union{Nothing, AbstractVector{<:Real}} = nothing,
    time_index::Union{Nothing, Int} = nothing
)::NamedTuple
    # Auto-detect if bathymetry NamedTuple was supplied as first positional argument
    if hydro_input isa NamedTuple && (hasproperty(hydro_input, :elevation) || hasproperty(hydro_input, :depth)) && !hasproperty(hydro_input, :temperature)
        bathymetry_data = hydro_input
        hydro_input = nothing
    end

    bathy = if bathymetry_data !== nothing
        bathymetry_data
    else
        load_open_bathymetry(source=:synthetic, bbox=bbox, grid_resolution=grid_resolution)
    end

    lons = bathy.lons
    lats = bathy.lats
    nx = length(lons)
    ny = length(lats)

    actual_depths = if depth_levels !== nothing
        collect(Float64, depth_levels)
    else
        collect(Float64, depths)
    end
    nz = length(actual_depths)
    z_levels = [z > 0.0 ? -z : z for z in actual_depths]

    # Resolve target depth index
    active_k = 1
    if depth !== nothing
        target_z = depth > 0.0 ? -Float64(depth) : Float64(depth)
        _, active_k = findmin(abs.(z_levels .- target_z))
    elseif depth_level !== nothing
        active_k = clamp(depth_level, 1, nz)
    end
    resolved_depth_m = z_levels[active_k]

    # Initialize 3D field arrays (nx × ny × nz)
    T_mat = zeros(Float64, nx, ny, nz)
    S_mat = zeros(Float64, nx, ny, nz)
    u_mat = zeros(Float64, nx, ny, nz)
    v_mat = zeros(Float64, nx, ny, nz)
    w_mat = zeros(Float64, nx, ny, nz)
    diff_v = zeros(Float64, nx, ny, nz)
    visc_v = zeros(Float64, nx, ny, nz)

    t_sec = time_seconds !== nothing ? Float64(time_seconds) :
            (time_index !== nothing ? Float64(time_index * 3600.0) : 0.0)

    # 1. Ingest from external model or NamedTuple if provided
    if hydro_input isa NamedTuple || hydro_input isa AbstractDict
        get_f(k_list, def) = begin
            for k in k_list
                if hydro_input isa NamedTuple && hasproperty(hydro_input, k)
                    return getproperty(hydro_input, k)
                elseif hydro_input isa AbstractDict && (haskey(hydro_input, k) || haskey(hydro_input, string(k)))
                    return haskey(hydro_input, k) ? hydro_input[k] : hydro_input[string(k)]
                end
            end
            return def
        end
        raw_u = get_f((:u, :u_velocity), nothing)
        raw_v = get_f((:v, :v_velocity), nothing)
        raw_t = get_f((:temperature, :T, :temp), nothing)
        raw_s = get_f((:salinity, :S, :sal), nothing)

        if raw_u !== nothing && size(raw_u, 1) == nx && size(raw_u, 2) == ny
            u_mat .= ndims(raw_u) == 2 ? repeat(raw_u, 1, 1, nz) : raw_u[:, :, 1:min(nz, size(raw_u, 3))]
        end
        if raw_v !== nothing && size(raw_v, 1) == nx && size(raw_v, 2) == ny
            v_mat .= ndims(raw_v) == 2 ? repeat(raw_v, 1, 1, nz) : raw_v[:, :, 1:min(nz, size(raw_v, 3))]
        end
        if raw_t !== nothing && size(raw_t, 1) == nx && size(raw_t, 2) == ny
            T_mat .= ndims(raw_t) == 2 ? repeat(raw_t, 1, 1, nz) : raw_t[:, :, 1:min(nz, size(raw_t, 3))]
        end
        if raw_s !== nothing && size(raw_s, 1) == nx && size(raw_s, 2) == ny
            S_mat .= ndims(raw_s) == 2 ? repeat(raw_s, 1, 1, nz) : raw_s[:, :, 1:min(nz, size(raw_s, 3))]
        end
    else
        # 2. Physics-based synthetic regional shelf circulation
        min_lon, max_lon, min_lat, max_lat = bathy.bbox
        for j in 1:ny
            y_norm = (lats[j] - min_lat) / (max_lat - min_lat)
            for i in 1:nx
                x_norm = (lons[i] - min_lon) / (max_lon - min_lon)
                h_bed = bathy.depth[i, j]

                # Tidal and seasonal modulation
                t_phase = 2.0 * π * (t_sec / 44714.0)
                u_tide = 0.08 * sin(t_phase + 2.0 * x_norm)
                v_tide = 0.06 * cos(t_phase + 1.5 * y_norm)

                for k in 1:nz
                    z = z_levels[k]

                    # If level is below seabed, mask as NaN
                    if abs(z) > h_bed
                        T_mat[i, j, k] = NaN
                        S_mat[i, j, k] = NaN
                        u_mat[i, j, k] = NaN
                        v_mat[i, j, k] = NaN
                        w_mat[i, j, k] = NaN
                        diff_v[i, j, k] = NaN
                        visc_v[i, j, k] = NaN
                        continue
                    end

                    # Temperature (°C): Surface warm, CIL minimum at -50m, slope warm
                    t_surface = 15.5 - 3.2 * y_norm + 1.5 * x_norm
                    t_cil = 2.0 + 0.9 * sin(π * x_norm)
                    t_slope = 7.8 + 1.2 * (1.0 - y_norm)

                    t_val = if z > -20.0
                        t_surface + (z / 20.0) * (t_surface - 6.0)
                    elseif z > -75.0
                        t_cil + ((z + 50.0) / 35.0)^2 * 2.8
                    else
                        t_cil + ((abs(z) - 75.0) / 100.0) * (t_slope - t_cil)
                    end
                    T_mat[i, j, k] = clamp(t_val, 0.2, 19.0)

                    # Salinity (PSU): Fresher coastal runoff to saline deep slope
                    s_val = 31.2 + 1.9 * (1.0 - y_norm) + 1.3 * x_norm + (abs(z) / 150.0) * 1.4
                    S_mat[i, j, k] = clamp(s_val, 29.8, 35.8)

                    # Advection: Southwestward Nova Scotia Current along coastal shelf
                    z_atten = exp(z / 80.0)
                    u_mean = (-0.18 - 0.12 * y_norm) * z_atten
                    v_mean = (-0.10 - 0.08 * (1.0 - x_norm)) * z_atten

                    u_mat[i, j, k] = u_mean + u_tide * (1.0 + z / 200.0)
                    v_mat[i, j, k] = v_mean + v_tide * (1.0 + z / 200.0)

                    # Vertical velocity w (m/s): Upwelling along shelf break
                    w_up = 0.0004 * sin(2.0 * π * x_norm) * cos(π * y_norm) * (z / max(1.0, h_bed))
                    w_mat[i, j, k] = w_up
                end
            end
        end
    end

    # Physical Density and Stratification Computation
    # UNESCO Linear Equation of State
    rho0 = 1025.0
    alpha_t = 2.0e-4
    beta_s = 7.6e-4
    T0 = 10.0
    S0 = 35.0
    g = 9.80665

    rho_mat = zeros(Float64, nx, ny, nz)
    strat_N2 = zeros(Float64, nx, ny, nz)
    sal_grad = zeros(Float64, nx, ny, nz)

    for k in 1:nz, j in 1:ny, i in 1:nx
        t = T_mat[i, j, k]
        s = S_mat[i, j, k]
        if !isnan(t) && !isnan(s)
            rho_mat[i, j, k] = rho0 * (1.0 - alpha_t * (t - T0) + beta_s * (s - S0))
        else
            rho_mat[i, j, k] = NaN
        end
    end

    # Vertical gradients
    for j in 1:ny, i in 1:nx
        for k in 1:nz
            if isnan(rho_mat[i, j, k])
                strat_N2[i, j, k] = NaN
                sal_grad[i, j, k] = NaN
                diff_v[i, j, k] = NaN
                visc_v[i, j, k] = NaN
                continue
            end

            d_rho_dz = if k == 1 && nz > 1
                (rho_mat[i, j, 1] - rho_mat[i, j, 2]) / (z_levels[1] - z_levels[2])
            elseif k == nz && nz > 1
                (rho_mat[i, j, nz - 1] - rho_mat[i, j, nz]) / (z_levels[nz - 1] - z_levels[nz])
            elseif nz >= 3
                (rho_mat[i, j, k - 1] - rho_mat[i, j, k + 1]) / (z_levels[k - 1] - z_levels[k + 1])
            else
                0.0
            end

            ds_dz = if k == 1 && nz > 1
                (S_mat[i, j, 1] - S_mat[i, j, 2]) / (z_levels[1] - z_levels[2])
            elseif k == nz && nz > 1
                (S_mat[i, j, nz - 1] - S_mat[i, j, nz]) / (z_levels[nz - 1] - z_levels[nz])
            elseif nz >= 3
                (S_mat[i, j, k - 1] - S_mat[i, j, k + 1]) / (z_levels[k - 1] - z_levels[k + 1])
            else
                0.0
            end

            sal_grad[i, j, k] = ds_dz
            n2_val = -(g / rho0) * d_rho_dz
            strat_N2[i, j, k] = max(1e-7, n2_val)

            # Turbulent Eddy Diffusivity & Viscosity
            du_dz = (nz > 1 && k < nz) ? (u_mat[i, j, k] - u_mat[i, j, k+1]) / (z_levels[k] - z_levels[k+1]) : 0.005
            dv_dz = (nz > 1 && k < nz) ? (v_mat[i, j, k] - v_mat[i, j, k+1]) / (z_levels[k] - z_levels[k+1]) : 0.005
            shear2 = max(1e-6, du_dz^2 + dv_dz^2)
            ri = clamp(strat_N2[i, j, k] / shear2, 0.05, 50.0)

            z = z_levels[k]
            h_bed = bathy.depth[i, j]
            k_surf = 1.2e-2 * exp(z / 15.0)
            k_pyc  = 1.5e-4 / (1.0 + 8.0 * ri)
            dist_to_bed = max(0.5, h_bed + z)
            k_bbl  = 2.5e-3 * exp(-dist_to_bed / 12.0)

            diff_v[i, j, k] = clamp(k_surf + k_pyc + k_bbl, 1e-5, 0.05)
            visc_v[i, j, k] = diff_v[i, j, k] * (1.0 + 0.5 * ri)
        end
    end

    spd_mat = hypot.(u_mat, v_mat)

    n_cells = nx * ny
    T_2d = reshape(T_mat, n_cells, nz)
    S_2d = reshape(S_mat, n_cells, nz)
    rho_2d = reshape(rho_mat, n_cells, nz)
    strat_N2_2d = reshape(strat_N2, n_cells, nz)
    sal_grad_2d = reshape(sal_grad, n_cells, nz)
    u_2d = reshape(u_mat, n_cells, nz)
    v_2d = reshape(v_mat, n_cells, nz)
    w_2d = reshape(w_mat, n_cells, nz)
    diff_v_2d = reshape(diff_v, n_cells, nz)
    visc_v_2d = reshape(visc_v, n_cells, nz)
    spd_2d = reshape(spd_mat, n_cells, nz)

    # Habitat suitability gradient tied to depth and water column temperature
    d_vec = bathy.depth_vec
    t_surf = T_2d[:, 1]
    hsi_raw = exp.(-((d_vec .- 175.0) ./ 60.0).^2 - ((t_surf .- 3.0) ./ 2.5).^2)
    valid_hsi = filter(!isnan, hsi_raw)
    hsi_min, hsi_max = isempty(valid_hsi) ? (0.0, 1.0) : extrema(valid_hsi)
    hsi_vec = (hsi_raw .- hsi_min) ./ max(1e-6, hsi_max - hsi_min) .* 0.85 .+ 0.1

    # Extract active 2D horizontal slice at resolved_depth_m
    slice_2d = (
        depth_m = resolved_depth_m,
        depth_level = active_k,
        temperature = T_mat[:, :, active_k],
        salinity = S_mat[:, :, active_k],
        density = rho_mat[:, :, active_k],
        stratification = strat_N2[:, :, active_k],
        salinity_gradient = sal_grad[:, :, active_k],
        u = u_mat[:, :, active_k],
        v = v_mat[:, :, active_k],
        w = w_mat[:, :, active_k],
        speed = spd_mat[:, :, active_k],
        diffusivity = diff_v[:, :, active_k],
        viscosity = visc_v[:, :, active_k]
    )

    return (
        lons = lons,
        lats = lats,
        depths = abs.(actual_depths),
        depth_levels = abs.(actual_depths),
        temperature = T_2d,
        temperature_3d = T_mat,
        salinity = S_2d,
        salinity_3d = S_mat,
        density = rho_2d,
        rho = rho_2d,
        density_3d = rho_mat,
        stratification_N2 = strat_N2_2d,
        N2 = strat_N2_2d,
        stratification_3d = strat_N2,
        salinity_gradient = sal_grad_2d,
        u = u_2d,
        advection_u = u_2d,
        v = v_2d,
        advection_v = v_2d,
        w = w_2d,
        speed = spd_2d,
        diffusivity_v = diff_v_2d,
        kappa_v = diff_v_2d,
        viscosity_v = visc_v_2d,
        nu_v = visc_v_2d,
        hsi = hsi_vec,
        elevation = zeros(Float64, n_cells),
        bathymetry = bathy.elevation,
        depth_vec = bathy.depth_vec,
        depths_vec = bathy.depth_vec,
        slice_2d = slice_2d,
        centroids = bathy.centroids,
        polygons = bathy.polygons,
        bbox = bathy.bbox
    )
end

