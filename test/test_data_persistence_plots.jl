# ==============================================================================
# TethysBayes Test Suite: Data Generator, Persistence (JLD2/DuckDB), Plots & Magnitude Validation
# ==============================================================================
using Dates
using DataFrames
using JLD2
using DuckDB
using Random
using Distributions
using DynamicPPL

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "TethysBayes_data Monolithic Synthetic Data Generator" begin
    @testset "Default scottish_lip Dataset & Covariate Completeness" begin
        p_out = TethysBayes.TethysBayes_data() # Default "scottish_lip"
        df = p_out.data
        au = p_out.au

        @test df isa DataFrame
        @test au.W isa AbstractMatrix
        @test size(df, 1) > 0

        required_cols = [
            :y, :y_rate, :y_bin, :y_gauss, :y_pois, :ordinal_y,
            :y_cat1, :y_cat2, :y_cat3, :counts, :t_idx, :year, :month, :day,
            :region, :district, :group, :group_id, :grouping_covariate, :cell_area,
            :effort, :removal, :removal_total, :proxy_val, :predator_pop,
            :recruitment, :habitat, :species_1, :species_2, :species_3,
            :age_1, :age_2, :age_3, :class_1, :class_2, :class_3, :class_4,
            :s_idx, :s_x, :s_y
        ]
        for col in required_cols
            @test col in propertynames(df)
        end
    end

    @testset "All Supported Synthetic Dataset Types" begin
        df_ord = TethysBayes.TethysBayes_data("ordinal"; n_obs=100)
        @test df_ord isa DataFrame
        @test :ordinal_y in propertynames(df_ord)
    end
end

if !HAS_PLOTTING
    @testset "Plotting Subsystem" begin
        @test_skip false
        @info "Skipping 'Plotting Subsystem' testset: optional plotting stack not installed."
    end
else
    @testset "Plotting Subsystem" begin
    # 1. Theme generator
    thm = TethysBayes.create_theme(fontsize=11)
    @test haskey(thm, :titlefontsize)
    @test thm.size == (900, 600)

    # 2. Choropleth plotting
    polys = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)]
    ]
    vals = [1.5, 2.5]
    p_choro = TethysBayes.choropleth(polys, vals; title="Test Choro")
    @test p_choro isa Plots.Plot

    # Test flipped argument order
    p_choro2 = TethysBayes.choropleth(vals, polys)
    @test p_choro2 isa Plots.Plot

    # 3. Timeseries with Credible Interval
    ts_p = TethysBayes.timeseries_ci(1:5, [1.0, 2.0, 3.0, 2.5, 2.0], [0.8, 1.7, 2.5, 2.0, 1.5],
        [1.2, 2.3, 3.5, 3.0, 2.5]; title="Trend")
    @test ts_p isa Plots.Plot

    # 4. Spatial Graph Plot
    cents = [(0.5, 0.5), (1.5, 0.5)]
    g = Graphs.SimpleGraph(2)
    Graphs.add_edge!(g, 1, 2)
    sg_p = TethysBayes.spatial_graph_plot(cents, g; polygons=polys)
    @test sg_p isa Plots.Plot

    # Test keyword-based au dispatch
    au_mock = (polygons=polys, centroids=cents, graph=g, hull_coords=polys[1])
    sg_p2 = TethysBayes.spatial_graph_plot(au=au_mock)
    @test sg_p2 isa Plots.Plot

    # 5. Render Paths & Occupancy
    path_mock = [[(0.5, 0.5), (1.5, 0.5)]]
    p_paths = TethysBayes.render_paths!(deepcopy(sg_p), path_mock)
    @test p_paths isa Plots.Plot

    occ_p = TethysBayes.map_point_occupancy(polys, cents, 1, 1)
    @test occ_p isa Plots.Plot

    # 6. Save Plot
    tmp_file = scratch_path("test_TethysBayes_plot.png")
    saved_path = TethysBayes.save_plot(p_choro, tmp_file)
    @test isfile(saved_path)
    try; rm(saved_path; force=true); catch; end

    # 7. Test Spatiotemporal Time Coordinates in TethysBayes_plots
    df_st_mock = DataFrame(
        s_idx = repeat(1:56, inner=10),
        year = repeat(1:10, outer=56),
        y_gauss = randn(560)
    )
    res_mock = (
        effects = Dict(
            :year => (
                structured = (
                    mean = repeat(sin.(1:10), outer=56),
                    lower = repeat(sin.(1:10) .- 0.2, outer=56),
                    upper = repeat(sin.(1:10) .+ 0.2, outer=56)
                ),
            )
        ),
        predictions = (
            denoised = (
                mean = randn(560),
                lower = randn(560) .- 0.5,
                upper = randn(560) .+ 0.5,
                observed = randn(560)
            ),
        )
    )
    M_mock_st = (
        s_N = 56,
        t_N = 10,
        t_idx_var = :year,
        t_idx = df_st_mock.year,
        s_idx = df_st_mock.s_idx,
        data = df_st_mock
    )
    plots_out = TethysBayes._TethysBayes_plots_impl(nothing, nothing, res_mock, M_mock_st; data=df_st_mock)
    @test haskey(plots_out.plots, :temporal)
    @test length(plots_out.plots_data[:temporal].time) == 10
    @test collect(plots_out.plots_data[:temporal].time) == 1:10
end

@testset "Leaflet Interactive HTML Visualization Engine" begin
    polys = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)],
        [(0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)],
        [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
    ]
    cents = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)]
    g = Graphs.SimpleGraph(4)
    Graphs.add_edge!(g, 1, 2)
    Graphs.add_edge!(g, 1, 3)
    Graphs.add_edge!(g, 2, 4)
    Graphs.add_edge!(g, 3, 4)
    W_mock = spzeros(4, 4)
    W_mock[1, 2] = W_mock[2, 1] = 1.0
    W_mock[1, 3] = W_mock[3, 1] = 1.0
    W_mock[2, 4] = W_mock[4, 2] = 1.0
    W_mock[3, 4] = W_mock[4, 3] = 1.0

    au = (polygons=polys, centroids=cents, graph=g, W=W_mock, hull_coords=polys[1])
    hsi = [0.2, 0.8, 0.5, 0.9]
    Gamma = [0.4 0.3 0.3 0.0;
             0.2 0.5 0.0 0.3;
             0.2 0.0 0.5 0.3;
             0.0 0.2 0.2 0.6]

    # 1. Leaflet Choropleth
    m_choro = TethysBayes.leaflet_choropleth(polys, hsi; title="Test HSI Choropleth")
    @test m_choro isa TethysBayes.LeafletMap
    @test occursin("<!DOCTYPE html>", m_choro.html_content)
    @test occursin("leaflet", m_choro.html_content)

    # 2. Leaflet Spatial Graph
    m_graph = TethysBayes.leaflet_spatial_graph(au; title="Test Spatial Graph")
    @test m_graph isa TethysBayes.LeafletMap

    # 3. Leaflet HSI, Diffusion, and Residence Time Maps
    m_hsi = TethysBayes.leaflet_hsi_map(hsi, au; title="HSI Leaflet")
    @test m_hsi isa TethysBayes.LeafletMap

    m_diff = TethysBayes.leaflet_diffusion_map(0.45, au; title="Diffusion Leaflet")
    @test m_diff isa TethysBayes.LeafletMap

    m_res = TethysBayes.leaflet_residence_time_map(Gamma, au; title="Residence Time Leaflet")
    @test m_res isa TethysBayes.LeafletMap

    # 4. Leaflet Advection Arrows / Velocity Field
    m_arr = TethysBayes.leaflet_advection_arrows(au; hsi=hsi, velocity=1.2, Gamma=Gamma, title="Advection Leaflet")
    @test m_arr isa TethysBayes.LeafletMap

    # 5. Leaflet Spatiotemporal Time Slider Map
    st_dict = Dict(2020 => [0.1, 0.3, 0.5, 0.7], 2021 => [0.2, 0.4, 0.6, 0.8], 2022 => [0.3, 0.5, 0.7, 0.9])
    m_st = TethysBayes.leaflet_spacetime_map([2020, 2021, 2022], st_dict, au; title="Spatiotemporal Evolution")
    @test m_st isa TethysBayes.LeafletMap
    @test occursin("timeSlider", m_st.html_content)

    # 6. HTML Persistence & save_plot / save_plots
    tmp_html = scratch_path("test_leaflet_st.html")
    saved_html = TethysBayes.save_plot(m_st, tmp_html)
    @test isfile(saved_html)
    @test filesize(saved_html) > 1000
    try; rm(saved_html; force=true); catch; end

    # Test save_plots with LeafletMap
    p_static = Plots.plot([1, 2], [3, 4])
    plots_bundle = (
        st_map = m_st,
        hsi_map = m_hsi,
        static_p = p_static
    )
    tmp_saved_dir = scratch_path("TethysBayes_test_plots_export")
    saved_files = TethysBayes.save_plots(plots_bundle, tmp_saved_dir)
    @test length(saved_files) == 3
    @test any(f -> endswith(f, "st_map.html"), saved_files)
    @test any(f -> endswith(f, "hsi_map.html"), saved_files)
    @test any(f -> endswith(f, "static_p.png"), saved_files)
    try; rm(tmp_saved_dir; recursive=true, force=true); catch; end

    # 7. Test mode=:leaflet flag across standard API functions
    @test TethysBayes.plot_hsi_choropleth(hsi, au; mode=:leaflet) isa TethysBayes.LeafletMap
    @test TethysBayes.plot_diffusion_map(0.5, au; mode=:leaflet) isa TethysBayes.LeafletMap
    @test TethysBayes.plot_residence_time_map(Gamma, au; mode=:leaflet) isa TethysBayes.LeafletMap
    @test TethysBayes.plot_advection_arrows(au; hsi=hsi, mode=:leaflet) isa TethysBayes.LeafletMap
    @test TethysBayes.choropleth(polys, hsi; mode=:leaflet) isa TethysBayes.LeafletMap
    @test TethysBayes.spatial_graph_plot(au=au, mode=:leaflet) isa TethysBayes.LeafletMap
    end # if HAS_PLOTTING
end

@testset "Model State & Results Persistence (JLD2 & DuckDB)" begin
    # 1. Setup simple model & sampling
    rng = MersenneTwister(123)
    df_test = DataFrame(
        y = rand(rng, 1:10, 25),
        x = randn(rng, 25),
        group = rand(rng, 1:5, 25)
    )
    m_test = @tbm(likelihood(y, family=poisson) ~ intercept() + fixed(x) + random(group,
        model=iid), df_test, verbose=false)
    chn_test = sample(m_test, MH(), 30; progress=false)
    res_test = model_results_comprehensive(m_test, chn_test)

    temp_dir = scratch_dir()
    jld2_path = joinpath(temp_dir, "test_model.jld2")
    duckdb_path = joinpath(temp_dir, "test_results.duckdb")
    bundle_base = joinpath(temp_dir, "test_bundle")

    # 2. Test JLD2 Model State Save & Load
    saved_file = save_TethysBayes_model(jld2_path, m_test; chain=chn_test,
        metadata=Dict("test_run"=>"v1"))
    @test isfile(saved_file)

    loaded_bundle = load_TethysBayes_model(saved_file)
    @test hasproperty(loaded_bundle, :model)
    @test loaded_bundle.model isa DynamicPPL.Model
    @test hasproperty(loaded_bundle, :chain)
    @test !isnothing(loaded_bundle.chain)
    @test loaded_bundle.metadata["test_run"] == "v1"

    # Test sampling on reloaded model
    chn_reloaded = sample(loaded_bundle.model, MH(), 10; progress=false)
    @test size(chn_reloaded, 1) == 10

    # 3. Test Chain Appending & Extension
    chn_extended = append_posterior_samples(chn_test, chn_reloaded)
    @test size(chn_extended, 1) == size(chn_test, 1) + size(chn_reloaded, 1)

    chn_ext_sampled = extend_sampling(m_test, chn_test, 10; sampler=MH(), progress=false)
    @test size(chn_ext_sampled, 1) == size(chn_test, 1) + 10

    # 4. Test DuckDB Results Save & Query
    saved_db = save_TethysBayes_results(duckdb_path, res_test; model=m_test, chain=chn_test)
    @test isfile(saved_db)

    # SQL Queries via DuckDB
    df_metrics = query_duckdb(duckdb_path, "SELECT * FROM metrics")
    @test df_metrics isa DataFrame
    @test "metric" in names(df_metrics)

    df_params = query_duckdb(duckdb_path, "SELECT * FROM parameter_stats")
    @test df_params isa DataFrame

    df_preds = query_duckdb(duckdb_path, "SELECT * FROM predictions")
    @test df_preds isa DataFrame
    @test nrow(df_preds) == 25

    # Reload DuckDB results
    res_reloaded = load_TethysBayes_results(duckdb_path)
    @test hasproperty(res_reloaded, :metrics)
    @test hasproperty(res_reloaded, :parameters)
    @test hasproperty(res_reloaded, :predictions)
    @test hasproperty(res_reloaded, :plots_data)

    # 5. Test Unified Bundle Save & Load
    bundle_paths = save_TethysBayes_bundle(bundle_base, m_test, chn_test, res_test;
        metadata=Dict("experiment"=>"bundle_test"))
    @test isfile(bundle_paths.model_file)
    @test isfile(bundle_paths.results_file)

    loaded_all = load_TethysBayes_bundle(bundle_base)
    @test loaded_all.model isa DynamicPPL.Model
    @test !isnothing(loaded_all.chain)
    @test hasproperty(loaded_all.results, :metrics)
    @test loaded_all.metadata["experiment"] == "bundle_test"

    # 6. Test Sequential Prior Extraction
    extracted_priors = extract_posterior_priors(res_test)
    @test haskey(extracted_priors, :intercept)
    @test extracted_priors[:intercept] isa Normal

    # 7. Test Multi-Model Ensembling & BMA
    ens_db = joinpath(temp_dir, "ensemble.duckdb")
    m2_test = @tbm(likelihood(y, family=poisson) ~ intercept() + fixed(x), df_test,
        verbose=false)
    c2_test = sample(m2_test, MH(), 30; progress=false)
    r2_test = model_results_comprehensive(m2_test, c2_test)

    ensemble_dict = Dict(
        :full => (model=m_test, chain=chn_test, results=res_test),
        :reduced => (model=m2_test, chain=c2_test, results=r2_test)
    )
    df_registry = save_model_ensemble(ens_db, ensemble_dict)
    @test nrow(df_registry) == 2
    @test "bma_weight" in names(df_registry)

    df_bma = bma_weighted_predictions(ens_db)
    @test nrow(df_bma) == 25
    @test "bma_pred_mean" in names(df_bma)

    # 8. Test CSV Export
    csv_path = joinpath(temp_dir, "predictions.csv")
    export_results_to_csv(duckdb_path, "predictions", csv_path)
    @test isfile(csv_path)

    # 9. Test DuckDB Compaction
    compact_duckdb(duckdb_path)

    # 10. Test GeoJSON Export
    au_dummy = (
        polygons = Vector{Tuple{Float64, Float64}}[
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
            [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        ],
        centroids = [(0.5, 0.5), (1.5, 0.5)]
    )
    geojson_path = joinpath(temp_dir, "spatial_units.geojson")
    export_spatial_results_to_geojson(geojson_path, res_test, au_dummy)
    @test isfile(geojson_path)
    @test filesize(geojson_path) > 0

    # Cleanup
    rm(temp_dir; recursive=true, force=true)
end

@testset "Prediction & Observation Magnitude and PPC Validation" begin
    # 1. Intercept-Only Model Magnitude Check
    @testset "Intercept-Only Model Magnitude" begin
        n_obs = 60
        true_mu = 4.5
        # Seeded, and drawn with enough samples that the statistic is actually powered.
        #
        # This testset used bare `randn`, an unseeded `sample`, and 150 draws. Under a
        # correctly recovered intercept the denoised mean should land within ~0.15 of the
        # data mean (posterior SD is 0.3/sqrt(60) = 0.039), but 150 MH draws leaves Monte
        # Carlo error of the same order as the 0.5 threshold, so the test failed at
        # 0.5024 / 0.5010. Measured across seeds, the gap scales as 1/sqrt(n) -- exactly
        # what Monte Carlo error does and what a model bias does NOT:
        #
        #     draws   150   400  1000  3000
        #     max gap 0.43  0.21  0.14  0.07
        #
        # So the assertion was sound and the test was underpowered. 2000 draws gives a ~5x
        # margin under 0.5 while keeping the threshold that was already there; the fix is
        # more draws, not a looser bound.
        y_sim = true_mu .+ randn(MersenneTwister(7), n_obs) .* 0.3
        df_intercept = DataFrame(y = y_sim)

        m_intercept = @tbm(likelihood(y) ~ 1, df_intercept, verbose=false)
        chn_intercept = sample(MersenneTwister(7), m_intercept, MH(), 2000, progress=false)
        res_intercept = TethysBayes.model_results_comprehensive(m_intercept, chn_intercept)
        
        y_obs_vec = res_intercept.predictions.observed
        pred_denoised = res_intercept.predictions.denoised.mean
        pred_noisy = res_intercept.predictions.noisy.mean
        
        @test length(pred_denoised) == n_obs
        @test length(pred_noisy) == n_obs
        @test abs(mean(pred_denoised) - mean(y_obs_vec)) < 0.5
        @test abs(mean(pred_noisy) - mean(y_obs_vec)) < 0.5
        @test minimum(pred_denoised) >= minimum(y_obs_vec) - 1.5
        @test maximum(pred_denoised) <= maximum(y_obs_vec) + 1.5
        @test res_intercept.metrics.rmse < 1.0
    end

    # 2. Complex Model (Fixed + Spatiotemporal) Magnitude and PPC Pearson Correlation Check
    @testset "Complex Model Magnitude and PPC Pearson Correlation" begin
        p_data = TethysBayes.TethysBayes_data("scottish_lip")
        df_cplx = p_data.data
        W_cplx = p_data.au.W
        
        m_cplx = @tbm(likelihood(y_gauss) ~ 1 + cov1 + random(s_idx,
            model=bym2) + random(year, model=ar1), df_cplx, W=W_cplx, verbose=false)
        chn_cplx = sample(m_cplx, NUTS(30, 0.65), 50, progress=false)
        res_cplx = TethysBayes.model_results_comprehensive(m_cplx, chn_cplx)
        
        y_obs_vec = res_cplx.predictions.observed
        pred_denoised = res_cplx.predictions.denoised.mean
        pred_noisy = res_cplx.predictions.noisy.mean
        
        @test abs(mean(pred_denoised) - mean(y_obs_vec)) < 1.0
        @test abs(mean(pred_noisy) - mean(y_obs_vec)) < 1.0
        @test abs(std(pred_noisy) - std(y_obs_vec)) < 1.5
        
        @test haskey(res_cplx.metrics, :r_pearson)
        r_val = res_cplx.metrics.r_pearson
        @test r_val isa Real
        @test !isnan(r_val)
        @test !isinf(r_val)
        @test -1.0 <= r_val <= 1.0
        calc_r = cor(y_obs_vec, pred_denoised)
        @test isapprox(r_val, calc_r, atol=1e-6)
        @test r_val > 0.0
        
        if HAS_PLOTTING
            plots_res = TethysBayes.TethysBayes_plots(res_cplx, df_cplx; au=p_data.au)
            @test haskey(plots_res.plots, :posterior_predictive_check)
            @test haskey(plots_res.plots_data, :posterior_predictive_check)
        else
            @test_skip false
            @info "Skipping TethysBayes_plots checks: optional plotting stack not installed."
        end
        @test res_cplx.metrics.r_pearson == r_val
    end

    # 3. Poisson with Log Offsets Count-Scale Magnitude Check
    @testset "Poisson with Log Offsets Magnitude" begin
        p_data = TethysBayes.TethysBayes_data("scottish_lip")
        df_lip = p_data.data
        W_lip = p_data.au.W
        
        m_pois = @tbm(likelihood(y_pois, family=poisson,
            log_offsets=log_offsets) ~ 1 + cov1 + random(s_idx, model=bym2), df_lip,
            W=W_lip, verbose=false)
        chn_pois = sample(m_pois, MH(), 100, progress=false)
        res_pois = TethysBayes.model_results_comprehensive(m_pois, chn_pois)
        
        preds_p = res_pois.predictions.denoised.mean
        @test median(preds_p) > 0.0
        @test median(preds_p) < 2.0 * maximum(df_lip.y_pois)
        @test abs(median(preds_p) - median(df_lip.y_pois)) < 25.0
    end

    # 4. Binomial with Trials Scale Check
    @testset "Binomial with Trials Magnitude" begin
        n_obs = 50
        n_trials = 40
        true_prob = 0.35
        y_sim = rand(Distributions.Binomial(n_trials, true_prob), n_obs)
        df_bin = DataFrame(y = y_sim, trials = fill(n_trials, n_obs))
        
        m_bin = @tbm(likelihood(y, family=binomial, trials=trials) ~ 1, df_bin, verbose=false)
        chn_bin = sample(m_bin, MH(), 100, progress=false)
        res_bin = TethysBayes.model_results_comprehensive(m_bin, chn_bin)
        
        preds_bin = res_bin.predictions.denoised.mean
        @test abs(mean(preds_bin) - mean(y_sim)) < 6.0
        @test all(v -> 0.0 <= v <= n_trials, preds_bin)
    end

      # 5. LogNormal Natural Scale Check
      @testset "LogNormal Natural Scale Magnitude" begin
          # Seeded: this drew from the global RNG, so its result depended on how many
          # draws earlier testsets happened to make.
          Random.seed!(4242)
          n_obs = 50
          true_mu = 1.5
          y_sim = exp.(true_mu .+ randn(n_obs) .* 0.2)
          df_ln = DataFrame(y = y_sim)


        m_ln = @tbm(likelihood(y, family=lognormal) ~ 1, df_ln, verbose=false)
        chn_ln = sample(m_ln, MH(), 100, progress=false)
        res_ln = TethysBayes.model_results_comprehensive(m_ln, chn_ln)
        
        preds_ln = res_ln.predictions.denoised.mean
        @test abs(mean(preds_ln) - mean(y_sim)) < 4.0
        @test minimum(preds_ln) > 0.0
    end

    # 6. Smooth Model (P-Spline) Magnitude Check
    @testset "Smooth Model (P-Spline) Magnitude" begin
        p_data = TethysBayes.TethysBayes_data("scottish_lip")
        df_lip = p_data.data
        
        m_smooth = @tbm(likelihood(y_gauss) ~ 1 + random(cov1, model=pspline), df_lip,
            verbose=false)
        chn_smooth = sample(m_smooth, MH(), 100, progress=false)
        res_smooth = TethysBayes.model_results_comprehensive(m_smooth, chn_smooth)
        
        preds_smooth = res_smooth.predictions.denoised.mean
        @test abs(mean(preds_smooth) - mean(df_lip.y_gauss)) < 2.5
        @test haskey(res_smooth.effects, :cov1)
    end

    @testset "Bundle Schema Versioning" begin
        Random.seed!(909)
        n_s = 8
        df_s = DataFrame(
            y = rand(Poisson(2.0), n_s),
            x = collect(1.0:n_s),
            s_idx = collect(1:n_s),
        )
        W_s = TethysBayes.spatial_knn_graph([(Float64(i), 0.0) for i in 1:n_s], 2)[2]
        m_s = @tbm(
            likelihood(y, family=poisson) ~ intercept() + fixed(x) +
            random(s_idx, model=bym2), df_s, W=W_s, verbose=false
        )

        path = joinpath(scratch_dir(), "schema.TethysBayes")
        TethysBayes.save_TethysBayes_model(path, m_s)

        # A bundle records the layout version that wrote it, in both places a reader
        # might look, and does not carry the old hardcoded package-version string.
        JLD2.jldopen(path, "r") do f
            @test haskey(f, "schema_version")
            @test f["schema_version"] == TethysBayes.TethysBayes_SCHEMA_VERSION
            @test f["metadata"]["schema_version"] == TethysBayes.TethysBayes_SCHEMA_VERSION
            @test !haskey(f["metadata"], "TethysBayes_version")
            @test haskey(f["metadata"], "dropped_spec_registry_keys")
        end

        # The recorded version survives a round trip, and the model is still usable.
        loaded = TethysBayes.load_TethysBayes_model(path)
        @test loaded.model isa DynamicPPL.Model

        # A bundle from a future TethysBayes must be refused outright, not silently misread.
        future = joinpath(scratch_dir(), "future.TethysBayes")
        TethysBayes.save_TethysBayes_model(future, m_s)
        JLD2.jldopen(future, "r+") do f
            delete!(f, "schema_version")
            f["schema_version"] = TethysBayes.TethysBayes_SCHEMA_VERSION + 5
        end
        @test_throws ErrorException TethysBayes.load_TethysBayes_model(future)

        # A bundle with no marker at all predates versioning: treat it as v1, not as
        # a corrupt file, and say so.
        legacy = joinpath(scratch_dir(), "legacy.TethysBayes")
        TethysBayes.save_TethysBayes_model(legacy, m_s)
        JLD2.jldopen(legacy, "r+") do f
            delete!(f, "schema_version")
            delete!(f, "metadata")
        end
        @test (@test_logs (:warn,) match_mode=:any TethysBayes.load_TethysBayes_model(legacy)).model isa
            DynamicPPL.Model
    end
end
