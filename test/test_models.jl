# ==============================================================================
# TethysBayes Test Suite: Model Instantiation, Smoke Tests, and Complex Inference
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Formulaic Interface & Model Instantiation" begin
    p_dummy = TethysBayes.TethysBayes_data("scottish_lip")
    dummy_df = p_dummy.data
    W_test = p_dummy.au.W

    test_cases = [
        ("ST_TypeIV_Poisson", "likelihood(y_pois, family=poisson) ~ 1 + cov1 + (random(s_idx, model=besag) ⊗ random(t_idx, model=ar1))", "poisson", Dict(:W => W_test)),
        ("MixedEffects_Gaussian", "likelihood(y_gauss) ~ 1 + cov1 + mixed((1 + cov1) | region) + random(s_idx, model=icar)", "gaussian", Dict(:W => W_test)),
        ("Multivariate_Gaussian",
            "likelihood(species_1 + species_2) ~ 1 + cov1 + random(s_idx, model=bym2)",
            "gaussian", Dict(:W => W_test)),
        ("Seasonal_RW2_Poisson", "likelihood(y_pois, family=poisson) ~ 1 + random(month, model=harmonic, period=12) + random(t_idx, model=rw2)", "poisson", Dict()),
        ("SVC_Gaussian", "likelihood(y_gauss) ~ 1 + (cov1 |> random(s_idx, model=icar))",
            "gaussian", Dict(:W => W_test)),
        ("Spatial_Smooth_RFF", "likelihood(y_gauss) ~ 1 + random(s_y, s_x, model=rff, n_features=15) + random(t_idx, model=ar1)", "gaussian", Dict()),
        ("Leroux_Gaussian", "likelihood(y_gauss) ~ 1 + random(s_idx, model=leroux)",
            "gaussian", Dict(:W => W_test)),
        ("GP_Smooth_Gaussian",
            "likelihood(y_gauss) ~ 1 + random(cov1, model=gp, kernel=\"se\")", "gaussian",
            Dict())
    ]

    for (name, f_str, fam, extra_args) in test_cases
        @testset "$name" begin
            println("  Testing Instantiation: $name")
            model = TethysBayes.TethysBayes_core(f_str, dummy_df; model_family=fam, extra_args...)
            @test size(sample(model, Turing.Prior(), 1), 1) >= 1
        end
    end
end

@testset "Integration Tests: Smoke Tests" begin
    p_smoke = TethysBayes.TethysBayes_data("scottish_lip")
    sim_df = p_smoke.data
    W_smoke = p_smoke.au.W

    @testset "IID Model Smoke Test" begin
        model_iid_sim = @tbm(likelihood(y_gauss) ~ intercept() + random(region,
            model=iid), sim_df)
        chain = sample(model_iid_sim, MH(), 100, progress=false)
        @test mean(chain[:sigma_region]) > 0
    end

    @testset "AR1 Model Smoke Test" begin
        model_ar1_sim = @tbm(likelihood(y_gauss) ~ intercept() + random(year, model=ar1),
            sim_df)
        chain = sample(model_ar1_sim, NUTS(10, 0.65), 10, progress=false)
        @test mean(chain[:sigma_year]) > 0
    end

    @testset "BYM2 ⊗ AR1 Interaction Smoke Test" begin
        model_int_sim = @tbm(
            likelihood(y_gauss) ~ intercept() + (random(s_idx, model=bym2) ⊗ random(t_idx,
                model=ar1)),
            sim_df, W=W_smoke
        )
        chain_int_sim = sample(model_int_sim, NUTS(10, 0.65), 10, progress=false)
        @test any(k -> occursin("sigma", string(k)) && (occursin("interaction",
            string(k)) || occursin("s_idx", string(k)) || occursin("t_idx", string(k))),
            keys(chain_int_sim))
    end

    @testset "AR1 spectral prior scale (variance, not just a shape check)" begin
        # The :spectral path is the DEFAULT and was missing the sqrt(1-rho^2) stationary
        # scale, giving the field a marginal variance of sigma^2 instead of
        # sigma^2/(1-rho^2) -- at rho = 0.9 that understated the temporal field variance by
        # a factor of 5.3, so the default method silently fitted a different model from
        # :centered.
        #
        # The stationary AR(1) prior pins Var(phi_1) = sigma^2/(1-rho^2) exactly, which
        # gives a sharp, reference-free assertion.
        n, sigma, noise = 12, 1.0, 1e-9
        for rho in (0.0, 0.3, 0.6, 0.9)
            tmpl = TethysBayes.build_structure_template(:ar1, n)
            lambda_vals = (1.0 + rho^2) .+ rho .* tmpl.L
            scale = TethysBayes.sigma_stable_scale(sigma, rho, Float64)
            D = scale ./ sqrt.(lambda_vals .+ noise)
            K = tmpl.U * Diagonal(D.^2) * tmpl.U'
            @test isapprox(K[1, 1], sigma^2 / (1 - rho^2); rtol=0.02)
        end

        # The factor is sigma / sqrt(1-rho^2), NOT sigma * sqrt(1-rho^2). An earlier fix
        # used the reciprocal and the diagonal assertion above is what caught it.
        @test isapprox(TethysBayes.sigma_stable_scale(1.0, 0.6, Float64), 1 / sqrt(1 - 0.36);
                       rtol=1e-12)
        # rho can round to +/-1 in floating point; the scale must stay finite and positive.
        @test isfinite(TethysBayes.sigma_stable_scale(1.0, 1.0, Float64))
        @test TethysBayes.sigma_stable_scale(1.0, 1.0, Float64) > 0

        # get_effects must use the same scale as the model, or reconstruction would
        # disagree with the model it was fitted from.
        @test hasmethod(TethysBayes.sigma_stable_scale, Tuple{Float64, Float64, Type{Float64}})
    end

    @testset "reshard SD propagates through the aggregation" begin
        # A coarse cell averaging k independent fine cells has SD sd/sqrt(k). The old
        # formula returned sd, overstating by exactly sqrt(k) (measured 2x for k=4).
        k, n_coarse = 4, 3
        P = zeros(n_coarse, n_coarse * k)
        for c in 1:n_coarse
            P[c, (c-1)*k+1 : c*k] .= 1/k
        end
        got = TethysBayes.reshard_spatial_field(P, (mean=ones(n_coarse * k), std=ones(n_coarse * k)))
        @test all(isapprox.(got.std, 1/sqrt(k); rtol=1e-12))
        @test all(got.std .< 1.0)          # strictly shrunk, as an average must be
        # The mean path is unchanged by the SD fix.
        @test all(isapprox.(got.mean, 1.0; rtol=1e-12))
        # Unequal SDs must be handled elementwise through P and P'.
        sd_fine = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        got2 = TethysBayes.reshard_spatial_field(P, (mean=zeros(12), std=copy(sd_fine)))
        want = sqrt.([sum(sd_fine[(c-1)*k+1 : c*k] .^ 2) / k^2 for c in 1:n_coarse])
        @test all(isapprox.(got2.std, want; rtol=1e-12))
    end
end

@testset "Complex Integration Tests" begin
    p_cplx = TethysBayes.TethysBayes_data("scottish_lip")
    data = p_cplx.data
    W = p_cplx.au.W

    @testset "Hurdle Model" begin
        model = @tbm(likelihood(y_pois, family=poisson, hurdle=5) ~ 1 + random(s_idx,
            model=bym2), data, W=W)
        chain = sample(model, MH(), 100, progress=false)
        @test any(k -> occursin("hurdle", string(k)), keys(chain))
    end

    @testset "Eigen Model" begin
        data_eigen = copy(data)
        data_eigen[!, :species_1] = randn(nrow(data_eigen))
        data_eigen[!, :species_2] = randn(nrow(data_eigen))
        data_eigen[!, :species_3] = randn(nrow(data_eigen))
        model = @tbm(likelihood(y_gauss) ~ 1 + eigen(species_1, species_2, species_3,
            n_factors=2), data_eigen)
        chain = sample(model, MH(), 10, progress=false)
        @test any(k -> occursin("pca_sd", string(k)) || occursin("species", string(k)),
            keys(chain))
    end

    @testset "Multifidelity Signal Transfer" begin
        df_hi = copy(data)
        if !hasproperty(df_hi, :t_idx)
            df_hi[!, :t_idx] = df_hi[!, :year]
        end
        W_mf = W
        df_lo = select(df_hi, :proxy_val => :y_low, :s_idx, :t_idx)

        model_mf = @tbm(
            likelihood(y_gauss) ~ 1 + random(s_idx, model=bym2) + random(t_idx,
                model=ar1) + nested(low_fi,
                formula="likelihood(y_low) ~ 1 + random(s_idx) + random(t_idx)",
                data_source=:low_quality_data),
            df_hi, W=W_mf, low_quality_data=df_lo
        )
        
        chain_mf = sample(model_mf, NUTS(10, 0.65), 10, progress=false, check_model=false)
        @test any(k -> occursin("nested_low_fi", string(k)), keys(chain_mf))
    end

    @testset "Prediction Engine" begin
        train_df = copy(data)
        if !hasproperty(train_df, :t_idx)
            train_df[!, :t_idx] = train_df[!, :year]
        end
        W_train = W
        
        centroids_matrix = reduce(hcat, [[c[1], c[2]] for c in p_cplx.au.centroids])
        s_N_train = length(p_cplx.au.centroids)
        t_N_train = length(unique(train_df.t_idx))

        test_pts = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)]
        n_s_test = length(test_pts)
        test_df = DataFrame(s_x=repeat([p[1] for p in test_pts], inner=t_N_train), 
                            s_y=repeat([p[2] for p in test_pts], inner=t_N_train), 
                            s_idx=repeat(1:n_s_test, inner=t_N_train),
                            t_idx=repeat(1:t_N_train, outer=n_s_test))

        model_obj = TethysBayes.TethysBayes_core("likelihood(y_gauss) ~ 1 + random(s_idx, model=bym2) + random(t_idx, model=ar1)", 
                         train_df; s_N=s_N_train, t_N=t_N_train, W=W_train)
        
        chain_train = sample(model_obj, MH(), 500, progress=false)
        res_pred = TethysBayes.predict(model_obj, chain_train, test_df; n_samples=100)
        
        @test res_pred.predictions_denoised isa NamedTuple
        @test length(res_pred.predictions_denoised.mean) == nrow(test_df)
    end

    @testset "Cross-Validation Orchestrator" begin
        cv_df = data
        cv_W = W
        cv_formula = "likelihood(y_gauss) ~ 1 + random(year, model=ar1)"

        @testset "k-fold CV" begin
            cv_results_kfold = TethysBayes.TethysBayes_cv_orchestrator(cv_formula, cv_df; method=:kfold,
                n_folds=3, n_samples=50, W=cv_W)
            @test length(cv_results_kfold.folds) == 3
            @test cv_results_kfold.mean_rmse isa Real
        end

        @testset "Temporal Forward-Chaining CV" begin
            cv_results_fchain = TethysBayes.TethysBayes_cv_orchestrator(cv_formula, cv_df;
                method=:temporal_forward_chain, cv_var=:year, n_folds=2, n_samples=50,
                W=cv_W)
            @test length(cv_results_fchain.folds) == 2
            @test cv_results_fchain.mean_r2 isa Real
        end
    end
end

@testset "Composite Gibbs Sampling & Optimal Sampler (Dual Promotion Verification)" begin
    W_test = [0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0]
    df_gibbs = DataFrame(
        s_idx = [1, 2, 3, 4, 1, 2, 3, 4],
        year = [1, 1, 1, 1, 2, 2, 2, 2],
        cov1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        log_offsets = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
        y = [2, 3, 1, 4, 3, 5, 2, 6]
    )
    m_gibbs = @tbm(
        likelihood(y, family=poisson, log_offsets=log_offsets) ~
            intercept() +
            fixed(cov1) +
            random(s_idx, model=icar) +
            random(year, model=ar1),
        df_gibbs,
        W = W_test,
        use_gpu = false,
        verbose = false
    )
    @test m_gibbs isa DynamicPPL.Model
    
    # Build optimal Gibbs sampler
    os = get_optimal_sampler(m_gibbs; adaptation_steps=10)

    chn_gibbs = sample(m_gibbs, os, 5; progress=false)
    @test size(chn_gibbs, 1) == 5

    # Test precompute_step_sizes with adaptation_steps
    step_sizes = precompute_step_sizes(m_gibbs; min_ϵ=0.01, max_ϵ=0.8, max_depth=8,
        adaptation_steps=:auto)
    @test haskey(step_sizes, :s_idx)
    @test haskey(step_sizes, :year)
    @test step_sizes[:s_idx].init_ϵ >= 0.01
    @test step_sizes[:s_idx].max_depth == 8
    @test step_sizes[:s_idx].adaptation_steps >= 100

    # Test get_optimal_sampler with :auto adaptation_steps
    os_auto = get_optimal_sampler(m_gibbs; adaptation_steps=:auto)

    # Test get_optimal_sampler with custom init_ϵ and max_depth
    os_custom = get_optimal_sampler(
        m_gibbs;
        init_ϵ = Dict(:s_idx => 0.03, :year => 0.08),
        max_depth = 6,
        adaptation_steps = 10
    )
    chn_custom = sample(m_gibbs, os_custom, 5; progress=false)
    @test size(chn_custom, 1) == 5

    # Test TethysBayes_sample with MCMCThreads backend and deepcopy validation
    chn_TethysBayes = TethysBayes_sample(m_gibbs, os_custom, 5; progress=false)
    @test size(chn_TethysBayes, 1) == 5

    # Test model_results_comprehensive on multi-chain / VNChain sample object
    res_gibbs = model_results_comprehensive(m_gibbs, chn_TethysBayes)
    @test haskey(res_gibbs.metrics, :rmse)
    @test haskey(res_gibbs, :parameters)

    # Test parameter name resolution with Parameter(...) and parameters. wrappers
    raw_wrapped_names = ["Parameter(sigma_s_idx)", "Parameter(innovations_s_idx[1])", "parameters.beta", "intercept"]
    @test TethysBayes._find_parameter(raw_wrapped_names, "sigma_s_idx") == "sigma_s_idx"
    @test TethysBayes._find_parameter(raw_wrapped_names, "innovations_s_idx") == "innovations_s_idx"
    @test TethysBayes._find_parameter(raw_wrapped_names, "beta") == "beta"
    @test TethysBayes._find_parameter(raw_wrapped_names, "intercept") == "intercept"

    # Test TethysBayes_Likelihood random generation
    lik_poisson = TethysBayes_Likelihood("poisson", 1.5)
    @test rand(lik_poisson) >= 0.0
    lik_gauss = TethysBayes_Likelihood("gaussian", 0.0; sigma_y=1.0)
    @test isfinite(rand(lik_gauss))
    lik_vec = [lik_poisson for _ in 1:10]
    sampled_vec = rand.(lik_vec)
    @test length(sampled_vec) == 10

    # Test extract_param_matrix and extract_param_vector
    mock_chain_dict = Dict(:sigma_s_idx => [0.5, 0.6, 0.7], :innovations_s_idx => [[0.1, 0.2], [0.3,
        0.4], [0.5, 0.6]])
    @test size(TethysBayes.extract_param_matrix(mock_chain_dict, :sigma_s_idx), 1) == 3
    @test size(TethysBayes.extract_param_matrix(mock_chain_dict, :innovations_s_idx), 2) == 2
    @test length(TethysBayes.extract_param_vector(mock_chain_dict, :sigma_s_idx)) == 3

    # Test multi-chain scalar parameter collapsing
    mock_multichain = (
        chains = [1, 2, 3],
        intercept = randn(10, 3),
        beta = randn(10, 3)
    )
    mat_mc = TethysBayes.extract_param_matrix(mock_multichain, :intercept)
    @test size(mat_mc) == (30, 1)

    # Test direct parameter summary statistics
    df_direct_summary = TethysBayes._compute_direct_parameter_summary(mock_chain_dict)
    @test df_direct_summary isa DataFrame
    @test "parameters" in names(df_direct_summary)
    @test "mean" in names(df_direct_summary)
    @test size(df_direct_summary, 1) >= 2

    # Test _generate_conditional_predictions type safety with integer columns
    cond_res = TethysBayes._generate_conditional_predictions(m_gibbs, chn_TethysBayes, m_gibbs.args.M, :cov1)
    @test !isnothing(cond_res)

    # Test temporal component get_effects with PS DataFrame
    ps_test = (data = DataFrame(s_idx = [1, 2], year = [1, 2], cov1 = [0.1, 0.2]), y_N = 2,
        y_obs = [0.0, 0.0])
    ar1_spec = filter(s -> s.structure == :temporal, m_gibbs.args.M.components)[1]
    ar1_effects = TethysBayes.get_effects(ar1_spec.component_obj, chn_TethysBayes, ar1_spec,
        m_gibbs.args.M, ps_test)
    @test haskey(ar1_effects, :structured)

    # Test predict with out-of-sample DataFrame
    df_new = DataFrame(s_idx = [1, 2, 3], year = [1, 2, 3], cov1 = [0.1, 0.2, 0.3])
    pred_out = TethysBayes.predict(m_gibbs, chn_TethysBayes, df_new)
    @test haskey(pred_out, :predictions_denoised)
    @test length(pred_out.predictions_denoised.mean) == 3

    # Test model with log_offsets end-to-end
    df_offset = DataFrame(
        y = [1.0, 2.0, 3.0, 4.0],
        s_idx = [1, 2, 1, 2],
        year = [1, 1, 2, 2],
        cov1 = [0.1, 0.2, 0.3, 0.4],
        log_e = [0.0, 0.1, 0.0, 0.1]
    )
    W_small = [0.0 1.0; 1.0 0.0]
    m_offset = TethysBayes.TethysBayes_core(
        :(likelihood(y, family=poisson,
            log_offsets=log_e) ~ intercept() + fixed(cov1) + random(s_idx,
            model=bym2) + random(year, model=ar1)),
        df_offset;
        W = W_small,
        verbose = false
    )
    chn_offset = TethysBayes.TethysBayes_sample(m_offset, 5; progress=false)
    au_test = (polygons = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0],
        [2.0, 0.0], [2.0, 1.0], [1.0, 0.0]]], centroids = [(0.5, 0.5), (1.5, 0.5)])
    res_offset = TethysBayes.model_results_comprehensive(m_offset, chn_offset)
    @test haskey(res_offset.metrics, :rmse)
    @test haskey(res_offset.metrics, :waic)
    @test haskey(res_offset, :parameters)
    @test haskey(res_offset, :effects)
    @test haskey(res_offset, :predictions)
    @test haskey(res_offset, :draws)
    @test haskey(res_offset.draws, :weights)
    @test haskey(res_offset, :model)
    @test haskey(res_offset, :chain)

    # Test separate TethysBayes_plots statement with data and spatial areal units (au).
    # `TethysBayes_plots` lives in the `TethysBayesPlotsExt` package extension and only exists once the
    # optional plotting stack (Plots/StatsPlots/ColorSchemes) is installed.
    if HAS_PLOTTING
        plot_dir = scratch_dir()
        plots_res = TethysBayes.TethysBayes_plots(res_offset; data=df_offset, au=au_test, save_dir=plot_dir)
        @test haskey(plots_res.plots, :posterior_predictive_check)
        @test haskey(plots_res.plots, :fixed_effects)
        @test haskey(plots_res.plots, :spatial)
        @test haskey(plots_res.plots, :spatial_observed)
        @test haskey(plots_res.plots, :spatial_fitted)
        @test haskey(plots_res.plots, :temporal)
        @test haskey(plots_res.plots, :spacetime_predictions)
        @test haskey(plots_res.plots_data, :spacetime_predictions)
        @test isfile(joinpath(plot_dir, "posterior_predictive_check.png"))
        @test !isfile(joinpath(plot_dir, "posterior_predictive_check.png.png"))
        @test isfile(joinpath(plot_dir, "spacetime_predictions.png"))

        # Test positional data argument
        plots_res_pos = TethysBayes.TethysBayes_plots(res_offset, df_offset; au=au_test)
        @test haskey(plots_res_pos.plots, :posterior_predictive_check)
    else
        @test_skip false
        @info "Skipping TethysBayes_plots checks: optional plotting stack not installed."
    end

    @testset "GMRF: every null direction is zeroed, not just the first" begin
        # The intrinsic sum-to-zero constraint was imposed by `diag_D[1] = 0`, which is
        # right only for a CONNECTED graph. A disconnected graph has one null direction per
        # component, and zeroing only the first leaves the rest carrying `sigma^2/noise`
        # variance -- measured at 1e9, giving a field mean of 2.0e8 instead of 0.
        using SparseArrays

        function chain_graph(n, edges)
            W = spzeros(n, n)
            for (i, j) in edges
                W[i, j] = 1.0; W[j, i] = 1.0
            end
            return W
        end

        # --- the reference implementation zeroes all null directions ---
        @testset "_zero_null_modes!" begin
            L = [0.0, 0.4, 2.0]
            d = [9.0, 9.0, 9.0]
            z = TethysBayes._zero_null_modes!(d, L)
            @test z == [1]
            @test d[1] == 0.0
            @test d[2] == 9.0 && d[3] == 9.0     # non-null untouched

            # Two null directions -> both zeroed.
            d2 = [9.0, 9.0, 9.0]
            @test TethysBayes._zero_null_modes!(d2, [0.0, 1e-16, 3.0]) == [1, 2]
            @test d2[1] == 0.0 && d2[2] == 0.0 && d2[3] == 9.0

            # Relative tolerance: scaling the whole precision must not change the verdict.
            d3 = [9.0, 9.0, 9.0]
            TethysBayes._zero_null_modes!(d3, [0.0, 1.0e-15, 3.0e6])
            @test d3[1] == 0.0 && d3[2] == 0.0 && d3[3] == 9.0

            # Degenerate template with no positive eigenvalue: everything is null.
            d4 = fill(9.0, 3)
            @test TethysBayes._zero_null_modes!(d4, [0.0, 0.0, 0.0]) == [1, 2, 3]
            @test all(iszero, d4)

            @test TethysBayes._zero_null_modes!(Float64[], Float64[]) == Int[]
        end

        # --- the connected-graph behaviour is unchanged (a no-op) ---
        @testset "connected graph is unaffected" begin
            W = chain_graph(5, [(1,2),(2,3),(3,4),(4,5)])
            t = TethysBayes.build_structure_template(:icar, 5; W=W)
            L = t.L
            new = 1.0 ./ sqrt.(L .+ 1e-9)
            old = copy(new)
            old[1] = 0.0                            # the previous behaviour
            TethysBayes._zero_null_modes!(new, L)
            @test isapprox(new, old; atol=1e-15)
            @test count(iszero, new) == 1            # exactly one null direction
        end

        # --- the disconnected graph now satisfies sum-to-zero ---
        @testset "disconnected graph has no null-space leak" begin
            W = chain_graph(5, [(1,2),(3,4),(4,5)])   # two components
            t = TethysBayes.build_structure_template(:icar, 5; W=W)
            L = t.L
            tol = 1e-10 * maximum(L)
            @test count(<=(tol), L) >= 2             # more than one null direction

            d = 1.0 ./ sqrt.(L .+ 1e-9)
            TethysBayes._zero_null_modes!(d, L)
            @test all(d[i] == 0.0 for i in findall(<=(tol), L))   # ALL of them

            C = t.U * Diagonal(d .^ 2) * t.U'
            # Sum-to-zero: the implied field has zero mean, so the covariance has zero row
            # sums.
            @test maximum(abs, vec(sum(C, dims=1))) < 1e-10

            # The inline form emitted into generated code must agree with the tested helper,
            # or the two can drift and the model stops satisfying the constraint it claims to.
            inline = copy(d)
            inline[L .<= (1e-10 * maximum(L))] .= 0.0
            @test isapprox(inline, d; atol=0.0)
        end

        # --- the generated model code carries the fix ---
        @testset "generated code contains the self-contained fix" begin
            n = 4
            W = chain_graph(n, [(1,2),(3,4)])
            idx = repeat(collect(1:n), 2)
            dfe = DataFrame(y = float.(1 .+ idx), s_idx = idx, x = float.(idx))
            m = TethysBayes.TethysBayes_core("likelihood(y, family=poisson) ~ intercept() + " *
                               "random(s_idx, model=icar)", dfe; W=W, s_N=n, verbose=false)
            code = m.args.M.generated_model_code
            # Must be self-contained: `_GeneratedModelRuntime` has no TethysBayes names in scope,
            # so a bare or `$(TethysBayes).`-qualified helper call raises UndefVarError there.
            @test occursin("zero EVERY null direction", code)
            @test !occursin("_zero_null_modes!", code)
        end
    end

    @testset "Sparse GP: :dtc and :fitc conditional variances" begin
        # Root cause this pins: the whitening was `A = K_XU L^-1` (computed as
        # `(L' \ K_XU')'`), but (LL')^-1 = L'^-1 L^-1, so the factor satisfying
        # `M M' = K_UU^-1` is `M = L'^-1` and the correct form is `A = K_XU L'^-1`, i.e.
        # `A' = L \ K_XU'` -- a solve against L, NOT L'. The wrong route inflated
        # diag(A A') by ~46x on one fixture, pushing lambda below zero so it CLAMPED TO 0:
        # silently reporting zero conditional variance, the over-confident direction.
        function rq_kernel(A, B, sigma, ls)
            K = zeros(size(A, 1), size(B, 1))
            for a in axes(A, 1), b in axes(B, 1)
                K[a, b] = sigma^2 * exp(-0.5 * sum((A[a, :] .- B[b, :]) .^ 2) / ls^2)
            end
            K
        end

        n, sigma, s2 = 60, 1.0, 1.0
        X = rand(MersenneTwister(3), n, 2) .* 10

        for (M, ls, gap) in [(15, 1.5, 1.0), (15, 1.5, 2.0), (30, 3.0, 2.0)]
            Z = rand(MersenneTwister(99), M, 2) .* 6 .+ (10 + gap)
            K_UU = rq_kernel(Z, Z, sigma, ls) + 1e-2 * I
            K_XU = rq_kernel(X, Z, sigma, ls)
            L = cholesky(Symmetric(K_UU)).L

            # References built by a DIFFERENT route: explicit inv and a dense LU solve.
            ref_dtc = max.(s2 .- diag(K_XU * (K_UU \ Matrix(K_XU'))), 0.0)
            A = K_XU * inv(Matrix(L'))
            B = Matrix{Float64}(I, M, M) + (A' * A) ./ s2
            ref_fitc = max.(s2 .- diag(A * inv(B) * A'), 0.0)

            got_dtc = TethysBayes._dtc_lambda_diag(L, K_XU, sigma)
            got_fitc = TethysBayes._fitc_lambda_diag(L, K_XU, sigma)

            @test isapprox(got_dtc, ref_dtc; atol=1e-10)
            @test isapprox(got_fitc, ref_fitc; atol=1e-10)
            @test all(>=(0.0), got_dtc)
            @test all(>=(0.0), got_fitc)
            # lambda_fitc >= lambda_dtc: lambda SUBTRACTS A B^-1 A', and B^-1 <= I, so the
            # subtracted term is smaller and lambda is larger.
            @test all(got_fitc .>= got_dtc .- 1e-10)
            @test all(ref_fitc .>= ref_dtc .- 1e-10)
        end

        @testset "the transposition is what makes the difference" begin
            # A regression guard that fails loudly if the `L'` form comes back.
            M, ls, gap = 30, 3.0, 2.0
            Z = rand(MersenneTwister(99), M, 2) .* 6 .+ (10 + gap)
            K_UU = rq_kernel(Z, Z, sigma, ls) + 1e-2 * I
            K_XU = rq_kernel(X, Z, sigma, ls)
            L = Matrix(cholesky(Symmetric(K_UU)).L)
            good = TethysBayes._dtc_lambda_diag(L, K_XU, sigma)
            wrong = max.(s2 .- vec(sum(((L' \ Matrix(K_XU')) .^ 2), dims=1)), 0.0)
            truth = max.(s2 .- diag(K_XU * (K_UU \ Matrix(K_XU'))), 0.0)
            @test isapprox(good, truth; atol=1e-10)
            @test !isapprox(wrong, truth; atol=1e-6)   # the old form really was different
        end

        @testset "both approaches sigma^2 as the conditioning weakens" begin
            prev = nothing
            for gap in (1.0, 2.0, 4.0)
                Z = rand(MersenneTwister(99), 12, 2) .* 6 .+ (10 + gap)
                K_UU = rq_kernel(Z, Z, sigma, 1.5) + 1e-2 * I
                K_XU = rq_kernel(X[1:20, :], Z, sigma, 1.5)
                L = cholesky(Symmetric(K_UU)).L
                d = mean(TethysBayes._dtc_lambda_diag(L, K_XU, sigma))
                @test isapprox(d, s2; atol=1e-3)
                isnothing(prev) || @test d >= prev - 1e-9   # monotonically toward sigma^2
                prev = d
            end
        end

        @testset "the dispatcher is the only path either side uses" begin
            @test TethysBayes._sparse_gp_lambda_diag(:dtc, Matrix(I, 2, 2), zeros(3, 2), 1.0) isa Vector
            @test_throws ErrorException TethysBayes._sparse_gp_lambda_diag(
                :vfe, Matrix(I, 2, 2), zeros(3, 2), 1.0)
            @test_throws ErrorException TethysBayes._sparse_gp_lambda_diag(
                :nonsense, Matrix(I, 2, 2), zeros(3, 2), 1.0)
            # Registered for the generated-code module, where bare TethysBayes names do not resolve.
            @test isdefined(TethysBayes, :_sparse_gp_lambda_diag)
        end
    end

    @testset "Sparse GP end-to-end: every method fits and reconstructs" begin
        # The expression-level tests cannot see any of this. They check
        # `_sparse_gp_lambda_diag` in isolation; they do not check that the generated model
        # body can call it (it runs in `_GeneratedModelRuntime`, which has no TethysBayes names in
        # scope), that `:dtc` -- a NEW method name -- is accepted by the constructor and the
        # formula parser, or that `get_effects` uses the same variance the model did.
        #
        # `sigma` and `length_scale` are pinned to constants, which `_prior_or_constant`
        # turns into `name = value` rather than a prior. That makes the latent field's
        # covariance under `Prior()` exactly known, so the whole path is checkable:
        #
        #     Cov(field) = K_XU K_UU^-1 K_XU'  +  diag(lambda)
        rng = MersenneTwister(4)
        npts = 40
        dfg = DataFrame(x = collect(range(0.0, 10.0; length=npts)),
                        y = collect(range(0.0, 10.0; length=npts)))
        dfg.depth = 5.0 .+ 0.1 .* dfg.x .+ 0.05 .* dfg.y .+ 0.01 .* randn(rng, npts)

        sigma0, ls0, nind, ndraws = 1.0, 3.0, 10, 3000

        # Analytic variance for one method, computed independently of the component.
        function expected_field_variance(meth, M, hyper, noise)
            K_UU = TethysBayes.evaluate_kernel_matrix(hyper.Z_inducing, sigma0, ls0, :se, noise)
            K_XU = TethysBayes.evaluate_cross_kernel_matrix(hyper.coords, hyper.Z_inducing,
                                                     sigma0, ls0, :se)
            L = cholesky(Symmetric(K_UU)).L
            Q = diag(K_XU * (K_UU \ Matrix(K_XU')))
            lam = meth === :vfe ? zeros(npts) : TethysBayes._sparse_gp_lambda_diag(meth, L, K_XU, sigma0)
            return Q .+ lam
        end

        fieldvars = Dict{Symbol, Vector{Float64}}()
        for meth in (:dtc, :fitc, :vfe)
            # Built as a STRING: `sigma=$sigma0` inside `@tbm(...)` makes the formula parser
            # emit the literal text `$sigma0`, which then fails `Core.eval`.
            formula = "likelihood(depth) ~ intercept() + " *
                      "random(x, y, model=fitc, n_inducing=$nind, method=$meth, " *
                      "sigma=$sigma0, length_scale=$ls0)"
            @testset "method = :$meth" begin
                m = TethysBayes.TethysBayes_core(formula, dfg; verbose=false)
                @test m isa DynamicPPL.Model

                ch = sample(MersenneTwister(5), m, Prior(), ndraws; progress=false)
                @test size(ch, 1) == ndraws

                comp = m.args.M.components[1]
                @test comp.component_obj isa TethysBayes.FITC
                @test comp.component_obj.method === meth   # the label survived construction

                eff = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M, nothing)
                field = eff.structured[1]
                @test size(field) == (npts, ndraws)
                @test all(isfinite, field)

                emp = vec(var(field, dims=2))
                fieldvars[meth] = emp

                # THE regression this whole testset exists for: with a pinned constant,
                # `get_effects` used to fail to find the parameter and return a ZERO matrix
                # with only a warning. A zero field would pass every "is it finite" check.
                @test all(>(0.0), emp)
                @test !isapprox(maximum(emp), 0.0; atol=1e-12)

                # And the variance must be the one the model actually used.
                want = expected_field_variance(meth, m.args.M, comp.hyper, m.args.M.noise)
                @test isapprox(mean(emp), mean(want); rtol=0.12)
                @test maximum(abs.(emp .- want) ./ max.(want, 1e-8)) < 0.25
            end
        end

        @testset "the three methods are genuinely different" begin
            # :vfe carries no diagonal term, so it must be the least variable of the three.
            @test mean(fieldvars[:vfe]) < mean(fieldvars[:fitc])
            # DTC keeps the smaller conditional variance, so :dtc <= :fitc pointwise.
            @test all(fieldvars[:fitc] .>= fieldvars[:dtc] .- 0.05)
            @test !isapprox(fieldvars[:dtc], fieldvars[:vfe]; rtol=1e-3)
        end

        @testset "a pinned hyperparameter reconstructs instead of collapsing to zero" begin
            # Regression guard for the constant fallback: without it every field here is 0.
            formula = "likelihood(depth) ~ intercept() + " *
                      "random(x, y, model=fitc, n_inducing=$nind, method=:dtc, " *
                      "sigma=$sigma0, length_scale=$ls0)"
            m = TethysBayes.TethysBayes_core(formula, dfg; verbose=false)
            ch = sample(MersenneTwister(5), m, Prior(), 400; progress=false)
            comp = m.args.M.components[1]
            @test TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M,
                                    nothing).structured[1] |> x -> all(isfinite, x)
            # With sigma pinned the whole field must be O(sigma), not O(0).
            f = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M, nothing).structured[1]
            @test maximum(abs, f) > 1e-6
        end
    end

    @testset "a pinned scale hyperparameter never reconstructs as a zero field" begin
        # A pinned `sigma` is NOT a chain parameter: `_prior_or_constant` emits
        # `name = value` rather than a `~` statement, so DynamicPPL never records it and
        # `_find_parameter` can never find it. 51 `get_effects` sites treated "not in the
        # chain" as "genuinely absent" and returned `zeros(...)` with only a warning -- a
        # silently wrong reconstruction for precisely the configuration a user pins a
        # hyperparameter in order to hold it fixed. Confirmed end-to-end for FITC, ICAR and
        # BYM2 before the fix; an all-components sweep put the total at 18.
        #
        # The check must be NONZERO, not finite: a zero matrix is finite, so the obvious
        # assertion passes and the warning is easy to miss.
        # 24 points, because `fitc` defaults to 20 inducing points and a 12-point fixture
        # fails its prior-predictive check with `B has first dimension 20 but needs 12`.
        npts = 24
        dsp = DataFrame(idx = collect(1:npts), x = collect(1.0:npts), y = zeros(Float64, npts))
        dsp.depth = 9.0 .+ 3.0 .* sin.(dsp.x ./ 3.0) .+ 0.1 .* cos.(dsp.x)
        Wsp = spzeros(Int, npts, npts)
        for j in 1:(npts - 1)
            Wsp[j, j + 1] = 1
            Wsp[j + 1, j] = 1
        end
        Wsp[1, npts] = 1
        Wsp[npts, 1] = 1   # connected, so this isolates the pin from the null-space work

        # Components confirmed to reconstruct a nonzero field with a pinned sigma.
        # Components confirmed to reconstruct a nonzero field with a pinned sigma. `fitc` is
        # deliberately NOT here: it needs a 2-D inducing layout to be non-degenerate, and it is
        # already covered strictly harder in the sparse-GP end-to-end testset, which checks
        # its variance against the closed form `K_XU K_UU^-1 K_XU' + diag(lambda)`.
        pinned_ok = [:icar, :besag, :bym2, :leroux, :cyclic, :rw1, :rw2, :iid, :moran,
                     :ar1, :ar2, :sar, :gp, :spectralgp, :adaptivesmooth, :spde,
                     :bspline, :pspline, :tps]
        for model in pinned_ok
            @testset "model=$model" begin
                formula = "likelihood(depth, family=gaussian) ~ intercept() + " *
                          "random(idx, model=$model, sigma=1.0)"
                m = TethysBayes.TethysBayes_core(formula, dsp; W=Wsp, verbose=false)
                ch = sample(MersenneTwister(3), m, MH(), 120; progress=false)
                comp = m.args.M.components[1]
                field = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M,
                                         nothing).structured[1]
                @test all(isfinite, field)
                @test maximum(abs, field) > 1e-8
            end
        end

        # `_resolve_hyper_samples` itself, which is the shared fix.
        @testset "_resolve_hyper_samples" begin
            fake = Dict(:sigma_idx => [2.0, 4.0])
            @test TethysBayes._resolve_hyper_samples(fake, ["sigma_idx"], :sigma_idx, 1.0, 1, false, 2) == [2.0, 4.0]
            # Absent from the chain but pinned: the constant, replicated.
            @test TethysBayes._resolve_hyper_samples(fake, [], :sigma_idx, 1.5, 1, false, 3) == fill(1.5, 3)
            # Absent and not pinned: genuinely unresolvable, so `nothing`.
            @test isnothing(TethysBayes._resolve_hyper_samples(fake, [], :sigma_idx, Normal(0, 1), 1, false, 3))
            # `as_matrix` gives the (n_samples, 1) shape some sites index as `[i, 1]`. It has
            # to be expressed here rather than as a trailing `reshape`, because
            # `reshape(nothing, ...)` would throw and break the genuinely-absent path.
            @test size(TethysBayes._resolve_hyper_samples(fake, ["sigma_idx"], :sigma_idx, 1.0,
                                                   1, false, 2; as_matrix = true)) == (2, 1)
            @test isnothing(TethysBayes._resolve_hyper_samples(fake, [], :sigma_idx, Normal(0, 1),
                                                       1, false, 3; as_matrix = true))
        end
    end

    @testset "every null direction is deflated, on a DISCONNECTED graph" begin
        # 14 sites across 7 component files used a hard-coded `diag_D[1] = 0` (sometimes
        # `diag_D[1] = 0.0; diag_D[2] = 0.0`) to impose sum-to-zero. That count is a
        # property of the eigen-spectrum, not of the source file, and it was wrong in both
        # directions on a disconnected graph:
        #
        #  * one zeroed direction, but a 2-component graph has 2 null directions. A null
        #    eigenvalue is `L ~ 0`, so `diag_D = sigma / sqrt(L + noise)` is LARGE for an
        #    undeflated mode -- the leftover direction carried variance `sigma^2 / noise`
        #    (100 at noise=0.1), i.e. a wildly over-dispersed component mean.
        #  * two zeroed directions, but a 1-D path penalty (RW2) has a one-dimensional null
        #    space however the index set is arranged, so the second zero deleted a real
        #    degree of freedom.
        #
        # All of them now call `_zero_null_modes!`, in the generated model body AND the
        # `get_effects` CPU reconstruction -- the two must agree or reconstruction stops
        # reproducing what the model sampled.
        ndisc = 8
        ddf = DataFrame(idx = collect(1:ndisc), x = collect(1.0:ndisc), y = zeros(Float64, ndisc))
        ddf.depth = 9.0 .+ 3.0 .* sin.(ddf.x ./ 3.0)

        Wdisc = spzeros(Int, ndisc, ndisc)
        for (i, j) in [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)]
            Wdisc[i, j] = 1
            Wdisc[j, i] = 1
        end
        Wconn = copy(Wdisc)
        Wconn[4, 5] = 1
        Wconn[5, 4] = 1
        Wconn[ndisc, 1] = 1
        Wconn[1, ndisc] = 1

        # The template must report one null direction per component.
        @testset "template null count" begin
            td = TethysBayes.build_structure_template(:icar, ndisc; W = Wdisc)
            tc = TethysBayes.build_structure_template(:icar, ndisc; W = Wconn)
            @test count(<=(1e-10 * maximum(td.L)), td.L) == 2
            @test count(<=(1e-10 * maximum(tc.L)), tc.L) == 1
            @test td.n_components == 2
        end

        # The observable consequence: a prior draw must have ~zero variance along EVERY
        # null direction. This goes through the generated model body and `get_effects`,
        # not just the template.
        @testset "no variance leaks onto the null space" begin
            noise_v = 0.1
            for model in (:icar, :besag, :bym2, :leroux, :cyclic, :rw1, :rw2)
                formula = "likelihood(depth, family=gaussian) ~ intercept() + " *
                          "random(idx, model=$model, sigma=1.0)"
                mv = TethysBayes.TethysBayes_core(formula, ddf; W = Wdisc, noise = noise_v, verbose = false)
                chv = sample(MersenneTwister(4), mv, Prior(), 400; progress = false)
                compv = mv.args.M.components[1]
                field = TethysBayes.get_effects(compv.component_obj, chv, compv, mv.args.M,
                                         nothing).structured[1]
                @test all(isfinite, field)
                # Sum-to-zero within each connected component. An undeflated null mode
                # shows up as a component mean wandering at ~sqrt(sigma^2/noise) = 3.2.
                for comp_idx in (1:4, 5:8)
                    cm = vec(mean(field[comp_idx, :], dims = 2))
                    @test maximum(abs, cm) < 0.5
                end
            end
        end

        # And the helper itself, including the over-deflation direction.
        @testset "_zero_null_modes! derives the count from the spectrum" begin
            d1 = ones(4)
            @test TethysBayes._zero_null_modes!(d1, [0.0, 1.0, 2.0, 3.0]) == [1]
            @test d1 == [0.0, 1.0, 1.0, 1.0]
            d2 = ones(4)
            @test TethysBayes._zero_null_modes!(d2, [0.0, 0.0, 2.0, 3.0]) == [1, 2]
            @test d2 == [0.0, 0.0, 1.0, 1.0]
            # A 1-D path penalty has ONE null direction, and the helper must not touch
            # the rest -- the failure mode of the old `[1]`-and-`[2]` form.
            d3 = ones(3)
            @test TethysBayes._zero_null_modes!(d3, [0.0, 1.0, 2.0]) == [1]
            @test d3[2] == 1.0
            @test TethysBayes._zero_null_modes!(ones(0), Float64[]) == Int[]
        end
    end

    @testset "components the generic sweep could not reach now have real fixtures" begin
        # `scripts/_sweep_pinned_sigma.jl` uses one generic fixture, so these components
        # failed for want of a required argument rather than because of a defect. Each is
        # given the geometry it documents here, which turns "never exercised" into covered.
        # None of them is a smoke test: every case asserts a NONZERO reconstructed field,
        # because that is the only assertion that catches the silent-zero-field class.
        nfix = 16

        function ring(r, n)
            ang = range(0, 2pi; length = n + 1)[1:n]
            DataFrame(x = r .* cos.(ang), y = r .* sin.(ang))
        end

        @testset "hyperbolic: 2-D coordinates inside the unit disk" begin
            # The sweep passed `idx` -- the integers 1:12 -- as the spatial variable, so the
            # "coordinates" lay far outside the unit disk and the kernel matrix was singular
            # (`PosDefException`). With genuine 2-D in-disk coordinates it samples fine, and
            # the kernel is PSD: eigenvalues [0.0015, 5.2]. So this was a fixture limitation.
            hf = ring(0.7, nfix)
            hf.depth = 9.0 .+ 2.0 .* cos.(range(0, 2pi; length = nfix + 1)[1:nfix])
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(x, y, model=hyperbolic, sigma=1.0, curvature=1.0)"
            m = TethysBayes.TethysBayes_core(f, hf; verbose = false)
            @test m isa DynamicPPL.Model
            ch = sample(MersenneTwister(3), m, MH(), 120; progress = false)
            comp = m.args.M.components[1]
            field = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M, nothing).structured[1]
            @test all(isfinite, field)
            @test maximum(abs, field) > 1e-8
            # The kernel itself must be positive definite on a valid configuration.
            ang = range(0, 2pi; length = nfix + 1)[1:nfix]
            C = hcat([0.7 * cos(a) for a in ang], [0.7 * sin(a) for a in ang])
            K = TethysBayes._evaluate_hyperbolic_kernel_matrix(C, 1.0, 1.0, 0.0)
            @test minimum(eigvals(Symmetric(Matrix(K)))) > 0
        end

        @testset "eigen: PCA needs at least two variables" begin
            # With a single column the factor count clamps to `n_vars - 1 = 0`, giving a 12x0
            # matrix and a `BoundsError`. That is correct behaviour for a PCA, not a defect --
            # but it means the component needs a genuine multi-column fixture.
            ef = DataFrame(idx = collect(1:20), a = collect(1.0:20.0),
                           b = sin.(collect(1.0:20.0) ./ 3.0),
                           c = cos.(collect(1.0:20.0) ./ 2.0))
            ef.depth = 9.0 .+ 0.5 .* ef.a .+ 0.3 .* ef.b
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(a, b, model=eigen, sigma=1.0, n_factors=1)"
            m = TethysBayes.TethysBayes_core(f, ef; verbose = false)
            ch = sample(MersenneTwister(3), m, MH(), 120; progress = false)
            comp = m.args.M.components[1]
            field = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M, nothing).structured[1]
            @test all(isfinite, field)
            @test maximum(abs, field) > 1e-8
        end

        @testset "networkflow: with the habitat column it documents" begin
            # `beta` is documented as optional with default `Normal(0, 1)`, but the
            # constructor used `p.beta` with no fallback -- so omitting it, the documented
            # usage, raised `FieldError: type NamedTuple has no field beta`.
            nf = DataFrame(idx = collect(1:nfix), x = collect(1.0:nfix),
                           y = zeros(Float64, nfix))
            nf.habitat = 1.0 .+ 0.5 .* sin.(nf.x ./ 2.0)
            nf.depth = 9.0 .+ 3.0 .* sin.(nf.x ./ 3.0)
            Wn = spzeros(Int, nfix, nfix)
            for j in 1:(nfix - 1)
                Wn[j, j + 1] = 1
                Wn[j + 1, j] = 1
            end
            Wn[1, nfix] = 1
            Wn[nfix, 1] = 1
            # `beta=` omitted on purpose: it is the documented default path.
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(idx, model=networkflow, sigma=1.0, habitat=habitat)"
            m = TethysBayes.TethysBayes_core(f, nf; W = Wn, verbose = false)
            ch = sample(MersenneTwister(3), m, MH(), 120; progress = false)
            comp = m.args.M.components[1]
            field = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M, nothing).structured[1]
            @test all(isfinite, field)
            @test maximum(abs, field) > 1e-8
            # And the pinned-beta path, which also required the struct field to accept Real.
            f2 = "likelihood(depth, family=gaussian) ~ intercept() + " *
                 "random(idx, model=networkflow, sigma=1.0, habitat=habitat, beta=0.5)"
            m2 = TethysBayes.TethysBayes_core(f2, nf; W = Wn, verbose = false)
            ch2 = sample(MersenneTwister(3), m2, MH(), 120; progress = false)
            c2 = m2.args.M.components[1]
            @test maximum(abs, TethysBayes.get_effects(c2.component_obj, ch2, c2, m2.args.M,
                                                nothing).structured[1]) > 1e-8
        end

        @testset "sparse GP: inducing count below the sample count" begin
            # The default is 20 inducing points, which cannot fit a 12-point fixture
            # (`B has first dimension 20 but needs 12`). Not a defect, but it meant three
            # sparse-GP components were never exercised at all.
            sg = DataFrame(x = collect(range(-2.0, 2.0; length = 30)),
                           y = collect(range(-2.0, 2.0; length = 30)))
            sg.depth = 9.0 .+ 2.0 .* sin.(sg.x)
            for model in (:fitc, :sparsegp, :nystrom)
                f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                    "random(x, y, model=$model, n_inducing=8, sigma=1.0)"
                m = TethysBayes.TethysBayes_core(f, sg; verbose = false)
                ch = sample(MersenneTwister(3), m, MH(), 80; progress = false)
                comp = m.args.M.components[1]
                field = TethysBayes.get_effects(comp.component_obj, ch, comp, m.args.M,
                                         nothing).structured[1]
                @test all(isfinite, field)
                @test maximum(abs, field) > 1e-8
            end
        end

        @testset "AR(1) boundary: finite and convergent as |rho| -> 1" begin
            # The stationary scale `sigma / sqrt(1 - rho^2)` genuinely diverges at rho = 1 --
            # the process has no stationary distribution there -- so a large scale is correct.
            # What must hold is that the marginal likelihood stays finite, free of NaN, and
            # *converges* as |rho| -> 1.
            #
            # Note it is NOT monotone in |rho|, and should not be: the likelihood is maximised
            # near the data's own autocorrelation, so it rises from rho = 0 to a peak and then
            # falls. Measured here the peak is at rho ~ 0.9. Asserting monotone decrease was
            # my error, not the code's.
            yy = [1.0, 1.2, 0.9, 1.5, 1.8, 1.1, 1.3, 1.0]
            tt = collect(1:length(yy))
            lls = Float64[]
            for rho in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999)
                v = TethysBayes._ar1_log_marginal_likelihood(yy, tt, length(yy), 0.5, rho, 0.2)
                @test isfinite(v)
                push!(lls, v)
            end
            # Bounded, and the tail settles rather than running off.
            @test maximum(lls) < 0.0
            @test abs(lls[end] - lls[end - 1]) < 0.01
            @test maximum(lls) - lls[end] < 2.0
            # At and just past rho = 1 the scale overflows. The implementation returns the
            # finite limiting value rather than a NaN, which is the graceful outcome: the
            # caller gets a usable number. Either that or a clearly non-finite value is
            # acceptable; a NaN masquerading as a likelihood is not.
            for rho in (1.0, 1.0000001)
                v = try
                    TethysBayes._ar1_log_marginal_likelihood(yy, tt, length(yy), 0.5, rho, 0.2)
                catch
                    NaN
                end
                @test !isnan(v)
            end
        end
    end

    @testset "`spacetime` is removed as a component; `svar` and the Kronecker form remain" begin
        # `spacetime` was never a distinct space-time model: it was a bare alias, with
        # `COMPONENT_CONSTRUCTORS[:spacetime] = COMPONENT_CONSTRUCTORS[:svar]` and both
        # registry entries pointing at the same `SVAR` struct. It was removed because the DSL
        # already expresses the structure the current way -- as a Kronecker product of a
        # spatial and a temporal effect -- and because leaving `svar` reachable under two names
        # is how the two drifted apart in the first place.
        #
        # `:spacetime` SURVIVES as a structure kind (not a component name): `svar`,
        # `tensorproductsmooth` and `dynamics` all map to it, and the assembly, reconstruction
        # and plotting code branches on `structure == :spacetime`. Only the alias is gone.
        @test !haskey(TethysBayes.COMPONENT_TYPE_REGISTRY, :spacetime)
        @test haskey(TethysBayes.COMPONENT_TYPE_REGISTRY, :svar)
        # A constructor is retained so the failure carries guidance rather than being a bare
        # "unknown model" -- the most common way a removed API name hurts someone.
        @test haskey(TethysBayes.COMPONENT_CONSTRUCTORS, :spacetime)

        ns, nt = 4, 3
        nst = ns * nt
        dst = DataFrame(s_idx = repeat(1:ns, nt), s_x = repeat(collect(1.0:ns), nt),
                        s_y = zeros(nst), t_idx = repeat(1:nt, inner = ns),
                        t = repeat(collect(1.0:nt), inner = ns))
        dst.depth = [10.0, 12.0, 9.0, 11.0][dst.s_idx] .+ dst.t
        Wst = spzeros(Int, ns, ns)
        for j in 1:(ns - 1)
            Wst[j, j + 1] = 1
            Wst[j + 1, j] = 1
        end
        Wst[1, ns] = 1
        Wst[ns, 1] = 1

        @testset "the removed name fails with guidance" begin
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(s_idx, model=spacetime, sigma=1.0)"
            err = try
                TethysBayes.TethysBayes_core(f, dst; W = Wst, verbose = false)
                nothing
            catch e
                e
            end
            @test err !== nothing
            msg = sprint(showerror, err)
            # It must point at both replacements, or the removal is user-hostile.
            @test occursin("svar", msg)
            @test occursin("Kronecker", msg)
        end

        @testset "the supported forms still work" begin
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(s_idx, model=svar, sigma=1.0)"
            m = TethysBayes.TethysBayes_core(f, dst; W = Wst, verbose = false)
            @test m isa DynamicPPL.Model
            @test sample(MersenneTwister(3), m, Prior(), 10; progress=false) isa Any

            # The modern form builds a SEPARABLE structure -- a spatial and a temporal
            # effect plus their composition -- rather than one monolithic component.
            f2 = "likelihood(depth, family=gaussian) ~ intercept() + " *
                 "random(s_idx, model=icar, sigma=1.0) " * "\u2297" * " " *
                 "random(t_idx, model=ar1, sigma=1.0)"
            m2 = TethysBayes.TethysBayes_core(f2, dst; W = Wst, verbose = false)
            @test m2 isa DynamicPPL.Model
            kinds = Set(nameof(typeof(c.component_obj)) for c in m2.args.M.components)
            @test :ICAR in kinds
            @test :AR1 in kinds
            @test sample(MersenneTwister(3), m2, Prior(), 10; progress=false) isa Any
        end

        @testset "the structure kind survives" begin
            @test TethysBayes.MODEL_TO_STRUCTURE_MAP[:svar] == :spacetime
            @test TethysBayes.MODEL_TO_STRUCTURE_MAP[:tensorproductsmooth] == :spacetime
            @test TethysBayes.MODEL_TO_STRUCTURE_MAP[:dynamics] == :spacetime
        end
    end
end
