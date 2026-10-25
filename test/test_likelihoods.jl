# ==============================================================================
# TethysBayes Test Suite: Likelihood Engine Taxonomy and Distribution Correctness
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Likelihood Engine Taxonomy" begin
    @testset "Discrete Families (ZI & Hurdle)" begin
        mu_p = exp(1.0)
        phi = 0.2
        d_pois = TethysBayes.TethysBayes_Likelihood("poisson", [1.0]; phi_zi=phi)
        ana_pois = LogExpFunctions.logsumexp(log(phi), log(1 - phi) + logpdf(Poisson(mu_p), 0))
        @test isapprox(Distributions.logpdf(d_pois, 0.0), ana_pois)

        mu_nb = exp(1.5)
        r_val = 2.0
        d_nb_h = TethysBayes.TethysBayes_Likelihood("negbin", [1.5]; r_nb=r_val, phi_hurdle=0.9, hurdle=0.0)
        dist_nb = NegativeBinomial(r_val, r_val / (r_val + mu_nb))
        ana_nb_h = log(0.9) + logpdf(dist_nb, 2.0) - logccdf(dist_nb, 0.0)
        @test isapprox(Distributions.logpdf(d_nb_h, 2.0), ana_nb_h)

        d_bin_int = TethysBayes.TethysBayes_Likelihood(
            "binomial", [0.0]; trial=10, censor_lower=3.0, censor_upper=5.0
        )
        dist_bin = Binomial(10, LogExpFunctions.logistic(0.0))
        ana_bin_int = stable_logdiffexp(logcdf(dist_bin, 5.0), logcdf(dist_bin, 2.0))
        @test isapprox(Distributions.logpdf(d_bin_int, NaN), ana_bin_int)
    end

    @testset "Continuous Families (Censoring)" begin
        d_gauss = TethysBayes.TethysBayes_Likelihood("gaussian", [1.0]; sigma_y=0.5, censor_lower=2.0)
        @test isapprox(Distributions.logpdf(d_gauss, NaN), logccdf(Normal(1.0, 0.5), 2.0))

        d_beta = TethysBayes.TethysBayes_Likelihood("beta", [-0.5]; extra_params=20.0)
        mu_b = LogExpFunctions.logistic(-0.5)
        dist_beta = Beta(mu_b * 20.0, (1 - mu_b) * 20.0)
        @test isapprox(Distributions.logpdf(d_beta, 0.4), logpdf(dist_beta, 0.4))

        d_ln = TethysBayes.TethysBayes_Likelihood("lognormal", [0.5]; sigma_y=0.3, censor_upper=1.5)
        mu_ln = 0.5 - (0.3^2) / 2.0
        @test isapprox(Distributions.logpdf(d_ln, NaN), logcdf(LogNormal(mu_ln, 0.3), 1.5))
    end

    @testset "Likelihood Families & Distributions Correctness" begin
        nb_lik = TethysBayes.TethysBayes_Likelihood("negbin", [1.0]; r_nb=2.0)
        logp_nb = logpdf(nb_lik, 3)
        @test isfinite(logp_nb)
    end

    @testset "Vectorized Observation Parameters & Censoring" begin
        # 1. Vectorized weights
        eta_w = [0.1, 0.2, -0.1, 0.5]
        y_w = [1, 2, 0, 3]
        weights = [1.0, 2.0, 0.5, 1.5]
        lik_w = TethysBayes.TethysBayes_Likelihood(:poisson, eta_w; weight=weights)
        lp_w = logpdf(lik_w, y_w)
        expected_lp_w = sum(weights[i] * logpdf(Poisson(exp(eta_w[i])), y_w[i]) for i in 1:4)
        @test isapprox(lp_w, expected_lp_w; atol=1e-12)

        # 2. Vectorized trials for Binomial
        eta_bin = [0.0, 1.0, -1.0]
        trials = [10, 25, 50]
        y_bin = [5, 15, 10]
        lik_bin = TethysBayes.TethysBayes_Likelihood(:binomial, eta_bin; trial=trials)
        lp_bin = logpdf(lik_bin, y_bin)
        p_bin = [1.0 / (1.0 + exp(-e)) for e in eta_bin]
        expected_bin_lp = sum(logpdf(Binomial(trials[i], p_bin[i]), y_bin[i]) for i in 1:3)
        @test isapprox(lp_bin, expected_bin_lp; atol=1e-12)

        # 3. Vectorized censoring thresholds
        eta_censor = [1.0, 1.0]
        censor_limits = [10.0, 20.0]
        y_censor = [10.0, 20.0]
        lik_censor = TethysBayes.TethysBayes_Likelihood(:gaussian, eta_censor; sigma_y=2.0, censor_lower=censor_limits)
        lp_censor = logpdf(lik_censor, y_censor)
        expected_censor_lp = sum(logccdf(Normal(eta_censor[i], 2.0), censor_limits[i]) for i in 1:2)
        @test isapprox(lp_censor, expected_censor_lp; atol=1e-6)

        # 4. Empty array raises ArgumentError
        @test_throws ArgumentError TethysBayes.TethysBayes_Likelihood(:poisson, eta_w; trial=Int[])

        # 5. Length mismatch raises DimensionMismatch
        @test_throws DimensionMismatch TethysBayes.TethysBayes_Likelihood(:poisson, eta_w; weight=[1.0, 2.0])
    end

    @testset "Vectorized Censoring in Model Formula and Config" begin
        rng = MersenneTwister(42)
        N = 10
        df_censor = DataFrame(
            y = rand(rng, 1.0:10.0, N),
            x = randn(rng, N),
            det_limit = fill(2.0, N)
        )
        df_censor.det_limit[1:5] .= 1.0
        df_censor.det_limit[6:10] .= 3.0

        # Formula with column symbol censor_lower
        cfg1 = TethysBayes.TethysBayes_config(
            "likelihood(y, censor_lower=det_limit) ~ intercept() + fixed(x)",
            df_censor
        )
        @test cfg1[:user_provided_censor_lower] == true
        @test size(cfg1[:censor_lower]) == (N, 1)
        @test cfg1[:censor_lower][1:5, 1] == fill(1.0, 5)
        @test cfg1[:censor_lower][6:10, 1] == fill(3.0, 5)

        # Formula with direct vector censor_lower
        my_limits = [10.0, 20.0, 10.0, 15.0, 12.0, 18.0, 14.0, 16.0, 11.0, 13.0]
        cfg2 = TethysBayes.TethysBayes_config(
            "likelihood(y, censor_lower=my_limits) ~ intercept() + fixed(x)",
            df_censor;
            my_limits = my_limits
        )
        @test cfg2[:censor_lower][:, 1] == my_limits
    end

    @testset "Zero-Inflation and Hurdle Mutual Exclusivity" begin
        eta = 0.5

        # Individual models succeed
        d_zi = TethysBayes.TethysBayes_Likelihood(:poisson, eta; phi_zi=0.2)
        @test d_zi.zi_state isa TethysBayes.ZeroInflated

        d_hurdle = TethysBayes.TethysBayes_Likelihood(:poisson, eta; phi_hurdle=0.2, hurdle=0.0)
        @test d_hurdle.phi_hurdle > -Inf
        @test d_hurdle.hurdle == 0.0

        # Conflicting combinations raise ArgumentError
        @test_throws ArgumentError TethysBayes.TethysBayes_Likelihood(:poisson, eta; phi_zi=0.3, hurdle=10.0)
        @test_throws ArgumentError TethysBayes.TethysBayes_Likelihood(:poisson, eta; phi_zi=0.3, phi_hurdle=0.5)

        # Formula parser raises ArgumentError when both specified
        df_conflict = DataFrame(y = [0, 1, 2], x = [0.1, 0.2, 0.3])
        @test_throws ArgumentError TethysBayes.decompose_TethysBayes_formula(
            "likelihood(y, family=poisson, zero_inflated=true, hurdle=0.0) ~ intercept() + fixed(x)",
            df_conflict
        )
    end

    @testset "Multivariate Normal Likelihood (MvNormalFamily)" begin
        lik_mvn = TethysBayes.TethysBayes_Likelihood(:mvnormal, [1.0, 2.0]; sigma_y = [0.5, 0.8])
        @test lik_mvn.family isa TethysBayes.MvNormalFamily
        y_obs_vec = [1.2, 1.9]
        lp_vec = logpdf(lik_mvn, y_obs_vec)
        expected_lp = logpdf(MvNormal([1.0, 2.0], Diagonal([0.5^2, 0.8^2])), y_obs_vec)
        @test isapprox(lp_vec, expected_lp; atol=1e-8)

        # Matrix evaluation (batch of multivariate observations)
        Y_obs_mat = [1.2 1.9; 0.9 2.1]
        lp_mat = logpdf(lik_mvn, Y_obs_mat)
        expected_mat_lp = logpdf(MvNormal([1.0, 2.0], Diagonal([0.5^2, 0.8^2])), [1.2, 1.9]) +
                          logpdf(MvNormal([1.0, 2.0], Diagonal([0.5^2, 0.8^2])), [0.9, 2.1])
        @test isapprox(lp_mat, expected_mat_lp; atol=1e-8)

        # Sampling
        mvn_sample = rand(lik_mvn)
        @test length(mvn_sample) == 2
    end

    @testset "Censoring Boundary Validation (cl < cu)" begin
        # 1. Scalar bounds lower >= upper
        @test_throws ArgumentError TethysBayes.TethysBayes_Likelihood(
            :gaussian, [1.0]; sigma_y=1.0, censor_lower=100.0, censor_upper=10.0
        )
        @test_throws ArgumentError TethysBayes.TethysBayes_Likelihood(
            :gaussian, [1.0]; sigma_y=1.0, censor_lower=5.0, censor_upper=5.0
        )

        # 2. Vector bounds lower >= upper
        @test_throws ArgumentError TethysBayes.TethysBayes_Likelihood(
            :gaussian, [1.0, 2.0]; sigma_y=1.0, censor_lower=[1.0, 10.0], censor_upper=[5.0, 8.0]
        )

        # 3. Vector length mismatch
        @test_throws DimensionMismatch TethysBayes.TethysBayes_Likelihood(
            :gaussian, [1.0, 2.0]; sigma_y=1.0, censor_lower=[1.0, 2.0], censor_upper=[5.0, 6.0, 7.0]
        )

        # 4. Valid interval censoring
        lik_valid_censor = TethysBayes.TethysBayes_Likelihood(
            :gaussian, [1.0, 2.0]; sigma_y=1.0, censor_lower=[1.0, 2.0], censor_upper=[5.0, 8.0]
        )
        @test lik_valid_censor.censoring_state isa TethysBayes.IntervalCensored
    end

    @testset "N1: truncated counts are ForwardDiff-capable" begin
        using ForwardDiff

        # The hurdle and censoring kernels normalise by a count tail. For Poisson that
        # normaliser is `logccdf`, which routes through Distributions' `_gammalogcdf` --
        # Float64-only -- so NUTS/HMC/ADVI all failed with a MethodError while MH worked.
        # `logpdf` was always AD-capable, which is why only the tail was the problem.
        _pois(lam) = TethysBayes.get_dist_ref(
            TethysBayes.get_model_family("poisson"), TethysBayes.TethysBayes_Likelihood("poisson", lam), lam, 1.0, 1)

        @testset "AD path reproduces the Float64 path" begin
            for (fam, lam) in [("poisson", 0.4), ("poisson", 1.3), ("poisson", 5.0),
                               ("negbin", 1.3)]
                dist = TethysBayes.get_dist_ref(
                    TethysBayes.get_model_family(fam), TethysBayes.TethysBayes_Likelihood(fam, lam), lam, 1.0, 1)
                for thr in (0.0, 0.5, 1.0, 2.0, 3.0)
                    @test isapprox(TethysBayes._logcdf_count_ad(dist, thr), logcdf(dist, thr); rtol=1e-11)
                    @test isapprox(TethysBayes._logccdf_count_ad(dist, thr), logccdf(dist, thr); rtol=1e-11)
                end
            end
        end

        @testset "gradients are correct, not merely finite" begin
            # An AD path that returned a plausible value with a wrong derivative would
            # break NUTS silently, so compare against finite differences.
            for lam in (0.5, 1.5, 4.0)
                f(l) = TethysBayes._logccdf_count_ad(_pois(l), 1.0)
                g = ForwardDiff.derivative(f, lam)
                h = 1e-6
                fd = (f(lam + h) - f(lam - h)) / (2h)
                @test isfinite(g)
                @test isapprox(g, fd; rtol=1e-5)
            end
        end

        @testset "the plain-float path is untouched" begin
            # `is_ad == false` must delegate to Distributions verbatim, so no reported
            # number moves. `===` rather than `isapprox`: it must be the identical value.
            for thr in (0.5, 1.0, 2.0)
                @test TethysBayes._logcdf_count(_pois(1.3), thr, false) === logcdf(_pois(1.3), thr)
                @test TethysBayes._logccdf_count(_pois(1.3), thr, false) === logccdf(_pois(1.3), thr)
            end
            # And the dual/Float64 discrimination used to pick the branch. Note it takes a
            # TYPE, which is how the kernels call it (`_is_dual_type(V)`).
            @test !TethysBayes._is_dual_type(Float64)
            @test TethysBayes._is_dual_type(typeof(ForwardDiff.Dual(1.0, 1.0)))
        end

        @testset "edge cases" begin
            # A bound below the support: P(X <= k) = 0 and P(X > k) = 1.
            @test TethysBayes._logcdf_count_ad(_pois(1.3), -1.0) == -Inf
            @test TethysBayes._logccdf_count_ad(_pois(1.3), -1.0) == 0.0
            # A non-integral hurdle bound must floor rather than throw an InexactError.
            @test isfinite(TethysBayes._logcdf_count_ad(_pois(1.3), 0.5))
            # An absurdly large bound reports an actionable error instead of grinding.
            @test_throws ErrorException TethysBayes._logcdf_count_ad(_pois(1.3), 1e7)
        end

        @testset "hurdle and censored models sample with NUTS" begin
            s_N = 6
            df = DataFrame(y = [1, 3, 2, 0, 4, 1, 2, 5, 1, 0, 3, 2], s_idx = repeat(1:s_N, 2))
            W = TethysBayes.spatial_knn_graph([(Float64(s), 0.0) for s in 1:s_N], 2)[2]
            for spec in ["hurdle=0.5", "hurdle=1.0", "hurdle=2.0",
                         "hurdle=0.5, censor_lower=2",
                         "hurdle=0.5, censor_upper=2",
                         "hurdle=0.5, censor_lower=1, censor_upper=3"]
                f = "likelihood(y, family=poisson, $spec) ~ 1 + random(s_idx, model=icar)"
                m = TethysBayes.TethysBayes_core(f, df; s_N=s_N, W=W, verbose=false)
                ch = sample(MersenneTwister(5), m, NUTS(), 40;
                            progress=false, check_model=false)
                @test size(ch, 1) == 40
            end
        end
    end
end
