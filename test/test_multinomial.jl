# ==============================================================================
# TethysBayes Test Suite: Multinomial, Categorical & Dirichlet Formulations
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Multinomial, Categorical & Dirichlet Formulations" begin
    # --------------------------------------------------------------------------
    # 1. Wide-Count Multinomial Model
    # --------------------------------------------------------------------------
    @testset "Wide-Count Multinomial Model" begin
        N_wide = 20
        df_wide = DataFrame(
            c1 = rand(Poisson(10), N_wide),
            c2 = rand(Poisson(5), N_wide),
            c3 = rand(Poisson(8), N_wide),
            x1 = randn(N_wide)
        )

        m_wide = @tbm(
            likelihood(c1 + c2 + c3, family=:multinomial) ~ intercept() + fixed(x1),
            df_wide,
            verbose = false
        )

        chn_wide = sample(m_wide, MH(), 50, progress=false)
        @test size(chn_wide, 1) == 50

        res_wide = TethysBayes.model_results_comprehensive(m_wide, chn_wide)

        @test haskey(res_wide.predictions, :probabilities)
        @test haskey(res_wide.predictions, :predicted_category)
        @test haskey(res_wide.metrics, :waic)
        @test isfinite(res_wide.metrics.waic)

        # Verify simplex property: sum(p_k) ≈ 1.0 across all observations
        probs_wide = res_wide.predictions.probabilities
        prob_keys = filter(k -> startswith(string(k), "p_"), keys(probs_wide))
        @test length(prob_keys) == 3

        prob_sums = sum([probs_wide[k].mean for k in prob_keys])
        @test all(isapprox.(prob_sums, 1.0, atol=1e-5))
    end

    # --------------------------------------------------------------------------
    # 2. Long-Categorical Factor Model with Custom Reference Category
    # --------------------------------------------------------------------------
    @testset "Long-Categorical Factor Model with Custom Reference" begin
        cats = ["cod", "haddock", "halibut"]
        species_vec = rand(cats, 30)
        df_cat = DataFrame(
            species = species_vec,
            temp = randn(30)
        )

        m_cat = @tbm(
            likelihood(species, family=:categorical, reference="cod") ~
                intercept() + fixed(temp),
            df_cat,
            verbose = false
        )

        chn_cat = sample(m_cat, MH(), 50, progress=false)
        @test size(chn_cat, 1) == 50

        res_cat = TethysBayes.model_results_comprehensive(m_cat, chn_cat)
        @test haskey(res_cat.predictions, :probabilities)
        @test haskey(res_cat.predictions, :predicted_category)
        @test isfinite(res_cat.metrics.waic)

        # Verify MAP dominant category values belong to original category levels
        pred_cats = res_cat.predictions.predicted_category.mean
        @test all(c -> c in cats, pred_cats)
    end

    # --------------------------------------------------------------------------
    # 3. Dirichlet-Multinomial Model with Dispersion
    # --------------------------------------------------------------------------
    @testset "Dirichlet-Multinomial Model with Dispersion" begin
        N_wide = 20
        df_wide = DataFrame(
            c1 = rand(Poisson(10), N_wide),
            c2 = rand(Poisson(5), N_wide),
            c3 = rand(Poisson(8), N_wide),
            x1 = randn(N_wide)
        )

        m_dm = @tbm(
            likelihood(c1 + c2 + c3, family=:dirichlet_multinomial) ~
                intercept() + fixed(x1),
            df_wide,
            verbose = false
        )

        chn_dm = sample(m_dm, MH(), 50, progress=false)
        @test mean(chn_dm[:dirichlet_phi]) > 0.0

        res_dm = TethysBayes.model_results_comprehensive(m_dm, chn_dm)
        @test isfinite(res_dm.metrics.waic)
    end

    # --------------------------------------------------------------------------
    # 4. Dirichlet Compositional Proportions Model
    # --------------------------------------------------------------------------
    @testset "Dirichlet Compositional Proportions Model" begin
        raw_mat = rand(Gamma(2.0, 1.0), 20, 3)
        prop_mat = raw_mat ./ sum(raw_mat, dims=2)
        df_dir = DataFrame(
            p1 = prop_mat[:, 1],
            p2 = prop_mat[:, 2],
            p3 = prop_mat[:, 3],
            x1 = randn(20)
        )

        m_dir = @tbm(
            likelihood(p1 + p2 + p3, family=:dirichlet) ~ intercept() + fixed(x1),
            df_dir,
            verbose = false
        )

        chn_dir = sample(m_dir, MH(), 50, progress=false)
        @test mean(chn_dir[:dirichlet_phi]) > 0.0
        res_dir = TethysBayes.model_results_comprehensive(m_dir, chn_dir)
        @test isfinite(res_dir.metrics.waic)
    end
end
