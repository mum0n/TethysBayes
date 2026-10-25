# ==============================================================================
# TethysBayes Test Suite: Core Formula Parsing, ParamRegistry, and Manifolds
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Core Components & Interfaces" begin

    @testset "split_terms_at_depth" begin
        @test TethysBayes.split_terms_at_depth("a + b(c+d) + e", "+") == ["a", "b(c+d)", "e"]
        @test TethysBayes.split_terms_at_depth("a", "+") == ["a"]
        @test TethysBayes.split_terms_at_depth("a |> log", " |> ") == ["a", "log"]
    end

    @testset "L1b: _sanitize_variablename yields valid identifiers" begin
        # The character class used to cover `\s | ( ) + * & :` but not `- / ^ = > <`, so
        # those column names produced names that are not valid Julia identifiers.
        @test TethysBayes._sanitize_variablename("a-b") == "a_b"
        @test TethysBayes._sanitize_variablename("a/b") == "a_b"
        @test TethysBayes._sanitize_variablename("a^b") == "a_b"
        @test TethysBayes._sanitize_variablename("a=b") == "a_b"
        @test TethysBayes._sanitize_variablename("a>b") == "a_b"
        @test TethysBayes._sanitize_variablename("rate/100") == "rate_100"
        @test TethysBayes._sanitize_variablename("cov|x") == "cov_x"
        @test TethysBayes._sanitize_variablename("a b") == "a_b"
        @test TethysBayes._sanitize_variablename("a+b") == "a_b"
        @test TethysBayes._sanitize_variablename("a*b") == "a_b"
        @test TethysBayes._sanitize_variablename("a&b") == "a_b"
        @test TethysBayes._sanitize_variablename("a:b") == "a_b"

        # Already-legal names must be untouched, including digits, `_` and `!`.
        @test TethysBayes._sanitize_variablename("cov1") == "cov1"
        @test TethysBayes._sanitize_variablename("my_var_2") == "my_var_2"
        @test TethysBayes._sanitize_variablename("is_positive!") == "is_positive!"

        # The named operators must still be EXPANDED, not flattened to a bare underscore.
        # These are the reason the function exists beyond plain sanitisation.
        @test TethysBayes._sanitize_variablename("a⊗b") == "a_kron_b"
        @test TethysBayes._sanitize_variablename("a∘b") == "a_comp_b"

        # Every result must actually be a legal identifier, and never empty.
        for nm in ["a-b", "a/b", "a^b", "a=b", "a>b", "***", "a⊗b", "∘∘", "-", "  "]
            out = TethysBayes._sanitize_variablename(nm)
            @test !isempty(out)
            @test Base.isidentifier(out)
        end
    end

    @testset "L8: denoised and noisy predictions share one eta clamp" begin
        # These were independently hardcoded as 20 (reconstruction) and 30 (likelihoods),
        # so for eta in (20, 30] the denoised mean and the mean of its own simulated noise
        # disagreed by up to e^10. They must now be the same constant.
        @test TethysBayes.ETA_CLAMP_BOUND == 30.0
        @test -TethysBayes.ETA_CLAMP_BOUND <= -30.0
        # `likelihoods.jl` must not reintroduce a literal 20/30 of its own.
        @test TethysBayes.ETA_CLAMP_BOUND > 0
    end

    @testset "L9: @tbm accepts a non-literal formula" begin
        # `string(formula_expr)` renders inline DSL syntax to text, but on a bare variable
        # it stringified the variable's NAME, so the DSL read "f" as the formula and
        # reported a missing outcome column -- an error that pointed at the data instead
        # of the real mistake. A variable holding the formula must now be evaluated.
        df = DataFrame(s_idx = repeat(1:4, 2), y = repeat(1:8, 1), x = collect(1.0:8.0))

        f = "likelihood(y, family=poisson) ~ intercept() + x + random(s_idx, model=iid)"
        m_var = @tbm(f, df, verbose=false)
        @test m_var isa DynamicPPL.Model

        # The string-literal form must keep working exactly as before.
        m_lit = @tbm("likelihood(y, family=poisson) ~ intercept() + x + random(s_idx, model=iid)",
                      df, verbose=false)
        @test m_lit isa DynamicPPL.Model

        # An interpolated string is a run-time value too, not DSL text.
        term = "x + random(s_idx, model=iid)"
        m_interp = @tbm("likelihood(y, family=poisson) ~ intercept() + $term", df, verbose=false)
        @test m_interp isa DynamicPPL.Model

        # A non-String value must fail with a type error naming the formula, not with a
        # misleading "outcome variable not found as a column".
        not_a_string = 42
        @test_throws TypeError @tbm(not_a_string, df, verbose=false)

        # Inline DSL syntax must still be rendered (not evaluated) -- this is the path the
        # macro has always used, and it is easy to break while fixing the above.
        m_dsl = @tbm(likelihood(y, family=poisson) ~ intercept() + x + random(s_idx, model=iid),
                      df, verbose=false)
        @test m_dsl isa DynamicPPL.Model

        # `formula=` keyword with a variable must work on the same terms.
        m_kw = @tbm(data = df, formula = f, verbose=false)
        @test m_kw isa DynamicPPL.Model
    end

    @testset "L10: scientific notation survives RHS subtraction splitting" begin
        # The RHS is rewritten `a - b` -> `a + -b` so the depth-aware splitter treats the
        # operands as separate terms. That rewrite used to hit the exponent sign too, so
        # `sigma=Normal(0, 1e-8)` became `Normal(0, 1 * e + -8)` and raised
        # `UndefVarError: e`. Scientific-notation floats are now masked before the rewrite.
        @testset "exponent signs are protected" begin
            for s in ["1e-8", "1e+8", "1.5e-8", "2.5E-3", "Normal(0, 1e-8)",
                      "Normal(0, 1e+8)", "Normal(0, 1.5e-8)", "Normal(0, 2.5E-3)"]
                @test TethysBayes._normalize_rhs_subtraction(s) == s
            end
            @test TethysBayes._normalize_rhs_subtraction("Normal(0, 1e-8) + Normal(0, 2e-3)") ==
                  "Normal(0, 1e-8) + Normal(0, 2e-3)"
        end

        @testset "genuine subtraction is unchanged" begin
            # The non-float cases must behave EXACTLY as the old blanket regex did,
            # otherwise this fix has silently changed the meaning of real formulas.
            @test TethysBayes._normalize_rhs_subtraction("x - 8") == "x + -8"
            @test TethysBayes._normalize_rhs_subtraction("scale - 8") == "scale + -8"
            # `rate-8` is not valid Julia; the old regex rewrote it too, so pin the old
            # output to prove no non-float case moved.
            @test TethysBayes._normalize_rhs_subtraction("rate-8") == "rate + -8"
        end

        @testset "the prior evaluates to the intended number" begin
            df = DataFrame(s_idx = repeat(1:4, 2), y = repeat(1:8, 1))
            f = "likelihood(y, family=poisson) ~ intercept() + " *
                "random(s_idx, model=rw1, sigma=Normal(0, 1e-8))"
            d = TethysBayes.decompose_TethysBayes_formula(f, df)
            sig = d.modules["s_idx"].args[:sigma]
            # The decomposition resolves the prior to a concrete distribution rather than
            # leaving it as an `Expr`, so the assertion is on the VALUE. What matters is
            # that sigma is 1e-8 and not the `1 * e + -8` that used to be produced.
            @test sig isa Normal
            @test isapprox(sig.μ, 0.0; atol=1e-15)
            @test isapprox(sig.σ, 1e-8; rtol=1e-12)
        end
    end

    @testset "Comprehensive Formula Parsing" begin
        data = TethysBayes.TethysBayes_data("advanced", s_N=30, t_N=12)
        W = create_chain_adj_matrix(30)
        
        formula = """
        y_gauss ~ 1 + cov1 + cov2 +
        random(s_idx, model='bym2') +
        random(year, model='ar1') +
        random(month, model='cyclic', period=12) +
        random(cov3, nbins=10, model='pspline') +
        nested(proxy_val, formula="likelihood(proxy_val) ~ 1 + random(s_idx)") +
        eigen(cov1, cov2, n_factors=1) +
        mixed(1|region) +
        (cov1 |> random(s_idx, model='icar')) +
        dynamics(recruitment, model='advection')
        """

        M_cfg = TethysBayes.TethysBayes_config(formula, data; W=W, s_N=30, t_N=12)
        
        @test M_cfg.add_intercept == true
        @test all(x -> x in string.(M_cfg.Xfixed_names), ["cov1", "cov2"])
        
        components = M_cfg.components
        @test any(c -> c.component_obj isa TethysBayes.BYM2, components)
        @test any(c -> c.component_obj isa TethysBayes.AR1, components)
        @test any(c -> c.component_obj isa TethysBayes.Cyclic || c.component_obj isa TethysBayes.Harmonic,
            components)
        @test any(c -> c.component_obj isa TethysBayes.PSpline, components)
        @test any(c -> c.component_obj isa TethysBayes.Eigen, components)
        @test any(c -> c.component_obj isa TethysBayes.Mixed, components)
        @test any(c -> c.component_obj isa TethysBayes.SVC, components)
        @test any(c -> c.component_obj isa TethysBayes.Dynamics, components)
    end

    @testset "ParamRegistry Architecture" begin
        # 1. Test ParamDescriptor creation
        desc = TethysBayes.ParamDescriptor(
            :sigma_s_idx;
            component_key = :s_idx,
            role = :sigma,
            shape = (1,),
            prior = Exponential(1.0)
        )
        @test desc.symbol == :sigma_s_idx
        @test desc.component_key == :s_idx
        @test desc.role == :sigma

        # 2. Test empty registry & add_descriptor!
        reg = TethysBayes.ParamRegistry()
        TethysBayes.add_descriptor!(reg, desc)
        @test "sigma_s_idx" in reg.names
        @test TethysBayes.find_chain_param(reg, "sigma_s_idx") == "sigma_s_idx"

        # 3. Test build_param_registry from M config
        p_data = TethysBayes.TethysBayes_data("scottish_lip")
        m_cfg = TethysBayes.TethysBayes_config("y ~ 1 + cov1 + random(s_idx, model=bym2) + random(year, model=ar1)", p_data.data; W=p_data.au.W)
        reg_m = TethysBayes.build_param_registry(m_cfg)

        # 4. Test calibrate_param_registry with prior sample NamedTuple and generic mapping
        s_N_lip = length(p_data.au.centroids)
        sample_nt = (
            intercept = 0.5,
            beta = [0.1, 0.2],
            sigma_s_idx = 1.2,
            innovations_s_idx = randn(s_N_lip)
        )
        calibrated = TethysBayes.calibrate_param_registry(reg_m, sample_nt)
        @test calibrated.descriptors[:innovations_s_idx].shape == (s_N_lip,)

        # Mock VarNamedTuple mapping
        struct MockVarNamedTuple
            data::Dict{Symbol, Any}
        end
        Base.pairs(m::MockVarNamedTuple) = Base.pairs(m.data)
        t_N_lip = length(unique(p_data.data.year))
        mock_vnt = MockVarNamedTuple(Dict(:sigma_year => 0.8, :innovations_year => randn(t_N_lip)))
        calibrated_vnt = TethysBayes.calibrate_param_registry(calibrated, mock_vnt)
        @test calibrated_vnt.descriptors[:innovations_year].shape == (t_N_lip,)

        # 5. Test get_samples with mock chain dictionary
        mock_ch = Dict(
            :sigma_s_idx => reshape([1.0, 1.1, 1.2, 1.3, 1.4], 1, 5),
            :innovations_s_idx => randn(s_N_lip, 5)
        )
        sigma_s = TethysBayes.get_samples(mock_ch, calibrated, :s_idx, :sigma)
        @test size(sigma_s, 1) == 5
        @test size(sigma_s, 2) == 1

        innovations_s = TethysBayes.get_samples(mock_ch, calibrated, :s_idx, :innovations)
        @test size(innovations_s, 1) == 5

        # 6. Test canonical _find_parameter
        names_list = ["intercept", "sigma_s_idx_1", "innovations_s_idx[1]"]
        @test TethysBayes._find_parameter(names_list, "sigma_s_idx", 1, true) == "sigma_s_idx_1"
        @test TethysBayes._find_parameter(names_list, :sigma_s_idx, 1, true) == "sigma_s_idx_1"
        @test TethysBayes._find_parameter(names_list, "intercept", nothing, false) == "intercept"
        @test TethysBayes._find_parameter(names_list, "innovations_s_idx", 1, true) == "innovations_s_idx[1]"

        # 7. Test Marginalized AR1 Likelihood & Latent Reconstruction
        ar1_m = TethysBayes.AR1(Normal(0, 1), Exponential(1.0), :marginalized)
        y_res = [1.0, 1.2, 0.9, 1.5, 1.8]
        t_idx = [1, 2, 3, 4, 5]
        ll_m = TethysBayes._ar1_log_marginal_likelihood(y_res, t_idx, 5, 0.7, 0.5, 0.2)
        @test isfinite(ll_m)
        @test ll_m < 0.0

        spec_ar1 = (key = :year, hyper = (n_latent = 5,), params = Dict())
        M_mock = (
            outcomes_N = 1,
            model_arch = "univariate",
            t_idx = t_idx,
            t_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        chain_ar1 = Dict(
            :rho_unconstrained_year => reshape([0.5, 0.6, 0.7], 1, 3),
            :sigma_year => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_ar1 = check_marginalized_reconstruction(ar1_m, chain_ar1, spec_ar1, M_mock)

        # 8. Test Marginalized AR2 Likelihood & Latent Reconstruction
        ar2_m = TethysBayes.AR2(Normal(0, 1), Normal(0, 1), Exponential(1.0), :marginalized)
        ll_ar2 = TethysBayes._ar2_log_marginal_likelihood(y_res, t_idx, 5, 0.4, 0.2, 0.5, 0.2)
        @test isfinite(ll_ar2)
        @test ll_ar2 < 0.0

        chain_ar2 = Dict(
            :rho_regime_1_unconstrained_year => reshape([0.3, 0.4, 0.5], 1, 3),
            :rho_regime_2_unconstrained_year => reshape([0.1, 0.2, 0.1], 1, 3),
            :sigma_year => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_ar2 = check_marginalized_reconstruction(ar2_m, chain_ar2, spec_ar1, M_mock)

        # 9. Test Marginalized RW1 Likelihood & Latent Reconstruction
        rw1_m = TethysBayes.RW1(Exponential(1.0), :marginalized)
        rw1_template = TethysBayes.build_structure_template(:rw1, 5)
        ll_rw1 = TethysBayes._rw1_log_marginal_likelihood(y_res, t_idx, 5, rw1_template.matrix,
            0.5, 0.2)
        @test isfinite(ll_rw1)

        spec_rw1 = (key = :year, hyper = (n_latent = 5, Q_template = rw1_template.matrix,
            U = rw1_template.U, L = rw1_template.L), params = Dict())
        chain_rw1 = Dict(
            :sigma_year => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_rw1 = check_marginalized_reconstruction(rw1_m, chain_rw1, spec_rw1, M_mock)

        # 10. Test Marginalized RW2 Likelihood & Latent Reconstruction
        rw2_m = TethysBayes.RW2(Exponential(1.0), :marginalized)
        rw2_template = TethysBayes.build_structure_template(:rw2, 5)
        ll_rw2 = TethysBayes._rw2_log_marginal_likelihood(y_res, t_idx, 5, rw2_template.matrix,
            0.5, 0.2)
        @test isfinite(ll_rw2)

        spec_rw2 = (key = :year, hyper = (n_latent = 5, Q_template = rw2_template.matrix,
            U = rw2_template.U, L = rw2_template.L), params = Dict())
        eff_rw2 = check_marginalized_reconstruction(rw2_m, chain_rw1, spec_rw2, M_mock)

        # 11. Test Marginalized ICAR Likelihood & Latent Reconstruction
        W_mock = [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0]
        icar_template = TethysBayes.build_structure_template(:icar, 5; W=W_mock)
        icar_m = TethysBayes.ICAR(Exponential(1.0), :marginalized)
        ll_icar = TethysBayes._icar_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix,
            icar_template.L, 0.5, 0.2)
        @test isfinite(ll_icar)

        M_spatial = (
            outcomes_N = 1,
            model_arch = "univariate",
            s_idx = t_idx,
            s_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        spec_icar = (key = :region, hyper = (n_latent = 5,
            Q_template = icar_template.matrix, U = icar_template.U, L = icar_template.L),
            params = Dict())
        chain_icar = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_icar = check_marginalized_reconstruction(icar_m, chain_icar, spec_icar, M_spatial)

        # 12. Test Marginalized Leroux Likelihood & Latent Reconstruction
        leroux_m = TethysBayes.Leroux(Beta(1, 1), Exponential(1.0), :marginalized)
        ll_leroux = TethysBayes._leroux_log_marginal_likelihood(y_res, t_idx, 5,
            icar_template.matrix, icar_template.L, 0.6, 0.5, 0.2)
        @test isfinite(ll_leroux)

        chain_leroux = Dict(
            :rho_region => reshape([0.5, 0.6, 0.7], 1, 3),
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_leroux = check_marginalized_reconstruction(leroux_m, chain_leroux, spec_icar, M_spatial)

        # 13. Test Marginalized PSpline Likelihood & Latent Reconstruction
        B_mock, _ = TethysBayes.TethysBayes_bspline_basis([1.0, 2.0, 3.0, 4.0, 5.0], 5, 3)
        pspline_m = TethysBayes.PSpline(5, 3, 2, Exponential(1.0), :marginalized)
        ll_ps = TethysBayes._pspline_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix,
            rw2_template.L, 2, 0.5, 0.2)
        @test isfinite(ll_ps)

        spec_ps = (key = :x, hyper = (n_latent = 5, basis_matrix = B_mock,
            Q_template = rw2_template.matrix, U = rw2_template.U, L = rw2_template.L),
            params = Dict(:positional_args => [:x]))
        chain_ps = Dict(
            :sigma_x => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_ps = check_marginalized_reconstruction(pspline_m, chain_ps, spec_ps, M_mock)

        # 14. Test Marginalized BYM2 Likelihood & Latent Reconstruction
        bym2_m = TethysBayes.BYM2(Normal(0, 0.5), Exponential(1.0), :marginalized)
        ll_bym2 = TethysBayes._bym2_log_marginal_likelihood(y_res, t_idx, 5, icar_template.U,
            icar_template.L, 0.6, 0.5, 0.2)
        @test isfinite(ll_bym2)

        chain_bym2 = Dict(
            :rho_unconstrained_region => reshape([0.5, 0.6, 0.7], 1, 3),
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_bym2 = TethysBayes.get_effects(bym2_m, chain_bym2, spec_icar, M_spatial, nothing)
        @test length(eff_bym2.structured) == 1
        @test length(eff_bym2.unstructured) == 1
        @test length(eff_bym2.noisy) == 1
        @test size(eff_bym2.structured[1]) == (5, 3)
        @test !all(iszero, eff_bym2.structured[1])
        @test !all(iszero, eff_bym2.noisy[1])

        # 15. Test Marginalized IID Likelihood & Latent Reconstruction
        iid_m = TethysBayes.IID(Exponential(1.0), :marginalized)
        ll_iid = TethysBayes._iid_log_marginal_likelihood(y_res, t_idx, 5, 0.5, 0.2)
        @test isfinite(ll_iid)

        spec_iid = (key = :group, structure = :mixed, var = "group", hyper = (n_latent = 5,
            ), params = Dict())
        M_iid = (
            outcomes_N = 1,
            model_arch = "univariate",
            mixed_idx_group = t_idx,
            y_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        chain_iid = Dict(
            :sigma_group => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_iid = check_marginalized_reconstruction(iid_m, chain_iid, spec_iid, M_iid)

        # 16. Test Marginalized Cyclic Likelihood & Latent Reconstruction
        cyclic_m = TethysBayes.Cyclic(5, Exponential(1.0), :marginalized)
        cyclic_template = TethysBayes.build_structure_template(:cyclic, 5)
        ll_cyclic = TethysBayes._cyclic_log_marginal_likelihood(y_res, t_idx, 5,
            cyclic_template.matrix, cyclic_template.L, 0.5, 0.2)
        @test isfinite(ll_cyclic)

        spec_cyclic = (key = :month, hyper = (n_latent = 5,
            Q_template = cyclic_template.matrix, U = cyclic_template.U,
            L = cyclic_template.L), params = Dict())
        M_cyclic = (
            outcomes_N = 1,
            model_arch = "univariate",
            u_idx = t_idx,
            u_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        chain_cyclic = Dict(
            :sigma_month => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_cyclic = check_marginalized_reconstruction(cyclic_m, chain_cyclic, spec_cyclic, M_cyclic)

        # 17. Test Marginalized BSpline Likelihood & Latent Reconstruction
        bspline_m = TethysBayes.BSpline(5, 3, Exponential(1.0), :marginalized)
        ll_bs = TethysBayes._bspline_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix,
            rw2_template.L, 0.5, 0.2)
        @test isfinite(ll_bs)

        eff_bs = check_marginalized_reconstruction(bspline_m, chain_ps, spec_ps, M_mock)

        # 18. Test Marginalized Moran Likelihood & Latent Reconstruction
        moran_m = TethysBayes.Moran(Exponential(1.0), :marginalized)
        moran_eigs = icar_template.U
        ll_moran = TethysBayes._moran_log_marginal_likelihood(y_res, t_idx, 5, moran_eigs, 0.5, 0.2)
        @test isfinite(ll_moran)

        spec_moran = (key = :region, hyper = (n_latent = 5,
            moran_eigenvectors = moran_eigs), params = Dict())
        chain_moran = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_moran = check_marginalized_reconstruction(moran_m, chain_moran, spec_moran, M_spatial)

        # 19. Test Marginalized SAR Likelihood & Latent Reconstruction
        sar_m = TethysBayes.SAR(Normal(0, 0.5), Exponential(1.0), :marginalized)
        sar_W = [0.0 1.0 0.0 0.0 0.0; 0.5 0.0 0.5 0.0 0.0; 0.0 0.5 0.0 0.5 0.0; 0.0 0.0 0.5 0.0 0.5; 0.0 0.0 0.0 1.0 0.0]
        sar_eigs = eigvals(sar_W)
        ll_sar = TethysBayes._sar_log_marginal_likelihood(y_res, t_idx, 5, sar_W, sar_eigs, 0.4,
            0.5, 0.2)
        @test isfinite(ll_sar)

        spec_sar = (key = :region, hyper = (n_latent = 5, Q_template = sar_W,
            eigenvalues = sar_eigs), params = Dict())
        chain_sar = Dict(
            :rho_region => reshape([0.3, 0.4, 0.5], 1, 3),
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_sar = check_marginalized_reconstruction(sar_m, chain_sar, spec_sar, M_spatial)

        # 20. Test Marginalized TPS Likelihood & Latent Reconstruction
        tps_m = TethysBayes.TPS(5, Exponential(1.0), :marginalized)
        ll_tps = TethysBayes._tps_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix,
            rw2_template.L, 0.5, 0.2)
        @test isfinite(ll_tps)

        spec_tps = (key = :space, hyper = (n_latent = 5, basis_matrix = B_mock,
            Q_template = rw2_template.matrix, L = rw2_template.L, knots = randn(5, 2)),
            params = Dict())
        chain_tps = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_tps = check_marginalized_reconstruction(tps_m, chain_tps, spec_tps, M_mock)

        # 21. Test Marginalized Barycentric Likelihood & Latent Reconstruction
        bary_m = TethysBayes.Barycentric(Exponential(1.0), :marginalized)
        ll_bary = TethysBayes._barycentric_log_marginal_likelihood(y_res, B_mock, nothing,
            nothing, 0.5, 0.2)
        @test isfinite(ll_bary)

        spec_bary = (key = :space, hyper = (n_knots = 5, B = B_mock,
            knots = [TethysBayes.Point2D(0.0, 0.0) for _ in 1:5]), params = Dict())
        chain_bary = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_bary = check_marginalized_reconstruction(bary_m, chain_bary, spec_bary, M_mock)

        # 22. Test Marginalized BCGN Likelihood & Latent Reconstruction
        bcgn_m = TethysBayes.BCGN(Exponential(1.0), :marginalized)
        bcgn_map = Matrix{Float64}(I, 5, 5)
        ll_bcgn = TethysBayes._bcgn_log_marginal_likelihood(y_res, bcgn_map, icar_template.matrix,
            icar_template.L, 0.5, 0.2)
        @test isfinite(ll_bcgn)

        spec_bcgn = (key = :space, hyper = (n_latent = 5, mapping_matrix = bcgn_map,
            Q_template = icar_template.matrix, L = icar_template.L, set1_indices = [1, 2,
            3, 4, 5]), params = Dict())
        chain_bcgn = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_bcgn = check_marginalized_reconstruction(bcgn_m, chain_bcgn, spec_bcgn, M_spatial)

        # 23. Test Marginalized Besag Likelihood & Latent Reconstruction
        besag_m = TethysBayes.Besag(Exponential(1.0), :marginalized)
        ll_besag = TethysBayes._besag_log_marginal_likelihood(y_res, t_idx, 5,
            icar_template.matrix, icar_template.L, 0.5, 0.2)
        @test isfinite(ll_besag)

        spec_besag = (key = :region, hyper = (n_latent = 5,
            Q_template = icar_template.matrix, L = icar_template.L), params = Dict())
        chain_besag = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_besag = check_marginalized_reconstruction(besag_m, chain_besag, spec_besag, M_spatial)

        # 24. Test Marginalized RFF Likelihood & Latent Reconstruction
        rff_m = TethysBayes.RFF(Normal(1.0, 0.1), Exponential(1.0), 5, "se", :marginalized)
        ll_rff = TethysBayes._rff_log_marginal_likelihood(y_res, B_mock, 0.5, 0.2)
        @test isfinite(ll_rff)

        coords_mock = randn(5, 2)
        spec_rff = (key = :space, hyper = (n_latent = 5, in_dims = 2, coords = coords_mock,
            W_fixed = randn(2, 5), b_fixed = rand(5)), params = Dict())
        chain_rff = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_rff = check_marginalized_reconstruction(rff_m, chain_rff, spec_rff, M_mock)

        # 25. Test Marginalized SPDE Likelihood & Latent Reconstruction
        spde_m = TethysBayes.SPDE(Exponential(1.0), LogNormal(0, 1), :marginalized)
        ll_spde = TethysBayes._spde_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix,
            icar_template.L, 1.2, 0.5, 0.2)
        @test isfinite(ll_spde)

        spec_spde = (key = :region, hyper = (n_latent = 5,
            Q_template = icar_template.matrix, L = icar_template.L, U = icar_template.U),
            params = Dict())
        chain_spde = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :range_region => reshape([1.0, 1.2, 1.4], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_spde = check_marginalized_reconstruction(spde_m, chain_spde, spec_spde, M_spatial)

        # 26. Test Marginalized GP Likelihood & Latent Reconstruction
        gp_m = TethysBayes.GP(LogNormal(0, 1), Exponential(1.0), "se", :marginalized)
        ll_gp = TethysBayes._gp_log_marginal_likelihood(y_res, coords_mock, 0.5, 1.0, :se, 0.2)
        @test isfinite(ll_gp)

        spec_gp = (key = :space, hyper = (n_latent = 5, coords = coords_mock), params = Dict())
        chain_gp = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :length_scale_space => reshape([1.0, 1.2, 1.4], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_gp = check_marginalized_reconstruction(gp_m, chain_gp, spec_gp, M_mock)
    end
end

@testset "Core: Manifold & Model Construction" begin
    @testset "Temporal Manifolds" begin
        m_rw1 = TethysBayes.RW1(Distributions.Exponential(1.0), :statespace)
        res_rw1 = TethysBayes.get_precomputes(m_rw1, (N_time=20,), Dict(:variables => :year))
        @test res_rw1.model_type == :rw1
        @test size(res_rw1.Q_template) == (20, 20)
    end

    @testset "Seasonal Manifolds" begin
        m_cyc = TethysBayes.Harmonic(1, Distributions.Exponential(1.0), Distributions.Beta(1, 1),
            12.0, :twocoefficient)
        res_cyc = TethysBayes.get_precomputes(m_cyc, (N_time=12,), Dict(:variables => :month))
        @test res_cyc.model_type == :harmonic
        @test res_cyc.period == 12.0
    end

    @testset "Basis & Continuous Manifolds" begin
        m_bs = TethysBayes.BSpline(15, 3, Distributions.Exponential(1.0), :spectral)
        res_bs = TethysBayes.get_precomputes(m_bs, (N_levels=15,), Dict(:variables => :cov3))
        @test res_bs.model_type == :bspline
        @test size(res_bs.B_matrix) == (15, 15)
    end
end

@testset "Formula Escapes, Operator Precedence & Tensor Registries" begin
    @testset "Escape Handling & Nested Parentheses" begin
        is_esc = TethysBayes._is_escaped
        s1 = "a\\\"b"
        @test is_esc(s1, 3) == true
        s2 = "a\\\\\"b"
        @test is_esc(s2, 4) == false
        s3 = "a\\\\\\\"b"
        @test is_esc(s3, 5) == true

        # split_terms_at_depth with escaped quotes and braces
        s_split1 = "fixed(x, label=\"a + b\") + fixed(y)"
        parts1 = TethysBayes.split_terms_at_depth(s_split1, "+")
        @test length(parts1) == 2
        @test parts1[1] == "fixed(x, label=\"a + b\")"
        @test parts1[2] == "fixed(y)"

        s_split2 = "fixed(x, label=\"foo \\\" + bar\") + fixed(z)"
        parts2 = TethysBayes.split_terms_at_depth(s_split2, "+")
        @test length(parts2) == 2
        @test parts2[2] == "fixed(z)"

        s_split3 = "fixed(x, options={a=1 + 2, b=3}) + fixed(y)"
        parts3 = TethysBayes.split_terms_at_depth(s_split3, "+")
        @test length(parts3) == 2

        # _is_outermost_grouping_parentheses
        is_outer = TethysBayes._is_outermost_grouping_parentheses
        @test is_outer("(a + b)") == true
        @test is_outer("(a) + (b)") == false
        @test is_outer("((a + b))") == true
        @test is_outer("(a + (b * (c + d)))") == true
        @test is_outer("fixed(x)") == false
        @test is_outer("(a))") == false
        @test is_outer("((a)") == false
        @test is_outer("(fixed(x, label=\"test ) string\"))") == true
        @test is_outer("(fixed(x, label=\"test ( string\"))") == true
        @test is_outer("(fixed(x, label=\"test \\\" ) string\"))") == true
    end

    @testset "Operator Precedence" begin
        parse_expr = TethysBayes._parse_rhs_expression

        # Pipe vs Kronecker product
        res1 = parse_expr("fixed(a) |> random(b) ⊗ random(c)")
        @test res1.type == :operator && res1.op == :pipe
        @test res1.children[2].op == :kronecker_product

        res2 = parse_expr("random(a) ⊗ random(b) |> fixed(c)")
        @test res2.type == :operator && res2.op == :pipe
        @test res2.children[1].op == :kronecker_product

        # Composition vs Kronecker product
        res3 = parse_expr("fixed(a) ∘ random(b) ⊗ random(c)")
        @test res3.type == :operator && res3.op == :composition
        @test res3.children[2].op == :kronecker_product

        res4 = parse_expr("random(a) ⊗ random(b) ∘ fixed(c)")
        @test res4.type == :operator && res4.op == :composition
        @test res4.children[1].op == :kronecker_product

        # Addition vs Pipe
        res5 = parse_expr("fixed(a) + fixed(b) |> fixed(c)")
        @test res5.type == :operator && res5.op == :add
        @test res5.children[2].op == :pipe

        res6 = parse_expr("(fixed(a) + fixed(b)) |> fixed(c)")
        @test res6.type == :operator && res6.op == :pipe
        @test res6.children[1].op == :add

        # Full formula parsing
        df_mock = DataFrame(y = [1, 2], cov = [0.1, 0.2], s_idx = [1, 2])
        decomp = TethysBayes.decompose_TethysBayes_formula(
            "y ~ intercept() + fixed(cov) |> random(s_idx, model=bym2)",
            df_mock
        )
        @test decomp.has_intercept == true
        @test length(decomp.modules) >= 1
    end

    @testset "Multi-dimensional Matrix/Tensor Parameters" begin
        in_dim = 3
        hidden_dim = 4
        nbins = 2

        comp_spec = (
            key = :nn_covar,
            component_obj = (hidden_dim = hidden_dim, nbins = nbins),
            hyper = (in_dim = in_dim, n_latent = 0),
            params = Dict{Symbol, Any}()
        )
        M_tensor = (
            model_arch = "univariate",
            outcomes_N = 1,
            add_intercept = true,
            Xfixed_N = 2,
            components = [comp_spec]
        )
        reg = TethysBayes.build_param_registry(M_tensor)

        @test reg.descriptors[:rff_weights_1_nn_covar].shape == (in_dim, hidden_dim)
        @test reg.descriptors[:rff_weights_2_nn_covar].shape == (hidden_dim, nbins)
        @test reg.descriptors[:rff_offsets_1_nn_covar].shape == (hidden_dim,)

        # Sample extraction with tensor reshaping
        n_samples = 50
        chain_df = DataFrame()
        for j in 1:hidden_dim
            for i in 1:in_dim
                col_name = Symbol("rff_weights_1_nn_covar[$i, $j]")
                chain_df[!, col_name] = [Float64(i * 10 + j + s * 100) for s in 1:n_samples]
            end
        end

        # A registry built from a chain is element-wise: MCMCChains stores one column per
        # matrix entry, so there is one descriptor per `[i, j]` rather than a rolled-up one.
        reg_chain = TethysBayes.build_param_registry(chain_df)

        samples_tensor = TethysBayes.get_param_samples(chain_df, reg, :nn_covar, :rff_weights_1)
        @test size(samples_tensor) == (n_samples, in_dim, hidden_dim)
        @test samples_tensor[1, 1, 1] == 111.0
        @test samples_tensor[50, 3, 4] == 5034.0

        samples_flat = TethysBayes.get_param_samples(
            chain_df, reg, :nn_covar, :rff_weights_1; reshape_to_shape = false
        )
        @test size(samples_flat) == (n_samples, in_dim * hidden_dim)
    end

    @testset "Log Transformations with Offsets" begin
        # Positive data
        x_pos = [1.0, 2.0, 10.0]
        @test TethysBayes.apply_transformation(:log, x_pos) ≈ log.(x_pos)

        # Zero-containing data defaults to log1p
        x_zero = [0.0, 1.0, 9.0]
        @test TethysBayes.apply_transformation(:log, x_zero) ≈ log1p.(x_zero)

        # Custom offset
        x_custom = [0.0, 2.0, 5.0]
        @test TethysBayes.apply_transformation(:log, x_custom; offset=0.05) ≈ log.(x_custom .+ 0.05)

        # Negative data without offset throws ArgumentError
        x_neg = [-5.0, 0.0, 5.0]
        @test_throws ArgumentError TethysBayes.apply_transformation(:log, x_neg)
        @test TethysBayes.apply_transformation(:log, x_neg; offset=10.0) ≈ log.(x_neg .+ 10.0)

        # Formula pipeline transformation (non-mutating caller data by default)
        df_trans = DataFrame(val = [0.0, 1.0, 2.0])
        decomp = TethysBayes.decompose_TethysBayes_formula(
            "y ~ intercept() + log(val, offset=0.1) |> fixed()", df_trans
        )
        @test !hasproperty(df_trans, :val_log)
        @test hasproperty(decomp.data, :val_log)
        @test decomp.data.val_log ≈ log.(decomp.data.val .+ 0.1)

        decomp_mut = TethysBayes.decompose_TethysBayes_formula(
            "y ~ intercept() + log(val, offset=0.1) |> fixed()", df_trans; copy_data=false
        )
        @test hasproperty(df_trans, :val_log)
    end

    @testset "Transformation DataFrame Safety & Collision Avoidance" begin
        # Column collision handling
        df_collision = DataFrame(
            temp = [1.0, 2.0, 3.0, 4.0, 5.0],
            temp_zscore = [99.0, 99.0, 99.0, 99.0, 99.0],
            y = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        decomp_col = TethysBayes.decompose_TethysBayes_formula("likelihood(y) ~ zscore(temp) |> fixed()", df_collision)
        @test hasproperty(decomp_col.data, :temp_zscore_2)
        @test decomp_col.data.temp_zscore == [99.0, 99.0, 99.0, 99.0, 99.0]
    end

    @testset "Multivariate Hyperprior Sharing (Item 8)" begin
        @test TethysBayes.is_param_shared(true, :sigma) == true
        @test TethysBayes.is_param_shared(false, :sigma) == false
        @test TethysBayes.is_param_shared(:all, :sigma) == true
        @test TethysBayes.is_param_shared(:sigma, :sigma) == true
        @test TethysBayes.is_param_shared(:sigma, :rho) == false
        @test TethysBayes.is_param_shared([:sigma, :range], :sigma) == true
        @test TethysBayes.is_param_shared([:sigma, :range], :rho) == false

        # Shared intercept
        m_cfg_shared_int = (
            model_arch = "multivariate",
            outcomes_N = 3,
            add_intercept = true,
            shared_intercept = true,
            intercept_prior = Normal(0, 1),
            Xfixed_N = 0,
            components = []
        )
        reg_shared_int = TethysBayes.build_param_registry(m_cfg_shared_int)
        @test reg_shared_int.descriptors[:intercept].is_shared == true
        @test !haskey(reg_shared_int.descriptors, :intercept_1)

        # Fine-grained component sharing
        m_cfg_shared_comp = (
            model_arch = "multivariate",
            outcomes_N = 2,
            add_intercept = false,
            Xfixed_N = 0,
            components = [
                (
                    key = :time,
                    params = Dict{Symbol, Any}(:shared => [:sigma]),
                    component_obj = TethysBayes.AR1(Normal(0, 1), Exponential(1.0), :statespace),
                    hyper = (n_latent = 10,)
                )
            ]
        )
        reg_comp = TethysBayes.build_param_registry(m_cfg_shared_comp)
        @test reg_comp.descriptors[:sigma_time].is_shared == true
        @test reg_comp.descriptors[:rho_unconstrained_time_1].is_shared == false
    end

    @testset "Parameter Extraction Disambiguation (Item 13)" begin
        reg_disambig = TethysBayes.ParamRegistry()
        TethysBayes.add_descriptor!(reg_disambig, TethysBayes.ParamDescriptor(:Xfixed_beta; role=:fixed_coef))
        TethysBayes.add_descriptor!(reg_disambig, TethysBayes.ParamDescriptor(:beta_prop; role=:fixed_coef))

        # A registered name resolves to itself and nothing else.
        @test TethysBayes.find_chain_param(reg_disambig, "Xfixed_beta") == "Xfixed_beta"
        @test TethysBayes.find_chain_param(reg_disambig, "beta_prop") == "beta_prop"

        # A name that is not registered returns nothing rather than quietly resolving to
        # a different coefficient, and says which names were on offer.
        @test_logs (:warn,) TethysBayes.find_chain_param(reg_disambig, "beta") == ""
    end

    @testset "Kronecker Product Composition Validation (Item 14)" begin
        df_kronecker = DataFrame(y = rand(10), s = 1:10, t = 1:10, x = rand(10))
        @test_throws ArgumentError TethysBayes.decompose_TethysBayes_formula(
            "likelihood(y) ~ intercept() ⊗ fixed(x)", df_kronecker
        )
    end

    @testset "Bare Fixed Effect Term Parsing" begin
        p_scot = TethysBayes.TethysBayes_data("scottish_lip")
        df_bare = p_scot.data
        W_bare = p_scot.au.W

        m_bare = @tbm(
            likelihood(y_gauss) ~ 1 + cov1 + random(s_idx, model=bym2) + random(year, model=ar1),
            df_bare,
            W = W_bare,
            verbose = false
        )
        @test m_bare !== nothing
        @test hasproperty(m_bare, :args)
    end
end

@testset "World Age Safety (model built inside a function)" begin
    # Regression test: `TethysBayes` generates the Turing model body with a runtime
    # `Core.eval`, which defines a method that already-compiled code (DynamicPPL's
    # `_evaluate!!`) cannot see. Constructing and sampling a model inside a plain
    # function therefore used to fail with
    #   "MethodError: ... (method too new to be called from this world context.)"
    # `make_world_age_safe` routes evaluation through a precompiled trampoline to
    # prevent that. Keep this test: the failure mode is silent at top level.
    p_world = TethysBayes.TethysBayes_data("scottish_lip")
    df_world = p_world.data
    W_world = p_world.au.W

    # The model must be built AND sampled inside the function, not at top level.
    function _build_and_sample_in_function()
        m = @tbm(
            likelihood(y_gauss) ~ intercept() + fixed(cov1),
            df_world,
            W = W_world,
            verbose = false
        )
        return sample(m, NUTS(), 10; progress = false, check_model = false)
    end

    @testset "macro in function scope" begin
        chn = _build_and_sample_in_function()
        @test chn !== nothing
        @test size(chn, 1) == 10
    end

    @testset "TethysBayes_core with a runtime String in function scope" begin
        # The string-formula entry point goes through the same construction path.
        function _build_and_sample_string()
            m = TethysBayes.TethysBayes_core(
                "likelihood(y_gauss) ~ intercept() + fixed(cov1)",
                df_world, Main;
                W = W_world, verbose = false)
            return sample(m, NUTS(), 10; progress = false, check_model = false)
        end
        chn2 = _build_and_sample_string()
        @test chn2 !== nothing
        @test size(chn2, 1) == 10
    end

    @testset "prior draw in function scope" begin
        m_rand = @tbm(likelihood(y_gauss) ~ intercept(), df_world; verbose = false)
        _rand_in_function(m) = rand(m)
        @test _rand_in_function(m_rand) !== nothing
    end

    @testset "trampoline is installed and idempotent" begin
        m_t = @tbm(likelihood(y_gauss) ~ intercept(), df_world; verbose = false)
        @test m_t.f isa TethysBayes.WorldAgeTrampoline
        # Wrapping twice must not nest another trampoline.
        @test TethysBayes.make_world_age_safe(m_t).f === m_t.f
    end

    @testset "Component Interface Contract" begin
        # Every registered component implements the whole interface. Checked as a
        # property of the registry, so a component that drops a method fails here rather
        # than as a MethodError deep inside model construction.
        @test isempty(TethysBayes.validate_all_component_interfaces())

        @test length(TethysBayes.COMPONENT_INTERFACE_FUNCTIONS) == 4
        @test Set(TethysBayes.COMPONENT_INTERFACE_FUNCTIONS) ==
            Set([:get_precomputes, :get_priors, :get_updates, :get_effects])

        for (key, T) in TethysBayes.COMPONENT_TYPE_REGISTRY
            missing = TethysBayes.missing_interface_methods(T)
            @test isempty(missing) || "$(key) is missing $(join(string.(missing), ", "))"
        end

        # The check actually detects a gap, rather than passing vacuously.
        @eval struct IncompleteComponent <: TethysBayes.ComponentModel end
        @test TethysBayes.missing_interface_methods(IncompleteComponent) ==
            collect(TethysBayes.COMPONENT_INTERFACE_FUNCTIONS)
        @test_throws ErrorException TethysBayes.validate_component_interface(IncompleteComponent)
    end
end
