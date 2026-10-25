# ==============================================================================
# TethysBayes Test Suite: Nested Multi-Fidelity Models & Sub-Model Linkage Architecture
# ==============================================================================
# Verifies the nested() module interface, recursive configuration isolation,
# parameter registry compilation, Turing model AST linear predictor coupling,
# dimension mismatch validation, observation mapping, and posterior reconstruction.
#
# Mathematical Formulation:
#   Main model:
#     y_hi ~ Likelihood(eta_hi, ...)
#     eta_hi = X_hi * beta_hi + sum_k(f_k(s, t)) + rho_sub * eta_sub[mapping]
#   Sub-model:
#     y_lo ~ Likelihood(eta_sub, ...)
#     eta_sub = X_lo * beta_sub + sum_j(g_j(s, t))
#     rho_sub ~ Normal(1.0, 0.5)
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Nested Architecture & Multi-Fidelity Sub-Models" begin
    Random.seed!(42)
    n_hi = 20
    n_lo = 20

    data_hi = DataFrame(
        y_hi = randn(n_hi),
        x = randn(n_hi),
        s_idx = rand(1:5, n_hi),
        t_idx = rand(1:4, n_hi)
    )

    data_lo = DataFrame(
        y_lo = randn(n_lo),
        x = randn(n_lo),
        s_idx = rand(1:5, n_lo),
        t_idx = rand(1:4, n_lo)
    )

    @testset "Sub-Config Keyword Isolation & Assembly" begin
        cfg = TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + fixed(x) + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1 + fixed(x)\", data_source=:low_data)",
            data_hi;
            low_data = data_lo
        )

        @test haskey(cfg, :nested_components)
        @test haskey(cfg.nested_components, :proxy)
        
        sub_cfg = cfg.nested_components[:proxy]
        @test sub_cfg.y_N == n_lo
        @test haskey(sub_cfg, :Xfixed)
        @test size(sub_cfg.Xfixed, 1) == n_lo
        # Ensure parent's data and observation arrays were not leaked
        @test sub_cfg.y_obs == data_lo.y_lo
        @test !haskey(sub_cfg, :nested_components)
    end

    @testset "Dimension Alignment & Explicit Observation Mapping" begin
        # 1. Observation count mismatch without mapping must raise ArgumentError
        data_lo_mismatch = DataFrame(
            y_lo = randn(35),
            x = randn(35)
        )

        @test_throws ArgumentError TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1\", data_source=:low_data)",
            data_hi;
            low_data = data_lo_mismatch
        )

        # 2. Valid mapping via column symbol
        data_hi_mapped = copy(data_hi)
        data_hi_mapped[!, :lo_idx] = rand(1:35, n_hi)

        cfg_col_map = TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1\", " *
            "data_source=:low_data, mapping=:lo_idx)",
            data_hi_mapped;
            low_data = data_lo_mismatch
        )
        @test haskey(cfg_col_map.nested_components[:proxy], :mapping)
        @test length(cfg_col_map.nested_components[:proxy].mapping) == n_hi
        @test cfg_col_map.nested_components[:proxy].mapping == data_hi_mapped.lo_idx

        # 3. Valid mapping via explicit vector
        explicit_idx = rand(1:35, n_hi)
        cfg_vec_map = TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1\", " *
            "data_source=:low_data, mapping=explicit_idx)",
            data_hi;
            low_data = data_lo_mismatch,
            explicit_idx = explicit_idx
        )
        @test cfg_vec_map.nested_components[:proxy].mapping == explicit_idx

        # 4. Out-of-bounds mapping index must raise BoundsError
        invalid_bounds_idx = copy(explicit_idx)
        invalid_bounds_idx[1] = 999  # Exceeds sub_N = 35
        @test_throws BoundsError TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1\", " *
            "data_source=:low_data, mapping=invalid_bounds_idx)",
            data_hi;
            low_data = data_lo_mismatch,
            invalid_bounds_idx = invalid_bounds_idx
        )

        # 5. Mapping length mismatch must raise DimensionMismatch
        short_idx = explicit_idx[1:10]
        @test_throws DimensionMismatch TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1\", " *
            "data_source=:low_data, mapping=short_idx)",
            data_hi;
            low_data = data_lo_mismatch,
            short_idx = short_idx
        )
    end

    @testset "ParamRegistry Integration & Prefixed Namespaces" begin
        cfg = TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + fixed(x) + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1 + fixed(x)\", data_source=:low_data)",
            data_hi;
            low_data = data_lo
        )
        reg = TethysBayes.build_param_registry(cfg)

        # Main model descriptors
        @test haskey(reg.descriptors, :intercept)
        @test haskey(reg.descriptors, :beta)
        @test haskey(reg.descriptors, :y_sigma)

        # Nested coupling weight
        @test haskey(reg.descriptors, :rho_nested_proxy)
        rho_desc = reg.descriptors[:rho_nested_proxy]
        @test rho_desc.role == :nested_weight
        @test rho_desc.shape == (1,)
        @test rho_desc.prior == Normal(1.0, 0.5)

        # Prefixed sub-model descriptors
        @test haskey(reg.descriptors, :intercept_proxy)
        @test haskey(reg.descriptors, :beta_proxy)
        @test haskey(reg.descriptors, :y_sigma_proxy)

        # Role-based query
        nested_weights = TethysBayes.get_descriptors_by_role(reg, :nested_weight)
        @test length(nested_weights) == 1
        @test nested_weights[1].symbol == :rho_nested_proxy
    end

    @testset "Turing Model Code Generation & Predictor Coupling" begin
        cfg = TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + fixed(x) + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1 + fixed(x)\", data_source=:low_data)",
            data_hi;
            low_data = data_lo
        )

        code_str, _, _ = TethysBayes.TethysBayes_text_assembler(cfg, :test_nested_model)

        # Sub-model weight and priors.
        # NOTE: these are emitted as plain `name ~ dist` statements; the legacy
        # `DynamicPPL.NamedDist(dist, :name)` wrapper was removed from DynamicPPL
        # and the left-hand side already supplies the variable name.
        @test occursin("rho_nested_proxy ~ Normal(1.0, 0.5)", code_str)
        @test !occursin("NamedDist", code_str)
        @test occursin("intercept_proxy ~", code_str)
        @test occursin("beta_proxy ~", code_str)
        @test occursin("y_sigma_proxy ~", code_str)

        # Sub-model fixed effects linear predictor update
        @test occursin("eta_sub_proxy = eta_sub_proxy .+ sub_M_proxy.Xfixed", code_str)

        # Sub-model likelihood evaluated on sub_M_proxy.y_obs
        @test occursin("Distributions.logpdf.(d_lik_vec, sub_M_proxy.y_obs)", code_str)

        # Linear predictor coupling into main eta
        @test occursin("eta = eta .+ rho_nested_proxy .* eta_sub_proxy", code_str)
    end

    @testset "End-to-End Prior Sampling & Posterior Reconstruction" begin
        model = @tbm(
            likelihood(y_hi) ~ 1 + fixed(x) +
            nested(proxy, formula="likelihood(y_lo) ~ 1 + fixed(x)", data_source=:low_data),
            data_hi,
            low_data = data_lo
        )

        n_draws = 10
        chain = sample(model, Prior(), n_draws, progress=false)
        @test size(chain, 1) == n_draws

        # Parameter samples extraction
        rho_samples = TethysBayes.get_samples(chain, :rho_nested_proxy)
        @test length(rho_samples) == n_draws
        @test all(isfinite, rho_samples)

        beta_proxy_samples = TethysBayes.get_samples(chain, :beta_proxy)
        @test length(beta_proxy_samples) == n_draws

        # Full reconstruction
        cfg = TethysBayes.TethysBayes_config(
            "likelihood(y_hi) ~ 1 + fixed(x) + " *
            "nested(proxy, formula=\"likelihood(y_lo) ~ 1 + fixed(x)\", data_source=:low_data)",
            data_hi;
            low_data = data_lo
        )
        res = TethysBayes.reconstruct(chain, cfg)

        @test haskey(res, :predictions_denoised)
        @test haskey(res, :predictions_noisy)
        @test haskey(res, :nested_results)
        @test haskey(res.nested_results, :proxy)
        @test length(res.predictions_denoised.mean) == n_hi
        @test all(isfinite, res.predictions_denoised.mean)
    end

    @testset "Alternative 2: Declarative Paired Multi-Equation Syntax" begin
        # 1. TethysBayes_config with paired specifications
        cfg_paired = TethysBayes.TethysBayes_config(
            :primary => (
                formula = "likelihood(y_hi) ~ 1 + fixed(x) + transfer(:proxy)",
                data    = data_hi
            ),
            :proxy => (
                formula = "likelihood(y_lo) ~ 1 + fixed(x)",
                data    = data_lo,
                prior   = Normal(0.0, 1.0)
            )
        )
        @test haskey(cfg_paired, :nested_components)
        @test haskey(cfg_paired.nested_components, :proxy)
        @test cfg_paired.nested_components[:proxy].coupling_prior == Normal(0.0, 1.0)

        # 2. ParamRegistry reflects custom coupling prior
        reg_paired = TethysBayes.build_param_registry(cfg_paired)
        @test haskey(reg_paired.descriptors, :rho_nested_proxy)
        @test reg_paired.descriptors[:rho_nested_proxy].prior == Normal(0.0, 1.0)

        # 3. Model instantiation with @tbm macro using unquoted paired expressions
        model_paired = @tbm(
            :primary => (
                formula = likelihood(y_hi) ~ 1 + fixed(x) + transfer(:proxy),
                data    = data_hi
            ),
            :proxy => (
                formula = likelihood(y_lo) ~ 1 + fixed(x),
                data    = data_lo,
                prior   = Normal(1.0, 0.5)
            )
        )

        n_draws = 10
        chain_paired = sample(model_paired, Prior(), n_draws, progress=false)
        @test size(chain_paired, 1) == n_draws
        @test any(k -> occursin("rho_nested_proxy", string(k)), keys(chain_paired))
        @test any(k -> occursin("beta_proxy", string(k)), keys(chain_paired))

        # 4. Reconstruction with transfer_results
        res_paired = TethysBayes.reconstruct(chain_paired, cfg_paired)
        @test haskey(res_paired, :transfer_results)
        @test haskey(res_paired.transfer_results, :proxy)
        @test length(res_paired.predictions_denoised.mean) == n_hi

        # 5. Paired specification with observation mapping
        data_lo_30 = DataFrame(y_lo = randn(30), x = randn(30))
        data_hi_map = copy(data_hi)
        data_hi_map[!, :lo_idx] = rand(1:30, n_hi)

        cfg_map_paired = TethysBayes.TethysBayes_config(
            :primary => (
                formula = "likelihood(y_hi) ~ 1 + transfer(:proxy)",
                data    = data_hi_map
            ),
            :proxy => (
                formula = "likelihood(y_lo) ~ 1",
                data    = data_lo_30,
                mapping = :lo_idx
            )
        )
        @test cfg_map_paired.nested_components[:proxy].mapping == data_hi_map.lo_idx
    end

    @testset "Enhanced Multi-Fidelity Nesting & Coupling Architecture" begin
        n_tier = 25
        data_hi_enh = DataFrame(
            y_hi = randn(n_tier),
            x = randn(n_tier),
            region = rand([:north, :south, :east], n_tier),
            depth_stratum = rand([:shallow, :deep], n_tier),
            temp = randn(n_tier)
        )
        data_mid_enh = DataFrame(
            y_mid = randn(n_tier),
            x = randn(n_tier),
            region = rand([:north, :south, :east], n_tier),
            depth_stratum = rand([:shallow, :deep], n_tier),
            temp = randn(n_tier)
        )
        data_lo_enh = DataFrame(
            y_lo = randn(n_tier),
            x = randn(n_tier),
            region = rand([:north, :south, :east], n_tier),
            depth_stratum = rand([:shallow, :deep], n_tier),
            temp = randn(n_tier)
        )

        # 1. Multi-Strata Factorial Resolution
        res_single = TethysBayes._resolve_nested_strata(data_hi_enh, :region)
        @test res_single.n_strata == 3
        @test length(res_single.strata_levels) == 3
        @test length(res_single.strata_indices) == n_tier

        res_multi = TethysBayes._resolve_nested_strata(
            data_hi_enh, [:region, :depth_stratum]
        )
        @test res_multi.n_strata == 6
        @test length(res_multi.strata_levels) == 6
        @test haskey(res_multi, :factor_dims)
        @test res_multi.factor_dims[:region] == 3
        @test res_multi.factor_dims[:depth_stratum] == 2
        @test length(res_multi.factor_indices[:region]) == n_tier

        # 2. Coupling Modes & ParamRegistry
        cfg_mult = TethysBayes.TethysBayes_config(
            :primary => (
                formula = "likelihood(y_hi) ~ 1 + fixed(x) + transfer(:proxy)",
                data = data_hi_enh
            ),
            :proxy => (
                formula = "likelihood(y_lo) ~ 1 + fixed(x)",
                data = data_lo_enh,
                coupling = :multiplicative
            )
        )
        @test cfg_mult.nested_components[:proxy].coupling == :multiplicative

        cfg_mod = TethysBayes.TethysBayes_config(
            :primary => (
                formula = "likelihood(y_hi) ~ 1 + fixed(x) + transfer(:proxy)",
                data = data_hi_enh
            ),
            :proxy => (
                formula = "likelihood(y_lo) ~ 1 + fixed(x)",
                data = data_lo_enh,
                coupling = :moderated,
                moderator = :temp
            )
        )
        reg_mod = TethysBayes.build_param_registry(cfg_mod)
        @test haskey(reg_mod.descriptors, :rho_nested_proxy_0)
        @test haskey(reg_mod.descriptors, :rho_nested_proxy_1)

        cfg_tensor = TethysBayes.TethysBayes_config(
            :primary => (
                formula = "likelihood(y_hi) ~ 1 + fixed(x) + transfer(:proxy)",
                data = data_hi_enh
            ),
            :proxy => (
                formula = "likelihood(y_lo) ~ 1 + fixed(x)",
                data = data_lo_enh,
                coupling = :tensor,
                strata = [:region, :depth_stratum]
            )
        )
        reg_tensor = TethysBayes.build_param_registry(cfg_tensor)
        @test haskey(reg_tensor.descriptors, :rho_nested_proxy_region)
        @test haskey(reg_tensor.descriptors, :rho_nested_proxy_depth_stratum)
        @test reg_tensor.descriptors[:rho_nested_proxy_region].shape == (3,)
        @test reg_tensor.descriptors[:rho_nested_proxy_depth_stratum].shape == (2,)

        # 3. Hierarchical 3-Tier Cascading Assembly
        cfg_3tier = TethysBayes.TethysBayes_config(
            :primary => (
                formula = "likelihood(y_hi) ~ 1 + fixed(x) + transfer(:mid)",
                data = data_hi_enh
            ),
            :mid => (
                formula = "likelihood(y_mid) ~ 1 + fixed(x) + transfer(:coarse)",
                data = data_mid_enh
            ),
            :coarse => (
                formula = "likelihood(y_lo) ~ 1 + fixed(x)",
                data = data_lo_enh
            )
        )
        @test haskey(cfg_3tier.nested_components, :mid)
        @test haskey(cfg_3tier.nested_components[:mid].nested_components, :coarse)

        reg_3tier = TethysBayes.build_param_registry(cfg_3tier)
        @test haskey(reg_3tier.descriptors, :rho_nested_mid)
        @test haskey(reg_3tier.descriptors, :rho_nested_mid_coarse)
        @test haskey(reg_3tier.descriptors, :intercept_mid_coarse)

        code_3tier, _, _ = TethysBayes.TethysBayes_text_assembler(cfg_3tier, :model_3tier)
        @test occursin("rho_nested_mid ~", code_3tier)
        @test occursin("rho_nested_mid_coarse ~", code_3tier)
        @test occursin("eta_sub_mid_coarse", code_3tier)
        @test occursin("eta_sub_mid = eta_sub_mid .+ rho_nested_mid_coarse .* eta_sub_mid_coarse", code_3tier)
        @test occursin("eta = eta .+ rho_nested_mid .* eta_sub_mid", code_3tier)

        # 4. End-to-End Prior Sampling & Reconstruction
        model_3tier = @tbm(
            :primary => (
                formula = likelihood(y_hi) ~ 1 + fixed(x) + transfer(:mid),
                data = data_hi_enh
            ),
            :mid => (
                formula = likelihood(y_mid) ~ 1 + fixed(x) + transfer(:coarse),
                data = data_mid_enh
            ),
            :coarse => (
                formula = likelihood(y_lo) ~ 1 + fixed(x),
                data = data_lo_enh
            )
        )
        chain_3tier = sample(model_3tier, Prior(), 5, progress=false)
        res_3tier = TethysBayes.reconstruct(chain_3tier, cfg_3tier)
        @test all(isfinite, res_3tier.predictions_denoised.mean)
        @test haskey(res_3tier.transfer_results, :mid)
        @test haskey(res_3tier.transfer_results[:mid].transfer_results, :coarse)
    end
end

