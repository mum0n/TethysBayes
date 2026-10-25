# ==============================================================================
# TethysBayes Test Suite: ComponentModel Interface, NNGP, and MCAR
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "ComponentModel Interface Unit Tests" begin
    # Test IID Component
    @testset "IID Component" begin
        N_obs, N_levels = 100, 10
        m_iid = TethysBayes.IID(Distributions.Exponential(1.0), :noncentered)
        
        mock_M_ds = Dict(:data => DataFrame(grouping_covariate=repeat(1:N_levels,
            inner=N_obs ÷ N_levels)[1:N_obs]))
        mock_mod_data = Dict(:variables => :grouping_covariate)
        
        mock_M_pc = (data=mock_M_ds[:data],)
        res_pc = TethysBayes.get_precomputes(m_iid, mock_M_pc, mock_mod_data)
        @test hasproperty(res_pc, :n_latent) || res_pc == NamedTuple()

        mock_M_priors = (technical=(component_levels=Dict(:grouping_covariate => N_levels),),)
        mock_spec_priors = mock_spec(:grouping_covariate)
        priors_str = TethysBayes.get_priors(m_iid, mock_spec_priors, "univariate", nothing,
            mock_M_priors)
        @test contains(priors_str, "sigma_grouping_covariate ~ Exponential(1.0)")
        @test contains(priors_str, "innovations_grouping_covariate ~ MvNormal(zeros(T,")

        mock_M_updates = (technical=(component_indices=Dict(:grouping_covariate => mock_M_ds[:data].grouping_covariate),), model_arch="univariate")
        mock_spec_updates = mock_spec(:grouping_covariate)
        updates_str = TethysBayes.get_updates(m_iid, mock_spec_updates, "univariate", nothing,
            mock_M_updates)
        @test contains(updates_str, "latent_field_grouping_covariate = innovations_grouping_covariate .* sigma_grouping_covariate")
        @test contains(updates_str, "view(latent_field_grouping_covariate,")

        mock_chain_effects = mock_chain(Dict(:sigma_grouping_covariate => 0.5,
            :innovations_grouping_covariate => randn(N_levels, 10)), 10)
        mock_M_effects = (technical=(component_indices=Dict(:grouping_covariate => mock_M_ds[:data].grouping_covariate),), model_arch="univariate")
        mock_spec_effects = mock_spec(:grouping_covariate)
        
        effects_result = TethysBayes.get_effects(m_iid, mock_chain_effects, mock_spec_effects,
            (outcomes_N=1, model_arch="univariate", technical=mock_M_effects.technical),
            nothing)
        @test size(effects_result.structured[1]) == (N_obs, 10)
    end

    # Test Leroux Component
    @testset "Leroux Component" begin
        N_areas, N_obs = 10, 100
        W_leroux = create_chain_adj_matrix(N_areas)
        m_leroux = TethysBayes.Leroux(Distributions.Beta(1, 1), Distributions.Exponential(1.0),
            :spectral)

        mock_M_ds = Dict(:data => DataFrame(s_idx=repeat(1:N_areas,
            inner=N_obs ÷ N_areas)[1:N_obs]), :W => W_leroux)
        mock_mod_data = Dict(:variables => :s_idx)

        mock_M_pc = (data=mock_M_ds[:data], W=W_leroux, s_N=N_areas)
        precomputes = TethysBayes.get_precomputes(m_leroux, mock_M_pc, mock_mod_data)
        @test hasproperty(precomputes, :Q_template)
        @test size(precomputes.Q_template) == (N_areas, N_areas)

        mock_M_priors = (technical=(component_levels=Dict(:s_idx => N_areas),),)
        mock_spec_priors = mock_spec(:s_idx, precomputes)
        priors_str = TethysBayes.get_priors(m_leroux, mock_spec_priors, "univariate", nothing,
            mock_M_priors)
        @test contains(priors_str, "sigma_s_idx ~ Exponential(1.0)")
        @test contains(priors_str, "rho_s_idx ~ Beta(1.0, 1.0)")
        @test contains(priors_str, "innovations_s_idx ~ MvNormal(zeros(T, 10), I)")

        mock_M_updates = (technical=(component_indices=Dict(:s_idx => repeat(1:N_areas,
            inner=N_obs ÷ N_areas)[1:N_obs]), ), model_arch="univariate")
        updates_str = TethysBayes.get_updates(m_leroux, mock_spec_priors, "univariate", nothing,
            mock_M_updates)
        @test contains(updates_str, "latent_field_s_idx = hyper.U * (diag_D .* innovations_s_idx)")
        @test contains(updates_str, "eta = eta .+ view(latent_field_s_idx, M.s_idx)")
    end

    # Test GP Component
    @testset "GP Component" begin
        N_obs, N_dims = 100, 2
        m_gp = TethysBayes.GP(Distributions.Gamma(2, 0.5), Distributions.Exponential(1.0), "se",
            :noncentered)
        
        mock_M_ds = Dict(:data => DataFrame(x=rand(N_obs), y=rand(N_obs)))
        mock_mod_data = Dict(:variables => [:x, :y], :params => Dict(:coords => rand(N_obs,
            N_dims)))

        mock_M_pc = (data=mock_M_ds[:data],)
        precomputes = TethysBayes.get_precomputes(m_gp, mock_M_pc, mock_mod_data)
        @test hasproperty(precomputes, :n_latent)
        @test precomputes.n_latent == N_obs

        mock_M_priors = (technical=(component_levels=Dict(),),)
        mock_spec_priors = mock_spec(:x_y, precomputes)
        priors_str = TethysBayes.get_priors(m_gp, mock_spec_priors, "univariate", nothing,
            mock_M_priors)
        @test contains(priors_str, "sigma_x_y ~ Exponential(1.0)")
        @test contains(priors_str, "length_scale_x_y ~ Gamma(2.0, 0.5)")
        @test contains(priors_str, "innovations_x_y ~ MvNormal(zeros(T, 100), I)")

        mock_M_updates = (technical=(component_indices=Dict(),), model_arch="univariate")
        updates_str = TethysBayes.get_updates(m_gp, mock_spec_priors, "univariate", nothing,
            mock_M_updates)
        @test contains(updates_str, "latent_field_x_y = F_gp.L * innovations_x_y")
        @test contains(updates_str, "eta = eta .+ latent_field_x_y")
    end

    # Test Harmonic Component
    @testset "Harmonic Component" begin
        N_obs, N_time, n_harm, period = 120, 12, 2, 12.0
        m_harm = TethysBayes.Harmonic(n_harm, Distributions.Exponential(1.0),
            Distributions.Beta(1, 1), period, :twocoefficient)
        
        mock_M_ds = Dict(:data => DataFrame(month=rand(1:N_time, N_obs)))
        mock_mod_data = Dict(:variables => :month)

        mock_M_pc = (data=mock_M_ds[:data], N_time=N_time)
        precomputes = TethysBayes.get_precomputes(m_harm, mock_M_pc, mock_mod_data)
        @test hasproperty(precomputes, :u_N)
        @test precomputes.u_N == N_time

        mock_M_priors = (technical=(component_levels=Dict(:month => N_time),),)
        mock_spec_priors = mock_spec(:month, precomputes)
        priors_str = TethysBayes.get_priors(m_harm, mock_spec_priors, "univariate", nothing,
            mock_M_priors)
        @test contains(priors_str, "beta_cos_month ~ filldist(Normal(0.0, 1.0), 2)")
        @test contains(priors_str, "beta_sin_month ~ filldist(Normal(0.0, 1.0), 2)")

        mock_M_updates = (technical=(component_indices=Dict(:month => mock_M_ds[:data].month),), model_arch="univariate")
        updates_str = TethysBayes.get_updates(m_harm, mock_spec_priors, "univariate", nothing,
            mock_M_updates)
        @test contains(updates_str, "latent_field_month = zeros(T_num, u_N_val)")
        @test contains(updates_str, "eta = eta .+ view(latent_field_month, u_idx_val)")
    end
end

@testset "NNGP and MCAR Spatial Components" begin
    # Test 1: NNGP model definition
    df_nngp = DataFrame(
        s_x = [0.1, 0.4, 0.7, 0.2, 0.8, 0.5, 0.9, 0.3],
        s_y = [0.2, 0.5, 0.1, 0.8, 0.9, 0.3, 0.6, 0.7],
        y = randn(MersenneTwister(42), 8)
    )
    m_nngp = @tbm(likelihood(y, family=gaussian) ~ intercept() + random(s_x, model=nngp,
        m=3, kernel=:exponential), df_nngp, verbose=false)
    @test m_nngp isa DynamicPPL.Model
    chn_nngp = sample(m_nngp, MH(), 20; progress=false)
    @test size(chn_nngp, 1) == 20

    # Test 2: MCAR model definition
    df_mcar = DataFrame(
        s_idx = [1, 2, 3, 4, 1, 2, 3, 4],
        y = rand(MersenneTwister(42), 0:10, 8)
    )
    W_4 = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0]
    m_mcar = @tbm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=mcar,
        K=2), df_mcar, W=W_4, verbose=false)
    @test m_mcar isa DynamicPPL.Model
    chn_mcar = sample(m_mcar, MH(), 20; progress=false)
    @test size(chn_mcar, 1) == 20

    @testset "every registered component at least builds and samples" begin
        # A net for the whole component surface. `fft` broke silently for a while and no
        # existing test caught it, because nothing exercised it: an automated rewrite of the
        # `diag_D` deflation sites assumed the eigenvalue variable had the same name in the
        # generated model body (`hyper.L`) and in the `get_effects` reconstruction (a local
        # `L`), got them backwards, and the result was
        # `UndefVarError: L not defined in _GeneratedModelRuntime` -- `fft` could not be
        # sampled AT ALL, and the suite stayed green.
        #
        # The lesson repeated: a component with no test is not a component that works. Four of
        # the original 51 un-migrated `get_effects` sites survived a 40-component sweep purely
        # because the sweep's list did not include them.
        #
        # This is deliberately NOT a correctness check -- it asserts only that a model can be
        # built and a prior draw taken. Anything stronger needs the component's own fixture,
        # and asking for that here would mean duplicating every testset. Its job is to make an
        # *unbuildable* component impossible to miss.
        registered = sort(collect(keys(TethysBayes.COMPONENT_TYPE_REGISTRY)))
        @test !isempty(registered)

        # Components that legitimately need a required argument, or whose reconstruction is
        # not implemented, are allowed to fail here -- but they must be listed ON PURPOSE, so
        # a NEW unbuildable component is a hard failure rather than a silent gap.
        #
        # `mcar` and `warp` were on this list until they were diagnosed one by one, and both
        # turned out to be small bugs rather than missing implementations: `mcar` fed a pinned
        # `sigma` to `_distribution_to_string`, and `warp` referenced the runtime-only name
        # `spec_registry` from `TethysBayes`'s own scope. Diagnosing beat assuming, again.
        #
        # `spacetime` is a space-time model and needs a temporal column this fixture does not
        # have. It looked buildable in a diagnostic that happened to supply one -- a reminder
        # that a passing probe says something about the probe's fixture, not the component.
        # (Its pinned-`sigma`-into-a-Distribution-only-field bug was real and is fixed.)
        # `spacetime` is gone: it was a bare alias for `svar` and has been removed, with a
        # constructor retained only to fail with guidance. It is deliberately absent from this
        # list, because the testset also asserts every entry is still registered.
        # `waveletgp` is no longer listed: it used `wavedec`, which the installed Wavelets
        # 0.10 does not provide, so it could not be built at all. It now derives its scale
        # bands from the length-preserving multiresolution layout directly, and builds.
        # `graph_wavelet`/`sgw` remain blocked on an undefined `graph_wavelet_basis_matrix`.
        expected_unbuildable = Set{Symbol}([
            :tar, :sciml, :networkflow, :barycentric, :composed, :mixed,
            :nonstationaryvariance, :tensorproductsmooth, :svar, :dynamics,
            :graph_wavelet, :sgw, :eigen, :fitc, :sparsegp, :nystrom, :tvc,
            :astar, :svc, :svgp,
        ])

        nsurf = 8
        ds = DataFrame(idx = collect(1:nsurf), x = collect(1.0:nsurf),
                        y = zeros(Float64, nsurf))
        ds.depth = 9.0 .+ 3.0 .* sin.(ds.x ./ 3.0)
        Ws = spzeros(Int, nsurf, nsurf)
        for j in 1:(nsurf - 1)
            Ws[j, j + 1] = 1
            Ws[j + 1, j] = 1
        end
        Ws[1, nsurf] = 1
        Ws[nsurf, 1] = 1

        unbuildable = Symbol[]
        for model in registered
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(idx, model=$model, sigma=1.0)"
            ok = try
                m = TethysBayes.TethysBayes_core(f, ds; W = Ws, verbose = false)
                # A prior draw is the cheapest thing that exercises the generated body, where
                # the helper allowlist, the code templates and the binding names all have to
                # resolve. `TethysBayes_core` alone would not touch any of them.
                sample(MersenneTwister(3), m, Prior(), 5; progress = false)
                true
            catch
                false
            end
            ok || push!(unbuildable, model)
        end

        @test isempty(setdiff(unbuildable, expected_unbuildable))
        # The list must not rot: every entry has to still name a registered component.
        @test issubset(expected_unbuildable, Set(registered))
    end

    @testset "WaveletGP scale bands and reconstruction" begin
        # `waveletgp` was permanently unbuildable: it called `wavedec`, which the installed
        # Wavelets 0.10 / WaveletsExt 0.2 do not provide. Everything below pins down the
        # replacement -- the length-preserving multiresolution layout -- so a future library
        # change cannot silently reintroduce a mislabelled or ill-sized band structure.

        @testset "band lengths sum to the resolution" begin
            for res in (4, 8, 16, 32, 64)
                lengths = TethysBayes._wavelet_decomposition_lengths(res)
                # One approximation band, then level l carries 2^(l-1) coefficients.
                @test lengths[1] == 1
                @test lengths[2:end] == [2^(l - 1) for l in 1:TethysBayes._wavelet_max_level(res)]
                # Length-preserving transform: the bands must tile the grid exactly, or
                # `innovations .* sqrt.(scale_variances)` is a dimension mismatch.
                @test sum(lengths) == res
                @test length(TethysBayes._wavelet_scale_indices_1d(res)) == res
                @test TethysBayes._wavelet_max_level(res) == trailing_zeros(res)
            end
        end

        @testset "coarsest-first, no negative scales" begin
            idx = TethysBayes._wavelet_scale_indices_1d(16)
            @test idx == [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
            # The approximation band is index 0; fine bands must be larger, never negative.
            # The old loop `max_lvl - (i - 1)` produced negative indices for the later bands,
            # which inflated their variances above sigma^2 for any alpha > 0.
            @test minimum(idx) == 0
            @test maximum(idx) == TethysBayes._wavelet_max_level(16)
            @test issorted(idx)
        end

        @testset "bands match the library's own basis functions" begin
            WV = TethysBayes.Wavelets
            wt = WV.wavelet(WV.WT.db4)
            for res in (8, 16)
                max_lvl = TethysBayes._wavelet_max_level(res)
                idx = TethysBayes._wavelet_scale_indices_1d(res)
                # L1/Linf is constant within a scale band, so this checks the assignment
                # against the basis rather than against a restatement of the same formula.
                # Compared with a tolerance: the values agree to ~1e-15 but not bitwise.
                ratios = map(1:res) do j
                    e = zeros(res); e[j] = 1.0
                    y = WV.idwt(e, wt, max_lvl)
                    sum(abs.(y)) / maximum(abs.(y))
                end
                @test all(isapprox(ratios[i], ratios[j]; rtol = 1e-9)
                          for i in eachindex(idx) for j in eachindex(idx) if idx[i] == idx[j])
                # A constant signal must load only the approximation coefficient.
                c = WV.dwt(ones(res), wt, max_lvl)
                @test findall(abs.(c) .> 1e-12) == [1]
            end
        end

        @testset "power law decays towards fine scales" begin
            idx = TethysBayes._wavelet_scale_indices_1d(16)
            for alpha in (0.5, 1.5, 3.0)
                var = 1.0 .* (2.0 .^ (-alpha .* idx))
                for lvl in 0:(maximum(idx) - 1)
                    @test var[findfirst(==(lvl), idx)] > var[findfirst(==(lvl + 1), idx)]
                end
            end
        end

        @testset "2-D index direction agrees with 1-D" begin
            s2 = reshape(TethysBayes._get_wavelet_scale_indices_2d(16), 16, 16)
            # Residual approximation is the coarsest block; the first (largest) detail
            # subband is the finest. The old code inverted both.
            @test s2[1, 1] == 0
            @test all(s2[1:8, 9:16] .== maximum(TethysBayes._get_wavelet_scale_indices_2d(16)))
            @test minimum(TethysBayes._get_wavelet_scale_indices_2d(16)) == 0
        end

        @testset "basis is orthonormal and analysis/synthesis is exact" begin
            WV = TethysBayes.Wavelets
            wt = WV.wavelet(WV.WT.db4)
            for res in (8, 16)
                max_lvl = TethysBayes._wavelet_max_level(res)
                Phi = zeros(res, res)
                for j in 1:res
                    e = zeros(res); e[j] = 1.0
                    Phi[:, j] = WV.idwt(e, wt, max_lvl)
                end
                @test maximum(abs.(Phi' * Phi - Matrix{Float64}(I, res, res))) < 1e-10
                x = collect(1.0:res) ./ res
                @test maximum(abs.(Phi * WV.dwt(x, wt, max_lvl) - x)) < 1e-10
            end
        end

        @testset "non-power-of-2 resolution is rejected" begin
            d = DataFrame(x = collect(1.0:16))
            d.depth = 9.0 .+ 3.0 .* sin.(d.x ./ 3.0)
            @test_throws Exception TethysBayes.TethysBayes_core(
                "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(x, model=waveletgp, resolution=17, wavelet=db4)", d; verbose = false)
        end

        @testset "builds and reconstructs a non-trivial field" begin
            # `sigma` is pinned here, which previously made `_find_parameter` miss all three
            # parameters and return an all-zero effect matrix: the fit silently looked empty.
            for (dd, expr) in (
                (DataFrame(x = collect(1.0:16)),
                    "random(x, model=waveletgp, sigma=1.0, resolution=16, wavelet=db4)"),
                (DataFrame(x = collect(1.0:16), y = collect(1.0:16) ./ 2),
                    "random(x, y, model=waveletgp, sigma=1.0, resolution=16, wavelet=db4)"))
                dd.depth = 9.0 .+ 3.0 .* sin.(dd.x ./ 3.0)
                m = TethysBayes.TethysBayes_core(
                    "likelihood(depth, family=gaussian) ~ intercept() + " * expr, dd;
                    verbose = false)
                spec = m.args.M.components[1]
                chn = sample(MersenneTwister(3), m, Prior(), 20; progress = false)
                eff = TethysBayes.get_effects(spec.component_obj, chn, spec, m.args.M, nothing)
                S = eff.structured[1]
                @test all(isfinite, S)
                @test any(abs.(S) .> 1e-8)
                @test size(S) == (16, 20)
                # No zero-matrix fallback and no missing-parameter warning path.
                @test std(S) > 1e-6
            end
        end

        @testset "wavelet families resolve to distinct bases" begin
            seen = Float64[]
            for w in (:db2, :db4, :db6, :db8, :haar, :coif2, :coif4, :sym4, :sym8)
                d = DataFrame(x = collect(1.0:16))
                d.depth = 9.0 .+ 3.0 .* sin.(d.x ./ 3.0)
                m = TethysBayes.TethysBayes_core(
                    "likelihood(depth, family=gaussian) ~ intercept() + " *
                    "random(x, model=waveletgp, resolution=16, wavelet=$(QuoteNode(w)))", d;
                    verbose = false)
                push!(seen, sum(abs.(m.args.M.components[1].hyper.Phi_wavelet[:, 2])))
            end
            @test length(unique(round.(seen, digits = 6))) == length(seen)
        end
    end
end
