# ==============================================================================
# TethysBayes Test Suite: Errors-in-Variables (EIV) Covariate Priors
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Errors-in-Variables (EIV) Covariate Priors" begin

    @testset "EIV Model Instantiation & Configuration" begin
        Random.seed!(42)
        N = 50
        x_true = randn(N)
        x_sd = fill(0.4, N)
        x_obs = x_true .+ randn(N) .* x_sd
        y = 2.0 .+ 1.5 .* x_true .+ randn(N) .* 0.3

        df_eiv = DataFrame(
            y = y,
            x_obs = x_obs,
            x_sd = x_sd
        )

        # Test constant error_sd
        m_const_sd = @tbm(
            likelihood(y) ~ intercept() + fixed(x_obs, error_sd=0.4),
            df_eiv, verbose=false
        )
        @test haskey(m_const_sd.args.M, :Xfixed_eiv_map)
        @test haskey(m_const_sd.args.M.Xfixed_eiv_map, :x_obs)
        @test length(m_const_sd.args.M.Xfixed_eiv_map[:x_obs]) == N

        # Test column-referenced error_sd
        m_col_sd = @tbm(
            likelihood(y) ~ intercept() + fixed(x_obs, error_sd=:x_sd),
            df_eiv, verbose=false
        )
        @test haskey(m_col_sd.args.M.Xfixed_eiv_map, :x_obs)
        @test m_col_sd.args.M.Xfixed_eiv_map[:x_obs] == x_sd

        # Verify generated Turing model string contains latent innovation prior
        code_str = m_col_sd.args.M.generated_model_code
        @test contains(code_str, "innovations_eiv_x_obs")
        @test contains(code_str, "lat_noise = M.Xfixed_eiv_map[:x_obs] .* innovations_eiv_x_obs")
        @test contains(code_str, "X_lat_x_obs = M.Xfixed[:, 1] .+ lat_noise")
    end

    @testset "EIV Mixed Fixed Effects & NUTS Sampling" begin
        Random.seed!(101)
        N = 40
        # True latent variables
        substrate_true = randn(N)
        substrate_se = fill(0.35, N)
        substrate_obs = substrate_true .+ randn(N) .* substrate_se
        temp = rand(Uniform(2.0, 10.0), N) # Exactly measured covariate

        y = 10.0 .- 2.5 .* substrate_true .+ 0.8 .* temp .+ randn(N) .* 0.2

        df_mixed = DataFrame(
            y = y,
            substrate = substrate_obs,
            substrate_se = substrate_se,
            temp = temp
        )

        # Model with one EIV covariate and one standard fixed effect
        m_mixed = @tbm(
            likelihood(y) ~ intercept() + fixed(substrate, error_sd=:substrate_se) + fixed(temp),
            df_mixed, verbose=false
        )

        # Sample with NUTS
        chn = sample(m_mixed, NUTS(20, 0.65), 30; progress=false)

        # Verify chains contain beta, intercept, and innovations_eiv_substrate
        p_names = string.(keys(chn))
        @test any(p -> occursin("beta", string(p)), p_names)
        @test any(p -> occursin("innovations_eiv_substrate", string(p)), p_names)

        # Verify sample extraction
        eiv_samples = TethysBayes.get_params_matrix(chn, "innovations_eiv_substrate", N)
        @test size(eiv_samples) == (30, N)
    end

end
