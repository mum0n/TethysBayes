# ==============================================================================
# TethysBayes Test Suite: Modular DAG Pipeline Orchestrator & Multi-Tier Resharding
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Modular DAG Pipeline Orchestrator Tests" begin

    @testset "Spatial Network Transfer Matrix & Resharding" begin
        Random.seed!(42)
        # Create Source Network (12 units)
        x_src = rand(Uniform(10.0, 90.0), 40)
        y_src = rand(Uniform(10.0, 90.0), 40)
        au_src = assign_spatial_units(x_src, y_src; target_units=12)

        # Create Destination Network (20 units)
        x_dest = rand(Uniform(10.0, 90.0), 60)
        y_dest = rand(Uniform(10.0, 90.0), 60)
        au_dest = assign_spatial_units(x_dest, y_dest; target_units=20)

        # Compute transfer operator P
        P = compute_network_transfer_matrix(au_src, au_dest)
        @test size(P) == (length(au_dest.centroids), length(au_src.centroids))
        
        # Verify convex combination row sums
        row_sums = vec(sum(P, dims=2))
        @test all(isapprox.(row_sums, 1.0, atol=1e-4))

        # Test field resharding with raw matrix (Full Monte Carlo sample draws)
        src_field_mat = randn(length(au_src.centroids), 50)
        dest_field_mat = reshard_spatial_field(src_field_mat, au_src, au_dest)
        @test size(dest_field_mat) == (length(au_dest.centroids), 50)

        # Test summarize_sample_matrix
        sum_stats = summarize_sample_matrix(dest_field_mat; alpha=0.05)
        @test length(sum_stats.mean) == length(au_dest.centroids)
        @test length(sum_stats.std) == length(au_dest.centroids)
        @test length(sum_stats.lower) == length(au_dest.centroids)
        @test length(sum_stats.upper) == length(au_dest.centroids)
        @test all(sum_stats.lower .<= sum_stats.upper)

        # Test reshard_spatial_field on NamedTuple with mode=:samples vs mode=:moments
        nt_src = (
            mean = vec(mean(src_field_mat, dims=2)),
            std  = vec(std(src_field_mat, dims=2)),
            lower = zeros(length(au_src.centroids)),
            upper = zeros(length(au_src.centroids)),
            samples = src_field_mat
        )
        resharded_mc = reshard_spatial_field(nt_src, au_src, au_dest; mode=:samples)
        @test resharded_mc.samples !== nothing
        @test size(resharded_mc.samples) == (length(au_dest.centroids), 50)
        @test isapprox(resharded_mc.mean, vec(mean(resharded_mc.samples, dims=2)), atol=1e-10)

        resharded_moments = reshard_spatial_field(nt_src, au_src, au_dest; mode=:moments)
        @test resharded_moments.samples === nothing
        @test length(resharded_moments.mean) == length(au_dest.centroids)
        @test isapprox(resharded_moments.mean, resharded_mc.mean, atol=1e-5)
    end

    @testset "3-Tier Hierarchical Pipeline End-to-End" begin
        Random.seed!(123)
        N_pts = 60
        x = rand(Uniform(10.0, 90.0), N_pts)
        y = rand(Uniform(10.0, 90.0), N_pts)

        # Tier 1 (Continuous Bathymetry)
        depth_true = [100.0 + 20.0 * sin(xi / 15.0) + 15.0 * cos(yi / 15.0) for (xi, yi) in zip(x, y)]
        df_bathy = DataFrame(s_x = x, s_y = y, depth = depth_true .+ randn(N_pts) .* 0.5)

        # Tier 2 (Discrete Substrate on 10-unit coarse mesh)
        au_sed = assign_spatial_units(x, y; target_units=10)
        sed_unit_true = randn(length(au_sed.centroids))
        s_idx_sed = au_sed.assignments
        grain_obs = [2.0 + 0.05 * depth_true[i] + sed_unit_true[s_idx_sed[i]] + randn() * 0.2 for i in 1:N_pts]
        df_sed = DataFrame(s_idx = s_idx_sed, s_x = x, s_y = y, grain = grain_obs)

        # Tier 3 (Focal Target Species on 18-unit master mesh)
        au_master = assign_spatial_units(x, y; target_units=18)
        s_idx_bio = au_master.assignments
        catch_obs = [exp(1.5 - 0.01 * depth_true[i] + 0.3 * grain_obs[i] + randn() * 0.1) for i in 1:N_pts]
        df_bio = DataFrame(s_idx = s_idx_bio, s_x = x, s_y = y, catch_val = catch_obs)

        # Run declarative pipeline
        pipe_res = TethysBayes_pipeline(
            :depth => (
                formula = "likelihood(depth) ~ intercept() + random(s_x, s_y, model=rff, n_features=15)",
                data = df_bathy,
                sampler = NUTS(15, 0.65),
                n_samples = 25,
                derivatives = [:slope, :curvature, :bpi],
                radii = [10.0]
            ),
            :substrate => (
                formula = "likelihood(grain) ~ intercept() + fixed(depth) + random(s_idx, model=bym2)",
                data = df_sed,
                au = au_sed,
                sampler = NUTS(15, 0.65),
                n_samples = 25
            ),
            :biology => (
                formula = "likelihood(catch_val, family=lognormal) ~ intercept() + fixed(substrate, error_sd=:substrate_se) + random(s_idx, model=bym2)",
                data = df_bio,
                au = au_master,
                sampler = NUTS(15, 0.65),
                n_samples = 25
            );
            master_au = au_master,
            duckdb_path = ":memory:",
            verbose = false
        )

        @test pipe_res isa PipelineResult
        @test pipe_res.tier_names == [:depth, :substrate, :biology]
        @test haskey(pipe_res.models, :depth)
        @test haskey(pipe_res.models, :substrate)
        @test haskey(pipe_res.models, :biology)
        @test haskey(pipe_res.derivatives, :depth)

        # Verify Master Harmonized Summary
        m_df = pipe_res.master_summary
        @test nrow(m_df) == length(au_master.centroids)
        @test hasproperty(m_df, :depth_mean)
        @test hasproperty(m_df, :depth_sd)
        @test hasproperty(m_df, :substrate_mean)
        @test hasproperty(m_df, :substrate_sd)
        @test hasproperty(m_df, :biology_mean)
        @test hasproperty(m_df, :biology_sd)
    end

end
