# ==============================================================================
# TethysBayes Test Suite: Analytical Surface Derivatives & Topographic Metrics
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Surface Derivatives & Differential Geometry Tests" begin

    @testset "Topographic Metrics Differential Geometry Formulation" begin
        # Test flat horizontal plane: z(x, y) = 10
        zx = [0.0]
        zy = [0.0]
        zxx = [0.0]
        zyy = [0.0]
        zxy = [0.0]
        topo_flat = TethysBayes.compute_topographic_metrics(zx, zy, zxx, zyy, zxy)
        @test topo_flat.slope_gradient[1] ≈ 0.0 atol=1e-6
        @test topo_flat.slope_degrees[1] ≈ 0.0 atol=1e-6
        @test topo_flat.aspect_degrees[1] == -1.0 # Flat surface aspect
        @test topo_flat.laplacian[1] ≈ 0.0 atol=1e-6

        # Test tilted plane: z(x, y) = 3x + 4y (gradient = (3, 4), slope = 5.0)
        zx_tilt = [3.0]
        zy_tilt = [4.0]
        topo_tilt = TethysBayes.compute_topographic_metrics(zx_tilt, zy_tilt, zxx, zyy, zxy)
        @test topo_tilt.slope_gradient[1] ≈ 5.0 atol=1e-6
        @test topo_tilt.slope_degrees[1] ≈ atan(5.0) * (180.0 / pi) atol=1e-4
        @test 0.0 <= topo_tilt.aspect_degrees[1] <= 360.0

        # Test parabolic bowl: z(x, y) = x^2 + y^2 at (1, 1)
        # zx = 2, zy = 2, zxx = 2, zyy = 2, zxy = 0 -> laplacian = 4.0
        topo_bowl = TethysBayes.compute_topographic_metrics([2.0], [2.0], [2.0], [2.0], [0.0])
        @test topo_bowl.slope_gradient[1] ≈ sqrt(8.0) atol=1e-6
        @test topo_bowl.laplacian[1] ≈ 4.0 atol=1e-6
        @test topo_bowl.profile_curvature[1] < 0.0 # Curving upward in direction of slope
    end

    @testset "RFF Analytical Surface Derivatives & Bessel BPI" begin
        Random.seed!(123)
        N = 60
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        # True bathymetric surface
        z_true = [100.0 + 20.0 * sin(x / 15.0) + 15.0 * cos(y / 15.0) + randn() * 0.5 for (x, y) in zip(x_pts, y_pts)]

        df_bathy = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true)

        m_rff = @tbm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=rff, n_features=25),
            df_bathy, verbose=false
        )
        chn_rff = sample(m_rff, NUTS(20, 0.65), 40; progress=false)

        # Query points on a test grid
        query_coords = DataFrame(
            s_x = [30.0, 50.0, 70.0],
            s_y = [30.0, 50.0, 70.0]
        )

        deriv_res = TethysBayes_surface_derivatives(
            m_rff, chn_rff, query_coords;
            metrics=[:slope, :curvature, :bpi],
            radii=[5.0, 15.0],
            return_samples=true
        )

        @test haskey(deriv_res, :summary)
        @test haskey(deriv_res, :metrics)
        @test haskey(deriv_res, :samples)

        sum_df = deriv_res.summary
        @test nrow(sum_df) == 3
        @test hasproperty(sum_df, :z_mean)
        @test hasproperty(sum_df, :slope_mean)
        @test hasproperty(sum_df, :slope_deg_mean)
        @test hasproperty(sum_df, :aspect_deg_mean)
        @test hasproperty(sum_df, :laplacian_mean)
        @test hasproperty(sum_df, :profile_curv_mean)
        @test hasproperty(sum_df, :planform_curv_mean)
        @test hasproperty(sum_df, :bpi_r5_0_mean)
        @test hasproperty(sum_df, :bpi_r15_0_mean)

        # Sanity checks on physical realism
        @test all(sum_df.slope_mean .>= 0.0)
        @test all(0.0 .<= sum_df.slope_deg_mean .<= 90.0)
        @test all(sum_df.z_mean .> 30.0)

        # Test sample matrices shape
        @test size(deriv_res.samples.elevation) == (3, 40)
        @test size(deriv_res.samples.slope_gradient) == (3, 40)
        @test size(deriv_res.samples.laplacian) == (3, 40)
        @test size(deriv_res.samples.bpi[5.0]) == (3, 40)
    end

    @testset "SpectralGP Exact Fourier Domain Derivatives" begin
        Random.seed!(456)
        N = 50
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        z_true = [150.0 - 0.5 * x + 0.3 * y + randn() * 0.5 for (x, y) in zip(x_pts, y_pts)]
        df_gp = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true)

        m_sgp = @tbm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=spectral_gp, resolution=16),
            df_gp, verbose=false
        )
        chn_sgp = sample(m_sgp, NUTS(20, 0.65), 40; progress=false)

        query_coords = Matrix{Float64}([40.0 40.0; 60.0 60.0])
        sgp_derivs = TethysBayes_surface_derivatives(
            m_sgp, chn_sgp, query_coords;
            radii=[10.0],
            return_samples=true
        )

        @test nrow(sgp_derivs.summary) == 2
        @test all(sgp_derivs.summary.slope_mean .>= 0.0)
        @test haskey(sgp_derivs.metrics, :slope)
        @test haskey(sgp_derivs.metrics, :laplacian)
        @test haskey(sgp_derivs.metrics, :bpi)
        @test size(sgp_derivs.samples.elevation) == (2, 40)
    end

    @testset "WaveletGP Multi-Scale Wavelet Derivatives & BPI" begin
        Random.seed!(789)
        N = 50
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        z_true = [120.0 + 10.0 * sin(x / 10.0) + randn() * 0.5 for (x, y) in zip(x_pts, y_pts)]
        df_wgp = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true)

        m_wgp = @tbm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=waveletgp, resolution=16, wavelet=:db4),
            df_wgp, verbose=false
        )
        chn_wgp = sample(m_wgp, NUTS(20, 0.65), 40; progress=false)

        query_coords = Matrix{Float64}([35.0 35.0; 65.0 65.0])
        wgp_derivs = TethysBayes_surface_derivatives(
            m_wgp, chn_wgp, query_coords;
            radii=[12.0],
            return_samples=true
        )

        @test nrow(wgp_derivs.summary) == 2
        @test all(wgp_derivs.summary.slope_mean .>= 0.0)
        @test haskey(wgp_derivs.metrics, :slope)
        @test haskey(wgp_derivs.metrics, :laplacian)
        @test haskey(wgp_derivs.metrics, :bpi)
        @test size(wgp_derivs.samples.elevation) == (2, 40)
    end

    @testset "SPDE Triangulation & Graph Laplacian Surface Derivatives" begin
        Random.seed!(321)
        N_units = 25
        # Create 2D grid centroids
        grid_1d = range(10.0, stop=90.0, length=5)
        cx = repeat(collect(grid_1d), inner=5)
        cy = repeat(collect(grid_1d), outer=5)
        
        # Build spatial adjacency matrix W
        W_spde = spzeros(Int, N_units, N_units)
        for i in 1:N_units
            for j in 1:N_units
                if i != j && sqrt((cx[i] - cx[j])^2 + (cy[i] - cy[j])^2) <= 25.0
                    W_spde[i, j] = 1
                end
            end
        end

        N_obs = 60
        s_indices = vcat(collect(1:N_units), rand(1:N_units, N_obs - N_units))
        z_obs = [100.0 + 0.5 * cx[s] - 0.2 * cy[s] + randn() * 0.5 for s in s_indices]
        df_spde = DataFrame(s_idx = s_indices, depth = z_obs)

        m_spde = @tbm(
            likelihood(depth) ~ intercept() + random(s_idx, model=spde),
            df_spde, W=W_spde, verbose=false
        )
        chn_spde = sample(m_spde, NUTS(20, 0.65), 40; progress=false)

        unit_coords = [cx cy]
        spde_derivs = TethysBayes_surface_derivatives(
            m_spde, chn_spde, unit_coords;
            radii=[20.0],
            return_samples=true
        )

        @test nrow(spde_derivs.summary) == N_units
        @test all(spde_derivs.summary.slope_mean .>= 0.0)
        @test haskey(spde_derivs.metrics, :slope)
        @test haskey(spde_derivs.metrics, :laplacian)
        @test haskey(spde_derivs.metrics, :bpi)
        @test size(spde_derivs.samples.elevation) == (N_units, 40)
    end

    @testset "Nystrom, FITC, and SVC Continuous Surface Derivatives" begin
        Random.seed!(987)
        N = 35
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        z_true = [
            60.0 + 10.0 * sin(x / 20.0) + 8.0 * cos(y / 20.0) + randn() * 0.2
            for (x, y) in zip(x_pts, y_pts)
        ]
        cov_val = randn(N)
        df_test = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true, temp = cov_val)

        query_coords = DataFrame(s_x = [30.0, 60.0], s_y = [30.0, 60.0])

        # 1. Nystrom Low-Rank Sparse GP
        m_nystrom = @tbm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=nystrom, n_inducing=8),
            df_test, verbose=false
        )
        chn_nystrom = sample(m_nystrom, Prior(), 20; progress=false)
        res_nystrom = TethysBayes_surface_derivatives(
            m_nystrom, chn_nystrom, query_coords;
            radii=[10.0], return_samples=true
        )

        @test nrow(res_nystrom.summary) == 2
        @test hasproperty(res_nystrom.summary, :slope_mean)
        @test hasproperty(res_nystrom.summary, :laplacian_mean)
        @test hasproperty(res_nystrom.summary, :bpi_r10_0_mean)
        @test all(res_nystrom.summary.slope_mean .>= 0.0)
        @test haskey(res_nystrom.metrics, :slope)
        @test haskey(res_nystrom.metrics, :bpi)
        @test size(res_nystrom.samples.elevation) == (2, 20)

        # 2. FITC Inducing Point Sparse GP
        m_fitc = @tbm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=fitc, n_inducing=8),
            df_test, verbose=false
        )
        chn_fitc = sample(m_fitc, Prior(), 20; progress=false)
        res_fitc = TethysBayes_surface_derivatives(
            m_fitc, chn_fitc, query_coords;
            radii=[10.0], return_samples=true
        )

        @test nrow(res_fitc.summary) == 2
        @test hasproperty(res_fitc.summary, :slope_mean)
        @test hasproperty(res_fitc.summary, :laplacian_mean)
        @test hasproperty(res_fitc.summary, :bpi_r10_0_mean)
        @test all(res_fitc.summary.slope_mean .>= 0.0)
        @test haskey(res_fitc.metrics, :slope)
        @test haskey(res_fitc.metrics, :bpi)
        @test size(res_fitc.samples.elevation) == (2, 20)

        # 3. Spatially Varying Coefficients (SVC)
        m_svc = @tbm(
            likelihood(depth) ~ intercept() + (temp |> random(s_x, s_y, model=rff, n_features=10)),
            df_test, verbose=false
        )
        chn_svc = sample(m_svc, Prior(), 20; progress=false)
        res_svc = TethysBayes_surface_derivatives(
            m_svc, chn_svc, query_coords;
            radii=[10.0], return_samples=true
        )

        @test nrow(res_svc.summary) == 2
        @test hasproperty(res_svc.summary, :slope_mean)
        @test hasproperty(res_svc.summary, :laplacian_mean)
        @test hasproperty(res_svc.summary, :bpi_r10_0_mean)
        @test all(res_svc.summary.slope_mean .>= 0.0)
        @test haskey(res_svc.metrics, :slope)
        @test haskey(res_svc.metrics, :bpi)
        @test size(res_svc.samples.elevation) == (2, 20)
    end

end
