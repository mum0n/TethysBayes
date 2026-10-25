# ==============================================================================
# TethysBayes Test Suite: Spatial Partitioning and Graph Construction Engine
# ==============================================================================

if !@isdefined(TethysBayes_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Spatial Partitioning Engine" begin
    s_N = 100
    t_N = 15
    coords = rand(s_N, 2) .* 100
    coords_tuples = [(coords[i, 1], coords[i, 2]) for i in 1:s_N]
    t_idx = repeat(1:t_N, inner=cld(s_N, t_N))[1:s_N]

    partitioning_methods = [:cvt, :kvt, :qvt, :bvt, :avt, :hvt, :lattice]

    for method in partitioning_methods
        @testset "Method: $method" begin
            au = TethysBayes.assign_spatial_units(
                coords_tuples;
                area_method = method,
                t_idx = t_idx,
                target_units = 20,
                min_points = 1
            )
            @test au isa NamedTuple
            @test hasproperty(au, :centroids)
            @test hasproperty(au, :W)
            @test length(au.centroids) > 0
            @test size(au.W, 1) == size(au.W, 2)
            @test size(au.W, 1) == length(au.centroids)
            @test length(au.s_idx) == s_N
        end
    end
end

@testset "Partitioning Subsystem Enhancements" begin
    rng = MersenneTwister(123)
    sx = rand(rng, 50) .* 10.0
    sy = rand(rng, 50) .* 10.0
    df_geo = DataFrame(s_x=sx, s_y=sy, t_idx=rand(rng, 1:4, 50))

    # 1. Hexagonal partitioning
    au_hex = TethysBayes.assign_spatial_units(sx, sy; area_method=:hexagonal, target_units=8)
    @test length(au_hex.centroids) > 0
    @test size(au_hex.W, 1) == length(au_hex.centroids)
    @test length(au_hex.s_idx) == 50

    # 2. Fast Lattice partitioning
    au_lat = TethysBayes.assign_spatial_units(sx, sy; area_method=:lattice, target_units=9)
    @test length(au_lat.centroids) > 0
    @test size(au_lat.W, 1) == length(au_lat.centroids)

    # 3. Spatial Weights Matrix (Row standardization)
    W_row = TethysBayes.spatial_weights_matrix(au_hex.W; style=:row_standardized)
    @test all(r -> isapprox(r, 1.0; atol=1e-5) || isapprox(r, 0.0; atol=1e-5),
        sum(W_row, dims=2))

    # 4. Spatial KNN and Radius Graphs
    coords_tuple = tuple.(sx, sy)
    g_knn, W_knn = TethysBayes.spatial_knn_graph(coords_tuple, 4)
    @test Graphs.nv(g_knn) == 50
    @test size(W_knn) == (50, 50)

    g_rad, W_rad = TethysBayes.spatial_radius_graph(coords_tuple, 3.0)
    @test Graphs.nv(g_rad) == 50

    # 5. Spatial Block Cross-Validation
    folds_km = TethysBayes.spatial_block_cv(sx, sy; n_folds=5, method=:kmeans)
    @test length(folds_km) == 50
    @test length(unique(folds_km)) <= 5

    folds_grid = TethysBayes.spatial_block_cv(sx, sy; n_folds=4, method=:grid)
    @test length(folds_grid) == 50

    # 6. Spatiotemporal Units Assignment
    st_res = TethysBayes.assign_spatiotemporal_units(df_geo; target_units=6, area_method=:hexagonal)
    @test length(st_res.st_idx) == 50
    @test st_res.S == length(st_res.au_spatial.centroids)
    @test st_res.T == 4
    @test st_res.scaling_factor_spatial > 0.0

    # 7. Granular Polygon Sizing & Exact Count Control
    au_exact = TethysBayes.assign_spatial_units(sx, sy; area_method=:cvt, target_units=7,
        exact_units=true)
    @test au_exact.n_units == 7
    @test length(au_exact.centroids) == 7
    @test length(au_exact.areas) == 7
    @test haskey(au_exact.metrics, :mean_area)
    @test haskey(au_exact.metrics, :total_area)

    au_area = TethysBayes.assign_spatial_units(sx, sy; area_method=:lattice, target_area=20.0)
    @test au_area.n_units > 0
    @test all(a -> a > 0.0, au_area.areas)

    au_grid_res = TethysBayes.assign_spatial_units(sx, sy; area_method=:lattice,
        grid_resolution=(4, 5))
    @test au_grid_res.n_units > 0

    au_min_area = TethysBayes.assign_spatial_units(sx, sy; area_method=:hexagonal,
        target_units=10, min_area=2.0, merge_small_polygons=true)
    @test all(a -> a >= 2.0 || isapprox(a, 2.0; atol=1e-3), au_min_area.areas)

    # 8. Hexagonal Gridding Unaggregated Regular Polygons & Observation Preservation
    rng_hex = MersenneTwister(42)
    n_hex = 300
    df_hex = DataFrame(
        lon = rand(rng_hex, n_hex) .* 100.0,
        lat = rand(rng_hex, n_hex) .* 50.0
    )
    au_hex = TethysBayes.assign_spatial_units(df_hex;
        x = :lon,
        y = :lat,
        area_method = :hexagonal,
        target_units = 16,
        exact_units = true,
        radius = 10.0,
        merge_small_polygons = true
    )
    @test au_hex.n_units == 16
    for p in au_hex.polygons
        @test length(p) == 7  # 6 vertices + 1 closed coordinate for regular hexagon
    end
    @test length(au_hex.s_idx) == n_hex
    @test extrema(au_hex.s_idx) == (1, 16)
end
