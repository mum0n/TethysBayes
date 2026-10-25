
# not central to TethysBayes. this is here for some simple comparisons
# to use it, you must read it in manually: include( "c:/home/jae/projects/TethysBayes/src/kriging.jl" )

if false 
    import Pkg
    Pkg.add(["TableTransforms", "CategoricalArrays", "CoordRefSystems", "GeoTables", "GeoStats", "CairoMakie"])
    
    using TableTransforms, CategoricalArrays, CoordRefSystems, GeoTables, GeoStats, CairoMakie
    using LinearAlgebra
    import CairoMakie as Mke
end

"""
    compute_kriging(df; coords=nothing, var=nothing, maxlag=80.0, nlags=15, range_val=40.0)

Performs Ordinary Kriging interpolation over a regular Cartesian grid for baseline spatial
comparison. Automatically detects coordinate columns (e.g. `(:plat, :plon)`, `(:s_x, :s_y)`,
`(:lon, :lat)`) and response variable (e.g. `:z`) if not explicitly specified.
"""
function compute_kriging(
    df;
    coords::Union{Tuple{Symbol, Symbol}, Vector{Symbol}, Nothing} = nothing,
    var::Union{Symbol, AbstractString, Nothing} = nothing,
    maxlag::Real = 80.0,
    nlags::Int = 15,
    range_val::Real = 40.0
)
    coord_pair = if coords === nothing
        _detect_xy_columns(df)
    else
        (Symbol(coords[1]), Symbol(coords[2]))
    end

    target_var = if var === nothing
        _detect_response_column(df; exclude=[coord_pair[1], coord_pair[2]])
    else
        Symbol(var)
    end

    geotable = georef(df, coord_pair)

    bbox = GeoStats.boundingbox(geotable.geometry)
    c_min, c_max = coords(bbox.min), coords(bbox.max)
    origin = (c_min.x, c_min.y)

    # Cartesian grid for prediction using fixed cell size
    u = oneunit(c_min.x)
    spacing = (5 * u, 5 * u)

    # Create the grid by specifying the min, max, and spacing directly
    topright = (c_max.x, c_max.y)
    grid = CartesianGrid(origin, topright, spacing)

    # Experimental variogram from the data for the target variable
    g_empirical = variogram(geotable, target_var, maxlag=maxlag, nlags=nlags)

    # Define a theoretical model manually
    g_fitted = GaussianVariogram(range=range_val * u)
    model = Kriging(g_fitted)

    # Perform the ordinary kriging interpolation over the grid
    solution = geotable |> Interpolate(grid, model=model)

    # 1. Extract conditioning points
    n = length(geotable.geometry)
    pts = collect(geotable.geometry)

    function pdist(p1, p2)
        c1, c2 = coords(p1), coords(p2)
        sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2)
    end

    # 2. Build the Kriging variogram matrix (n+1) x (n+1)
    var_type = typeof(g_fitted(pdist(pts[1], pts[1])))
    K = zeros(var_type, n+1, n+1)
    for i in 1:n, j in 1:n
        K[i,j] = g_fitted(pdist(pts[i], pts[j]))
    end

    # Add unbiasedness constraint
    for i in 1:n
        K[i, n+1] = oneunit(var_type)
        K[n+1, i] = oneunit(var_type)
    end
    K[n+1, n+1] = zero(var_type)

    K_fact = lu(K)

    # 3. Predict variance at each grid centroid
    grid_pts = collect(centroid.(grid))
    m = length(grid_pts)
    krig_vars = zeros(var_type, m)

    for (idx, p) in enumerate(grid_pts)
        k = zeros(var_type, n+1)
        for i in 1:n
            k[i] = g_fitted(pdist(pts[i], p))
        end
        k[n+1] = oneunit(var_type)

        weights = K_fact \ k
        krig_vars[idx] = dot(weights, k)
    end

    kriging_stds = [Float64(ustrip(sqrt(max(zero(v), v)))) for v in krig_vars]
    
    return (
        geotable = geotable,
        grid = grid,
        g_empirical = g_empirical,
        g_fitted = g_fitted,
        solution = solution,
        kriging_stds = kriging_stds,
        target_var = target_var,
        coords = coord_pair
    )
end

function plot_kriging(results)
    geotable = results.geotable
    grid = results.grid
    g_empirical = results.g_empirical
    g_fitted = results.g_fitted
    solution = results.solution
    kriging_stds = results.kriging_stds
    target_var = haskey(results, :target_var) ? results.target_var : :z

    # --- FIGURE 1: Variogram Fit ---
    fig1 = Mke.Figure(size = (600, 400))
    ax1 = Mke.Axis(fig1[1, 1], title = "Variogram Fit", xlabel = "Lag Distance", ylabel = "Semivariance")
    Mke.plot!(ax1, g_empirical, color = :red, marker = :circle, label = "Empirical")
    Mke.plot!(ax1, g_fitted, maxlag=80.0, color = :blue, label = "Fitted Model")
    Mke.axislegend(ax1, position = :rb)
    display(fig1)

    # --- FIGURE 2: Prediction Spatial Map ---
    fig2 = Mke.Figure(size = (600, 450))

    # Subplot A: Kriging Predictions (Mean)
    ax_pred = Mke.Axis(fig2[1, 1], title = "Kriging Predictions (:$(target_var))")
    sol_vals = getproperty(solution, target_var)
    p1 = viz!(ax_pred, solution.geometry, color = sol_vals, colormap = :viridis)
    viz!(ax_pred, geotable.geometry, color = :black, pointsize = 10, pointmarker = :cross)
    z_extrema = extrema(sol_vals)
    Mke.Colorbar(fig2[1, 2], colormap = :viridis, limits = z_extrema)

    display(fig2)

    # --- FIGURE 3: Kriging Standard Deviation Map ---
    fig3 = Mke.Figure(size = (600, 450))
    ax_std = Mke.Axis(fig3[1, 1], title = "Kriging Standard Deviation")

    p_std = viz!(ax_std, grid, color = kriging_stds, colormap = :magma)
    viz!(ax_std, geotable.geometry, color = :white, pointsize = 10, pointmarker = :cross)

    std_extrema = extrema(kriging_stds)
    Mke.Colorbar(fig3[1, 2], colormap = :magma, limits = std_extrema)

    display(fig3)
end


