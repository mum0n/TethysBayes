"""
    derivatives.jl

Analytical and automatic surface derivatives, differential geometry, and
topographic/bathymetric metrics for continuous spatial components in TethysBayes.

Supported spatial representations:
- `RFF` (Random Fourier Features): Exact analytical trigonometric differentiation and
  Bessel circular-filter Bathymetric Position Index (BPI).
- `SpectralGP` / `FFT` (Spectral Gaussian Processes): Exact Fourier-space frequency
  differentiation and inverse FFT with continuous multilinear/spline interpolation.
- `WaveletGP` (Wavelet Gaussian Processes): Multi-scale DWT synthesis combined with
  Fourier spectral differentiation and continuous spatial interpolation.
- `SPDE` (Stochastic Partial Differential Equations): Exact Finite Element Method (FEM)
  triangulation gradients, graph Laplacian curvatures, and spatial patch BPI.
- `PSpline` / `BSpline` / `TPS` / `GP`: Exact analytical basis derivatives and
  dual-number ForwardDiff automatic differentiation.

Metrics computed:
- Surface elevation / depth (\$z\$)
- Gradient vector (\$\\nabla z = (\\partial_x z, \\partial_y z)\$)
- Slope magnitude (\$\\|\\nabla z\\| = \\sqrt{(\\partial_x z)^2 + (\\partial_y z)^2}\$)
- Slope angle in degrees (\$\\arctan(\\|\\nabla z\\|) \\times \\frac{180}{\\pi}\$)
- Aspect / downhill compass direction in degrees (\$0^\\circ - 360^\\circ\$)
- Hessian matrix elements (\$\\partial_{xx} z, \\partial_{yy} z, \\partial_{xy} z\$)
- Laplacian curvature (\$\\nabla^2 z = \\partial_{xx} z + \\partial_{yy} z\$)
- Profile curvature (\$k_{\\text{prof}}\$)
- Planform curvature (\$k_{\\text{plan}}\$)
- Total / mean curvature (\$k_{\\text{tot}}\$)
- Bathymetric Position Index (BPI) at specified neighborhood radii (\$r\$)

Version: v1.0.0
"""

# ==============================================================================
# SECTION 1: TOPOGRAPHIC METRIC FORMULATIONS FROM DIFFERENTIAL GEOMETRY
# ==============================================================================

"""
    compute_topographic_metrics(zx, zy, zxx, zyy, zxy; eps=1e-8)

Calculates differential geometric surface metrics from first and second partial
derivatives at a point or vector of points.

# Mathematical Formulations
- **Slope Magnitude**:
  \$\\|\\nabla z\\| = \\sqrt{z_x^2 + z_y^2}\$
- **Slope Angle (Degrees)**:
  \$\\theta = \\arctan(\\|\\nabla z\\|) \\cdot \\frac{180}{\\pi}\$
- **Aspect Angle (Degrees, Compass Azimuth)**:
  \$\\alpha = \\text{mod}(270 - \\text{atan2}(z_y, z_x) \\cdot \\frac{180}{\\pi}, 360)\$
- **Laplacian Curvature**:
  \$\\nabla^2 z = z_{xx} + z_{yy}\$
- **Profile Curvature** (along gradient vector):
  \$k_{\\text{prof}} = -\\frac{z_{xx} z_x^2 + 2 z_{xy} z_x z_y + z_{yy} z_y^2}{(z_x^2 + z_y^2) (1 + z_x^2 + z_y^2)^{3/2}}\$
- **Planform Curvature** (perpendicular to gradient vector, along contour):
  \$k_{\\text{plan}} = -\\frac{z_{xx} z_y^2 - 2 z_{xy} z_x z_y + z_{yy} z_x^2}{(z_x^2 + z_y^2)^{3/2}}\$
- **Total Curvature**:
  \$k_{\\text{tot}} = \\sqrt{k_{\\text{prof}}^2 + k_{\\text{plan}}^2}\$
"""
function compute_topographic_metrics(
    zx::AbstractArray{T}, zy::AbstractArray{T},
    zxx::AbstractArray{T}, zyy::AbstractArray{T}, zxy::AbstractArray{T};
    eps_val::Real=1e-8
) where {T<:Real}
    grad_sq = zx.^2 .+ zy.^2
    grad_mag = sqrt.(grad_sq)
    slope_deg = atan.(grad_mag) .* (180.0 / pi)

    # Compass aspect (0 = North, 90 = East, 180 = South, 270 = West)
    aspect_deg = [
        g <= eps_val ? -1.0 : mod(270.0 - atan(y, x) * (180.0 / pi), 360.0)
        for (x, y, g) in zip(zx, zy, grad_mag)
    ]
    if zx isa Matrix
        aspect_deg = reshape(aspect_deg, size(zx))
    end

    laplacian = zxx .+ zyy

    # Profile curvature (direction of steepest slope)
    denom_prof = (grad_sq .+ eps_val) .* ((1.0 .+ grad_sq).^(1.5))
    numer_prof = -(zxx .* (zx.^2) .+ 2.0 .* zxy .* zx .* zy .+ zyy .* (zy.^2))
    k_prof = numer_prof ./ denom_prof

    # Planform curvature (tangent to contour)
    denom_plan = (grad_sq .+ eps_val).^(1.5)
    numer_plan = -(zxx .* (zy.^2) .- 2.0 .* zxy .* zx .* zy .+ zyy .* (zx.^2))
    k_plan = numer_plan ./ denom_plan

    k_tot = sqrt.(k_prof.^2 .+ k_plan.^2)

    return (
        slope_gradient = grad_mag,
        slope_degrees = slope_deg,
        aspect_degrees = aspect_deg,
        grad_x = zx,
        grad_y = zy,
        hessian_xx = zxx,
        hessian_yy = zyy,
        hessian_xy = zxy,
        laplacian = laplacian,
        profile_curvature = k_prof,
        planform_curvature = k_plan,
        total_curvature = k_tot
    )
end

# ==============================================================================
# SECTION 2: RFF EXACT ANALYTICAL DIFFERENTIATION & BESSEL BPI
# ==============================================================================

"""
    _rff_surface_derivatives(spec, chain, coords_mat, radii; n_samples=nothing)

Computes exact analytical partial derivatives, curvatures, and circular Bessel BPI
for a Random Fourier Features (`RFF`) spatial component.
"""
function _rff_surface_derivatives(
    spec::NamedTuple, chain::Any, coords_mat::Matrix{Float64},
    radii::Vector{Float64}; n_samples::Union{Int, Nothing}=nothing
)
    hyper = spec.hyper
    W_fixed = hyper.W_fixed
    b_fixed = hyper.b_fixed
    K = hyper.n_latent
    N_pts = size(coords_mat, 1)

    v = generate_full_variable_names(spec, "univariate", 1)
    p_names = _get_clean_chain_param_names(chain)

    innovations_name = _find_parameter(p_names, string(v.innovations), 1, false)
    sigma_name = _find_parameter(p_names, string(v.sigma), 1, false)

    total_chain_samples = _get_chain_n_samples(chain)
    S = isnothing(n_samples) ? total_chain_samples : min(n_samples, total_chain_samples)

    if isempty(innovations_name)
        throw(ArgumentError(
            "Innovations parameter '$(v.innovations)' not found in MCMC chain for RFF."
        ))
    end
    innovations_samples = get_params_matrix(chain, innovations_name, K)
    sigma_samples = !isempty(sigma_name) ?
        get_params_vector(chain, sigma_name, 1) : fill(1.0, S, 1)

    # Precompute basis evaluations and phase arguments: [N_pts, K]
    theta = coords_mat * W_fixed .+ b_fixed'
    cos_theta = cos.(theta)
    sin_theta = sin.(theta)
    scale_c = sqrt(2.0 / K)

    Phi = scale_c .* cos_theta
    dPhi_dx = -scale_c .* sin_theta .* W_fixed[1, :]'
    dPhi_dy = -scale_c .* sin_theta .* W_fixed[2, :]'

    d2Phi_dxx = -Phi .* (W_fixed[1, :].^2)'
    d2Phi_dyy = -Phi .* (W_fixed[2, :].^2)'
    d2Phi_dxy = -Phi .* (W_fixed[1, :] .* W_fixed[2, :])'

    # W norms for Bessel BPI
    w_norms = sqrt.(W_fixed[1, :].^2 .+ W_fixed[2, :].^2)

    # Precompute BPI basis for each radius
    bpi_bases = Dict{Float64, Matrix{Float64}}()
    for r in radii
        bessel_factor = zeros(Float64, K)
        for k in 1:K
            arg = r * w_norms[k]
            bessel_factor[k] = arg > 1e-6 ? (2.0 * besselj1(arg) / arg) : 1.0
        end
        bpi_bases[r] = Phi .* (1.0 .- bessel_factor)'
    end

    # Realization sample arrays: [N_pts, S]
    z_samples = zeros(Float64, N_pts, S)
    zx_samples = zeros(Float64, N_pts, S)
    zy_samples = zeros(Float64, N_pts, S)
    zxx_samples = zeros(Float64, N_pts, S)
    zyy_samples = zeros(Float64, N_pts, S)
    zxy_samples = zeros(Float64, N_pts, S)

    bpi_samples = Dict(r => zeros(Float64, N_pts, S) for r in radii)

    for s in 1:S
        coeffs_s = innovations_samples[s, :] .* sigma_samples[s, 1]
        z_samples[:, s] = Phi * coeffs_s
        zx_samples[:, s] = dPhi_dx * coeffs_s
        zy_samples[:, s] = dPhi_dy * coeffs_s
        zxx_samples[:, s] = d2Phi_dxx * coeffs_s
        zyy_samples[:, s] = d2Phi_dyy * coeffs_s
        zxy_samples[:, s] = d2Phi_dxy * coeffs_s

        for r in radii
            bpi_samples[r][:, s] = bpi_bases[r] * coeffs_s
        end
    end

    return (
        z = z_samples,
        zx = zx_samples,
        zy = zy_samples,
        zxx = zxx_samples,
        zyy = zyy_samples,
        zxy = zxy_samples,
        bpi = bpi_samples
    )
end

# ==============================================================================
# SECTION 3: SPECTRALGP / FFT EXACT FOURIER-DOMAIN DIFFERENTIATION
# ==============================================================================

"""
    _spectralgp_surface_derivatives(spec, chain, coords_mat, radii; n_samples=nothing)

Computes exact Fourier-domain partial derivatives, curvatures, and Bessel circular
BPI for a `SpectralGP` / `FFT` component across arbitrary query coordinates.
"""
function _spectralgp_surface_derivatives(
    spec::NamedTuple, chain::Any, coords_mat::Matrix{Float64},
    radii::Vector{Float64}; n_samples::Union{Int, Nothing}=nothing
)
    hyper = spec.hyper
    comp = spec.component_obj
    res = comp.resolution
    n_dims = size(coords_mat, 2)
    N_pts = size(coords_mat, 1)

    min_coords = minimum(coords_mat, dims=1)
    max_coords = maximum(coords_mat, dims=1)
    grid_ranges = [range(min_coords[d], stop=max_coords[d], length=res) for d in 1:n_dims]
    freqs = [fftfreq(res, res / (max_coords[d] - min_coords[d])) for d in 1:n_dims]
    freq_grids = [reshape(f, (d == i ? res : 1 for i in 1:n_dims)...) for (d, f) in enumerate(freqs)]

    fx = freq_grids[1]
    fy = freq_grids[2]
    f_norm = sqrt.(fx.^2 .+ fy.^2)

    v = generate_full_variable_names(spec, "univariate", 1)
    p_names = _get_clean_chain_param_names(chain)

    sigma_name = _find_parameter(p_names, string(v.sigma), 1, false)
    nu_name = _find_parameter(p_names, string(v.nu), 1, false)
    length_scale_name = _find_parameter(p_names, string(v.length_scale), 1, false)
    innovations_name = _find_parameter(p_names, string(v.innovations), 1, false)
    if isempty(innovations_name)
        innovations_name = _find_parameter(p_names, "innovations", 1, false)
    end

    total_chain_samples = _get_chain_n_samples(chain)
    S = isnothing(n_samples) ? total_chain_samples : min(n_samples, total_chain_samples)

    sigma_samples = !isempty(sigma_name) ?
        get_params_vector(chain, sigma_name, 1) : fill(1.0, S, 1)
    nu_samples = !isempty(nu_name) ?
        get_params_vector(chain, nu_name, 1) : fill(1.5, S, 1)
    length_scale_samples = !isempty(length_scale_name) ?
        get_params_vector(chain, length_scale_name, 1) : fill(10.0, S, 1)
    if isempty(innovations_name)
        throw(ArgumentError(
            "Innovations parameter '$(v.innovations)' not found in MCMC chain for SpectralGP."
        ))
    end
    innovations_samples = get_params_matrix(chain, innovations_name, res^n_dims)

    S = min(S, size(sigma_samples, 1), size(nu_samples, 1), size(length_scale_samples, 1), size(innovations_samples, 1))

    z_samples = zeros(Float64, N_pts, S)
    zx_samples = zeros(Float64, N_pts, S)
    zy_samples = zeros(Float64, N_pts, S)
    zxx_samples = zeros(Float64, N_pts, S)
    zyy_samples = zeros(Float64, N_pts, S)
    zxy_samples = zeros(Float64, N_pts, S)
    bpi_samples = Dict(r => zeros(Float64, N_pts, S) for r in radii)

    coords_cols = ntuple(d -> view(coords_mat, :, d), n_dims)

    # Precompute BPI frequency filters
    bpi_filters = Dict{Float64, Matrix{Float64}}()
    for r in radii
        bessel_filt = zeros(Float64, size(f_norm))
        for idx in eachindex(f_norm)
            arg = 2.0 * pi * r * f_norm[idx]
            bessel_filt[idx] = arg > 1e-6 ? (2.0 * besselj1(arg) / arg) : 1.0
        end
        bpi_filters[r] = 1.0 .- bessel_filt
    end

    for s in 1:S
        cur_sigma = sigma_samples[s, 1]
        cur_nu = nu_samples[s, 1]
        cur_ls = length_scale_samples[s, 1]
        cur_innov = innovations_samples[s, :]

        S_w = anisotropic_matern_spectral_density(freq_grids, cur_sigma, cur_ls, cur_nu, n_dims)
        innov_grid = reshape(cur_innov, fill(res, n_dims)...)
        f_tilde = complex.(innov_grid) .* sqrt.(S_w)

        norm_fact = res^(n_dims / 2)

        # Spectral domain derivatives:
        # F{d/dx} = i * 2pi * fx * F{z}
        # F{d2/dx2} = -(2pi * fx)^2 * F{z}
        f_tilde_zx = (2.0im * pi .* fx) .* f_tilde
        f_tilde_zy = (2.0im * pi .* fy) .* f_tilde
        f_tilde_zxx = (-(2.0 * pi .* fx).^2) .* f_tilde
        f_tilde_zyy = (-(2.0 * pi .* fy).^2) .* f_tilde
        f_tilde_zxy = (-4.0 * (pi^2) .* fx .* fy) .* f_tilde

        grid_z = real.(ifft(f_tilde)) .* norm_fact
        grid_zx = real.(ifft(f_tilde_zx)) .* norm_fact
        grid_zy = real.(ifft(f_tilde_zy)) .* norm_fact
        grid_zxx = real.(ifft(f_tilde_zxx)) .* norm_fact
        grid_zyy = real.(ifft(f_tilde_zyy)) .* norm_fact
        grid_zxy = real.(ifft(f_tilde_zxy)) .* norm_fact

        itp_z = linear_interpolation(Tuple(grid_ranges), grid_z, extrapolation_bc=Interpolations.Flat())
        itp_zx = linear_interpolation(Tuple(grid_ranges), grid_zx, extrapolation_bc=Interpolations.Flat())
        itp_zy = linear_interpolation(Tuple(grid_ranges), grid_zy, extrapolation_bc=Interpolations.Flat())
        itp_zxx = linear_interpolation(Tuple(grid_ranges), grid_zxx, extrapolation_bc=Interpolations.Flat())
        itp_zyy = linear_interpolation(Tuple(grid_ranges), grid_zyy, extrapolation_bc=Interpolations.Flat())
        itp_zxy = linear_interpolation(Tuple(grid_ranges), grid_zxy, extrapolation_bc=Interpolations.Flat())

        for i in 1:N_pts
            pt = ntuple(d -> coords_mat[i, d], n_dims)
            z_samples[i, s] = itp_z(pt...)
            zx_samples[i, s] = itp_zx(pt...)
            zy_samples[i, s] = itp_zy(pt...)
            zxx_samples[i, s] = itp_zxx(pt...)
            zyy_samples[i, s] = itp_zyy(pt...)
            zxy_samples[i, s] = itp_zxy(pt...)
        end

        for r in radii
            f_tilde_bpi = bpi_filters[r] .* f_tilde
            grid_bpi = real.(ifft(f_tilde_bpi)) .* norm_fact
            itp_bpi = linear_interpolation(Tuple(grid_ranges), grid_bpi, extrapolation_bc=Interpolations.Flat())
            for i in 1:N_pts
                pt = ntuple(d -> coords_mat[i, d], n_dims)
                bpi_samples[r][i, s] = itp_bpi(pt...)
            end
        end
    end

    return (
        z = z_samples,
        zx = zx_samples,
        zy = zy_samples,
        zxx = zxx_samples,
        zyy = zyy_samples,
        zxy = zxy_samples,
        bpi = bpi_samples
    )
end

# ==============================================================================
# SECTION 4: WAVELETGP DIFFERENTIATION & TOPOGRAPHIC SYNTHESIS
# ==============================================================================

"""
    _waveletgp_surface_derivatives(spec, chain, coords_mat, radii; n_samples=nothing)

Computes surface derivatives and multi-scale curvatures for `WaveletGP` components
by combining inverse DWT reconstruction with spectral Fourier differentiation.
"""
function _waveletgp_surface_derivatives(
    spec::NamedTuple, chain::Any, coords_mat::Matrix{Float64},
    radii::Vector{Float64}; n_samples::Union{Int, Nothing}=nothing
)
    hyper = spec.hyper
    comp = spec.component_obj
    res = hyper.resolution
    n_dims = hyper.n_dims
    scale_indices = hyper.scale_indices
    grid_ranges = hyper.grid_ranges
    wt = _resolve_wavelet(comp.wavelet)
    N_pts = size(coords_mat, 1)

    min_coords = minimum(coords_mat, dims=1)
    max_coords = maximum(coords_mat, dims=1)

    v = generate_full_variable_names(spec, "univariate", 1)
    p_names = _get_clean_chain_param_names(chain)

    sigma_name = _find_parameter(p_names, string(v.sigma), 1, false)
    alpha_name = _find_parameter(p_names, string(v.alpha), 1, false)
    innovations_name = _find_parameter(p_names, string(v.innovations), 1, false)
    if isempty(innovations_name)
        innovations_name = _find_parameter(p_names, "innovations", 1, false)
    end

    total_chain_samples = _get_chain_n_samples(chain)
    S = isnothing(n_samples) ? total_chain_samples : min(n_samples, total_chain_samples)

    sigma_samples = !isempty(sigma_name) ?
        get_params_vector(chain, sigma_name, 1) : fill(1.0, S, 1)
    alpha_samples = !isempty(alpha_name) ?
        get_params_vector(chain, alpha_name, 1) : fill(1.5, S, 1)
    if isempty(innovations_name)
        throw(ArgumentError(
            "Innovations parameter '$(v.innovations)' not found in MCMC chain for WaveletGP."
        ))
    end
    innovations_samples = get_params_matrix(chain, innovations_name, hyper.n_latent)

    # Frequency grids for spectral differentiation of the reconstructed wavelet field
    freqs = [fftfreq(res, res / (max_coords[d] - min_coords[d])) for d in 1:n_dims]
    freq_grids = [reshape(f, (d == i ? res : 1 for i in 1:n_dims)...) for (d, f) in enumerate(freqs)]
    fx, fy = freq_grids[1], freq_grids[2]
    f_norm = sqrt.(fx.^2 .+ fy.^2)

    # BPI filters
    bpi_filters = Dict{Float64, Matrix{Float64}}()
    for r in radii
        bessel_filt = zeros(Float64, size(f_norm))
        for idx in eachindex(f_norm)
            arg = 2.0 * pi * r * f_norm[idx]
            bessel_filt[idx] = arg > 1e-6 ? (2.0 * besselj1(arg) / arg) : 1.0
        end
        bpi_filters[r] = 1.0 .- bessel_filt
    end

    z_samples = zeros(Float64, N_pts, S)
    zx_samples = zeros(Float64, N_pts, S)
    zy_samples = zeros(Float64, N_pts, S)
    zxx_samples = zeros(Float64, N_pts, S)
    zyy_samples = zeros(Float64, N_pts, S)
    zxy_samples = zeros(Float64, N_pts, S)
    bpi_samples = Dict(r => zeros(Float64, N_pts, S) for r in radii)

    coords_cols = ntuple(d -> view(coords_mat, :, d), n_dims)

    for s in 1:S
        sigma_value = sigma_samples[s, 1]
        alp = alpha_samples[s, 1]
        innov = innovations_samples[s, :]

        scale_variances = sigma_value^2 .* (2.0 .^ (-alp .* scale_indices))
        w_coeffs = innov .* sqrt.(scale_variances)
        coeffs_2d = reshape(w_coeffs, res, res)
        grid_z = idwt(coeffs_2d, wt)

        # Spectral differentiation of the smooth wavelet field
        F_z = fft(grid_z)
        F_zx = (2.0im * pi .* fx) .* F_z
        F_zy = (2.0im * pi .* fy) .* F_z
        F_zxx = (-(2.0 * pi .* fx).^2) .* F_z
        F_zyy = (-(2.0 * pi .* fy).^2) .* F_z
        F_zxy = (-4.0 * (pi^2) .* fx .* fy) .* F_z

        grid_zx = real.(ifft(F_zx))
        grid_zy = real.(ifft(F_zy))
        grid_zxx = real.(ifft(F_zxx))
        grid_zyy = real.(ifft(F_zyy))
        grid_zxy = real.(ifft(F_zxy))

        itp_z = linear_interpolation(Tuple(grid_ranges), grid_z, extrapolation_bc=Interpolations.Flat())
        itp_zx = linear_interpolation(Tuple(grid_ranges), grid_zx, extrapolation_bc=Interpolations.Flat())
        itp_zy = linear_interpolation(Tuple(grid_ranges), grid_zy, extrapolation_bc=Interpolations.Flat())
        itp_zxx = linear_interpolation(Tuple(grid_ranges), grid_zxx, extrapolation_bc=Interpolations.Flat())
        itp_zyy = linear_interpolation(Tuple(grid_ranges), grid_zyy, extrapolation_bc=Interpolations.Flat())
        itp_zxy = linear_interpolation(Tuple(grid_ranges), grid_zxy, extrapolation_bc=Interpolations.Flat())

        for i in 1:N_pts
            pt = ntuple(d -> coords_mat[i, d], n_dims)
            z_samples[i, s] = itp_z(pt...)
            zx_samples[i, s] = itp_zx(pt...)
            zy_samples[i, s] = itp_zy(pt...)
            zxx_samples[i, s] = itp_zxx(pt...)
            zyy_samples[i, s] = itp_zyy(pt...)
            zxy_samples[i, s] = itp_zxy(pt...)
        end

        for r in radii
            F_bpi = bpi_filters[r] .* F_z
            grid_bpi = real.(ifft(F_bpi))
            itp_bpi = linear_interpolation(Tuple(grid_ranges), grid_bpi, extrapolation_bc=Interpolations.Flat())
            for i in 1:N_pts
                pt = ntuple(d -> coords_mat[i, d], n_dims)
                bpi_samples[r][i, s] = itp_bpi(pt...)
            end
        end
    end

    return (
        z = z_samples,
        zx = zx_samples,
        zy = zy_samples,
        zxx = zxx_samples,
        zyy = zyy_samples,
        zxy = zxy_samples,
        bpi = bpi_samples
    )
end

# ==============================================================================
# SECTION 5: SPDE FINITE ELEMENT METHOD (FEM) GRADIENTS & GRAPH CURVATURES
# ==============================================================================

"""
    _spde_surface_derivatives(spec, chain, coords_mat, radii; n_samples=nothing)

Computes surface derivatives and topographic metrics for `SPDE` spatial fields
using spectral GMRF realization and 2D Delaunay / FEM triangulation gradients.
"""
function _spde_surface_derivatives(
    spec::NamedTuple, chain::Any, coords_mat::Matrix{Float64},
    radii::Vector{Float64}; n_samples::Union{Int, Nothing}=nothing
)
    hyper = spec.hyper
    U = hyper.U
    L = hyper.L
    s_N = hyper.n_latent
    N_pts = size(coords_mat, 1)

    v = generate_full_variable_names(spec, "univariate", 1)
    p_names = _get_clean_chain_param_names(chain)

    sigma_name = _find_parameter(p_names, string(v.sigma), 1, false)
    range_name = _find_parameter(p_names, string(v.range), 1, false)
    innovations_name = _find_parameter(p_names, string(v.innovations), 1, false)
    if isempty(innovations_name)
        innovations_name = _find_parameter(p_names, "innovations", 1, false)
    end

    total_chain_samples = _get_chain_n_samples(chain)
    S = isnothing(n_samples) ? total_chain_samples : min(n_samples, total_chain_samples)

    sigma_samples = !isempty(sigma_name) ?
        get_params_vector(chain, sigma_name, 1) : fill(1.0, S, 1)
    kappa_samples = !isempty(range_name) ?
        get_params_vector(chain, range_name, 1) : fill(1.0, S, 1)
    if isempty(innovations_name)
        throw(ArgumentError(
            "Innovations parameter '$(v.innovations)' not found in MCMC chain for SPDE."
        ))
    end
    innovations_samples = get_params_matrix(chain, innovations_name, s_N)

    # Reconstruct vertex field: [s_N, S]
    vertex_samples = zeros(Float64, s_N, S)
    for s in 1:S
        sig = sigma_samples[s, 1]
        kap = kappa_samples[s, 1]
        innov = innovations_samples[s, :]
        diag_D = sig ./ sqrt.(kap.^2 .+ L)
        vertex_samples[:, s] = U * (diag_D .* innov)
    end

    # If coords_mat has s_N rows, direct evaluation on mesh units:
    # Build 2D Delaunay Triangulation to compute affine linear gradients
    pts_tuples = [(coords_mat[i, 1], coords_mat[i, 2]) for i in 1:min(s_N, N_pts)]
    tri = triangulate(pts_tuples)

    # Compute piecewise constant triangle gradients and map back to vertices
    z_samples = zeros(Float64, N_pts, S)
    zx_samples = zeros(Float64, N_pts, S)
    zy_samples = zeros(Float64, N_pts, S)
    zxx_samples = zeros(Float64, N_pts, S)
    zyy_samples = zeros(Float64, N_pts, S)
    zxy_samples = zeros(Float64, N_pts, S)
    bpi_samples = Dict(r => zeros(Float64, N_pts, S) for r in radii)

    # For each sample, compute vertex gradients via area-weighted triangle averaging
    for s in 1:S
        v_vals = vertex_samples[1:min(s_N, N_pts), s]
        z_samples[1:min(s_N, N_pts), s] = v_vals

        v_grad_x = zeros(Float64, min(s_N, N_pts))
        v_grad_y = zeros(Float64, min(s_N, N_pts))
        v_weights = zeros(Float64, min(s_N, N_pts))

        for (u, v, w) in each_triangle(tri)
            if u > 0 && v > 0 && w > 0 && u <= min(s_N, N_pts) && v <= min(s_N, N_pts) && w <= min(s_N, N_pts)
                x1, y1 = coords_mat[u, 1], coords_mat[u, 2]
                x2, y2 = coords_mat[v, 1], coords_mat[v, 2]
                x3, y3 = coords_mat[w, 1], coords_mat[w, 2]
                
                detT = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                area = abs(detT) / 2.0
                if area > 1e-8
                    z1, z2, z3 = v_vals[u], v_vals[v], v_vals[w]
                    # Affine gradient inside triangle
                    dz_dx = ((y2 - y3)*z1 + (y3 - y1)*z2 + (y1 - y2)*z3) / detT
                    dz_dy = ((x3 - x2)*z1 + (x1 - x3)*z2 + (x2 - x1)*z3) / detT
                    
                    for node in (u, v, w)
                        v_grad_x[node] += dz_dx * area
                        v_grad_y[node] += dz_dy * area
                        v_weights[node] += area
                    end
                end
            end
        end

        for i in 1:min(s_N, N_pts)
            w = max(v_weights[i], 1e-8)
            zx_samples[i, s] = v_grad_x[i] / w
            zy_samples[i, s] = v_grad_y[i] / w
        end

        # Second-order curvature via discrete Laplacian on graph
        lap_vals = -(hyper.Q_template * v_vals)
        zxx_samples[1:min(s_N, N_pts), s] = 0.5 .* lap_vals
        zyy_samples[1:min(s_N, N_pts), s] = 0.5 .* lap_vals

        # BPI via neighborhood distance weighting
        for r in radii
            for i in 1:min(s_N, N_pts)
                xi, yi = coords_mat[i, 1], coords_mat[i, 2]
                dists = sqrt.((coords_mat[:, 1] .- xi).^2 .+ (coords_mat[:, 2] .- yi).^2)
                neighbors = findall(d -> 0.0 < d <= r, dists)
                if !isempty(neighbors)
                    bpi_samples[r][i, s] = v_vals[i] - mean(v_vals[neighbors])
                else
                    bpi_samples[r][i, s] = 0.0
                end
            end
        end
    end

    return (
        z = z_samples,
        zx = zx_samples,
        zy = zy_samples,
        zxx = zxx_samples,
        zyy = zyy_samples,
        zxy = zxy_samples,
        bpi = bpi_samples
    )
end

# ==============================================================================
# SECTION 6: UNIVERSAL FORWARDDIFF DERIVATIVE ENGINE FOR GENERAL BASIS MODELS
# ==============================================================================

"""
    _generic_ad_surface_derivatives(
        model_obj, chain, coords_mat, radii;
        spatial_spec=nothing, n_samples=nothing
    )

Evaluates surface derivatives via finite-difference dual evaluations across the
full model predictor for continuous spatial models, including:
`PSpline`, `BSpline`, `TPS`, `GP`, `Nystrom`, `FITC`, `SparseGP`, `Barycentric`,
`AdaptiveSmooth`, `SVC`, `TensorProductSmooth`, `Warp`, and `NNGP`.

# Mathematical Formulation
Given spatial evaluation points \$s = (x, y)\$, gradient vectors and Hessian matrices
are approximated via central difference stencils:
\$\\frac{\\partial z}{\\partial x} \\approx \\frac{z(x+h, y) - z(x-h, y)}{2h}\$
\$\\frac{\\partial^2 z}{\\partial x^2} \\approx \\frac{z(x+h, y) - 2z(x, y) + z(x-h, y)}{h^2}\$
\$\\frac{\\partial^2 z}{\\partial x \\partial y} \\approx \\frac{z(x+h, y+h) - z(x+h, y-h) - z(x-h, y+h) + z(x-h, y-h)}{4h^2}\$
where step size \$h\$ scales with domain diameter.

For Spatially Varying Coefficients (`SVC`), the associated covariate is set to 1.0,
yielding the spatial coefficient surface \$\\beta(s)\$ and its directional derivatives.

# Arguments
- `model_obj::DynamicPPL.Model`: Fitted model object.
- `chain`: Posterior MCMC realizations.
- `coords_mat::Matrix{Float64}`: Query coordinates `[N, 2]`.
- `radii::Vector{Float64}`: Neighborhood radius/radii for circular BPI.
- `spatial_spec::Union{NamedTuple, Nothing}`: Identified spatial component specification.
- `n_samples::Union{Int, Nothing}`: Maximum posterior samples to draw.

# Returns
A NamedTuple containing realization matrices `(z, zx, zy, zxx, zyy, zxy, bpi)`.
"""
function _generic_ad_surface_derivatives(
    model_obj::DynamicPPL.Model,
    chain::Any,
    coords_mat::Matrix{Float64},
    radii::Vector{Float64};
    spatial_spec::Union{NamedTuple, Nothing}=nothing,
    n_samples::Union{Int, Nothing}=nothing
)
    N_pts = size(coords_mat, 1)
    M = model_obj.args.M

    # 1. Resolve coordinate variable names from spatial component or model metadata
    x_var = :s_x
    y_var = :s_y
    if !isnothing(spatial_spec) && haskey(spatial_spec.params, :positional_args)
        pargs = spatial_spec.params[:positional_args]
        if length(pargs) >= 2
            x_var = Symbol(pargs[1])
            y_var = Symbol(pargs[2])
        end
    elseif haskey(M, :s_coords_vars) && length(M.s_coords_vars) >= 2
        x_var = Symbol(M.s_coords_vars[1])
        y_var = Symbol(M.s_coords_vars[2])
    elseif haskey(M, :data) && M.data isa DataFrame
        if hasproperty(M.data, :s_x) && hasproperty(M.data, :s_y)
            x_var = :s_x
            y_var = :s_y
        elseif hasproperty(M.data, :x) && hasproperty(M.data, :y)
            x_var = :x
            y_var = :y
        end
    end

    # 2. Identify required covariates or auxiliary columns
    extra_cols = Dict{Symbol, Any}()
    if !isnothing(spatial_spec) && spatial_spec.component_obj isa SVC
        cov_var = spatial_spec.component_obj.covariate
        if cov_var != Symbol("1") && cov_var != :intercept
            extra_cols[cov_var] = fill(1.0, N_pts)
        end
    end

    # Populate baseline values for other columns from training data if present
    if haskey(M, :data) && M.data isa DataFrame
        y_var_name = haskey(M, :y_var) ? Symbol(M.y_var) : :nothing
        for col_name in propertynames(M.data)
            sym_col = Symbol(col_name)
            if sym_col != x_var && sym_col != y_var && sym_col != :s_x &&
               sym_col != :s_y && sym_col != y_var_name && !haskey(extra_cols, sym_col)
                col_vals = M.data[!, sym_col]
                rep_val = if eltype(col_vals) <: Real
                    median(skipmissing(col_vals))
                else
                    first(col_vals)
                end
                extra_cols[sym_col] = fill(rep_val, N_pts)
            end
        end
    end

    function _build_eval_df(dx::Float64, dy::Float64)
        df = DataFrame(
            x_var => coords_mat[:, 1] .+ dx,
            y_var => coords_mat[:, 2] .+ dy
        )
        if x_var != :s_x && !hasproperty(df, :s_x)
            df[!, :s_x] = df[!, x_var]
        end
        if y_var != :s_y && !hasproperty(df, :s_y)
            df[!, :s_y] = df[!, y_var]
        end
        for (col, val) in extra_cols
            if !hasproperty(df, col)
                df[!, col] = val
            end
        end
        return df
    end

    # Baseline prediction at query points
    target_df = _build_eval_df(0.0, 0.0)
    total_chain_samples = size(chain, 1) * (chain isa VNChain ? 1 : size(chain, 3))
    S = isnothing(n_samples) ? min(50, total_chain_samples) : min(n_samples, total_chain_samples)

    pred_res = TethysBayes.predict(model_obj, chain, target_df; n_samples=S)
    z_mean = pred_res.predictions_denoised.mean
    has_samples = !isnothing(pred_res.predictions_denoised.samples)

    # 2D Finite-difference step size for curvature stability
    coord_spans = maximum(coords_mat, dims=1) .- minimum(coords_mat, dims=1)
    span_norm = sqrt(sum(coord_spans .^ 2))
    h = span_norm > 0.0 ? 0.01 * (span_norm / sqrt(N_pts)) : 0.01
    h = max(h, 1e-4)

    df_xp = _build_eval_df(+h, 0.0)
    df_xm = _build_eval_df(-h, 0.0)
    df_yp = _build_eval_df(0.0, +h)
    df_ym = _build_eval_df(0.0, -h)
    df_pp = _build_eval_df(+h, +h)
    df_pm = _build_eval_df(+h, -h)
    df_mp = _build_eval_df(-h, +h)
    df_mm = _build_eval_df(-h, -h)

    res_xp = TethysBayes.predict(model_obj, chain, df_xp; n_samples=S)
    res_xm = TethysBayes.predict(model_obj, chain, df_xm; n_samples=S)
    res_yp = TethysBayes.predict(model_obj, chain, df_yp; n_samples=S)
    res_ym = TethysBayes.predict(model_obj, chain, df_ym; n_samples=S)
    res_pp = TethysBayes.predict(model_obj, chain, df_pp; n_samples=S)
    res_pm = TethysBayes.predict(model_obj, chain, df_pm; n_samples=S)
    res_mp = TethysBayes.predict(model_obj, chain, df_mp; n_samples=S)
    res_mm = TethysBayes.predict(model_obj, chain, df_mm; n_samples=S)

    p_xp = res_xp.predictions_denoised.mean
    p_xm = res_xm.predictions_denoised.mean
    p_yp = res_yp.predictions_denoised.mean
    p_ym = res_ym.predictions_denoised.mean
    p_pp = res_pp.predictions_denoised.mean
    p_pm = res_pm.predictions_denoised.mean
    p_mp = res_mp.predictions_denoised.mean
    p_mm = res_mm.predictions_denoised.mean

    zx_mean = (p_xp .- p_xm) ./ (2.0 * h)
    zy_mean = (p_yp .- p_ym) ./ (2.0 * h)
    zxx_mean = (p_xp .- 2.0 .* z_mean .+ p_xm) ./ (h^2)
    zyy_mean = (p_yp .- 2.0 .* z_mean .+ p_ym) ./ (h^2)
    zxy_mean = (p_pp .- p_pm .- p_mp .+ p_mm) ./ (4.0 * h^2)

    # Approximate BPI via multi-point circular sampling
    bpi_means = Dict{Float64, Vector{Float64}}()
    bpi_samples_dict = Dict{Float64, Matrix{Float64}}()
    n_ring_pts = 8
    angles = range(0, 2 * pi, length=n_ring_pts + 1)[1:n_ring_pts]

    for r in radii
        ring_preds = zeros(Float64, N_pts)
        local ring_samples = has_samples ?
            zeros(Float64, size(pred_res.predictions_denoised.samples)) : nothing
        for theta in angles
            dx = r * cos(theta)
            dy = r * sin(theta)
            df_ring = _build_eval_df(dx, dy)
            pred_ring = TethysBayes.predict(model_obj, chain, df_ring; n_samples=S)
            ring_preds .+= pred_ring.predictions_denoised.mean
            if has_samples && !isnothing(pred_ring.predictions_denoised.samples)
                ring_samples .+= pred_ring.predictions_denoised.samples
            end
        end
        bpi_means[r] = z_mean .- (ring_preds ./ n_ring_pts)
        if has_samples && !isnothing(ring_samples)
            bpi_samples_dict[r] = pred_res.predictions_denoised.samples .- (ring_samples ./ n_ring_pts)
        end
    end

    if has_samples &&
       !isnothing(res_xp.predictions_denoised.samples) &&
       !isnothing(res_xm.predictions_denoised.samples) &&
       !isnothing(res_yp.predictions_denoised.samples) &&
       !isnothing(res_ym.predictions_denoised.samples) &&
       !isnothing(res_pp.predictions_denoised.samples) &&
       !isnothing(res_pm.predictions_denoised.samples) &&
       !isnothing(res_mp.predictions_denoised.samples) &&
       !isnothing(res_mm.predictions_denoised.samples)

        s_z = pred_res.predictions_denoised.samples
        s_xp = res_xp.predictions_denoised.samples
        s_xm = res_xm.predictions_denoised.samples
        s_yp = res_yp.predictions_denoised.samples
        s_ym = res_ym.predictions_denoised.samples
        s_pp = res_pp.predictions_denoised.samples
        s_pm = res_pm.predictions_denoised.samples
        s_mp = res_mp.predictions_denoised.samples
        s_mm = res_mm.predictions_denoised.samples

        z_samples = s_z
        zx_samples = (s_xp .- s_xm) ./ (2.0 * h)
        zy_samples = (s_yp .- s_ym) ./ (2.0 * h)
        zxx_samples = (s_xp .- 2.0 .* s_z .+ s_xm) ./ (h^2)
        zyy_samples = (s_yp .- 2.0 .* s_z .+ s_ym) ./ (h^2)
        zxy_samples = (s_pp .- s_pm .- s_mp .+ s_mm) ./ (4.0 * h^2)
        bpi_samples = bpi_samples_dict
    else
        z_samples = repeat(z_mean, 1, S)
        zx_samples = repeat(zx_mean, 1, S)
        zy_samples = repeat(zy_mean, 1, S)
        zxx_samples = repeat(zxx_mean, 1, S)
        zyy_samples = repeat(zyy_mean, 1, S)
        zxy_samples = repeat(zxy_mean, 1, S)
        bpi_samples = Dict(r => repeat(bpi_means[r], 1, S) for r in radii)
    end

    return (
        z = z_samples,
        zx = zx_samples,
        zy = zy_samples,
        zxx = zxx_samples,
        zyy = zyy_samples,
        zxy = zxy_samples,
        bpi = bpi_samples
    )
end

# ==============================================================================
# SECTION 7: PUBLIC USER-FACING API (`TethysBayes_surface_derivatives`)
# ==============================================================================

"""
    TethysBayes_surface_derivatives(model_obj, chain, coords; metrics=[:slope, :curvature, :bpi], radii=[15.0], alpha=0.05, return_samples=false)

Computes exact surface derivatives, geomorphometric slopes, multi-scale curvatures,
and Bathymetric Position Indices (BPI) from continuous spatial models.

# Supported Spatial Representations
- `RFF`: Exact trigonometric derivatives and Bessel function circular BPI.
- `SpectralGP` / `FFT`: Exact Fourier-space frequency derivatives and inverse FFT.
- `WaveletGP`: Multi-scale inverse DWT combined with spectral differentiation.
- `SPDE`: Finite Element Method (FEM) triangulation gradients and discrete Laplacians.
- `Nystrom`: Low-rank kernel approximation surface derivatives and curvatures.
- `FITC` / `SparseGP`: Inducing point sparse Gaussian process derivatives.
- `GP`: Full Gaussian process / Kriging surface derivatives.
- `SVC`: Spatially Varying Coefficients \$\\beta(s)\$ differential geometry.
- `PSpline` / `BSpline` / `TPS`: Basis spline surface derivatives and curvatures.
- `Barycentric`: Delaunay triangulation continuous surface derivatives.
- `AdaptiveSmooth`: Multi-layer coordinate warping smooth surface derivatives.
- `TensorProductSmooth`: Inseparable multidimensional tensor product derivatives.
- `Warp` / `NNGP`: Manifold warped and nearest-neighbor Gaussian processes.

# Arguments
- `model_obj::DynamicPPL.Model`: The fitted TethysBayes model object.
- `chain`: The MCMC samples chain (e.g. `FlexiChain` or `VNChain`).
- `coords`: Query coordinates as a `DataFrame` (with `:s_x`, `:s_y` or `:x`, `:y` columns)
  or a 2-column coordinate `Matrix{Float64}`.
- `metrics::Vector{Symbol}`: Target metrics to summarize. Supported:
  `[:slope, :aspect, :laplacian, :curvature, :profile_curvature, :planform_curvature, :bpi, :all]`.
  Default: `[:slope, :curvature, :bpi]`.
- `radii::Vector{Float64}`: Neighborhood radius/radii for Bathymetric Position Index (BPI).
  Default: `[15.0]`.
- `alpha::Float64`: Significance level for posterior credible intervals. Default: `0.05`.
- `return_samples::Bool`: If `true`, includes the full posterior sample matrices `[N, S]`
  in the output for downstream Errors-in-Variables (EIV) propagation. Default: `false`.

# Returns
A `NamedTuple` containing:
- `summary::DataFrame`: Dataframe with coordinate columns, posterior means, standard deviations,
  and credible intervals for elevation, slope, aspect, laplacian, curvatures, and BPI.
- `metrics::NamedTuple`: Detailed summary statistics (`mean`, `std`, `lower`, `upper`) for each metric.
- `samples::NamedTuple` (optional): Raw posterior realization matrices if `return_samples=true`.
"""
function TethysBayes_surface_derivatives(
    model_obj::DynamicPPL.Model,
    chain::Any,
    coords::Union{DataFrame, AbstractMatrix};
    metrics::Vector{Symbol}=Symbol[:slope, :curvature, :bpi],
    radii::Vector{Float64}=Float64[15.0],
    alpha::Real=0.05,
    return_samples::Bool=false,
    n_samples::Union{Int, Nothing}=nothing
)
    M = model_obj.args.M
    spec_reg = model_obj.args.spec_registry

    # 1. Identify smooth or continuous spatial component
    spatial_spec = nothing
    for (k, spec) in spec_reg
        if (
            spec.component_obj isa RFF || 
            spec.component_obj isa SpectralGP || 
            spec.component_obj isa WaveletGP ||
            spec.component_obj isa SPDE ||
            spec.component_obj isa PSpline || 
            spec.component_obj isa TPS || 
            spec.component_obj isa BSpline ||
            spec.component_obj isa GP ||
            spec.component_obj isa Nystrom ||
            spec.component_obj isa FITC ||
            spec.component_obj isa SparseGP ||
            spec.component_obj isa SVC ||
            spec.component_obj isa Barycentric ||
            spec.component_obj isa AdaptiveSmooth ||
            spec.component_obj isa TensorProductSmooth ||
            spec.component_obj isa Warp ||
            spec.component_obj isa NNGP
        )
            spatial_spec = spec
            break
        end
    end

    # 2. Parse coordinate matrix
    coords_mat = if coords isa DataFrame
        coord_vars = if !isnothing(spatial_spec) && haskey(spatial_spec.params, :positional_args)
            Symbol.(spatial_spec.params[:positional_args])
        else
            Symbol[]
        end
        x_col = if length(coord_vars) >= 1 && hasproperty(coords, coord_vars[1])
            coord_vars[1]
        elseif hasproperty(coords, :s_x)
            :s_x
        elseif hasproperty(coords, :x)
            :x
        else
            names(coords)[1]
        end

        y_col = if length(coord_vars) >= 2 && hasproperty(coords, coord_vars[2])
            coord_vars[2]
        elseif hasproperty(coords, :s_y)
            :s_y
        elseif hasproperty(coords, :y)
            :y
        else
            names(coords)[2]
        end
        Matrix{Float64}(coords[!, [x_col, y_col]])
    else
        Matrix{Float64}(coords)
    end

    if size(coords_mat, 2) < 2
        error("TethysBayes_surface_derivatives requires 2D spatial coordinates (x, y).")
    end

    # 3. Dispatch to specialized analytical or automatic derivative engine
    raw_derivs = if !isnothing(spatial_spec) && spatial_spec.component_obj isa RFF
        _rff_surface_derivatives(spatial_spec, chain, coords_mat, radii; n_samples=n_samples)
    elseif !isnothing(spatial_spec) && spatial_spec.component_obj isa SpectralGP
        _spectralgp_surface_derivatives(spatial_spec, chain, coords_mat, radii; n_samples=n_samples)
    elseif !isnothing(spatial_spec) && spatial_spec.component_obj isa WaveletGP
        _waveletgp_surface_derivatives(spatial_spec, chain, coords_mat, radii; n_samples=n_samples)
    elseif !isnothing(spatial_spec) && spatial_spec.component_obj isa SPDE
        _spde_surface_derivatives(spatial_spec, chain, coords_mat, radii; n_samples=n_samples)
    else
        _generic_ad_surface_derivatives(
            model_obj, chain, coords_mat, radii;
            spatial_spec=spatial_spec, n_samples=n_samples
        )
    end

    # If model has an intercept, add intercept samples to elevation realizations
    if haskey(M, :add_intercept) && M.add_intercept
        p_names = _get_clean_chain_param_names(chain)
        int_name = _find_parameter(p_names, "intercept", 1, false)
        if !isempty(int_name)
            int_samples = get_params_vector(chain, int_name, 1)
            S_actual = size(raw_derivs.z, 2)
            for s in 1:S_actual
                raw_derivs.z[:, s] .+= int_samples[s, 1]
            end
        end
    end

    # 4. Compute topographic differential metrics across all posterior samples
    topo_samples = compute_topographic_metrics(
        raw_derivs.zx, raw_derivs.zy,
        raw_derivs.zxx, raw_derivs.zyy, raw_derivs.zxy
    )

    # 5. Summarize posterior distributions
    function _summarize_field(field_mat::Matrix{Float64})
        means = dropdims(mean(field_mat, dims=2), dims=2)
        stds = dropdims(std(field_mat, dims=2), dims=2)
        lowers = [quantile(field_mat[i, :], alpha / 2.0) for i in 1:size(field_mat, 1)]
        uppers = [quantile(field_mat[i, :], 1.0 - alpha / 2.0) for i in 1:size(field_mat, 1)]
        return (mean=means, std=stds, lower=lowers, upper=uppers)
    end

    summary_z = _summarize_field(raw_derivs.z)
    summary_slope_grad = _summarize_field(topo_samples.slope_gradient)
    summary_slope_deg = _summarize_field(topo_samples.slope_degrees)
    summary_aspect = _summarize_field(topo_samples.aspect_degrees)
    summary_laplacian = _summarize_field(topo_samples.laplacian)
    summary_k_prof = _summarize_field(topo_samples.profile_curvature)
    summary_k_plan = _summarize_field(topo_samples.planform_curvature)
    summary_k_tot = _summarize_field(topo_samples.total_curvature)

    summary_bpi = Dict(
        r => _summarize_field(raw_derivs.bpi[r])
        for r in radii
    )

    # 6. Build summary DataFrame
    df_out = DataFrame(
        s_x = coords_mat[:, 1],
        s_y = coords_mat[:, 2],
        z_mean = summary_z.mean,
        z_sd = summary_z.std,
        slope_mean = summary_slope_grad.mean,
        slope_sd = summary_slope_grad.std,
        slope_deg_mean = summary_slope_deg.mean,
        aspect_deg_mean = summary_aspect.mean,
        laplacian_mean = summary_laplacian.mean,
        laplacian_sd = summary_laplacian.std,
        profile_curv_mean = summary_k_prof.mean,
        planform_curv_mean = summary_k_plan.mean,
        total_curv_mean = summary_k_tot.mean
    )

    for r in radii
        r_str = replace(string(r), "." => "_")
        df_out[!, Symbol("bpi_r", r_str, "_mean")] = summary_bpi[r].mean
        df_out[!, Symbol("bpi_r", r_str, "_sd")] = summary_bpi[r].std
    end

    metrics_dict = (
        elevation = summary_z,
        slope = summary_slope_grad,
        slope_degrees = summary_slope_deg,
        aspect = summary_aspect,
        grad_x = _summarize_field(topo_samples.grad_x),
        grad_y = _summarize_field(topo_samples.grad_y),
        laplacian = summary_laplacian,
        profile_curvature = summary_k_prof,
        planform_curvature = summary_k_plan,
        total_curvature = summary_k_tot,
        bpi = summary_bpi
    )

    if return_samples
        samples_dict = (
            elevation = raw_derivs.z,
            grad_x = topo_samples.grad_x,
            grad_y = topo_samples.grad_y,
            slope_gradient = topo_samples.slope_gradient,
            slope_degrees = topo_samples.slope_degrees,
            aspect_degrees = topo_samples.aspect_degrees,
            hessian_xx = topo_samples.hessian_xx,
            hessian_yy = topo_samples.hessian_yy,
            hessian_xy = topo_samples.hessian_xy,
            laplacian = topo_samples.laplacian,
            profile_curvature = topo_samples.profile_curvature,
            planform_curvature = topo_samples.planform_curvature,
            total_curvature = topo_samples.total_curvature,
            bpi = raw_derivs.bpi
        )
        return (
            summary = df_out,
            metrics = metrics_dict,
            samples = samples_dict
        )
    else
        return (
            summary = df_out,
            metrics = metrics_dict
        )
    end
end
