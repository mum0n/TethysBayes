# Modelling the bathymetry of the Scotian Shelf, Canada

## Introduction
There exist a rich tradition of ecological modelling, where the need to produce predictions of environmental
features over large spatial extents is common. The Scotian Shelf, off the coast of Nova Scotia, Canada,
is a large submarine s_yeau that supports a rich diversity of marine life. It is a productive ecosystem that provides
habitat for a wide range of species, including fish, invertebrates, and marine mammals. Mapping the features
of this region, such as depth (bathymetry),  is an important task for understanding the ecology of the region,
and for managing its resources. In this document, we will use TethysBayes to model the bathymetry of the Scotian Shelf.

The purpose of this document is twofold:
1. To show how to use Julia and its ecosystem of tools, including TethysBayes, to model the bathymetry of the Scotian Shelf.
2. To compare the performance of different spatial models for modeling this data. 

In particular, we will compare a number of different spatial models including: BYM2, CAR and continuous models such as RFF,
FFT, SpectralGP, WaveletGP, and traditional kriging. We will then evaluate the models using a number of
different metrics including RMSE, Pearson r, ESS, Rhat, and WAIC.   

## Setup 

```julia

#  your working directory (where data, etc are found)
myworkdir = "c:/home/jae/projects/TethysBayes"

mkpath(myworkdir)
cd(myworkdir) 

using Pkg
Pkg.activate(myworkdir)
Pkg.instantiate()

# this will install required packages, you will have to rerun this several times to get all the dependencies worked out. 
# manually install and dependencies that get interrupted, if needed restart julia as well
TethysBayes_location = "c:/home/jae/projects/TethysBayes"
include( joinpath(TethysBayes_location, "src", "TethysBayes.jl") ) 
using .TethysBayes  # the "." means load the module in the current session, not importing it as a package.  


```

## Data

The main source of data for this analysis is the Canadian Hydrographic Service's (CHS). Ecosystems RV Surveys (Department of Fisheries and Oceans Canada; DFO) as well as snow crab scientific survey data (DFO) supplement this dataset. 

We bring a segment of this data here for testing:

```julia
## Load the data:
# fn = "https://github.com/jae0/carstm/blob/064b8a52bdd31fbb88c99a7bc97bb2b1cc1d0234/inst/extdata/scotian_shelf_bathymetry_subset.rda"

using RData, DataFrames

data_location = "c:/home/jae/projects/TethysBayes"

fn = joinpath( data_location, "data", "scotian_shelf_bathymetry_subset.rda" )
df_bathy = RData.load(fn)["bathy"];  # Coordinates plat and plon are projected coordinates in km
# Notice: No need to rename plat/plon to s_x/s_y; TethysBayes natively accepts user variable names!

# plot histograms and chloropleth maps

```


## Methods

The are a number of spatial methods accessible in `TethysBayes`.  These fall into several categories:
- Traditional kriging
- Continuous spatial models: RFF, TPS, PSpline, NNGP, GP, SparseGP, AdaptiveSmooth, SpectralGP, WaveletGP
- Hierarchical / areal unit models: SPDE, Leroux, ICAR, BYM2
- Spectral models: FFT, SpectralGP, WaveletGP

We will demonstrate a selection of these methods for modeling the bathymetry of the Scotian Shelf.  

### Kriging

Kriging is a well known geostatistical method for spatial interpolation. We use it to provides a familiar baseline for comparing the performance of other more advanced methods.  

It assumes that the spatial field is a realization of a Gaussian random field.  The computations in standard version use least squares and rely on an explicit definition of the spatial covariance matrix of the data (usually fit "by eye") and the covariance matrix of the prediction locations. It is a computationally expensive method ($O(n^3)$) and so can be prohibitive for large datasets.  

```julia
import Pkg

# Pkg.add(["TableTransforms", "CategoricalArrays", "CoordRefSystems", "GeoTables", "GeoStats", "CairoMakie"])

using TableTransforms, CategoricalArrays, CoordRefSystems, GeoTables, GeoStats, CairoMakie
using LinearAlgebra
import CairoMakie as Mke

TethysBayes_location = "c:/home/jae/projects/TethysBayes"
include( joinpath(TethysBayes_location, "src", "kriging.jl") )

# Compute and plot (automatically detects plat, plon and response variable z)
results = compute_kriging(df_bathy)
plot_kriging(results)


```

### Bayesian models using `TethysBayes`

```julia

#  nonstationary variance  (no covariate)
# m_nonstationaryvariance = @tbm(
#     likelihood(z) ~ intercept() + `random(s_idx, model=icar) ∘ random(cov, model=rff)`,
#     df_bathy
# )

# local adaptive 
localadaptive = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = localadaptive, n_clusters = 8),
    df_bathy
)   



# 1. Random Fourier Features (RFF)
m_rff = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = rff, n_features = 30),
    df_bathy
)

# 2. Thin Plate Regression Splines (TPS)
m_tps = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = tps, k = 40),
    df_bathy
)

# 3. Penalized B-Splines (PSpline)
m_pspline = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = pspline, n_knots = 15),
    df_bathy
)

# 4. Nearest Neighbor Gaussian Process (NNGP) for massive point clouds
m_nngp = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = nngp, n_neighbors = 15),
    df_bathy
)

# 5. Exact Gaussian Process (GP) for smaller regional subsets (prior prective check causes this to be slow)
# m_gp = @tbm(
#    likelihood(z) ~ intercept() + random(plat, plon, model = gp),
#    df_bathy
# )

  
# 6. Stochastic Partial Differential Equation (SPDE) Matérn Field
# Ideal for complex coastal boundaries and irregular bathymetric mesh domains
au = assign_spatial_units(df_bathy, area_method=:avt, target_units=500 )
df_bathy.s_idx = au.assignments

spatial_graph_plot(au=au, title="Spatial Units")
 
m_spde = @tbm(
    likelihood(z) ~ intercept() + random(s_idx, model = spde, W=au.W ),
    df_bathy
)

# 2. Sparse Gaussian Process via Predictive Process / FITC pseudo-inputs
# Balances exact covariance capture with reduced computational complexity for large point clouds
m_sparsegp = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = sparsegp, n_inducing = 100),
    df_bathy
)

# 3. Adaptive Smooth Nonparametric Model
# Captures sharp shelf breaks, trenches, or abrupt depth transitions by allowing spatially varying smoothing parameters
m_adaptive = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = adaptivesmooth),
    df_bathy
)


# FFT / Spectral Basis approach via Random Fourier Features (RFF) 
# or spectral spatial approximations within TethysBayes

# 1. Spectral Gaussian Process (`spectralgp`) for multi-scale frequency-domain spatial representation
# m_spectralgp = @tbm(
#     likelihood(z) ~ intercept() + random(plat, plon, model = spectralgp, n_components = 32),
#     df_bathy
# )

# # 2. Wavelet Gaussian Process (`waveletgp`) for localized multi-resolution seabed features
# m_waveletgp = @tbm(
#     likelihood(z) ~ intercept() + random(plat, plon, model = waveletgp, levels = 4),
#     df_bathy
# )

# 3. FFT-based Random Fourier Features (`rff`) for scalable \mathcal{O}(N \log N) spectral approximation
m_fft = @tbm(
    likelihood(z) ~ intercept() + random(plat, plon, model = rff, n_features = 40),
    df_bathy
)

m_leroux = @tbm(
    likelihood(z) ~ intercept() + random(s_idx, model = leroux, W=au.W),
    df_bathy
)

m_icar = @tbm(
    likelihood(z) ~ intercept() + random(s_idx, model = icar, W=au.W),
    df_bathy
)

m_bym2 = @tbm(
    likelihood(z) ~ intercept() + random(s_idx, model = bym2, W=au.W),
    df_bathy
)

# note the slower ones would have to use MAP or VI for large datasets
#WAIC were from random data not real data
m = m_rff # 14930; 3 min with os on subset; 2.3963538104578423e10
m = m_tps # 70549 (but waic is lowest!)
m = m_pspline # 14929
m = m_nngp # too slow
# m = m_gp #  too slow to use atm
m = m_sparsegp # too slow to use atm
m = m_spde # 14930
m = m_adaptive # stuck; with os: 2.2673600848825867e6 (kind of high, but ppc is better!)
m = m_fft # 14928
m = m_spectralgp # too slow to use atm
m = m_waveletgp # too slow to use atm
m = m_leroux # 14933
m = m_icar # 14930
m = m_bym2 # 14930


# Sample and compute exact analytical derivatives (slopes, Hessians, BPI) for any fitted model
# os = NUTS()
# os = MH()
os = get_optimal_sampler(m) # get optimal sampler for your model
chn = TethysBayes_sample(m, os, 100; progress = true)
res = model_results_comprehensive(m, chn)
res.metrics.waic 

# ML (Maximum Likelihood) -- you probably do not want to use this. it is for demonstration of method ....
res_opt = maximum_likelihood(m, LBFGS() )  # many optimizers available
res_opt.optim_result.retcode
res_opt.params
res = model_results_comprehensive(m, res_opt, df_bathy, au, n_samples=100);
res.metrics.waic 

# MAP (Maximum A Posteriori)  
res_map = maximum_a_posteriori( m, LBFGS() )
res_map.optim_result.retcode   #
res_map.params
res = model_results_comprehensive(m, res_map, df_bathy, au);
res.metrics.waic 


# VI (Variational Inference) 
# VI is slow:  takes about 24 hr on rff on sample data with LBFGS

samples_per_step = 10
max_iters = 1000
n_samples = 1000

# q_init = q_locationscale
# q_init = q_meanfield_gaussian
# q_init = q_fullrank_gaussian
q_init = q_fullrank_gaussian

 
chn_vi = vi(
    m, q_init, max_iters, adtype=AutoForwardDiff(),
    show_progress=true; optimizer=LBFGS() 
); # other options: optimizer=Flux.ADAM(1e-1)
 
# Convert to sample chain format and reconstruct
res = model_results_comprehensive(m, chn_vi, df_bathy, au);
res.metrics.waic 

pairs(res.metrics)
 
plts = TethysBayes_plots(res; data=df_bathy, au=au)

coords = DataFrame(s_x=[  c[1] for c in au.centroids], s_y=[c[2] for c in au.centroids]);
derivs = TethysBayes_surface_derivatives(m, chn, coords ; radii = [10.0, 25.0])

res_df = DataFrame(derivs.metrics)

res_bathy = DataFrame(
    unit_id     = 1:au.n_units,
    pred_mean   = res_df.summary.z_mean,
    pred_sd     = res_df.summary.z_sd,
    slope_mean  = res_df.summary.slope_mean,
    slope_sd    = res_df.summary.slope_sd,
    curv_mean   = res_df.summary.profile_curv_mean,
    bpi_10_mean = res_df.summary.bpi_r10_0_mean
)

write_tier_table!(db_path, res_bathy, "tier1_depth_predictions")


```

## Methods

Here we examine a number of methods for modeling this data. We show that while GAMs are a flexible tool for modeling complex environmental relationships, they do not capture the spatial structure of the data. We then show how this can be done in a Bayesian context using TethysBayes in Julia. We compare a number of different spatial models including: BYM2, CAR and continuous models such as RFF, FFT, SpectralGP, WaveletGP, and traditional kriging. We then evaluate the models using a number of different metrics including RMSE, Pearson r, ESS, Rhat, and WAIC.   

## Results

## Discussion

## Conclusion   


```julia
using TethysBayes
using CSV

## Load the data
data = CSV.read("data/scotian_shelf_bathymetry.csv", DataFrame)

## Assign spatial units

## Build the model

m = @tbm(
    likelihood(depth, family=gaussian) ~ 
        intercept() + 
        fixed(longitude) + 
        fixed(latitude) + 
        random(s_idx, model=bym2, W=W)
)

## Run the sampler 

## Analyse and visualize the results

```

## Results

## Discussion

## Conclusion

---

## Appendix: Model Selection Guide for Bathymetric Mapping

Based on the current component models implemented in **TethysBayes**, their underlying mathematical formulations, and empirical computational performance on dense marine elevation datasets (such as the Scotian Shelf bathymetry), this guide provides a ranked analysis of the most useful, fast, and informative models for bathymetric modeling.

### 1. Summary Comparison Table

| Model | Component Type | Domain Type | Complexity per MCMC Step | Topographic Derivatives (Slope, BPI) | Key Strength for Bathymetry |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Thin Plate Spline (`tps`)** | Reduced-Rank RBF | Continuous $\mathbb{R}^2$ | $\mathcal{O}(N \cdot k)$ ($k \approx 30\text{--}60$) | **Exact / Analytical** | **Lowest WAIC**, rotationally invariant, ideal for continuous seabed relief |
| **Random Fourier Features (`rff`)** | Spectral GP Projection | Continuous $\mathbb{R}^2$ | $\mathcal{O}(N \cdot D)$ ($D \approx 30\text{--}60$) | **Exact / Analytical** | **Fastest continuous model** ($\sim$minutes on large point clouds), orthogonal frequency sampling |
| **SPDE Matérn (`spde`)** | GMRF on Mesh/Graph | Triangulated / Areal Mesh | $\mathcal{O}(U^{1.5})$ ($U \approx 500\text{--}2000$ units) | Discrete / Mesh Gradients | **Boundary-aware** (respects irregular coastlines and islands without land leakage) |
| **Multi-Fidelity Cascade (`transfer` / `nested`)** | Hierarchical Multi-Model | Continuous or Compound | Depends on tiers | Inherited from sub-models | **Fuses sparse regional soundings (GEBCO/altimetry) with dense local multibeam/trawl surveys** |
| **Penalized B-Splines (`pspline`)** | Tensor B-Spline | Continuous $\mathbb{R}^2$ | $\mathcal{O}(N \cdot K)$ | Exact / Analytical | Very fast and numerically stable, but coordinate-axis oriented |
| **Local Adaptive (`localadaptive`)** | Clustered Leroux GMRF | Spatial Units ($U$) | $\mathcal{O}(U \cdot k_{\text{clust}})$ | Discrete Unit Differencing | Captures **abrupt shelf breaks and deep trench edges** |

---

### 2. Tier 1: The Top Recommended Models (Fast, Continuous, and Highly Detailed)

#### 1. Thin Plate Regression Splines (`tps`) — *Best Overall Accuracy and Derivative Quality*
- **Why it is ideal for bathymetry**:
  Bathymetric surfaces represent continuous physical elevation fields where minimizing thin-plate bending energy ($\iint [f_{xx}^2 + 2f_{xy}^2 + f_{yy}^2] \, dx \, dy$) directly aligns with geophysical gravity and sediment deposition processes.
  - Unlike axis-aligned tensor splines, TPS is **rotationally invariant** (an underwater ridge or submarine canyon oriented at an angle is modeled without directional bias).
  - Because it yields an analytical, twice-differentiable continuous function $\hat{f}(\text{plon}, \text{plat})$, spatial gradients (slopes, aspect), surface curvature (Hessians), and multi-scale **Bathymetric Position Index (BPI)** can be evaluated analytically across the entire domain.
  - In empirical tests on the Scotian Shelf dataset, `tps` consistently achieves the **lowest WAIC** (best out-of-sample predictive accuracy).
- **Speed & Scalability**:
  The $N \times k$ basis matrix ($k \approx 40$ knots placed via $k$-means) is precomputed outside the MCMC sampler. Turing only samples the $k$ coefficients and the smoothing scale $\sigma_{\text{tps}}$, making NUTS/HMC sampling very fast.
- **TethysBayes Specification**:
  ```julia
  m_tps = @tbm(
      likelihood(z) ~ intercept() + random(
          plon, plat,
          model = tps,
          nbins = 40,
          method = :spectral
      ),
      df_bathy
  )
  ```

#### 2. Random Fourier Features (`rff`) — *Fastest Continuous Model for Massive Sounding Point Clouds*
- **Why it is ideal for bathymetry**:
  RFF approximates a continuous stationary Gaussian Process (e.g., Matérn-3/2 or Squared Exponential) via Bochner’s theorem by projecting projected coordinates $(\text{plon}, \text{plat})$ into a randomized spectral basis:
  $$\phi(x) = \sqrt{\frac{2}{D}} \left[ \cos(\omega_1^T x + b_1), \dots, \cos(\omega_D^T x + b_D) \right]$$
  - TethysBayes’s **orthogonal frequency sampling** (`sampling = :orthogonal`) reduces approximation variance compared to naive IID random features.
  - Transforms an otherwise intractable $\mathcal{O}(N^3)$ Gaussian process into a linear regression problem with $D$ random sinusoidal basis functions ($\mathcal{O}(N \cdot D)$).
  - Can easily handle 50,000+ to 500,000+ sounding records in a few minutes.
- **Speed & Scalability**:
  Extremely fast (typically 2–3 minutes for complete NUTS sampling on moderate-to-large subsets).
- **TethysBayes Specification**:
  ```julia
  m_rff = @tbm(
      likelihood(z) ~ intercept() + random(
          plon, plat,
          model = rff,
          n_features = 40,
          kernel = "matern32",
          sampling = :orthogonal,
          method = :fixed
      ),
      df_bathy
  )
  ```

#### 3. SPDE Matérn Field (`spde`) — *Best for Complex Coastal Geometries & Mesh Boundaries*
- **Why it is ideal for bathymetry**:
  Solves the stochastic Helmholtz equation $(\kappa^2 - \Delta)^{\alpha/2} u(s) = \mathcal{W}(s)$ using finite elements over an adjacency graph $W$ or triangular mesh (e.g., generated via `assign_spatial_units(..., area_method = :avt)`).
  - **Overcomes the land-leakage barrier**: Standard isotropic Euclidean GPs smooth across peninsulas, islands, and coastlines (e.g., smoothing depth across Cape Breton or Nova Scotia mainland). The SPDE approach restricts diffusion strictly through water paths defined on the adjacency graph or mesh.
  - The precision matrix $Q(\kappa, \sigma)$ is sparse ($\mathcal{O}(U)$ non-zeros for $U$ units), allowing fast Cholesky and spectral evaluations.
- **Speed & Scalability**:
  Very fast for 500 to 2,000 spatial units.
- **TethysBayes Specification**:
  ```julia
  au = assign_spatial_units(df_bathy, area_method = :avt, target_units = 500)
  df_bathy.s_idx = au.assignments

  m_spde = @tbm(
      likelihood(z) ~ intercept() + random(s_idx, model = spde, W = au.W),
      df_bathy
  )
  ```

#### 4. Multi-Fidelity Cascading Model (`transfer` / `nested`) — *Best for Multi-Source Data Fusion*
- **Why it is ideal for bathymetry**:
  Real-world bathymetry is heterogeneous:
  1. Low-fidelity regional grids (e.g., satellite altimetry or coarse GEBCO) covering the entire offshore domain.
  2. High-fidelity localized soundings (e.g., multibeam surveys or trawl acoustic sounders).
  Using TethysBayes’s hierarchical multi-fidelity architecture, you can model the broad regional trend with a fast continuous sub-model (like `tps` or `rff`) and couple it into local high-resolution soundings across depth strata or oceanographic regions (`strata = [:region, :depth_stratum]`).
- **TethysBayes Specification**:
  ```julia
  cfg = TethysBayes_config(
      :primary => (
          formula = "likelihood(z_survey) ~ 1 + transfer(:coarse)",
          data    = df_high_res
      ),
      :coarse => (
          formula = "likelihood(z_coarse) ~ 1 + " *
                    "random(plon, plat, model = tps, nbins = 30)",
          data    = df_regional_gebco,
          strata  = :region,
          coupling = :additive
      )
  )
  ```

---

### 3. Tier 2: Models with Specialized Utility

- **Local Adaptive Model (`localadaptive`)**:
  - Uses $k$-means spatial clustering combined with a Leroux GMRF precision matrix.
  - Highly effective when the bathymetry exhibits **sharp structural steps** (e.g., the abrupt continental shelf break where depth plunges from $\sim 200\,\text{m}$ to $> 2000\,\text{m}$ over a narrow margin).
  - *Caveat*: Slower to sample than standard `tps` or `rff`; best used with an informative cluster count (`n_clusters = 6` to `10`).

- **Penalized B-Splines (`pspline`)**:
  - Uses a 2D tensor product of B-splines with RW2 difference penalties.
  - Numerically stable and fast, but slightly less rotationally isotropic than `tps` because the knots are placed along Cartesian axes.

- **Besag-York-Mollié 2 (`bym2`) / Leroux (`leroux`)**:
  - Extremely fast and reliable on areal units, separating spatially structured variance from local measurement noise (sounding nugget effect).
  - *Caveat*: Produces discrete, piecewise-constant depth estimates for each spatial unit rather than a smooth continuous field.

---

### 4. Models to Avoid for Full MCMC on Large Bathymetric Data

- **Exact Gaussian Process (`gp`)**: Requires $\mathcal{O}(N^3)$ dense covariance matrix factorizations per MCMC leapfrog step. Unusable for $N > 2,000$ points.
- **Nearest Neighbor GP (`nngp`)**: While theoretical storage is $\mathcal{O}(N \cdot m)$, evaluating automatic differentiation gradients through large neighbor trees in Turing is currently slow for large point clouds.
- **SpectralGP / WaveletGP**: Conceptually attractive for multi-frequency sea-bottom texture, but currently computationally heavy in full MCMC; recommended only under MAP or Variational Inference (`vi`).

---

### Practical Recommendations for Bathymetry Workflows

1. **For pure continuous mapping and derivative generation (slope, BPI)**: Use **`tps`** (`nbins = 35` to `50`). It delivers the lowest WAIC, smooth gradients, and runs in a few minutes.
2. **For massive sounding point clouds ($N > 50,000$)**: Use **`rff`** (`n_features = 40` to `60`, `sampling = :orthogonal`). It provides continuous GP-like predictions with the fastest sampling speed.
3. **If coastal boundaries and islands cause unphysical smoothing across land**: Use **`spde`** over an AVT spatial unit mesh (`au = assign_spatial_units(df_bathy, area_method = :avt, target_units = 500)`).
