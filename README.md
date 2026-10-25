# TethysBayes: Bayesian Spatiotemporal Models in Julia

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org)
[![Turing.jl](https://img.shields.io/badge/Powered%20By-Turing.jl-purple.svg)](https://turinglang.org)
[![DuckDB](https://img.shields.io/badge/Analytics-DuckDB-yellow.svg)](https://duckdb.org)

The **`TethysBayes`** framework provides a composable, formula-driven probabilistic programming interface for hierarchical Bayesian spatiotemporal modeling in Julia. Built on top of **Turing.jl** and Julia's stellar scientific computing ecosystem, `TethysBayes` separates observation likelihood specifications from latent process dynamics. This separation allows researchers to flexibly assemble spatial, temporal, non-linear, and mechanistic differential components into an integrated, differentiable probabilistic model.

Inspired by high-level formula interfaces like R's `brms` and `INLA`, `TethysBayes` and learning from `bugs`, `jags` and `stan`, it provides automated code generation, automatic differentiation (ForwardDiff, ReverseDiff, Zygote), adaptive composite block-sampling, full posterior reconstruction, spatial tessellation, analytical SQL querying, and publication-ready diagnostic visualization. It represents the evolution of the  `aegis`-based approach and workflow implemented in R (https://github.com/jae0/aegis, https://github.com/jae0/carstm, and https://github.com/jae0/bio.snowcrab) to embrace Julia's performance, ecosystem and extensibility. It is designed to make complex ecological modeling faster, more flexible, transparent and accessible to scientists, in the spirit and ideals of open and reproducible science. It is limited by my own human limits knowledge and experience, and so I look forward to other scientists running with it to make something truly great! 

Full disclosure: I have made heavy use of Gemini AI to help develop this package. It has been an invaluable catalyst to make coherent the code, unit tests, visualizations and consistent documentation. 

Best regards,
Jae


## What it looks like

By design, this should look familiar and comprehensible to most people. All that is required for a user is to define a likelihood and any latent processes. The rest is handled by `TethysBayes` and you have a full hierarchical Bayesian model! You can take the generated Turing.jl model and alter it by adding any custom components you desire! Then use the best inference method available, such as cutting edge Variational inference methods, or MCMC methods (NUTS, HMC, etc.) to estimate the features you are interested in. 

```julia

m = @tbm(
    likelihood(y, family=poisson) ~
        intercept() +
        fixed(elevation) +
        random(s_idx, model=bym2, W=W) +
        random(year_idx, model=ar1),
    data = st_data
);

```

---

## Key Features

- **Intuitive Formula DSL (`@tbm`)**:
  Declarative formula syntax separating observation likelihoods (`likelihood(y, family=...)`) on the LHS from latent processes (`intercept() + fixed() + random()`) on the RHS.
- **Rich Latent Component Library (50+ Components)**:
  - **Discrete Spatial GMRFs**: `BYM2` (Besag-York-Mollié with spectral scaling), `ICAR`, `Besag`, `Leroux`, `MCAR` (Multivariate Conditional Autoregressive), `SAR` (Simultaneous Autoregressive), `LocalAdaptive` (cluster-specific regime shifts), and `BCGN`.
  - **Continuous Geostatistics**: `NNGP` (Nearest Neighbor Gaussian Process scaling to $N > 10^5$), Exact Gaussian Processes (`GP`), `SparseGP` (FITC pseudo-inputs), `RFF` (Random Fourier Features), and `SPDE` (triangulated mesh Matérn fields).
  - **Temporal & Seasonal**: `AR1`, `AR2`, `RW1` (stochastic level), `RW2` (stochastic curvature), `Harmonic` / `Cyclic` (Fourier seasonality), and `TAR` (Threshold Autoregressive).
  - **Nonparametric Smooths**: Penalized B-splines (`PSpline`), `BSpline`, Thin Plate Regression Splines (`TPS`), and `AdaptiveSmooth`.
  - **Mechanistic & Movement Dynamics**: Advection-Diffusion-Reaction (`movement()`), state-space dynamical systems (`dynamics()`), Bayesian factor analysis (`eigen()`), and ODE/SciML integration (`sciml()`).
- **Component Algebra & Compositional Operators**:
  - **Kronecker Interaction (`⊗`)**: Models non-separable space-time interactions (Knorr-Held Types I–IV, $Q_{st} = Q_t \otimes Q_s$).
  - **Pipe Operator (`|>`)**: Constructs Spatially-Varying Coefficients (SVC) and spatially-varying temporal curves (e.g. `covariate |> random(s_idx, model=icar)`).
  - **Composition (`∘`)**: Links intensity fields to point processes (e.g. Log-Gaussian Cox Processes `LGCP`).
- **Comprehensive Spatial Partitioning Subsystem** (provided by the [`Tethys`](../Tethys) partitioning subsystem, re-exported here):
  - 9 automated spatial discretization algorithms: `:hexagonal` (honeycomb packing), `:cvt` (Lloyd's centroidal relaxation), `:kvt` (K-means density balancing), `:qvt`, `:bvt`, `:hvt`, `:avt`, `:lattice` (fast raster), and spring-layout coordinate inference.
  - Granular sizing and polygon count controls: `target_units`, `exact_units`, `target_area`, `min_area`, `merge_small_polygons`, `prune_empty`, and KD-tree island bridging (`ensure_connected!`).
  - Joint space-time synchronization: `assign_spatiotemporal_units` ($st = (t-1)S + s$).
- **Flexible Likelihoods & Observation Models**:
  - 19 distribution families: `:gaussian`, `:poisson`, `:negbin`, `:bernoulli`, `:binomial`, `:beta`, `:gamma`, `:lognormal`, `:studentt`, `:exponential`, `:weibull`, `:gev`, `:zipoisson`, `:zinegbin`, `:ordered_logistic`, `:ordered_probit`, `:categorical`, `:multinomial`, `:dirichlet`.
  - Observation modifiers: `log_offsets` (epidemiological rate modeling), `weights`, `trials`, `zero_inflated`, `hurdle`, `volatility`, `censor_lower`, and `censor_upper`.
- **Adaptive Sampler Optimization (`get_optimal_sampler`)**:
  Introspects parameter supports and constructs composite Gibbs samplers assigning `PG` to discrete variables, `ESS` to Gaussian latent vectors, `Slice` to bounded parameters, and `NUTS` to continuous blocks.
- **Two-Tier Model Persistence & Analytical SQL Engine (`src/input_output.jl`)**:
  - **Tier 1 (JLD2)**: Full binary serialization of live callable Turing models (`m`), configurations, data, and MCMC chains (`chn`).
  - **Tier 2 (DuckDB)**: Embedded relational SQL database storing normalized metrics, parameter statistics, predictions, WKT spatial geometries, and diagnostic plot data for zero-copy querying, multi-model Bayesian Model Averaging (BMA), sequential prior extraction, and CSV/GeoJSON export.
- **Diagnostics, Post-Processing & Visualization (`src/plotting.jl`)**:
  One-line extraction of posterior credible intervals, predictive error metrics (RMSE, $R^2$, DIC, WAIC), spatial choropleth maps (`choropleth`), spatial adjacency graphs (`spatial_graph_plot`), animal movement paths (`render_paths!`), and timeseries ribbons (`timeseries_ci`).
- **Spatial Block Cross-Validation (`TethysBayes_cv_orchestrator`)**:
  Assesses out-of-sample generalization using `:spatial_block`, `:temporal_block`, `:lolo` (leave-one-location-out), and `:temporal_forward_chain` to prevent spatial autocorrelation leakage.

- **Modular Bayesian DAG Pipeline & Cross-Mesh Resharding (`src/pipeline.jl`)**:
  - Declarative workflow orchestrator (`TethysBayes_pipeline`) running sequential multi-tier cut-posterior models.
  - Linear geometric transfer operators ($P \in \mathbb{R}^{N_{\text{dest}} \times N_{\text{src}}}$) mapping spatial latent fields across mismatched irregular polygonal tessellations via `:area_weighted`, `:gaussian_kernel`, and `:inverse_distance` weighting (`compute_network_transfer_matrix`).
  - Full Monte Carlo sample matrix transformation ($U_{\text{dest}} = P U_{\text{src}}$) preserving non-Gaussian posterior geometry (`reshard_spatial_field`, `summarize_sample_matrix`).
  - Canonical master mesh harmonization across all tiers with automated DuckDB persistence.
- **Continuous Surface Derivatives & Differential Geometry (`src/derivatives.jl`)**:
  - Exact analytical gradients ($\nabla z$), geomorphometric slopes ($\|\nabla z\|$, $\arctan\|\nabla z\|$), compass aspects, Hessians ($z_{xx}, z_{yy}, z_{xy}$), Laplacians ($\nabla^2 z$), profile/planform curvatures ($k_{\text{prof}}, k_{\text{plan}}$), and circular Bessel Bathymetric Position Indices ($\text{BPI}_r$) across `RFF`, `SpectralGP`, `WaveletGP`, `SPDE`, `PSpline`, and `TPS` (`TethysBayes_surface_derivatives`).
- **Errors-in-Variables (EIV) Covariate Priors (`src/model.jl`)**:
  - Propagate upstream posterior uncertainty directly into downstream linear predictors as latent Gaussian measurement errors: $x_i \sim \mathcal{N}(\mu_{x, i}, \sigma^2_{x, i})$ via `fixed(cov, error_sd=:cov_sd)` or `eiv(cov, cov_sd)`.

---

## Installation & Setup

`TethysBayes` is structured as a self-contained Julia project environment. Clone the repository and load the module:

```bash
git clone https://github.com/mum0n/TethysBayes.git
TethysBayes_location = "where/you/saved/TethysBayes"  # e.g. TethysBayes_location = "C:/home/jae/projects/TethysBayes"

```

Start Julia within the repository:

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

# or, you can of course install and import as a package too:  
# Pkg.add(url="https://github.com/mum0n/TethysBayes.git"); using TethysBayes

# or to develop:
cd(TethysBayes_location)
] dev .
using TethysBayes


```

## Quick Start Examples

### Example 1: Spatial Disease Mapping (Scottish Lip Cancer Data)

Fit a hierarchical BYM2 spatial model with an AR1 temporal trend, extract diagnostics, and persist to a DuckDB bundle:

```julia

Random.seed!(42)

# 1. Load benchmark dataset (56 Scottish districts across time)
data_scot = TethysBayes_data(); # Scottish Lip Cancer
df = data_scot.data # dataframe with response and covariates
W = data_scot.au.W  # graph (adjacency matrix)

# 2. Specify Hierarchical Spatiotemporal Model
m = @tbm(
    likelihood(y, family=poisson, log_offsets=log_offsets) ~
        intercept() +
        fixed(cov1) +
        random(s_idx, model=bym2, W=W) +
        random(year, model=ar1),
    df     
);

# 3. Sample from Posterior with NUTS
chn = sample(m, NUTS(), 30; progress=false)

# 4. Extract Comprehensive Diagnostics & Summaries (Pure Data)
res = model_results_comprehensive(m, chn);
println("Model WAIC: ", res.metrics.waic)
display(res.parameters)

# 5. Generate, Display, and Export Diagnostic Plots
plots_res = TethysBayes_plots(res; au=data_scot.au, save_dir="output/plots");
display(plots_res.plots[:spatial])

# 6. Persist Unified Bundle to DuckDB and JLD2
save_TethysBayes_bundle("output/scot_lip_model", m, chn, res; au=data_scot.au);
bn_bundle = load_TethysBayes_bundle("output/scot_lip_model") ;
keys(bn_bundle)

# 7. Query prior
# Extract prior from the saved bundle
prior_posterior = extract_prior_posterior(bn_bundle);

# Show posterior summaries of the prior (first 1000 draws)
display(prior_posterior[:prior, :parameters, 1:1000])
 
```

---

### Example 2: Continuous Point Data Partitioning with Hexagonal Binning

Discretize continuous GPS coordinates into exact regular hexagons, construct neighborhood topology, and fit a spatiotemporal model:

```julia

rng = MersenneTwister(42)
N = 400

# 1. Continuous point observations
df = DataFrame(
    lon = rand(rng, N) .* 100.0,
    lat = rand(rng, N) .* 50.0,
    year = rand(rng, 2020:2024, N),
    elevation = randn(rng, N),
    y = rand(rng, 0:20, N)
);

# 2. Partition space into hexagons across 5 years
st_data = assign_spatiotemporal_units(df;
    space_x = :lon,
    space_y = :lat,
    time_var = :year,
    area_method = :hexagonal,
    target_units = 16,
    radius = 10.0,
    exact_units = true
);

spatial_graph_plot(au=st_data.au_spatial, title="Spatial Units")

df.s_idx = st_data.s_idx ;
df.year_idx = st_data.t_idx ;

# 3. Fit Spatiotemporal Model with BYM2 + AR1
m = @tbm(
    likelihood(y, family=poisson) ~
        intercept() +
        fixed(elevation) +
        random(s_idx, model=bym2) +
        random(year_idx, model=ar1),
    df,
    W = st_data.W_spatial,
    verbose = false
);

# 4. Process Results and Generate Plots
chn = sample(m, NUTS(), 300; progress=false);
res = model_results_comprehensive(m, chn);
plots_res = TethysBayes_plots(res; data=df, au=st_data.au_spatial);

p_map = plots_res.plots[:spatial]
p_ppc = plots_res.plots[:posterior_predictive_check]
plot(p_map, p_ppc, layout=(1, 2), size=(1000, 450))
```


---

### Example 3: Multivariate Hierarchical Spatiotemporal Model

```julia

data_scot = TethysBayes_data(); # Scottish Lip Cancer
df = data_scot.data # dataframe with response and covariates
W = data_scot.au.W  # graph (adjacency matrix)

m = @tbm(
    # Multivariate joint likelihood: specify two outputs and their families
    likelihood((y, y_rate),
               family = (poisson, gaussian),
               # pass offsets/trials if needed per outcome: log_offsets=(offset1, offset2)
    ) ~
        # Common fixed effects (applied to both outcomes), plus outcome-specific terms
        intercept() +
        fixed(cov1) +
        # name the spatial component "shared_spatial" so it's the same latent effect for both outcomes
        random(s_idx, model=bym2, W = W, key = :shared_spatial) +
        # outcome-specific temporal effect: this will be added per outcome automatically
        random(t_idx, model=ar1, key = :temporal_by_outcome),
    df;
    model_arch = "multivariate",    # enable multivariate architecture
    verbose = true
)
 
```



### Example 4: Analytical SQL Querying, Prior Extraction & Chain Extension

Leverage DuckDB for relational SQL analytics and resume sampling on previously saved models:

```julia
using TethysBayes

# 1. Query previously saved model results directly via SQL
df_high_risk = query_duckdb("output/scot_lip_model.duckdb", """
    SELECT unit_id, latent_field_mean, latent_field_lower, latent_field_upper 
    FROM plot_data_latent_field_spatial 
    WHERE latent_field_mean > 1.0 
    ORDER BY latent_field_mean DESC
""")
display(df_high_risk)

# 2. Extract posterior parameters as informative priors for a new model
priors = extract_posterior_priors("output/scot_lip_model.duckdb")

# 3. Load model state and extend MCMC chain with 500 additional samples
bundle = load_TethysBayes_bundle("output/scot_lip_model")
chn_extended = extend_sampling(bundle.model, bundle.chain, 500; progress=false)
```

---

### Example 5: Modular Hierarchical DAG Pipeline & Surface Derivatives

Execute a multi-tier ecological workflow with analytical surface derivatives, cross-mesh resharding, Errors-in-Variables priors, and DuckDB table persistence. This was the `aegis` workflow, but with `TethysBayes` and some major enhancements enabled by the Julia ecosystem.

```julia
using TethysBayes, DataFrames, Turing

# 1. Declarative Multi-Tier Pipeline Execution
pipe_result = TethysBayes_pipeline(
    :depth => (
        formula = "likelihood(depth) ~ intercept() + random(s_x, s_y, model=rff, n_features=25)",
        data = df_bathy,
        derivatives = [:slope, :curvature, :bpi],
        radii = [10.0, 25.0]
    ),
    :substrate => (
        formula = "likelihood(grain) ~ intercept() + fixed(depth_mu, error_sd=:depth_sd) + random(s_idx, model=bym2)",
        data = df_sub,
        au = au_sub
    ),
    :biology => (
        formula = "likelihood(biomass, family=gamma) ~ intercept() + fixed(grain_mu, error_sd=:grain_sd) + random(s_idx, model=bym2)",
        data = df_bio,
        au = au_master
    );
    master_au = au_master,
    duckdb_path = "project_db/ecosystem_pipeline.duckdb",
    geojson_path = "project_gis/master_habitat.geojson"
)

# 2. Query Master Harmonized Summary Table from DuckDB
df_master = query_duckdb("project_db/ecosystem_pipeline.duckdb", "SELECT * FROM master_harmonized_summary LIMIT 10")
display(df_master)
```

---

## Documentation

There is a lot more functionality available. Comprehensive guides and technical documentation are available in the `docs/` directory:

- [**Integrated Hierarchical Workflows & ADR Telemetry** (`docs/hierarchical_workflow/hierarchical_workflow.md`)](docs/hierarchical_workflow/hierarchical_workflow.md):
  Directed Acyclic Graphs (DAGs), multi-scale cross-mesh resharding, Full Monte Carlo matrix transformations, Errors-in-Variables (EIV) priors, second-last tier Habitat Suitability (HSI) determination, analytical surface derivatives, Advection-Diffusion-Reaction (ADR) population dynamics, Lagrangian telemetry, and DuckDB SQL analytics, Tempered Power Posteriors (fractional feedback $\lambda$), full empirical spatial covariance in EIV, hybrid continuous basis-polygon quadrature resharding, continuous soft-sigmoid physiological HSI, and coupled oceanographic-active advection.
- [**Architectural & Methodological Overview** (`docs/TethysBayes.md`)](docs/TethysBayes.md):
  Design principles, formula syntax, component algebra, prior systems, and inference engines.
- [**Technical API Reference** (`docs/TethysBayes_api.md`)](docs/TethysBayes_api.md):
  Developer reference for the `ComponentModel` lifecycle, `ParamRegistry`, likelihood distributions, plotting functions, and AST parser.
- [**Spatial & Spatiotemporal Partitioning Guide** (`docs/TethysBayes_spatial_partitioning.md`)](docs/TethysBayes_spatial_partitioning.md):
  Mathematical formulations, Lloyd's relaxation, hexagonal geometry, MAUP mitigation, island bridging, cross-mesh transfer operators, and BYM2 spectral scaling. The engine itself lives in [`Tethys`](../Tethys); this document is mirrored there as `docs/spatial_partitioning.md`.
- [**Input / Output & Persistence Guide** (`docs/TethysBayes_input_output.md`)](docs/TethysBayes_input_output.md):
  Two-tier persistence architecture, JLD2 model serialization, DuckDB analytical results storage, sample extension, SQL analytics, and GIS export.
- [**Custom Components & Spatial SEIR Modeling Guide** (`docs/TethysBayes_custom.md`)](docs/TethysBayes_custom.md):
  Mechanistic process modeling, raw Turing code injection with `custom()`, first-class `ComponentModel` implementation, and spatial SEIR disease dynamics. OR, create a new ComposedModel specific to your needs by extending existing components. Use the many examples provided in 'src/composed/' as a guide. 

---

## Citation & References

If you use `TethysBayes` in your research, please cite:

1. **Besag, J.** (1974). Spatial interaction and the statistical analysis of lattice systems. *Journal of the Royal Statistical Society: Series B*, 36(2), 192–225.
2. **Hooten, M. B., & Hefley, T. J.** (2019). *Bringing Bayesian Models to Life*. CRC Press.
3. **Knorr-Held, L.** (2000). Bayesian modelling of inseparable space-time variation in disease risk. *Statistical Methods in Medical Research*, 9(3), 205–220.
4. **Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H.** (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
5. **Ge, H., Xu, K., & Ghahramani, Z.** (2018). Turing: A language for flexible probabilistic programming. *International Conference on Artificial Intelligence and Statistics (AISTATS)*.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
