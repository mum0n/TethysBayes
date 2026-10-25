---
title: "TethysBayes Command-Line Interface Runner (`TethysBayesr.jl`): Operating Guide & Technical Manual"
format: html
---

# TethysBayes Command-Line Interface Runner (`TethysBayesr.jl`)

## 1. Overview

The `TethysBayesr.jl` command-line utility provides a declarative, reproducible, and scriptable interface to the `@tbm` Bayesian Spatio-Temporal Modeling framework. It reads a structured text configuration file (formatted in TOML), resolves environment dependencies, executes user-specified data preprocessing, compiles dynamic Turing probabilistic models, performs MCMC sampling, and persists analytical outputs to high-performance formats: **JLD2** (binary model snapshots for warm-restarts) and **DuckDB** (compressed relational analytical tables).

```
                      ┌─────────────────────────────────┐
                      │    Configuration File (.toml)   │
                      │  - [environment], [data]        │
                      │  - [preprocessing], [model]     │
                      │  - [sampling], [output]         │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │        scripts/TethysBayesr.jl         │
                      │   - Dynamic package loading     │
                      │   - User data preprocessing     │
                      │   - Model compilation & MCMC    │
                      │   - Analytical summary metrics  │
                      └────────┬───────────────┬────────┘
                               │               │
                               ▼               ▼
                      ┌────────────────┐ ┌──────────────┐
                      │   Tier 1 JLD2  │ │ Tier 2 DuckDB│
                      │  Model State   │ │ Relational DB│
                      │  (.jld2)       │ │ (.duckdb)    │
                      └────────────────┘ └──────────────┘
```

---

## 2. Command-Line Invocation & Options

### 2.1 General Syntax

```bash
julia --project=. scripts/TethysBayesr.jl <config.toml> [options]
```

or using the explicit `--config` option:

```bash
julia --project=. scripts/TethysBayesr.jl --config=<path_to_config.toml> [options]
```

### 2.2 CLI Flags & Options

Command-line options override settings declared in the TOML configuration file:

| Option | Short Flag | Description | Default / Fallback |
| :--- | :--- | :--- | :--- |
| `--config=<path>` | `-c=<path>` | Path to the TOML configuration file | First positional argument |
| `--output=<path>` | `-o=<path>` | Destination output file or directory | `[output].path` in TOML |
| `--format=<fmt>` | `-f=<fmt>` | Output format: `jld2`, `duckdb`, or `both` | `[output].format` in TOML |
| `--samples=<N>` | `-s=<N>` | Number of posterior MCMC iterations | `[sampling].n_samples` (100) |
| `--warmup=<N>` | `-w=<N>` | Number of warmup / adaptation steps | `[sampling].n_warmup` (50) |
| `--sampler=<name>`| — | Sampling algorithm: `optimal`, `nuts`, `gibbs`, `mh`, `hmc` | `[sampling].sampler` (`optimal`) |
| `--no-sample` | — | Compiles and validates model without running MCMC | False |
| `--dry-run` | — | Validates TOML, data, and formula syntax and exits | False |
| `--verbose` | `-v` | Enables detailed logging of compilation & execution | `[model].verbose` (False) |
| `--help` | `-h` | Prints formatted help guide with options and examples | — |

---

## 3. TOML Configuration Specification

A `TethysBayesr` configuration file is organized into six declarative tables:

### 3.1 `[metadata]`
Provides human-readable provenance and documentation stored directly in output files:

```toml
[metadata]
name = "scotian_shelf_rff"
description = "Continuous spatial bathymetry model using Random Fourier Features"
project = "Scotian Shelf Ecosystem Mapping"
author = "Jae S. Choi"
```

### 3.2 `[environment]`
Defines run-specific Julia libraries required for the workflow. These packages are dynamically imported into `Main` before data loading or preprocessing begins:

```toml
[environment]
# List of Julia package names to load into the execution runtime
packages = ["RData", "DataFrames", "SpatialData"]
```

### 3.3 `[data]`
Specifies the input dataset source, format, and optional spatial geometry dependencies:

```toml
[data]
# Relative or absolute path to the data file (.rda, .rdata, .jld2, .csv, .duckdb)
path = "data/scotian_shelf_bathymetry_subset.rda"

# Object key (for .rda / .jld2) or table name (for .duckdb)
key = "bathy"

# Format indicator: "auto" (detected from extension), "rda", "jld2", "csv", "duckdb"
format = "auto"

# Optional path to spatial areal units container (.jld2) for CAR/BYM2/SPDE models
# au_path = "data/scotian_shelf_areal_units.jld2"

# Optional custom SQL query when reading from DuckDB
# query = "SELECT plat, plon, z FROM bathymetry_table WHERE z IS NOT NULL"
```

### 3.4 `[preprocessing]`
Enables customized data transformation prior to model compilation and sampling. Multiple preprocessing strategies can be combined and are executed in sequence:

```toml
[preprocessing]
# 1. Run-specific packages needed specifically for data transformation
packages = ["Statistics"]

# 2. External Julia script: executes with `df_input` in Main and captures `df_output`
# script = "scripts/preprocess_bathymetry.jl"

# 3. User-defined function: calls a function already defined in Main on the DataFrame
# transform = "clean_survey_observations"

# 4. Inline Julia expression or block transforming `df`
code = "filter(:z => z -> !isnan(z) && z > -1500.0, df)"
```

#### Preprocessing Execution Sequence:
1. If `script` is specified, `df` is assigned to `Main.df_input`, the script is evaluated via `Base.include(Main, script_file)`, and the result is read from `Main.df_output` (or `Main.df_input`).
2. If `transform` is specified, the named function is retrieved from `Main` and called via `Base.invokelatest(func, df)`.
3. If `code` is specified, `df` is assigned to `Main.df`, and the code string is evaluated. If it evaluates to a `DataFrame`, it is captured.

### 3.5 `[model]`
Defines the symbolic model formula and compilation options:

```toml
[model]
# Model formula using @tbm syntax:
formula = "likelihood(z, family = gaussian) ~ intercept() + random(plat, plon, model = rff, n_features = 30)"

# Computational flags
verbose = false
use_gpu = false
prior_scheme = "pcpriors"  # Options: "pcpriors", "default", "informative"
```

### 3.6 `[sampling]`
Configures posterior simulation and MCMC convergence parameters:

```toml
[sampling]
# Enable or disable MCMC sampling
sample = true

# Sampler strategy: "optimal" (auto-calibrated / promoted), "nuts", "gibbs", "mh", "hmc"
sampler = "optimal"

# Sampling dimensions
n_samples = 100       # Number of retained posterior draws
n_warmup = 50         # Adaptation / warmup iterations
n_chains = 1          # Number of independent chains
target_accept = 0.8   # NUTS target acceptance rate
progress = true       # Display interactive terminal progress bar
```

### 3.7 `[output]`
Controls persistence destinations, file formats, and storage compression:

```toml
[output]
# Destination path (.jld2 or .duckdb)
path = "output/scotian_shelf_rff.duckdb"

# Format: "jld2", "duckdb", or "both"
format = "duckdb"

# Enable Zstd / native compression
compress = true

# Optional table name prefix for DuckDB relational export
table_prefix = "rff_"
```

---

## 4. End-to-End Practical Examples

### 4.1 Example 1: Bathymetry Modeling with DuckDB Export

**Configuration File (`scripts/config_bathymetry_rff_duckdb.toml`)**:
```toml
[metadata]
name = "scotian_shelf_rff_duckdb"
description = "Random Fourier Features (RFF) bathymetry model with DuckDB relational output"
project = "Scotian Shelf Ecosystem Mapping"
author = "Jae S. Choi"

[environment]
packages = ["RData"]

[data]
path = "data/scotian_shelf_bathymetry_subset.rda"
key = "bathy"
format = "rda"

[preprocessing]
code = "filter(:z => z -> !isnan(z) && z > -1500.0, df)"

[model]
formula = "likelihood(z, family = gaussian) ~ intercept() + random(plat, plon, model = rff, n_features = 30)"
verbose = false
use_gpu = false
prior_scheme = "pcpriors"

[sampling]
sample = true
sampler = "optimal"
n_samples = 100
n_warmup = 50
n_chains = 1
target_accept = 0.8
progress = true

[output]
path = "output/scotian_shelf_rff.duckdb"
format = "duckdb"
compress = true
table_prefix = "rff_"
```

**Execution**:
```bash
julia --project=. scripts/TethysBayesr.jl scripts/config_bathymetry_rff_duckdb.toml
```

**Output Inspection**:
The resulting DuckDB database contains normalized relational tables ready for SQL querying or GIS integration:
- `rff_model_metadata`: Model formula, sampler details, and runtime metadata.
- `rff_metrics`: Model evaluation metrics (RMSE, Pearson $r$, WAIC, $\hat{R}$, ESS).
- `rff_parameter_stats`: Posterior mean, standard deviations, and credible intervals ($q_{0.025}, q_{0.5}, q_{0.975}$).
- `rff_predictions`: Denoised latent surfaces ($\mu$), linear predictor ($\eta$), and observation-level intervals.

### 4.2 Example 2: Fast Model Validation (`--no-sample` / `--dry-run`)

To verify configuration syntax, data accessibility, and model compilation without running expensive MCMC sampling:

```bash
# Validates configuration, loads data, and instantiates model without MCMC
julia --project=. scripts/TethysBayesr.jl scripts/config_bathymetry_rff.toml --no-sample

# Quick dry-run check
julia --project=. scripts/TethysBayesr.jl scripts/config_bathymetry_rff.toml --dry-run
```

### 4.3 Example 3: Overriding Sampling Dimensions via CLI

Run production sampling with 1,000 iterations and 500 warmup steps, overriding the TOML defaults:

```bash
julia --project=. scripts/TethysBayesr.jl scripts/config_bathymetry_rff.toml -s=1000 -w=500 --sampler=nuts
```

### 4.4 Example 4: Exporting Both JLD2 and DuckDB Simultaneously

Save both the complete model state snapshot for warm-restarts and the relational DuckDB database:

```bash
julia --project=. scripts/TethysBayesr.jl scripts/config_bathymetry_rff.toml --format=both -o=output/scotian_shelf_bundle
```

This creates:
- `output/scotian_shelf_bundle.jld2`: Tier 1 model state snapshot.
- `output/scotian_shelf_bundle.duckdb`: Tier 2 compressed relational analytics database.

---

## 5. Troubleshooting & Best Practices

1. **Missing User Packages**:
   If an environment or preprocessing package is not installed, `TethysBayesr.jl` raises an informative error directing the user to run `Pkg.add("<PackageName>")`.

2. **World-Age Issues in Dynamic Scripts**:
   The `TethysBayesr.jl` runner utilizes `Base.invokelatest` on model instantiation and sampling boundaries to guarantee seamless execution when dynamic models and user functions are compiled on the fly.

3. **Memory Management on Large Datasets**:
   For large datasets (e.g. $>100{,}000$ points), consider using continuous spatial approximations like Random Fourier Features (`model = rff`) or NNGP (`model = nngp`) rather than full covariance Gaussian Processes.

4. **DuckDB File Locking**:
   DuckDB establishes exclusive file locks during write operations. Ensure that external database viewers (e.g. DBeaver or DuckDB CLI) are closed before overwriting an existing database file.
