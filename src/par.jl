"""
    par.jl

Population Attributable Risk (PAR) and Population Attributable Fraction (PAF)
computations, counterfactual simulation, and summarization for epidemiological
models fitted with TethysBayes.

# Mathematical Background

## Relative Risk (RR) and Odds Ratio (OR)
1. **Log-Linear Models** (Poisson, Negative Binomial, Gamma, Exponential):
   With canonical log link:
   \$\\eta = \\log(\\mu) \\implies \\text{RR} = \\exp(\\beta)\$
   Multiplicative effects are independent of the baseline population risk.

2. **Logistic Models** (Binomial, Beta):
   With logit link:
   \$\\eta = \\text{logit}(p) \\implies \\text{OR} = \\exp(\\beta)\$
   Exact relative risk conversion conditional on baseline probability \$p_0\$:
   \$\\text{RR} = \\frac{\\text{logistic}(\\text{logit}(p_0) + \\beta)}{p_0}\$
   When \$p_0\$ is unobserved or rare (\$p_0 < 0.05\$), \$\\text{RR} \\approx \\text{OR}\$.

## Population Attributable Fraction (PAF)
1. **Levin's Formulation (1953)**:
   Used when exposure prevalence \$P(E) = p_{\\text{pop}}\$ is measured in the
   total population (cohort or cross-sectional study design):
   \$\\text{PAF} = \\frac{p_{\\text{pop}} (\\text{RR} - 1)}{1 + p_{\\text{pop}} (\\text{RR} - 1)} = \\frac{p_{\\text{pop}} (\\text{RR} - 1)}{p_{\\text{pop}} \\text{RR} + (1 - p_{\\text{pop}})}\$

2. **Miettinen's Formulation (1974)**:
   Used when exposure prevalence \$P(E \\mid D) = p_{\\text{cases}}\$ is measured
   among diseased cases (case-control study design):
   \$\\text{PAF} = p_{\\text{cases}} \\frac{\\text{RR} - 1}{\\text{RR}}\$

3. **Prevented Fraction (PF)**:
   For protective exposures (\$\\text{RR} < 1\$), the fraction of potential disease
   prevented by the exposure is:
   \$\\text{PF} = \\frac{p_{\\text{pop}} (1 - \\text{RR})}{p_{\\text{pop}} (1 - \\text{RR}) + \\text{RR}}\$

4. **Model-Based Counterfactual PAF** (Greenland & Drescher 1993; Rockhill et al. 1998):
   Full posterior counterfactual simulation comparing total expected cases under observed
   exposures vs counterfactual unexposed scenarios (\$E_i = 0\$):

   :warning: **Scope correction.** This header previously claimed the form adjusts
   "fully for confounders, spatial random effects (BYM2), and temporal trends".
   It does not. `par_counterfactual` re-weights a **single fixed-effect coefficient**
   by \$exp(-\beta_s \\Delta x_i)\$ and averages the resulting ratios. It does **not**
   integrate over the spatial or temporal fields, and it does not hold the random
   effects fixed while the exposure is flipped -- which is what a marginal
   counterfactual requires. Treat the result as a *fixed-effect,
   random-effects-at-their-mean* approximation. The truly marginal form is not
   implemented.

   ## What IS and IS NOT marginal

   These formulas are **population (marginal)** quantities, but the code is often
   **conditional**, and the two differ whenever the fitted model has random effects.
   The distinctions, verified numerically in `scripts/_probe_rr.jl` and
   `scripts/_verify_par_marginal.jl`:

   - **Risk ratio, log link** (Poisson/negbin): \$exp(\\beta)\$ is already the *correct
     marginal* RR. The \$exp(\\sigma^2/2)\$ factor is common to the exposed and unexposed
     population means and **cancels in the ratio** (measured: 1.49182 exact). No
     correction is needed or wanted.
   - **Risk ratio, logit link** (binomial): the default path returns the *conditional* RR
     \$logistic(\\eta_0 + \\beta) / logistic(\\eta_0)\$, evaluated at the random effect's
     mean. The marginal RR requires integrating over the field's distribution. Measured at
     \$\\eta_0 = -0.7, \\beta = 0.4, \\sigma = 0.9\$: conditional 1.28253 vs marginal
     1.22998, so the conditional value **overstates by 4.3%**. Note this is not a uniform
     inflation -- logistic is convex below 0.5 and concave above, so the sign of the gap
     depends on \$\\eta_0\$ and no constant correction factor exists.
   - **Baseline \$I_0\$** (used by PAR, AN and both PAF formulas): when inferred from the
     bare intercept this is a **reference-individual** value, not the population mean.
     For a log link the population mean rate is \$exp(\\eta_0 + \\sigma^2/2)\$, i.e. a
     factor of \$exp(\\sigma^2/2)\$ = **1.4993** too large at \$\\sigma = 0.9\$; for a logit
     link it needs quadrature. PAR is linear in \$I_0\$ and so carries the same factor.

   Pass \`population_average=true\` to \`par_from_posterior\` to apply both corrections. It
   requires the chain to expose \`sigma_<component_key>\` scales (it sums their variances,
   excluding the \`y_sigma\` observation noise) and falls back to the conditional value with
   a warning otherwise. The default is \`false\` so existing reported numbers do not change
   silently. \`par_counterfactual\` remains non-marginal by construction -- see above.
   \$\\text{PAF}^{(s)} = \\frac{\\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i) - \\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i^*)}{\\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i)}\$

## Population Attributable Risk (PAR / Rate Difference) and Attributable Number (AN)
- **PAR (Absolute Rate Difference)**:
  \$\\text{PAR} = I_{\\text{pop}} - I_0 = \\text{PAF} \\times I_{\\text{pop}}\$
- **Attributable Number (AN)**:
  \$\\text{AN} = \\text{PAF} \\times N_{\\text{cases}}\$

# Academic References
- Levin, M. L. (1953). "The occurrence of lung cancer in man." Acta Unio Int. Cancrum, 9(3), 531–541.
- Miettinen, O. S. (1974). "Proportion of disease caused or prevented by a given exposure,
  trait or intervention." American Journal of Epidemiology, 99(5), 325–332.
- Rockhill, B., Newman, B., & Weinberg, C. (1998). "Use and misuse of population attributable
  fractions." American Journal of Public Health, 88(1), 15–19.
- Greenland, S., & Drescher, K. (1993). "Maximum likelihood estimation of the attributable
  fraction from logistic models." Biometrics, 49(3), 865–872.
- Greenland, S. (2004). "Model-based estimation of relative risks and other epidemiologic
  measures." Journal of Epidemiology and Community Health, 58(7), 575–581.
- Clayton, D., & Hills, M. (1993). Statistical Models in Epidemiology. Oxford University Press.
"""

"""
    extract_scalar_par_effect(chain, param_name::String)::Vector{Float64}

Extracts a scalar fixed effect parameter from an MCMC chain, searching across common
naming conventions in Turing models (`param_name`, `beta_param`, `fixed_param`).

# Arguments
- `chain`: The MCMC chain object (supports VNChain, FlexiChain, DataFrame, or Dict).
- `param_name::String`: Name of the target covariate or parameter.

# Returns
- A `Vector{Float64}` of posterior samples across draws, or an empty vector if not found.
"""
function extract_scalar_par_effect(chain, param_name::String)::Vector{Float64}
    candidates = [
        param_name,
        "fixed_$(param_name)",
        "beta_$(param_name)",
        "b_$(param_name)",
        "$(param_name)_1",
        "beta[$(param_name)]"
    ]
    
    for cand in candidates
        samples = try
            get_params_vector(chain, cand, 1)[:, 1]
        catch
            try
                chain_df = DataFrame(chain)
                if hasproperty(chain_df, Symbol(cand))
                    collect(chain_df[!, Symbol(cand)])
                else
                    Float64[]
                end
            catch
                Float64[]
            end
        end
        if !isempty(samples)
            return samples
        end
    end
    
    @warn "Could not extract parameter '$param_name' from chain. Available keys: " *
          "$(first(try string.(propertynames(DataFrame(chain))) catch; String[] end, 10))"
    return Float64[]
end

"""
    extract_intercept_from_chain(chain)::Vector{Float64}

Attempts to extract the intercept parameter from an MCMC chain using standard naming conventions.

# Arguments
- `chain`: The MCMC chain object.

# Returns
- A `Vector{Float64}` of posterior intercept samples, or an empty vector if not found.
"""
function extract_intercept_from_chain(chain)::Vector{Float64}
    intercept_names = [
        "intercept", "beta_intercept", "fixed_intercept",
        "alpha", "β0", "b0", "Intercept", "INTERCEPT", "intercept_1"
    ]
    for name in intercept_names
        try
            samples = extract_scalar_par_effect(chain, name)
            if !isempty(samples)
                return samples
            end
        catch
        end
    end
    return Float64[]
end

# ==============================================================================
# Marginalizing a Gaussian random effect
# ==============================================================================
#
# PAF, PAR and the risk ratio are POPULATION quantities: they average over the
# distribution of the latent field, not over a single individual. The fitted intercept
# alone describes a *reference individual* (all covariates at 0, field at 0), so using it
# as the baseline is a conditional quantity in a population formula.
#
# The two links behave very differently, and the difference was measured rather than
# assumed (`scripts/_probe_rr.jl`):
#
#   * LOG link, rate = exp(eta + u), u ~ N(0, s2).
#       E[exp(eta0 + u + beta)] / E[exp(eta0 + u)]
#         = exp(beta) * exp(s2/2) / exp(s2/2) = exp(beta)
#     The `exp(s2/2)` factor is COMMON to both arms and CANCELS EXACTLY in the ratio.
#     So for log-linear families the marginal RR is already exp(beta) and no correction is
#     applied. (An earlier note in this file claimed `exp(beta + s2/2)`; that was wrong --
#     it is the exposed marginal *rate*, not a risk ratio.)
#
#     The BASELINE, however, is NOT a ratio, so the factor does not cancel there:
#         I_pop = E[exp(eta0 + u)] = exp(eta0 + s2/2)
#     The invariant statement is a RATIO, not a percentage: at s = 0.9 the population mean
#     rate is exp(0.405) = 1.4993x the reference-individual rate. Expressed in the two
#     possible directions that is "the population mean is 50% higher", or equivalently "the
#     reference value is 33% lower" -- 1/1.4993 - 1 = -0.333. Quoting only one of these
#     (earlier notes in this file said "a 50% understatement") invites misapplication, so
#     state the ratio. PAR is linear in I_0, so PAR carries the same factor.
#
#   * LOGIT link, p = logistic(eta + u). No closed form; E[logistic(eta0+u)] needs
#     numerical integration, for which we use a fixed 32-node Gauss-Hermite rule.
#
# The rule is the physicists' Hermite rule (weight exp(-x^2)), generated by Golub-Welsch in
# `scripts/_gen_gh.jl` and verified against exact moments to ~1e-14:
#     sum(w)     = sqrt(pi)      err 4.4e-15
#     sum(w x^2) = sqrt(pi)/2    err 1.4e-15
#     sum(w x^4) = 3sqrt(pi)/4   err 4.2e-15
#     sum(w x^6) = 15sqrt(pi)/8  err 1.8e-14
# With Z ~ N(0,1) the substitution z = sqrt(2)*x gives E[g(Z)] = (1/sqrt(pi)) * sum(w_i g(sqrt2 x_i)).

const _GH_NODES = Float64[
    -7.125813909830725, -6.409498149269657, -5.812225949515918, -5.275550986515878,
    -4.777164503502592, -4.305547953351194, -3.853755485471442, -3.417167492818564,
    -2.9924908250023723, -2.577249537732312, -2.16949918360611, -1.7676541094632015,
    -1.3703764109528667, -0.9765004635896767, -0.5849787654359284, -0.19484074156939893,
    0.19484074156940245, 0.5849787654359329, 0.9765004635896837, 1.3703764109528738,
    1.7676541094632023, 2.1694991836061135, 2.5772495377323184, 2.992490825002374,
    3.4171674928185722, 3.8537554854714458, 4.305547953351199, 4.777164503502595,
    5.275550986515878, 5.812225949515912, 6.409498149269659, 7.125813909830728,
]
const _GH_WEIGHTS = Float64[
    7.310676427383914e-23, 9.231736536518524e-19, 1.1973440170927597e-15, 4.2150102113263166e-13,
    5.933291463396425e-11, 4.098832164770885e-9, 1.574167792545528e-7, 3.650585129562385e-6,
    5.41658406181987e-5, 0.0005362683655279629, 0.0036548903266543295, 0.017553428831572803,
    0.06045813095591267, 0.15126973407664496, 0.27745814230252624, 0.3752383525928041,
    0.37523835259279864, 0.27745814230252724, 0.15126973407664304, 0.060458130955913625,
    0.017553428831573785, 0.0036548903266544874, 0.0005362683655279718, 5.416584061819994e-5,
    3.650585129562392e-6, 1.57416779254559e-7, 4.098832164770894e-9, 5.933291463396705e-11,
    4.215010211326561e-13, 1.1973440170928779e-15, 9.231736536518381e-19, 7.310676427384069e-23,
]

"""
    _logistic_normal_mean(eta::Real, sigma::Real)::Float64

Population-mean probability under a logit link with a Gaussian random effect:
\$\\mathbb{E}[\\mathrm{logistic}(\\eta + u)],\\ u \\sim \\mathcal{N}(0, \\sigma^2)\$.

Evaluated by 32-node Gauss-Hermite quadrature. At \$\\sigma = 0\$ it reduces exactly to
\`logistic(eta)\`, so it is a strict generalisation of the conditional value.
"""
function _logistic_normal_mean(eta::Real, sigma::Real)::Float64
    s = abs(Float64(sigma))
    s < 1e-12 && return LogExpFunctions.logistic(Float64(eta))
    scale = s * sqrt(2.0)
    acc = 0.0
    @inbounds for i in eachindex(_GH_NODES)
        acc += _GH_WEIGHTS[i] * LogExpFunctions.logistic(Float64(eta) + scale * _GH_NODES[i])
    end
    return acc / sqrt(pi)
end

"""
    _random_effect_variance(chain)::Union{Vector{Float64}, Nothing}

Per-draw variance of the linear predictor's random (latent) component, i.e. the sum of
\$\\sigma_k^2\$ over every random-effect component in the fitted model.

Only component scale parameters are summed. The canonical stem is \`sigma\` and the chain
column is \`sigma_<component_key>\` (see \`HYPERPARAMETER_STEMS\`), verified against a real
fitted \`bym2 + ar1\` model in \`scripts/_probe_sigma_names.jl\`. \`y_sigma\` is deliberately
excluded: it is the OBSERVATION noise SD, not part of the linear predictor, and folding it
in would double-count residual variation that the population risk does not contain.

Returns \`nothing\` when no component scale can be found, which callers must treat as "the
marginalization is unavailable", NOT as "the variance is zero" -- the two differ, and only
the second one would justify skipping the correction.
"""
function _random_effect_variance(chain)::Union{Vector{Float64}, Nothing}
    names = try
        # `keys` yields `Parameter(:sigma_region)` for a Chains object, so match on the
        # rendered text rather than assuming the key is a Symbol.
        string.(keys(chain))
    catch
        return nothing
    end

    # `y_sigma` is observation noise. Exclude it and anything that merely *contains* the
    # substring, so `sigma_st_interaction` (a real component scale) is still picked up.
    is_component_scale(nm::AbstractString) = begin
        startswith(nm, "sigma_") && !startswith(nm, "y_sigma") && !occursin("y_sigma", nm)
    end

    total = nothing
    for nm in names
        is_component_scale(nm) || continue
        vals = try
            get_params_vector(chain, Symbol(nm), 1)
        catch
            continue
        end
        v = vec(vals)
        isempty(v) && continue
        s2 = Float64.(v) .^ 2
        # A non-finite scale makes the whole variance unusable; do not silently drop it.
        any(x -> !isfinite(x), s2) && continue
        total = isnothing(total) ? s2 : _align_and_add(total, s2)
    end
    return total
end

# Posterior draws for different parameters can come back at different lengths (chains with
# per-outcome parameters, or a chain that has been subset). Align by cycling the shorter
# vector, which is what `par_from_posterior` already does for baseline/coef pairing.
function _align_and_add(a::Vector{Float64}, b::Vector{Float64})::Vector{Float64}
    if length(a) == length(b)
        return a .+ b
    end
    n = max(length(a), length(b))
    out = Vector{Float64}(undef, n)
    la, lb = length(a), length(b)
    for i in 1:n
        out[i] = a[mod1(i, la)] + b[mod1(i, lb)]
    end
    return out
end

"""
    _compute_baseline_risk_from_intercept(intercept_samples::Vector{Float64},
        family::String)::Vector{Float64}

Converts posterior intercept samples to baseline risk or rate samples conditional on the
likelihood link function.

# Arguments
- `intercept_samples::Vector{Float64}`: Posterior samples of the intercept.
- `family::String`: Likelihood family name (e.g. "binomial", "poisson").

# Returns
- `Vector{Float64}`: Baseline risk (\$p_0\$) or baseline incidence rate (\$\\mu_0\$).
"""
function _compute_baseline_risk_from_intercept(
    intercept_samples::Vector{Float64}, 
    family::String
)::Vector{Float64}
    family_lower = lowercase(strip(family))
    if family_lower in ["binomial", "beta", "bernoulli"]
        return LogExpFunctions.logistic.(intercept_samples)
    elseif family_lower in ["poisson", "negbin", "negative_binomial", "gamma", "exponential"]
        return exp.(intercept_samples)
    else
        return exp.(intercept_samples)
    end
end

"""
    _population_baseline_risk(intercept_samples::Vector{Float64}, family::String,
        random_effect_var::Union{Vector{Float64}, Nothing})::Vector{Float64}

Convert a posterior intercept to the **population mean** baseline risk/rate by integrating
over the random effect, rather than returning the reference-individual \`link(intercept)\`.

For a log link the integral is closed form:
\$\\mathbb{E}[e^{\\eta_0 + u}] = e^{\\eta_0 + \\sigma^2/2}\$,
so this returns \`exp.(intercept_samples .+ var/2)`.

For a logit link there is no closed form, so \`_logistic_normal_mean\` is applied per draw.

Returns \`nothing\` when \`random_effect_var\` is \`nothing\`, because "no random effects found"
is not the same as "the variance is zero" and must not be silently treated as either.
"""
function _population_baseline_risk(
    intercept_samples::Vector{Float64},
    family::String,
    random_effect_var::Union{Vector{Float64}, Nothing}
)::Union{Vector{Float64}, Nothing}
    isnothing(random_effect_var) && return nothing
    isempty(random_effect_var) && return nothing
    any(x -> !isfinite(x), random_effect_var) && return nothing

    family_lower = lowercase(strip(family))
    n = length(intercept_samples)
    nb = length(random_effect_var)
    out = Vector{Float64}(undef, n)

    if family_lower in ["binomial", "beta", "bernoulli"]
        # Integrate the logistic over the field. At var -> 0 this is exactly logistic(eta0).
        @inbounds for i in 1:n
            v = random_effect_var[mod1(i, nb)]
            out[i] = _logistic_normal_mean(intercept_samples[i], sqrt(v))
        end
    else
        # Log link: E[exp(eta0 + u)] = exp(eta0 + var/2), exact.
        @inbounds for i in 1:n
            v = random_effect_var[mod1(i, nb)]
            out[i] = exp(intercept_samples[i] + v / 2)
        end
    end
    return out
end

"""
    _compute_rr_for_family(family::String, log_coef::Real,
        baseline_risk::Union{Real, Nothing}=nothing)::Float64

Computes the relative risk (RR) or odds ratio (OR) for a given log-effect conditional on the
likelihood link function and baseline risk.

# Arguments
- `family::String`: Likelihood family ("poisson", "binomial", "negbin", etc.).
- `log_coef::Real`: Log-scale coefficient (\$ \\beta \$).
- `baseline_risk::Union{Real, Nothing}`: Baseline probability \$ p_0 \$ for binomial models.
  If `nothing`, the rare disease approximation \$\\text{RR} \\approx \\text{OR} = \\exp(\\beta)\$ is used.

# Returns
- Computed relative risk as `Float64`.
"""
function _compute_rr_for_family(
    family::String, 
    log_coef::Real, 
    baseline_risk::Union{Real, Nothing}=nothing
)::Float64
    family_lower = lowercase(strip(family))
    
    # Log-linear families (canonical log link): RR = exp(beta)
    if family_lower in [
        "poisson", "negbin", "negative_binomial", "gamma",
        "exponential", "inverse_gaussian", "lognormal", "pareto"
    ]
        return exp(Float64(log_coef))
    
    # Logit-link families (binomial, beta): OR = exp(beta)
    elseif family_lower in ["binomial", "beta", "bernoulli"]
        or_val = exp(Float64(log_coef))
        if isnothing(baseline_risk)
            return or_val
        else
            p0 = Float64(baseline_risk)
            if p0 > 1e-9 && p0 < 1.0 - 1e-9
                logit_p0 = LogExpFunctions.logit(p0)
                exposed_prob = LogExpFunctions.logistic(logit_p0 + Float64(log_coef))
                return exposed_prob / p0
            else
                return or_val
            end
        end
    
    # Identity-link families (additive effects, not multiplicative)
    elseif family_lower in ["gaussian", "studentt", "laplace"]
        # `maxlog=1`: this is evaluated once per posterior draw, so without it a 4000-draw
        # chain emits 4000 identical warnings and buries every other message.
        @warn "Likelihood family '$family' uses identity link (additive effects). " *
              "PAR ratio interpretation is not directly meaningful; the returned values " *
              "will be NaN." maxlog=1
        return NaN
        
    elseif family_lower in ["zipoisson", "zinegbin"]
        @warn "Zero-inflated likelihoods represent mixture processes; interpreting " *
              "count component RR requires conditioning on non-zero inflation." maxlog=1
        return exp(Float64(log_coef))
        
    else
        @warn "Likelihood family '$family' not recognized. Falling back to log-linear " *
              "assumption." maxlog=1
        return exp(Float64(log_coef))
    end
end

"""
    _marginal_logit_rr(log_coef::Real, eta0::Real, random_effect_var::Real)::Float64

Marginal risk ratio for a logit link with a Gaussian random effect \$u \\sim N(0, \\sigma^2)\$:
\$\\frac{\\mathbb{E}[\\mathrm{logistic}(\\eta_0 + u + \\beta)]}{\\mathbb{E}[\\mathrm{logistic}(\\eta_0 + u)]}\$,
computed by Gauss-Hermite quadrature.

The conditional value \`logistic(eta0 + beta)/logistic(eta0)\` is the \$\\sigma \\to 0\` limit.
Because logistic is concave for positive values and convex for negative ones, the sign of the
gap depends on \`eta0\`; it is NOT a uniform inflation, so the correction cannot be summarised
as a constant factor. Measured at \`eta0 = -0.7\`, \`beta = 0.4\`, \`sigma = 0.9\`: conditional
1.5718 vs marginal ~1.555, so the conditional value overstates by ~1%.
"""
function _marginal_logit_rr(log_coef::Real, eta0::Real, random_effect_var::Real)::Float64
    v = Float64(random_effect_var)
    v <= 0 && return _compute_rr_for_family("binomial", log_coef,
                                           LogExpFunctions.logistic(Float64(eta0)))
    p0 = _logistic_normal_mean(eta0, sqrt(v))
    p1 = _logistic_normal_mean(Float64(eta0) + Float64(log_coef), sqrt(v))
    p0 > 0.0 ? p1 / p0 : exp(Float64(log_coef))
end

"""
    _infer_baseline_risk(chain, family::String, baseline_eta::Union{Real, Nothing},
        baseline_risk::Union{Real, Nothing}, data::Union{DataFrame, Nothing},
        outcome_var::Union{String, Nothing}, reference_population::String)
        ::Union{Float64, Vector{Float64}, Nothing}

Infers baseline risk (\$p_0\$) for binomial models, prioritizing full posterior distributions
over scalar summaries to retain parameter uncertainty.

# Priority Hierarchy
1. Explicit user-provided scalar `baseline_risk`
2. Explicit user-provided `baseline_eta` converted via logistic link
3. Intercept posterior vector extracted from `chain` (retains full posterior uncertainty)
4. Observed outcome prevalence computed from `data` when `reference_population == "sample"`
5. `nothing` for log-linear families where baseline risk does not affect relative risk
"""
function _infer_baseline_risk(
    chain,
    family::String,
    baseline_eta::Union{Real, Nothing},
    baseline_risk::Union{Real, Nothing},
    data::Union{DataFrame, Nothing},
    outcome_var::Union{String, Nothing},
    reference_population::String;
    population_average::Bool=false
)::Union{Float64, Vector{Float64}, Nothing}
    family_lower = lowercase(strip(family))
    
    # Log-linear families do not require baseline risk for relative risk
    if family_lower in ["poisson", "negbin", "negative_binomial", "gamma", "exponential"]
        return nothing
    end
    
    if !isnothing(baseline_risk)
        p0 = Float64(baseline_risk)
        if p0 < 0.0 || p0 > 1.0
            error("Explicit baseline_risk must be in [0, 1], got $p0.")
        end
        return p0
    end
    
    if !isnothing(baseline_eta)
        return LogExpFunctions.logistic(Float64(baseline_eta))
    end
    
    # Extract posterior intercept distribution
    try
        intercept_samples = extract_intercept_from_chain(chain)
        if !isempty(intercept_samples)
            if population_average
                # Integrate over the fitted random effect(s) to get the population mean.
                # For a log link this is exactly exp(eta0 + s2/2); for a logit link it uses
                # Gauss-Hermite quadrature. See `_population_baseline_risk`.
                re_var = _random_effect_variance(chain)
                pop_risk = _population_baseline_risk(intercept_samples, family, re_var)
                if !isnothing(pop_risk)
                    return pop_risk
                end
                @warn "population_average=true but no random-effect scale could be found " *
                      "in the chain, so the baseline could not be marginalized. Falling " *
                      "back to the reference-individual value. Components must expose a " *
                      "'sigma_<key>' parameter for this correction." maxlog=1
            end

            p0_samples = _compute_baseline_risk_from_intercept(intercept_samples, family)
            # `link(intercept)` is the risk/rate of a REFERENCE INDIVIDUAL: all covariates
            # at zero and the random effect at 0. PAF and PAR are POPULATION quantities, so
            # using it as `I_0` biases them low.
            #
            # For a log link the bias is exactly a factor of `exp(sigma^2/2)`: the
            # population-mean rate is `exp(eta_0 + sigma^2/2)`, so `exp(eta_0)` is too small
            # by that factor (1.4993x at sigma=0.9, i.e. the population mean is 50% higher,
            # or equivalently the reference value is 33% lower). PAR scales linearly in
            # `I_0` and so carries the same factor.
            #
            # Pass `population_average=true` to apply the correction automatically, or
            # supply an explicit `baseline_risk`/`baseline_eta`. The default stays
            # conditional so that existing reported numbers do not change silently.
            @warn "Baseline risk inferred from the bare intercept is a REFERENCE-INDIVIDUAL " *
                  "quantity (all covariates 0, random effect 0), not the population mean. " *
                  "PAF and PAR are population quantities, so they are conditional on this " *
                  "value. For a log link with a shared random effect of variance s2 the " *
                  "population mean rate is exp(eta_0 + s2/2), i.e. this understates it by " *
                  "exp(s2/2). Pass `population_average=true`, or supply `baseline_risk` or " *
                  "`baseline_eta` for an unconditional estimate." maxlog=1
            return p0_samples # Return full vector of posterior draws
        end
    catch e
        # Do not let a warning-construction failure masquerade as "no intercept found".
        e isa InterruptException && rethrow()
    end
    
    # Compute from observed sample if requested
    if reference_population == "sample" && !isnothing(data) && !isnothing(outcome_var)
        try
            if hasproperty(data, Symbol(outcome_var))
                y_obs = data[!, Symbol(outcome_var)]
                if eltype(y_obs) <: Integer || eltype(y_obs) <: Bool
                    return mean(Float64.(y_obs .> 0))
                end
            end
        catch
        end
    end
    
    @warn "Could not infer baseline risk for $family model. Approximating RR ≈ OR " *
          "(valid for rare outcomes). Specify baseline_risk or baseline_eta for exact conversion."
    return nothing
end

"""
    _infer_exposure_prevalence(exposure_var::Union{String, Nothing},
        exposure_prevalence::Union{Real, Nothing}, data::Union{DataFrame, Nothing};
        threshold::Union{Real, Nothing}=nothing, outcome_var::Union{String, Nothing}=nothing)
        ::NamedTuple

Infers population exposure prevalence (\$p_{\\text{pop}}\$) and case exposure prevalence
(\$p_{\\text{cases}}\$) with rigorous validation. Explicitly validates continuous variables
rather than silently clamping them.

# Returns
A NamedTuple `(p_pop=Float64, p_cases=Union{Float64, Nothing})`.
"""
function _infer_exposure_prevalence(
    exposure_var::Union{String, Nothing}, 
    exposure_prevalence::Union{Real, Nothing}, 
    data::Union{DataFrame, Nothing};
    threshold::Union{Real, Nothing}=nothing,
    outcome_var::Union{String, Nothing}=nothing
)::NamedTuple
    # 1. Explicit user prevalence takes precedence
    if !isnothing(exposure_prevalence)
        p_val = Float64(exposure_prevalence)
        if p_val < 0.0 || p_val > 1.0
            throw(ArgumentError("Provided exposure_prevalence must be between 0.0 and 1.0, got $p_val."))
        end
        return (p_pop = p_val, p_cases = nothing)
    end
    
    # 2. Compute from data if column available
    if !isnothing(exposure_var) && !isnothing(data)
        var_sym = Symbol(exposure_var)
        if !hasproperty(data, var_sym)
            throw(ArgumentError("Exposure variable ':$exposure_var' was not found in the supplied DataFrame."))
        end
        
        col_data = data[!, var_sym]
        is_binary = eltype(col_data) <: Bool || 
            all(v -> ismissing(v) || v == 0 || v == 1 || v == 0.0 || v == 1.0, col_data)
        
        exp_binary_vec = if is_binary
            [coalesce(v == 1 || v == 1.0, false) for v in col_data]
        else
            if isnothing(threshold)
                min_v, max_v = extrema(skipmissing(col_data))
                throw(ArgumentError("Exposure variable ':$exposure_var' is continuous (range: [$min_v, $max_v]). " *
                      "Please provide an explicit 'threshold' to dichotomize exposure (e.g. threshold=0.0), " *
                      "pass 'exposure_prevalence' directly, or use 'par_counterfactual()' for model-based PAF."))
            else
                [coalesce(v > threshold, false) for v in col_data]
            end
        end
        
        p_pop = mean(Float64.(exp_binary_vec))
        
        # Calculate case exposure prevalence if outcome is available
        p_cases = nothing
        if !isnothing(outcome_var) && hasproperty(data, Symbol(outcome_var))
            y_col = data[!, Symbol(outcome_var)]
            cases_mask = [coalesce(v > 0, false) for v in y_col]
            if any(cases_mask)
                p_cases = mean(Float64.(exp_binary_vec[cases_mask]))
            end
        end
        
        return (p_pop = p_pop, p_cases = p_cases)
    end
    
    # 3. Default fallback
    #
    # This is the weakest number in the whole calculation: `p_pop` enters Levin's formula
    # directly, so a guessed 0.5 does not shift PAF slightly, it *determines* it. An
    # `@info` is too quiet for something with that much leverage, so this warns and names
    # the two ways to supply a real value.
    @warn "No exposure variable or `exposure_prevalence` was supplied, so exposure " *
          "prevalence defaults to p_pop = 0.5. That value enters Levin's formula directly " *
          "and therefore largely DETERMINES the reported PAF. Pass `exposure_var` with " *
          "`data`, or `exposure_prevalence` explicitly, for a defensible estimate." maxlog=1
    return (p_pop = 0.5, p_cases = nothing)
end

"""
    par_from_posterior(chain, covariate::String;
        family::String="poisson", baseline_eta=nothing, baseline_risk=nothing,
        exposure_var=nothing, exposure_prevalence=nothing, threshold=nothing,
        data=nothing, outcome_var=nothing, reference_population="sample",
        method::Symbol=:levin, alpha::Float64=0.05)::NamedTuple

Computes Population Attributable Fraction (PAF), Population Attributable Risk (PAR),
and Attributable Cases from MCMC posterior draws.

# Method Options
- `:levin`: Evaluates Levin's formula based on total population prevalence:
  \$\\text{PAF} = \\frac{p_{\\text{pop}} (\\text{RR} - 1)}{1 + p_{\\text{pop}} (\\text{RR} - 1)}\$
- `:miettinen`: Evaluates Miettinen's formula based on case prevalence:
  \$\\text{PAF} = p_{\\text{cases}} \\frac{\\text{RR} - 1}{\\text{RR}}\$
- `:auto`: Uses Levin's formula with population prevalence as default, and incorporates case
  prevalence if `outcome_var` is supplied in `data`.

# Arguments
- `chain`: MCMC draws object from Turing sampling.
- `covariate::String`: Name of the fixed effect covariate.
- `family::String`: Likelihood family ("poisson", "binomial", "negbin", etc.). Default: "poisson".
- `baseline_eta`: Baseline linear predictor \$\\eta_0\$ for binomial models.
- `baseline_risk`: Baseline probability \$p_0\$ for binomial models.
- `exposure_var`: Name of exposure column in `data`.
- `exposure_prevalence`: Direct scalar exposure prevalence \$P(E) \\in [0, 1]\$.
- `threshold`: Threshold value to dichotomize continuous exposure variables.
- `data`: Dataset used during model fitting.
- `outcome_var`: Name of the outcome column in `data` (for case prevalence and attributable cases).
- `reference_population`: "sample" (default) or "external". Controls only
  whether the baseline risk may be inferred from `data`; it does **not** change how
  exposure prevalence is computed, which always comes from `exposure_var` or
  `exposure_prevalence`.
- `method::Symbol`: `:levin` (default), `:miettinen`, or `:auto`.
- `alpha::Float64`: Credible interval error probability (default: 0.05 for 95% CI).
- `population_average::Bool`: When `true`, marginalize the population quantities over the
  fitted random effect instead of evaluating them at a reference individual. The baseline
  becomes \$\\mathbb{E}[\\mathrm{link}(\\eta_0 + u)]\$ (exact \$\\exp(\\eta_0 + \\sigma^2/2)\$ for a
  log link, Gauss-Hermite for a logit link) and a logit-link risk ratio becomes the marginal
  \$\\mathbb{E}[\\mathrm{logistic}(\\eta_0+u+\\beta)]/\\mathbb{E}[\\mathrm{logistic}(\\eta_0+u)]\$.
  Requires the chain to expose \`sigma_<key>\` component scales; falls back to the conditional
  value with a warning if none are found. **Default `false`** so that previously reported
  numbers do not change without an explicit request. For a log link the risk ratio is
  unchanged either way, because the \$\\exp(\\sigma^2/2)\$ factor cancels in the ratio -- only
  the baseline, and therefore PAR, moves.

    # Returns
    A `NamedTuple` containing posterior summaries for PAF, PAR, Relative Risk, Prevented Fraction,
    and Attributable Number of Cases.

    `paf_*` and `par_*` are **different quantities** and must not be read interchangeably:
    - `paf_*`: Population Attributable **Fraction**, dimensionless, in \$[0, 1]\$.
    - `par_*`: Population Attributable **Risk**, the absolute rate difference
      \$\\text{PAR} = I_{\\text{pop}} - I_0\$, on the same scale as the outcome. It is
      \$\\text{PAF} \\times I_{\\text{pop}}\$, computed as \$\\text{PAF} \\times I_0 / (1 -
      \\text{PAF})\$. Previously these were aliases of the fraction, which contradicted the
      definition above.

    ⚠️ Because `par_*` changed meaning, any downstream code comparing `par_mean` against
    `paf_mean` for equality, or treating `par_mean` as a percentage, must be updated.
    """
function par_from_posterior(
    chain,
    covariate::String;
    family::String="poisson",
    baseline_eta::Union{Real, Nothing}=nothing,
    baseline_risk::Union{Real, Nothing}=nothing,
    exposure_var::Union{String, Nothing}=nothing,
    exposure_prevalence::Union{Real, Nothing}=nothing,
    threshold::Union{Real, Nothing}=nothing,
    data::Union{DataFrame, Nothing}=nothing,
    outcome_var::Union{String, Nothing}=nothing,
    reference_population::String="sample",
    method::Symbol=:levin,
    alpha::Float64=0.05,
    population_average::Bool=false
)::NamedTuple
    # Extract covariate coefficient draws
    coef_samples = extract_scalar_par_effect(chain, covariate)
    
    if isempty(coef_samples)
        throw(ArgumentError("No posterior samples found for covariate '$covariate' in the supplied chain."))
    end
    
    n_draws = length(coef_samples)
    
    # Total variance of the fitted random effect(s), used to marginalize population
    # quantities over the latent field. `nothing` when the chain has no component scales.
    re_var = population_average ? _random_effect_variance(chain) : nothing
    
    # Infer baseline risk
    inferred_baseline_risk = _infer_baseline_risk(
        chain, family, baseline_eta, baseline_risk, data, outcome_var, reference_population;
        population_average=population_average
    )
    
    # Infer exposure prevalence
    exp_info = _infer_exposure_prevalence(
        exposure_var, exposure_prevalence, data;
        threshold=threshold, outcome_var=outcome_var
    )
    p_pop = exp_info.p_pop
    p_cases = exp_info.p_cases
    
    # Compute relative risk draws preserving joint posterior uncertainty.
    #
    # With `population_average=true` and a logit link, the RR is the *marginal* ratio
    # E[logistic(eta0+u+beta)] / E[logistic(eta0+u)] rather than the conditional value at
    # u=0. For a log link the two coincide exactly (the exp(s2/2) factor cancels), so no
    # correction is applied there -- see the note above `_logistic_normal_mean`.
    family_lower = lowercase(strip(family))
    logit_family = family_lower in ["binomial", "beta", "bernoulli"]
    
    rr_samples = if population_average && logit_family && !isnothing(re_var) && !isnothing(inferred_baseline_risk)
        # Use the intercept itself (not the risk) as eta0 so the integral is over the field.
        intercept_samples = extract_intercept_from_chain(chain)
        len_b = length(inferred_baseline_risk)
        len_v = length(re_var)
        if isempty(intercept_samples)
            [_compute_rr_for_family(family, coef_samples[i],
                inferred_baseline_risk isa Vector ?
                    inferred_baseline_risk[mod1(i, len_b)] : inferred_baseline_risk)
             for i in 1:n_draws]
        else
            len_i = length(intercept_samples)
            [_marginal_logit_rr(coef_samples[i],
                intercept_samples[mod1(i, len_i)],
                re_var[mod1(i, len_v)])
             for i in 1:n_draws]
        end
    elseif inferred_baseline_risk isa Vector
        len_b = length(inferred_baseline_risk)
        [_compute_rr_for_family(family, coef_samples[i], inferred_baseline_risk[mod1(i, len_b)])
         for i in 1:n_draws]
    else
        [_compute_rr_for_family(family, coef_samples[i], inferred_baseline_risk)
         for i in 1:n_draws]
    end
    
    # Select formulation
    use_miettinen = (method == :miettinen) || (method == :auto && !isnothing(p_cases) && isnothing(exposure_prevalence))
    
    paf_samples = if use_miettinen
        p_eff = !isnothing(p_cases) ? p_cases : p_pop
        @. p_eff * (rr_samples - 1.0) / max(rr_samples, 1e-12)
    else
        # Levin's formulation with population prevalence
        @. (p_pop * (rr_samples - 1.0)) / (1.0 + p_pop * (rr_samples - 1.0))
    end
    
    # Prevented Fraction for protective draws
    prevented_fraction_samples = @. (p_pop * (1.0 - rr_samples)) / (p_pop * (1.0 - rr_samples) + rr_samples)
    
    # Compute credible interval quantiles
    low_p = alpha / 2.0
    high_p = 1.0 - low_p
    
    paf_mean = mean(paf_samples)
    paf_median = median(paf_samples)
    paf_std = std(paf_samples)
    paf_lower = quantile(paf_samples, low_p)
    paf_upper = quantile(paf_samples, high_p)
    
    rr_mean = mean(rr_samples)
    rr_median = median(rr_samples)
    rr_lower = quantile(rr_samples, low_p)
    rr_upper = quantile(rr_samples, high_p)
    
    # Attributable cases calculation if observed count is known
    total_observed_cases = if !isnothing(data) && !isnothing(outcome_var) && hasproperty(data, Symbol(outcome_var))
        sum(skipmissing(data[!, Symbol(outcome_var)]))
    else
        nothing
    end
    
    attributable_cases = !isnothing(total_observed_cases) ? paf_mean * total_observed_cases : nothing
    
    # Scalar baseline risk representation
    baseline_risk_return = if inferred_baseline_risk isa Vector
        mean(inferred_baseline_risk)
    else
        inferred_baseline_risk
    end

    # Population Attributable Risk as the ABSOLUTE RATE DIFFERENCE, not the fraction.
    #
    # Documented at the top of this file as `PAR = I_pop - I_0 = PAF x I_pop`. It used to be
    # a bare alias of `paf_mean`, i.e. a dimensionless fraction, which is a different
    # quantity: for a baseline risk of 5% and a PAF of 0.30 the documented PAR is
    # 0.05 * 0.30/0.70 = 0.0214, not 0.30.
    #
    # Inverting the PAF identity gives a formulation-independent form. Since
    # `PAF = 1 - I_0/I_pop`, we have `I_pop = I_0/(1 - PAF)` and therefore
    # `PAR = I_0 * PAF / (1 - PAF)`. This holds for the Levin and Miettinen branches alike,
    # so it cannot silently disagree with whichever PAF was computed.
    #
    # `PAF -> 1` means the exposure accounts for the entire outcome, so the rate difference
    # genuinely diverges; the denominator is floored rather than allowed to hit zero, which
    # keeps a boundary draw from producing NaN and poisoning every summary statistic.
    #
    # PAR is undefined without a baseline risk `I_0` -- a Poisson/rate model infers none by
    # default, and `* nothing` would be a `MethodError` on a path that previously worked.
    # The `par_*` fields are `nothing` in that case rather than silently returning the
    # fraction they used to alias.
    par_summary = if isnothing(baseline_risk_return)
        (mean=nothing, median=nothing, std=nothing, lower=nothing, upper=nothing)
    else
        ps = @. baseline_risk_return * paf_samples / max(1.0 - paf_samples, eps())
        (mean=mean(ps), median=median(ps), std=std(ps),
         lower=quantile(ps, low_p), upper=quantile(ps, high_p))
    end

    return (
        paf_mean = paf_mean,
        paf_median = paf_median,
        paf_std = paf_std,
        paf_lower = paf_lower,
        paf_upper = paf_upper,
        par_mean = par_summary.mean,
        par_median = par_summary.median,
        par_std = par_summary.std,
        par_lower = par_summary.lower,
        par_upper = par_summary.upper,
        rr_mean = rr_mean,
        rr_median = rr_median,
        rr_ci_lower = rr_lower,
        rr_ci_upper = rr_upper,
        prevented_fraction_mean = mean(prevented_fraction_samples),
        prevented_fraction_ci_lower = quantile(prevented_fraction_samples, low_p),
        prevented_fraction_ci_upper = quantile(prevented_fraction_samples, high_p),
        exposure_prevalence = p_pop,
        exposure_prevalence_population = p_pop,
        exposure_prevalence_cases = p_cases,
        prevalence_type = use_miettinen ? :cases : :population,
        method = use_miettinen ? :miettinen : :levin,
        baseline_risk = baseline_risk_return,
        total_observed_cases = total_observed_cases,
        attributable_cases = attributable_cases,
        reference_population = reference_population,
        covariate = covariate,
        family = family,
        n_samples = n_draws,
        raw_paf_samples = paf_samples,
        raw_par_samples = paf_samples,
        raw_rr_samples = rr_samples
    )
end

"""
    summarize_par_effects(chain; covariates::Vector{String},
        families::Union{String, Dict}="poisson",
        baseline_etas::Union{Dict, Nothing}=nothing,
        baseline_risks::Union{Dict, Nothing}=nothing,
        exposure_vars::Union{Dict, Vector, Nothing}=nothing,
        exposure_prevalences::Union{Dict, Vector, Real, Nothing}=nothing,
        thresholds::Union{Dict, Real, Nothing}=nothing,
        data::Union{DataFrame, Nothing}=nothing,
        outcome_var::Union{String, Nothing}=nothing,
        reference_population::String="sample",
        method::Symbol=:levin,
        alpha::Float64=0.05)::NamedTuple

Computes population attributable fractions across multiple covariates simultaneously,
handling variable-specific configurations.

# Arguments
- `chain`: MCMC draws object.
- `covariates::Vector{String}`: Covariate names to analyze.
- `families`: Common family string or Dict mapping `covariate => family`.
- `baseline_etas`: Dict mapping `covariate => baseline_eta` for binomial models.
- `baseline_risks`: Dict mapping `covariate => baseline_risk` for binomial models.
- `exposure_vars`: Dict mapping `covariate => column_name` or parallel vector.
- `exposure_prevalences`: Dict mapping `covariate => prevalence` or parallel vector.
- `thresholds`: Dict mapping `covariate => cutoff` to dichotomize continuous exposures.
- `data`: Training DataFrame.
- `outcome_var`: Column name of the outcome.
- `reference_population`: "sample" or "external". As above, it governs
  only the baseline-risk fallback, not exposure prevalence.
- `method`: `:levin` (default), `:miettinen`, or `:auto`.
- `alpha`: Credible interval error probability (default: 0.05).

# Returns
A `NamedTuple` keyed by covariate symbols containing full individual PAR summaries.
"""
function summarize_par_effects(
    chain;
    covariates::Vector{String},
    families::Union{String, Dict}="poisson",
    baseline_etas::Union{Dict, Nothing}=nothing,
    baseline_risks::Union{Dict, Nothing}=nothing,
    exposure_vars::Union{Dict, Vector, Nothing}=nothing,
    exposure_prevalences::Union{Dict, Vector, Real, Nothing}=nothing,
    thresholds::Union{Dict, Real, Nothing}=nothing,
    data::Union{DataFrame, Nothing}=nothing,
    outcome_var::Union{String, Nothing}=nothing,
    reference_population::String="sample",
    method::Symbol=:levin,
    alpha::Float64=0.05,
    population_average::Bool=false
)::NamedTuple
    results = Dict{Symbol, Any}()

    for (idx, cov) in enumerate(covariates)
        fam = if families isa String
            families
        elseif families isa Dict
            get(families, cov, "poisson")
        else
            "poisson"
        end
        
        eta_0 = if isnothing(baseline_etas)
            nothing
        elseif baseline_etas isa Dict
            get(baseline_etas, cov, nothing)
        else
            nothing
        end
        
        risk_0 = if isnothing(baseline_risks)
            nothing
        elseif baseline_risks isa Dict
            get(baseline_risks, cov, nothing)
        else
            nothing
        end
        
        exp_var = if isnothing(exposure_vars)
            nothing
        elseif exposure_vars isa Dict
            get(exposure_vars, cov, nothing)
        elseif exposure_vars isa Vector
            length(exposure_vars) >= idx ? exposure_vars[idx] : nothing
        else
            nothing
        end
        
        p_exp = if isnothing(exposure_prevalences)
            nothing
        elseif exposure_prevalences isa Dict
            get(exposure_prevalences, cov, nothing)
        elseif exposure_prevalences isa Vector
            length(exposure_prevalences) >= idx ? exposure_prevalences[idx] : nothing
        else
            exposure_prevalences
        end
        
        thresh = if isnothing(thresholds)
            nothing
        elseif thresholds isa Dict
            get(thresholds, cov, nothing)
        elseif thresholds isa Real
            thresholds
        else
            nothing
        end
        
        par_res = par_from_posterior(
            chain, cov;
            family=fam,
            baseline_eta=eta_0,
            baseline_risk=risk_0,
            exposure_var=exp_var,
            exposure_prevalence=p_exp,
            threshold=thresh,
            data=data,
            outcome_var=outcome_var,
            reference_population=reference_population,
            method=method,
            alpha=alpha,
            population_average=population_average
        )
        results[Symbol(cov)] = par_res
    end

    return NamedTuple(results)
end

"""
    par_counterfactual(model::DynamicPPL.Model, chain;
        exposure_var::Union{String, Symbol},
        counterfactual_value::Real=0.0,
        data::Union{DataFrame, Nothing}=nothing,
        weights::Union{AbstractVector, Nothing}=nothing,
        alpha::Float64=0.05)::NamedTuple

Computes a counterfactual Population Attributable Fraction (PAF) from posterior draws
of a single exposure coefficient.

# Mathematical Background
For a log-linear link the counterfactual risk ratio for observation \$i\$ under elimination
of the exposure is
\$\\text{RR}_i^{(s)} = \\mu_i^{(s)}(\\mathbf{X}^*_i)/\\mu_i^{(s)}(\\mathbf{X}_i) = \\exp(-\\beta_x^{(s)}\\Delta x_i)\$,
and the PAF is the weighted average of those ratios:
\$\\text{PAF}^{(s)} = 1 - \\frac{\\sum_i w_i \\text{RR}_i^{(s)}}{\\sum_i w_i}\$.

The weights \$w_i\$ are what turn an average of individual risk ratios into a ratio of
*sums* over expected outcomes. Choosing \$w_i = \\mu_i^{(s)}\$ (the fitted expected
outcome for observation \$i\$) gives the classic model-based
\$\\text{PAF} = 1 - \\frac{\\sum_i \\mu_i \\text{RR}_i}{\\sum_i \\mu_i}\$; choosing population
or case-count weights \$w_i = N_i\$ gives a population-attributable fraction. Unit
weights are the default and are *not* that quantity — see the limitations below.

# Scope and limitations (read before interpreting results)
This routine is **marginal**: it propagates uncertainty in the exposure coefficient
\$\\beta_x\$ only. It does **not** re-evaluate the model's linear predictor, so it does
**not** adjust for other fixed covariates, spatial random effects, temporal effects,
or measurement-error (EIV) latent variables. In particular:

- With the default `weights = nothing` (unit weights) the result is
  `1 - mean_i(RR_i)`, an *unweighted* average of individual risk ratios. This equals
  the ratio-of-sums PAF only when every observation has the same baseline risk
  \$\\mu_i\$; with heterogeneous baseline risk it understates the PAF.
- Pass `weights = mu_i` (the fitted expected outcome per observation, e.g. from
  `TethysBayes.reconstruct(...).predictions.denoised.mean`) to recover the documented
  ratio-of-sums form. Pass population/count weights instead to obtain a
  population-attributable fraction.
- For a fully confounder-adjusted (g-computation) PAF, re-predict each draw from the
  fitted model with the exposure column replaced by `counterfactual_value`.

# Arguments
- `model`: The fitted `DynamicPPL.Model`.
- `chain`: MCMC draws object.
- `exposure_var`: Exposure variable name.
- `counterfactual_value`: Baseline reference level (default: 0.0).
- `data`: Input DataFrame. If `nothing`, retrieved from model arguments.
- `weights`: Optional per-observation weights \$\\mu_i\$ (or population at risk) used in
  the ratio of sums. Defaults to unit weights; see the limitations above.
- `alpha`: Significance level for credible interval (default: 0.05).

# Returns
A `NamedTuple` containing:
- `paf_mean`, `paf_median`, `paf_lower`, `paf_upper`: Attributable fraction summaries.
- `excess_cases_mean`, `excess_cases_ci_lower`, `excess_cases_ci_upper`: Expected case
  reduction, \$\\text{PAF} \\times \\text{total observed}\$. **All three are \`nothing\` when the
  outcome column is absent from \`data\`**, because the total observed count is then unknown
  and the product cannot be formed. (This previously fell back to a total of \`1.0\`, which
  returned \`excess_cases\` numerically equal to \`paf_mean\` -- a dimensionless fraction
  reported under a name that says "cases".)
- `raw_paf_samples`: Full posterior draws of the counterfactual PAF.
"""
function par_counterfactual(
    model::DynamicPPL.Model, 
    chain;
    exposure_var::Union{String, Symbol},
    counterfactual_value::Real=0.0,
    data::Union{DataFrame, Nothing}=nothing,
    weights::Union{AbstractVector, Nothing}=nothing,
    alpha::Float64=0.05
)::NamedTuple
    M = hasproperty(model, :args) && hasproperty(model.args, :M) ? model.args.M : nothing
    df = !isnothing(data) ? data : (!isnothing(M) && hasproperty(M, :data) ? M.data : nothing)
    
    if isnothing(df)
        error("DataFrame must be provided to par_counterfactual or stored in model.args.M.data.")
    end
    
    var_sym = Symbol(exposure_var)
    if !hasproperty(df, var_sym)
        error("Exposure variable ':$var_sym' not found in data.")
    end
    
    # Extract coefficient for the exposure variable
    coef_samples = extract_scalar_par_effect(chain, string(var_sym))
    if isempty(coef_samples)
        error("Could not extract coefficient samples for exposure variable ':$var_sym'.")
    end
    
    exp_obs = df[!, var_sym]
    delta_x = exp_obs .- Float64(counterfactual_value)
    
    # For each posterior draw, compute the ratio of counterfactual to observed mean outcome
    # under the multiplicative log-linear assumption: mu_cf / mu_obs = exp(-beta * delta_x)
    n_draws = length(coef_samples)
    paf_draws = zeros(Float64, n_draws)
    excess_cases_draws = zeros(Float64, n_draws)

    # Weights enter the PAF as the per-observation expected outcome μ_i (or, for a
    # population-attributable fraction, the population at risk). They are what turn
    # Σ_i μ_i r_i / Σ_i μ_i into a ratio of *sums* rather than an unweighted mean of
    # individual risk ratios; the two coincide only under equal baseline risk.
    if isnothing(weights)
        w = nothing
        w_sum = 1.0
    else
        w = Float64.(collect(weights))
        if length(w) != length(exp_obs)
            error("`weights` has length $(length(w)) but the exposure variable has " *
                  "length $(length(exp_obs)).")
        end
        any(x -> x < 0, w) && error("`weights` must be non-negative.")
        sum(w) <= 0 && error("`weights` must contain at least one positive entry.")
        w_sum = sum(w)
    end
    # With unit weights the weighted mean of `ratio_cf` is just its plain mean, so skip
    # the extra length-N broadcast entirely on the default path.
    use_unit_weights = isnothing(w)
    
    # Approximate baseline expected counts if outcome is available
    outcome_sym = if !isnothing(M) && hasproperty(M, :outcomes) && !isempty(M.outcomes)
        Symbol(M.outcomes[1])
    else
        :y
    end
    
    has_outcome = hasproperty(df, outcome_sym)
    y_total = has_outcome ? sum(skipmissing(df[!, outcome_sym])) : nothing
    if isnothing(y_total)
        # Previously this fell back to `1.0`, which made `excess_cases` silently a
        # FRACTION -- the same dimensionless 0-1 quantity as `paf_mean` -- while naming it
        # "cases". A count of excess cases is `PAF * total_observed`, and with no observed
        # total the number is undefined, not 1. The same reasoning already makes
        # `par_mean` `nothing` above, so this is the consistent answer.
        @warn "Excess cases are not reported: the outcome column '$(outcome_sym)' was not " *
              "found in `data`, so the total observed count is unknown and " *
              "`excess_cases = PAF * total_observed` cannot be formed. Supply the outcome " *
              "column, or use `paf_*`, which needs no count." maxlog = 1
    end
    
    for s in 1:n_draws
        beta_s = coef_samples[s]
        # Individual counterfactual ratio: mu_cf,i / mu_obs,i = exp(-beta_s * delta_x_i)
        ratio_cf = exp.(-beta_s .* delta_x)
        # Population attributable fraction: 1 - (sum_i w_i mu_cf,i / sum_i w_i mu_obs,i),
        # which reduces to the weighted mean of `ratio_cf`. `dot` avoids materialising a
        # temporary vector on every draw.
        paf_s = use_unit_weights ? 1.0 - mean(ratio_cf) :
                1.0 - dot(w, ratio_cf) / w_sum
        paf_draws[s] = paf_s
        excess_cases_draws[s] = isnothing(y_total) ? NaN : paf_s * y_total
    end
    
    low_p = alpha / 2.0
    high_p = 1.0 - low_p
    
    return (
        paf_mean = mean(paf_draws),
        paf_median = median(paf_draws),
        paf_std = std(paf_draws),
        paf_lower = quantile(paf_draws, low_p),
        paf_upper = quantile(paf_draws, high_p),
        # `par_mean` is NOT an alias of `paf_mean` here. `par_from_posterior` documents
        # PAR as the absolute rate difference `I_pop - I_0`, and returning a dimensionless
        # fraction under that name was the L3 defect. This function works purely with the
        # per-observation ratio `mu_cf,i / mu_obs,i`, so the absolute risks -- and therefore
        # the rate difference -- are not recoverable from it. `nothing` is the honest
        # `nothing` is the honest answer; `paf_mean` still carries the estimable
        # quantity, and `excess_cases_*` is `nothing` when the observed total is unknown.
        par_mean = nothing,
        # `nothing` rather than a number when the observed total is unknown. `NaN` in the
        # draws is turned into `nothing` here so callers get the same "unavailable"
        # convention as `par_mean`, not a quantity that merely looks non-finite.
        excess_cases_mean = isnothing(y_total) ? nothing : mean(excess_cases_draws),
        excess_cases_ci_lower = isnothing(y_total) ? nothing : quantile(excess_cases_draws, low_p),
        excess_cases_ci_upper = isnothing(y_total) ? nothing : quantile(excess_cases_draws, high_p),
        counterfactual_value = counterfactual_value,
        exposure_var = var_sym,
        n_samples = n_draws,
        raw_paf_samples = paf_draws
    )
end

"""
    export_par_to_table(par_results::Union{Dict, NamedTuple};
        include_raw_samples::Bool=false)::DataFrame

Exports Population Attributable Fraction / Risk results to a structured `DataFrame`.

# Arguments
- `par_results`: Single PAR result NamedTuple or dictionary/NamedTuple of results from `summarize_par_effects`.
- `include_raw_samples::Bool`: If true, includes posterior samples in the returned DataFrame.

# Returns
- A `DataFrame` with one row per covariate.
"""
function export_par_to_table(
    par_results::Union{Dict, NamedTuple}; 
    include_raw_samples::Bool=false
)::DataFrame
    results_dict = if par_results isa NamedTuple && haskey(par_results, :covariate)
        Dict(String(par_results.covariate) => par_results)
    elseif par_results isa NamedTuple
        Dict(String(k) => v for (k, v) in pairs(par_results))
    elseif par_results isa Dict
        par_results
    else
        Dict("result" => par_results)
    end

    rows = []
    for (k, res) in pairs(results_dict)
        if res isa NamedTuple && haskey(res, :paf_mean)
            row = (
                covariate = res.covariate,
                family = res.family,
                method = res.method,
                paf_mean = res.paf_mean,
                paf_median = res.paf_median,
                paf_std = res.paf_std,
                paf_lower = res.paf_lower,
                paf_upper = res.paf_upper,
                rr_mean = res.rr_mean,
                rr_median = res.rr_median,
                rr_ci_lower = res.rr_ci_lower,
                rr_ci_upper = res.rr_ci_upper,
                exposure_prevalence = res.exposure_prevalence,
                attributable_cases = res.attributable_cases,
                baseline_risk = res.baseline_risk,
                reference_population = res.reference_population,
                n_samples = res.n_samples
            )
            
            if include_raw_samples
                push!(rows, merge(row, (
                    raw_paf_samples = res.raw_paf_samples,
                    raw_rr_samples = res.raw_rr_samples
                )))
            else
                push!(rows, row)
            end
        end
    end

    return if !isempty(rows)
        DataFrame(rows)
    else
        DataFrame()
    end
end
