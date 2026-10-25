"""
    likelihoods.jl

Observation models, likelihood evaluation kernels, censoring, and hurdle transformations
for Bayesian Spatio-Temporal Models (TethysBayes).

Version: v1.0.0
"""


abstract type AbstractTethysBayes_Family end

struct PoissonFamily <: AbstractTethysBayes_Family end
struct GaussianFamily <: AbstractTethysBayes_Family end
struct LogNormalFamily <: AbstractTethysBayes_Family end
struct NegativeBinomialFamily <: AbstractTethysBayes_Family end
struct BinomialFamily <: AbstractTethysBayes_Family end
struct GammaFamily <: AbstractTethysBayes_Family end
struct ExponentialFamily <: AbstractTethysBayes_Family end
struct BetaFamily <: AbstractTethysBayes_Family end
struct InverseGaussianFamily <: AbstractTethysBayes_Family end
struct StudentTFamily <: AbstractTethysBayes_Family end
struct HalfNormalFamily <: AbstractTethysBayes_Family end
struct HalfStudentTFamily <: AbstractTethysBayes_Family end
struct LaplaceFamily <: AbstractTethysBayes_Family end
struct ParetoFamily <: AbstractTethysBayes_Family end
struct DirichletFamily <: AbstractTethysBayes_Family end
struct InverseWishartFamily <: AbstractTethysBayes_Family end
struct MultinomialFamily <: AbstractTethysBayes_Family end
struct CategoricalFamily <: AbstractTethysBayes_Family end
struct DirichletMultinomialFamily <: AbstractTethysBayes_Family end
struct OrdinalFamily <: AbstractTethysBayes_Family end
struct MvNormalFamily <: AbstractTethysBayes_Family end

abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end


abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end
struct LeftCensored <: AbstractCensoringState end
struct RightCensored <: AbstractCensoringState end
struct IntervalCensored <: AbstractCensoringState end


const TethysBayes_FAMILY_REGISTRY = Dict{String, AbstractTethysBayes_Family}(
    "poisson"               => PoissonFamily(),
    "gaussian"              => GaussianFamily(),
    "lognormal"             => LogNormalFamily(),
    "bernoulli"             => BinomialFamily(),
    "binomial"              => BinomialFamily(),
    "negbin"                => NegativeBinomialFamily(),
    "gamma"                 => GammaFamily(),
    "exponential"           => ExponentialFamily(),
    "beta"                  => BetaFamily(),
    "inverse_gaussian"      => InverseGaussianFamily(),
    "student_t"             => StudentTFamily(),
    "half_normal"           => HalfNormalFamily(),
    "half_student_t"        => HalfStudentTFamily(),
    "laplace"               => LaplaceFamily(),
    "pareto"                => ParetoFamily(),
    "dirichlet"             => DirichletFamily(),
    "inverse_wishart"       => InverseWishartFamily(),
    "multinomial"           => MultinomialFamily(),
    "categorical"           => CategoricalFamily(),
    "dirichlet_multinomial" => DirichletMultinomialFamily(),
    "ordinal"               => OrdinalFamily(),
    "mvnormal"              => MvNormalFamily(),
    "multivariate_normal"   => MvNormalFamily()
)

const STATSMODELS_CONTRASTS = Dict(
    :dummy => StatsModels.DummyCoding(),
    :effects => StatsModels.EffectsCoding(),
    :helmert => StatsModels.HelmertCoding(),
    :treatment => StatsModels.DummyCoding()
)




"""
    TethysBayes_Likelihood

Observation likelihood distribution parameterized by the linear predictor `param` (`eta`),
supporting zero-inflation, hurdles, left/right/interval censoring, and observation weights.
"""
struct TethysBayes_Likelihood{
    F, Z, C, W, P, PH, R, S, PR, TR, TL, TU, HT, EX
} <: ContinuousMultivariateDistribution
    family::F
    param::PR
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    phi_hurdle::PH
    r_nb::R
    sigma_y::S
    trial::TR
    censor_lower::TL
    censor_upper::TU
    hurdle::HT
    extra_params::EX
end

Base.length(d::TethysBayes_Likelihood) = length(d.param)
Base.size(d::TethysBayes_Likelihood) = (length(d.param),)

"""
    get_model_family(model_family::String)::AbstractTethysBayes_Family

Maps a string identifier to its corresponding concrete `AbstractTethysBayes_Family` singleton instance.
"""
function get_model_family(model_family::String)
    family_key = lowercase(strip(model_family))
    if haskey(TethysBayes_FAMILY_REGISTRY, family_key)
        return TethysBayes_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$(model_family)'. " *
              "Supported families are: $(keys(TethysBayes_FAMILY_REGISTRY))")
    end
end

# --- Distribution Reference Generators ---

@inline _get_obs(v, i::Int) = v isa AbstractVector ? v[i] : v

"""
    const ETA_CLAMP_BOUND

Symmetric bound applied to the linear predictor before exponentiating, shared by the
likelihood families here and by the denoised-prediction path in `reconstruction.jl`.

These two MUST agree. They were independently hardcoded as 30 and 20, so for
\$\$\\eta \\in (20, 30]\\\$\$ the denoised prediction reported \$\$\\exp(20)\\\$ while a noisy
draw from the fitted likelihood returned \$\$\\exp(\\eta)\\\$ -- a discrepancy of up to
\$\$e^{10}\\\$ between a reported mean and the mean of its own simulated noise. One shared
constant removes the class of bug rather than the single instance.

30 is far below \$\$\\exp(709)\\\$, the actual overflow point, so this guards against
pathological inputs rather than serving a numerical necessity.
"""
const ETA_CLAMP_BOUND = 30.0

# Generic fallback: distribution reference ignores obs_idx unless specialized
get_dist_ref(fam::AbstractTethysBayes_Family, d, eta, sig, obs_idx::Int) =
    get_dist_ref(fam, d, eta, sig)

function get_dist_ref(::PoissonFamily, d, eta::V, sig, obs_idx::Int=1) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Poisson(1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Poisson(clamp(exp(eta_clamped), V(1e-9), V(1e9)))
end    

function get_dist_ref(::PoissonFamily, d, eta, sig, obs_idx::Int=1)
    if isnan(eta) || isinf(eta)
        return Poisson(1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Poisson(clamp(exp(eta_clamped), 1e-9, 1e9))
end

function get_dist_ref(::InverseWishartFamily, d, eta, sig, obs_idx::Int=1)
    error("The Inverse-Wishart likelihood is for covariance matrix outcomes and is not " *
          "supported in the univariate framework.")
end

function get_dist_ref(::GaussianFamily, d, eta::V, sig::S, obs_idx::Int=1) where {V<:Real, S<:Real}
    T = promote_type(V, S)
    if isnan(eta) || isinf(eta)
        return Normal(zero(T), one(T))
    end
    sig_val = _get_obs(sig, obs_idx)
    return Normal(T(eta), T(sig_val) + T(1e-9))
end

function get_dist_ref(::LogNormalFamily, d, eta::V, sig::S, obs_idx::Int=1) where {V<:Real, S<:Real}
    T = promote_type(V, S)
    if isnan(eta) || isinf(eta)
        return LogNormal(zero(T), one(T))
    end
    sig_val = _get_obs(sig, obs_idx)
    mu = clamp(T(eta) - (T(sig_val)^2) / T(2.0), T(-30.0), T(30.0))
    return LogNormal(mu, T(sig_val) + T(1e-9))
end

function get_dist_ref(::NegativeBinomialFamily, d, eta::V, sig, obs_idx::Int=1) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return NegativeBinomial(1, 0.5)
    end
    r = V(_get_obs(d.r_nb, obs_idx))
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    mu = clamp(exp(eta_clamped), V(1e-9), V(1e9))
    p = clamp(r / (r + mu), V(1e-12), V(1.0 - 1e-12))
    return NegativeBinomial(r, p)
end

function get_dist_ref(::BinomialFamily, d, eta::V, sig, obs_idx::Int=1) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Binomial(1, 0.5)
    end
    n = _get_obs(d.trial, obs_idx)
    return Binomial(Int(n), LogExpFunctions.logistic(eta))
end

function get_dist_ref(::BinomialFamily, d, eta, sig, obs_idx::Int=1)
    if isnan(eta) || isinf(eta)
        return Binomial(1, 0.5)
    end
    n = _get_obs(d.trial, obs_idx)
    return Binomial(Int(n), LogExpFunctions.logistic(eta))
end

function get_dist_ref(::GammaFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Gamma(1.0, 1.0)
    end
    alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Gamma(alpha, clamp(exp(eta_clamped), V(1e-9), V(1e9)) / alpha)
end

function get_dist_ref(::GammaFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Gamma(1.0, 1.0)
    end
    alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Gamma(alpha, clamp(exp(eta_clamped), 1e-9, 1e9) / alpha)
end

function get_dist_ref(::ExponentialFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Exponential(1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Exponential(clamp(exp(eta_clamped), V(1e-9), V(1e9)))
end

function get_dist_ref(::ExponentialFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Exponential(1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Exponential(clamp(exp(eta_clamped), 1e-9, 1e9))
end

function get_dist_ref(::BetaFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Beta(1.0, 1.0)
    end
    mu = LogExpFunctions.logistic(eta)
    phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0
    return Beta(clamp(mu * phi, V(1e-9), V(1e9)), clamp((V(1.0) - mu) * phi, V(1e-9), V(1e9)))
end

function get_dist_ref(::BetaFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Beta(1.0, 1.0)
    end
    mu = LogExpFunctions.logistic(eta)
    phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0
    return Beta(clamp(mu * phi, 1e-9, 1e9), clamp((1.0 - mu) * phi, 1e-9, 1e9))
end

function get_dist_ref(::InverseGaussianFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return InverseGaussian(1.0, 1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    mu = clamp(exp(eta_clamped), V(1e-9), V(1e9))
    lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    return InverseGaussian(mu, lambda)
end

function get_dist_ref(::InverseGaussianFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return InverseGaussian(1.0, 1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    mu = clamp(exp(eta_clamped), 1e-9, 1e9)
    lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    return InverseGaussian(mu, lambda)
end

function get_dist_ref(::StudentTFamily, d, eta, sig)
    nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0
    loc = isnan(eta) || isinf(eta) ? 0.0 : eta
    return LocationScale(loc, max(sig, 1e-9), TDist(nu))
end

function get_dist_ref(::HalfNormalFamily, d, eta, sig)
    # `eta` is the location, exactly as for `StudentTFamily`/`LaplaceFamily`; the
    # truncation at 0 is what makes the support half-line, not the location. Pinning the
    # location to 0.0 would make the linear predictor have no effect on the mean.
    loc = isnan(eta) || isinf(eta) ? 0.0 : eta
    return truncated(Normal(loc, max(sig, 1e-9)), 0.0, Inf)
end

function get_dist_ref(::HalfStudentTFamily, d, eta, sig)
    nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0
    loc = isnan(eta) || isinf(eta) ? 0.0 : eta
    return truncated(LocationScale(loc, max(sig, 1e-9), TDist(nu)), 0.0, Inf)
end

function get_dist_ref(::LaplaceFamily, d, eta, sig)
    loc = isnan(eta) || isinf(eta) ? 0.0 : eta
    return Laplace(loc, max(sig, 1e-9))
end

function get_dist_ref(::ParetoFamily, d, eta, sig)
    shape = d.extra_params isa Number && d.extra_params > 1.0 ? d.extra_params : 1.1
    eta_clamped = clamp(eta, -30.0, 30.0)
    mean_val = clamp(exp(eta_clamped), 1e-9, 1e9)
    scale = mean_val * (shape - 1.0) / shape
    return Pareto(shape, scale)
end




"""
    get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `DirichletMultinomial` distribution instance parameterized by `d.trial` and
  `softmax(eta_vec)`.
"""
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)
    sig_val = _get_obs(sig, obs_idx)
    alpha_0 = max(sig_val, 1e-4)
    mean_probs = NNlib.softmax(eta_vec)
    alpha_params = max.(alpha_0 .* mean_probs, 1e-6)
    n_total = _get_obs(d.trial, obs_idx)
    return DirichletMultinomial(Int(n_total), alpha_params)
end

function TethysBayes_kernel(
    fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

"""
    get_dist_ref(::MultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `Multinomial` distribution instance parameterized by `d.trial` and `softmax(eta_vec)`.
"""
function get_dist_ref(::MultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)
    probs = NNlib.softmax(eta_vec)
    n_total = _get_obs(d.trial, obs_idx)
    return Multinomial(Int(n_total), probs)
end

function TethysBayes_kernel(
    fam::MultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

"""
    get_dist_ref(::CategoricalFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `Categorical` distribution instance parameterized by `softmax(eta_vec)`.
"""
function get_dist_ref(::CategoricalFamily, d, eta_vec, sig, obs_idx::Int=1)
    probs = NNlib.softmax(eta_vec)
    return Categorical(probs)
end

function TethysBayes_kernel(
    fam::CategoricalFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_val,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, Int(y_val))
end

"""
    get_dist_ref(::DirichletFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `Dirichlet` distribution instance parameterized by `alpha = phi .* softmax(eta_vec)`.
"""
function get_dist_ref(::DirichletFamily, d, eta_vec, sig, obs_idx::Int=1)
    sig_val = _get_obs(sig, obs_idx)
    phi = max(sig_val, 1e-4)
    probs = NNlib.softmax(eta_vec)
    alpha_params = max.(phi .* probs, 1e-6)
    return Dirichlet(alpha_params)
end

function TethysBayes_kernel(
    fam::DirichletFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

"""
    get_dist_ref(::MvNormalFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a multivariate normal (`MvNormal`) distribution instance.
Mean vector is given by `eta_vec`. The covariance structure is resolved from
`d.extra_params[:cov]`, correlation Cholesky factor `d.extra_params[:correlation_cholesky]`,
matrix `sig`, or diagonal elements from vector/scalar `sig`.

# Mathematical Formulation
\$ Y \\sim \\text{MvNormal}(\\mu, \\Sigma) \$
where \$\\mu = \\eta\$ and \$\\Sigma = D(s) L L^T D(s)\$ or \$\\text{diag}(s^2)\$.
"""
function get_dist_ref(::MvNormalFamily, d, eta_vec, sig, obs_idx::Int=1)
    K = length(eta_vec)
    cov_mat = if !isnothing(d.extra_params) && haskey(d.extra_params, :cov)
        d.extra_params[:cov]
    elseif !isnothing(d.extra_params) && haskey(d.extra_params, :correlation_cholesky)
        Lraw = d.extra_params[:correlation_cholesky]
        # The correlation factor arrives in several shapes: a `Cholesky`, an `LKJCholesky`
        # (whose Cholesky lives in its `.d` field, NOT `.L`), or a plain matrix.
        # `Cholesky * Adjoint` has no method and `Matrix(::LKJCholesky)` is not the
        # factor, so unwrap before forming the covariance.
        Lc = if Lraw isa LinearAlgebra.Cholesky
            Lraw
        elseif hasproperty(Lraw, :d) && getfield(Lraw, :d) isa LinearAlgebra.Cholesky
            getfield(Lraw, :d)            # LKJCholesky
        elseif hasproperty(Lraw, :L) && getfield(Lraw, :L) isa LinearAlgebra.Cholesky
            getfield(Lraw, :L)            # other wrappers
        else
            Lraw
        end
        Lmat = Lc isa AbstractMatrix ? Lc : Matrix(Lc)
        s_raw = sig isa AbstractVector ? sig : fill(sig, K)
        # `D R D` is positive definite only if every `s` is strictly positive AND `R = L L'`
        # is itself positive definite. `LKJCholesky` permits correlations approaching +-1,
        # where `lambda_min(R) -> 0`, and `y_sigma` is an unconstrained positive parameter
        # that can be driven arbitrarily small -- or negative, or non-finite when HMC
        # diverges. All are handled relative to the data's own scale rather than with an
        # absolute epsilon, so the safeguard does not depend on units.
        #
        # Non-finite `s` is replaced by the floor rather than thrown on. `cholesky` reports a
        # NaN covariance as `PosDefException`, which is indistinguishable from genuine
        # ill-conditioning -- the reason this was misdiagnosed as conditioning for so long.
        # Throwing here instead kills the whole test suite before it can report. Collapsing
        # to the floor makes the log-density hugely negative, so HMC rejects the proposal
        # exactly as it should for an invalid state, and the run continues.
        finite_s = filter(isfinite, s_raw)
        smax = isempty(finite_s) ? one(real(float(eltype(s_raw)))) :
               max(maximum(abs, finite_s), floatmin(float(eltype(s_raw))))
        floorv = 1e-8 * smax
        s = map(v -> (isfinite(v) && v > floorv) ? v : floorv, s_raw)
        # Form the covariance as a GRAM matrix, `Sigma = B B'` with `B = Diagonal(s) * L`,
        # rather than as `Diagonal(s) * (L L') * Diagonal(s)`.
        #
        # The two are algebraically identical -- `(D L)(D L)' = D L L' D` -- but the Gram
        # form is positive *semidefinite by construction*. Every operation is a multiply or
        # an add, so it stays valid under ForwardDiff's dual numbers, and there is no
        # intermediate `R` that rounding can push below zero.
        B = Diagonal(s) * Lmat
        Σ = B * B'
        Σ = (Σ + Σ') / 2
        # A diverging HMC trajectory can drive the correlation factor itself to NaN/Inf.
        # `cholesky` reports such a covariance as `PosDefException`, indistinguishable from
        # genuine ill-conditioning -- which is why this was misdiagnosed as conditioning
        # for so long. Throwing kills the test suite before it can report, and the factor
        # is not repairable: a NaN Cholesky has no meaning.
        #
        # Instead, return a valid distribution with a vanishingly small isotropic
        # covariance. Its log-density is -Inf for any observation differing from the mean,
        # so the proposal is rejected -- the same treatment Stan gives an invalid state --
        # and the run continues instead of dying.
        if !all(isfinite, Σ)
            return MvNormal(collect(eta_vec),
                            Symmetric(Matrix{eltype(eta_vec)}(I, K, K) * eps(one(real(float(eltype(Σ)))))))
        end
        # Relative diagonal jitter, needed only for the rank-deficient case where some
        # diagonal of `L` has reached 0. Bounded by the matrix scale, so it repairs
        # ill-conditioning without distorting a well-conditioned fit.
        Σ = Σ + (1e-8 * (tr(Σ) / K + one(eltype(Σ)))) * Matrix{eltype(Σ)}(I, K, K)
    elseif sig isa AbstractMatrix
        sig
    elseif sig isa AbstractVector
        Diagonal(sig .^ 2)
    else
        Diagonal(fill(Float64(sig)^2, K))
    end
    return MvNormal(collect(eta_vec), Symmetric(cov_mat))
end

function TethysBayes_kernel(
    fam::MvNormalFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily,
                                   MultinomialFamily, CategoricalFamily,
                                   DirichletMultinomialFamily})
    return true
end

function is_discrete_family(::AbstractTethysBayes_Family)
    return false
end

"""
    TethysBayes_Likelihood(family_input, param; kwargs...)

Constructs a `TethysBayes_Likelihood` distribution wrapper parameterized by linear predictor `param`
(`eta`), incorporating censoring, zero-inflation, hurdles, dispersion, and observation weights.
Supports both scalar and vectorized per-observation modifiers with rigorous validation.
"""
function TethysBayes_Likelihood(
    family_input::Union{String, Symbol},
    param;
    zi_state = nothing,
    censoring_state = nothing,
    weight = 1.0,
    phi_zi = -Inf,
    phi_hurdle = -Inf,
    r_nb = 1.0,
    sigma_y = 1.0,
    trial = 1,
    censor_lower = -Inf,
    censor_upper = Inf,
    hurdle = -Inf,
    extra_params = nothing
)
    f_trait = get_model_family(string(family_input))
    param_vec = param isa AbstractVector ? param : [param]
    n_obs = length(param_vec)
    
    function _validate_obs_param(x, name::Symbol)
        if x isa AbstractArray
            if isempty(x)
                throw(ArgumentError("Parameter ':$name' in TethysBayes_Likelihood cannot be an empty array."))
            elseif length(x) == 1
                return first(x)
            elseif length(x) != n_obs
                throw(DimensionMismatch(
                    "Parameter ':$name' has length $(length(x)), but expected scalar or " *
                    "vector matching observation count ($n_obs)."
                ))
            else
                return x
            end
        end
        return x
    end

    function _validate_int_param(x, name::Symbol)
        val = _validate_obs_param(x, name)
        return val isa AbstractArray ? Int.(val) : Int(val)
    end

    tr_s = _validate_int_param(trial, :trial)
    w_s = _validate_obs_param(weight, :weight)
    pzi_s = _validate_obs_param(phi_zi, :phi_zi)
    phu_s = _validate_obs_param(phi_hurdle, :phi_hurdle)
    rnb_s = _validate_obs_param(r_nb, :r_nb)
    sig_s = _validate_obs_param(sigma_y, :sigma_y)
    cl_s = _validate_obs_param(censor_lower, :censor_lower)
    cu_s = _validate_obs_param(censor_upper, :censor_upper)
    hu_s = _validate_obs_param(hurdle, :hurdle)

    # Validate mutual exclusivity of zero-inflation and hurdle formulations
    has_zi = (zi_state isa ZeroInflated) ||
             (pzi_s isa AbstractVector ? any(v -> v > -Inf, pzi_s) : pzi_s > -Inf)
    has_hurdle = (phu_s isa AbstractVector ? any(v -> v > -Inf, phu_s) : phu_s > -Inf) ||
                 (hu_s isa AbstractVector ? any(v -> isfinite(v) && v > -Inf, hu_s) :
                  (isfinite(hu_s) && hu_s > -Inf))
    if has_zi && has_hurdle
        throw(ArgumentError(
            "Likelihood specification error: Zero-inflation and hurdle models are mutually exclusive. " *
            "Received zero-inflation (phi_zi = $(pzi_s)) with hurdle (phi_hurdle = $(phu_s), hurdle = $(hu_s)). " *
            "Specify either zero-inflation or a hurdle threshold, not both."
        ))
    end

    zi_trait = if !isnothing(zi_state)
        zi_state
    else
        has_zi ? ZeroInflated() : NonZeroInflated()
    end

    has_lower_censor = cl_s isa AbstractVector ? any(isfinite, cl_s) : isfinite(cl_s)
    has_upper_censor = cu_s isa AbstractVector ? any(isfinite, cu_s) : isfinite(cu_s)

    # Validate censoring bounds: lower must be strictly less than upper (Item 15)
    if has_lower_censor && has_upper_censor
        if cl_s isa AbstractVector && cu_s isa AbstractVector
            if length(cl_s) != length(cu_s)
                throw(DimensionMismatch(
                    "Censoring bounds vector length mismatch: censor_lower has $(length(cl_s)) " *
                    "elements but censor_upper has $(length(cu_s)) elements."
                ))
            end
            for idx in eachindex(cl_s, cu_s)
                if isfinite(cl_s[idx]) && isfinite(cu_s[idx]) && cl_s[idx] >= cu_s[idx]
                    throw(ArgumentError(
                        "Censoring bounds invalid at index $(idx): lower bound ($(cl_s[idx])) " *
                        "must be strictly less than upper bound ($(cu_s[idx]))."
                    ))
                end
            end
        elseif cl_s isa AbstractVector
            for idx in eachindex(cl_s)
                if isfinite(cl_s[idx]) && isfinite(cu_s) && cl_s[idx] >= cu_s
                    throw(ArgumentError(
                        "Censoring bounds invalid at index $(idx): lower bound ($(cl_s[idx])) " *
                        "must be strictly less than upper bound ($(cu_s))."
                    ))
                end
            end
        elseif cu_s isa AbstractVector
            for idx in eachindex(cu_s)
                if isfinite(cl_s) && isfinite(cu_s[idx]) && cl_s >= cu_s[idx]
                    throw(ArgumentError(
                        "Censoring bounds invalid at index $(idx): lower bound ($(cl_s)) " *
                        "must be strictly less than upper bound ($(cu_s[idx]))."
                    ))
                end
            end
        else
            if isfinite(cl_s) && isfinite(cu_s) && cl_s >= cu_s
                throw(ArgumentError(
                    "Censoring bounds invalid: lower bound ($(cl_s)) must be strictly " *
                    "less than upper bound ($(cu_s))."
                ))
            end
        end
    end

    censor_trait = if !isnothing(censoring_state)
        censoring_state
    elseif !has_lower_censor && !has_upper_censor
        Uncensored()
    elseif has_lower_censor && !has_upper_censor
        RightCensored()
    elseif !has_lower_censor && has_upper_censor
        LeftCensored()
    else 
        IntervalCensored() 
    end

    return TethysBayes_Likelihood(
        f_trait, param_vec, zi_trait, censor_trait,
        w_s, pzi_s, phu_s, rnb_s, sig_s, tr_s,
        cl_s, cu_s, hu_s, extra_params
    )
end

function Distributions._logpdf(d::TethysBayes_Likelihood, y::AbstractVector{V}) where {V<:Real}
    logp = zero(V) 
    
    if d.family isa MvNormalFamily
        eta = d.param
        sig = d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return TethysBayes_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y, 1) * w
    elseif d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily}
        eta = d.param
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return TethysBayes_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y, 1) * w
    else
        for i in 1:length(y)
            eta_i = d.param isa AbstractVector ? d.param[i] : d.param
            sig_i = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            w_i = d.weight isa AbstractVector ? d.weight[i] : d.weight
            k_val = TethysBayes_kernel(d.family, d.censoring_state, d.zi_state, d, eta_i, sig_i, y[i], i)
            logp += k_val * w_i
        end
        return logp
    end
end

"""
    Distributions.logpdf(d::TethysBayes_Likelihood, Y::AbstractMatrix{<:Real})

Evaluates joint log-likelihood for a batch of \$N\$ multivariate observations
represented as rows of matrix \$Y \\in \\mathbb{R}^{N \\times K}\$.
"""
function Distributions.logpdf(d::TethysBayes_Likelihood, Y::AbstractMatrix{<:Real})
    if d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily, MvNormalFamily}
        logp = 0.0
        N = size(Y, 1)
        for i in 1:N
            eta_i = d.param isa AbstractMatrix ? d.param[i, :] : d.param
            sig_i = if d.family isa MvNormalFamily
                (d.sigma_y isa AbstractMatrix && size(d.sigma_y, 1) == N) ? d.sigma_y[i, :] : d.sigma_y
            else
                d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            end
            w_i = d.weight isa AbstractVector ? d.weight[i] : d.weight
            k_val = TethysBayes_kernel(d.family, d.censoring_state, d.zi_state, d, eta_i, sig_i, Y[i, :], i)
            logp += k_val * w_i
        end
        return logp
    else
        throw(ArgumentError(
            "Matrix observation evaluation is only supported for multivariate likelihood families, " *
            "got $(typeof(d.family))."
        ))
    end
end

function Distributions.logpdf(d::TethysBayes_Likelihood, y::Real)
    if d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily, MvNormalFamily}
        error("$(typeof(d.family)) likelihood requires a vector of observations, " *
              "but received a scalar.")
    end
    
    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    
    return TethysBayes_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y, 1) * w
end

function Distributions.logpdf(d::TethysBayes_Likelihood, y::AbstractVector{<:Real})
    return Distributions._logpdf(d, y)
end

function Base.rand(rng::Random.AbstractRNG, d::TethysBayes_Likelihood)
    if d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily,
                          CategoricalFamily, MvNormalFamily}
        dist = get_dist_ref(d.family, d, d.param, d.sigma_y, 1)
        return rand(rng, dist)
    end

    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    phi_zi = _get_obs(d.phi_zi, 1)
    phi_hu = _get_obs(d.phi_hurdle, 1)
    hu_val = _get_obs(d.hurdle, 1)

    # Zero-inflation draw
    if d.zi_state isa ZeroInflated
        if rand(rng) < phi_zi
            return zero(Float64)
        end
    elseif phi_hu > -Inf
        if rand(rng) < (1.0 - phi_hu)
            return Float64(hu_val)
        end
    end

    dist = get_dist_ref(d.family, d, eta, sig, 1)
    raw_draw = Float64(rand(rng, dist))

    # Censoring bounds
    if d.censoring_state isa LeftCensored
        l_b = _get_obs(d.censor_lower, 1)
        return max(raw_draw, Float64(l_b))
    elseif d.censoring_state isa RightCensored
        u_b = _get_obs(d.censor_upper, 1)
        return min(raw_draw, Float64(u_b))
    elseif d.censoring_state isa IntervalCensored
        l_b = _get_obs(d.censor_lower, 1)
        u_b = _get_obs(d.censor_upper, 1)
        return clamp(raw_draw, Float64(l_b), Float64(u_b))
    end

    return raw_draw
end

Base.rand(d::TethysBayes_Likelihood) = rand(Random.default_rng(), d)

function Distributions._rand!(rng::Random.AbstractRNG, d::TethysBayes_Likelihood, x::AbstractArray)
    for i in eachindex(x)
        eta_i = d.param isa AbstractVector ? d.param[i] : d.param
        sig_i = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        phi_zi_i = _get_obs(d.phi_zi, i)
        phi_hu_i = _get_obs(d.phi_hurdle, i)
        hu_i = _get_obs(d.hurdle, i)
        
        draw = if d.zi_state isa ZeroInflated && rand(rng) < phi_zi_i
            zero(Float64)
        elseif phi_hu_i > -Inf && rand(rng) < (1.0 - phi_hu_i)
            Float64(hu_i)
        else
            dist_i = get_dist_ref(d.family, d, eta_i, sig_i, i)
            Float64(rand(rng, dist_i))
        end
        
        cl_i = _get_obs(d.censor_lower, i)
        cu_i = _get_obs(d.censor_upper, i)
        if d.censoring_state isa LeftCensored
            x[i] = max(draw, Float64(cl_i))
        elseif d.censoring_state isa RightCensored
            x[i] = min(draw, Float64(cu_i))
        elseif d.censoring_state isa IntervalCensored
            x[i] = clamp(draw, Float64(cl_i), Float64(cu_i))
        else
            x[i] = draw
        end
    end
    return x
end

function TethysBayes_kernel(
    fam::AbstractTethysBayes_Family, ::Uncensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    # Is the linear predictor a ForwardDiff dual? Decided once, from `V`.
    is_ad = _is_dual_type(V)
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))

        if y == zero(V)
            if is_discrete_family(fam)
                lp0 = log_one_minus_phi + logpdf(dist, zero(V))
                return LogExpFunctions.logsumexp(log_phi, lp0)
            else
                return log_phi
            end
        else
            return log_one_minus_phi + logpdf(dist, V(y))
        end
    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        log_one_minus_phi = V(log1p(-phi_hu_val))

        if y <= hu_val
            return log_one_minus_phi
        else
            logp_truncated = logpdf(dist, V(y)) - _logccdf_count(dist, hu_val, is_ad)
            return log_phi + logp_truncated
        end
    else
        return logpdf(dist, V(y))
    end
end

function TethysBayes_kernel(
    fam::AbstractTethysBayes_Family, ::LeftCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    upper_bound = V(_get_obs(d.censor_upper, obs_idx))
    if !isfinite(upper_bound)
        return TethysBayes_kernel(fam, Uncensored(), zero_inflated, d, eta, sig, y, obs_idx)
    end

    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    # Is the linear predictor a ForwardDiff dual? Decided once, from `V`.
    is_ad = _is_dual_type(V)
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))
        lp_base = _logcdf_count(dist, upper_bound, is_ad)
        if upper_bound >= V(0.0)
            return LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            return log_one_minus_phi + lp_base
        end
    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        log_one_minus_phi = V(log1p(-phi_hu_val))
        if upper_bound <= hu_val
            return log_one_minus_phi
        end
        log_prob_interval = _stable_logsubexp(
            _logcdf_count(dist, upper_bound, is_ad), _logcdf_count(dist, hu_val, is_ad)
        ) - _logccdf_count(dist, hu_val, is_ad)
        return LogExpFunctions.logsumexp(log_one_minus_phi, log_phi + log_prob_interval)
    else
        return _logcdf_count(dist, upper_bound, is_ad)
    end
end

function TethysBayes_kernel(
    fam::AbstractTethysBayes_Family, ::RightCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    lower_bound = V(_get_obs(d.censor_lower, obs_idx))
    if !isfinite(lower_bound)
        return TethysBayes_kernel(fam, Uncensored(), zero_inflated, d, eta, sig, y, obs_idx)
    end

    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    # Is the linear predictor a ForwardDiff dual? Decided once, from `V`.
    is_ad = _is_dual_type(V)
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))
        
        log_p_le_L = if lower_bound < V(0.0)
            log_one_minus_phi + _logcdf_count(dist, lower_bound, is_ad)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + _logcdf_count(dist, lower_bound, is_ad))
        end
        return LogExpFunctions.log1mexp(log_p_le_L)

    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        adj_hurdle = is_discrete_family(fam) ? hu_val - V(1.0) : hu_val

        if lower_bound > hu_val
            return log_phi + _logccdf_count(dist, adj_L, is_ad) - _logccdf_count(dist, adj_hurdle, is_ad)
        else
            return log_phi
        end
    else
        return _logccdf_count(dist, adj_L, is_ad)
    end
end

function TethysBayes_kernel(
    fam::AbstractTethysBayes_Family, ::IntervalCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    lower_bound = V(_get_obs(d.censor_lower, obs_idx))
    upper_bound = V(_get_obs(d.censor_upper, obs_idx))

    if !isfinite(lower_bound) && !isfinite(upper_bound)
        return TethysBayes_kernel(fam, Uncensored(), zero_inflated, d, eta, sig, y, obs_idx)
    elseif isfinite(lower_bound) && !isfinite(upper_bound)
        return TethysBayes_kernel(fam, RightCensored(), zero_inflated, d, eta, sig, y, obs_idx)
    elseif !isfinite(lower_bound) && isfinite(upper_bound)
        return TethysBayes_kernel(fam, LeftCensored(), zero_inflated, d, eta, sig, y, obs_idx)
    end

    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    # Is the linear predictor a ForwardDiff dual? Decided once, from `V`.
    is_ad = _is_dual_type(V)
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))

        log_p_le_U = if upper_bound < V(0.0)
            log_one_minus_phi + _logcdf_count(dist, upper_bound, is_ad)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + _logcdf_count(dist, upper_bound, is_ad))
        end

        log_p_le_L = if lower_bound < V(0.0)
            log_one_minus_phi + _logcdf_count(dist, lower_bound, is_ad)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + _logcdf_count(dist, lower_bound, is_ad))
        end
        return _stable_logsubexp(log_p_le_U, log_p_le_L)

    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        adj_hurdle = is_discrete_family(fam) ? hu_val - V(1.0) : hu_val

        if upper_bound <= hu_val
            return V(-Inf)
        end

        effective_lower = max(adj_L, adj_hurdle)
        log_prob_in_interval = _stable_logsubexp(
            _logcdf_count(dist, upper_bound, is_ad), _logcdf_count(dist, effective_lower, is_ad)
        )
        log_normalizer = _logccdf_count(dist, adj_hurdle, is_ad)
        return log_phi + log_prob_in_interval - log_normalizer
    else
        return _stable_logsubexp(_logcdf_count(dist, upper_bound, is_ad), _logcdf_count(dist, adj_L, is_ad))
    end
end

"""
    _stable_logsubexp(a::Real, b::Real)

Computes `log(exp(a) - exp(b))` in a numerically stable manner via `a + log1mexp(b - a)`.
Returns `-Inf` if `a <= b`.
"""
function _stable_logsubexp(a::Real, b::Real)
    if a <= b
        return -Inf
    end
    return a + LogExpFunctions.log1mexp(b - a)
end

# ==============================================================================
# ForwardDiff-capable truncated counts (N1)
# ==============================================================================
#
# The hurdle and censoring kernels normalise by a count tail, e.g. the uncensored hurdle
# computes `logpdf(y) - logccdf(hu)` because the continuous part is the distribution
# conditioned on X > hu. Those normalisers are what broke gradient sampling:
# `logcdf`/`logccdf` for Poisson route through Distributions' `_gammalogcdf`, which has
# methods only for `Float64` and `Float16`/`Float32`. So with a `ForwardDiff.Dual` linear
# predictor they raise
#
#     MethodError: no method matching _gammalogcdf(::ForwardDiff.Dual, ::ForwardDiff.Dual,
#                                                     ::ForwardDiff.Dual)
#
# and NUTS/HMC/ADVI all fail while MH (which needs no gradient) works. `logpdf`, by
# contrast, IS AD-capable -- only the tails are not, which is why the plain Poisson model
# samples fine and the hurdle one does not.
#
# The fix is NOT to add a method for `_gammalogcdf`. That is a Distributions internal, so
# defining one would be type piracy: it would be silently overwritten on upgrade and it
# would apply to every user of the package in the session. Instead, when the linear
# predictor is a dual, evaluate the truncated sum directly from `logpdf`, which is
# AD-capable.
#
# The plain-float path is deliberately left calling `logcdf`/`logccdf`, so reported
# results are bit-identical to before. Only the AD path -- which previously threw -- changes.
#
# `logccdf` is obtained as `log1mexp(logcdf)`, i.e. via the complement. This is exact in
# exact arithmetic and fine for the small thresholds a hurdle uses, but it loses precision
# when `cdf -> 1` (a threshold far into the upper tail), where it tends to `-Inf`. The
# float path does not have that weakness, which is one more reason not to route it there.

# A hurdle threshold is a small count. Beyond this the explicit sum stops being sensible,
# so the AD path reports an actionable error instead of silently grinding.
const _HURDLE_SUM_MAX_TERMS = 4096

# `ForwardDiff.Dual <: Real` but NOT `<: AbstractFloat`, so this separates the AD path from
# the plain-float path without referencing ForwardDiff (which is not a direct dependency).
_is_dual_type(::Type{T}) where {T} = !(T <: AbstractFloat)

"""
    _count_threshold_index(x)::Int

Floor a censoring/hurdle bound to the integer count index it selects. Uses `floor` rather
than a bare `Int()` so a non-integral bound (a hurdle of 0.5) degrades to a well-defined
index instead of throwing an `InexactError` on the AD path.
"""
_count_threshold_index(x::Real) = isfinite(x) ? floor(Int, x) : (x < 0 ? typemin(Int) - 1 : _HURDLE_SUM_MAX_TERMS + 1)

"""
    _logcdf_count_ad(dist, x)

`log P(X <= x)` for a discrete `dist`, computed as `logsumexp` of the point masses, so it
works when the parameter is a `ForwardDiff.Dual`.
"""
function _logcdf_count_ad(dist, x::Real)
    k = _count_threshold_index(x)
    k < 0 && return oftype(float(x), -Inf)
    if k > _HURDLE_SUM_MAX_TERMS
        error("Hurdle/censoring bound $(k) exceeds the AD-capable sum limit of " *
              "$(_HURDLE_SUM_MAX_TERMS) terms. Gradient samplers cannot evaluate a " *
              "truncated count normaliser that far into the tail; supply a smaller " *
              "threshold, or sample with MH.")
    end
    return LogExpFunctions.logsumexp([logpdf(dist, i) for i in 0:k])
end

"""
    _logccdf_count_ad(dist, x)

`log P(X > x)`, via the complement of `_logcdf_count_ad`. See the note above on precision
when the CDF approaches 1.
"""
function _logccdf_count_ad(dist, x::Real)
    k = _count_threshold_index(x)
    k < 0 && return zero(float(x))
    return LogExpFunctions.log1mexp(_logcdf_count_ad(dist, x))
end

# Dispatch on `is_ad`, decided once per kernel call from the linear predictor's type
# rather than by probing the distribution's parameters (which have no common accessor
# across Poisson / NegativeBinomial / Binomial).
_logcdf_count(dist, x, is_ad::Bool) = is_ad ? _logcdf_count_ad(dist, x) : logcdf(dist, x)
_logccdf_count(dist, x, is_ad::Bool) = is_ad ? _logccdf_count_ad(dist, x) : logccdf(dist, x)


