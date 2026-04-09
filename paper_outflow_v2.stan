// ============================================================================
// Azose & Raftery (2019) — OUTFLOW MODEL (Emigration Rate) — V2
//
// Paper equations:
//   N_it = floor(δ_it · P_it + 1/2)
//   log δ_it ~ Normal((1 - φ)·μ_i + φ·log δ_{i,t-1}, σ_i²)
//   φ ~ Uniform(0, 1)
//   μ_i ~ Normal(ν, τ₀²)
//   ν ~ Normal(μ₀, 100²)
//   σ_i ~ Beta(a₀, b₀)
//
// τ₀ is USER-SPECIFIED (fixed from data), not estimated.
// Paper: "We set this value equal to approximately three times the SD
//         of the mean log origin outflow rate over t."
// ============================================================================

data {
  int<lower=1> N_orig;

  // Initial observations (first period per origin)
  int<lower=0> N_init;
  array[N_init] int<lower=1, upper=N_orig> init_origin;
  vector[N_init] log_delta_init;

  // AR observations
  int<lower=0> N_ar;
  array[N_ar] int<lower=1, upper=N_orig> ar_origin;
  vector[N_ar] log_delta_ar;
  vector[N_ar] lag_log_delta;

  // Prediction
  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=N_orig> pred_origin;
  vector[N_pred] pred_lag_log_delta;

  // Hyperparameter constants (ALL user-specified from data)
  real mu_0;                // Prior mean for ν
  real<lower=0> tau_0;      // SD of μ_i across origins (FIXED, not estimated)
  real<lower=0> a_0;        // Beta shape1 for σ_i
  real<lower=0> b_0;        // Beta shape2 for σ_i
}

parameters {
  real<lower=0, upper=1> phi;              // AR(1) coefficient
  vector[N_orig] mu;                       // Origin-specific long-run mean
  real nu;                                 // Grand mean of μ_i
  vector<lower=0, upper=1>[N_orig] sigma;  // Per-origin residual SD (Beta prior)
}

model {
  // === PRIORS (matching paper exactly) ===
  phi ~ uniform(0, 1);                    // φ ~ Uniform(0,1)
  nu ~ normal(mu_0, 100);                 // ν ~ Normal(μ₀, 100²)
  mu ~ normal(nu, tau_0);                 // μ_i ~ Normal(ν, τ₀²)  — τ₀ fixed
  sigma ~ beta(a_0, b_0);                 // σ_i ~ Beta(a₀, b₀)

  // === INITIAL OBS (stationary distribution) ===
  for (n in 1:N_init) {
    int i = init_origin[n];
    real sigma_stat = sigma[i] / sqrt(1.0 - square(phi));
    log_delta_init[n] ~ normal(mu[i], sigma_stat);
  }

  // === AR OBS ===
  // log δ_it ~ Normal((1-φ)μ_i + φ·log δ_{i,t-1}, σ_i²)
  for (n in 1:N_ar) {
    int i = ar_origin[n];
    real mu_ar = (1.0 - phi) * mu[i] + phi * lag_log_delta[n];
    log_delta_ar[n] ~ normal(mu_ar, sigma[i]);
  }
}

generated quantities {
  // Out-of-sample predictions
  vector[N_pred] log_delta_pred;
  for (n in 1:N_pred) {
    int i = pred_origin[n];
    real mu_pred = (1.0 - phi) * mu[i] + phi * pred_lag_log_delta[n];
    log_delta_pred[n] = normal_rng(mu_pred, sigma[i]);
  }
}
