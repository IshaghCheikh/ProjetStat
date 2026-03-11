// ============================================================================
// Emigration Shares Model (Azose & Raftery 2019)
//
// Models log-shares η_lt = log(π_ijt) where π_ijt = m_ijt / E_it
// Decomposition: η_lt = k_l + ε_lt
//   k_l = corridor-specific mean (hierarchical within origin)
//   ε_lt = AR(1) deviation: ε_lt = φ·ε_{l,t-1} + ν_lt
//
// Equivalent formulation:
//   Initial:  η_l1 ~ N(k_l, σ_ε / √(1-φ²))
//   AR:       η_lt ~ N(k_l + φ·(η_{l,t-1} - k_l), σ_ε)
// ============================================================================

data {
  int<lower=1> N_orig;               // Number of origin countries
  int<lower=1> N_corridors;          // Number of active corridors (i,j)
  array[N_corridors] int<lower=1, upper=N_orig> corridor_origin;  // Origin of each corridor

  // --- Initial observations (no lag available) ---
  int<lower=0> N_init;
  array[N_init] int<lower=1, upper=N_corridors> corr_init;
  vector[N_init] eta_init;

  // --- AR observations (consecutive 5-year lag available) ---
  int<lower=0> N_ar;
  array[N_ar] int<lower=1, upper=N_corridors> corr_ar;
  vector[N_ar] eta_ar;
  vector[N_ar] eta_lag;

  // --- Prediction: corridors with 2005 observation ---
  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=N_corridors> pred_corridor;
  vector[N_pred] pred_eta_last;      // η from 2005 (lag for 2010 prediction)
}

parameters {
  // Corridor means (non-centered parameterization)
  vector[N_corridors] k_raw;

  // Origin-level hyperparameters for corridor means
  vector[N_orig] mu_k;               // Mean corridor effect per origin
  vector<lower=0>[N_orig] tau_k;     // SD of corridor effects per origin

  // Global hyperpriors
  real mu_mu_k;                       // Grand mean of μ_k
  real<lower=0> sigma_mu_k;          // SD of μ_k across origins
  real<lower=0> tau_hyper;           // Scale for τ_k

  // AR(1) coefficient
  real<lower=-0.99, upper=0.99> phi;

  // Residual SD
  real<lower=0> sigma_eps;
}

transformed parameters {
  // Non-centered: k_l = μ_{k,i(l)} + τ_{k,i(l)} · k_raw_l
  vector[N_corridors] k;
  for (l in 1:N_corridors)
    k[l] = mu_k[corridor_origin[l]] + tau_k[corridor_origin[l]] * k_raw[l];
}

model {
  // === HYPERPRIORS ===
  mu_mu_k ~ normal(-4, 3);           // Log-shares are negative on average
  sigma_mu_k ~ normal(0, 2);
  tau_hyper ~ normal(0, 1);

  // === ORIGIN-LEVEL PRIORS ===
  mu_k ~ normal(mu_mu_k, sigma_mu_k);
  tau_k ~ normal(0, tau_hyper);       // Half-normal (lower=0 constraint)

  // === CORRIDOR MEANS ===
  k_raw ~ std_normal();

  // === AR(1) & RESIDUAL ===
  phi ~ normal(0.3, 0.3);
  sigma_eps ~ normal(0, 2);

  // === LIKELIHOOD: INITIAL OBS (stationary distribution) ===
  {
    real sigma_init = sigma_eps / sqrt(1.0 - square(phi));
    vector[N_init] mu_init;
    for (n in 1:N_init)
      mu_init[n] = k[corr_init[n]];
    eta_init ~ normal(mu_init, sigma_init);
  }

  // === LIKELIHOOD: AR OBS ===
  {
    vector[N_ar] mu_ar;
    for (n in 1:N_ar)
      mu_ar[n] = k[corr_ar[n]] + phi * (eta_lag[n] - k[corr_ar[n]]);
    eta_ar ~ normal(mu_ar, sigma_eps);
  }
}

generated quantities {
  // --- Log-likelihood for model comparison ---
  vector[N_init + N_ar] log_lik;
  {
    real sigma_init = sigma_eps / sqrt(1.0 - square(phi));
    for (n in 1:N_init)
      log_lik[n] = normal_lpdf(eta_init[n] | k[corr_init[n]], sigma_init);
    for (n in 1:N_ar) {
      real mu_n = k[corr_ar[n]] + phi * (eta_lag[n] - k[corr_ar[n]]);
      log_lik[N_init + n] = normal_lpdf(eta_ar[n] | mu_n, sigma_eps);
    }
  }

  // --- Out-of-sample predictions (2010) ---
  vector[N_pred] eta_pred;
  for (n in 1:N_pred) {
    real mu_pred = k[pred_corridor[n]]
                   + phi * (pred_eta_last[n] - k[pred_corridor[n]]);
    eta_pred[n] = normal_rng(mu_pred, sigma_eps);
  }
}
