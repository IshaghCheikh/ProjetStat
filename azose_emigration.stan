// ============================================================================
// Total Emigration Model (Azose & Raftery 2019)
//
// Models log(E_it) = total emigration from origin i in period t
// AR(1) structure: log(E_it) = (1-ψ)·a_i + ψ·log(E_{i,t-1}) + u_it
//   a_i = origin-specific long-run mean (hierarchical)
//   ψ   = AR(1) persistence coefficient
//
// Initial obs use stationary distribution:
//   log(E_i1) ~ N(a_i, σ_u / √(1-ψ²))
// ============================================================================

data {
  int<lower=1> N_orig;

  // --- Initial observations (first period per origin) ---
  int<lower=0> N_init;
  array[N_init] int<lower=1, upper=N_orig> init_origin;
  vector[N_init] log_E_init;

  // --- AR observations (consecutive lag available) ---
  int<lower=0> N_ar;
  array[N_ar] int<lower=1, upper=N_orig> ar_origin;
  vector[N_ar] log_E_ar;
  vector[N_ar] lag_log_E;

  // --- Prediction: origins with 2005 data ---
  int<lower=0> N_pred;
  array[N_pred] int<lower=1, upper=N_orig> pred_origin;
  vector[N_pred] pred_lag_log_E;     // log(E_{i,2005})
}

parameters {
  // Origin effects (non-centered)
  vector[N_orig] a_raw;
  real mu_a;                          // Grand mean
  real<lower=0> sigma_a;             // SD across origins

  // AR(1) coefficient
  real<lower=-0.99, upper=0.99> psi;

  // Residual SD
  real<lower=0> sigma_u;
}

transformed parameters {
  vector[N_orig] a = mu_a + sigma_a * a_raw;
}

model {
  // === PRIORS ===
  mu_a ~ normal(10, 5);              // log(E) ~ 10 → E ~ 22K migrants
  sigma_a ~ normal(0, 3);
  a_raw ~ std_normal();
  psi ~ normal(0.5, 0.3);
  sigma_u ~ normal(0, 2);

  // === INITIAL OBS (stationary) ===
  {
    real sigma_stat = sigma_u / sqrt(1.0 - square(psi));
    log_E_init ~ normal(a[init_origin], sigma_stat);
  }

  // === AR OBS ===
  // log(E_it) ~ N((1-ψ)·a_i + ψ·log(E_{i,t-1}), σ_u)
  {
    vector[N_ar] mu_ar;
    for (n in 1:N_ar)
      mu_ar[n] = (1.0 - psi) * a[ar_origin[n]] + psi * lag_log_E[n];
    log_E_ar ~ normal(mu_ar, sigma_u);
  }
}

generated quantities {
  // --- Out-of-sample predictions (2010) ---
  vector[N_pred] log_E_pred;
  for (n in 1:N_pred) {
    real mu_pred = (1.0 - psi) * a[pred_origin[n]] + psi * pred_lag_log_E[n];
    log_E_pred[n] = normal_rng(mu_pred, sigma_u);
  }
}
