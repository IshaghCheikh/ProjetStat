// ============================================================================
// Hierarchical Bayesian Model for Bilateral Migration Flows
// Re-implementation of Azose & Raftery (2019) methodology
//
// Two-part hurdle log-normal model:
//   Part 1 (zero model):  Bernoulli-logit for P(flow > 0)
//   Part 2 (positive model): Log-normal for magnitude given flow > 0
//
// Hierarchical structure:
//   - Origin random effects (push factors)
//   - Destination random effects (pull factors)
//   - Time fixed effects (period dummies)
//   - Corridor-level AR(1) persistence (via lagged log-flow)
// ============================================================================

data {
  // --- Dimensions ---
  int<lower=1> N_all;           // Total number of observations (all flows)
  int<lower=1> N_pos;           // Number of positive flows
  int<lower=1> N_orig;          // Number of origin countries
  int<lower=1> N_dest;          // Number of destination countries
  int<lower=1> N_time;          // Number of time periods

  // --- Full data (for zero model) ---
  array[N_all] int<lower=1, upper=N_orig> orig_all;   // Origin index for each obs
  array[N_all] int<lower=1, upper=N_dest> dest_all;   // Destination index for each obs
  array[N_all] int<lower=1, upper=N_time> time_all;   // Time index for each obs
  array[N_all] int<lower=0, upper=1> is_positive;     // 1 if flow > 0

  // --- Positive flows only (for positive model) ---
  array[N_pos] int<lower=1, upper=N_orig> orig_pos;   // Origin index (positive flows)
  array[N_pos] int<lower=1, upper=N_dest> dest_pos;   // Destination index (positive flows)
  array[N_pos] int<lower=1, upper=N_time> time_pos;   // Time index (positive flows)
  vector[N_pos] log_flow;                              // log(migrantCount) for positive flows

  // --- AR(1) component: lagged log-flow (0 if no lag available) ---
  vector[N_pos] lag_log_flow;       // Lagged log-flow (from previous period)
  array[N_pos] int<lower=0, upper=1> has_lag;  // 1 if lag is available
}

parameters {
  // --- Zero model (logistic) ---
  real gamma_0;                         // Intercept
  vector[N_orig] gamma_orig_raw;        // Raw origin effects (non-centered)
  vector[N_dest] gamma_dest_raw;        // Raw destination effects (non-centered)
  real<lower=0> sigma_gamma_orig;       // SD of origin effects
  real<lower=0> sigma_gamma_dest;       // SD of destination effects

  // --- Positive model (log-normal) ---
  real alpha;                           // Global intercept
  vector[N_orig] alpha_orig_raw;        // Raw origin effects (non-centered)
  vector[N_dest] beta_dest_raw;         // Raw destination effects (non-centered)
  vector[N_time] delta_time;            // Time fixed effects (sum-to-zero)
  real<lower=0> sigma_alpha;            // SD of origin effects
  real<lower=0> sigma_beta;             // SD of destination effects
  real<lower=0> sigma;                  // Residual SD

  // --- AR(1) persistence ---
  real<lower=-1, upper=1> rho;          // AR(1) coefficient on lagged log-flow
}

transformed parameters {
  // Non-centered parameterization for random effects (better sampling)
  vector[N_orig] gamma_orig = sigma_gamma_orig * gamma_orig_raw;
  vector[N_dest] gamma_dest = sigma_gamma_dest * gamma_dest_raw;
  vector[N_orig] alpha_orig = sigma_alpha * alpha_orig_raw;
  vector[N_dest] beta_dest  = sigma_beta  * beta_dest_raw;
}

model {
  // ========================
  // PRIORS
  // ========================

  // --- Zero model priors ---
  gamma_0 ~ normal(0, 5);
  gamma_orig_raw ~ std_normal();    // implies gamma_orig ~ N(0, sigma_gamma_orig)
  gamma_dest_raw ~ std_normal();    // implies gamma_dest ~ N(0, sigma_gamma_dest)
  sigma_gamma_orig ~ normal(0, 2);  // half-normal (due to <lower=0>)
  sigma_gamma_dest ~ normal(0, 2);

  // --- Positive model priors ---
  alpha ~ normal(5, 5);             // log-flow intercept (e^5 ≈ 148)
  alpha_orig_raw ~ std_normal();    // implies alpha_orig ~ N(0, sigma_alpha)
  beta_dest_raw  ~ std_normal();    // implies beta_dest  ~ N(0, sigma_beta)
  sigma_alpha ~ normal(0, 3);
  sigma_beta  ~ normal(0, 3);
  delta_time ~ normal(0, 2);        // Time effects
  sigma ~ normal(0, 5);             // Residual SD (half-normal)
  rho ~ normal(0.5, 0.3);           // AR(1) prior centered on moderate persistence

  // ========================
  // LIKELIHOOD
  // ========================

  // --- Part 1: Zero/non-zero model (logistic) ---
  {
    vector[N_all] logit_p;
    for (n in 1:N_all) {
      logit_p[n] = gamma_0 + gamma_orig[orig_all[n]] + gamma_dest[dest_all[n]];
    }
    is_positive ~ bernoulli_logit(logit_p);
  }

  // --- Part 2: Positive flow model (log-normal) ---
  {
    vector[N_pos] mu;
    for (n in 1:N_pos) {
      mu[n] = alpha + alpha_orig[orig_pos[n]] + beta_dest[dest_pos[n]]
              + delta_time[time_pos[n]];
      // Add AR(1) component if lagged value is available
      if (has_lag[n] == 1) {
        mu[n] += rho * lag_log_flow[n];
      }
    }
    log_flow ~ normal(mu, sigma);
  }
}

generated quantities {
  // Posterior predictive checks: log-likelihood for LOO-CV
  vector[N_pos] log_lik;
  vector[N_pos] y_rep;  // Posterior predictive draws

  for (n in 1:N_pos) {
    real mu_n = alpha + alpha_orig[orig_pos[n]] + beta_dest[dest_pos[n]]
                + delta_time[time_pos[n]];
    if (has_lag[n] == 1) {
      mu_n += rho * lag_log_flow[n];
    }
    log_lik[n] = normal_lpdf(log_flow[n] | mu_n, sigma);
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}
