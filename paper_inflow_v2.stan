// ============================================================================
// Azose & Raftery (2019) — INFLOW MODEL (Emigration Shares) — V2
//
// Paper equations:
//   π_ijt = exp(η_ijt) / Σ_k exp(η_ikt)        (softmax / inverse CLR)
//   η_ijt ~ Normal(κ_ij, ψ_ij²)                 (iid over time)
//   κ_ij  ~ Normal(0, 10²)                       (flat prior)
//   ψ_ij  ~ Beta(p₀, q₀)                         (per-corridor SD)
//
// Identifiability constraint (paper):
//   "A sum-to-zero constraint on κ_ij makes this model identifiable."
//   For each origin i: Σ_j κ_{i,j} = 0
//   (CLR vectors sum to zero by definition)
//
// Observations:
//   m_{i,·,t} | π_{i,·,t}, δ_it ~ Multinomial(N_it, π_{i,·,t})
// ============================================================================

data {
  int<lower=1> N_orig;             // Number of origins
  int<lower=1> N_corridors;        // Total active corridors
  int<lower=1> N_obs_groups;       // Number of (origin, time) groups with flows

  // Corridor metadata
  array[N_corridors] int<lower=1, upper=N_orig> corridor_origin;

  // Grouped observations: for each (origin, time) group
  array[N_obs_groups] int<lower=1> group_size;           // # corridors in group
  array[N_obs_groups] int<lower=1> group_start;          // Start index in flat arrays
  array[N_obs_groups] int<lower=1> group_N;              // N_it = total emigrants

  // Flat arrays (concatenated across groups)
  int<lower=0> N_flat;
  array[N_flat] int<lower=1, upper=N_corridors> flat_corridor;  // Corridor index
  array[N_flat] int<lower=0> flat_count;                        // m_ijt

  // Prediction
  int<lower=0> N_pred_groups;
  array[N_pred_groups] int<lower=1> pred_group_size;
  array[N_pred_groups] int<lower=1> pred_group_start;
  int<lower=0> N_pred_flat;
  array[N_pred_flat] int<lower=1, upper=N_corridors> pred_flat_corridor;

  // Hyperparameters
  real<lower=0> p_0;   // Beta shape1 for ψ_ij
  real<lower=0> q_0;   // Beta shape2 for ψ_ij
}

parameters {
  vector[N_corridors] kappa;                          // Corridor means κ_ij
  vector<lower=0, upper=1>[N_corridors] psi;          // Corridor SDs ψ_ij

  // Latent log-shares for each observation
  vector[N_flat] eta;                                 // η_ijt
}

transformed parameters {
  // Sum of κ per origin (for sum-to-zero constraint)
  vector[N_orig] kappa_sum = rep_vector(0.0, N_orig);
  for (c in 1:N_corridors) {
    kappa_sum[corridor_origin[c]] += kappa[c];
  }
}

model {
  // === PRIORS (matching paper) ===
  kappa ~ normal(0, 10);            // κ_ij ~ Normal(0, 10²)
  psi ~ beta(p_0, q_0);             // ψ_ij ~ Beta(p₀, q₀)

  // === SUM-TO-ZERO CONSTRAINT (paper identifiability) ===
  // "A sum-to-zero constraint on κ_ij makes this model identifiable."
  // Soft constraint: for each origin i, Σ_j κ_{i,j} ≈ 0
  kappa_sum ~ normal(0, 0.001);

  // === LATENT η ===
  // η_ijt ~ Normal(κ_ij, ψ_ij²)
  for (n in 1:N_flat) {
    int c = flat_corridor[n];
    eta[n] ~ normal(kappa[c], psi[c]);
  }

  // === MULTINOMIAL LIKELIHOOD ===
  // For each (origin, time) group: m ~ Multinomial(N, softmax(η))
  for (g in 1:N_obs_groups) {
    int s = group_start[g];
    int sz = group_size[g];

    // Extract η for this group and compute log_softmax
    vector[sz] eta_group;
    for (k in 1:sz)
      eta_group[k] = eta[s + k - 1];

    vector[sz] log_pi = log_softmax(eta_group);

    // Multinomial log-likelihood: Σ m_j · log(π_j) + const
    for (k in 1:sz)
      target += flat_count[s + k - 1] * log_pi[k];
  }
}

generated quantities {
  // Predict shares (draw η from posterior, compute softmax)
  vector[N_pred_flat] pred_eta;
  for (n in 1:N_pred_flat) {
    int c = pred_flat_corridor[n];
    pred_eta[n] = normal_rng(kappa[c], psi[c]);
  }
}
