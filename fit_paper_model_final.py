"""
==============================================================================
Azose & Raftery (2019) — FINAL FIT  (V2 — exact paper hyperparameters)
==============================================================================
All countries (200) · All periods (1990–2010) · No hold-out

Hyperparameters are computed from the data exactly as described in
the paper's "Prior Specification" section:

  a₀, b₀   — Beta params for σ_i: fitted so the 2.5 & 97.5% quantiles of
              Beta(a₀, b₀) match (0.15, 0.99)
  τ₀        — Fixed to 3 × SD(mean_i(log δ_it) across origins)
  μ₀        — Mean over origins i of per-origin mean log δ
  p₀, q₀   — Beta params for ψ_ij: fitted so the 2.5 & 97.5% quantiles of
              Beta(p₀, q₀) match empirical quantiles of SD grand mean of
              CLR-transformed shares

Posteriors saved to posterior_final_V2/
==============================================================================
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from scipy.special import softmax

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
N_CHAINS = 4
N_WARMUP = 500
N_SAMPLES = 500
SEED = 42
PROJ = Path(__file__).parent
DATA_FLOWS = PROJ / "data" / "azoseRaftery2019flows.csv"
DATA_GRAVITY = PROJ / "data_final" / "FINAL_GRAVITY_TRAINING_MATRIX.csv"
OUTFLOW_STAN = PROJ / "paper_outflow_v2.stan"   # V2: τ₀ as data, not parameter
INFLOW_STAN = PROJ / "paper_inflow_v2.stan"  # V2: sum-to-zero constraint on κ
SAVE_DIR = PROJ / "posterior_final_V2"


# ============================================================================
# HYPERPARAMETER CALIBRATION (exact paper procedure)
# ============================================================================

def fit_beta_quantiles(target_low, target_high, q_low=0.025, q_high=0.975):
    """
    Find Beta(a, b) parameters whose q_low and q_high quantiles
    match target_low and target_high, by minimizing the sum of
    squared differences.

    Paper: "We did this by minimizing the sum of the differences
    of the 2.5 and 97.5% quantiles from Beta(a₀, b₀) and (0.15, 0.99)."
    """
    def objective(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        q_lo = beta_dist.ppf(q_low, a, b)
        q_hi = beta_dist.ppf(q_high, a, b)
        return (q_lo - target_low) ** 2 + (q_hi - target_high) ** 2

    best = None
    best_val = 1e10
    # Grid search over starting points for robustness
    for a_init in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        for b_init in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            res = minimize(objective, [a_init, b_init],
                           method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
    return best[0], best[1]


def calibrate_outflow_hyperparams(E):
    """
    Calibrate a₀, b₀, μ₀, τ₀ from the data.

    Paper procedure:
      μ₀  = mean over origins i of (mean_t log δ_it within origin i)
      τ₀  = 3 × SD of (mean_t log δ_it) across origins
      a₀, b₀ = Beta params such that 2.5% & 97.5% quantiles ≈ (0.15, 0.99)
    """
    # Per-origin mean log δ
    origin_mean_log_delta = E.groupby('origIso')['log_delta'].mean()

    mu_0 = origin_mean_log_delta.mean()
    tau_0 = 3.0 * origin_mean_log_delta.std()

    # a₀, b₀: Beta quantiles match (0.15, 0.99)
    a_0, b_0 = fit_beta_quantiles(0.15, 0.99)

    print("\n  OUTFLOW hyperparameters (calibrated from data):")
    print(f"    μ₀  = {mu_0:.4f}  (mean of per-origin mean log δ)")
    print(f"    τ₀  = {tau_0:.4f}  (3 × SD of per-origin mean log δ = "
          f"3 × {origin_mean_log_delta.std():.4f})")
    print(f"    a₀  = {a_0:.4f}")
    print(f"    b₀  = {b_0:.4f}")
    q_lo = beta_dist.ppf(0.025, a_0, b_0)
    q_hi = beta_dist.ppf(0.975, a_0, b_0)
    print(f"    Beta({a_0:.2f}, {b_0:.2f}) → 2.5%={q_lo:.4f}, 97.5%={q_hi:.4f}  "
          f"(target: 0.15, 0.99)")

    return mu_0, tau_0, a_0, b_0


def calibrate_inflow_hyperparams(df, countries, c2i):
    """
    Calibrate p₀, q₀ from the data.

    Paper: "We found p₀ and q₀ by minimizing the sum of the differences
    between the 2.5 and 97.5% quantiles of the Beta(p₀, q₀) distribution
    and the quantiles of means by origin of SDs of clr(π_{ij,t}) over t
    for all positive flows."

    Steps:
      1. For each (origin, time), compute empirical shares π_ijt
      2. Apply CLR transform: η_ijt = log(π_ijt / geomean(π_{i·t}))
      3. For each corridor (i,j), compute SD of η_ijt over t
      4. For each origin i, compute mean of corridor SDs
      5. Target quantiles = 2.5% and 97.5% of these origin-level mean SDs
      6. Fit Beta(p₀, q₀) to match those quantiles
    """
    pos = df[df['migrantCount'] > 0].copy()

    # Step 1-2: Compute CLR-transformed shares per (origin, time)
    corridor_eta_values = {}  # (orig, dest) → list of η values over time

    for (orig, year), grp in pos.groupby(['origIso', 'year']):
        counts = grp[['destIso', 'migrantCount']].set_index('destIso')['migrantCount']
        total = counts.sum()
        if total <= 0:
            continue
        shares = counts / total
        # CLR transform: log(π_j) - mean(log(π_j))
        log_shares = np.log(shares.values)
        clr_vals = log_shares - log_shares.mean()

        for dest, clr_val in zip(shares.index, clr_vals):
            key = (orig, dest)
            if key not in corridor_eta_values:
                corridor_eta_values[key] = []
            corridor_eta_values[key].append(clr_val)

    # Step 3: SD of CLR values over time for each corridor
    corridor_sds = {}
    for (orig, dest), vals in corridor_eta_values.items():
        if len(vals) >= 2:
            corridor_sds[(orig, dest)] = np.std(vals, ddof=1)

    # Step 4: Mean SD per origin
    origin_mean_sds = {}
    for (orig, dest), sd_val in corridor_sds.items():
        if orig not in origin_mean_sds:
            origin_mean_sds[orig] = []
        origin_mean_sds[orig].append(sd_val)

    origin_mean_sd_values = np.array([np.mean(v) for v in origin_mean_sds.values()])

    # Step 5: Target quantiles
    target_low = np.percentile(origin_mean_sd_values, 2.5)
    target_high = np.percentile(origin_mean_sd_values, 97.5)

    # Step 6: Fit Beta
    p_0, q_0 = fit_beta_quantiles(target_low, target_high)

    print("\n  INFLOW hyperparameters (calibrated from data):")
    print(f"    CLR SD statistics:")
    print(f"      # corridors with ≥2 obs: {len(corridor_sds):,}")
    print(f"      # origins: {len(origin_mean_sds)}")
    print(f"      Origin mean SD: mean={origin_mean_sd_values.mean():.4f}, "
          f"median={np.median(origin_mean_sd_values):.4f}")
    print(f"    Target quantiles (from data):")
    print(f"      2.5%  = {target_low:.4f}")
    print(f"      97.5% = {target_high:.4f}")
    print(f"    p₀  = {p_0:.4f}")
    print(f"    q₀  = {q_0:.4f}")
    q_lo = beta_dist.ppf(0.025, p_0, q_0)
    q_hi = beta_dist.ppf(0.975, p_0, q_0)
    print(f"    Beta({p_0:.2f}, {q_0:.2f}) → 2.5%={q_lo:.4f}, 97.5%={q_hi:.4f}")

    return p_0, q_0


# ============================================================================
# 1. DATA PREPARATION — ALL COUNTRIES, ALL PERIODS
# ============================================================================

def prepare_data():
    print("=" * 70)
    print("DATA PREPARATION — FINAL (all countries, 1990–2010)")
    print("=" * 70)

    df = pd.read_csv(DATA_FLOWS)
    df = df[df['origIso'] != df['destIso']].copy()

    # --- Population ---
    grav = pd.read_csv(DATA_GRAVITY)
    pop = grav[['iso3_o', 'year', 'pop_o']].drop_duplicates(['iso3_o', 'year'])
    pop.columns = ['origIso', 'year', 'pop']

    df = df.merge(pop, on=['origIso', 'year'], how='left')
    for c in df['origIso'].unique():
        mask = (df['origIso'] == c) & df['pop'].isna()
        if mask.any():
            med = df.loc[df['origIso'] == c, 'pop'].median()
            df.loc[mask, 'pop'] = med if not pd.isna(med) else 1e6

    # --- Total emigration & rate ---
    E = df.groupby(['origIso', 'year'])['migrantCount'].sum().reset_index()
    E.columns = ['origIso', 'year', 'E_it']
    E = E.merge(pop, on=['origIso', 'year'], how='left')
    E['pop'] = E['pop'].fillna(E.groupby('origIso')['pop'].transform('median'))
    E['pop'] = E['pop'].fillna(1e6)
    E['delta'] = E['E_it'] / E['pop']
    E.loc[E['delta'] <= 0, 'delta'] = 1e-10
    E['log_delta'] = np.log(E['delta'])

    # --- Country & corridor indexing ---
    countries = sorted(set(df['origIso'].unique()) | set(df['destIso'].unique()))
    c2i = {c: i + 1 for i, c in enumerate(countries)}  # 1-indexed for Stan

    pos = df[df['migrantCount'] > 0].copy()
    pos['pair'] = pos['origIso'] + '_' + pos['destIso']
    corridors = sorted(pos['pair'].unique())
    corr2i = {c: i + 1 for i, c in enumerate(corridors)}
    corr_origin = [c2i[c.split('_')[0]] for c in corridors]

    print(f"Countries: {len(countries)}")
    print(f"Corridors (>0 flow): {len(corridors):,}")
    print(f"Total obs: {len(df):,}")
    print(f"Positive obs: {len(pos):,}")
    print(f"Periods: {sorted(df['year'].unique())}")
    print(f"Emigration rate — mean: {E['delta'].mean():.4f}, "
          f"mean log δ: {E['log_delta'].mean():.2f}")

    # --- Calibrate hyperparameters from data (exact paper procedure) ---
    print(f"\n{'=' * 70}")
    print("HYPERPARAMETER CALIBRATION (paper procedure)")
    print(f"{'=' * 70}")

    mu_0, tau_0, a_0, b_0 = calibrate_outflow_hyperparams(E)
    p_0, q_0 = calibrate_inflow_hyperparams(df, countries, c2i)

    return {
        'df': df, 'E': E,
        'countries': countries, 'c2i': c2i,
        'corridors': corridors, 'corr2i': corr2i, 'corr_origin': corr_origin,
        'pop': pop,
        # Calibrated hyperparameters
        'mu_0': mu_0, 'tau_0': tau_0, 'a_0': a_0, 'b_0': b_0,
        'p_0': p_0, 'q_0': q_0,
    }


# ============================================================================
# 2. BUILD OUTFLOW STAN DATA
# ============================================================================

def build_outflow_data(data):
    print("\n--- Building OUTFLOW model data ---")

    E = data['E']
    c2i = data['c2i']

    E = E.sort_values(['origIso', 'year']).reset_index(drop=True)

    init_origin, init_ld = [], []
    ar_origin, ar_ld, ar_lag = [], [], []

    for orig, grp in E.groupby('origIso'):
        oidx = c2i[orig]
        years = grp['year'].values
        lds = grp['log_delta'].values

        init_origin.append(oidx)
        init_ld.append(lds[0])

        for i in range(1, len(grp)):
            if years[i] - years[i - 1] == 5:
                ar_origin.append(oidx)
                ar_ld.append(lds[i])
                ar_lag.append(lds[i - 1])
            else:
                init_origin.append(oidx)
                init_ld.append(lds[i])

    print(f"  Init: {len(init_origin)} | AR: {len(ar_origin)}")

    stan_data = {
        'N_orig': len(data['countries']),
        'N_init': len(init_origin),
        'init_origin': np.array(init_origin, dtype=int),
        'log_delta_init': np.array(init_ld),
        'N_ar': len(ar_origin),
        'ar_origin': np.array(ar_origin, dtype=int),
        'log_delta_ar': np.array(ar_ld),
        'lag_log_delta': np.array(ar_lag),
        # No prediction — set to empty
        'N_pred': 0,
        'pred_origin': np.array([], dtype=int),
        'pred_lag_log_delta': np.array([]),
        # Calibrated hyperparameters (paper procedure)
        'mu_0': data['mu_0'],
        'tau_0': data['tau_0'],
        'a_0': data['a_0'],
        'b_0': data['b_0'],
    }
    return stan_data


# ============================================================================
# 3. BUILD INFLOW STAN DATA
# ============================================================================

def build_inflow_data(data):
    print("\n--- Building INFLOW model data ---")

    df = data['df']
    corr2i = data['corr2i']
    corr_origin = data['corr_origin']

    pos = df[df['migrantCount'] > 0].copy()
    pos['pair'] = pos['origIso'] + '_' + pos['destIso']
    pos = pos[pos['pair'].isin(corr2i)].copy()

    groups = []
    flat_corridor = []
    flat_count = []

    for (orig, year), grp in pos.groupby(['origIso', 'year']):
        corridors_in_grp = []
        counts_in_grp = []
        for _, row in grp.iterrows():
            cidx = corr2i[row['pair']]
            corridors_in_grp.append(cidx)
            counts_in_grp.append(int(row['migrantCount']))

        start = len(flat_corridor) + 1
        groups.append({
            'size': len(corridors_in_grp),
            'start': start,
            'N_it': grp['migrantCount'].sum(),
        })
        flat_corridor.extend(corridors_in_grp)
        flat_count.extend(counts_in_grp)

    N_flat = len(flat_corridor)
    print(f"  Groups: {len(groups):,} | Flat obs: {N_flat:,}")

    stan_data = {
        'N_orig': len(data['countries']),
        'N_corridors': len(data['corridors']),
        'N_obs_groups': len(groups),
        'corridor_origin': np.array(corr_origin, dtype=int),
        'group_size': np.array([g['size'] for g in groups], dtype=int),
        'group_start': np.array([g['start'] for g in groups], dtype=int),
        'group_N': np.array([g['N_it'] for g in groups], dtype=int),
        'N_flat': N_flat,
        'flat_corridor': np.array(flat_corridor, dtype=int),
        'flat_count': np.array(flat_count, dtype=int),
        # No prediction — set to empty
        'N_pred_groups': 0,
        'pred_group_size': np.array([], dtype=int),
        'pred_group_start': np.array([], dtype=int),
        'N_pred_flat': 0,
        'pred_flat_corridor': np.array([], dtype=int),
        # Calibrated hyperparameters (paper procedure)
        'p_0': data['p_0'],
        'q_0': data['q_0'],
    }
    return stan_data


# ============================================================================
# 4. FIT
# ============================================================================

def fit_stan(stan_file, stan_data, label):
    from cmdstanpy import CmdStanModel

    print(f"\n{'=' * 70}")
    print(f"FITTING: {label}")
    print(f"{'=' * 70}")

    model = CmdStanModel(stan_file=str(stan_file))
    print("Compiled.")

    t0 = time.time()
    fit = model.sample(
        data=stan_data,
        chains=N_CHAINS,
        iter_warmup=N_WARMUP,
        iter_sampling=N_SAMPLES,
        seed=SEED,
        show_console=False,
        adapt_delta=0.95,
        max_treedepth=12,
        threads_per_chain=4,
        inits=0,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(fit.diagnose())

    return fit


# ============================================================================
# 5. SAVE POSTERIORS & DIAGNOSTICS
# ============================================================================

def save_and_report(fit_out, fit_in, data):
    print(f"\n{'=' * 70}")
    print("POSTERIORS & DIAGNOSTICS")
    print(f"{'=' * 70}")

    # --- Outflow diagnostics ---
    phi = fit_out.stan_variable('phi')
    mu_draws = fit_out.stan_variable('mu')
    sigma_draws = fit_out.stan_variable('sigma')
    nu = fit_out.stan_variable('nu')

    print("\n  OUTFLOW MODEL:")
    print(f"    φ  = {phi.mean():.3f} [{np.percentile(phi, 2.5):.3f}, "
          f"{np.percentile(phi, 97.5):.3f}]")
    print(f"    ν  = {nu.mean():.3f} ± {nu.std():.3f}")
    print(f"    τ₀ = {data['tau_0']:.4f}  (fixed from data)")
    print(f"    σ_i mean = {sigma_draws.mean():.3f}, range "
          f"[{sigma_draws.mean(axis=0).min():.3f}, {sigma_draws.mean(axis=0).max():.3f}]")
    print(f"    μ_i mean = {mu_draws.mean():.3f}, SD across countries = "
          f"{mu_draws.mean(axis=0).std():.3f}")

    # --- Inflow diagnostics ---
    kappa = fit_in.stan_variable('kappa')
    psi = fit_in.stan_variable('psi')

    print("\n  INFLOW MODEL:")
    print(f"    κ_ij mean = {kappa.mean():.3f}, SD across corridors = "
          f"{kappa.mean(axis=0).std():.3f}")
    print(f"    ψ_ij mean = {psi.mean():.3f}, range "
          f"[{psi.mean(axis=0).min():.3f}, {psi.mean(axis=0).max():.3f}]")

    # --- Save posteriors ---
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.savez(SAVE_DIR / 'outflow_posteriors.npz',
             phi=phi, mu=mu_draws, sigma=sigma_draws,
             nu=nu)
    np.savez(SAVE_DIR / 'inflow_posteriors.npz',
             kappa=kappa, psi=psi)

    # Save hyperparameters & metadata
    meta = {
        'countries': data['countries'],
        'corridors': data['corridors'],
        'hyperparameters': {
            'mu_0': float(data['mu_0']),
            'tau_0': float(data['tau_0']),
            'a_0': float(data['a_0']),
            'b_0': float(data['b_0']),
            'p_0': float(data['p_0']),
            'q_0': float(data['q_0']),
        },
        'description': (
            'Hyperparameters calibrated from data using exact paper procedure. '
            'tau_0 is fixed (not estimated). '
            'a_0/b_0 fitted to Beta quantiles (0.15, 0.99). '
            'p_0/q_0 fitted to empirical CLR SD quantiles.'
        ),
    }
    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Posteriors saved to {SAVE_DIR}/")
    print(f"    outflow_posteriors.npz  — phi({phi.shape}), mu{mu_draws.shape}, "
          f"sigma{sigma_draws.shape}")
    print(f"    inflow_posteriors.npz   — kappa{kappa.shape}, psi{psi.shape}")
    print(f"    metadata.json           — {len(data['countries'])} countries, "
          f"{len(data['corridors']):,} corridors, calibrated hyperparams")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  AZOSE & RAFTERY (2019) — FINAL FIT V2")
    print("  Exact paper hyperparameter calibration")
    print("  All countries · All periods (1990–2010)")
    print("=" * 70)

    data = prepare_data()

    outflow_data = build_outflow_data(data)
    inflow_data = build_inflow_data(data)

    fit_out = fit_stan(OUTFLOW_STAN, outflow_data, "OUTFLOW — all countries")
    fit_in = fit_stan(INFLOW_STAN, inflow_data, "INFLOW — all corridors")

    save_and_report(fit_out, fit_in, data)

    print(f"\n{'=' * 70}")
    print("  DONE — Final model V2 fitted on all data.")
    print(f"  Posteriors: {SAVE_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
