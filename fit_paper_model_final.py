"""
==============================================================================
Azose & Raftery (2019) — FINAL FIT
==============================================================================
All countries (200) · All periods (1990–2010) · No hold-out
Posteriors saved to posteriors_final/
==============================================================================
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

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
OUTFLOW_STAN = PROJ / "paper_outflow.stan"
INFLOW_STAN = PROJ / "paper_inflow.stan"
SAVE_DIR = PROJ / "posteriors_final"

# Hyperparameters (paper Table 3)
MU_0 = -5.0
A_0 = 2.0
B_0 = 5.0
P_0 = 2.0
Q_0 = 5.0


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

    return {
        'df': df, 'E': E,
        'countries': countries, 'c2i': c2i,
        'corridors': corridors, 'corr2i': corr2i, 'corr_origin': corr_origin,
        'pop': pop,
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
        'mu_0': MU_0,
        'a_0': A_0,
        'b_0': B_0,
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
        'p_0': P_0,
        'q_0': Q_0,
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
    tau_0 = fit_out.stan_variable('tau_0')

    print("\n  OUTFLOW MODEL:")
    print(f"    φ  = {phi.mean():.3f} [{np.percentile(phi, 2.5):.3f}, "
          f"{np.percentile(phi, 97.5):.3f}]")
    print(f"    ν  = {nu.mean():.3f} ± {nu.std():.3f}")
    print(f"    τ₀ = {tau_0.mean():.3f} ± {tau_0.std():.3f}")
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
             nu=nu, tau_0=tau_0)
    np.savez(SAVE_DIR / 'inflow_posteriors.npz',
             kappa=kappa, psi=psi)

    meta = {'countries': data['countries'], 'corridors': data['corridors']}
    with open(SAVE_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f)

    print(f"\n  Posteriors saved to {SAVE_DIR}/")
    print(f"    outflow_posteriors.npz  — phi({phi.shape}), mu{mu_draws.shape}, "
          f"sigma{sigma_draws.shape}")
    print(f"    inflow_posteriors.npz   — kappa{kappa.shape}, psi{psi.shape}")
    print(f"    metadata.json           — {len(data['countries'])} countries, "
          f"{len(data['corridors']):,} corridors")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  AZOSE & RAFTERY (2019) — FINAL FIT")
    print("  All countries · All periods (1990–2010)")
    print("=" * 70)

    data = prepare_data()

    outflow_data = build_outflow_data(data)
    inflow_data = build_inflow_data(data)

    fit_out = fit_stan(OUTFLOW_STAN, outflow_data, "OUTFLOW — all countries")
    fit_in = fit_stan(INFLOW_STAN, inflow_data, "INFLOW — all corridors")

    save_and_report(fit_out, fit_in, data)

    print(f"\n{'=' * 70}")
    print("  DONE — Final model fitted on all data.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
