"""
==============================================================================
Azose & Raftery (2019) — Exact Paper Equations
==============================================================================

Observations:   m_{i,·,t} | π, δ ~ Multinomial(N_it, π_{i,·,t})

Outflow:        log δ_it ~ N((1-φ)μ_i + φ·log δ_{i,t-1}, σ_i²)
                φ ~ Uniform(0,1)
                μ_i ~ N(ν, τ₀²)
                ν ~ N(μ₀, 100²)
                σ_i ~ Beta(a₀, b₀)

Inflow:         π_ijt = softmax(η_{i,·,t})_j
                η_ijt ~ N(κ_ij, ψ_ij²)
                κ_ij ~ N(0, 10²)
                ψ_ij ~ Beta(p₀, q₀)

Forecast:       m̂_ij,2010 = δ̂_i,2010 · P_i,2010 · π̂_ij,2010

Train: 1990-2005, Test: 2010 (out-of-sample)
==============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import softmax

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
TOP_N = 50
N_CHAINS = 4
N_WARMUP = 1000
N_SAMPLES = 1000
SEED = 42
PROJ = Path(__file__).parent
DATA_FLOWS = PROJ / "data" / "azoseRaftery2019flows.csv"
DATA_GRAVITY = PROJ / "data_final" / "FINAL_GRAVITY_TRAINING_MATRIX.csv"
OUTFLOW_STAN = PROJ / "paper_outflow.stan"
INFLOW_STAN = PROJ / "paper_inflow.stan"
OUT = PROJ / "graphs"

# Hyperparameters (paper Table 3)
MU_0 = -5.0       # Prior mean for ν (log emigration rate ~ -5 → 0.7%)
A_0 = 2.0          # Beta(a₀, b₀) for σ_i — weakly informative
B_0 = 5.0           
P_0 = 2.0          # Beta(p₀, q₀) for ψ_ij
Q_0 = 5.0


# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

def prepare_data():
    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    df = pd.read_csv(DATA_FLOWS)

    # --- Get population from gravity matrix ---
    grav = pd.read_csv(DATA_GRAVITY)
    pop = grav[['iso3_o', 'year', 'pop_o']].drop_duplicates(['iso3_o', 'year'])
    pop.columns = ['origIso', 'year', 'pop']
    print(f"Population data: {len(pop)} origin-year pairs")

    # --- Top N countries ---
    vol = pd.concat([
        df.groupby('origIso')['migrantCount'].sum(),
        df.groupby('destIso')['migrantCount'].sum()
    ], axis=1).sum(axis=1).sort_values(ascending=False)
    top = vol.head(TOP_N).index.tolist()

    df = df[df['origIso'].isin(top) & df['destIso'].isin(top)].copy()
    df = df[df['origIso'] != df['destIso']].copy()

    # --- Merge population ---
    df = df.merge(pop, on=['origIso', 'year'], how='left')
    # Fill missing population with median per country
    for c in df['origIso'].unique():
        mask = (df['origIso'] == c) & df['pop'].isna()
        if mask.any():
            med = df.loc[df['origIso'] == c, 'pop'].median()
            df.loc[mask, 'pop'] = med if not np.isnan(med) else 1e6

    # --- Total emigration & emigration rate ---
    E = df.groupby(['origIso', 'year'])['migrantCount'].sum().reset_index()
    E.columns = ['origIso', 'year', 'E_it']
    E = E.merge(pop, on=['origIso', 'year'], how='left')
    # Fill any remaining NaN population
    E['pop'] = E['pop'].fillna(E.groupby('origIso')['pop'].transform('median'))
    E['pop'] = E['pop'].fillna(1e6)
    E['delta'] = E['E_it'] / E['pop']  # emigration rate
    E.loc[E['delta'] <= 0, 'delta'] = 1e-10
    E['log_delta'] = np.log(E['delta'])

    # --- Train / Test ---
    df_train = df[df['year'] < 2010].copy()
    df_test = df[df['year'] == 2010].copy()
    E_train = E[E['year'] < 2010].copy()
    E_test = E[E['year'] == 2010].copy()

    print(f"Subset to top {TOP_N}: {len(df):,} obs")
    print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")
    print(f"Emigration rate stats (train):")
    print(f"  mean δ = {E_train['delta'].mean():.4f} ({E_train['delta'].mean()*100:.2f}%)")
    print(f"  mean log δ = {E_train['log_delta'].mean():.2f}")

    # --- Country index ---
    countries = sorted(set(df['origIso'].unique()) | set(df['destIso'].unique()))
    c2i = {c: i + 1 for i, c in enumerate(countries)}

    # --- Corridor index ---
    # Only corridors with at least one positive flow in training
    pos_train = df_train[df_train['migrantCount'] > 0].copy()
    pos_train['pair'] = pos_train['origIso'] + '_' + pos_train['destIso']
    corridors = sorted(pos_train['pair'].unique())
    corr2i = {c: i + 1 for i, c in enumerate(corridors)}
    corr_origin = [c2i[c.split('_')[0]] for c in corridors]

    print(f"Countries: {len(countries)} | Corridors: {len(corridors):,}")

    return {
        'df': df, 'df_train': df_train, 'df_test': df_test,
        'E': E, 'E_train': E_train, 'E_test': E_test,
        'countries': countries, 'c2i': c2i,
        'corridors': corridors, 'corr2i': corr2i, 'corr_origin': corr_origin,
        'pop': pop,
    }


# ============================================================================
# 2. BUILD OUTFLOW STAN DATA
# ============================================================================

def build_outflow_data(data):
    print("\n--- Building OUTFLOW model data ---")

    E_train = data['E_train']
    c2i = data['c2i']

    E_train = E_train.sort_values(['origIso', 'year']).reset_index(drop=True)

    init_origin, init_ld = [], []
    ar_origin, ar_ld, ar_lag = [], [], []

    for orig, grp in E_train.groupby('origIso'):
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

    # Prediction: 2005 values as lag for 2010
    E_2005 = E_train[E_train['year'] == 2005]
    pred_origin = [c2i[r['origIso']] for _, r in E_2005.iterrows()]
    pred_lag = E_2005['log_delta'].values.tolist()

    print(f"  Init: {len(init_origin)} | AR: {len(ar_origin)} | Pred: {len(pred_origin)}")

    stan_data = {
        'N_orig': len(data['countries']),
        'N_init': len(init_origin),
        'init_origin': np.array(init_origin, dtype=int),
        'log_delta_init': np.array(init_ld),
        'N_ar': len(ar_origin),
        'ar_origin': np.array(ar_origin, dtype=int),
        'log_delta_ar': np.array(ar_ld),
        'lag_log_delta': np.array(ar_lag),
        'N_pred': len(pred_origin),
        'pred_origin': np.array(pred_origin, dtype=int),
        'pred_lag_log_delta': np.array(pred_lag),
        'mu_0': MU_0,
        'a_0': A_0,
        'b_0': B_0,
    }
    data['outflow_pred_origins'] = E_2005['origIso'].values
    data['outflow_pred_pop'] = E_2005.merge(
        data['E'][data['E']['year'] == 2010][['origIso', 'pop']],
        on='origIso', how='left', suffixes=('_2005', '_2010')
    )['pop_2010'].values

    return stan_data


# ============================================================================
# 3. BUILD INFLOW STAN DATA
# ============================================================================

def build_inflow_data(data):
    print("\n--- Building INFLOW model data ---")

    df_train = data['df_train']
    corr2i = data['corr2i']
    corr_origin = data['corr_origin']

    # --- Build (origin, time) groups ---
    pos = df_train[df_train['migrantCount'] > 0].copy()
    pos['pair'] = pos['origIso'] + '_' + pos['destIso']
    # Only include corridors in our index
    pos = pos[pos['pair'].isin(corr2i)].copy()

    groups = []
    flat_corridor = []
    flat_count = []

    for (orig, year), grp in pos.groupby(['origIso', 'year']):
        N_it = grp['migrantCount'].sum()
        corridors_in_group = []
        counts_in_group = []
        for _, row in grp.iterrows():
            cidx = corr2i[row['pair']]
            corridors_in_group.append(cidx)
            counts_in_group.append(int(row['migrantCount']))

        start = len(flat_corridor) + 1  # 1-indexed for Stan
        groups.append({
            'size': len(corridors_in_group),
            'start': start,
            'N_it': N_it,
        })
        flat_corridor.extend(corridors_in_group)
        flat_count.extend(counts_in_group)

    # --- Prediction groups (2010) ---
    df_test = data['df_test']
    pos_test = df_test[df_test['migrantCount'] > 0].copy()
    pos_test['pair'] = pos_test['origIso'] + '_' + pos_test['destIso']
    pos_test = pos_test[pos_test['pair'].isin(corr2i)].copy()

    pred_groups = []
    pred_flat_corridor = []
    pred_pairs = []

    for (orig, year), grp in pos_test.groupby(['origIso', 'year']):
        start = len(pred_flat_corridor) + 1
        corrs = []
        for _, row in grp.iterrows():
            cidx = corr2i[row['pair']]
            corrs.append(cidx)
            pred_flat_corridor.append(cidx)
            pred_pairs.append(row['pair'])
        pred_groups.append({'size': len(corrs), 'start': start})

    N_flat = len(flat_corridor)
    N_pred_flat = len(pred_flat_corridor)

    print(f"  Train groups: {len(groups)} | Flat obs: {N_flat}")
    print(f"  Pred groups: {len(pred_groups)} | Pred corridors: {N_pred_flat}")

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
        'N_pred_groups': len(pred_groups),
        'pred_group_size': np.array([g['size'] for g in pred_groups], dtype=int),
        'pred_group_start': np.array([g['start'] for g in pred_groups], dtype=int),
        'N_pred_flat': N_pred_flat,
        'pred_flat_corridor': np.array(pred_flat_corridor, dtype=int),
        'p_0': P_0,
        'q_0': Q_0,
    }

    data['pred_pairs_inflow'] = pred_pairs

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
        inits=0,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(fit.diagnose())

    return fit


# ============================================================================
# 5. PREDICTION
# ============================================================================

def predict_2010(fit_outflow, fit_inflow, data):
    print(f"\n{'=' * 70}")
    print("OUT-OF-SAMPLE PREDICTION (2010)")
    print(f"{'=' * 70}")

    # --- Outflow: predicted emigration rate ---
    log_delta_pred = fit_outflow.stan_variable('log_delta_pred')  # (draws, N_pred)
    n_draws = log_delta_pred.shape[0]
    pred_origins = data['outflow_pred_origins']
    pred_pop_2010 = data['outflow_pred_pop']

    # Fix NaN populations
    pred_pop_2010 = np.where(np.isnan(pred_pop_2010), 1e6, pred_pop_2010)

    # Map origin → (draw-level) predicted E_it = δ_it · P_it
    orig_to_E_draws = {}
    for idx, orig in enumerate(pred_origins):
        delta_draws = np.exp(log_delta_pred[:, idx])  # (n_draws,)
        E_draws = delta_draws * pred_pop_2010[idx]     # N_it = δ · P
        orig_to_E_draws[orig] = E_draws

    # --- Inflow: predicted shares ---
    pred_eta = fit_inflow.stan_variable('pred_eta')  # (draws, N_pred_flat)
    pred_pairs = data['pred_pairs_inflow']

    # Group predicted corridors by origin
    origin_groups = {}
    for i, pair in enumerate(pred_pairs):
        orig = pair.split('_')[0]
        if orig not in origin_groups:
            origin_groups[orig] = []
        origin_groups[orig].append((i, pair))

    # --- Combine: m̂_ijt = Ê_it × π̂_ijt ---
    predictions = []
    for orig, corridors in origin_groups.items():
        if orig not in orig_to_E_draws:
            continue

        E_draws = orig_to_E_draws[orig]   # (n_draws,)
        share_indices = [c[0] for c in corridors]
        pair_labels = [c[1] for c in corridors]

        pred_flows = np.zeros(len(share_indices))
        for d in range(n_draws):
            eta_vec = pred_eta[d, share_indices]
            pi_vec = softmax(eta_vec)  # compositional constraint
            pred_flows += E_draws[d] * pi_vec

        pred_flows /= n_draws  # posterior mean

        for j, pair in enumerate(pair_labels):
            dest = pair.split('_')[1]
            predictions.append({
                'origIso': orig, 'destIso': dest, 'pair': pair,
                'pred_flow': pred_flows[j],
            })

    pred_df = pd.DataFrame(predictions)

    # Merge with actual
    actual = data['df_test'][['origIso', 'destIso', 'migrantCount']].copy()
    actual['pair'] = actual['origIso'] + '_' + actual['destIso']
    eval_df = pred_df.merge(actual[['pair', 'migrantCount']], on='pair', how='inner')
    eval_df.rename(columns={'migrantCount': 'actual_flow'}, inplace=True)

    print(f"Predictions for {len(eval_df):,} corridors in 2010")

    return eval_df


# ============================================================================
# 6. EVALUATE
# ============================================================================

def evaluate(eval_df, output_dir):
    print(f"\n{'=' * 70}")
    print("EVALUATION — OUT-OF-SAMPLE (2010)")
    print(f"{'=' * 70}")

    actual = eval_df['actual_flow'].values
    pred = eval_df['pred_flow'].values

    mask = actual > 0
    a = actual[mask]
    p = pred[mask]

    log_a = np.log(a)
    log_p = np.log(np.maximum(p, 1e-10))

    mse_log = np.mean((log_a - log_p) ** 2)
    mae_log = np.mean(np.abs(log_a - log_p))
    corr = np.corrcoef(log_a, log_p)[0, 1]
    mape = np.mean(np.abs(a - p) / a) * 100
    medape = np.median(np.abs(a - p) / a) * 100

    print(f"\n  Corridors (actual>0): {mask.sum():,}")
    print(f"  MSE  (log):   {mse_log:.4f}")
    print(f"  MAE  (log):   {mae_log:.4f}")
    print(f"  Corr:         {corr:.4f}")
    print(f"  MAPE:         {mape:.1f}%")
    print(f"  MedAPE:       {medape:.1f}%")

    # --- Plots ---
    os.makedirs(output_dir, exist_ok=True)
    resid = log_a - log_p

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.scatter(log_a, log_p, alpha=0.3, s=10, color='steelblue')
    lims = [min(log_a.min(), log_p.min()) - 0.5, max(log_a.max(), log_p.max()) + 0.5]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel('Observed log(flow) — 2010')
    ax.set_ylabel('Predicted log(flow) — 2010')
    ax.set_title(f'Out-of-Sample (Paper Model)\nr = {corr:.3f}')

    ax = axes[0, 1]
    ax.hist(resid, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', lw=2)
    ax.set_xlabel('Residual (log obs − log pred)')
    ax.set_title(f'Residuals: mean={resid.mean():.2f}, SD={resid.std():.2f}')

    ax = axes[1, 0]
    top100 = eval_df[eval_df['actual_flow'] > 0].nlargest(100, 'actual_flow')
    ax.scatter(top100['actual_flow']/1e3, top100['pred_flow']/1e3, alpha=0.6, s=30, color='coral')
    mx = max(top100['actual_flow'].max(), top100['pred_flow'].max()) / 1e3
    ax.plot([0, mx], [0, mx], 'r--', lw=2)
    ax.set_xlabel('Actual flow (thousands)')
    ax.set_ylabel('Predicted flow (thousands)')
    ax.set_title('Top 100 Corridors (level scale)')

    ax = axes[1, 1]
    ape = np.clip(np.abs(a - p) / a * 100, 0, 500)
    ax.hist(ape, bins=60, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(medape, color='blue', lw=2, label=f'MedAPE = {medape:.0f}%')
    ax.set_xlabel('Abs. Percentage Error (%)')
    ax.set_title('Distribution of Errors (capped 500%)')
    ax.legend()

    plt.suptitle('Azose & Raftery (2019) — Paper Model — Out-of-Sample 2010',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'paper_model_oos.png', dpi=150)
    plt.close()
    print(f"\n  Plot: {output_dir / 'paper_model_oos.png'}")

    return {'mse_log': mse_log, 'mae_log': mae_log, 'corr': corr,
            'mape': mape, 'medape': medape}


# ============================================================================
# 7. PARAMETER DIAGNOSTICS
# ============================================================================

def diagnostics(fit_outflow, fit_inflow, data, output_dir):
    print(f"\n{'=' * 70}")
    print("PARAMETER DIAGNOSTICS")
    print(f"{'=' * 70}")

    # --- Outflow ---
    summary_out = fit_outflow.summary(sig_figs=4)
    mc = 'Mean' if 'Mean' in summary_out.columns else 'mean'
    sc = 'StdDev' if 'StdDev' in summary_out.columns else 'sd'
    rc = 'R_hat' if 'R_hat' in summary_out.columns else 'r_hat'

    print("\n  OUTFLOW MODEL:")
    for p in ['phi', 'nu', 'tau_0']:
        if p in summary_out.index:
            r = summary_out.loc[p]
            print(f"    {p:<12s}: {r[mc]:>8.4f} ± {r[sc]:.4f}  R-hat={r[rc]:.3f}")

    phi = fit_outflow.stan_variable('phi')
    sigma_draws = fit_outflow.stan_variable('sigma')  # (draws, N_orig)
    mu_draws = fit_outflow.stan_variable('mu')

    print(f"\n    φ = {phi.mean():.3f} [{np.percentile(phi, 2.5):.3f}, {np.percentile(phi, 97.5):.3f}]")
    print(f"    σ_i mean = {sigma_draws.mean():.3f}, range [{sigma_draws.mean(axis=0).min():.3f}, {sigma_draws.mean(axis=0).max():.3f}]")

    # --- Inflow ---
    kappa = fit_inflow.stan_variable('kappa')  # (draws, N_corridors)
    psi = fit_inflow.stan_variable('psi')

    print("\n  INFLOW MODEL:")
    print(f"    κ_ij mean = {kappa.mean():.3f}, SD across corridors = {kappa.mean(axis=0).std():.3f}")
    print(f"    ψ_ij mean = {psi.mean():.3f}, range [{psi.mean(axis=0).min():.3f}, {psi.mean(axis=0).max():.3f}]")

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Outflow: phi
    ax = axes[0, 0]
    ax.hist(phi, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(phi.mean(), color='red', lw=2)
    ax.set_title(f'φ (outflow AR) = {phi.mean():.3f}')
    ax.set_xlabel('φ')

    # Outflow: σ_i distribution
    ax = axes[0, 1]
    sigma_means = sigma_draws.mean(axis=0)
    ax.hist(sigma_means, bins=30, density=True, alpha=0.7, color='seagreen', edgecolor='white')
    ax.set_title(f'Per-origin σ_i (n={len(sigma_means)})')
    ax.set_xlabel('σ_i (posterior mean)')

    # Outflow: μ_i (top 15 origins)
    ax = axes[0, 2]
    mu_means = mu_draws.mean(axis=0)
    countries = data['countries']
    top_idx = np.argsort(mu_means)[-15:]
    ax.barh(range(15), mu_means[top_idx], color='coral', alpha=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels([countries[i] for i in top_idx])
    ax.set_xlabel('μ_i (long-run log emigration rate)')
    ax.set_title('Top 15 Emigrating Countries')

    # Inflow: κ_ij distribution
    ax = axes[1, 0]
    kappa_means = kappa.mean(axis=0)
    ax.hist(kappa_means, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_title(f'κ_ij corridor means (n={len(kappa_means):,})')
    ax.set_xlabel('κ_ij')

    # Inflow: ψ_ij distribution
    ax = axes[1, 1]
    psi_means = psi.mean(axis=0)
    ax.hist(psi_means, bins=60, density=True, alpha=0.7, color='purple', edgecolor='white')
    ax.set_title(f'ψ_ij corridor SDs (n={len(psi_means):,})')
    ax.set_xlabel('ψ_ij')

    # Inflow: top corridors by κ
    ax = axes[1, 2]
    corridors = data['corridors']
    top_corr = np.argsort(kappa_means)[-15:]
    ax.barh(range(15), kappa_means[top_corr], color='steelblue', alpha=0.7)
    ax.set_yticks(range(15))
    ax.set_yticklabels([corridors[i] for i in top_corr], fontsize=8)
    ax.set_xlabel('κ_ij')
    ax.set_title('Top 15 Corridors by Mean Share')

    plt.suptitle('Paper Model — Parameter Posteriors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'paper_model_params.png', dpi=150)
    plt.close()
    print(f"\n  Plot: {output_dir / 'paper_model_params.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  AZOSE & RAFTERY (2019) — EXACT PAPER EQUATIONS")
    print("=" * 70)

    os.makedirs(OUT, exist_ok=True)
    data = prepare_data()

    outflow_data = build_outflow_data(data)
    inflow_data = build_inflow_data(data)

    fit_out = fit_stan(OUTFLOW_STAN, outflow_data, "OUTFLOW (emigration rate)")
    fit_in = fit_stan(INFLOW_STAN, inflow_data, "INFLOW (shares, Multinomial)")

    diagnostics(fit_out, fit_in, data, OUT)

    eval_df = predict_2010(fit_out, fit_in, data)
    metrics = evaluate(eval_df, OUT)

    # --- Final summary ---
    phi = fit_out.stan_variable('phi').mean()
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — Paper Model")
    print(f"{'=' * 70}")
    print(f"  φ  (outflow AR):     {phi:.3f}")
    print(f"  Out-of-sample 2010:")
    print(f"    Corr (log):  {metrics['corr']:.4f}")
    print(f"    MedAPE:      {metrics['medape']:.1f}%")
    print(f"    MAPE:        {metrics['mape']:.1f}%")
    print(f"\n  Files: {OUT}")


if __name__ == "__main__":
    main()
