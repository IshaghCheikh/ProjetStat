"""
==============================================================================
Hierarchical Bayesian Model for Bilateral Migration Flows
Re-implementation of Azose & Raftery (2019) methodology using Stan/CmdStanPy

This script:
1. Loads the Azose & Raftery bilateral migration flow data
2. Prepares the data for a two-part hurdle log-normal HBM
3. Fits the model via MCMC (NUTS) in Stan
4. Produces diagnostics, posterior summaries, and visualizations
5. Compares with a simple OLS baseline
==============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
TOP_N_COUNTRIES = 50     # Use top N countries by total flow volume (for tractability)
N_CHAINS = 4             # Number of MCMC chains
N_WARMUP = 1000          # Warmup iterations per chain
N_SAMPLES = 1000         # Sampling iterations per chain
SEED = 42
DATA_PATH = Path(__file__).parent / "data" / "azoseRaftery2019flows.csv"
STAN_MODEL_PATH = Path(__file__).parent / "azose_raftery_hbm.stan"
OUTPUT_DIR = Path(__file__).parent / "graphs"

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare_data(data_path, top_n=TOP_N_COUNTRIES):
    """Load migration flows and prepare for the hurdle HBM."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(data_path)
    print(f"Raw data: {df.shape[0]:,} observations, {df['origIso'].nunique()} countries, "
          f"{df['year'].nunique()} periods")
    print(f"Zero flows: {(df['migrantCount'] == 0).sum():,} ({(df['migrantCount'] == 0).mean():.1%})")

    # --- Subset to top N countries by total flow volume ---
    total_flow_by_country = pd.concat([
        df.groupby('origIso')['migrantCount'].sum(),
        df.groupby('destIso')['migrantCount'].sum()
    ], axis=1).sum(axis=1).sort_values(ascending=False)

    top_countries = total_flow_by_country.head(top_n).index.tolist()
    df_sub = df[df['origIso'].isin(top_countries) & df['destIso'].isin(top_countries)].copy()
    df_sub = df_sub[df_sub['origIso'] != df_sub['destIso']].copy()  # remove self-flows

    print(f"\nSubset to top {top_n} countries: {df_sub.shape[0]:,} observations")
    print(f"Zero flows in subset: {(df_sub['migrantCount'] == 0).sum():,} "
          f"({(df_sub['migrantCount'] == 0).mean():.1%})")

    # --- Create integer indices ---
    orig_cats = pd.Categorical(df_sub['origIso'])
    dest_cats = pd.Categorical(df_sub['destIso'])
    # Ensure same category set for origin and destination
    all_countries = sorted(set(df_sub['origIso'].unique()) | set(df_sub['destIso'].unique()))
    orig_cats = pd.Categorical(df_sub['origIso'], categories=all_countries)
    dest_cats = pd.Categorical(df_sub['destIso'], categories=all_countries)

    df_sub['orig_idx'] = orig_cats.codes + 1  # Stan is 1-indexed
    df_sub['dest_idx'] = dest_cats.codes + 1

    years_sorted = sorted(df_sub['year'].unique())
    year_map = {y: i + 1 for i, y in enumerate(years_sorted)}
    df_sub['time_idx'] = df_sub['year'].map(year_map)

    df_sub['is_positive'] = (df_sub['migrantCount'] > 0).astype(int)

    # --- Positive flows ---
    df_pos = df_sub[df_sub['migrantCount'] > 0].copy()
    df_pos['log_flow'] = np.log(df_pos['migrantCount'].astype(float))

    # --- Compute lagged log-flow (AR(1) component) ---
    # Sort by origin, destination, year
    df_pos = df_pos.sort_values(['origIso', 'destIso', 'year']).reset_index(drop=True)
    df_pos['lag_log_flow'] = 0.0
    df_pos['has_lag'] = 0

    for (o, d), group in df_pos.groupby(['origIso', 'destIso']):
        idx = group.index
        for i in range(1, len(group)):
            curr_year = group.iloc[i]['year']
            prev_year = group.iloc[i - 1]['year']
            if curr_year - prev_year == 5:  # consecutive 5-year periods
                df_pos.loc[idx[i], 'lag_log_flow'] = group.iloc[i - 1]['log_flow']
                df_pos.loc[idx[i], 'has_lag'] = 1

    n_with_lag = df_pos['has_lag'].sum()
    print(f"Positive flows with AR(1) lag available: {n_with_lag:,} / {len(df_pos):,}")

    N_orig = len(all_countries)
    N_dest = len(all_countries)
    N_time = len(years_sorted)

    # --- Build Stan data dictionary ---
    stan_data = {
        'N_all': len(df_sub),
        'N_pos': len(df_pos),
        'N_orig': N_orig,
        'N_dest': N_dest,
        'N_time': N_time,
        # Full data
        'orig_all': df_sub['orig_idx'].values.astype(int),
        'dest_all': df_sub['dest_idx'].values.astype(int),
        'time_all': df_sub['time_idx'].values.astype(int),
        'is_positive': df_sub['is_positive'].values.astype(int),
        # Positive flows
        'orig_pos': df_pos['orig_idx'].values.astype(int),
        'dest_pos': df_pos['dest_idx'].values.astype(int),
        'time_pos': df_pos['time_idx'].values.astype(int),
        'log_flow': df_pos['log_flow'].values,
        'lag_log_flow': df_pos['lag_log_flow'].values,
        'has_lag': df_pos['has_lag'].values.astype(int),
    }

    meta = {
        'countries': all_countries,
        'years': years_sorted,
        'df_sub': df_sub,
        'df_pos': df_pos,
        'year_map': year_map,
    }

    print(f"\nStan data dimensions:")
    print(f"  N_all  = {stan_data['N_all']:,}")
    print(f"  N_pos  = {stan_data['N_pos']:,}")
    print(f"  N_orig = {stan_data['N_orig']}")
    print(f"  N_dest = {stan_data['N_dest']}")
    print(f"  N_time = {stan_data['N_time']}")

    return stan_data, meta


# ============================================================================
# 2. FIT THE MODEL
# ============================================================================

def fit_model(stan_data, stan_model_path, n_chains=N_CHAINS,
              n_warmup=N_WARMUP, n_samples=N_SAMPLES, seed=SEED):
    """Compile and fit the Stan HBM."""
    from cmdstanpy import CmdStanModel

    print("\n" + "=" * 70)
    print("COMPILING STAN MODEL")
    print("=" * 70)

    model = CmdStanModel(stan_file=str(stan_model_path))
    print("Compilation successful.")

    print("\n" + "=" * 70)
    print(f"FITTING MODEL ({n_chains} chains × {n_warmup}+{n_samples} iterations)")
    print("=" * 70)

    t0 = time.time()
    fit = model.sample(
        data=stan_data,
        chains=n_chains,
        iter_warmup=n_warmup,
        iter_sampling=n_samples,
        seed=seed,
        show_console=False,
        adapt_delta=0.9,
        max_treedepth=12,
        inits=0,            # Start from zero to avoid inf during init
    )
    elapsed = time.time() - t0
    print(f"\nSampling completed in {elapsed:.1f} seconds ({elapsed / 60:.1f} min)")

    return fit


# ============================================================================
# 3. DIAGNOSTICS
# ============================================================================

def run_diagnostics(fit, meta, output_dir):
    """Run MCMC diagnostics and produce summary."""
    import arviz as az

    print("\n" + "=" * 70)
    print("MCMC DIAGNOSTICS")
    print("=" * 70)

    # --- CmdStanPy diagnostics ---
    print(fit.diagnose())

    # --- Key parameter summaries ---
    key_params = [
        'gamma_0', 'sigma_gamma_orig', 'sigma_gamma_dest',
        'alpha', 'sigma_alpha', 'sigma_beta', 'sigma', 'rho',
    ]
    # Add time effects
    key_params += [f'delta_time[{i}]' for i in range(1, len(meta['years']) + 1)]

    summary = fit.summary(sig_figs=4)
    print("\n--- Summary columns:", summary.columns.tolist())
    print("\n--- Key Parameter Estimates ---")

    # Detect column names (varies across cmdstanpy versions)
    mean_col = 'Mean' if 'Mean' in summary.columns else 'mean'
    sd_col = 'StdDev' if 'StdDev' in summary.columns else 'sd'
    rhat_col = 'R_hat' if 'R_hat' in summary.columns else 'r_hat'
    ess_col = next((c for c in summary.columns if 'ess' in c.lower() or 'n_eff' in c.lower()), None)

    # Filter to key params
    for p in key_params:
        if p in summary.index:
            row = summary.loc[p]
            line = f"  {p:25s}: mean={row[mean_col]:8.4f}  sd={row[sd_col]:7.4f}"
            if rhat_col in row.index:
                line += f"  R-hat={row[rhat_col]:6.3f}"
            if ess_col and ess_col in row.index:
                line += f"  ESS={row[ess_col]:8.0f}"
            print(line)

    # --- Check convergence ---
    if rhat_col in summary.columns:
        rhat_vals = summary[rhat_col].dropna()
        n_bad_rhat = (rhat_vals > 1.05).sum()
        print(f"\nParameters with R-hat > 1.05: {n_bad_rhat} / {len(rhat_vals)}")
    else:
        print("\nR-hat column not found in summary")

    # --- Convert to ArviZ InferenceData for plots ---
    idata = az.from_cmdstanpy(fit)

    # --- Trace plots for key parameters ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    trace_params = ['gamma_0', 'alpha', 'sigma', 'rho',
                    'sigma_gamma_orig', 'sigma_gamma_dest',
                    'sigma_alpha', 'sigma_beta']
    for ax, pname in zip(axes.flat, trace_params):
        if pname in idata.posterior:
            vals = idata.posterior[pname].values  # shape: (chains, draws)
            for c in range(vals.shape[0]):
                ax.plot(vals[c], alpha=0.6, lw=0.5)
            ax.set_title(pname, fontsize=10)
            ax.set_ylabel('Value')
    plt.suptitle('Trace Plots - Key Parameters', fontsize=14)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir / 'hbm_trace_plots.png', dpi=150)
    plt.close()
    print(f"\nTrace plots saved to {output_dir / 'hbm_trace_plots.png'}")

    # --- Posterior density plots ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, pname in zip(axes.flat, trace_params):
        if pname in idata.posterior:
            vals = idata.posterior[pname].values.flatten()
            ax.hist(vals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
            ax.axvline(np.mean(vals), color='red', lw=2, label=f'mean={np.mean(vals):.3f}')
            ax.set_title(pname, fontsize=10)
            ax.legend(fontsize=8)
    plt.suptitle('Posterior Distributions - Key Parameters', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hbm_posterior_densities.png', dpi=150)
    plt.close()
    print(f"Posterior density plots saved to {output_dir / 'hbm_posterior_densities.png'}")

    return summary, idata


# ============================================================================
# 4. POSTERIOR PREDICTIVE CHECKS & EVALUATION
# ============================================================================

def posterior_predictive_checks(fit, stan_data, meta, output_dir):
    """Posterior predictive checks and model evaluation."""

    print("\n" + "=" * 70)
    print("POSTERIOR PREDICTIVE CHECKS")
    print("=" * 70)

    df_pos = meta['df_pos']

    # --- Extract posterior predictive draws ---
    y_rep = fit.stan_variable('y_rep')  # shape: (n_draws, N_pos)
    log_flow_obs = stan_data['log_flow']

    y_rep_mean = y_rep.mean(axis=0)
    y_rep_sd = y_rep.std(axis=0)

    # --- Posterior predictive: observed vs predicted ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Observed vs Predicted (log scale)
    ax = axes[0]
    ax.scatter(log_flow_obs, y_rep_mean, alpha=0.1, s=5, color='steelblue')
    lims = [min(log_flow_obs.min(), y_rep_mean.min()),
            max(log_flow_obs.max(), y_rep_mean.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='y = x')
    ax.set_xlabel('Observed log(flow)')
    ax.set_ylabel('Predicted log(flow) [posterior mean]')
    ax.set_title('Observed vs Predicted (log scale)')
    ax.legend()
    corr = np.corrcoef(log_flow_obs, y_rep_mean)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 2: Residual distribution
    ax = axes[1]
    resid = log_flow_obs - y_rep_mean
    ax.hist(resid, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', lw=2)
    ax.set_xlabel('Residual (observed - predicted)')
    ax.set_ylabel('Density')
    ax.set_title(f'Residual Distribution\nMean={resid.mean():.3f}, SD={resid.std():.3f}')

    # Panel 3: Posterior predictive distribution vs observed
    ax = axes[2]
    ax.hist(log_flow_obs, bins=80, density=True, alpha=0.5, color='blue', label='Observed')
    # Overlay a few posterior predictive draws
    for i in range(min(50, y_rep.shape[0])):
        ax.hist(y_rep[i], bins=80, density=True, alpha=0.02, color='orange')
    ax.hist(y_rep[0], bins=80, density=True, alpha=0.3, color='orange', label='Posterior predictive')
    ax.set_xlabel('log(flow)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Predictive Check')
    ax.legend()

    plt.suptitle('Posterior Predictive Checks - HBM', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hbm_posterior_predictive.png', dpi=150)
    plt.close()
    print(f"Posterior predictive plots saved to {output_dir / 'hbm_posterior_predictive.png'}")

    # --- Compute metrics ---
    # On log scale
    mse_log = np.mean(resid ** 2)
    mae_log = np.mean(np.abs(resid))
    # On original scale
    pred_flow = np.exp(y_rep_mean)
    obs_flow = np.exp(log_flow_obs)
    mape = np.mean(np.abs(obs_flow - pred_flow) / np.maximum(obs_flow, 1)) * 100
    mse_level = np.mean((obs_flow - pred_flow) ** 2)

    print(f"\n--- Model Performance (positive flows only) ---")
    print(f"  MSE  (log scale):    {mse_log:.4f}")
    print(f"  MAE  (log scale):    {mae_log:.4f}")
    print(f"  Corr (log scale):    {corr:.4f}")
    print(f"  MAPE (level scale):  {mape:.1f}%")
    print(f"  RMSE (level scale):  {np.sqrt(mse_level):.0f}")

    return {
        'mse_log': mse_log, 'mae_log': mae_log, 'corr': corr,
        'mape': mape, 'rmse_level': np.sqrt(mse_level)
    }


# ============================================================================
# 5. RANDOM EFFECTS ANALYSIS
# ============================================================================

def analyze_random_effects(fit, meta, output_dir):
    """Visualize origin and destination random effects."""

    print("\n" + "=" * 70)
    print("RANDOM EFFECTS ANALYSIS")
    print("=" * 70)

    countries = meta['countries']

    # --- Origin effects (push factors) ---
    alpha_orig = fit.stan_variable('alpha_orig')  # (n_draws, N_orig)
    alpha_orig_mean = alpha_orig.mean(axis=0)
    alpha_orig_sd = alpha_orig.std(axis=0)

    # --- Destination effects (pull factors) ---
    beta_dest = fit.stan_variable('beta_dest')
    beta_dest_mean = beta_dest.mean(axis=0)
    beta_dest_sd = beta_dest.std(axis=0)

    # --- Plot top/bottom random effects ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top 15 positive origin effects (biggest "push")
    idx_orig_top = np.argsort(alpha_orig_mean)[-15:]
    ax = axes[0, 0]
    ax.barh(range(15), alpha_orig_mean[idx_orig_top], xerr=alpha_orig_sd[idx_orig_top],
            color='coral', alpha=0.7, capsize=3)
    ax.set_yticks(range(15))
    ax.set_yticklabels([countries[i] for i in idx_orig_top])
    ax.set_xlabel('Effect size')
    ax.set_title('Top 15 Origin Effects (Push Factors)')

    # Bottom 15 origin effects
    idx_orig_bot = np.argsort(alpha_orig_mean)[:15]
    ax = axes[0, 1]
    ax.barh(range(15), alpha_orig_mean[idx_orig_bot], xerr=alpha_orig_sd[idx_orig_bot],
            color='steelblue', alpha=0.7, capsize=3)
    ax.set_yticks(range(15))
    ax.set_yticklabels([countries[i] for i in idx_orig_bot])
    ax.set_xlabel('Effect size')
    ax.set_title('Bottom 15 Origin Effects')

    # Top 15 destination effects (biggest "pull")
    idx_dest_top = np.argsort(beta_dest_mean)[-15:]
    ax = axes[1, 0]
    ax.barh(range(15), beta_dest_mean[idx_dest_top], xerr=beta_dest_sd[idx_dest_top],
            color='coral', alpha=0.7, capsize=3)
    ax.set_yticks(range(15))
    ax.set_yticklabels([countries[i] for i in idx_dest_top])
    ax.set_xlabel('Effect size')
    ax.set_title('Top 15 Destination Effects (Pull Factors)')

    # Bottom 15 destination effects
    idx_dest_bot = np.argsort(beta_dest_mean)[:15]
    ax = axes[1, 1]
    ax.barh(range(15), beta_dest_mean[idx_dest_bot], xerr=beta_dest_sd[idx_dest_bot],
            color='steelblue', alpha=0.7, capsize=3)
    ax.set_yticks(range(15))
    ax.set_yticklabels([countries[i] for i in idx_dest_bot])
    ax.set_xlabel('Effect size')
    ax.set_title('Bottom 15 Destination Effects')

    plt.suptitle('Hierarchical Random Effects - Origin & Destination', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hbm_random_effects.png', dpi=150)
    plt.close()
    print(f"Random effects plot saved to {output_dir / 'hbm_random_effects.png'}")

    # --- Time effects ---
    delta_time = fit.stan_variable('delta_time')  # (n_draws, N_time)
    delta_mean = delta_time.mean(axis=0)
    delta_sd = delta_time.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(meta['years'], delta_mean, yerr=1.96 * delta_sd,
                fmt='o-', color='steelblue', capsize=5, lw=2, markersize=8)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel('Period')
    ax.set_ylabel('Time Effect (δ_t)')
    ax.set_title('Temporal Effects on log(Migration Flow)')
    ax.set_xticks(meta['years'])
    plt.tight_layout()
    plt.savefig(output_dir / 'hbm_time_effects.png', dpi=150)
    plt.close()
    print(f"Time effects plot saved to {output_dir / 'hbm_time_effects.png'}")


# ============================================================================
# 6. OLS BASELINE COMPARISON
# ============================================================================

def ols_baseline(meta):
    """Fit a simple OLS gravity-style model for comparison."""
    from scipy import stats as sp_stats

    print("\n" + "=" * 70)
    print("OLS BASELINE COMPARISON")
    print("=" * 70)

    df_pos = meta['df_pos'].copy()

    # Simple model: log(flow) ~ origin_FE + dest_FE + time_FE
    # Using demeaned approach for speed
    y = df_pos['log_flow'].values
    y_mean = y.mean()

    # Origin FE
    orig_means = df_pos.groupby('orig_idx')['log_flow'].mean()
    pred_orig = df_pos['orig_idx'].map(orig_means).values

    # Dest FE
    dest_means = df_pos.groupby('dest_idx')['log_flow'].mean()
    pred_dest = df_pos['dest_idx'].map(dest_means).values

    # Time FE
    time_means = df_pos.groupby('time_idx')['log_flow'].mean()
    pred_time = df_pos['time_idx'].map(time_means).values

    # Simple additive prediction (origin + dest + time - 2*grand_mean)
    pred = pred_orig + pred_dest + pred_time - 2 * y_mean

    resid = y - pred
    mse = np.mean(resid ** 2)
    mae = np.mean(np.abs(resid))
    corr = np.corrcoef(y, pred)[0, 1]

    pred_flow = np.exp(pred)
    obs_flow = np.exp(y)
    mape = np.mean(np.abs(obs_flow - pred_flow) / np.maximum(obs_flow, 1)) * 100

    print(f"  MSE  (log scale):    {mse:.4f}")
    print(f"  MAE  (log scale):    {mae:.4f}")
    print(f"  Corr (log scale):    {corr:.4f}")
    print(f"  MAPE (level scale):  {mape:.1f}%")

    return {'mse_log': mse, 'mae_log': mae, 'corr': corr, 'mape': mape}


# ============================================================================
# 7. EMIGRATION SHARE DECOMPOSITION (η model)
# ============================================================================

def emigration_share_analysis(meta, output_dir):
    """Analyze emigration shares following Azose & Raftery decomposition."""

    print("\n" + "=" * 70)
    print("EMIGRATION SHARE DECOMPOSITION (η_lt model)")
    print("=" * 70)

    df_sub = meta['df_sub'].copy()

    # Compute total emigration per origin-year
    total_emi = df_sub.groupby(['origIso', 'year'])['migrantCount'].sum().reset_index()
    total_emi.columns = ['origIso', 'year', 'total_emigration']

    df_sub = df_sub.merge(total_emi, on=['origIso', 'year'])

    # Emigration shares (only where total > 0)
    df_sub = df_sub[df_sub['total_emigration'] > 0].copy()
    df_sub['pi_ijt'] = df_sub['migrantCount'] / df_sub['total_emigration']

    # Log shares (only for positive flows)
    df_shares = df_sub[df_sub['migrantCount'] > 0].copy()
    df_shares['eta_lt'] = np.log(df_shares['pi_ijt'])

    # Corridor-specific means
    df_shares['pair'] = df_shares['origIso'] + '_' + df_shares['destIso']
    corridor_means = df_shares.groupby('pair')['eta_lt'].mean()
    df_shares['k_ij'] = df_shares['pair'].map(corridor_means)

    # R² of corridor fixed effects model
    ss_total = np.sum((df_shares['eta_lt'] - df_shares['eta_lt'].mean()) ** 2)
    ss_resid = np.sum((df_shares['eta_lt'] - df_shares['k_ij']) ** 2)
    r2 = 1 - ss_resid / ss_total

    print(f"  Number of corridors: {df_shares['pair'].nunique():,}")
    print(f"  R² of corridor fixed-effects model: {r2:.4f}")
    print(f"  This means {r2:.1%} of variance in log-shares is explained by corridor identity")

    # --- Plot distribution of corridor means ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(corridor_means.values, bins=80, density=True, alpha=0.7,
            color='steelblue', edgecolor='white')
    ax.set_xlabel('k_ij (corridor mean log-share)')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Corridor Effects k_ij\n(n={len(corridor_means):,})')

    ax = axes[1]
    # Residual variance by corridor
    resid_var = df_shares.groupby('pair').apply(
        lambda g: np.var(g['eta_lt'] - g['k_ij']) if len(g) > 1 else np.nan
    ).dropna()
    ax.hist(resid_var.values, bins=60, density=True, alpha=0.7,
            color='coral', edgecolor='white')
    ax.set_xlabel('Residual variance within corridor')
    ax.set_ylabel('Density')
    ax.set_title('Within-Corridor Variance of η_lt')

    plt.suptitle('Emigration Share Decomposition (Azose & Raftery)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hbm_emigration_shares.png', dpi=150)
    plt.close()
    print(f"Emigration share plots saved to {output_dir / 'hbm_emigration_shares.png'}")

    return r2


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  AZOSE & RAFTERY (2019) - HBM RE-IMPLEMENTATION WITH STAN")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    stan_data, meta = load_and_prepare_data(DATA_PATH, top_n=TOP_N_COUNTRIES)

    # 2. OLS baseline
    ols_metrics = ols_baseline(meta)

    # 3. Emigration share decomposition
    r2_shares = emigration_share_analysis(meta, OUTPUT_DIR)

    # 4. Fit HBM
    fit = fit_model(stan_data, STAN_MODEL_PATH)

    # 5. Diagnostics
    summary, idata = run_diagnostics(fit, meta, OUTPUT_DIR)

    # 6. Posterior predictive checks
    hbm_metrics = posterior_predictive_checks(fit, stan_data, meta, OUTPUT_DIR)

    # 7. Random effects analysis
    analyze_random_effects(fit, meta, OUTPUT_DIR)

    # 8. Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: HBM vs OLS")
    print("=" * 70)
    print(f"{'Metric':<25s} {'OLS':>12s} {'HBM':>12s}")
    print("-" * 50)
    print(f"{'MSE (log scale)':<25s} {ols_metrics['mse_log']:>12.4f} {hbm_metrics['mse_log']:>12.4f}")
    print(f"{'MAE (log scale)':<25s} {ols_metrics['mae_log']:>12.4f} {hbm_metrics['mae_log']:>12.4f}")
    print(f"{'Correlation':<25s} {ols_metrics['corr']:>12.4f} {hbm_metrics['corr']:>12.4f}")
    print(f"{'MAPE (%)':<25s} {ols_metrics['mape']:>11.1f}% {hbm_metrics['mape']:>11.1f}%")

    # --- Rho (AR(1) persistence) ---
    rho_samples = fit.stan_variable('rho')
    print(f"\n--- AR(1) Persistence Parameter ---")
    print(f"  ρ = {rho_samples.mean():.3f} (95% CI: [{np.percentile(rho_samples, 2.5):.3f}, "
          f"{np.percentile(rho_samples, 97.5):.3f}])")
    print(f"  Interpretation: {'Strong' if rho_samples.mean() > 0.5 else 'Moderate'} "
          f"temporal persistence in migration corridors")

    print("\n" + "=" * 70)
    print("LIMITATIONS OF THIS RE-IMPLEMENTATION")
    print("=" * 70)
    print("""
    1. DATA SUBSET: We use the top {top_n} countries (by total flow) instead of all
       200, reducing from ~199K to ~{n_all:,} observations. This is necessary for
       Stan's NUTS sampler to converge in reasonable time, but excludes small
       island nations and territories with sparse flows.

    2. NO COVARIATES: The paper's full model incorporates gravity-type covariates
       (distance, contiguity, common language, colonial ties, GDP, population).
       Our HBM uses only origin/destination random effects and AR(1) persistence.
       This means the model cannot explain *why* flows differ, only *that* they
       differ across corridors.

    3. SIMPLIFIED ZERO MODEL: The paper's demographic accounting approach handles
       zeros through the estimation procedure itself. Our hurdle model treats
       zero/non-zero as a separate logistic component, which may not capture
       the structural reasons for zero flows (non-existence of corridor vs.
       very small unobserved flows).

    4. NO DEMOGRAPHIC ACCOUNTING: Azose & Raftery (2019) estimate flows from
       stock data using a demographic accounting identity. We take their
       estimated flows as given data, so we cannot assess uncertainty from
       the flow estimation step itself.

    5. TEMPORAL STRUCTURE: The paper uses a more sophisticated temporal model
       with age-specific components and cohort effects. Our model uses simple
       period dummies + AR(1), which captures persistence but not demographic
       dynamics like aging cohorts.

    6. NO UNCERTAINTY PROPAGATION: The original paper propagates uncertainty
       from the stock-to-flow estimation into the hierarchical model. Our
       approach treats the point estimates as exact, underestimating total
       uncertainty.

    7. SCALABILITY: Stan's NUTS sampler scales poorly to the full 200×200×5
       matrix. The original paper likely used custom MCMC (Gibbs sampling with
       conjugate updates) or variational methods for the full dataset.

    8. NO RETURN/TRANSIT MIGRATION: The paper distinguishes emigration, return
       migration, and transit migration. Our model treats all flows identically.
    """.format(top_n=TOP_N_COUNTRIES, n_all=stan_data['N_all']))

    print("Done! All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
