"""
Germany Migration Forecast Plots — Azose & Raftery (2019)
Uses saved posteriors (no refitting).
Panels: A (Net), C (Inflow), D (Outflow),
        F (TUR→DEU), G (POL→DEU), H (USA→DEU)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJ = Path(__file__).resolve().parent
TOP_N = 50
HIST_YEARS = [1990, 1995, 2000, 2005, 2010]
FORECAST_YEARS = [2015, 2020, 2025, 2030, 2035, 2040, 2045]

np.random.seed(42)

# ── 1. Rebuild country & corridor lists (same logic as fit_paper_model.py) ───

df_raw = pd.read_csv(PROJ / 'data' / 'azoseRaftery2019flows.csv')
vol = pd.concat([
    df_raw.groupby('origIso')['migrantCount'].sum(),
    df_raw.groupby('destIso')['migrantCount'].sum()
], axis=1).sum(axis=1).sort_values(ascending=False)

top = vol.head(TOP_N).index.tolist()
df = df_raw[
    (df_raw['origIso'].isin(top)) & (df_raw['destIso'].isin(top))
    & (df_raw['origIso'] != df_raw['destIso'])
].copy()

countries = sorted(set(df['origIso'].unique()) | set(df['destIso'].unique()))
c2i = {c: i for i, c in enumerate(countries)}

df_train = df[df['year'] < 2010].copy()
pos_train = df_train[df_train['migrantCount'] > 0].copy()
pos_train = pos_train.copy()
pos_train['pair'] = pos_train['origIso'] + '_' + pos_train['destIso']
corridors = sorted(pos_train['pair'].unique())
corr2i = {c: i for i, c in enumerate(corridors)}

print(f"Countries: {len(countries)}, Corridors: {len(corridors)}")
print(f"DEU index: {c2i.get('DEU')}")

# ── 2. Load posteriors ────────────────────────────────────────────────────────

out = np.load(PROJ / 'posteriors' / 'outflow_posteriors.npz')
phi, mu, sigma = out['phi'], out['mu'], out['sigma']
N_DRAWS = len(phi)

inp = np.load(PROJ / 'posteriors' / 'inflow_posteriors.npz')
kappa, psi = inp['kappa'], inp['psi']

print(f"Posterior draws: {N_DRAWS}")

# ── 3. Population data ───────────────────────────────────────────────────────

grav = pd.read_csv(PROJ / 'data_final' / 'FINAL_GRAVITY_TRAINING_MATRIX.csv')
pop_df = grav[['iso3_o', 'year', 'pop_o']].drop_duplicates(['iso3_o', 'year'])
pop_df.columns = ['iso', 'year', 'pop']


def get_pop(iso, year):
    """Get or extrapolate population."""
    row = pop_df.loc[(pop_df['iso'] == iso) & (pop_df['year'] == year), 'pop']
    if len(row) > 0:
        return row.values[0]
    avail = pop_df.loc[pop_df['iso'] == iso].sort_values('year')
    if len(avail) >= 2:
        p1, p2 = avail.iloc[-2]['pop'], avail.iloc[-1]['pop']
        y1, y2 = avail.iloc[-2]['year'], avail.iloc[-1]['year']
        g = (p2 / max(p1, 1)) ** (1.0 / max(y2 - y1, 1))
        return p2 * g ** (year - y2)
    elif len(avail) == 1:
        return avail.iloc[0]['pop']
    return 1e6


# Pre-compute populations
pop_cache = {}
for c in countries:
    for y in HIST_YEARS + FORECAST_YEARS:
        pop_cache[(c, y)] = get_pop(c, y)

# ── 4. Historical flows ──────────────────────────────────────────────────────

DEU_IDX = c2i['DEU']

hist_outflow = {
    y: df.loc[(df['origIso'] == 'DEU') & (df['year'] == y), 'migrantCount'].sum()
    for y in HIST_YEARS
}
hist_inflow = {
    y: df.loc[(df['destIso'] == 'DEU') & (df['year'] == y), 'migrantCount'].sum()
    for y in HIST_YEARS
}
hist_net = {y: hist_inflow[y] - hist_outflow[y] for y in HIST_YEARS}

TRACKED = {'TUR_DEU': 'TUR', 'POL_DEU': 'POL', 'USA_DEU': 'USA'}
hist_corr = {}
for pair, orig in TRACKED.items():
    hist_corr[pair] = {
        y: df.loc[
            (df['origIso'] == orig) & (df['destIso'] == 'DEU') & (df['year'] == y),
            'migrantCount',
        ].sum()
        for y in HIST_YEARS
    }

print("Historical DEU outflow:", hist_outflow)
print("Historical DEU inflow:", hist_inflow)

# ── 5. Last observed log emigration rate (2010) ──────────────────────────────

log_delta_2010 = np.zeros(len(countries))
for i, c in enumerate(countries):
    outflow = df.loc[
        (df['origIso'] == c) & (df['year'] == 2010), 'migrantCount'
    ].sum()
    p = pop_cache[(c, 2010)]
    log_delta_2010[i] = np.log(max(outflow / p, 1e-12))

# ── 6. Origin → corridor mapping ─────────────────────────────────────────────

corridors_from = {}  # orig_iso -> [(local_idx, global_corr_idx, dest_iso)]
for j, c in enumerate(corridors):
    o, d = c.split('_')
    if o not in corridors_from:
        corridors_from[o] = []
    corridors_from[o].append((len(corridors_from[o]), j, d))

# ── 7. Forecast ──────────────────────────────────────────────────────────────

n_fc = len(FORECAST_YEARS)
log_delta_cur = np.tile(log_delta_2010, (N_DRAWS, 1))  # (4000, N)

fc_outflow = np.zeros((N_DRAWS, n_fc))
fc_inflow = np.zeros((N_DRAWS, n_fc))
fc_corr = {k: np.zeros((N_DRAWS, n_fc)) for k in TRACKED}

for t, yr in enumerate(FORECAST_YEARS):
    print(f"  Forecasting {yr} ...")

    # AR(1) step for all countries
    eps = np.random.randn(N_DRAWS, len(countries))
    log_delta_new = (
        (1 - phi[:, None]) * mu + phi[:, None] * log_delta_cur + sigma * eps
    )

    # DEU total outflow
    fc_outflow[:, t] = np.exp(log_delta_new[:, DEU_IDX]) * pop_cache[('DEU', yr)]

    # Inflow to DEU: sum over all origins with a corridor to DEU
    inflow_draws = np.zeros(N_DRAWS)
    for orig_iso, corr_list in corridors_from.items():
        # Does this origin have a corridor to DEU?
        deu_local_idx = None
        for local_i, global_i, dest in corr_list:
            if dest == 'DEU':
                deu_local_idx = local_i
                break
        if deu_local_idx is None:
            continue

        orig_idx = c2i[orig_iso]
        pop_orig = pop_cache[(orig_iso, yr)]
        total_emig = np.exp(log_delta_new[:, orig_idx]) * pop_orig

        # Draw shares via softmax over all corridors from this origin
        global_indices = [gi for _, gi, _ in corr_list]
        eta = (
            np.random.randn(N_DRAWS, len(corr_list)) * psi[:, global_indices]
            + kappa[:, global_indices]
        )
        eta_max = eta.max(axis=1, keepdims=True)
        exp_eta = np.exp(eta - eta_max)
        shares = exp_eta / exp_eta.sum(axis=1, keepdims=True)

        flow_to_deu = total_emig * shares[:, deu_local_idx]
        inflow_draws += flow_to_deu

        # Track specific corridors
        pair_key = f'{orig_iso}_DEU'
        if pair_key in fc_corr:
            fc_corr[pair_key][:, t] = flow_to_deu

    fc_inflow[:, t] = inflow_draws
    log_delta_cur = log_delta_new

fc_net = fc_inflow - fc_outflow

# ── 8. Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


def plot_panel(ax, hist, fc, title):
    years_h = sorted(hist.keys())
    vals_h = [hist[y] for y in years_h]
    ax.plot(years_h, vals_h, 'k-o', lw=2, ms=4, label='Observed')

    med = np.median(fc, axis=0)
    q025, q10, q90, q975 = [
        np.percentile(fc, p, axis=0) for p in [2.5, 10, 90, 97.5]
    ]

    ax.plot(FORECAST_YEARS, med, 'b-', lw=2, label='Median')
    ax.fill_between(FORECAST_YEARS, q10, q90, alpha=0.3, color='blue', label='80% PI')
    ax.fill_between(
        FORECAST_YEARS, q025, q975, alpha=0.15, color='blue', label='95% PI'
    )
    # Connect last historical to first forecast
    ax.plot(
        [years_h[-1], FORECAST_YEARS[0]], [vals_h[-1], med[0]], 'b--', alpha=0.5
    )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Persons')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')


plot_panel(axes[0, 0], hist_net, fc_net, 'A. Net Migration — Germany')
plot_panel(axes[0, 1], hist_inflow, fc_inflow, 'C. Total Immigration — Germany')
plot_panel(axes[0, 2], hist_outflow, fc_outflow, 'D. Total Emigration — Germany')
plot_panel(
    axes[1, 0], hist_corr['TUR_DEU'], fc_corr['TUR_DEU'],
    'F. Immigration from Turkey'
)
plot_panel(
    axes[1, 1], hist_corr['POL_DEU'], fc_corr['POL_DEU'],
    'G. Immigration from Poland'
)
plot_panel(
    axes[1, 2], hist_corr['USA_DEU'], fc_corr['USA_DEU'],
    'H. Immigration from USA'
)

fig.suptitle(
    'Migration Forecasts for Germany — Azose & Raftery (2019) Model',
    fontsize=15, fontweight='bold', y=1.02,
)
fig.tight_layout()
fig.savefig(PROJ / 'graphs' / 'germany_forecasts.png', dpi=200, bbox_inches='tight')
print("\nSaved → graphs/germany_forecasts.png")
plt.close()
