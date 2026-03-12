"""
Germany 2010 Out-of-Sample Validation — Azose & Raftery (2019)
Training: 1990–2005.  Predicted vs Observed: 2010.
Same 6-panel layout as forecast plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJ = Path(__file__).resolve().parent
TOP_N = 50
TRAIN_YEARS = [1990, 1995, 2000, 2005]
PRED_YEAR = 2010

np.random.seed(42)

# ── 1. Rebuild country & corridor lists ──────────────────────────────────────

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

DEU_IDX = c2i['DEU']
print(f"Countries: {len(countries)}, Corridors: {len(corridors)}, DEU idx: {DEU_IDX}")

# ── 2. Load posteriors ───────────────────────────────────────────────────────

out = np.load(PROJ / 'posteriors' / 'outflow_posteriors.npz')
phi, mu, sigma = out['phi'], out['mu'], out['sigma']
N_DRAWS = len(phi)

inp = np.load(PROJ / 'posteriors' / 'inflow_posteriors.npz')
kappa, psi = inp['kappa'], inp['psi']

print(f"Posterior draws: {N_DRAWS}")

# ── 3. Population data ──────────────────────────────────────────────────────

grav = pd.read_csv(PROJ / 'data_final' / 'FINAL_GRAVITY_TRAINING_MATRIX.csv')
pop_df = grav[['iso3_o', 'year', 'pop_o']].drop_duplicates(['iso3_o', 'year'])
pop_df.columns = ['iso', 'year', 'pop']


def get_pop(iso, year):
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


pop_cache = {}
for c in countries:
    for y in TRAIN_YEARS + [PRED_YEAR]:
        pop_cache[(c, y)] = get_pop(c, y)

# ── 4. Historical & observed 2010 flows ──────────────────────────────────────

TRACKED = {'TUR_DEU': 'TUR', 'POL_DEU': 'POL', 'USA_DEU': 'USA'}
ALL_YEARS = TRAIN_YEARS + [PRED_YEAR]

hist_outflow = {
    y: df.loc[(df['origIso'] == 'DEU') & (df['year'] == y), 'migrantCount'].sum()
    for y in ALL_YEARS
}
hist_inflow = {
    y: df.loc[(df['destIso'] == 'DEU') & (df['year'] == y), 'migrantCount'].sum()
    for y in ALL_YEARS
}
hist_net = {y: hist_inflow[y] - hist_outflow[y] for y in ALL_YEARS}

hist_corr = {}
for pair, orig in TRACKED.items():
    hist_corr[pair] = {
        y: df.loc[
            (df['origIso'] == orig) & (df['destIso'] == 'DEU') & (df['year'] == y),
            'migrantCount',
        ].sum()
        for y in ALL_YEARS
    }

print(f"Observed 2010 — Outflow: {hist_outflow[2010]:,.0f}, "
      f"Inflow: {hist_inflow[2010]:,.0f}, Net: {hist_net[2010]:,.0f}")

# ── 5. Log emigration rate at 2005 (last training period) ────────────────────

log_delta_2005 = np.zeros(len(countries))
for i, c in enumerate(countries):
    outflow = df.loc[
        (df['origIso'] == c) & (df['year'] == 2005), 'migrantCount'
    ].sum()
    p = pop_cache[(c, 2005)]
    log_delta_2005[i] = np.log(max(outflow / p, 1e-12))

# ── 6. Origin → corridor mapping ────────────────────────────────────────────

corridors_from = {}
for j, c in enumerate(corridors):
    o, d = c.split('_')
    if o not in corridors_from:
        corridors_from[o] = []
    corridors_from[o].append((len(corridors_from[o]), j, d))

# ── 7. One-step-ahead prediction for 2010 ───────────────────────────────────

print("Predicting 2010 …")
log_delta_prev = np.tile(log_delta_2005, (N_DRAWS, 1))  # (4000, N)

eps = np.random.randn(N_DRAWS, len(countries))
log_delta_pred = (1 - phi[:, None]) * mu + phi[:, None] * log_delta_prev + sigma * eps

# DEU outflow prediction
pred_outflow = np.exp(log_delta_pred[:, DEU_IDX]) * pop_cache[('DEU', PRED_YEAR)]

# Inflow to DEU
pred_inflow = np.zeros(N_DRAWS)
pred_corr = {k: np.zeros(N_DRAWS) for k in TRACKED}

for orig_iso, corr_list in corridors_from.items():
    deu_local_idx = None
    for local_i, global_i, dest in corr_list:
        if dest == 'DEU':
            deu_local_idx = local_i
            break
    if deu_local_idx is None:
        continue

    orig_idx = c2i[orig_iso]
    pop_orig = pop_cache[(orig_iso, PRED_YEAR)]
    total_emig = np.exp(log_delta_pred[:, orig_idx]) * pop_orig

    global_indices = [gi for _, gi, _ in corr_list]
    eta = (
        np.random.randn(N_DRAWS, len(corr_list)) * psi[:, global_indices]
        + kappa[:, global_indices]
    )
    eta_max = eta.max(axis=1, keepdims=True)
    exp_eta = np.exp(eta - eta_max)
    shares = exp_eta / exp_eta.sum(axis=1, keepdims=True)

    flow_to_deu = total_emig * shares[:, deu_local_idx]
    pred_inflow += flow_to_deu

    pair_key = f'{orig_iso}_DEU'
    if pair_key in pred_corr:
        pred_corr[pair_key] = flow_to_deu

pred_net = pred_inflow - pred_outflow

# Print summary
for label, pred, obs in [
    ('Outflow', pred_outflow, hist_outflow[2010]),
    ('Inflow', pred_inflow, hist_inflow[2010]),
    ('Net', pred_net, hist_net[2010]),
]:
    med = np.median(pred)
    q025, q975 = np.percentile(pred, [2.5, 97.5])
    print(f"  {label}: Obs={obs:,.0f}  Med={med:,.0f}  95%CI=[{q025:,.0f}, {q975:,.0f}]")

for pair in TRACKED:
    med = np.median(pred_corr[pair])
    q025, q975 = np.percentile(pred_corr[pair], [2.5, 97.5])
    obs = hist_corr[pair][2010]
    print(f"  {pair}: Obs={obs:,.0f}  Med={med:,.0f}  95%CI=[{q025:,.0f}, {q975:,.0f}]")

# ── 8. Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


def plot_panel(ax, hist_dict, pred_draws, title):
    """Plot training history + 2010 observed point + 2010 prediction PI."""
    # Training data (1990–2005)
    train_yrs = sorted([y for y in hist_dict if y < PRED_YEAR])
    train_vals = [hist_dict[y] for y in train_yrs]
    ax.plot(train_yrs, train_vals, 'k-o', lw=2, ms=5, label='Training data')

    # Observed 2010
    obs_val = hist_dict[PRED_YEAR]
    ax.plot(PRED_YEAR, obs_val, 's', color='red', ms=10, zorder=5,
            label=f'Observed {PRED_YEAR}')

    # Predicted 2010 distribution
    med = np.median(pred_draws)
    q025, q10, q90, q975 = [
        np.percentile(pred_draws, p) for p in [2.5, 10, 90, 97.5]
    ]

    ax.plot(PRED_YEAR, med, 'D', color='blue', ms=8, zorder=5,
            label=f'Predicted median')

    # Vertical bars for PI
    ax.vlines(PRED_YEAR, q10, q90, color='blue', lw=6, alpha=0.35, label='80% PI')
    ax.vlines(PRED_YEAR, q025, q975, color='blue', lw=2, alpha=0.25, label='95% PI')

    # Connect last training point to prediction
    ax.plot([train_yrs[-1], PRED_YEAR], [train_vals[-1], med],
            'b--', alpha=0.4)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Persons')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    ax.set_xticks(ALL_YEARS)


plot_panel(axes[0, 0], hist_net, pred_net,
           'A. Net Migration — Germany')
plot_panel(axes[0, 1], hist_inflow, pred_inflow,
           'C. Total Immigration — Germany')
plot_panel(axes[0, 2], hist_outflow, pred_outflow,
           'D. Total Emigration — Germany')
plot_panel(axes[1, 0], hist_corr['TUR_DEU'], pred_corr['TUR_DEU'],
           'F. Immigration from Turkey')
plot_panel(axes[1, 1], hist_corr['POL_DEU'], pred_corr['POL_DEU'],
           'G. Immigration from Poland')
plot_panel(axes[1, 2], hist_corr['USA_DEU'], pred_corr['USA_DEU'],
           'H. Immigration from USA')

fig.suptitle(
    'Out-of-Sample Validation (2010) — Germany\n'
    'Azose & Raftery (2019) Model, trained on 1990–2005',
    fontsize=14, fontweight='bold', y=1.03,
)
fig.tight_layout()
fig.savefig(PROJ / 'graphs' / 'germany_validation_2010.png',
            dpi=200, bbox_inches='tight')
print(f"\nSaved → graphs/germany_validation_2010.png")
plt.close()
