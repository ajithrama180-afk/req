import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ========= OPENPYXL CHECK =========
XLS_ENGINES_AVAILABLE = True
try:
    import openpyxl  # noqa
except ImportError:
    XLS_ENGINES_AVAILABLE = False

WEIGHTS = {
    "TT": 0.20,
    "MA": 0.20,
    "MO": 0.25,
    "VOL": 0.10,
    "SUP": 0.10,
    "EXT": 0.10,
    "DAY": 0.05,
}

# Alternate composite requested weights
ALT_METRIC_WEIGHTS = {
    "ALT_MomW_Score": 0.30,   # Mom (10) 1W
    "ALT_Chg1W_Score": 0.25,  # Change % 1W
    "ALT_Chg1M_Score": 0.20,  # Change % 1M
    "ALT_MA_Pos": 0.15,       # MA Position / crossover quality (reuse F_MA)
    "ALT_RVol_Score": 0.10,   # Rel Volume
    # 0% weight metric intentionally omitted
}

ST_FACTOR_DESCRIPTIONS = {
    "TT": "Trend Turn: Prior negatives (6M/YTD/1Y) + healthy positive 1M",
    "MA": "MA Recovery: Price above EMA9/EMA21/VWMA20 & EMA9 > EMA21 (not over-stretched)",
    "MO": "Momentum Shift: Weekly turning positive & daily stable",
    "VOL": "Rel Volume scaled (target ≥ 1.0)",
    "SUP": "Distance above S1 (5–15% sweet spot)",
    "EXT": "Extension Filter (avoid huge recent runs / extreme collapses)",
    "DAY": "Intraday Follow-through (positive but controlled)",
}

st.set_page_config(page_title="Reversal Ranking Dashboard", layout="wide")

# ---------- Utilities ----------
def normalize_text(s: str):
    if not isinstance(s, str):
        return s
    return s.replace('−', '-').replace('\u00A0', ' ').strip()

def parse_numeric(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = normalize_text(str(x))
    if s == '':
        return np.nan
    mult = 1.0
    if re.search(r'\bB\b', s):
        mult = 1e9
    if re.search(r'\bM\b', s):
        mult = 1e6
    s = (s.replace('USD','')
           .replace('B','')
           .replace('M','')
           .replace('%','')
           .replace(',','')
           .replace('+','')
           .strip())
    try:
        return float(s) * mult
    except ValueError:
        return np.nan

def parse_percent(x):
    return parse_numeric(x)

def extract_ticker_and_name(symbol_field: str):
    if not isinstance(symbol_field, str):
        return symbol_field, symbol_field
    s = normalize_text(symbol_field)
    m = re.match(r'^([A-Z]+)\s+(.*)$', s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return s.strip(), s.strip()

# ---------- Core Factor Functions ----------
def factor_trend_turn(row):
    perf6, perfytd, perf1y, perf1m = row['Perf6M'], row['PerfYTD'], row['Perf1Y'], row['Perf1M']
    negatives = sum([1 for v in [perf6, perfytd, perf1y] if not np.isnan(v) and v < 0])
    base = (negatives / 3) * 60.0
    if np.isnan(perf1m):
        onem = 0
    else:
        if perf1m < 0: onem = 0
        elif perf1m <= 8: onem = 40
        elif perf1m <= 15: onem = 25
        else: onem = 15
    return min(100.0, base + onem)

def factor_ma_recovery(row):
    price, ema9, ema21, vwma = row['Price'], row['EMA9'], row['EMA21'], row['VWMA20']
    pts = 0
    if price > ema9: pts += 1
    if price > ema21: pts += 1
    if price > vwma: pts += 1
    if ema9 > ema21: pts += 1
    core = (pts / 4) * 100
    dist = row['ClusterDist']
    if np.isnan(dist): return core
    if dist <= 4: scale = 1.0
    elif dist <= 8: scale = max(0.0, 1 - (dist - 4)/4)
    else: scale = 0.0
    return core * scale

def factor_momentum_shift(row):
    mw, md = row['Mom10_1W'], row['Mom10_D']
    if mw < -20: w = 10
    elif mw < 0: w = 40
    elif mw < 40: w = 100
    elif mw < 80: w = 60
    else: w = 30
    if md <= -6: d = 20
    elif md < -2: d = 50
    elif md <= 5: d = 100
    elif md <= 10: d = 70
    else: d = 40
    return 0.6 * w + 0.4 * d

def factor_volume(row):
    rv = row['RelVolume']
    if np.isnan(rv): return 0
    return min(rv/1.2, 1.0) * 100

def factor_support(row):
    dist = row['DistS1']
    if np.isnan(dist) or dist < 0: return 0
    if dist < 5: return (dist/5)*40
    if dist <= 15: return 100 - (abs(dist - 10)/5)*20
    if dist <= 25: return 80 - ((dist - 15)/10)*80
    return 0

def factor_extension(row):
    p6, p1y = row['Perf6M'], row['Perf1Y']
    if np.isnan(p6): base = 50
    else:
        if p6 < -40: base = 30
        elif p6 < -10: base = 70
        elif p6 <= 20: base = 100
        elif p6 <= 50: base = 70
        elif p6 <= 100: base = 40
        else: base = 20
    if not np.isnan(p1y) and p1y > 100: base -= 10
    return max(0, min(100, base))

def factor_intraday(row):
    cp = row['ChangePctDay']
    if np.isnan(cp): return 50
    if cp < -4: return 10
    if cp < -2: return 30
    if cp < 0: return 50
    if cp <= 2: return 100
    if cp <= 4: return 80
    return 60

# ---------- Alt Composite Scaling Helpers ----------
def scale_weekly_momentum(mw):
    if np.isnan(mw): return 50
    if mw < -20: return 10
    if mw < 0: return 40
    if mw < 40: return 100
    if mw < 80: return 60
    return 30

def scale_weekly_change(pct):
    if np.isnan(pct): return 50
    if pct <= -15: return 5
    if pct <= -8: return 25
    if pct <= -3: return 60
    if pct <= 3: return 80
    if pct <= 8: return 100
    if pct <= 15: return 70
    return 40

def scale_monthly_change(pct):
    if np.isnan(pct): return 50
    if pct < -20: return 10
    if pct < -10: return 35
    if pct < 0: return 55
    if pct <= 8: return 100
    if pct <= 15: return 80
    if pct <= 30: return 60
    return 40

def scale_rel_volume(rv):
    if np.isnan(rv): return 0
    return min(rv/1.2, 1.0) * 100

# ---------- Sidebar Upload ----------
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx","xls","csv"])
if uploaded is None:
    st.info("Upload a data file to proceed.")
    st.stop()

# ---------- Read File ----------
file_lower = uploaded.name.lower()
try:
    if file_lower.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        if not XLS_ENGINES_AVAILABLE:
            st.error("openpyxl not installed: pip install openpyxl")
            st.stop()
        df_raw = pd.read_excel(uploaded, engine='openpyxl')
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df_raw.columns = [normalize_text(c) for c in df_raw.columns]
df = df_raw.copy()

rename_map = {
    'Symbol': 'Symbol',
    'Mom (10) 1W': 'Mom10_1W',
    'Mom (10)': 'Mom10_D',
    'Change %': 'ChangePctDay',
    'Change%': 'ChangePctDay',
    'Change': 'ChangeUSD',
    'Price': 'Price',
    'Perf % 6M': 'Perf6M',
    'Perf % YTD': 'PerfYTD',
    'Perf % 1Y': 'Perf1Y',
    'Perf % 5Y': 'Perf5Y',
    'Perf % 10Y': 'Perf10Y',
    'Perf % All Time': 'PerfAllTime',
    'Change % 1M': 'Perf1M',
    'EMA (21)': 'EMA21',
    'EMA (9)': 'EMA9',
    'VWMA (20)': 'VWMA20',
    'Rel Volume': 'RelVolume',
    'ADR': 'ADR',
    'Change 1M': 'Change1MUSD',
    'Classic S1': 'ClassicS1',
    'Market cap': 'MarketCap',
    'Market cap perf % 1W': 'MktCapPerf1W',
    'Change % 1W': 'ChangePct1W',
    'Change % (1W)': 'ChangePct1W',
    'Change% 1W': 'ChangePct1W',
    'Perf % 1W': 'ChangePct1W'
}
for col in list(df.columns):
    base = col.replace('  ', ' ').strip()
    if base in rename_map:
        df.rename(columns={col: rename_map[base]}, inplace=True)

# Ticker / Name split
tickers, names = [], []
if 'Symbol' not in df.columns:
    st.error("Required column 'Symbol' missing.")
    st.stop()
for val in df['Symbol']:
    t, n = extract_ticker_and_name(val)
    tickers.append(t)
    names.append(n)
df['Ticker'] = tickers
df['Name'] = names

num_cols_pct = ['Perf6M','PerfYTD','Perf1Y','Perf5Y','Perf10Y','PerfAllTime','Perf1M',
                'ChangePctDay','ChangePct1W','MktCapPerf1W']
num_cols_price = ['Price','EMA21','EMA9','VWMA20','RelVolume','ADR','ClassicS1','ChangeUSD',
                  'Change1MUSD','Mom10_1W','Mom10_D','MarketCap']
for c in num_cols_pct:
    if c in df.columns:
        df[c] = df[c].apply(parse_percent)
for c in num_cols_price:
    if c in df.columns:
        df[c] = df[c].apply(parse_numeric)

# Derived moving average cluster
df['ClusterMean'] = df[['EMA9','EMA21','VWMA20']].mean(axis=1, skipna=True)
df['ClusterDist'] = (df['Price'] - df['ClusterMean']) / df['ClusterMean'] * 100
df.loc[df['ClusterMean'].isna() | (df['ClusterMean'] == 0), 'ClusterDist'] = np.nan
df['ClusterDist'].replace([np.inf, -np.inf], np.nan, inplace=True)

if 'ClassicS1' in df.columns:
    df['DistS1'] = (df['Price'] - df['ClassicS1']) / df['ClassicS1'] * 100
    df.loc[df['ClassicS1'].isna() | (df['ClassicS1'] == 0), 'DistS1'] = np.nan
    df['DistS1'].replace([np.inf, -np.inf], np.nan, inplace=True)
else:
    df['DistS1'] = np.nan

# Core factor scores
df['F_TT'] = df.apply(factor_trend_turn, axis=1)
df['F_MA'] = df.apply(factor_ma_recovery, axis=1)
df['F_MO'] = df.apply(factor_momentum_shift, axis=1)
df['F_VOL'] = df.apply(factor_volume, axis=1)
df['F_SUP'] = df.apply(factor_support, axis=1)
df['F_EXT'] = df.apply(factor_extension, axis=1)
df['F_DAY'] = df.apply(factor_intraday, axis=1)

df['Score'] = (
    df['F_TT'] * WEIGHTS['TT'] +
    df['F_MA'] * WEIGHTS['MA'] +
    df['F_MO'] * WEIGHTS['MO'] +
    df['F_VOL'] * WEIGHTS['VOL'] +
    df['F_SUP'] * WEIGHTS['SUP'] +
    df['F_EXT'] * WEIGHTS['EXT'] +
    df['F_DAY'] * WEIGHTS['DAY']
)

# ---------- Alternate Composite Construction ----------
alt_source_presence = {
    'Mom10_1W': 'Mom10_1W' in df.columns,
    'ChangePct1W': 'ChangePct1W' in df.columns,
    'Perf1M': 'Perf1M' in df.columns,
    'RelVolume': 'RelVolume' in df.columns,
    'EMA9': 'EMA9' in df.columns,
    'EMA21': 'EMA21' in df.columns,
    'VWMA20': 'VWMA20' in df.columns
}

# Build ALT_* columns safely
df['ALT_MomW_Score'] = df['Mom10_1W'].apply(scale_weekly_momentum) if 'Mom10_1W' in df.columns else np.nan

if 'ChangePct1W' in df.columns:
    df['ALT_Chg1W_Score'] = df['ChangePct1W'].apply(scale_weekly_change)
elif 'MktCapPerf1W' in df.columns:
    df['ALT_Chg1W_Score'] = df['MktCapPerf1W'].apply(scale_weekly_change)
else:
    df['ALT_Chg1W_Score'] = np.nan

df['ALT_Chg1M_Score'] = df['Perf1M'].apply(scale_monthly_change) if 'Perf1M' in df.columns else np.nan
df['ALT_MA_Pos'] = df['F_MA']  # reuse MA factor
df['ALT_RVol_Score'] = df['RelVolume'].apply(scale_rel_volume) if 'RelVolume' in df.columns else np.nan

# Ensure all expected alt columns exist (even if user edited earlier code)
alt_metric_cols = ["ALT_MomW_Score","ALT_Chg1W_Score","ALT_Chg1M_Score","ALT_MA_Pos","ALT_RVol_Score"]
for col in alt_metric_cols:
    if col not in df.columns:
        df[col] = np.nan

df['AltComposite'] = (
    df['ALT_MomW_Score'] * ALT_METRIC_WEIGHTS['ALT_MomW_Score'] +
    df['ALT_Chg1W_Score'] * ALT_METRIC_WEIGHTS['ALT_Chg1W_Score'] +
    df['ALT_Chg1M_Score'] * ALT_METRIC_WEIGHTS['ALT_Chg1M_Score'] +
    df['ALT_MA_Pos'] * ALT_METRIC_WEIGHTS['ALT_MA_Pos'] +
    df['ALT_RVol_Score'] * ALT_METRIC_WEIGHTS['ALT_RVol_Score']
)

for comp_col, w in ALT_METRIC_WEIGHTS.items():
    contrib_col = f'AltWeightedContribution_{comp_col}'
    df[contrib_col] = df[comp_col] * w

df.sort_values('Score', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- UI ----------
st.title("Reversal / Early Push Ranking Dashboard")
st.caption("Example analytics – not investment advice.")

# Debug sidebar for missing columns
with st.sidebar.expander("Debug / Columns", expanded=False):
    st.write("Alt source fields presence:")
    st.json(alt_source_presence)
    missing_alt_cols = [c for c in alt_metric_cols if c not in df.columns]
    if missing_alt_cols:
        st.warning(f"Missing ALT columns (filled as NaN): {missing_alt_cols}")
    else:
        st.success("All ALT metric columns present.")

st.sidebar.subheader("Factor Weights")
with st.sidebar.expander("Adjust Core Weights", expanded=False):
    new_weights = {}
    total_preview = 0
    for k, v in WEIGHTS.items():
        new_weights[k] = st.slider(f"{k} ({ST_FACTOR_DESCRIPTIONS[k].split(':')[0]})", 0.0, 0.5, v, 0.01)
        total_preview += new_weights[k]
    st.write(f"Sum: {total_preview:.2f}")
    if st.button("Apply New Core Weights"):
        if abs(total_preview - 1.0) > 1e-6:
            st.warning("Weights must sum to 1.0.")
        else:
            WEIGHTS.update(new_weights)
            df['Score'] = (
                df['F_TT'] * WEIGHTS['TT'] +
                df['F_MA'] * WEIGHTS['MA'] +
                df['F_MO'] * WEIGHTS['MO'] +
                df['F_VOL'] * WEIGHTS['VOL'] +
                df['F_SUP'] * WEIGHTS['SUP'] +
                df['F_EXT'] * WEIGHTS['EXT'] +
                df['F_DAY'] * WEIGHTS['DAY']
            )
            df.sort_values('Score', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

st.sidebar.subheader("Filters / Sections")
min_score = st.sidebar.slider("Minimum Core Score", 0.0, 100.0, 0.0, 1.0)
show_extended = st.sidebar.checkbox("Show Extended Columns", value=False)
show_bar_section = st.sidebar.checkbox("Show Core Score Charts", value=True)
show_alt_section = st.sidebar.checkbox("Show Alt Composite Section", value=True)

df_view = df[df['Score'] >= min_score].copy()

st.subheader("Ranking Table")
basic_cols = ['Ticker','Name','Score','AltComposite',
              'F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY',
              'Perf6M','PerfYTD','Perf1Y','Perf1M','Mom10_1W','Mom10_D',
              'Price','EMA9','EMA21','VWMA20','RelVolume','DistS1','ClusterDist']
extended_cols = basic_cols + [
    'ChangePctDay','ChangeUSD','Change1MUSD','ClassicS1','MarketCap','ADR','ClusterMean',
    'ChangePct1W','MktCapPerf1W',
    'ALT_MomW_Score','ALT_Chg1W_Score','ALT_Chg1M_Score','ALT_MA_Pos','ALT_RVol_Score'
]
cols_to_show = extended_cols if show_extended else basic_cols
cols_to_show = [c for c in cols_to_show if c in df_view.columns]

def color_score(val):
    if pd.isna(val): return ''
    return f'background-color: rgba({int(255 - (val/100)*255)}, {int((val/100)*200)}, 120, 0.35)'

style_cols_color = ['Score','F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY','AltComposite'] + \
                   [c for c in alt_metric_cols if c in df_view.columns]

styled = (
    df_view[cols_to_show]
    .style
    .format({c: "{:.2f}" for c in cols_to_show if c not in ['Ticker','Name']})
    .applymap(color_score, subset=[c for c in style_cols_color if c in cols_to_show])
)
st.dataframe(styled, use_container_width=True)

# ---------- Scatter ----------
st.subheader("Momentum Scatter (Weekly vs Daily)")
if not df_view.empty and 'Mom10_D' in df_view.columns and 'Mom10_1W' in df_view.columns:
    fig_scatter = px.scatter(
        df_view,
        x='Mom10_D',
        y='Mom10_1W',
        color='Score',
          size='RelVolume' if 'RelVolume' in df_view.columns else None,
        hover_data=[c for c in ['Ticker','Score','Perf1M','Perf6M','Price'] if c in df_view.columns],
        color_continuous_scale='Tealrose',
        title="Weekly Momentum vs Daily Momentum"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Momentum columns not available for scatter plot.")

# ---------- Core Score Charts ----------
if show_bar_section and not df_view.empty:
    st.subheader("Core Score Bar Charts")
    colA, colB = st.columns(2)
    with colA:
        factor_cols = ['F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY']
        existing_factors = [c for c in factor_cols if c in df_view.columns]
        if existing_factors:
            avg_series = df_view[existing_factors].mean().round(2)
            avg_df = avg_series.reset_index()
            avg_df.columns = ['Factor','AverageScore']
            avg_fig = px.bar(
                avg_df, x='Factor', y='AverageScore', text='AverageScore',
                range_y=[0,100], title="Mean Core Factor Scores"
            )
            avg_fig.update_traces(textposition='outside')
            st.plotly_chart(avg_fig, use_container_width=True)
        else:
            st.info("No core factor columns found.")
    with colB:
        top_n = st.slider("Top N (Core Composite)", 3, min(50, max(3, len(df_view))), min(10, len(df_view)))
        top_df = df_view.nlargest(top_n, 'Score')[['Ticker','Score']].copy().iloc[::-1]
        bar_fig = px.bar(
            top_df, x='Score', y='Ticker', orientation='h',
            text='Score', range_x=[0,100], title=f"Top {top_n} Core Composite Scores"
        )
        bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("**Stacked Weighted Core Factor Contributions**")
    show_stack = st.checkbox("Show Core Factor Stack", value=True)
    if show_stack:
        top_n_stack = st.slider("Top N (Core Stack)", 3, min(30, max(3, len(df_view))), min(8, len(df_view)))
        top_stack_df = df_view.nlargest(top_n_stack, 'Score').copy()
        factor_map = {'TT':'F_TT','MA':'F_MA','MO':'F_MO','VOL':'F_VOL','SUP':'F_SUP','EXT':'F_EXT','DAY':'F_DAY'}
        contrib_rows = []
        for _, r in top_stack_df.iterrows():
            for short, col in factor_map.items():
                if col in r:
                    rv = r[col]
                    contrib_rows.append({
                        'Ticker': r['Ticker'],
                        'Factor': short,
                        'RawScore': rv,
                        'Weight': WEIGHTS[short],
                        'WeightedContribution': rv * WEIGHTS[short]
                    })
        contrib_long = pd.DataFrame(contrib_rows)
        if not contrib_long.empty:
            ticker_order = top_stack_df.sort_values('Score', ascending=True)['Ticker'].tolist()
            stack_fig = px.bar(
                contrib_long, x='WeightedContribution', y='Ticker', color='Factor',
                category_orders={'Ticker': ticker_order},
                title=f"Weighted Core Contributions (Top {top_n_stack})",
                hover_data=['RawScore','Weight'], orientation='h'
            )
            stack_fig.update_layout(barmode='stack', xaxis_title="Weighted Contribution")
            st.plotly_chart(stack_fig, use_container_width=True)

# ---------- Alternate Composite Section ----------
if show_alt_section and not df_view.empty:
    st.subheader("Alternate Momentum Composite")
    st.caption("Weights: Mom(10)1W 30% | Change%1W 25% | Change%1M 20% | MA Pos 15% | RelVol 10%")

    try:
        alt_metrics = ['ALT_MomW_Score','ALT_Chg1W_Score','ALT_Chg1M_Score','ALT_MA_Pos','ALT_RVol_Score']
        existing_alt = [c for c in alt_metrics if c in df_view.columns]
        if existing_alt:
            alt_avg = df_view[existing_alt].mean().round(2).reset_index()
            alt_avg.columns = ['Metric','AverageScore']
            alt_avg_fig = px.bar(
                alt_avg, x='Metric', y='AverageScore', text='AverageScore',
                range_y=[0,100], title="Mean Alt Metric Scores"
            )
            alt_avg_fig.update_traces(textposition='outside')
            st.plotly_chart(alt_avg_fig, use_container_width=True)
        else:
            st.info("No Alt metric columns available (all missing).")

        col1, col2 = st.columns(2)
        with col1:
            if 'AltComposite' in df_view.columns:
                top_n_alt = st.slider("Top N (Alt Composite)", 3, min(50, max(3, len(df_view))),
                                      min(10, len(df_view)), key="alt_top_n")
                alt_top_df = df_view.nlargest(top_n_alt, 'AltComposite')[['Ticker','AltComposite']].copy().iloc[::-1]
                alt_bar = px.bar(
                    alt_top_df, x='AltComposite', y='Ticker',
                    orientation='h', text='AltComposite', range_x=[0,100],
                    title=f"Top {top_n_alt} AltComposite Scores"
                )
                alt_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(alt_bar, use_container_width=True)
            else:
                st.warning("AltComposite column missing.")
        with col2:
            show_alt_stack = st.checkbox("Show Alt Weighted Contribution Stack", value=True)
            if show_alt_stack and 'AltComposite' in df_view.columns:
                top_n_alt_stack = st.slider("Top N (Alt Stack)", 3, min(30, max(3, len(df_view))),
                                            min(8, len(df_view)), key="alt_stack_n")
                alt_stack_df = df_view.nlargest(top_n_alt_stack, 'AltComposite').copy()
                alt_contrib_rows = []
                for _, r in alt_stack_df.iterrows():
                    for metric_col, w in ALT_METRIC_WEIGHTS.items():
                        raw_val = r.get(metric_col, np.nan)
                        alt_contrib_rows.append({
                            'Ticker': r['Ticker'],
                            'Metric': metric_col.replace('ALT_','').replace('_Score',''),
                            'RawScore': raw_val,
                            'Weight': w,
                            'WeightedContribution': raw_val * w
                        })
                alt_contrib_long = pd.DataFrame(alt_contrib_rows)
                if not alt_contrib_long.empty:
                    ticker_order_alt = alt_stack_df.sort_values('AltComposite', ascending=True)['Ticker'].tolist()
                    alt_stack_fig = px.bar(
                        alt_contrib_long,
                        x='WeightedContribution', y='Ticker', color='Metric',
                        category_orders={'Ticker': ticker_order_alt},
                        title=f"Alt Composite Weighted Contributions (Top {top_n_alt_stack})",
                        hover_data=['RawScore','Weight'], orientation='h'
                    )
                    alt_stack_fig.update_layout(barmode='stack', xaxis_title="Weighted Contribution")
                    st.plotly_chart(alt_stack_fig, use_container_width=True)

        st.markdown("**Single Ticker Alt Composite Breakdown**")
        if 'AltComposite' in df_view.columns:
            alt_tickers = df_view[~df_view['AltComposite'].isna()].sort_values('AltComposite', ascending=False)['Ticker'].tolist()
            if alt_tickers:
                sel_alt = st.selectbox("Select Ticker (Alt)", alt_tickers, index=0, key="alt_single_select")
                alt_row = df_view[df_view['Ticker'] == sel_alt].iloc[0]
                contrib_rows = []
                for metric_col, w in ALT_METRIC_WEIGHTS.items():
                    raw_val = alt_row.get(metric_col, np.nan)
                    contrib_rows.append({
                        'Metric': metric_col.replace('ALT_','').replace('_Score',''),
                        'RawScore': raw_val,
                        'Weight': w,
                        'WeightedContribution': raw_val * w
                    })
                contrib_df = pd.DataFrame(contrib_rows)
                contrib_fig = px.bar(
                    contrib_df, x='Metric', y='WeightedContribution',
                    text='WeightedContribution', hover_data=['RawScore','Weight'],
                    title=f"Alt Composite Contributions: {sel_alt}"
                )
                contrib_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(contrib_fig, use_container_width=True)
            else:
                st.info("No tickers have AltComposite values.")
    except Exception as e:
        st.error(f"Alt Composite section error: {e}")

# ---------- Original Factor Radar ----------
st.subheader("Factor Radar / Single Ticker (Core)")
tickers_core = df_view['Ticker'].tolist()
if tickers_core:
    selected = st.selectbox("Select Ticker (Core)", tickers_core, index=0, key="core_single_ticker")
    row_core = df_view[df_view['Ticker'] == selected].iloc[0]
    factor_vals = {
        "TT": row_core['F_TT'],
        "MA": row_core['F_MA'],
        "MO": row_core['F_MO'],
        "VOL": row_core['F_VOL'],
        "SUP": row_core['F_SUP'],
        "EXT": row_core['F_EXT'],
        "DAY": row_core['F_DAY']
    }
    radar_df = pd.DataFrame({'Factor': list(factor_vals.keys()), 'Score': list(factor_vals.values())})
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_df['Score'],
        theta=radar_df['Factor'],
        fill='toself',
        name=selected
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("Weighted Factor Contributions (Selected Ticker)")
    contrib = []
    for k, v in factor_vals.items():
        w = WEIGHTS[k]
        contrib.append({'Factor': k, 'RawScore': v, 'Weight': w, 'WeightedContribution': v * w})
    contrib_df = pd.DataFrame(contrib)
    bar_fig = px.bar(
        contrib_df, x='Factor', y='WeightedContribution',
        hover_data=['RawScore','Weight'], text='WeightedContribution',
        title=f"Core Composite Build: {selected}"
    )
    bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("**Factor Definitions:**")
    for k, desc in ST_FACTOR_DESCRIPTIONS.items():
        st.markdown(f"- **{k}**: {desc}")

# ---------- Download ----------
st.subheader("Download Ranked Data")
download_cols = [
    'Ticker','Name','Score','AltComposite',
    'F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY',
    'Perf6M','PerfYTD','Perf1Y','Perf1M','Mom10_1W','Mom10_D',
    'Price','EMA9','EMA21','VWMA20','RelVolume','DistS1','ClusterDist',
    'ChangePctDay','MarketCap','ChangePct1W',
    'ALT_MomW_Score','ALT_Chg1W_Score','ALT_Chg1M_Score','ALT_MA_Pos','ALT_RVol_Score'
]
download_cols = [c for c in download_cols if c in df_view.columns]
out_csv = df_view[download_cols].to_csv(index=False)
st.download_button("Download CSV", data=out_csv, file_name="reversal_ranking.csv", mime="text/csv")

st.caption("© Reversal Ranking Dashboard – Example analytics. Not investment advice.")
