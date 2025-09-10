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

# ---------- Factors ----------
def factor_trend_turn(row):
    perf6, perfytd, perf1y, perf1m = row['Perf6M'], row['PerfYTD'], row['Perf1Y'], row['Perf1M']
    negatives = sum([1 for v in [perf6, perfytd, perf1y] if not np.isnan(v) and v < 0])
    base = (negatives / 3) * 60.0
    if np.isnan(perf1m): onem = 0
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

# ---------- Sidebar Upload ----------
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx","xls","csv"])

if uploaded is None:
    st.info("Upload the file (e.g. 9_92025.xlsx) to proceed.")
    st.stop()

# ---------- Read File ----------
file_lower = uploaded.name.lower()
try:
    if file_lower.endswith('.csv'):
        df_raw = pd.read_csv(uploaded)
    else:
        if not XLS_ENGINES_AVAILABLE:
            st.error("openpyxl is not installed. Install with: pip install openpyxl (then re-run).")
            st.stop()
        df_raw = pd.read_excel(uploaded, engine='openpyxl')
except ImportError as ie:
    st.error(f"ImportError: {ie}. Install openpyxl: pip install openpyxl")
    st.stop()
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
    'Market cap perf % 1W': 'MktCapPerf1W'
}

existing = df.columns.tolist()
for col in existing:
    base = col.replace('  ', ' ').strip()
    if base in rename_map:
        df.rename(columns={col: rename_map[base]}, inplace=True)

tickers, names = [], []
for val in df['Symbol']:
    t, n = extract_ticker_and_name(val)
    tickers.append(t)
    names.append(n)
df['Ticker'] = tickers
df['Name'] = names

num_cols_pct = ['Perf6M','PerfYTD','Perf1Y','Perf5Y','Perf10Y','PerfAllTime','Perf1M','ChangePctDay']
num_cols_price = ['Price','EMA21','EMA9','VWMA20','RelVolume','ADR','ClassicS1','ChangeUSD','Change1MUSD','Mom10_1W','Mom10_D','MarketCap','MktCapPerf1W']
for c in num_cols_pct:
    if c in df.columns:
        df[c] = df[c].apply(parse_percent)
for c in num_cols_price:
    if c in df.columns:
        df[c] = df[c].apply(parse_numeric)

# ---------- Vectorized Derived Metrics ----------
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

# ---------- Factor Scores ----------
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

df.sort_values('Score', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- UI ----------
st.title("Reversal / Early Push Ranking Dashboard")
st.caption("Emerging from downtrends and pushing higher (example analytics; not investment advice).")

st.sidebar.subheader("Factor Weights")
with st.sidebar.expander("Adjust Weights", expanded=False):
    new_weights = {}
    total_preview = 0
    for k, v in WEIGHTS.items():
        new_weights[k] = st.slider(f"{k} ({ST_FACTOR_DESCRIPTIONS[k].split(':')[0]})", 0.0, 0.5, v, 0.01)
        total_preview += new_weights[k]
    st.write(f"Sum: {total_preview:.2f}")
    if st.button("Apply New Weights"):
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

st.sidebar.subheader("Filters / Display")
min_score = st.sidebar.slider("Minimum Score", 0.0, 100.0, 0.0, 1.0)
show_extended = st.sidebar.checkbox("Show Extended Columns", value=False)
show_bar_section = st.sidebar.checkbox("Show Score Bar Charts Section", value=True)

df_view = df[df['Score'] >= min_score].copy()

st.subheader("Ranking Table")
basic_cols = ['Ticker','Name','Score','F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY',
              'Perf6M','PerfYTD','Perf1Y','Perf1M','Mom10_1W','Mom10_D','Price','EMA9','EMA21','VWMA20',
              'RelVolume','DistS1','ClusterDist']
extended_cols = basic_cols + ['ChangePctDay','ChangeUSD','Change1MUSD','ClassicS1','MarketCap','ADR','ClusterMean']
cols_to_show = extended_cols if show_extended else basic_cols
cols_to_show = [c for c in cols_to_show if c in df_view.columns]

def color_score(val):
    if pd.isna(val): return ''
    return f'background-color: rgba({int(255 - (val/100)*255)}, {int((val/100)*200)}, 120, 0.35)'

styled = (df_view[cols_to_show]
          .style
          .format({c: "{:.2f}" for c in df_view.columns if c not in ['Ticker','Name']})
          .applymap(color_score, subset=['Score','F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY'])
         )
st.dataframe(styled, use_container_width=True)

# ---------- Momentum Scatter ----------
st.subheader("Momentum Scatter (Weekly vs Daily)")
fig_scatter = px.scatter(
    df_view,
    x='Mom10_D',
    y='Mom10_1W',
    color='Score',
    size='RelVolume',
    hover_data=['Ticker','Score','Perf1M','Perf6M','Price'],
    color_continuous_scale='Tealrose',
    title="Weekly Momentum vs Daily Momentum"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- NEW: Score Bar Charts Section ----------
if show_bar_section and not df_view.empty:
    st.subheader("Score Bar Charts")

    colA, colB = st.columns([1,1])
    with colA:
        st.markdown("**Average Raw Factor Scores** (Filtered Universe)")
        factor_cols = ['F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY']
        avg_series = df_view[factor_cols].mean().round(2)
        avg_df = avg_series.reset_index()
          # rename columns
        avg_df.columns = ['Factor','AverageScore']
        avg_fig = px.bar(
            avg_df,
            x='Factor',
            y='AverageScore',
            text='AverageScore',
            range_y=[0,100],
            title="Mean Factor Scores"
        )
        avg_fig.update_traces(textposition='outside')
        st.plotly_chart(avg_fig, use_container_width=True)

    with colB:
        top_n = st.slider("Top N for Composite Score Bar", 3, min(50, max(3, len(df_view))), min(10, len(df_view)))
        top_df = df_view.nlargest(top_n, 'Score')[['Ticker','Score']].copy()
        top_df = top_df.iloc[::-1]  # reverse for horizontal bar (lowest at bottom)
        bar_fig = px.bar(
            top_df,
            x='Score',
            y='Ticker',
            orientation='h',
            text='Score',
            range_x=[0, 100],
            title=f"Top {top_n} Composite Scores"
        )
        bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("**Stacked Weighted Factor Contributions (Top N)**")
    show_stack = st.checkbox("Show Stacked Weighted Contributions", value=True)
    if show_stack:
        top_n_stack = st.slider("Top N for Contribution Stack", 3, min(30, max(3, len(df_view))), min(8, len(df_view)))
        top_stack_df = df_view.nlargest(top_n_stack, 'Score').copy()
        factor_map = {
            'TT': 'F_TT',
            'MA': 'F_MA',
            'MO': 'F_MO',
            'VOL': 'F_VOL',
            'SUP': 'F_SUP',
            'EXT': 'F_EXT',
            'DAY': 'F_DAY'
        }
        contrib_rows = []
        for _, r in top_stack_df.iterrows():
            for short, col in factor_map.items():
                raw_val = r[col]
                weight = WEIGHTS[short]
                contrib_rows.append({
                    'Ticker': r['Ticker'],
                    'Factor': short,
                    'RawScore': raw_val,
                    'Weight': weight,
                    'WeightedContribution': raw_val * weight
                })
        contrib_long = pd.DataFrame(contrib_rows)
        # order tickers by total score ascending for readable horizontal stacking
        ticker_order = top_stack_df.sort_values('Score', ascending=True)['Ticker'].tolist()
        stack_fig = px.bar(
            contrib_long,
            x='WeightedContribution',
            y='Ticker',
            color='Factor',
            category_orders={'Ticker': ticker_order},
            title=f"Weighted Factor Contributions (Top {top_n_stack})",
            hover_data=['RawScore','Weight'],
            orientation='h'
        )
        stack_fig.update_layout(barmode='stack', xaxis_title="Weighted Contribution (Composite Points)")
        st.plotly_chart(stack_fig, use_container_width=True)

# ---------- Single Ticker Radar & Detailed Contribution ----------
st.subheader("Factor Radar / Single Ticker")
tickers = df_view['Ticker'].tolist()
if tickers:
    selected = st.selectbox("Select Ticker", tickers, index=0, key="single_ticker_select")
    row = df_view[df_view['Ticker'] == selected].iloc[0]
    factor_vals = {
        "TT": row['F_TT'],
        "MA": row['F_MA'],
        "MO": row['F_MO'],
        "VOL": row['F_VOL'],
        "SUP": row['F_SUP'],
        "EXT": row['F_EXT'],
        "DAY": row['F_DAY']
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
        contrib_df,
        x='Factor',
        y='WeightedContribution',
        hover_data=['RawScore','Weight'],
        text='WeightedContribution',
        title=f"Composite Build: {selected}"
    )
    bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("**Factor Definitions:**")
    for k, desc in ST_FACTOR_DESCRIPTIONS.items():
        st.markdown(f"- **{k}**: {desc}")

# ---------- Download ----------
st.subheader("Download Ranked Data")
download_cols = ['Ticker','Name','Score','F_TT','F_MA','F_MO','F_VOL','F_SUP','F_EXT','F_DAY',
                 'Perf6M','PerfYTD','Perf1Y','Perf1M','Mom10_1W','Mom10_D','Price','EMA9','EMA21','VWMA20',
                 'RelVolume','DistS1','ClusterDist','ChangePctDay','MarketCap']
download_cols = [c for c in download_cols if c in df_view.columns]
out_csv = df_view[download_cols].to_csv(index=False)
st.download_button("Download CSV", data=out_csv, file_name="reversal_ranking.csv", mime="text/csv")

st.caption("© Reversal Ranking Dashboard – Example analytics. Not investment advice.")