"""
============================================================
 PROJECT 2 — SMART MONEY CONCEPT (SMC) SCREENER
 pip install yfinance pandas plotly
 python smc_screener.py
============================================================
CONCEPTS YOU WILL LEARN:
  - Order Block detection (OB) — bullish and bearish
  - Fair Value Gap detection (FVG / Imbalance)
  - Break of Structure (BOS) — trend continuation
  - Change of Character (CHoCH) — reversal signal
  - Swing high / swing low detection algorithm
  - Liquidity pools (BSL / SSL)
  - How to read OHLC data with pandas
  - Interactive charts with plotly
============================================================
HOW TO USE:
  1. Run the script — it fetches live data (no API key needed)
  2. It prints a screening summary in the terminal
  3. It saves interactive HTML charts — open them in a browser
  4. Try changing PAIRS, TIMEFRAME, LOOKBACK_DAYS below
============================================================
"""

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  ★ EDIT THESE TO CUSTOMISE YOUR SCREEN
# ──────────────────────────────────────────────────────────

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
}

TIMEFRAME    = "1h"    # Options: 1m 5m 15m 30m 1h 4h 1d
LOOKBACK_DAYS = 14     # How many days back to fetch
SWING_N      = 3       # Candles each side to define a swing point
OB_MULT      = 1.5     # Impulse candle must be this × avg range to qualify


# ──────────────────────────────────────────────────────────
#  STEP 1 — FETCH DATA
# ──────────────────────────────────────────────────────────

def fetch(symbol: str, days: int, interval: str) -> pd.DataFrame:
    """
    Pull OHLC candles from Yahoo Finance for free.
    Returns a clean DataFrame with lowercase column names.

    LEARNING NOTE:
      yfinance wraps Yahoo Finance — completely free, no key needed.
      Returned columns: Open, High, Low, Close, Volume
      We rename them to lowercase for convenience.
    """
    print(f"  Fetching {symbol} ({interval}, {days}d)...", end=" ")
    t = yf.Ticker(symbol)
    df = t.history(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        interval=interval,
    )
    if df.empty:
        print("NO DATA")
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    print(f"{len(df)} candles ✓")
    return df


# ──────────────────────────────────────────────────────────
#  STEP 2 — DETECT SWING HIGHS & LOWS
# ──────────────────────────────────────────────────────────

def find_swings(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Identify swing highs and swing lows.

    LEARNING NOTE:
      A SWING HIGH at candle [i] means:
        candle[i].high is the HIGHEST of all candles in [i-n … i+n]
      A SWING LOW at candle [i] means:
        candle[i].low is the LOWEST of all candles in [i-n … i+n]

      These points are the FOUNDATION of all ICT analysis.
      Higher 'n' = fewer, more significant swings.
      Lower  'n' = more swings (noisier but more responsive).
    """
    df = df.copy()
    df["sh"] = False   # swing high
    df["sl"] = False   # swing low
    for i in range(n, len(df) - n):
        window_h = df["high"].iloc[i - n : i + n + 1]
        window_l = df["low"].iloc[i - n : i + n + 1]
        if df["high"].iloc[i] == window_h.max():
            df.loc[df.index[i], "sh"] = True
        if df["low"].iloc[i] == window_l.min():
            df.loc[df.index[i], "sl"] = True
    return df


# ──────────────────────────────────────────────────────────
#  STEP 3 — DETECT MARKET STRUCTURE (BOS / CHoCH)
# ──────────────────────────────────────────────────────────

def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each candle with its market structure event.

    LEARNING NOTE:
      BOS  (Break of Structure) = price closes BEYOND a swing in
           the SAME direction as current trend → trend continues.

      CHoCH (Change of Character) = price closes BEYOND a swing
           AGAINST the current trend → potential REVERSAL.
           CHoCH is always the FIRST opposing BOS.

    Algorithm (simple version):
      - Track last swing high (last_sh) and last swing low (last_sl)
      - If close > last_sh and trend was bullish  → BOS_BULL
      - If close > last_sh and trend was bearish  → CHoCH_BULL (reversal!)
      - If close < last_sl and trend was bearish  → BOS_BEAR
      - If close < last_sl and trend was bullish  → CHoCH_BEAR (reversal!)
    """
    df = df.copy()
    df["structure"] = ""
    df["trend"] = "NEUTRAL"

    last_sh = None
    last_sl = None
    trend = "NEUTRAL"

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        if prev["sh"]:
            last_sh = prev["high"]
        if prev["sl"]:
            last_sl = prev["low"]

        if last_sh is None or last_sl is None:
            continue

        close = df["close"].iloc[i]
        prev_trend = trend

        if close > last_sh:
            trend = "BULLISH"
            label = "BOS_BULL" if prev_trend == "BULLISH" else "CHOCH_BULL"
            df.loc[df.index[i], "structure"] = label
        elif close < last_sl:
            trend = "BEARISH"
            label = "BOS_BEAR" if prev_trend == "BEARISH" else "CHOCH_BEAR"
            df.loc[df.index[i], "structure"] = label

        df.loc[df.index[i], "trend"] = trend

    return df


# ──────────────────────────────────────────────────────────
#  STEP 4 — DETECT ORDER BLOCKS
# ──────────────────────────────────────────────────────────

def detect_obs(df: pd.DataFrame, mult: float = 1.5) -> list:
    """
    Find Order Blocks in the price data.

    LEARNING NOTE:
      An ORDER BLOCK is the LAST opposing candle before a strong
      impulse move. It represents a zone where institutions placed
      large orders and price is likely to respect it when revisited.

      Bullish OB:
        - candle[i] is bearish (close < open)
        - candle[i+1] is bullish AND moves strongly upward
        - candle[i+1].close > candle[i].high (displacement)

      Bearish OB:
        - candle[i] is bullish (close > open)
        - candle[i+1] is bearish AND moves strongly downward
        - candle[i+1].close < candle[i].low (displacement)

      'mult' controls how large the impulse must be relative
      to the average candle range to count as a valid OB.
    """
    obs = []
    avg_range = (df["high"] - df["low"]).mean()
    threshold = avg_range * mult

    for i in range(1, len(df) - 2):
        c  = df.iloc[i]
        cn = df.iloc[i + 1]
        impulse = abs(cn["close"] - cn["open"])
        if impulse < threshold:
            continue

        # ── Bullish OB ──
        if (c["close"] < c["open"]               # candle is bearish
                and cn["close"] > cn["open"]      # next is bullish
                and cn["close"] > c["high"]):     # displacement
            obs.append({
                "type":      "BULL",
                "top":        c["high"],
                "bottom":     c["low"],
                "time":       df.index[i],
                "mitigated":  False,
                "color_fill": "rgba(0,229,160,.12)",
                "color_line": "#00e5a0",
            })

        # ── Bearish OB ──
        elif (c["close"] > c["open"]              # candle is bullish
                and cn["close"] < cn["open"]      # next is bearish
                and cn["close"] < c["low"]):      # displacement
            obs.append({
                "type":      "BEAR",
                "top":        c["high"],
                "bottom":     c["low"],
                "time":       df.index[i],
                "mitigated":  False,
                "color_fill": "rgba(255,74,107,.12)",
                "color_line": "#ff4a6b",
            })

    # ── Mark mitigated OBs (price returned to the OB zone) ──
    for ob in obs:
        idx = df.index.get_loc(ob["time"])
        future = df.iloc[idx + 2 :]
        if ob["type"] == "BULL":
            ob["mitigated"] = bool((future["low"] <= ob["top"]).any())
        else:
            ob["mitigated"] = bool((future["high"] >= ob["bottom"]).any())

    return obs


# ──────────────────────────────────────────────────────────
#  STEP 5 — DETECT FAIR VALUE GAPS
# ──────────────────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame) -> list:
    """
    Find Fair Value Gaps (FVG / Imbalance / CISD).

    LEARNING NOTE:
      An FVG is a 3-candle pattern where price moves so fast that
      it leaves a GAP between candle 1 and candle 3.

      Bullish FVG:
        candle[0].high  <  candle[2].low
        → Gap exists ABOVE candle[0] and BELOW candle[2]
        → The middle candle was a strong up-move

      Bearish FVG:
        candle[0].low   >  candle[2].high
        → Gap exists BELOW candle[0] and ABOVE candle[2]
        → The middle candle was a strong down-move

      ICT says price often returns to FILL the FVG before
      continuing the original direction. Enter at the 50% level.
    """
    fvgs = []
    avg_range = (df["high"] - df["low"]).mean()

    for i in range(len(df) - 2):
        c1 = df.iloc[i]
        c3 = df.iloc[i + 2]

        # Bullish FVG
        if c1["high"] < c3["low"]:
            gap = c3["low"] - c1["high"]
            if gap > avg_range * 0.25:   # only significant gaps
                fvgs.append({
                    "type":   "BULL",
                    "top":     c3["low"],
                    "bottom":  c1["high"],
                    "mid":    (c3["low"] + c1["high"]) / 2,
                    "time":    df.index[i + 1],
                    "size":    gap,
                    "filled":  False,
                    "color_fill": "rgba(45,156,255,.08)",
                    "color_line": "#2d9cff",
                })

        # Bearish FVG
        elif c1["low"] > c3["high"]:
            gap = c1["low"] - c3["high"]
            if gap > avg_range * 0.25:
                fvgs.append({
                    "type":   "BEAR",
                    "top":     c1["low"],
                    "bottom":  c3["high"],
                    "mid":    (c1["low"] + c3["high"]) / 2,
                    "time":    df.index[i + 1],
                    "size":    gap,
                    "filled":  False,
                    "color_fill": "rgba(212,168,67,.08)",
                    "color_line": "#d4a843",
                })

    # Mark filled FVGs
    for fvg in fvgs:
        idx = df.index.get_loc(fvg["time"])
        future = df.iloc[idx + 2 :]
        if fvg["type"] == "BULL":
            fvg["filled"] = bool((future["low"] <= fvg["bottom"]).any())
        else:
            fvg["filled"] = bool((future["high"] >= fvg["top"]).any())

    return fvgs


# ──────────────────────────────────────────────────────────
#  STEP 6 — DETECT LIQUIDITY POOLS
# ──────────────────────────────────────────────────────────

def detect_liquidity(df: pd.DataFrame) -> dict:
    """
    Find Buy-Side and Sell-Side Liquidity pools.

    LEARNING NOTE:
      LIQUIDITY in ICT = areas where retail traders have stops.

      BSL (Buy-Side Liquidity):
        Clusters of SWING HIGHS above current price.
        Retail longs placed stops ABOVE swing highs.
        Smart money will HUNT these stops by pushing price up,
        triggering buys, then reversing downward.

      SSL (Sell-Side Liquidity):
        Clusters of SWING LOWS below current price.
        Retail shorts placed stops BELOW swing lows.
        Smart money hunts these before reversing upward.

      EQUAL HIGHS/LOWS are the most powerful liquidity:
        Two or more swing highs at the same price level
        = double stop cluster = very likely to be swept.
    """
    pip = 0.0001
    tol = 5 * pip

    sh_vals = df[df["sh"]]["high"]
    sl_vals = df[df["sl"]]["low"]
    price   = df["close"].iloc[-1]

    bsl = sh_vals[sh_vals > price].sort_values()
    ssl = sl_vals[sl_vals < price].sort_values(ascending=False)

    return {
        "current":      round(float(price), 5),
        "nearest_bsl":  round(float(bsl.iloc[0]), 5) if not bsl.empty else None,
        "nearest_ssl":  round(float(ssl.iloc[0]), 5) if not ssl.empty else None,
        "bsl_count":    len(bsl),
        "ssl_count":    len(ssl),
    }


# ──────────────────────────────────────────────────────────
#  STEP 7 — BUILD PLOTLY CHART
# ──────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, pair: str, obs: list, fvgs: list) -> go.Figure:
    """
    Build an interactive chart with all SMC annotations.

    LEARNING NOTE:
      Plotly makes browser-based interactive charts.
      - Hover over candles to see OHLC values
      - Scroll to zoom, drag to pan
      - Click legend items to toggle layers
      The chart is saved as a self-contained HTML file.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.8, 0.2],
        subplot_titles=[f"{pair} — SMC Analysis ({TIMEFRAME})", "Volume"],
    )

    # ── Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="Price",
            increasing_line_color="#00e5a0",
            decreasing_line_color="#ff4a6b",
            increasing_fillcolor="rgba(0,229,160,.6)",
            decreasing_fillcolor="rgba(255,74,107,.6)",
        ),
        row=1, col=1,
    )

    # ── Volume bars
    vol_colors = [
        "#00e5a0" if c >= o else "#ff4a6b"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df.get("volume", pd.Series([0]*len(df))),
               marker_color=vol_colors, opacity=0.4, name="Volume"),
        row=2, col=1,
    )

    # ── Swing Highs
    sh_df = df[df["sh"]]
    fig.add_trace(
        go.Scatter(
            x=sh_df.index, y=sh_df["high"] * 1.0003,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=9, color="#ff4a6b"),
            text=["SH"] * len(sh_df),
            textposition="top center",
            textfont=dict(size=8, color="#ff4a6b"),
            name="Swing High",
        ),
        row=1, col=1,
    )

    # ── Swing Lows
    sl_df = df[df["sl"]]
    fig.add_trace(
        go.Scatter(
            x=sl_df.index, y=sl_df["low"] * 0.9997,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=9, color="#00e5a0"),
            text=["SL"] * len(sl_df),
            textposition="bottom center",
            textfont=dict(size=8, color="#00e5a0"),
            name="Swing Low",
        ),
        row=1, col=1,
    )

    # ── BOS / CHoCH markers
    for evt, color in [
        ("BOS_BULL", "#00e5a0"), ("BOS_BEAR", "#ff4a6b"),
        ("CHOCH_BULL", "#2d9cff"), ("CHOCH_BEAR", "#d4a843"),
    ]:
        sub = df[df["structure"] == evt]
        if sub.empty:
            continue
        is_bull = "BULL" in evt
        y_vals = sub["high"] * 1.0004 if is_bull else sub["low"] * 0.9996
        fig.add_trace(
            go.Scatter(
                x=sub.index, y=y_vals,
                mode="markers+text",
                marker=dict(symbol="diamond", size=11, color=color),
                text=[evt.replace("_", " ")] * len(sub),
                textposition="top center" if is_bull else "bottom center",
                textfont=dict(size=8, color=color),
                name=evt.replace("_", " "),
            ),
            row=1, col=1,
        )

    # ── Order Block rectangles (last 6 unmitigated)
    active_obs = [o for o in obs if not o["mitigated"]][-6:]
    for ob in active_obs:
        fig.add_hrect(
            y0=ob["bottom"], y1=ob["top"],
            fillcolor=ob["color_fill"],
            line=dict(color=ob["color_line"], width=1, dash="dot"),
            annotation_text=f"{'Bull' if ob['type']=='BULL' else 'Bear'} OB",
            annotation_font=dict(size=8, color=ob["color_line"]),
            annotation_position="right",
            row=1, col=1,
        )

    # ── FVG rectangles (last 6 unfilled)
    active_fvgs = [f for f in fvgs if not f["filled"]][-6:]
    for fvg in active_fvgs:
        fig.add_hrect(
            y0=fvg["bottom"], y1=fvg["top"],
            fillcolor=fvg["color_fill"],
            line=dict(color=fvg["color_line"], width=1, dash="dash"),
            annotation_text=f"{'Bull' if fvg['type']=='BULL' else 'Bear'} FVG",
            annotation_font=dict(size=8, color=fvg["color_line"]),
            annotation_position="right",
            row=1, col=1,
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{pair} — ICT SMC Analysis</b>  "
                f"<span style='font-size:11px;color:#304a61'>"
                f"Active OBs: {len(active_obs)}  |  "
                f"Unfilled FVGs: {len(active_fvgs)}  |  "
                f"TF: {TIMEFRAME}</span>"
            ),
            font=dict(color="#d4a843", size=15),
        ),
        paper_bgcolor="#020810",
        plot_bgcolor="#06101a",
        font=dict(family="monospace", color="#ccd8e8", size=10),
        xaxis_rangeslider_visible=False,
        height=720,
        legend=dict(bgcolor="rgba(6,16,26,.85)", bordercolor="#0e2236"),
        margin=dict(l=60, r=140, t=80, b=40),
    )
    fig.update_xaxes(gridcolor="#0e2236", showgrid=True)
    fig.update_yaxes(gridcolor="#0e2236", showgrid=True)
    return fig


# ──────────────────────────────────────────────────────────
#  STEP 8 — PRINT TERMINAL SUMMARY
# ──────────────────────────────────────────────────────────

def print_summary(pair: str, df: pd.DataFrame, obs: list, fvgs: list):
    price   = df["close"].iloc[-1]
    trend   = df["trend"].iloc[-1]
    last_ev = df[df["structure"] != ""]["structure"].tail(1)
    last_ev = last_ev.iloc[0] if not last_ev.empty else "—"

    active_obs  = [o for o in obs  if not o["mitigated"]]
    active_fvgs = [f for f in fvgs if not f["filled"]]

    bull_obs  = [o for o in active_obs if o["type"] == "BULL" and o["top"]    < price]
    bear_obs  = [o for o in active_obs if o["type"] == "BEAR" and o["bottom"] > price]
    bull_fvgs = [f for f in active_fvgs if f["type"] == "BULL" and f["top"]   < price]
    bear_fvgs = [f for f in active_fvgs if f["type"] == "BEAR" and f["bottom"]> price]

    icon = "📈" if trend == "BULLISH" else ("📉" if trend == "BEARISH" else "➡️")
    print(f"\n  {'─'*52}")
    print(f"  {icon}  {pair:<10}  Price: {price:.5f}")
    print(f"  {'─'*52}")
    print(f"  Trend          : {trend}")
    print(f"  Last Event     : {last_ev}")
    print(f"  Bull OBs below : {len(bull_obs)}  (support zones)")
    print(f"  Bear OBs above : {len(bear_obs)}  (resistance zones)")
    print(f"  Bull FVGs below: {len(bull_fvgs)}  (fill magnets)")
    print(f"  Bear FVGs above: {len(bear_fvgs)}  (fill magnets)")

    # Nearest OB levels
    if bull_obs:
        print(f"  Nearest Support OB : {bull_obs[-1]['bottom']:.5f} – {bull_obs[-1]['top']:.5f}")
    if bear_obs:
        print(f"  Nearest Resist OB  : {bear_obs[0]['bottom']:.5f} – {bear_obs[0]['top']:.5f}")

    # Bias
    print()
    if trend == "BULLISH":
        print("  ✅ BIAS: Look for LONG setups at discount OBs/FVGs")
        print("  ✅ Best entry: OB + FVG confluence in London/NY KZ")
    elif trend == "BEARISH":
        print("  🔴 BIAS: Look for SHORT setups at premium OBs/FVGs")
        print("  🔴 Best entry: OB + FVG confluence in London/NY KZ")
    else:
        print("  ⚪ BIAS: No clear trend — reduce size or stay flat")


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  SMC SCREENER — SMART MONEY CONCEPT ANALYSIS")
    print(f"  Pairs: {list(PAIRS.keys())}  |  TF: {TIMEFRAME}")
    print("="*55)

    summaries = []

    for pair, symbol in PAIRS.items():
        # 1. Fetch
        df = fetch(symbol, LOOKBACK_DAYS, TIMEFRAME)
        if df.empty:
            continue

        # 2. Detect all SMC elements
        df = find_swings(df, n=SWING_N)
        df = detect_structure(df)
        obs  = detect_obs(df, mult=OB_MULT)
        fvgs = detect_fvgs(df)
        liq  = detect_liquidity(df)

        # 3. Terminal summary
        print_summary(pair, df, obs, fvgs)

        # 4. Interactive chart
        print(f"\n  Building chart for {pair}...", end=" ")
        fig  = build_chart(df, pair, obs, fvgs)
        fname = f"smc_{pair.replace('/','')}.html"
        fig.write_html(fname)
        print(f"saved → {fname}")

        # 5. Collect for CSV
        price  = df["close"].iloc[-1]
        active_obs  = [o for o in obs  if not o["mitigated"]]
        active_fvgs = [f for f in fvgs if not f["filled"]]
        summaries.append({
            "pair":         pair,
            "price":        round(price, 5),
            "trend":        df["trend"].iloc[-1],
            "last_event":   df[df["structure"] != ""]["structure"].tail(1).values[0]
                            if (df["structure"] != "").any() else "",
            "active_obs":   len(active_obs),
            "active_fvgs":  len(active_fvgs),
            "nearest_bsl":  liq["nearest_bsl"],
            "nearest_ssl":  liq["nearest_ssl"],
        })

    # Save screening results
    if summaries:
        out = pd.DataFrame(summaries)
        out.to_csv("smc_results.csv", index=False)
        print("\n" + "="*55)
        print("  SCREENING SUMMARY")
        print("="*55)
        print(out.to_string(index=False))
        print("\n  ✓ Saved: smc_results.csv")

    print("\n" + "="*55)
    print("  KEY TAKEAWAYS:")
    print("  OB  = institutional order zone — high-prob entry area")
    print("  FVG = price imbalance — price likes to fill these")
    print("  BOS = trend confirmed — trade in BOS direction")
    print("  CHoCH = FIRST reversal signal — wait for confirmation")
    print("  Always combine: OB + FVG + Killzone = highest prob")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()