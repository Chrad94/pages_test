#!/usr/bin/env python3
"""
Interactive private-safe trading dashboard (Plotly HTML + metrics.json + index.html).

Option B:
- Keep raw CSV/TXT private (not committed)
- Publish only /docs/* to GitHub Pages

Inputs (in --input-dir):
- portfolio.csv          (required)
- trade_stats.csv        (optional)
- macro_means_log.csv    (optional)

Outputs (in --output-dir):
- equity.html
- drawdown.html
- exposure.html
- leverage.html
- positions.html         (nr_longs / nr_shorts)
- turnover.html          (from trade_stats.csv)
- fees.html              (optional, if trade_stats has turnover[%])
- holdings.html          (current holdings histogram)
- metrics.json
- index.html

Features:
- Charts are fully interactive (hover, zoom, pan).
- All time-series charts default to last 7 days (you can zoom out).
- KPIs show 7d return, 7d drawdown, latest market exposure, etc.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import ast

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


FEE_RATE_NOTIONAL = 0.00043  # 0.043% per trade notionally


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _detect_step_jumps(equity: pd.Series, z: float = 8.0) -> list[int]:
    if len(equity) < 10:
        return []
    d = equity.diff().dropna()
    mad = np.median(np.abs(d - np.median(d)))
    if mad == 0:
        mad = np.median(np.abs(d)) + 1e-9
    thresh = z * mad
    return list(d.index[np.abs(d) > thresh])


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _write_plotly_html(fig: go.Figure, outpath: Path, title: str, x_range=None) -> None:
    # Use Plotly CDN (small files; great for GitHub Pages)
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    fig.write_html(
        outpath,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,   # zoom with mousewheel/trackpad
            "responsive": True,
        },
    )


def _compute_7d_range(df_time: pd.Series):
    """Return (start, end) for last 7 days, or None if not enough data."""
    if len(df_time) == 0:
        return None
    end = df_time.iloc[-1]
    start_candidate = end - pd.Timedelta(days=7)
    start = max(start_candidate, df_time.iloc[0])
    if start >= end:
        return None
    return (start, end)


def build_dashboard(input_dir: Path, output_dir: Path, fee_rate: float = FEE_RATE_NOTIONAL) -> dict:
    portfolio_path = input_dir / "portfolio.csv"
    if not portfolio_path.exists():
        raise FileNotFoundError(f"Missing required file: {portfolio_path}")

    pf = pd.read_csv(portfolio_path)
    if "timestamp" not in pf.columns or "total_value_usd" not in pf.columns:
        raise ValueError("portfolio.csv must include columns: timestamp, total_value_usd")

    pf["timestamp"] = _parse_datetime_series(pf["timestamp"])
    pf["total_value_usd"] = pd.to_numeric(pf["total_value_usd"], errors="coerce")
    pf = pf.dropna(subset=["timestamp", "total_value_usd"]).sort_values("timestamp").reset_index(drop=True)

    equity = pf["total_value_usd"]
    dd_full = _compute_drawdown(equity)
    jumps = _detect_step_jumps(equity)
    x_range_7d = _compute_7d_range(pf["timestamp"])

    # --- Equity ---
    fig = px.line(pf, x="timestamp", y="total_value_usd")
    if jumps:
        for j in jumps:
            try:
                t = pf.loc[j, "timestamp"]
                fig.add_vline(x=t, line_dash="dash")
            except Exception:
                pass
        fig.add_annotation(
            xref="paper", yref="paper", x=0, y=0,
            text="Dashed lines = detected step jumps (e.g., deposits / accounting changes)",
            showarrow=False, font=dict(size=11),
        )
    _write_plotly_html(fig, output_dir / "equity.html", "Equity Curve (total_value_usd)", x_range=x_range_7d)

    # --- Drawdown ---
    dd_df = pd.DataFrame({"timestamp": pf["timestamp"], "drawdown": dd_full})
    fig = px.line(dd_df, x="timestamp", y="drawdown")
    fig.update_yaxes(tickformat=".0%")
    _write_plotly_html(fig, output_dir / "drawdown.html", "Drawdown", x_range=x_range_7d)

    # --- Net Market Exposure ---
    last_market_exposure = None
    if "market_exposure" in pf.columns:
        pf["market_exposure"] = pd.to_numeric(pf["market_exposure"], errors="coerce")
        last_market_exposure = _safe_float(pf["market_exposure"].iloc[-1])
        fig = px.line(pf, x="timestamp", y="market_exposure")
        _write_plotly_html(fig, output_dir / "exposure.html", "Net Market Exposure (market_exposure)", x_range=x_range_7d)
    else:
        fig = go.Figure()
        fig.add_annotation(text="Missing 'market_exposure' column in portfolio.csv", showarrow=False)
        _write_plotly_html(fig, output_dir / "exposure.html", "Net Market Exposure")

    # --- Leverage ---
    if "leverage" in pf.columns:
        pf["leverage"] = pd.to_numeric(pf["leverage"], errors="coerce")
        fig = px.line(pf, x="timestamp", y="leverage")
        _write_plotly_html(fig, output_dir / "leverage.html", "Leverage", x_range=x_range_7d)
    else:
        fig = go.Figure()
        fig.add_annotation(text="Missing 'leverage' column in portfolio.csv", showarrow=False)
        _write_plotly_html(fig, output_dir / "leverage.html", "Leverage")

    # --- Nr longs & shorts (positions.html) ---
    if "nr_longs" in pf.columns or "nr_shorts" in pf.columns:
        fig = go.Figure()
        if "nr_longs" in pf.columns:
            pf["nr_longs"] = pd.to_numeric(pf["nr_longs"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=pf["timestamp"],
                    y=pf["nr_longs"],
                    mode="lines",
                    name="Longs",
                )
            )
        if "nr_shorts" in pf.columns:
            pf["nr_shorts"] = pd.to_numeric(pf["nr_shorts"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=pf["timestamp"],
                    y=pf["nr_shorts"],
                    mode="lines",
                    name="Shorts",
                )
            )
        _write_plotly_html(fig, output_dir / "positions.html", "Number of Longs & Shorts", x_range=x_range_7d)

    # --- Current holdings as histogram (holdings.html) ---
    # Uses last row's usd_value dict-like column if available
    if "usd_value" in pf.columns:
        try:
            raw = pf["usd_value"].iloc[-1]
            # Usually stored as a Python dict string: "{'BTC': 100, 'ETH': -50, ...}"
            holdings_dict = ast.literal_eval(raw)
            if isinstance(holdings_dict, dict) and holdings_dict:
                hold_df = pd.DataFrame(
                    {
                        "symbol": list(holdings_dict.keys()),
                        "usd_value": [float(v) for v in holdings_dict.values()],
                    }
                )
                # Sort by absolute usd value descending so largest exposures are prominent
                hold_df["abs_usd"] = hold_df["usd_value"].abs()
                hold_df = hold_df.sort_values("abs_usd", ascending=False)
                fig = px.bar(hold_df, x="symbol", y="usd_value")
                fig.update_layout(yaxis_title="USD value")
                _write_plotly_html(fig, output_dir / "holdings.html", "Current Holdings (USD)")
        except Exception as e:
            # If parsing fails, skip holdings chart silently
            print("Could not parse usd_value for holdings histogram:", e)

    # --- Optional trade_stats-derived fee & turnover charts ---
    ts = _read_csv_if_exists(input_dir / "trade_stats.csv")
    fee_est = {}
    fees_written = False
    turnover_written = False
    if ts is not None and "timestamp" in ts.columns:
        ts["timestamp"] = _parse_datetime_series(ts["timestamp"])
        ts = ts.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        x_range_7d_ts = _compute_7d_range(ts["timestamp"])

        # Turnover chart (turnover.html)
        if "turnover[%]" in ts.columns:
            turnover = pd.to_numeric(ts["turnover[%]"], errors="coerce")
            ts["turnover[%]"] = turnover
            # Turnover time-series
            turn_df = ts[["timestamp", "turnover[%]"]].dropna()
            if len(turn_df) > 0:
                fig = px.bar(turn_df, x="timestamp", y="turnover[%]")
                fig.update_yaxes(title_text="Turnover (%)")
                _write_plotly_html(
                    fig,
                    output_dir / "turnover.html",
                    "Turnover per Session (%)",
                    x_range=x_range_7d_ts,
                )
                turnover_written = True

            # Fee estimates
            ts["turnover_ratio"] = turnover / 100.0
            ts["fee_est_pct_capital"] = fee_rate * ts["turnover_ratio"] * 100.0

            fee_est = {
                "fee_rate_notional": fee_rate,
                "avg_turnover_pct": float(np.nanmean(turnover)),
                "median_turnover_pct": float(np.nanmedian(turnover)),
                "avg_fee_est_pct_capital": float(np.nanmean(ts["fee_est_pct_capital"])),
                "median_fee_est_pct_capital": float(np.nanmedian(ts["fee_est_pct_capital"])),
            }

            fee_df = ts[["timestamp", "fee_est_pct_capital"]].dropna()
            if len(fee_df) > 1:
                fig = px.bar(fee_df, x="timestamp", y="fee_est_pct_capital")
                fig.update_yaxes(title_text="Estimated fees (% of capital)")
                _write_plotly_html(
                    fig,
                    output_dir / "fees.html",
                    "Estimated Fees per Session (% of capital)",
                    x_range=x_range_7d_ts,
                )
                fees_written = True

        if "trade_sess_capital_change[%]" in ts.columns:
            sess = pd.to_numeric(ts["trade_sess_capital_change[%]"], errors="coerce")
            fee_est.update(
                {
                    "avg_session_return_pct": float(np.nanmean(sess)),
                    "median_session_return_pct": float(np.nanmedian(sess)),
                    "worst_session_return_pct": float(np.nanmin(sess)),
                    "best_session_return_pct": float(np.nanmax(sess)),
                }
            )

    # --- Macro snapshot (optional) ---
    macro = _read_csv_if_exists(input_dir / "macro_means_log.csv")
    macro_summary = {}
    if macro is not None and "Timestamp" in macro.columns:
        macro["Timestamp"] = _parse_datetime_series(macro["Timestamp"])
        macro = macro.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        last = macro.iloc[-1].to_dict()
        keep = [
            "ROC4h", "ROC24h", "ROC7", "ROC14", "ROC30",
            "Stoch4h", "Stoch24h", "Stoch20", "Price", "Volume",
        ]
        macro_summary = {k: _safe_float(last.get(k)) for k in keep if k in last}

    # --- Metrics ---
    last_equity = float(equity.iloc[-1])
    first_equity = float(equity.iloc[0])

    def window_return(days: int) -> float | None:
        if len(pf) < 2:
            return None
        cutoff = pf["timestamp"].iloc[-1] - pd.Timedelta(days=days)
        sub = pf[pf["timestamp"] >= cutoff]
        if len(sub) < 2:
            return None
        return float(sub["total_value_usd"].iloc[-1] / sub["total_value_usd"].iloc[0] - 1.0)

    # 7d max drawdown (based on last 7 days)
    max_dd_7d_pct = None
    if x_range_7d is not None:
        start_7d, end_7d = x_range_7d
        mask_7d = (pf["timestamp"] >= start_7d) & (pf["timestamp"] <= end_7d)
        sub_eq = equity[mask_7d]
        if len(sub_eq) >= 2:
            dd_7d = _compute_drawdown(sub_eq)
            max_dd_7d_pct = float(dd_7d.min() * 100.0)

    gross_exposure_x = None
    if "totalNtlPos" in pf.columns:
        total_ntl = pd.to_numeric(pf["totalNtlPos"], errors="coerce")
        gross_exposure_x = float(np.nanmean(total_ntl / equity))

    avg_leverage = float(np.nanmean(pf["leverage"])) if "leverage" in pf.columns else None
    avg_abs_market_exposure = float(np.nanmean(np.abs(pf["market_exposure"]))) if "market_exposure" in pf.columns else None

    metrics = {
        "last_updated_utc": _utc_now_iso(),
        "data_time_start": pf["timestamp"].iloc[0].isoformat(),
        "data_time_end": pf["timestamp"].iloc[-1].isoformat(),
        "equity_start_usd": first_equity,
        "equity_last_usd": last_equity,
        "return_total_pct": float((last_equity / first_equity - 1.0) * 100.0),
        "return_7d_pct": None if window_return(7) is None else float(window_return(7) * 100.0),
        "return_30d_pct": None if window_return(30) is None else float(window_return(30) * 100.0),
        "max_drawdown_pct": float(dd_full.min() * 100.0),
        "max_drawdown_7d_pct": max_dd_7d_pct,
        "avg_leverage_x": avg_leverage,
        "avg_gross_exposure_x": gross_exposure_x,
        "avg_abs_market_exposure": avg_abs_market_exposure,
        "last_market_exposure": last_market_exposure,
        "last_nr_longs": int(pf["nr_longs"].iloc[-1]) if "nr_longs" in pf.columns else None,
        "last_nr_shorts": int(pf["nr_shorts"].iloc[-1]) if "nr_shorts" in pf.columns else None,
        "detected_step_jumps_count": int(len(jumps)),
        "fee_and_session_stats": fee_est,
        "macro_snapshot": macro_summary,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # --- index.html ---
    fees_block = ""
    if fees_written:
        fees_block = """
    <div class="card">
      <h3>Estimated Fees per Session</h3>
      <iframe src="fees.html"></iframe>
    </div>
"""

    turnover_block = ""
    if turnover_written:
        turnover_block = """
    <div class="card">
      <h3>Turnover per Session</h3>
      <iframe src="turnover.html"></iframe>
    </div>
"""

    positions_block = ""
    if (output_dir / "positions.html").exists():
        positions_block = """
    <div class="card">
      <h3>Number of Longs & Shorts</h3>
      <iframe src="positions.html"></iframe>
    </div>
"""

    holdings_block = ""
    if (output_dir / "holdings.html").exists():
        holdings_block = """
    <div class="card">
      <h3>Current Holdings (USD)</h3>
      <iframe src="holdings.html"></iframe>
    </div>
"""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trading Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 0; }}
    .sub {{ color: #555; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; margin-top: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; }}
    iframe {{ width: 100%; height: 440px; border: 1px solid #eee; border-radius: 8px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .kpi div {{ padding: 10px; border: 1px solid #eee; border-radius: 8px; }}
    .small {{ font-size: 12px; color: #666; }}
    @media (max-width: 600px) {{
      iframe {{ height: 360px; }}
    }}
  </style>
</head>
<body>
  <h1>Trading Dashboard</h1>
  <div class="sub">Interactive charts (hover/zoom/pan). Default view: last 7 days. Only aggregates are published—no raw trades/positions.</div>

  <div class="card" style="margin-top:16px;">
    <div class="kpi" id="kpi"></div>
    <div class="small" id="updated"></div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Equity</h3>
      <iframe src="equity.html"></iframe>
    </div>
    <div class="card">
      <h3>Drawdown</h3>
      <iframe src="drawdown.html"></iframe>
    </div>
    <div class="card">
      <h3>Net Market Exposure</h3>
      <iframe src="exposure.html"></iframe>
    </div>
    <div class="card">
      <h3>Leverage</h3>
      <iframe src="leverage.html"></iframe>
    </div>
    {positions_block}
    {holdings_block}
    {turnover_block}
    {fees_block}
  </div>

  <div class="card" style="margin-top:18px;">
    <h3>metrics.json</h3>
    <pre id="metrics"></pre>
  </div>

<script>
async function main() {{
  try {{
    const r = await fetch('metrics.json', {{ cache: 'no-store' }});
    const m = await r.json();
    document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);

    const fmt = (x, suf='') => (x === null || x === undefined || Number.isNaN(x)) ? '—' : (typeof x === 'number' ? x.toFixed(2) : x) + suf;

    const kpis = [
      ['Equity (last)', fmt(m.equity_last_usd, ' USD')],
      ['7D return', fmt(m.return_7d_pct, ' %')],
      ['7D max drawdown', fmt(m.max_drawdown_7d_pct, ' %')],
      ['Last market exposure', fmt(m.last_market_exposure, '')],
      ['Max drawdown (all)', fmt(m.max_drawdown_pct, ' %')],
      ['Avg leverage', fmt(m.avg_leverage_x, 'x')],
      ['Avg gross exposure', fmt(m.avg_gross_exposure_x, 'x')],
    ];

    const kpiEl = document.getElementById('kpi');
    kpiEl.innerHTML = '';
    for (const [k,v] of kpis) {{
      const d = document.createElement('div');
      d.innerHTML = `<div class="small">${{k}}</div><div style="font-size:18px; font-weight:600;">${{v}}</div>`;
      kpiEl.appendChild(d);
    }}
    document.getElementById('updated').textContent =
      `Data range: ${{m.data_time_start}} → ${{m.data_time_end}} | Last generated: ${{m.last_updated_utc}}`;
  }} catch (e) {{
    document.getElementById('metrics').textContent = 'Failed to load metrics.json: ' + e;
  }}
}}
main();
</script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")

    return metrics


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default="raw", help="Directory containing your private CSV/TXT files")
    ap.add_argument("--output-dir", type=str, default="docs", help="Directory where public dashboard files are written")
    ap.add_argument("--fee-rate", type=float, default=FEE_RATE_NOTIONAL, help="Fee rate (decimal, e.g. 0.00043 for 0.043%)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(output_dir)

    metrics = build_dashboard(input_dir=input_dir, output_dir=output_dir, fee_rate=float(args.fee_rate))
    print("✅ Interactive dashboard generated in:", output_dir)
    print("✅ Last equity:", metrics.get("equity_last_usd"))
    return metrics


if __name__ == "__main__":
    main()
