import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from io import BytesIO

try:
    from openpyxl.utils import get_column_letter
except ImportError:
    get_column_letter = None

st.set_page_config(page_title="DCF Valuation App", page_icon="📊", layout="wide")

st.title("📊 DCF Valuation App")
st.caption("Teaching focused DCF model with transparent assumptions, forecast detail, and intrinsic value comparison.")

def money(x, decimals=2):
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.{decimals}f}"

def pct(x, decimals=2):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def bounded(v, low, high):
    return max(low, min(high, v))


def range_text(low, mid, high):
    return f"{pct(low, 1)} to {pct(high, 1)} (base case around {pct(mid, 1)})"

def _scalar_missing_or_nan(x):
    """True if x is None, missing/NaN scalar, or non-scalar (list/ndarray), so it is not a single usable number."""
    if x is None:
        return True
    if not pd.api.types.is_scalar(x):
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return True

def safe_float(x, default):
    """Parse yfinance / form values that may be missing, NaN, or oddly typed."""
    if _scalar_missing_or_nan(x):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def _is_rate_limit_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return "too many requests" in msg or "rate limit" in msg or "429" in msg

# yfinance hits Yahoo many times per ticker (info, history, statements). Cache aggressively
# and avoid refetching on every widget interaction (see sidebar load flow).
@st.cache_data(show_spinner=False, ttl=3600)
def load_ticker_data(ticker):
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    hist = tk.history(period="6mo")
    financials = tk.financials
    if financials is None or getattr(financials, "empty", True):
        financials = getattr(tk, "income_stmt", None)
    if financials is None:
        financials = pd.DataFrame()

    cashflow = tk.cashflow
    if cashflow is None or getattr(cashflow, "empty", True):
        cashflow = getattr(tk, "quarterly_cashflow", None)
    if cashflow is None:
        cashflow = pd.DataFrame()

    current_price = None
    if hist is not None and not hist.empty:
        current_price = float(hist["Close"].dropna().iloc[-1])

    def get_first_value(df, labels):
        if df is None or df.empty:
            return None
        for label in labels:
            exact = [idx for idx in df.index if str(idx).strip().lower() == label.lower()]
            if exact:
                vals = pd.to_numeric(df.loc[exact[0]], errors="coerce").dropna()
                if not vals.empty:
                    return float(vals.iloc[0])
        for label in labels:
            for idx in df.index:
                if label.lower() in str(idx).lower():
                    vals = pd.to_numeric(df.loc[idx], errors="coerce").dropna()
                    if not vals.empty:
                        return float(vals.iloc[0])
        return None

    revenue = get_first_value(financials, ["Total Revenue", "Revenue", "Operating Revenue"])
    ebit = get_first_value(financials, ["EBIT", "Operating Income"])
    dep = get_first_value(cashflow, ["Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion"])
    capex = get_first_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    wc_change = get_first_value(cashflow, ["Change In Working Capital", "Changes In Working Capital"])

    tax_rate = None
    tax_provision = get_first_value(financials, ["Tax Provision", "Income Tax Expense"])
    pretax = get_first_value(financials, ["Pretax Income", "Pretax Income Loss"])
    if tax_provision is not None and pretax not in [None, 0]:
        implied_tax = tax_provision / pretax
        if not pd.isna(implied_tax):
            tax_rate = min(max(implied_tax, 0.0), 0.40)

    hist_revenue_cagr = None
    hist_revenue_avg_growth = None
    hist_ebit_margin = None
    if financials is not None and not financials.empty:
        rev_row = None
        for label in ["Total Revenue", "Revenue", "Operating Revenue"]:
            exact = [idx for idx in financials.index if str(idx).strip().lower() == label.lower()]
            if exact:
                rev_row = financials.loc[exact[0]]
                break
        if rev_row is not None:
            rev_series = pd.to_numeric(rev_row, errors="coerce").dropna()
            if len(rev_series) >= 2:
                rev_series = rev_series.sort_index()
                first_rev = float(rev_series.iloc[0])
                last_rev = float(rev_series.iloc[-1])
                periods = len(rev_series) - 1
                if first_rev > 0 and periods > 0:
                    hist_revenue_cagr = (last_rev / first_rev) ** (1 / periods) - 1
                growths = rev_series.pct_change().dropna()
                if not growths.empty:
                    hist_revenue_avg_growth = float(growths.mean())

        ebit_row = None
        for label in ["EBIT", "Operating Income"]:
            exact = [idx for idx in financials.index if str(idx).strip().lower() == label.lower()]
            if exact:
                ebit_row = financials.loc[exact[0]]
                break
        if rev_row is not None and ebit_row is not None:
            rev_series = pd.to_numeric(rev_row, errors="coerce")
            ebit_series = pd.to_numeric(ebit_row, errors="coerce")
            merged = pd.concat([rev_series, ebit_series], axis=1).dropna()
            if not merged.empty:
                margins = merged.iloc[:, 1] / merged.iloc[:, 0].replace(0, np.nan)
                margins = margins.replace([np.inf, -np.inf], np.nan).dropna()
                if not margins.empty:
                    hist_ebit_margin = float(margins.median())

    return {
        "name": info.get("longName", ticker.upper()),
        "ticker": ticker.upper(),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "price": current_price,
        "shares": info.get("sharesOutstanding"),
        "market_cap": info.get("marketCap"),
        "debt": info.get("totalDebt"),
        "cash": info.get("totalCash"),
        "beta": info.get("beta"),
        "revenue": revenue,
        "ebit": ebit,
        "dep": dep,
        "capex": capex,
        "wc_change": wc_change,
        "tax_rate": tax_rate,
        "revenue_growth_hint": info.get("revenueGrowth"),
        "hist_revenue_cagr": hist_revenue_cagr,
        "hist_revenue_avg_growth": hist_revenue_avg_growth,
        "hist_ebit_margin": hist_ebit_margin,
    }

def build_projection(revenue, growth_rates, margin, tax_rate, reinvest_rate):
    rows = []
    rev = revenue

    for year, growth in enumerate(growth_rates, start=1):
        rev = rev * (1 + growth)
        ebit = rev * margin
        nopat = ebit * (1 - tax_rate)
        reinvestment = nopat * reinvest_rate
        fcf = nopat - reinvestment

        rows.append({
            "Year": year,
            "Revenue": rev,
            "Growth Rate": growth,
            "EBIT": ebit,
            "EBIT Margin": margin,
            "NOPAT": nopat,
            "Reinvestment": reinvestment,
            "FCF": fcf,
        })

    return pd.DataFrame(rows)

def discount_valuation(df, wacc, terminal_growth, debt, cash, shares):
    out = df.copy()
    out["Discount Factor"] = [1 / ((1 + wacc) ** yr) for yr in out["Year"]]
    out["PV of FCF"] = out["FCF"] * out["Discount Factor"]

    terminal_fcf = float(out.iloc[-1]["FCF"])
    terminal_value = terminal_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** int(out.iloc[-1]["Year"]))

    enterprise_value = out["PV of FCF"].sum() + pv_terminal
    equity_value = enterprise_value - debt + cash
    value_per_share = equity_value / shares if shares not in [0, None] else np.nan

    return out, terminal_value, pv_terminal, enterprise_value, equity_value, value_per_share

def _autosize_excel_columns(writer):
    """Widen columns slightly beyond header/data length so sheets are easier to scan."""
    if get_column_letter is None:
        return
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        for idx, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200), values_only=False), start=1):
            maxlen = 0
            for cell in col:
                if cell.value is None:
                    continue
                maxlen = max(maxlen, len(str(cell.value)))
            letter = get_column_letter(idx)
            ws.column_dimensions[letter].width = min(max(maxlen + 2, 10), 48)


def build_dcf_excel_bytes(
    ticker,
    ticker_data,
    revenue,
    years,
    growth_rates,
    margin,
    tax_rate,
    reinvest,
    wacc,
    terminal_growth,
    debt,
    cash,
    shares,
    discounted_df,
    enterprise_value,
    equity_value,
    value_per_share,
    market_price,
    terminal_value,
    pv_terminal,
    sensitivity_df,
):
    """Multi-sheet workbook: overview, assumptions, forecast, walkthrough, sensitivity."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    company = ticker_data["name"] if ticker_data else ""
    sector = ticker_data["sector"] if ticker_data else ""
    industry = ticker_data["industry"] if ticker_data else ""

    upside_vs_market = np.nan
    if market_price not in [None, 0] and not pd.isna(value_per_share):
        upside_vs_market = value_per_share / market_price - 1

    summary_rows = [
        ("Document type", "DCF valuation workbook (formula-based)"),
        ("Generated", generated),
        ("Company", company),
        ("Ticker", ticker or ""),
        ("Sector", sector),
        ("Industry", industry),
        (None, None),
        ("Key results", None),
        ("Enterprise value ($)", None),
        ("Equity value ($)", None),
        ("Intrinsic value per share ($)", None),
        ("Market price ($)", market_price if market_price is not None else np.nan),
        ("Upside vs market (ratio vs price)", None),
        (None, None),
        ("Valuation bridge ($)", None),
        ("PV of forecast period FCF", None),
        ("PV of terminal value", None),
        ("Enterprise value", None),
        ("Less: debt", None),
        ("Plus: cash", None),
        ("Equity value", None),
        ("Shares outstanding (count)", None),
        ("Intrinsic value per share", None),
        ("Terminal value (undiscounted, exit year)", None),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Description", "Value"])

    assumption_records = [
        ("Ticker", ticker, "Company symbol"),
        ("Projection years", years, "Explicit forecast horizon"),
        ("Current revenue ($)", revenue, "Starting revenue base"),
    ]
    for i in range(len(growth_rates)):
        assumption_records.append(
            (f"Year {i + 1} revenue growth", growth_rates[i], "Annual revenue growth assumption"),
        )
    assumption_records.extend([
        ("EBIT margin", margin, "Operating margin on revenue"),
        ("Tax rate", tax_rate, "Corporate tax on EBIT"),
        ("Reinvestment rate", reinvest, "NOPAT reinvested"),
        ("WACC", wacc, "Discount rate"),
        ("Terminal growth rate", terminal_growth, "Perpetuity growth"),
        ("Debt ($)", debt, "Auto-loaded from ticker data (editable in Excel)"),
        ("Cash ($)", cash, "Auto-loaded from ticker data (editable in Excel)"),
        ("Shares outstanding", shares, "Auto-loaded from ticker data (editable in Excel)"),
    ])
    assumptions_df = pd.DataFrame(assumption_records, columns=["Assumption", "Value", "Notes"])

    forecast_cols = [
        "Year",
        "Revenue",
        "Growth Rate",
        "EBIT",
        "EBIT Margin",
        "NOPAT",
        "Reinvestment",
        "FCF",
        "Discount Factor",
        "PV of FCF",
    ]
    forecast_df = discounted_df[[c for c in forecast_cols if c in discounted_df.columns]].copy()
    walkthrough_df = discounted_df[["Year", "FCF", "Discount Factor", "PV of FCF"]].copy()

    sens_export = sensitivity_df.copy()
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        forecast_df.to_excel(writer, sheet_name="Forecast_DCF", index=False)
        walkthrough_df.to_excel(writer, sheet_name="DCF_walkthrough", index=False)
        sens_export.to_excel(writer, sheet_name="Sensitivity", index=True)

        ws_assump = writer.book["Assumptions"]
        ws_fore = writer.book["Forecast_DCF"]
        ws_sum = writer.book["Summary"]

        row_map = {}
        for r in range(2, ws_assump.max_row + 1):
            key = ws_assump[f"A{r}"].value
            if key is not None:
                row_map[str(key)] = r

        years_row = row_map["Projection years"]
        rev_row = row_map["Current revenue ($)"]
        margin_row = row_map["EBIT margin"]
        tax_row = row_map["Tax rate"]
        reinvest_row = row_map["Reinvestment rate"]
        wacc_row = row_map["WACC"]
        tg_row = row_map["Terminal growth rate"]
        debt_row = row_map["Debt ($)"]
        cash_row = row_map["Cash ($)"]
        shares_row = row_map["Shares outstanding"]

        first_data_row = 2
        last_data_row = first_data_row + years - 1
        for r in range(first_data_row, last_data_row + 1):
            year_idx = r - 1
            ws_fore[f"A{r}"] = year_idx
            ws_fore[f"C{r}"] = f"=INDEX(Assumptions!$B:$B,{row_map['Year 1 revenue growth']}+A{r}-1)"
            if r == first_data_row:
                ws_fore[f"B{r}"] = f"=Assumptions!$B${rev_row}*(1+C{r})"
            else:
                ws_fore[f"B{r}"] = f"=B{r-1}*(1+C{r})"
            ws_fore[f"D{r}"] = f"=B{r}*Assumptions!$B${margin_row}"
            ws_fore[f"E{r}"] = f"=Assumptions!$B${margin_row}"
            ws_fore[f"F{r}"] = f"=D{r}*(1-Assumptions!$B${tax_row})"
            ws_fore[f"G{r}"] = f"=F{r}*Assumptions!$B${reinvest_row}"
            ws_fore[f"H{r}"] = f"=F{r}-G{r}"
            ws_fore[f"I{r}"] = f"=1/(1+Assumptions!$B${wacc_row})^A{r}"
            ws_fore[f"J{r}"] = f"=H{r}*I{r}"

        ws_sum["B10"] = "=B19"
        ws_sum["B11"] = "=B22"
        ws_sum["B12"] = "=B24"
        ws_sum["B14"] = "=IF(B13=0,NA(),B12/B13-1)"
        ws_sum["B17"] = f"=SUM(Forecast_DCF!J{first_data_row}:J{last_data_row})"
        ws_sum["B25"] = f"=Forecast_DCF!H{last_data_row}*(1+Assumptions!$B${tg_row})/(Assumptions!$B${wacc_row}-Assumptions!$B${tg_row})"
        ws_sum["B18"] = f"=B25/(1+Assumptions!$B${wacc_row})^Forecast_DCF!A{last_data_row}"
        ws_sum["B19"] = "=B17+B18"
        ws_sum["B20"] = f"=-Assumptions!$B${debt_row}"
        ws_sum["B21"] = f"=Assumptions!$B${cash_row}"
        ws_sum["B22"] = "=B19+B20+B21"
        ws_sum["B23"] = f"=Assumptions!$B${shares_row}"
        ws_sum["B24"] = "=B22/B23"

        ws_walk = writer.book["DCF_walkthrough"]
        for r in range(first_data_row, last_data_row + 1):
            ws_walk[f"A{r}"] = f"=Forecast_DCF!A{r}"
            ws_walk[f"B{r}"] = f"=Forecast_DCF!H{r}"
            ws_walk[f"C{r}"] = f"=Forecast_DCF!I{r}"
            ws_walk[f"D{r}"] = f"=Forecast_DCF!J{r}"

        _autosize_excel_columns(writer)
    buffer.seek(0)
    return buffer.getvalue()


def sensitivity_table(revenue, growth_rates, margin, tax_rate, reinvest_rate, debt, cash, shares, base_wacc, base_tg):
    waccs = [max(0.01, round(base_wacc + x, 4)) for x in [-0.02, -0.01, 0.00, 0.01, 0.02]]
    tgs = [max(0.00, round(base_tg + x, 4)) for x in [-0.01, -0.005, 0.00, 0.005, 0.01]]

    table = pd.DataFrame(index=[f"{tg*100:.1f}%" for tg in tgs])

    for w in waccs:
        vals = []
        for tg in tgs:
            if w <= tg:
                vals.append(np.nan)
                continue
            proj = build_projection(revenue, growth_rates, margin, tax_rate, reinvest_rate)
            _, _, _, _, _, vps = discount_valuation(proj, w, tg, debt, cash, shares)
            vals.append(vps)
        table[f"{w*100:.1f}%"] = vals

    table.index.name = "Terminal Growth"
    return table

with st.expander("What this app does", expanded=True):
    st.write(
        """
        This app starts with your simple DCF structure and improves it by adding:

        • ticker based company lookup  
        • automatic market price retrieval  
        • editable DCF assumptions  
        • full forecast breakdown  
        • intrinsic value versus current price comparison  
        • sensitivity analysis  
        • exportable model output

        Core model used here:

        EBIT = Revenue × Margin  
        NOPAT = EBIT × (1 − Tax Rate)  
        Reinvestment = NOPAT × Reinvestment Rate  
        FCF = NOPAT − Reinvestment  
        Enterprise Value = PV of Forecast FCF + PV of Terminal Value
        """
    )

st.sidebar.header("Inputs")

if "dcf_loaded_ticker" not in st.session_state:
    st.session_state.dcf_loaded_ticker = None
if "dcf_ticker_data" not in st.session_state:
    st.session_state.dcf_ticker_data = None

ticker_input = st.sidebar.text_input(
    "Ticker",
    value="AAPL",
    help="Enter a symbol, then click Load. Each load calls Yahoo several times; wait if you see rate limits.",
).strip().upper()
load_clicked = st.sidebar.button("Load ticker data", use_container_width=True)

ticker = ticker_input if ticker_input else "AAPL"

# Fetch only on first visit (warm defaults) or when the user explicitly loads — not on every rerun/slider move.
need_fetch = load_clicked or (st.session_state.dcf_loaded_ticker is None)

ticker_data = None
if need_fetch and ticker:
    with st.spinner("Loading market data from Yahoo…"):
        try:
            ticker_data = load_ticker_data(ticker)
            st.session_state.dcf_ticker_data = ticker_data
            st.session_state.dcf_loaded_ticker = ticker
        except Exception as e:
            if _is_rate_limit_error(e) and st.session_state.dcf_ticker_data is not None \
                    and st.session_state.dcf_loaded_ticker == ticker:
                ticker_data = st.session_state.dcf_ticker_data
                st.sidebar.info(
                    "Yahoo Finance rate-limited the request. Showing the last successful data from this session "
                    "for this ticker. Wait several minutes, then click **Load ticker data** again."
                )
            else:
                st.sidebar.warning(f"Could not load ticker data: {e}")
                if _is_rate_limit_error(e):
                    st.sidebar.caption(
                        "Yahoo limits how often unofficial clients can pull data. "
                        "Use **Load ticker data** only when needed, wait a few minutes, or enter assumptions manually below."
                    )
elif ticker == st.session_state.dcf_loaded_ticker:
    ticker_data = st.session_state.dcf_ticker_data
else:
    st.sidebar.caption(
        "Enter a symbol and click **Load ticker data** to pull figures from Yahoo. "
        "Until then, the model uses the placeholder defaults below."
    )

if ticker_data:
    default_revenue = safe_float(ticker_data.get("revenue"), 1000.0)
    default_growth = safe_float(ticker_data.get("revenue_growth_hint"), 0.05)
    rev = ticker_data.get("revenue")
    eb = ticker_data.get("ebit")
    default_margin = 0.20
    try:
        if _scalar_missing_or_nan(rev) or _scalar_missing_or_nan(eb):
            pass
        else:
            rev_f = float(rev)
            eb_f = float(eb)
            if rev_f != 0.0:
                m = eb_f / rev_f
                if np.isfinite(m) and m > 0:
                    default_margin = m
    except (TypeError, ValueError):
        default_margin = 0.20
    default_tax = safe_float(ticker_data.get("tax_rate"), 0.25)
    default_debt = safe_float(ticker_data.get("debt"), 500.0)
    default_cash = safe_float(ticker_data.get("cash"), 100.0)
    default_shares = safe_float(ticker_data.get("shares"), 100.0)
    hist_growth_cagr = safe_float(ticker_data.get("hist_revenue_cagr"), default_growth)
    hist_growth_avg = safe_float(ticker_data.get("hist_revenue_avg_growth"), default_growth)
    hist_margin = safe_float(ticker_data.get("hist_ebit_margin"), default_margin)
    beta_hint = safe_float(ticker_data.get("beta"), 1.0)
else:
    default_revenue = 1000.0
    default_growth = 0.05
    default_margin = 0.20
    default_tax = 0.25
    default_debt = 500.0
    default_cash = 100.0
    default_shares = 100.0
    hist_growth_cagr = default_growth
    hist_growth_avg = default_growth
    hist_margin = default_margin
    beta_hint = 1.0

revenue = st.sidebar.number_input("Current Revenue ($)", min_value=0.0, value=default_revenue)
years = st.sidebar.slider("Projection Years", 3, 10, 5)

st.sidebar.subheader("Growth Assumptions")
growth_base = bounded((hist_growth_cagr + hist_growth_avg) / 2, -0.10, 0.30)
growth_low = bounded(growth_base - 0.03, -0.20, 0.25)
growth_high = bounded(growth_base + 0.03, -0.05, 0.40)

st.sidebar.caption(
    f"Historical guide: revenue CAGR ~ {pct(hist_growth_cagr)} | avg YoY growth ~ {pct(hist_growth_avg)}."
)
with st.sidebar.expander("How to set growth (beginner guide)", expanded=False):
    st.markdown(
        f"""
        - **What this means:** Revenue growth controls how fast sales expand each year.
        - **Historical anchor:** This company has grown around **{pct(hist_growth_cagr)} CAGR** and **{pct(hist_growth_avg)} average YoY**.
        - **Practical range:** Try **{range_text(growth_low, growth_base, growth_high)}** for the early forecast years.
        - **Simple rule:** Start closer to history, then fade growth down each year as the business matures.
        - **Red flag:** If near-term growth is much higher than history, keep a higher reinvestment rate to stay realistic.
        """
    )
growth_rates = []
for i in range(years):
    default_i = max(default_growth - 0.01 * i, -0.50)
    g = st.sidebar.number_input(f"Year {i+1} Growth Rate (%)", value=float(default_i * 100), step=0.5) / 100
    growth_rates.append(g)

margin = st.sidebar.number_input("EBIT Margin (%)", value=float(default_margin * 100), step=0.5) / 100
tax_rate = st.sidebar.number_input("Tax Rate (%)", value=float(default_tax * 100), step=0.5) / 100
reinvest = st.sidebar.number_input("Reinvestment Rate (%)", value=50.0, step=0.5) / 100
margin_low = bounded(hist_margin - 0.03, 0.01, 0.60)
margin_high = bounded(hist_margin + 0.03, 0.03, 0.75)
st.sidebar.caption(
    f"Historical guide: median EBIT margin ~ {pct(hist_margin)}. "
    "Keep reinvestment higher when growth assumptions are aggressive."
)
with st.sidebar.expander("How to set margin, tax, and reinvestment", expanded=False):
    st.markdown(
        f"""
        - **EBIT margin:** % of revenue left after operating costs (before interest/taxes).  
          Historical median here is about **{pct(hist_margin)}**; a reasonable range is **{pct(margin_low,1)} to {pct(margin_high,1)}**.
        - **Tax rate:** Usually stable over time. If unsure, start near **{pct(default_tax)}** and adjust only with a clear reason.
        - **Reinvestment rate:** % of NOPAT put back into the business.  
          Higher growth usually needs higher reinvestment; lower growth can support lower reinvestment.
        - **Quick check:** If you raise growth and margin together, make sure reinvestment is not unrealistically low.
        """
    )

st.sidebar.subheader("Valuation Assumptions")
rf_default = 0.045
erp_default = 0.050
wacc_hint = min(max(rf_default + beta_hint * erp_default, 0.06), 0.16)
st.sidebar.caption(
    f"Valuation guide: beta-implied WACC is roughly {pct(wacc_hint)} "
    f"(using {pct(rf_default)} risk-free + {pct(erp_default)} equity risk premium). "
    "Terminal growth is usually conservative (about 2%-3%)."
)
wacc = st.sidebar.number_input("WACC (%)", value=10.0, step=0.25) / 100
terminal_growth = st.sidebar.number_input("Terminal Growth (%)", value=2.5, step=0.25) / 100
with st.sidebar.expander("How to set WACC and terminal growth", expanded=False):
    st.markdown(
        f"""
        - **WACC:** Your required annual return for this company.  
          Beta-implied starting point is about **{pct(wacc_hint)}**.
        - **Terminal growth:** Long-run growth after the forecast period.  
          A common conservative range is **2.0% to 3.0%** for mature businesses.
        - **Important rule:** WACC must be greater than terminal growth, or terminal value breaks mathematically.
        - **Sensitivity tip:** If valuation moves a lot when WACC changes by +/-1%, treat output as a range, not a single target.
        """
    )

debt = default_debt
cash = default_cash
shares = max(default_shares, 0.000001)
st.sidebar.caption(
    f"Equity bridge inputs are auto-loaded: debt {money(debt)}, cash {money(cash)}, shares {shares:,.0f}."
)

if wacc <= terminal_growth:
    st.error("WACC must be greater than terminal growth for the terminal value formula to work.")
    st.stop()

if ticker_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Company", ticker_data["name"])
    c2.metric("Current Price", money(ticker_data["price"]))
    c3.metric("Sector", ticker_data["sector"])
    c4.metric("Industry", ticker_data["industry"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Valuation Summary",
    "Assumptions",
    "Projection",
    "DCF Walkthrough",
    "Sensitivity",
])

projection_df = build_projection(revenue, growth_rates, margin, tax_rate, reinvest)
discounted_df, terminal_value, pv_terminal, enterprise_value, equity_value, value_per_share = discount_valuation(
    projection_df, wacc, terminal_growth, debt, cash, shares
)

market_price = ticker_data["price"] if ticker_data else None
valuation_gap = None
if market_price not in [None, 0] and not pd.isna(value_per_share):
    valuation_gap = value_per_share / market_price - 1

sensitivity_df = sensitivity_table(
    revenue, growth_rates, margin, tax_rate, reinvest, debt, cash, shares, wacc, terminal_growth
)

with tab1:
    st.header("Valuation Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Enterprise Value", money(enterprise_value))
    m2.metric("Equity Value", money(equity_value))
    m3.metric("Intrinsic Value Per Share", money(value_per_share))
    m4.metric("Current Market Price", money(market_price))

    if valuation_gap is not None:
        st.metric("Upside / Downside", f"{valuation_gap*100:,.2f}%")

    bridge = pd.DataFrame({
        "Line Item": [
            "Present Value of Forecast FCF",
            "Present Value of Terminal Value",
            "Enterprise Value",
            "Less Debt",
            "Add Cash",
            "Equity Value",
            "Intrinsic Value Per Share",
            "Current Market Price",
        ],
        "Amount": [
            discounted_df["PV of FCF"].sum(),
            pv_terminal,
            enterprise_value,
            -debt,
            cash,
            equity_value,
            value_per_share,
            market_price if market_price is not None else np.nan,
        ]
    })

    bridge_display = bridge.copy()
    bridge_display["Amount"] = bridge_display["Amount"].apply(lambda x: money(x))
    st.subheader("Valuation Bridge")
    st.dataframe(bridge_display, use_container_width=True, hide_index=True)

with tab2:
    st.header("Assumptions")
    st.markdown(
        "Use this section as a teaching view: start from historical anchors, choose a base case, "
        "then test optimistic and conservative cases in the sensitivity tab."
    )

    guide = pd.DataFrame({
        "Assumption": [
            "Revenue Growth (Years 1-2)",
            "Revenue Growth (Terminal years)",
            "EBIT Margin",
            "Tax Rate",
            "Reinvestment Rate",
            "WACC",
            "Terminal Growth",
        ],
        "Historical / market anchor": [
            f"CAGR {pct(hist_growth_cagr)}; Avg YoY {pct(hist_growth_avg)}",
            "Should trend down from early years",
            f"Median historical EBIT margin {pct(hist_margin)}",
            f"Current implied tax estimate {pct(default_tax)}",
            "Linked to growth intensity",
            f"Beta-implied starting point {pct(wacc_hint)}",
            "Usually tied to long-run GDP/inflation-like growth",
        ],
        "Beginner-friendly starting point": [
            range_text(growth_low, growth_base, growth_high),
            f"{pct(bounded(growth_base - 0.03, -0.05, 0.12),1)} to {pct(bounded(growth_base - 0.01, 0.00, 0.15),1)}",
            f"{pct(margin_low,1)} to {pct(margin_high,1)}",
            f"{pct(bounded(default_tax - 0.03, 0.10, 0.35),1)} to {pct(bounded(default_tax + 0.03, 0.15, 0.40),1)}",
            "30% to 70% (higher when growth is high)",
            f"{pct(bounded(wacc_hint - 0.01, 0.05, 0.18),1)} to {pct(bounded(wacc_hint + 0.01, 0.06, 0.20),1)}",
            "2.0% to 3.0% for mature companies",
        ],
        "What happens if too high": [
            "Can overstate forecast sales and valuation",
            "Can make terminal value unrealistic",
            "Can overstate profitability and FCF",
            "Can understate taxes and inflate value",
            "Can overstate free cash flow",
            "Can discount cash flows too heavily",
            "Can create unstable/overstated terminal value",
        ],
    })
    st.subheader("Assumption Playbook")
    st.dataframe(guide, use_container_width=True, hide_index=True)

    assumptions = pd.DataFrame({
        "Input": [
            "Ticker",
            "Current Revenue",
            "Projection Years",
            "Year 1 Growth",
            "Year 2 Growth",
            "Year 3 Growth",
            "Year 4 Growth" if years >= 4 else None,
            "Year 5 Growth" if years >= 5 else None,
            "EBIT Margin",
            "Tax Rate",
            "Reinvestment Rate",
            "WACC",
            "Terminal Growth",
            "Debt",
            "Cash",
            "Shares Outstanding",
        ],
        "Value": [
            ticker,
            revenue,
            years,
            growth_rates[0] if years >= 1 else None,
            growth_rates[1] if years >= 2 else None,
            growth_rates[2] if years >= 3 else None,
            growth_rates[3] if years >= 4 else None,
            growth_rates[4] if years >= 5 else None,
            margin,
            tax_rate,
            reinvest,
            wacc,
            terminal_growth,
            debt,
            cash,
            shares,
        ],
        "Why It Matters": [
            "Identifies the company and allows automatic market data retrieval.",
            "Sets the starting point for the forecast.",
            "Controls the explicit forecast horizon.",
            "Drives top line growth in year 1.",
            "Drives top line growth in year 2.",
            "Drives top line growth in year 3.",
            "Drives top line growth in year 4." if years >= 4 else None,
            "Drives top line growth in year 5." if years >= 5 else None,
            "Converts revenue into operating profit.",
            "Converts EBIT into after tax operating profit.",
            "Represents how much NOPAT must be reinvested to sustain growth.",
            "Discount rate used to value future cash flows.",
            "Long run perpetual growth assumption used in terminal value.",
            "Subtracted from enterprise value to get equity value.",
            "Added back to enterprise value to get equity value.",
            "Used to calculate value per share.",
        ],
    }).dropna()

    display_assumptions = assumptions.copy()
    display_assumptions["Value"] = display_assumptions.apply(
        lambda row: money(row["Value"]) if row["Input"] in ["Current Revenue", "Debt", "Cash"] else (
            f"{row['Value']:,.0f}" if row["Input"] in ["Projection Years", "Shares Outstanding"] else (
                pct(row["Value"]) if row["Input"] not in ["Ticker"] else row["Value"]
            )
        ),
        axis=1
    )
    st.dataframe(display_assumptions, use_container_width=True, hide_index=True)

with tab3:
    st.header("Projection")
    display_proj = discounted_df.copy()
    for col in ["Revenue", "EBIT", "NOPAT", "Reinvestment", "FCF", "PV of FCF"]:
        display_proj[col] = display_proj[col].map(lambda x: money(x))
    for col in ["Growth Rate", "EBIT Margin"]:
        display_proj[col] = discounted_df[col].map(lambda x: pct(x))
    display_proj["Discount Factor"] = discounted_df["Discount Factor"].map(lambda x: f"{x:,.4f}")

    st.dataframe(display_proj, use_container_width=True, hide_index=True)

    st.subheader("Charts")
    st.line_chart(projection_df.set_index("Year")[["Revenue", "FCF"]])
    st.bar_chart(projection_df.set_index("Year")[["EBIT", "NOPAT", "Reinvestment"]])

with tab4:
    st.header("DCF Walkthrough")

    st.markdown("""
    **Step 1**  
    Forecast revenue using your annual growth assumptions.

    **Step 2**  
    Estimate EBIT using EBIT Margin.

    **Step 3**  
    Convert EBIT to NOPAT using the tax rate.

    **Step 4**  
    Estimate reinvestment as a percent of NOPAT.

    **Step 5**  
    Compute free cash flow:

    `FCF = NOPAT − Reinvestment`

    **Step 6**  
    Discount each future FCF using WACC.

    **Step 7**  
    Estimate terminal value:

    `Terminal Value = Final Year FCF × (1 + g) / (WACC − g)`

    **Step 8**  
    Convert enterprise value to equity value:

    `Equity Value = Enterprise Value − Debt + Cash`

    **Step 9**  
    Divide by shares outstanding to get intrinsic value per share.
    """)

    walkthrough = pd.DataFrame({
        "Year": discounted_df["Year"],
        "FCF": discounted_df["FCF"],
        "Discount Factor": discounted_df["Discount Factor"],
        "PV of FCF": discounted_df["PV of FCF"],
    })
    walkthrough_display = walkthrough.copy()
    walkthrough_display["FCF"] = walkthrough_display["FCF"].map(lambda x: money(x))
    walkthrough_display["Discount Factor"] = walkthrough_display["Discount Factor"].map(lambda x: f"{x:,.4f}")
    walkthrough_display["PV of FCF"] = walkthrough_display["PV of FCF"].map(lambda x: money(x))
    st.dataframe(walkthrough_display, use_container_width=True, hide_index=True)

    st.write(f"Terminal Value: {money(terminal_value)}")
    st.write(f"Present Value of Terminal Value: {money(pv_terminal)}")

with tab5:
    st.header("Sensitivity Analysis")
    st.caption("Rows are terminal growth assumptions and columns are WACC assumptions.")
    sens_display = sensitivity_df.copy()
    for col in sens_display.columns:
        sens_display[col] = sens_display[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    st.dataframe(sens_display, use_container_width=True)

    st.write("DCF outputs are highly sensitive to discount rate and terminal growth assumptions. Use the table as a valuation range, not a single perfect number.")

st.divider()
st.markdown("##### Export to Excel")
st.caption(
    "Downloads a workbook with Summary, Assumptions, Forecast_DCF, DCF_walkthrough, and Sensitivity tabs "
    "(numeric values—format currency or % in Excel as you prefer)."
)

safe_name = (ticker or "model").lower().replace(" ", "_")
try:
    excel_bytes = build_dcf_excel_bytes(
        ticker,
        ticker_data,
        revenue,
        years,
        growth_rates,
        margin,
        tax_rate,
        reinvest,
        wacc,
        terminal_growth,
        debt,
        cash,
        shares,
        discounted_df,
        enterprise_value,
        equity_value,
        value_per_share,
        market_price,
        terminal_value,
        pv_terminal,
        sensitivity_df,
    )
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.download_button(
            label="Download Excel workbook",
            data=excel_bytes,
            file_name=f"{safe_name}_dcf_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
            help="Structured sheets with valuation summary, inputs, forecast, discounting detail, and sensitivity grid.",
        )
except ImportError:
    st.caption("Excel download requires **openpyxl**. Install with `pip install openpyxl` and restart the app.")
