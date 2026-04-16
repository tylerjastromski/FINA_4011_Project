import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from io import BytesIO

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

@st.cache_data(show_spinner=False)
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

ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()

ticker_data = None
if ticker:
    try:
        ticker_data = load_ticker_data(ticker)
    except Exception as e:
        st.sidebar.warning(f"Could not load ticker data: {e}")

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
else:
    default_revenue = 1000.0
    default_growth = 0.05
    default_margin = 0.20
    default_tax = 0.25
    default_debt = 500.0
    default_cash = 100.0
    default_shares = 100.0

revenue = st.sidebar.number_input("Current Revenue ($)", min_value=0.0, value=default_revenue)
years = st.sidebar.slider("Projection Years", 3, 10, 5)

st.sidebar.subheader("Growth Assumptions")
growth_rates = []
for i in range(years):
    default_i = max(default_growth - 0.01 * i, -0.50)
    g = st.sidebar.number_input(f"Year {i+1} Growth Rate (%)", value=float(default_i * 100), step=0.5) / 100
    growth_rates.append(g)

margin = st.sidebar.number_input("EBIT Margin (%)", value=float(default_margin * 100), step=0.5) / 100
tax_rate = st.sidebar.number_input("Tax Rate (%)", value=float(default_tax * 100), step=0.5) / 100
reinvest = st.sidebar.number_input("Reinvestment Rate (%)", value=50.0, step=0.5) / 100

st.sidebar.subheader("Valuation Assumptions")
wacc = st.sidebar.number_input("WACC (%)", value=10.0, step=0.25) / 100
terminal_growth = st.sidebar.number_input("Terminal Growth (%)", value=2.5, step=0.25) / 100

st.sidebar.subheader("Capital Structure")
debt = st.sidebar.number_input("Debt ($)", min_value=0.0, value=default_debt)
cash = st.sidebar.number_input("Cash ($)", min_value=0.0, value=default_cash)
shares = st.sidebar.number_input("Shares Outstanding", min_value=0.000001, value=default_shares)

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
    sens = sensitivity_table(revenue, growth_rates, margin, tax_rate, reinvest, debt, cash, shares, wacc, terminal_growth)
    st.caption("Rows are terminal growth assumptions and columns are WACC assumptions.")
    sens_display = sens.copy()
    for col in sens_display.columns:
        sens_display[col] = sens_display[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    st.dataframe(sens_display, use_container_width=True)

    st.write("DCF outputs are highly sensitive to discount rate and terminal growth assumptions. Use the table as a valuation range, not a single perfect number.")

try:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        discounted_df.to_excel(writer, sheet_name="Projection", index=False)
        pd.DataFrame({
            "Metric": ["Enterprise Value", "Equity Value", "Intrinsic Value Per Share", "Current Market Price"],
            "Value": [enterprise_value, equity_value, value_per_share, market_price]
        }).to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    safe_name = (ticker or "model").lower().replace(" ", "_")
    st.download_button(
        "Download valuation output to Excel",
        data=buffer.getvalue(),
        file_name=f"{safe_name}_dcf_model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
except ImportError:
    st.caption("Excel download requires **openpyxl**. Install with `pip install openpyxl` and restart the app.")
