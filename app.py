import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st



st.set_page_config(
    page_title="DCF Valuation App",
    page_icon="💰",
    layout="wide",
)

st.title("DCF Valuation App")
st.caption("A teaching focused discounted cash flow model built in Streamlit.")


def format_money(value: Optional[float], decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_pct(value: Optional[float], decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    try:
        if n is None or d in [None, 0] or pd.isna(n) or pd.isna(d):
            return None
        return n / d
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_company_data(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)

    info = tk.info if tk.info else {}
    financials = tk.financials
    cashflow = tk.cashflow
    balance_sheet = tk.balance_sheet
    history = tk.history(period="1y")

    latest_price = None
    if history is not None and not history.empty:
        latest_price = float(history["Close"].dropna().iloc[-1])

    shares_outstanding = info.get("sharesOutstanding")
    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt")
    cash = info.get("totalCash")
    beta = info.get("beta")
    company_name = info.get("longName", ticker.upper())
    currency = info.get("currency", "USD")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")

    def find_row(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
        if df is None or df.empty:
            return None
        for candidate in candidates:
            matches = [idx for idx in df.index if str(idx).strip().lower() == candidate.lower()]
            if matches:
                series = df.loc[matches[0]]
                series = pd.to_numeric(series, errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[0])
        for candidate in candidates:
            for idx in df.index:
                if candidate.lower() in str(idx).lower():
                    series = df.loc[idx]
                    series = pd.to_numeric(series, errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[0])
        return None

    revenue = find_row(financials, ["Total Revenue", "Revenue", "Operating Revenue"])
    ebit = find_row(financials, ["EBIT", "Operating Income"])
    tax_provision = find_row(financials, ["Tax Provision", "Income Tax Expense"])
    pretax_income = find_row(financials, ["Pretax Income", "Pretax Income Loss"])
    depreciation = find_row(cashflow, ["Depreciation And Amortization", "Depreciation", "Depreciation Amortization Depletion"])
    capex = find_row(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    wc_change = find_row(cashflow, ["Change In Working Capital", "Changes In Working Capital"])

    tax_rate = None
    if tax_provision is not None and pretax_income not in [None, 0]:
        raw_tax = tax_provision / pretax_income
        if not pd.isna(raw_tax):
            tax_rate = max(0.0, min(0.40, raw_tax))

    revenue_growth_hint = info.get("revenueGrowth")
    operating_margin_hint = safe_div(ebit, revenue)
    dep_pct_hint = safe_div(depreciation, revenue)
    capex_pct_hint = None
    if capex is not None and revenue not in [None, 0]:
        capex_pct_hint = abs(capex) / revenue
    nwc_pct_hint = None
    if wc_change is not None and revenue not in [None, 0]:
        nwc_pct_hint = abs(wc_change) / revenue

    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "currency": currency,
        "latest_price": latest_price,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "beta": beta,
        "revenue": revenue,
        "ebit": ebit,
        "depreciation": depreciation,
        "capex": capex,
        "wc_change": wc_change,
        "tax_rate": tax_rate,
        "revenue_growth_hint": revenue_growth_hint,
        "operating_margin_hint": operating_margin_hint,
        "dep_pct_hint": dep_pct_hint,
        "capex_pct_hint": capex_pct_hint,
        "nwc_pct_hint": nwc_pct_hint,
    }


def compute_cost_of_equity(risk_free: float, equity_risk_premium: float, beta: float) -> float:
    return risk_free + beta * equity_risk_premium


def compute_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity: float,
    pre_tax_cost_of_debt: float,
    tax_rate: float,
) -> float:
    total_capital = max(market_cap, 0) + max(total_debt, 0)
    if total_capital <= 0:
        return cost_of_equity
    weight_equity = max(market_cap, 0) / total_capital
    weight_debt = max(total_debt, 0) / total_capital
    after_tax_cost_of_debt = pre_tax_cost_of_debt * (1 - tax_rate)
    return weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt


def build_forecast(
    revenue_base: float,
    growth_years_1_5: List[float],
    operating_margin: float,
    tax_rate: float,
    dep_pct: float,
    capex_pct: float,
    nwc_pct: float,
) -> pd.DataFrame:
    rows = []
    revenue = revenue_base

    for i, growth in enumerate(growth_years_1_5, start=1):
        revenue = revenue * (1 + growth)
        ebit = revenue * operating_margin
        nopat = ebit * (1 - tax_rate)
        depreciation = revenue * dep_pct
        capex = revenue * capex_pct
        change_nwc = revenue * nwc_pct
        fcff = nopat + depreciation - capex - change_nwc

        rows.append(
            {
                "Year": i,
                "Revenue": revenue,
                "Growth": growth,
                "EBIT": ebit,
                "EBIT Margin": operating_margin,
                "NOPAT": nopat,
                "D&A": depreciation,
                "Capex": capex,
                "Change in NWC": change_nwc,
                "FCFF": fcff,
            }
        )

    return pd.DataFrame(rows)


def discount_forecast(
    forecast_df: pd.DataFrame,
    wacc: float,
    terminal_growth: float,
    total_debt: float,
    cash: float,
    shares_outstanding: float,
) -> Dict:
    df = forecast_df.copy()
    df["Discount Factor"] = 1 / ((1 + wacc) ** df["Year"])
    df["PV of FCFF"] = df["FCFF"] * df["Discount Factor"]

    terminal_year_fcff = float(df.iloc[-1]["FCFF"])
    terminal_value = terminal_year_fcff * (1 + terminal_growth) / (wacc - terminal_growth)
    terminal_pv = terminal_value / ((1 + wacc) ** int(df.iloc[-1]["Year"]))

    enterprise_value = df["PV of FCFF"].sum() + terminal_pv
    equity_value = enterprise_value - total_debt + cash
    intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else np.nan

    return {
        "discounted_df": df,
        "terminal_value": terminal_value,
        "terminal_pv": terminal_pv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
    }


def build_sensitivity(
    revenue_base: float,
    growth_years_1_5: List[float],
    operating_margin: float,
    tax_rate: float,
    dep_pct: float,
    capex_pct: float,
    nwc_pct: float,
    wacc_values: List[float],
    terminal_growth_values: List[float],
    total_debt: float,
    cash: float,
    shares_outstanding: float,
) -> pd.DataFrame:
    grid = pd.DataFrame(index=[f"{g*100:.1f}%" for g in terminal_growth_values])

    for wacc in wacc_values:
        values = []
        for tg in terminal_growth_values:
            if wacc <= tg:
                values.append(np.nan)
                continue
            forecast = build_forecast(
                revenue_base=revenue_base,
                growth_years_1_5=growth_years_1_5,
                operating_margin=operating_margin,
                tax_rate=tax_rate,
                dep_pct=dep_pct,
                capex_pct=capex_pct,
                nwc_pct=nwc_pct,
            )
            result = discount_forecast(
                forecast_df=forecast,
                wacc=wacc,
                terminal_growth=tg,
                total_debt=total_debt,
                cash=cash,
                shares_outstanding=shares_outstanding,
            )
            values.append(result["intrinsic_value_per_share"])
        grid[f"{wacc*100:.1f}%"] = values

    grid.index.name = "Terminal Growth"
    return grid


with st.expander("How this app works", expanded=True):
    st.markdown(
        """
        This app estimates intrinsic value with a free cash flow to firm DCF.

        The model follows this flow:

        1. Pull company data using the ticker.
        2. Let you review or override key assumptions.
        3. Forecast revenue, operating profit, taxes, reinvestment, and free cash flow.
        4. Discount projected cash flows at WACC.
        5. Estimate terminal value and convert enterprise value into equity value per share.

        The app is intentionally transparent. Every major input and intermediate step is shown below.
        """
    )

ticker = st.text_input("Ticker", value="AAPL").strip().upper()

if not ticker:
    st.stop()

try:
    data = load_company_data(ticker)
except Exception as exc:
    st.error(f"Could not load data for {ticker}. Error: {exc}")
    st.stop()

left, right = st.columns([1.2, 1])

with left:
    st.subheader(f"{data['company_name']} ({data['ticker']})")
    st.write(f"Sector: {data['sector']}")
    st.write(f"Industry: {data['industry']}")
    st.write(f"Currency: {data['currency']}")

with right:
    st.metric("Current Market Price", format_money(data["latest_price"]))
    st.metric("Shares Outstanding", f"{data['shares_outstanding']:,.0f}" if data["shares_outstanding"] else "N/A")
    st.metric("Market Cap", format_money(data["market_cap"], 0))

st.divider()

st.subheader("Step 1: Review base company data")

base_df = pd.DataFrame(
    {
        "Metric": [
            "Latest Revenue",
            "Latest EBIT",
            "Tax Rate Estimate",
            "Depreciation and Amortization",
            "Capital Expenditures",
            "Change in Working Capital",
            "Cash",
            "Debt",
            "Beta",
        ],
        "Value": [
            format_money(data["revenue"], 0),
            format_money(data["ebit"], 0),
            format_pct(data["tax_rate"] * 100 if data["tax_rate"] is not None else None),
            format_money(data["depreciation"], 0),
            format_money(data["capex"], 0),
            format_money(data["wc_change"], 0),
            format_money(data["cash"], 0),
            format_money(data["total_debt"], 0),
            f"{data['beta']:.2f}" if data["beta"] is not None else "N/A",
        ],
        "Why it matters": [
            "Starting point for forecasting future sales",
            "Used to estimate operating profitability",
            "Converts EBIT into after tax operating profit",
            "Non cash add back in FCFF",
            "Investment needed to maintain and grow the business",
            "Cash tied up in operations",
            "Added back when converting enterprise value to equity value",
            "Subtracted when converting enterprise value to equity value",
            "Used in CAPM to estimate cost of equity",
        ],
    }
)
st.dataframe(base_df, use_container_width=True, hide_index=True)

st.subheader("Step 2: Set valuation assumptions")

col1, col2, col3 = st.columns(3)

default_rev_growth = data["revenue_growth_hint"] if data["revenue_growth_hint"] is not None else 0.08
default_margin = data["operating_margin_hint"] if data["operating_margin_hint"] is not None else 0.20
default_tax = data["tax_rate"] if data["tax_rate"] is not None else 0.21
default_dep_pct = data["dep_pct_hint"] if data["dep_pct_hint"] is not None else 0.03
default_capex_pct = data["capex_pct_hint"] if data["capex_pct_hint"] is not None else 0.04
default_nwc_pct = data["nwc_pct_hint"] if data["nwc_pct_hint"] is not None else 0.01
default_beta = data["beta"] if data["beta"] is not None else 1.0

with col1:
    st.markdown("**Operating assumptions**")
    growth_y1 = st.number_input("Revenue Growth Year 1", min_value=-0.50, max_value=1.00, value=float(default_rev_growth), step=0.01, format="%.2f",
                                help="Expected revenue growth in the first forecast year.")
    growth_y2 = st.number_input("Revenue Growth Year 2", min_value=-0.50, max_value=1.00, value=float(max(default_rev_growth - 0.01, -0.50)), step=0.01, format="%.2f")
    growth_y3 = st.number_input("Revenue Growth Year 3", min_value=-0.50, max_value=1.00, value=float(max(default_rev_growth - 0.02, -0.50)), step=0.01, format="%.2f")
    growth_y4 = st.number_input("Revenue Growth Year 4", min_value=-0.50, max_value=1.00, value=float(max(default_rev_growth - 0.03, -0.50)), step=0.01, format="%.2f")
    growth_y5 = st.number_input("Revenue Growth Year 5", min_value=-0.50, max_value=1.00, value=float(max(default_rev_growth - 0.04, -0.50)), step=0.01, format="%.2f")
    operating_margin = st.number_input("Operating Margin", min_value=-0.20, max_value=0.80, value=float(default_margin), step=0.01, format="%.2f",
                                       help="EBIT as a percent of revenue.")
    tax_rate = st.number_input("Tax Rate", min_value=0.00, max_value=0.50, value=float(default_tax), step=0.01, format="%.2f",
                               help="Tax applied to EBIT to estimate NOPAT.")

with col2:
    st.markdown("**Reinvestment assumptions**")
    dep_pct = st.number_input("D&A as Percent of Revenue", min_value=0.00, max_value=0.30, value=float(default_dep_pct), step=0.005, format="%.3f",
                              help="Non cash expense added back in FCFF.")
    capex_pct = st.number_input("Capex as Percent of Revenue", min_value=0.00, max_value=0.40, value=float(default_capex_pct), step=0.005, format="%.3f",
                                help="Capital investment deducted in FCFF.")
    nwc_pct = st.number_input("Change in NWC as Percent of Revenue", min_value=-0.10, max_value=0.20, value=float(default_nwc_pct), step=0.005, format="%.3f",
                              help="Incremental working capital needs deducted in FCFF.")
    terminal_growth = st.number_input("Terminal Growth Rate", min_value=0.00, max_value=0.08, value=0.025, step=0.005, format="%.3f",
                                      help="Long run perpetual growth rate after Year 5.")

with col3:
    st.markdown("**Discount rate assumptions**")
    use_manual_wacc = st.checkbox("Use manual WACC", value=False)
    risk_free_rate = st.number_input("Risk Free Rate", min_value=0.00, max_value=0.15, value=0.043, step=0.001, format="%.3f",
                                     help="Usually based on a long term government bond yield.")
    equity_risk_premium = st.number_input("Equity Risk Premium", min_value=0.00, max_value=0.15, value=0.055, step=0.001, format="%.3f",
                                          help="Extra return investors require for equities over the risk free rate.")
    beta = st.number_input("Beta", min_value=0.10, max_value=3.00, value=float(default_beta), step=0.05, format="%.2f")
    pre_tax_cost_of_debt = st.number_input("Pre Tax Cost of Debt", min_value=0.00, max_value=0.20, value=0.050, step=0.001, format="%.3f")
    manual_wacc = st.number_input("Manual WACC", min_value=0.01, max_value=0.30, value=0.090, step=0.001, format="%.3f",
                                  help="Used only if the box above is checked.")

base_revenue = st.number_input(
    "Starting Revenue",
    min_value=0.0,
    value=float(data["revenue"]) if data["revenue"] is not None else 1000000000.0,
    step=1000000.0,
    help="Most recent revenue used as the base year for projections.",
)

cash = float(data["cash"]) if data["cash"] is not None else 0.0
debt = float(data["total_debt"]) if data["total_debt"] is not None else 0.0
market_cap = float(data["market_cap"]) if data["market_cap"] is not None else (
    float(data["latest_price"]) * float(data["shares_outstanding"])
    if data["latest_price"] is not None and data["shares_outstanding"] is not None
    else 0.0
)
shares_outstanding = float(data["shares_outstanding"]) if data["shares_outstanding"] is not None else 1.0

cost_of_equity = compute_cost_of_equity(risk_free_rate, equity_risk_premium, beta)
auto_wacc = compute_wacc(
    market_cap=market_cap,
    total_debt=debt,
    cost_of_equity=cost_of_equity,
    pre_tax_cost_of_debt=pre_tax_cost_of_debt,
    tax_rate=tax_rate,
)
selected_wacc = manual_wacc if use_manual_wacc else auto_wacc

if selected_wacc <= terminal_growth:
    st.error("WACC must be greater than terminal growth rate for the terminal value formula to work.")
    st.stop()

growth_list = [growth_y1, growth_y2, growth_y3, growth_y4, growth_y5]

assumption_df = pd.DataFrame(
    {
        "Input": [
            "Revenue Growth Year 1",
            "Revenue Growth Year 2",
            "Revenue Growth Year 3",
            "Revenue Growth Year 4",
            "Revenue Growth Year 5",
            "Operating Margin",
            "Tax Rate",
            "D&A as Percent of Revenue",
            "Capex as Percent of Revenue",
            "Change in NWC as Percent of Revenue",
            "Risk Free Rate",
            "Equity Risk Premium",
            "Beta",
            "Pre Tax Cost of Debt",
            "Selected WACC",
            "Terminal Growth",
        ],
        "Value": [
            format_pct(growth_y1 * 100),
            format_pct(growth_y2 * 100),
            format_pct(growth_y3 * 100),
            format_pct(growth_y4 * 100),
            format_pct(growth_y5 * 100),
            format_pct(operating_margin * 100),
            format_pct(tax_rate * 100),
            format_pct(dep_pct * 100),
            format_pct(capex_pct * 100),
            format_pct(nwc_pct * 100),
            format_pct(risk_free_rate * 100),
            format_pct(equity_risk_premium * 100),
            f"{beta:.2f}",
            format_pct(pre_tax_cost_of_debt * 100),
            format_pct(selected_wacc * 100),
            format_pct(terminal_growth * 100),
        ],
        "Explanation": [
            "Top line growth for forecast year 1",
            "Top line growth for forecast year 2",
            "Top line growth for forecast year 3",
            "Top line growth for forecast year 4",
            "Top line growth for forecast year 5",
            "Converts revenue into EBIT",
            "Converts EBIT into NOPAT",
            "Non cash add back assumption",
            "Capital investment assumption",
            "Working capital investment assumption",
            "Base return for CAPM",
            "Market premium for CAPM",
            "Measures stock sensitivity to the market",
            "Debt financing cost",
            "Discount rate applied to FCFF",
            "Long run growth after explicit forecast period",
        ],
    }
)
st.dataframe(assumption_df, use_container_width=True, hide_index=True)

st.subheader("Step 3: Forecast the business")

forecast_df = build_forecast(
    revenue_base=base_revenue,
    growth_years_1_5=growth_list,
    operating_margin=operating_margin,
    tax_rate=tax_rate,
    dep_pct=dep_pct,
    capex_pct=capex_pct,
    nwc_pct=nwc_pct,
)

display_forecast = forecast_df.copy()
for col in ["Revenue", "EBIT", "NOPAT", "D&A", "Capex", "Change in NWC", "FCFF"]:
    display_forecast[col] = display_forecast[col].map(lambda x: format_money(x, 0))
display_forecast["Growth"] = forecast_df["Growth"].map(lambda x: format_pct(x * 100))
display_forecast["EBIT Margin"] = forecast_df["EBIT Margin"].map(lambda x: format_pct(x * 100))
st.dataframe(display_forecast, use_container_width=True, hide_index=True)

with st.expander("Show FCFF formula explanation", expanded=False):
    st.markdown(
        """
        Free Cash Flow to Firm is calculated as:

        **FCFF = NOPAT + D&A - Capex - Change in NWC**

        Where:

        * **NOPAT** = EBIT × (1 - Tax Rate)
        * **D&A** is added back because it is a non cash expense
        * **Capex** is subtracted because it represents investment in long lived assets
        * **Change in NWC** is subtracted because growing operations usually require more working capital
        """
    )

st.subheader("Step 4: Discount projected cash flows and estimate value")

valuation_result = discount_forecast(
    forecast_df=forecast_df,
    wacc=selected_wacc,
    terminal_growth=terminal_growth,
    total_debt=debt,
    cash=cash,
    shares_outstanding=shares_outstanding,
)

discount_df = valuation_result["discounted_df"].copy()
display_discount = discount_df.copy()
for col in ["FCFF", "PV of FCFF"]:
    display_discount[col] = display_discount[col].map(lambda x: format_money(x, 0))
display_discount["Discount Factor"] = discount_df["Discount Factor"].map(lambda x: f"{x:.4f}")
st.dataframe(display_discount[["Year", "FCFF", "Discount Factor", "PV of FCFF"]], use_container_width=True, hide_index=True)

bridge_df = pd.DataFrame(
    {
        "Valuation Bridge": [
            "Present Value of Forecast Period FCFF",
            "Present Value of Terminal Value",
            "Enterprise Value",
            "Less Debt",
            "Add Cash",
            "Equity Value",
            "Intrinsic Value Per Share",
            "Current Market Price",
            "Upside / Downside",
        ],
        "Amount": [
            format_money(discount_df["PV of FCFF"].sum(), 0),
            format_money(valuation_result["terminal_pv"], 0),
            format_money(valuation_result["enterprise_value"], 0),
            format_money(-debt, 0),
            format_money(cash, 0),
            format_money(valuation_result["equity_value"], 0),
            format_money(valuation_result["intrinsic_value_per_share"]),
            format_money(data["latest_price"]),
            format_pct(
                (
                    (valuation_result["intrinsic_value_per_share"] / data["latest_price"] - 1) * 100
                    if data["latest_price"] not in [None, 0]
                    else None
                )
            ),
        ],
    }
)
st.dataframe(bridge_df, use_container_width=True, hide_index=True)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Intrinsic Value Per Share", format_money(valuation_result["intrinsic_value_per_share"]))
metric2.metric("Current Price", format_money(data["latest_price"]))
if data["latest_price"] not in [None, 0]:
    pct_gap = (valuation_result["intrinsic_value_per_share"] / data["latest_price"] - 1) * 100
    metric3.metric("Valuation Gap", format_pct(pct_gap))
else:
    metric3.metric("Valuation Gap", "N/A")

st.subheader("Step 5: Understand the discount rate")

capital_structure_df = pd.DataFrame(
    {
        "Component": [
            "Risk Free Rate",
            "Equity Risk Premium",
            "Beta",
            "Cost of Equity",
            "Pre Tax Cost of Debt",
            "Tax Rate",
            "After Tax Cost of Debt",
            "Market Cap",
            "Debt",
            "Auto WACC",
            "Selected WACC",
        ],
        "Value": [
            format_pct(risk_free_rate * 100),
            format_pct(equity_risk_premium * 100),
            f"{beta:.2f}",
            format_pct(cost_of_equity * 100),
            format_pct(pre_tax_cost_of_debt * 100),
            format_pct(tax_rate * 100),
            format_pct(pre_tax_cost_of_debt * (1 - tax_rate) * 100),
            format_money(market_cap, 0),
            format_money(debt, 0),
            format_pct(auto_wacc * 100),
            format_pct(selected_wacc * 100),
        ],
        "Explanation": [
            "Government bond yield used as the base rate",
            "Extra expected return for equities",
            "Sensitivity to market movements",
            "CAPM result: risk free rate plus beta times ERP",
            "Borrowing cost before tax benefit",
            "Used for debt tax shield",
            "Debt cost after accounting for taxes",
            "Equity portion of capital structure",
            "Debt portion of capital structure",
            "Weighted blend of equity and debt financing costs",
            "Final discount rate actually used in valuation",
        ],
    }
)
st.dataframe(capital_structure_df, use_container_width=True, hide_index=True)

st.subheader("Sensitivity analysis")

wacc_range = [selected_wacc - 0.02, selected_wacc - 0.01, selected_wacc, selected_wacc + 0.01, selected_wacc + 0.02]
tg_range = [terminal_growth - 0.01, terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005, terminal_growth + 0.01]
wacc_range = [max(0.01, x) for x in wacc_range]
tg_range = [max(0.0, x) for x in tg_range]

sensitivity_df = build_sensitivity(
    revenue_base=base_revenue,
    growth_years_1_5=growth_list,
    operating_margin=operating_margin,
    tax_rate=tax_rate,
    dep_pct=dep_pct,
    capex_pct=capex_pct,
    nwc_pct=nwc_pct,
    wacc_values=wacc_range,
    terminal_growth_values=tg_range,
    total_debt=debt,
    cash=cash,
    shares_outstanding=shares_outstanding,
)

st.caption("Rows are terminal growth assumptions. Columns are WACC assumptions. Values are intrinsic value per share.")
st.dataframe(
    sensitivity_df.style.format("${:,.2f}", na_rep="N/A"),
    use_container_width=True,
)

with st.expander("Interpretation guide", expanded=False):
    st.markdown(
        """
        A DCF is extremely sensitive to a few assumptions:

        * Revenue growth
        * Operating margin
        * WACC
        * Terminal growth

        Treat the result as a valuation range, not a single perfect number.
        """
    )

st.subheader("Download model output")

download_forecast = forecast_df.copy()
download_discount = discount_df.copy()
download_assumptions = assumption_df.copy()
download_bridge = bridge_df.copy()
download_sensitivity = sensitivity_df.copy()

from io import BytesIO

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    download_assumptions.to_excel(writer, sheet_name="Assumptions", index=False)
    download_forecast.to_excel(writer, sheet_name="Forecast", index=False)
    download_discount.to_excel(writer, sheet_name="Discounted_Cash_Flows", index=False)
    download_bridge.to_excel(writer, sheet_name="Valuation_Bridge", index=False)
    download_sensitivity.to_excel(writer, sheet_name="Sensitivity")
buffer.seek(0)

st.download_button(
    "Download DCF workbook as Excel",
    data=buffer,
    file_name=f"{ticker}_dcf_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(f"Last refreshed: {datetime.now().strftime('%Y %m %d %H:%M:%S')}")
