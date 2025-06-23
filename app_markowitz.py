"""
A Streamlit web application for performing a comparative walk-forward backtest
of Cost-Aware vs. Classic Markowitz portfolio optimization strategies.

This application implements a robust, point-in-time backtesting methodology
designed to prevent common biases and data leakage. The key design principles are:

1.  Dynamic Asset Universe & Survivorship Bias Elimination:
    The simulation begins by fetching data for a broad "master universe" of
    tickers. At each rebalancing step, the universe of tradable assets is
    dynamically determined by filtering for stocks that existed and had
    uninterrupted data throughout the entire lookback (training) window. This
    point-in-time selection process naturally handles stock IPOs and delistings,
    thus eliminating survivorship bias.

2.  Drift-Aware Cost Calculation:
    To accurately model transaction costs, the simulation accounts for
    portfolio drift. Before each rebalance, it calculates the "drifted" weights
    based on the actual performance of the held assets. The turnover cost is
    then calculated against these drifted weights, not the previous target weights.

3.  Fair & Optional Risk Management:
    A portfolio-level trailing stop-loss can be applied as a risk management
    overlay. To ensure a fair comparison, this rule is applied equally to all
    three strategies (Cost-Aware, Classic, and Equal-Weight benchmark).

4.  Point-in-Time & Unbiased Logic:
    All calculations are performed without future information. The Classic
    Markowitz optimizer is seeded with a neutral guess (equal weight) to ensure
    a fair comparison. The Forward Signal for the next trading period is
    correctly generated using the price on the decision date.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional

# Try to import quantstats, with fallback
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    QUANTSTATS_AVAILABLE = False
    st.warning(f"quantstats not available ({str(e)}). Using built-in statistics functions.")

# Fallback statistics functions if quantstats is not available
def calculate_cagr(returns_series):
    """Calculate Compound Annual Growth Rate"""
    if len(returns_series) < 2:
        return 0.0
    total_return = (1 + returns_series).prod() - 1
    years = len(returns_series) / 252  # Assuming 252 trading days per year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    if len(returns_series) < 2:
        return 0.0
    excess_returns = returns_series - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(returns_series):
    """Calculate Maximum Drawdown"""
    if len(returns_series) < 2:
        return 0.0
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_volatility(returns_series):
    """Calculate Annualized Volatility"""
    if len(returns_series) < 2:
        return 0.0
    return returns_series.std() * np.sqrt(252)

# --- Data Acquisition and Preparation ---

@st.cache_data(show_spinner="Fetching historical data for master universe...")
def get_survivorship_bias_free_data(master_ticker_list: list[str], period_days: int) -> pd.DataFrame:
    """
    Downloads historical data for a master list of tickers, creating a
    point-in-time-aware DataFrame. NaN values correctly represent periods
    where a stock was not listed or data is unavailable. This is the foundation
    for a survivorship-bias-free backtest.
    """
    st.info(f"Attempting to download data for {len(master_ticker_list)} tickers. This may take a moment...")

    df = yf.download(
        " ".join(master_ticker_list),
        period=f"{period_days}d",
        progress=False,
        auto_adjust=True,
        group_by='ticker'
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close_df = df.stack(level=0, future_stack=True)['Close'].unstack()
    else:
        close_df = df[['Close']] if 'Close' in df.columns else df

    initial_cols = len(close_df.columns)
    close_df.dropna(axis=1, how='all', inplace=True)
    final_cols = len(close_df.columns)

    st.success(f"Successfully created master price data. Kept {final_cols} of {initial_cols} requested tickers with valid data.")
    return close_df

def get_tradable_universe_for_segment(price_df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    """
    Determines the tradable universe for a specific time segment by filtering
    the master price history for assets with continuous data.
    """
    training_period_prices = price_df.iloc[start_idx:end_idx]
    tradable_assets = training_period_prices.dropna(axis=1, how='any')
    return tradable_assets.columns.tolist()

# --- Optimization and Portfolio Logic ---

def apply_stop_loss(series: pd.Series, stop_loss_pct: float) -> Tuple[pd.Series, Optional[pd.Timestamp]]:
    """
    Applies a portfolio-level trailing stop-loss on a daily, point-in-time basis.

    Args:
        series: A pandas Series of portfolio daily returns for a single test segment.
        stop_loss_pct: The drawdown percentage (e.g., 10.0 for 10%) to trigger the stop.
                       If this value is zero or negative, the function returns the series unmodified.

    Returns:
        A tuple containing:
        - A pandas Series of returns with the stop-loss logic applied.
        - The timestamp of the day the stop-loss was triggered, or None if not triggered.
    """
    if stop_loss_pct <= 0:
        return series, None

    stop_loss_decimal = stop_loss_pct / 100.0
    modified_series = series.copy()
    high_water_mark = 1.0
    equity = 1.0
    stop_triggered = False
    stop_loss_trigger_date = None

    for date, ret in series.items():
        if stop_triggered:
            # Once stopped, the strategy remains in cash (0% return) for the rest of the segment.
            modified_series.loc[date] = 0.0
            continue

        equity *= (1 + ret)
        high_water_mark = max(high_water_mark, equity)
        drawdown = (high_water_mark - equity) / high_water_mark

        if drawdown > stop_loss_decimal:
            stop_triggered = True
            stop_loss_trigger_date = date
            # The loss on the trigger day is realized; subsequent days will have 0% return.

    return modified_series, stop_loss_trigger_date

def sharpe_objective_classic(w, mu, sigma):
    """Objective for Classic Markowitz: Maximize Sharpe Ratio."""
    expected_return = w.T @ mu
    vol = np.sqrt(w.T @ sigma @ w)
    if vol < 1e-9: return 1e9
    return -expected_return / vol

def cost_aware_objective(w, mu, sigma, w_prev, turnover_lambda):
    """
    Objective for Cost-Aware Markowitz. Maximizes Sharpe Ratio while
    penalizing for portfolio turnover.
    """
    expected_return = w.T @ mu
    vol = np.sqrt(w.T @ sigma @ w)
    if vol < 1e-9: return 1e9

    sharpe_ratio = expected_return / vol
    turnover = 0.5 * np.sum(np.abs(w - w_prev))
    turnover_penalty = turnover * turnover_lambda

    return -(sharpe_ratio - turnover_penalty)

def calculate_drifted_weights(start_weights: dict, period_returns: pd.DataFrame) -> dict:
    """
    Calculates the drifted portfolio weights after a period of market movement.
    """
    if not start_weights: return {}
    cumulative_returns = (1 + period_returns).prod()
    end_values = {
        ticker: start_weights.get(ticker, 0) * cumulative_returns.get(ticker, 1)
        for ticker in start_weights
    }
    total_end_value = sum(end_values.values())
    if total_end_value == 0: return {ticker: 0.0 for ticker in start_weights}
    return {ticker: value / total_end_value for ticker, value in end_values.items()}

# --- CORE SIMULATION ENGINE ---

@st.cache_data(show_spinner="Running backtest simulation for all strategies...")
def run_walk_forward_simulation(
    price_df: pd.DataFrame, train_window: int, test_window: int,
    top_n: int, cost_rate: float, turnover_lambda: float, cov_reg: float,
    stop_loss_pct: float
):
    """
    Executes the main walk-forward simulation on the master price data.

    Args:
        price_df: The master DataFrame of historical prices.
        ... (other simulation parameters) ...
        stop_loss_pct: The portfolio stop-loss percentage. Disabled if <= 0.
    """
    if price_df.empty or price_df.shape[1] < 2: return {}, [], {}
    all_returns = price_df.pct_change()
    n_days = len(price_df)
    if n_days < train_window + test_window: return {}, [], {}

    results = {'cost_aware': [], 'classic': [], 'equal_weight': []}
    weights_log = []
    stop_loss_log = {'cost_aware': [], 'classic': [], 'equal_weight': []}
    w_drifted = {'cost_aware': {}, 'classic': {}, 'equal_weight': {}}

    segments_to_process = range(0, n_days - (train_window + test_window), test_window)
    progress_bar = st.progress(0, text="Processing Segments...")

    for i, start in enumerate(segments_to_process):
        train_start_idx, train_end_idx = start, start + train_window
        test_start_idx, test_end_idx = train_end_idx, train_end_idx + test_window
        if test_end_idx > n_days: continue

        tradable_tickers = get_tradable_universe_for_segment(price_df, train_start_idx, train_end_idx)

        if len(tradable_tickers) < 2:
            for key in w_drifted:
                if w_drifted[key]:
                    test_returns_for_drift = all_returns.iloc[train_end_idx:test_end_idx]
                    w_drifted[key] = calculate_drifted_weights(w_drifted[key], test_returns_for_drift.fillna(0))
            continue

        train_returns_full = all_returns.iloc[train_start_idx:train_end_idx][tradable_tickers]
        segment_tickers = tradable_tickers
        if top_n is not None and 0 < top_n < len(tradable_tickers):
            momentum = train_returns_full.mean()
            segment_tickers = momentum.nlargest(top_n).index.tolist()

        train_returns_seg = train_returns_full[segment_tickers].fillna(0)
        test_returns_seg = all_returns.iloc[test_start_idx:test_end_idx][segment_tickers].fillna(0)
        mu = train_returns_seg.mean().values
        sigma = train_returns_seg.cov().values + np.eye(len(mu)) * cov_reg
        w_opts = {}
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(mu)
        n_assets = len(mu)

        w_prev_ca = np.array([w_drifted['cost_aware'].get(t, 0.0) for t in segment_tickers])
        res_ca = minimize(cost_aware_objective, w_prev_ca, args=(mu, sigma, w_prev_ca, turnover_lambda), method='SLSQP', bounds=bounds, constraints=cons)
        w_opts['cost_aware'] = res_ca.x

        w_initial_guess_classic = np.ones(n_assets) / n_assets
        res_cl = minimize(sharpe_objective_classic, w_initial_guess_classic, args=(mu, sigma), method='SLSQP', bounds=bounds, constraints=cons)
        w_opts['classic'] = res_cl.x

        for key in w_opts:
            w_opts[key][w_opts[key] < 1e-5] = 0
            if w_opts[key].sum() > 0: w_opts[key] /= w_opts[key].sum()

        w_prev_all = {'cost_aware': w_prev_ca, 'classic': np.array([w_drifted['classic'].get(t, 0.0) for t in segment_tickers])}

        for key in ['cost_aware', 'classic']:
            turnover = np.sum(np.abs(w_opts[key] - w_prev_all[key]))
            transaction_cost = turnover * cost_rate
            gross_returns = pd.Series(test_returns_seg.values @ w_opts[key], index=test_returns_seg.index)
            net_returns = gross_returns.copy()
            if not net_returns.empty:
                net_returns.iloc[0] = (1 - transaction_cost) * (1 + net_returns.iloc[0]) - 1
            final_returns, stop_date = apply_stop_loss(net_returns, stop_loss_pct)
            if stop_date:
                stop_loss_log[key].append(stop_date)
            results[key].append(pd.DataFrame({'ret': final_returns}))
            w_drifted[key] = calculate_drifted_weights({t: w for t, w in zip(segment_tickers, w_opts[key])}, test_returns_seg)

        w_prev_eq = np.array([w_drifted['equal_weight'].get(t, 0.0) for t in segment_tickers])
        w_eq = np.ones(len(segment_tickers)) / len(segment_tickers)
        eq_turnover = np.sum(np.abs(w_eq - w_prev_eq))
        eq_transaction_cost = eq_turnover * cost_rate
        eq_gross_returns = pd.Series(test_returns_seg.values @ w_eq, index=test_returns_seg.index)
        eq_net_returns = eq_gross_returns.copy()
        if not eq_net_returns.empty:
            eq_net_returns.iloc[0] = (1 - eq_transaction_cost) * (1 + eq_net_returns.iloc[0]) - 1
        eq_final_returns, eq_stop_date = apply_stop_loss(eq_net_returns, stop_loss_pct)
        if eq_stop_date:
            stop_loss_log['equal_weight'].append(eq_stop_date)
        results['equal_weight'].append(pd.DataFrame({'ret': eq_final_returns}))
        w_drifted['equal_weight'] = calculate_drifted_weights({t: w for t, w in zip(segment_tickers, w_eq)}, test_returns_seg)

        decision_date_idx = train_end_idx - 1
        weights_log.append((segment_tickers, w_opts['cost_aware'], w_opts['classic'], w_eq, decision_date_idx))
        progress_bar.progress((i + 1) / len(segments_to_process), text=f"Processing Segment {i+1}/{len(segments_to_process)}")

    final_dfs = {}
    for key, res_list in results.items():
        if res_list:
            df = pd.concat(res_list)
            df['cum_ret'] = (1 + df['ret']).cumprod()
            final_dfs[key] = df
    return final_dfs, weights_log, stop_loss_log


# --- Main Streamlit App UI ---
def main():
    st.set_page_config(page_title="Comparative Markowitz Optimizer", layout="wide")
    st.title("⚖️ Comparative Markowitz Walk-Forward Optimizer")
    st.info(__doc__.split('principles are:')[1])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Stock Universe")
    default_tickers = "AAPL, ADBE, ADI, ADP, ADSK, AKAM, AMAT, AMD, ANSS, ASML, AVGO, CDNS, CDW, CSCO, CTSH, DDOG, DOCU, EA, EBAY, ENPH, ENTG, FTNT, GFS, GOOGL, INTC, INTU, KLAC, LRCX, MCHP, MDB, META, MPWR, MSFT, MU, NVDA, NXPI, OKTA, ON, PANW, PAYX, PCAR, PTC, QCOM, QRVO, SNPS, TEAM, TTWO, TXN, WDAY, ZS, ALGN, AMGN, APLS, BIIB, BMRN, BPMC, CRSP, DXCM, EXAS, GILD, HALO, IDXX, ILMN, INCY, IONS, ISRG, MRNA, NBIX, REGN, RGEN, RPRX, SGEN, SRPT, TECH, UTHR, VCYT, VRTX, AMZN, BKNG, CMG, CPRT, CSGP, CZR, DASH, DRI, EXPE, HAS, LULU, MAR, MELI, NFLX, NKE, ORLY, PDD, ROST, SBUX, TSLA, ULTA, WYNN, YUM, ABNB, ETSY, LCID, LVS, TTD, VRSN, ZG"

    tickers_input = st.sidebar.text_area("Enter tickers (comma-separated)", default_tickers, height=150)
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameters")
    days = st.sidebar.slider("Total Historical Days", 200, 7000, 3000, 100)
    initial_equity = st.sidebar.number_input("Initial Equity ($)", 1000, 10000000, 100000, 1000)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimization Settings")
    train_window = st.sidebar.slider("Training Window (days)", 20, 500, 120, 10)
    test_window = st.sidebar.slider("Rebalance Window (days)", 1, 100, 5, 1)
    top_n_default = min(20, len(tickers)) if len(tickers) > 0 else 0
    top_n = st.sidebar.number_input("Portfolio Size (Top N Stocks)", 0, len(tickers), top_n_default, 1, help="0 uses all available stocks.")
    cost_pct = st.sidebar.number_input("Transaction Cost (%)", 0.0, 5.0, 0.8, 0.01)
    turnover_lambda = st.sidebar.number_input("Turnover Regularization (Lambda)", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Risk Management")
    stop_loss_pct = st.sidebar.number_input(
        "Portfolio Trailing Stop-Loss (%)", min_value=-1.0, max_value=50.0, value=-1.0, step=0.5,
        help="-1.0 to disable. If positive, this rule is applied to ALL strategies. "
             "It liquidates a strategy's segment portfolio if its equity drops by this percentage from its peak within that segment."
    )

    if not tickers:
        st.warning("Please enter at least two stock tickers to begin.")
        return

    if st.button("🚀 Run Comparative Simulation", use_container_width=True, type="primary"):
        try:
            price_df = get_survivorship_bias_free_data(tickers, days)
            if price_df.empty:
                st.error("Could not fetch valid data. Check tickers or increase 'Total Historical Days'.")
            else:
                results, weights_log, stop_loss_log = run_walk_forward_simulation(
                    price_df, train_window, test_window, top_n, cost_pct / 100,
                    turnover_lambda, 1e-4, stop_loss_pct
                )
                if not results:
                    st.error("Simulation failed. This is likely due to insufficient data for the chosen window parameters.")
                else:
                    st.session_state['results'] = results
                    st.session_state['weights_log'] = weights_log
                    st.session_state['stop_loss_log'] = stop_loss_log
                    st.session_state['price_df'] = price_df
                    st.session_state['initial_equity'] = initial_equity
                    st.success("Comparative Simulation Complete!")
        except Exception as e:
            st.error(f"An error occurred during the simulation: {e}")
            st.exception(e)

    st.markdown("---")

    if 'results' in st.session_state:
        results = st.session_state['results']
        weights_log = st.session_state['weights_log']
        price_df = st.session_state['price_df']
        initial_equity = st.session_state['initial_equity']
        stop_loss_log = st.session_state.get('stop_loss_log', {})

        st.header("🎯 Forward Trade Signal (based on Cost-Aware Strategy)")
        if weights_log:
            last_tickers, w_ca, _, _, last_decision_idx = weights_log[-1]
            trade_details = []
            for ticker, weight in zip(last_tickers, w_ca):
                if weight > 1e-5:
                    dollar_value = weight * initial_equity
                    price = price_df.iloc[last_decision_idx].get(ticker)
                    if pd.notna(price) and price > 0:
                        shares = np.floor(dollar_value / price)
                        trade_details.append({"Ticker": ticker, "Weight": weight, "Allocation ($)": dollar_value, "Price": price, "Shares to Buy": shares})
            if trade_details:
                trade_df = pd.DataFrame(trade_details)
                st.dataframe(trade_df.style.format({"Weight": "{:.2%}", "Allocation ($)": "${:,.2f}", "Price": "${:,.2f}"}), use_container_width=True)
            else:
                st.info("The Cost-Aware strategy recommends a 100% cash position for the next period.")

        st.header("🔙 Backtest Results")
        st.subheader("📈 Cumulative Returns")
        
        # --- NEW: Add checkboxes for marker visibility ---
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            show_rebalance_markers = st.checkbox("Show Rebalance ('o')", value=True)
        with col2:
            show_stop_loss_markers = st.checkbox("Show Stop-Loss ('x')", value=True)


        fig = go.Figure()
        colors = {'cost_aware': 'blue', 'classic': 'red', 'equal_weight': 'grey'}

        rebalance_dates = {price_df.index[log[-1] + 1] for log in weights_log if log[-1] + 1 < len(price_df.index)}

        for name, df in results.items():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['cum_ret'],
                name=name.replace('_', ' ').title(),
                line=dict(width=3 if name == 'cost_aware' else 2, color=colors[name],
                          dash='solid' if name != 'equal_weight' else 'dash'),
                legendgroup=name
            ))

            # --- MODIFIED: Conditionally add 'o' markers ---
            if show_rebalance_markers:
                rebal_points = df[df.index.isin(rebalance_dates)]
                if not rebal_points.empty:
                     fig.add_trace(go.Scatter(
                        x=rebal_points.index, y=rebal_points['cum_ret'],
                        mode='markers',
                        marker=dict(symbol='circle-open', color=colors[name], size=8, line=dict(width=2)),
                        name=f'{name} Rebalance',
                        showlegend=False, legendgroup=name, hoverinfo='none'
                     ))

            # --- MODIFIED: Conditionally add 'x' markers ---
            if show_stop_loss_markers:
                strategy_stop_dates = stop_loss_log.get(name, [])
                if strategy_stop_dates:
                    stop_points = df[df.index.isin(strategy_stop_dates)]
                    if not stop_points.empty:
                        fig.add_trace(go.Scatter(
                            x=stop_points.index, y=stop_points['cum_ret'],
                            mode='markers',
                            marker=dict(symbol='x', color='black', size=10, line=dict(width=2)),
                            name=f'{name} Stop Loss',
                            showlegend=False, legendgroup=name, hoverinfo='none'
                        ))
        
        # --- NEW: Dynamically build the title ---
        title_text = 'Strategy Equity Curves (Net of Costs)'
        subtitle_parts = []
        if show_rebalance_markers:
            subtitle_parts.append("o = Rebalance")
        if show_stop_loss_markers:
            subtitle_parts.append("x = Stop-Loss Trigger")
        if subtitle_parts:
            title_text += f"<br><sup>{', '.join(subtitle_parts)}</sup>"
            
        fig.update_layout(title=title_text,
                          yaxis_title="Equity Multiplier",
                          legend_title="Strategy", yaxis_tickformat=".2f")
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("⚙️ Performance Statistics (Annualized)")
        stats_data = {}
        for name, df in results.items():
            ret_series = df['ret']
            if not ret_series.empty and ret_series.std() != 0:
                stats_data[name.replace('_', ' ').title()] = {
                    "CAGR": calculate_cagr(ret_series), "Sharpe Ratio": calculate_sharpe_ratio(ret_series),
                    "Max Drawdown": calculate_max_drawdown(ret_series), "Volatility (Ann.)": calculate_volatility(ret_series)
                }
        stats_df = pd.DataFrame(stats_data).T
        st.dataframe(stats_df.style.format({"CAGR": "{:.2%}", "Max Drawdown": "{:.2%}", "Volatility (Ann.)": "{:.2%}", "Sharpe Ratio": "{:.2f}"}), use_container_width=True)

        st.subheader("⚖️ Historical Segment Allocations")
        segment_options = [f"Segment {i+1} (Decision on {price_df.index[d].date()})" for i, (_,_,_,_,d) in enumerate(weights_log)]
        if segment_options:
            selected_segment = st.selectbox("Select a historical segment to inspect:", reversed(segment_options))
            idx = segment_options.index(selected_segment)
            seg_tickers, seg_w_ca, seg_w_cl, _, _ = weights_log[idx]
            def create_trade_df(tickers, weights):
                df = pd.DataFrame([{'Ticker': t, 'Weight': w} for t, w in zip(tickers, weights) if w > 1e-5])
                return df.sort_values(by='Weight', ascending=False).set_index('Ticker')
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Cost-Aware Portfolio**")
                st.dataframe(create_trade_df(seg_tickers, seg_w_ca).style.format({'Weight': '{:.2%}'}), use_container_width=True)
            with col2:
                st.write("**Classic Portfolio**")
                st.dataframe(create_trade_df(seg_tickers, seg_w_cl).style.format({'Weight': '{:.2%}'}), use_container_width=True)


if __name__ == "__main__":
    main()
