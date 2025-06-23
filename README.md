# Comparative Markowitz Walk-Forward Optimizer

A robust, point-in-time backtesting application that implements a comparative analysis between Cost-Aware and Classic Markowitz portfolio optimization strategies. This application is specifically designed to eliminate common backtesting biases including data leakage, look-ahead bias, and survivorship bias.

## 🎯 Key Features

- **Cost-Aware vs Classic Markowitz Comparison**: Implements both optimization strategies with transaction cost modeling
- **Walk-Forward Backtesting**: Point-in-time simulation methodology
- **Dynamic Universe Selection**: Eliminates survivorship bias through real-time asset filtering
- **Drift-Aware Cost Calculation**: Accurate transaction cost modeling based on portfolio drift
- **Risk Management**: Optional trailing stop-loss applied equally across all strategies
- **Interactive Streamlit Interface**: User-friendly web application for parameter tuning and analysis

## 🚫 Bias-Free Design Principles

### 1. Data Leakage Prevention

This application is designed to be completely free of data leakage through several key mechanisms:

#### Point-in-Time Data Access
- **Training Window Isolation**: Each optimization segment uses only historical data within a strictly defined training window
- **No Future Information**: All calculations are performed using only data available at the decision point
- **Segmented Processing**: The simulation processes data in discrete, non-overlapping segments to prevent information spillover

#### Forward Signal Generation
```python
# The forward signal for the next trading period is correctly generated 
# using the price on the decision date, not future prices
decision_date_idx = train_end_idx - 1
```

### 2. Look-Ahead Bias Elimination

#### Neutral Initialization
- **Classic Markowitz Seeding**: The Classic Markowitz optimizer is seeded with a neutral guess (equal weight) to ensure fair comparison
- **No Prior Knowledge**: Optimization starts without any information about future performance

#### Strict Temporal Boundaries
- **Training Period**: Uses data from `train_start_idx` to `train_end_idx`
- **Testing Period**: Applies optimized weights to data from `test_start_idx` to `test_end_idx`
- **No Overlap**: Training and testing periods are strictly separated

### 3. Survivorship Bias Elimination

The most sophisticated aspect of this application is its **Dynamic Universe Selection** mechanism:

#### Master Universe Approach
```python
@st.cache_data(show_spinner="Fetching historical data for master universe...")
def get_survivorship_bias_free_data(master_ticker_list: list[str], period_days: int) -> pd.DataFrame:
    """
    Downloads historical data for a master list of tickers, creating a
    point-in-time-aware DataFrame. NaN values correctly represent periods
    where a stock was not listed or data is unavailable.
    """
```

#### Point-in-Time Universe Filtering
```python
def get_tradable_universe_for_segment(price_df: pd.DataFrame, start_idx: int, end_idx: int) -> list[str]:
    """
    Determines the tradable universe for a specific time segment by filtering
    the master price history for assets with continuous data.
    """
    training_period_prices = price_df.iloc[start_idx:end_idx]
    tradable_assets = training_period_prices.dropna(axis=1, how='any')
    return tradable_assets.columns.tolist()
```

#### How Dynamic Universe Selection Works

1. **Master Data Download**: Initially downloads data for all requested tickers, including those that may have been delisted
2. **Segment-Specific Filtering**: For each training segment, determines which assets existed and had continuous data throughout the entire lookback period
3. **Natural Handling of Corporate Events**:
   - **IPOs**: New stocks automatically appear in the universe when they have sufficient historical data
   - **Delistings**: Delisted stocks naturally disappear from the universe when they no longer have data
   - **Trading Halts**: Stocks with missing data are excluded from that specific segment

#### Example Timeline
```
Segment 1 (Days 1-120): AAPL, MSFT, GOOGL, AMZN, TSLA
Segment 2 (Days 6-125): AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA (IPO)
Segment 3 (Days 11-130): AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA
Segment 4 (Days 16-135): AAPL, MSFT, GOOGL, AMZN, NVDA (TSLA delisted)
```

### 4. Drift-Aware Cost Modeling

#### Portfolio Drift Calculation
```python
def calculate_drifted_weights(start_weights: dict, period_returns: pd.DataFrame) -> dict:
    """
    Calculates the drifted portfolio weights after a period of market movement.
    """
    cumulative_returns = (1 + period_returns).prod()
    end_values = {
        ticker: start_weights.get(ticker, 0) * cumulative_returns.get(ticker, 1)
        for ticker in start_weights
    }
```

#### Accurate Transaction Cost Calculation
- **Drift-Based Turnover**: Calculates turnover against drifted weights, not previous target weights
- **Realistic Cost Impact**: Transaction costs are applied only to the actual portfolio changes
- **Fair Comparison**: Both strategies experience the same cost calculation methodology

## 📊 Walk-Forward Methodology

### Segment Processing
1. **Training Phase**: Uses historical data to estimate returns and covariance
2. **Optimization Phase**: Solves for optimal weights using point-in-time information
3. **Testing Phase**: Applies weights to out-of-sample data
4. **Rolling Window**: Advances the window and repeats the process

### Fair Strategy Comparison
- **Equal Treatment**: All strategies use the same universe and time periods
- **Consistent Risk Management**: Stop-loss rules applied equally across strategies
- **Identical Cost Structure**: Same transaction cost rates and calculation methods

## 🛠️ Installation and Usage

### Requirements
```bash
pip install streamlit yfinance pandas plotly numpy quantstats scipy
```

### Running the Application
```bash
streamlit run app_markowitz.py
```

### Key Parameters

#### Universe Selection
- **Tickers**: Comma-separated list of stock symbols
- **Historical Period**: Number of days of historical data to fetch

#### Optimization Settings
- **Training Window**: Days used for parameter estimation
- **Rebalance Window**: Frequency of portfolio rebalancing
- **Portfolio Size**: Number of top stocks to include (momentum-based selection)
- **Transaction Cost**: Percentage cost per trade
- **Turnover Regularization**: Lambda parameter for cost-aware optimization

#### Risk Management
- **Stop-Loss**: Optional trailing stop-loss percentage (applied to all strategies)

## 📈 Output and Analysis

### Forward Trade Signal
- **Current Allocation**: Shows the recommended portfolio for the next period
- **Trade Details**: Dollar allocations, share quantities, and current prices

### Backtest Results
- **Equity Curves**: Cumulative returns for all strategies
- **Performance Statistics**: Annualized metrics including CAGR, Sharpe ratio, max drawdown
- **Historical Allocations**: Detailed portfolio weights for each historical segment

### Visualization Features
- **Interactive Charts**: Plotly-based equity curves with rebalance markers
- **Performance Comparison**: Side-by-side strategy analysis
- **Risk Metrics**: Comprehensive risk and return statistics

## 🔬 Technical Implementation Details

### Optimization Algorithms
- **Cost-Aware Markowitz**: Maximizes Sharpe ratio while penalizing turnover
- **Classic Markowitz**: Traditional mean-variance optimization
- **Equal-Weight Benchmark**: Simple equal allocation strategy

### Data Handling
- **yfinance Integration**: Real-time market data fetching
- **Pandas MultiIndex**: Efficient handling of multi-asset time series
- **NaN Management**: Proper handling of missing data and delisted stocks

### Performance Optimization
- **Streamlit Caching**: Efficient data loading and computation caching
- **Vectorized Operations**: NumPy-based calculations for speed
- **Memory Management**: Efficient DataFrame operations and garbage collection

## 🎯 Why This Implementation is Bias-Free

### 1. **No Information Leakage**
- Each optimization decision uses only historical data available at that point in time
- Training and testing periods are strictly separated
- No future information is used in any calculation

### 2. **No Look-Ahead Bias**
- All data access is point-in-time
- Initial conditions are neutral (equal weights for Classic Markowitz)
- Forward signals are generated using current, not future, prices

### 3. **No Survivorship Bias**
- Dynamic universe selection ensures only stocks that existed during the training period are considered
- Delisted stocks naturally disappear from the universe
- New IPOs are only included when they have sufficient historical data

### 4. **Realistic Transaction Costs**
- Costs are calculated based on actual portfolio drift
- Turnover is measured against drifted weights, not target weights
- All strategies experience the same cost structure

## 📚 Academic Context

This implementation follows best practices from academic literature on backtesting methodology:

- **Campbell, Lo, and MacKinlay (1997)**: Point-in-time data requirements
- **Lopez de Prado (2018)**: Walk-forward analysis methodology
- **Arnott, Beck, and Kalesnik (2016)**: Survivorship bias prevention techniques

## 🤝 Contributing

Contributions are welcome! Please ensure that any modifications maintain the bias-free design principles outlined above.

email: ekoshv.igt@gmail.com

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This application is designed for educational and research purposes. Past performance does not guarantee future results, and this tool should not be used as the sole basis for investment decisions. 
