

# The table provided contains several key performance metrics for a systematic quantitative trading strategy. Below is an explanation of each metric along with the relevant formulas for their calculation:

# ## Key Metrics and Formulas

# ### **1. Number of Trades**
# - **# Trades**: Total number of trades executed.
# - **# Trades / year**: Average number of trades executed per year.

# ### **2. Average Return per Trade [%]**
# - **Formula**: $$\text{Avg. Return per Trade} = \frac{\text{Total Return}}{\text{Total Number of Trades}}$$
# - **Explanation**: This metric indicates the average percentage return per trade.

# ### **3. Best/Worst Trade [%]**
# - **Best Trade**: The highest percentage return from a single trade.
# - **Worst Trade**: The lowest percentage return from a single trade.

# ### **4. Trade Duration**
# - **Max. Trade Duration [days]**: Longest duration of any single trade.
# - **Avg. Trade Duration [days]**: Average duration of all trades.
#   - **Formula**: $$\text{Avg. Trade Duration} = \frac{\text{Total Duration of All Trades}}{\text{Total Number of Trades}}$$

# ### **5. Win Rate [%]**
# - **Formula**: $$\text{Win Rate} = \left( \frac{\text{Number of Winning Trades}}{\text{Total Number of Trades}} \right) \times 100$$
# - **Explanation**: Percentage of trades that were profitable.

# ### **6. Profit Factor**
# - **Formula**: $$\text{Profit Factor} = \frac{\text{Total Profit}}{\text{Total Loss}}$$
# - **Explanation**: Ratio of total profits to total losses. A value greater than 1 indicates profitability.

# ### **7. Win/Loss Ratio**
# - **Formula**: $$\text{Win/Loss Ratio} = \frac{\text{Number of Winning Trades}}{\text{Number of Losing Trades}}$$
# - **Explanation**: Ratio of the number of winning trades to losing trades.

# ### **8. Payoff Ratio**
# - **Formula**: $$\text{Payoff Ratio} = \frac{\text{Average Profit per Winning Trade}}{\text{Average Loss per Losing Trade}}$$
# - **Explanation**: Ratio of the average profit of winning trades to the average loss of losing trades.

# ### **9. CPC Index**
# - **Explanation**: This metric is not standard and may refer to a custom performance measure specific to the strategy. CPC could stand for "Cost Per Click" in other contexts, but in trading, it might be a proprietary index.

# ### **10. Expectancy [$]**
# - **Formula**: $$\text{Expectancy} = \left( \text{Win Rate} \times \text{Average Win Size} \right) - \left( \text{Loss Rate} \times \text{Average Loss Size} \right)$$
# - **Explanation**: Average amount expected to be made per trade. A positive expectancy indicates a profitable strategy.

# ## Example Calculations

# ### **Average Return per Trade [%]**
# Given:
# - Total Return: 398.58%
# - Total Number of Trades: 4,518

# $$\text{Avg. Return per Trade} = \frac{398.58}{4518} \approx 0.88\%$$

# ### **Win Rate [%]**
# Given:
# - Number of Winning Trades: 3,001
# - Total Number of Trades: 4,518

# $$\text{Win Rate} = \left( \frac{3001}{4518} \right) \times 100 \approx 66.4\%$$

# ### **Profit Factor**
# Given:
# - Total Profit: Sum of all profits from winning trades
# - Total Loss: Sum of all losses from losing trades

# $$\text{Profit Factor} = \frac{\text{Total Profit}}{\text{Total Loss}} = 1.41$$

# ### **Expectancy [$]**
# Given:
# - Win Rate: 66.4%
# - Average Win Size: $3,200 (assuming $100 per 1%)
# - Loss Rate: 33.6% (100% - Win Rate)
# - Average Loss Size: $3,700 (assuming $100 per 1%)

# $$\text{Expectancy} = (0.664 \times 3200) - (0.336 \times 3700) = 2124.8 - 1243.2 = 881.6$$

# These metrics provide a comprehensive view of the trading strategy's performance and can be used to assess its profitability and risk profile.

# Citations:
# [1] https://www.forex.academy/how-to-count-win-rate-and-profit-factors-forex/
# [2] https://www.investopedia.com/terms/a/averagereturn.asp
# [3] https://www.pineconnector.com/blogs/pico-blog/what-is-expectancy-ratio
# [4] https://www.forex.in.rs/pay-off-ratio/
# [5] https://www.incredibuild.com/blog/quant-trading-strategies-definitions-and-importance
# [6] https://wiki.paperswithbacktest.com/glossary/quantitative-trading-strategies
# [7] https://www.investopedia.com/terms/q/quantitative-trading.asp
# [8] https://blog.quantinsti.com/performance-metrics-risk-metrics-optimization/
# [9] https://pplx-res.cloudinary.com/image/upload/v1722621440/user_uploads/gocffkpte/Stats-Trading.jpg

# Certainly! Below is a Python code function that calculates the relevant statistics from a daily cumulative return series of a trading strategy. This function assumes you have a Pandas DataFrame with a column for daily cumulative returns.

To modify the code to calculate the relevant statistics based on the given cumulative strategy return and the weights DataFrame, we need to identify trades based on changes in the weights. Here is the updated Python code:

```python
import pandas as pd
import numpy as np

def calculate_trading_statistics(df, weights):
    """
    Calculate trading statistics from a DataFrame of daily cumulative returns and weights.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'cumulative_return' column.
    weights (pd.DataFrame): DataFrame containing weights of the assets traded.
    
    Returns:
    dict: Dictionary containing calculated statistics.
    
    'Number of Trades': num_trades,
    'Average Return per Trade (%)': avg_return_per_trade,
    'Best Trade (%)': best_trade,
    'Worst Trade (%)': worst_trade,
    'Max Trade Duration (days)': max_trade_duration,
    'Avg Trade Duration (days)': avg_trade_duration,
    'Win Rate (%)': win_rate,
    'Profit Factor': profit_factor,
    'Win/Loss Ratio': win_loss_ratio,
    'Payoff Ratio': payoff_ratio,
    'CPC Index': cpc_index,
    'Expectancy ($)': expectancy
    


    """
    # Calculate daily returns
    df['daily_return'] = df['cumulative_return'].pct_change().fillna(0)
    
    # Identify trades based on changes in weights
    trades = ((weights > 0).diff() != 0).any(axis=1).astype(int)
    df['trade'] = trades
    
    # Calculate number of trades
    num_trades = df['trade'].sum()
    
    # Calculate average return per trade
    avg_return_per_trade = df['daily_return'][df['trade'] == 1].mean() * 100
    
    # Calculate best and worst trades
    best_trade = df['daily_return'][df['trade'] == 1].max() * 100
    worst_trade = df['daily_return'][df['trade'] == 1].min() * 100
    
    # Calculate trade durations
    trade_durations = df['trade'].groupby((df['trade'] != df['trade'].shift()).cumsum()).cumsum()
    max_trade_duration = trade_durations.max()
    avg_trade_duration = trade_durations.mean()
    
    # Calculate win rate
    win_rate = (df['daily_return'][df['trade'] == 1] > 0).sum() / num_trades * 100
    
    # Calculate profit factor
    total_profit = df['daily_return'][df['daily_return'] > 0].sum()
    total_loss = -df['daily_return'][df['daily_return'] < 0].sum()
    profit_factor = total_profit / total_loss if total_loss != 0 else np.inf
    
    # Calculate win/loss ratio
    num_winning_trades = (df['daily_return'][df['trade'] == 1] > 0).sum()
    num_losing_trades = (df['daily_return'][df['trade'] == 1] < 0).sum()
    win_loss_ratio = num_winning_trades / num_losing_trades if num_losing_trades != 0 else np.inf
    
    # Calculate payoff ratio
    avg_profit_per_winning_trade = df['daily_return'][df['daily_return'] > 0].mean()
    avg_loss_per_losing_trade = -df['daily_return'][df['daily_return'] < 0].mean()
    payoff_ratio = avg_profit_per_winning_trade / avg_loss_per_losing_trade if avg_loss_per_losing_trade != 0 else np.inf
    
    # Calculate expectancy
    expectancy = (win_rate / 100 * avg_profit_per_winning_trade) - ((100 - win_rate) / 100 * avg_loss_per_losing_trade)
    
    # CPC Index (custom metric, assumed to be the same as profit factor for this example)
    cpc_index = profit_factor
    
    statistics = {
        'Number of Trades': num_trades,
        'Average Return per Trade (%)': avg_return_per_trade,
        'Best Trade (%)': best_trade,
        'Worst Trade (%)': worst_trade,
        'Max Trade Duration (days)': max_trade_duration,
        'Avg Trade Duration (days)': avg_trade_duration,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Win/Loss Ratio': win_loss_ratio,
        'Payoff Ratio': payoff_ratio,
        'CPC Index': cpc_index,
        'Expectancy ($)': expectancy
    }
    
    return statistics

# Example usage
# Assuming df is a DataFrame with a 'cumulative_return' column
# and weights is a DataFrame with the weights of the assets traded
# df = pd.read_csv('cumulative_returns.csv')
# weights = pd.read_csv('weights.csv')
# stats = calculate_trading_statistics(df, weights)
# print(stats)
```

# ### Explanation of Changes:
# 1. **Trade Identification**: Trades are identified based on changes in the weights of the assets traded. If there is any change in the weights from one day to the next, it is considered a trade.
# 2. **Daily Returns Calculation**: The daily returns are calculated using the percentage change of the cumulative returns.
# 3. **Number of Trades**: The total number of trades is the sum of the identified trades.
# 4. **Average Return per Trade**: The mean of the daily returns where trades were identified.
# 5. **Best and Worst Trades**: The maximum and minimum daily returns where trades were identified.
# 6. **Trade Durations**: The duration of each trade is calculated, and the maximum and average durations are derived.
# 7. **Win Rate**: The percentage of positive daily returns where trades were identified.
# 8. **Profit Factor**: The ratio of the sum of positive returns to the sum of negative returns.
# 9. **Win/Loss Ratio**: The ratio of the number of winning trades to losing trades.
# 10. **Payoff Ratio**: The ratio of the average profit per winning trade to the average loss per losing trade.
# 11. **Expectancy**: The expected return per trade considering the win rate and average returns.
# 12. **CPC Index**: Assumed to be the same as the profit factor for this example.

# This modified function now takes into account the changes in weights to identify trades and calculate the relevant statistics.

# Citations:
# [1] https://pplx-res.cloudinary.com/image/upload/v1722621440/user_uploads/gocffkpte/Stats-Trading.jpg

