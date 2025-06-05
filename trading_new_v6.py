import pandas as pd
import yfinance as yf
from datetime import timedelta
import time
from collections import defaultdict
import math

def fetch_unique_ticker_data(df, start_date, end_date):
    """Fetch historical data for unique tickers"""
    ticker_columns = ['Ticker A', 'Ticker B', 'Ticker C', 'Ticker D']
    unique_tickers = pd.unique(df[ticker_columns].values.ravel('K'))
    unique_tickers = [t for t in unique_tickers if pd.notna(t)]
    ticker_data = defaultdict(dict)
    print(f"Fetching data for {len(unique_tickers)} tickers...")
    
    for ticker in unique_tickers:
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date)
            if not data.empty:
                data.index = data.index.strftime('%Y-%m-%d')
                ticker_data[ticker] = data
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
    return ticker_data

# def compute_positions(df, threshold=0.005):
#     """Calculate daily trading units with dual prediction columns"""
#     daily_units = defaultdict(lambda: defaultdict(int))
#     ticker_groups = [['Ticker A', 'Ticker B'], ['Ticker C', 'Ticker D']]
#     df['var_check_pos'] = 0
#     df['var_check_neg'] = 0
    
#     for idx, row in df.iterrows():
#         date_str = row['Date']
#         pos_val = 1 if row['pred_avg_positive'] > threshold else (-1 if row['pred_avg_positive'] < -threshold else 0)
#         neg_val = 1 if row['pred_avg_negative'] > threshold else (-1 if row['pred_avg_negative'] < -threshold else 0)
#         df.at[idx, 'var_check_pos'] = pos_val
#         df.at[idx, 'var_check_neg'] = neg_val

#         signal = pos_val
        
#         if signal == 1:
#             for ticker in ticker_groups[0]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] += 1
#             for ticker in ticker_groups[1]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] -= 1
#         elif signal == -1:
#             for ticker in ticker_groups[0]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] -= 1
#             for ticker in ticker_groups[1]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] += 1
#         else:  # Handle signal=0 case
#             # Close all positions by setting units to 0 for all tickers
#             for group in ticker_groups:
#                 for ticker_col in group:
#                     ticker = row[ticker_col]
#                     if pd.notna(ticker):
#                         daily_units[date_str][ticker] = 0
                    
#     return daily_units, df

def compute_positions(df, threshold=0.005):
    """Calculate daily trading units with dual prediction columns"""
    # Use dict instead of defaultdict for the initial collection to have more control
    daily_signals = {}
    daily_units = defaultdict(lambda: defaultdict(int))
    ticker_groups = [['Ticker A', 'Ticker B'], ['Ticker C', 'Ticker D']]
    df['var_check_pos'] = 0
    # df['var_check_neg'] = 0
    
    # First pass: collect all signals by date and set signal flags
    for idx, row in df.iterrows():
        date_str = row['Date']
        pos_val = 1 if row['Predicted'] > threshold else (-1 if row['Predicted'] < -threshold else 0)
        # neg_val = 1 if row['pred_avg_negative'] > threshold else (-1 if row['pred_avg_negative'] < -threshold else 0)
        df.at[idx, 'var_check_pos'] = pos_val
        # df.at[idx, 'var_check_neg'] = neg_val
        
        # Store the signal with the tickers for this row
        if date_str not in daily_signals:
            daily_signals[date_str] = []
        
        # Store the signal and relevant tickers
        tickers = {}
        for group_idx, group in enumerate(ticker_groups):
            for ticker_col in group:
                ticker = row[ticker_col]
                if pd.notna(ticker):
                    tickers[(group_idx, ticker_col)] = ticker
        
        daily_signals[date_str].append((pos_val, tickers))
    
    # Second pass: process signals by date, prioritizing non-zero signals
    for date_str, signals in daily_signals.items():
        # First process all non-zero signals
        non_zero_processed_tickers = set()
        
        # Process non-zero signals first
        for signal, tickers in signals:
            if signal != 0:
                if signal == 1:
                    for (group_idx, ticker_col), ticker in tickers.items():
                        if group_idx == 0:  # First group (Ticker A, Ticker B)
                            daily_units[date_str][ticker] += 1
                            non_zero_processed_tickers.add(ticker)
                        else:  # Second group (Ticker C, Ticker D)
                            daily_units[date_str][ticker] -= 1
                            non_zero_processed_tickers.add(ticker)
                            
                elif signal == -1:
                    for (group_idx, ticker_col), ticker in tickers.items():
                        if group_idx == 0:  # First group (Ticker A, Ticker B)
                            daily_units[date_str][ticker] -= 1
                            non_zero_processed_tickers.add(ticker)
                        else:  # Second group (Ticker C, Ticker D)
                            daily_units[date_str][ticker] += 1
                            non_zero_processed_tickers.add(ticker)
        
        # Then process zero signals only for tickers that weren't already processed
        for signal, tickers in signals:
            if signal == 0:
                for (_, _), ticker in tickers.items():
                    if ticker not in non_zero_processed_tickers:
                        daily_units[date_str][ticker] = 0
    
    return daily_units, df
        
                    
#     return daily_units, df
# def compute_positions(df, threshold=0.005):
#     """Calculate daily trading units with dual prediction columns"""
#     daily_units = defaultdict(lambda: defaultdict(int))
#     ticker_groups = [['Ticker A', 'Ticker B'], ['Ticker C', 'Ticker D']]
#     df['var_check_pos'] = 0
#     df['var_check_neg'] = 0
    
#     for idx, row in df.iterrows():
#         date_str = row['Date']
#         pos_val = 1 if row['pred_avg_positive'] > threshold else (-1 if row['pred_avg_positive'] < -threshold else 0)
#         neg_val = 1 if row['pred_avg_negative'] > threshold else (-1 if row['pred_avg_negative'] < -threshold else 0)
#         df.at[idx, 'var_check_pos'] = pos_val
#         df.at[idx, 'var_check_neg'] = neg_val

#         signal = pos_val
        
#         if signal == 1:
#             for ticker in ticker_groups[0]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] += 1
#             for ticker in ticker_groups[1]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] -= 1
#         elif signal == -1:
#             for ticker in ticker_groups[0]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] -= 1
#             for ticker in ticker_groups[1]:
#                 if pd.notna(row[ticker]):
#                     daily_units[date_str][row[ticker]] += 1
    #     else:  # Handle signal=0 case
    #         # Close all positions by setting units to 0 for all tickers
    #         for group in ticker_groups:
    #             for ticker_col in group:
    #                 ticker = row[ticker_col]
    #                 if pd.notna(ticker):
    #                     daily_units[date_str][ticker] = 0
                    
    # return daily_units, df

def calculate_shares_and_trading_costs(daily_units, ticker_data, df):
    """Calculate daily trading metrics with new position sizing and trading rules"""
    results = []
    cash = 100000
    prev_shares = defaultdict(int)
    inventory = defaultdict(int)
    dates = pd.to_datetime(df['Date'], format='%d-%m-%Y').sort_values().unique()
    prev_day_nav = None  # Track previous day's NAV

    for date in dates:
        date_str = date.strftime('%d-%m-%Y')
        yfinance_date = date.strftime('%Y-%m-%d')
        daily_data = daily_units.get(date_str, {})
        total_trading_cost = 0
        daily_cash_impact = 0
        current_prices = {}
        shares_data = {}
        
        # Calculate position sizing parameters
        sum_abs_units = sum(abs(u) for u in daily_data.values())
        prev_nav = prev_day_nav if prev_day_nav is not None else 100000

        # First pass: calculate all changes
        for ticker, units in daily_data.items():
            if ticker in ticker_data and yfinance_date in ticker_data[ticker].index:
                price = ticker_data[ticker].loc[yfinance_date, 'Close']
                
                # Calculate position size
                if sum_abs_units > 100:
                    position_size = (1/sum_abs_units) * units * prev_nav
                else:
                    position_size = 0.01 * units * prev_nav
                
                # Initialize delta and shares
                shares = 0
                delta = 0
                
                # Trading conditions
                if price <= 30:
                    # Liquidate position if price <= 30
                    shares = 0
                    delta = -prev_shares.get(ticker, 0)
                else:
                    # Calculate original shares
                    shares = math.ceil(abs(position_size) / price) * (1 if units >= 0 else -1)
                    delta = shares - prev_shares.get(ticker, 0)
                    
                    # Check delta threshold for stocks >30
                    if abs(delta) <= 50:
                        shares = 0
                        delta = -prev_shares.get(ticker, 0)

                current_prices[ticker] = price
                shares_data[ticker] = shares
                daily_cash_impact += delta * price
                total_trading_cost += math.ceil(abs(delta) / 100)

        # Update cash balance
        cash -= daily_cash_impact
        
        # Calculate NAV
        nav = cash + sum(shares * current_prices.get(t, 0) for t, shares in shares_data.items())
        prev_day_nav = nav  # Store for next day's calculation

        # Create records
        for ticker in daily_data:
            if ticker in current_prices:
                units = daily_data[ticker]
                # Recalculate position size for output
                if sum_abs_units > 100:
                    position_size = (1/sum_abs_units) * units * prev_nav
                else:
                    position_size = 0.01 * units * prev_nav
                
                results.append({
                    'Date': date_str,
                    'Ticker Name': ticker,
                    'Year': date.strftime('%Y'),
                    'No. Of Units': units,
                    'Adj Closing Price': round(current_prices[ticker], 6),
                    'Position Size': position_size,
                    'No. Of Shares': shares_data[ticker],
                    'Delta Shares': shares_data[ticker] - prev_shares.get(ticker, 0),
                    'Trading Cost': math.ceil(abs(shares_data[ticker] - prev_shares.get(ticker, 0)) / 100),
                    'Inventory': shares_data[ticker],
                    'Total Trading Cost': total_trading_cost,
                    'Cash': cash,
                    'NAV': nav
                })

        # Update previous shares after processing all tickers
        prev_shares.update(shares_data)

    return pd.DataFrame(results)

def main():
    """Main execution flow"""
    try:
        # Load and process data
        df = pd.read_csv('2022_2000d_comp.csv')
        daily_units, modified_df = compute_positions(df, threshold=0.005)

        # Set date range with buffer
        dates = pd.to_datetime(modified_df['Date'], format='%d-%m-%Y')
        start_date = dates.min() - timedelta(days=5)
        end_date = dates.max() + timedelta(days=5)

        # Fetch market data
        ticker_data = fetch_unique_ticker_data(
            modified_df,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Generate final results
        trading_results = calculate_shares_and_trading_costs(daily_units, ticker_data, modified_df)

        # Ensure column order and formatting
        output_columns = [
            'Date', 'Year', 'Ticker Name', 'No. Of Units', 'Adj Closing Price',
            'Position Size', 'No. Of Shares', 'Delta Shares', 'Trading Cost',
            'Inventory', 'Total Trading Cost', 'Cash', 'NAV'
        ]
        trading_results = trading_results[output_columns]
        trading_results.to_csv('trading_res_2022_2000.csv', index=False)
        print("Processing completed successfully")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
