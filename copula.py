#tfius 
import json
import inspect
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import norm # For Gaussian copula
from scipy.interpolate import interp1d
import talib

def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Displaying the first few entries to understand the structure of the data
    type(data), list(data.keys())
    return data

# Function to transform data to uniform marginals using empirical CDF
def to_uniform_marginals(data):
    ranks = rankdata(data)  # Get ranks of each data point
    uniform = (ranks - 0.5) / len(data)  # Convert ranks to uniform distribution
    return uniform

# empirical cumulative distribution function (ECDF) and its inverse
def create_ecdf(data):
    # Sort the data in ascending order
    sorted_data = np.sort(data)
    # Calculate percentile ranks
    percentiles = np.arange(len(sorted_data)) / len(sorted_data)
    # Create ECDF and inverse ECDF functions using interpolation
    ecdf = interp1d(sorted_data, percentiles, bounds_error=False, fill_value='extrapolate')
    inverse_ecdf = interp1d(percentiles, sorted_data, bounds_error=False, fill_value='extrapolate')
    return ecdf, inverse_ecdf



def apply_ta_indicator(indicator_name, data, *args, **kwargs):
    """
    Apply a TA-Lib indicator by its name.

    :param indicator_name: String, the name of the TA-Lib function (e.g., 'SMA', 'RSI').
    :param data: Pandas Series, the data on which to apply the indicator (e.g., closing prices).
    :param args: Additional positional arguments for the TA-Lib function.
    :param kwargs: Additional keyword arguments for the TA-Lib function.
    :return: The result of the TA-Lib function.
    """
    try:
        # Get the TA-Lib function by name
        func = getattr(talib, indicator_name)
        # Apply the function to the data
        result = func(data, *args, **kwargs)
        return result
    except AttributeError:
        print(f"TA-Lib does not have an indicator named '{indicator_name}'")
        return None

def apply_and_plot_ta_indicators(data, column_name):
    """
    Apply TA-Lib indicators to the specified column of a DataFrame and plot the results.

    :param data: Pandas DataFrame with financial data.
    :param column_name: String, the name of the column to which to apply the indicators.
    """

    # List of function categories to consider (e.g., functions that require a single price series)
    function_categories = ['Overlap Studies', 'Momentum Indicators', 'Volume Indicators', 'Volatility Indicators']
    # Get all TA-Lib functions grouped by category
    function_groups = talib.get_function_groups()
    print("function groups", function_groups)
    # Filter functions based on the selected categories
    selected_functions = [func for group in function_categories for func in function_groups[group]]

    # Filter functions that require a single input series
    # single_input_functions = [func for func in talib.get_functions() if 'real' in talib.get_function_groups()[func]]
    plt.figure(figsize=(10, 4))
    plt.title("Indicators")
    plt.plot(data.index, data[column_name], label='Price', alpha=0.5)
    for func_name in selected_functions:
        try:
            result = apply_ta_indicator(func_name, data[column_name], timeperiod=14)
            # Plot if the result is not None and is a valid series/array
            if result is not None and len(result) > 0:
                plt.plot(data.index, result, label=func_name)
         
        except Exception as e:
            print(f"INDICATOR Could not apply {func_name}: {e}")

    plt.legend()
    plt.show()

def apply_and_plot_ta_indicators_ohlc(data):
    """
    Apply TA-Lib indicators that require OHLC data and plot the results.

    :param data: Pandas DataFrame with OHLC data.
    """
    # List of function categories to consider (functions that usually require OHLC data)
    function_categories = ['Overlap Studies', 'Momentum Indicators', 'Volume Indicators', 'Volatility Indicators']
    
    # Get all TA-Lib functions grouped by category
    function_groups = talib.get_function_groups()
    
    # Filter functions based on the selected categories
    selected_functions = [func for group in function_categories for func in function_groups[group]]

    for func_name in selected_functions:
        try:
            # Get the TA-Lib function by name
            func = getattr(talib, func_name)

            # Extract the docstring and parse for parameters
            doc = func.__doc__
            # if doc:
            #     # Find the section after 'Parameters:'
            #     params_section = re.search('Inputs:(.*)', doc, re.DOTALL)
            #     if params_section:
            #         params_text = params_section.group(1)
            #         # Find all parameter names in the section
            #         params = re.findall(r'\b(timeperiod|int|real|high|low|close|open|volume)\b', params_text, re.IGNORECASE)
            if doc:
                # Extract the section between 'Inputs:' and 'Parameters:'
                match = re.search(r'Inputs:(.*?)Parameters:', doc, re.DOTALL)
                if match:
                    inputs_text = match.group(1)
                    # Find all input names in this section
                    params = re.findall(r'\b(timeperiod|int|real|high|low|close|open|volume)\b', inputs_text, re.IGNORECASE)



            print(f"OHLC Applying {func_name} with parameters: {params}")
            # Prepare the arguments based on the parsed parameters
            args = []
            for param in params:
                param = param.lower()  # Ensure lowercase for matching column names
                # capitalise the first letter of the parameter name
                param = param[0].upper() + param[1:]
                if param in data.columns:
                    args.append(data[param])
                elif param == 'Real':
                    args.append(data['Close'])  # Default to 'Close' for 'real'
                elif param == 'Timeperiod':  # Default time period if not specified
                    args.append(14)

            # Get the function's signature
            sig = inspect.signature(func)

            # Prepare the arguments based on the DataFrame columns and function signature
            # args = []
            # for param in sig.parameters.values():
            #     if param.name.upper() in data.columns:
            #         args.append(data[param.name.upper()])
            #     elif param.name == 'real':
            #         args.append(data['Close'])  # Default to 'Close' for 'real'
            #     elif param.name == 'timeperiod':
            #         args.append(14)  # Default time period


            # Apply the function with the prepared arguments
            result = func(*args)

            # Plotting
            plt.figure(figsize=(10, 4))
            plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)

            # Some functions return a tuple of arrays
            if isinstance(result, tuple):
                for i, res in enumerate(result):
                    if res is not None:
                        plt.plot(data.index, res, label=f'{func_name} - {i}')
            else:
                plt.plot(data.index, result, label=func_name)

            plt.title("OHLC " + func_name)
            plt.legend()
            plt.show()   

        except Exception as e:
            print(f"OHLC Could not apply {func_name}: {e}")

    #plt.show()


# Example Usage
# apply_and_plot_ta_indicators_ohlc(synthetic_ohlc_df)

            
# Function to generate synthetic OHLC data from daily returns
def generate_synthetic_ohlc(df, return_col):
    ohlc = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        "Volume": 0
    }

    # Starting value for the open (can be any arbitrary value)
    open_price = 100

    for daily_return in df[return_col]:
        close_price = open_price * (1 + daily_return)  # Apply the return to get the close price
        high_price = open_price * (1 + abs(daily_return) * 1.5)  # Example rule for high
        low_price = open_price * (1 - abs(daily_return) * 1.5)  # Example rule for low

        # Ensure Low is not higher than Open or Close, and High is not lower
        low_price = min(open_price, close_price, low_price)
        high_price = max(open_price, close_price, high_price)

        np.random.seed(0)  # For reproducibility
        volume_base = 1000000  # Base volume
        # Create volume data that somewhat correlates with the price movement
        ohlc['Volume'] = volume_base + (np.random.rand(len(df)) * 
                                             (high_price - low_price) * 
                                             volume_base / close_price)

        # Append to the lists
        ohlc['Open'].append(open_price)
        ohlc['High'].append(high_price)
        ohlc['Low'].append(low_price)
        ohlc['Close'].append(close_price)


        # Set the next open to the current close
        open_price = close_price

    return pd.DataFrame(ohlc, index=df.index)



bitcoin_file_path = './history/bitcoin.json'
ethereum_file_path = './history/ethereum.json'

bitcoin_data = load_data_from_file(bitcoin_file_path)
ethereum_data = load_data_from_file(ethereum_file_path)

# Converting the data into Pandas DataFrames for ease of manipulation
btc_df = pd.DataFrame(bitcoin_data['prices'], columns=['timestamp', 'btc_price'])
eth_df = pd.DataFrame(ethereum_data['prices'], columns=['timestamp', 'eth_price'])

# Converting timestamps from milliseconds to standard datetime format
btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')

# Setting the timestamp as the index
btc_df.set_index('timestamp', inplace=True)
eth_df.set_index('timestamp', inplace=True)

# Calculate daily returns
btc_df['btc_return'] = btc_df['btc_price'].pct_change()
eth_df['eth_return'] = eth_df['eth_price'].pct_change()

# Display the first few rows of both datasets to verify
btc_df.head(), eth_df.head()

# Basic statistics for Bitcoin and Ethereum returns
btc_stats = btc_df['btc_return'].describe()
eth_stats = eth_df['eth_return'].describe()

btc_stats, eth_stats

# Calculating the correlation coefficient between Bitcoin and Ethereum daily returns
correlation = btc_df['btc_return'].corr(eth_df['eth_return'])
correlation

# Apply the transformation to Bitcoin and Ethereum returns
btc_uniform = to_uniform_marginals(btc_df['btc_return'].dropna())
eth_uniform = to_uniform_marginals(eth_df['eth_return'].dropna())

# Creating a DataFrame for the transformed data
uniform_returns_df = pd.DataFrame({
    'btc_uniform': btc_uniform,
    'eth_uniform': eth_uniform
})

uniform_returns_df.head()

# Correlation coefficient
correlation_coefficient = 0.827

# Defining the 2x2 correlation matrix
correlation_matrix = np.array([
    [1, correlation_coefficient],
    [correlation_coefficient, 1]
])

print(correlation_matrix)

# Plotting the daily returns of Bitcoin and Ethereum
plt.figure(figsize=(12, 6))
plt.plot(btc_df.index, btc_df['btc_return'], label='Bitcoin')
plt.plot(eth_df.index, eth_df['eth_return'], label='Ethereum', alpha=0.7)
plt.title('Daily Returns of Bitcoin and Ethereum Over the Last Year')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
# plt.show()


# Step 2: Simulate data from the Gaussian copula
num_simulations = 10000
copula_samples = np.random.multivariate_normal(mean=[0, 0], cov=correlation_matrix, size=num_simulations)

# Convert the samples to uniform using the cumulative distribution function of the normal distribution
uniform_samples = norm.cdf(copula_samples)

# Create ECDF and inverse ECDF for Bitcoin and Ethereum returns
btc_ecdf, btc_inverse_ecdf = create_ecdf(btc_df['btc_return'].dropna())
eth_ecdf, eth_inverse_ecdf = create_ecdf(eth_df['eth_return'].dropna())

# Step 3: Transform the uniform data back to the original scale (returns)
# Assuming the empirical CDFs are represented by btc_ecdf and eth_ecdf functions (to be defined based on data)
btc_returns_simulated = btc_inverse_ecdf(uniform_samples[:, 0])
eth_returns_simulated = eth_inverse_ecdf(uniform_samples[:, 1])

# Step 4: Calculate VaR
confidence_level = 0.95
btc_var = np.percentile(btc_returns_simulated, 100 * (1 - confidence_level))
eth_var = np.percentile(eth_returns_simulated, 100 * (1 - confidence_level))

print(f"Bitcoin VaR at {confidence_level*100}% confidence level: {btc_var}")
print(f"Ethereum VaR at {confidence_level*100}% confidence level: {eth_var}")

# Calculate CVaR
btc_cvar = btc_returns_simulated[btc_returns_simulated <= btc_var].mean()
eth_cvar = eth_returns_simulated[eth_returns_simulated <= eth_var].mean()

print(f"Bitcoin CVaR at {confidence_level*100}% confidence level: {btc_cvar}")
print(f"Ethereum CVaR at {confidence_level*100}% confidence level: {eth_cvar}")


# Calculating Simple Moving Averages (SMA) for Bitcoin and Ethereum
btc_df['btc_30d_sma'] = btc_df['btc_price'].rolling(window=30).mean()
btc_df['btc_90d_sma'] = btc_df['btc_price'].rolling(window=90).mean()

eth_df['eth_30d_sma'] = eth_df['eth_price'].rolling(window=30).mean()
eth_df['eth_90d_sma'] = eth_df['eth_price'].rolling(window=90).mean()

# Plotting the SMAs along with the actual prices
plt.figure(figsize=(15, 8))

# Bitcoin
plt.subplot(2, 1, 1)
plt.plot(btc_df.index, btc_df['btc_price'], label='Bitcoin Price', color='blue')
plt.plot(btc_df.index, btc_df['btc_30d_sma'], label='30-Day SMA', color='orange')
plt.plot(btc_df.index, btc_df['btc_90d_sma'], label='90-Day SMA', color='green')
plt.title('Bitcoin Price and Simple Moving Averages')
plt.ylabel('Price')
plt.legend()

# Ethereum
plt.subplot(2, 1, 2)
plt.plot(eth_df.index, eth_df['eth_price'], label='Ethereum Price', color='blue')
plt.plot(eth_df.index, eth_df['eth_30d_sma'], label='30-Day SMA', color='orange')
plt.plot(eth_df.index, eth_df['eth_90d_sma'], label='90-Day SMA', color='green')
plt.title('Ethereum Price and Simple Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
#plt.show()

# List all functions in TA-Lib
# all_functions = talib.get_functions()

# Print each function
# for func in all_functions:
#    print(func)

# Example: Applying the Simple Moving Average (SMA) to Bitcoin prices
# btc_sma = apply_ta_indicator('SMA', btc_df['btc_price'], timeperiod=30)
# Example: Applying the Relative Strength Index (RSI) to Ethereum prices
# eth_rsi = apply_ta_indicator('RSI', eth_df['eth_price'], timeperiod=14)

apply_and_plot_ta_indicators(btc_df, 'btc_price')
#apply_and_plot_ta_indicators(eth_df, 'eth_price')

#plt.show()
# Generate synthetic OHLC data for Bitcoin
synthetic_btc_ohlc = generate_synthetic_ohlc(btc_df, 'btc_return')
synthetic_btc_ohlc.head()


apply_and_plot_ta_indicators_ohlc(synthetic_btc_ohlc)

plt.show()