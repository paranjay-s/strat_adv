"""
    interval: 60, "D"
    from: integer from timestamp in seconds
    limit: max size of 200
"""

from config_strategy_api import session
from config_strategy_api import timeframe
from config_strategy_api import kline_limit
import datetime
import time
 
 
# Get start times 
time_start_date = 0 
if timeframe == 60: 
    time_start_date = datetime.datetime.now() - datetime.timedelta(hours=kline_limit)
if timeframe == "D": 
    time_start_date = datetime.datetime.now() - datetime.timedelta(days=kline_limit)
 
time_start_milliseconds = int(time_start_date.timestamp())*1000 
 
 
 
#Get Historical Prices (klines)
def get_price_klines(symbol):
 
    #Get prices
    prices = session.get_mark_price_kline(
        symbol = symbol,
        interval = timeframe,
        limit = kline_limit,
        start = time_start_milliseconds
    )
 
    #Manage API calls
    time.sleep(0.1)
 
 
    #Return output
    if len(prices["result"]["list"]) != kline_limit:
        return []
    return prices["result"]["list"]


# # Get Extended Historical Prices (klines)
# def get_extended_price_klines(symbol, intervals=10):
#     """
#     Fetch extended price history by breaking the data into multiple intervals.
#     """
#     all_prices = []
#     for i in range(intervals):
#         try:
#             # Calculate start and end times for each interval
#             start_time = time_start_milliseconds + (i * kline_limit * timeframe * 1000)
#             end_time = start_time + (kline_limit * timeframe * 1000)
            
#             # Fetch prices for the interval
#             prices = session.get_mark_price_kline(
#                 symbol=symbol,
#                 interval=timeframe,
#                 limit=kline_limit,
#                 start=start_time,
#                 end=end_time
#             )
            
#             # Append results
#             if "result" in prices and "list" in prices["result"]:
#                 all_prices.extend(prices["result"]["list"])
            
#             # Avoid API rate limits
#             time.sleep(0.1)
#         except Exception as e:
#             print(f"Error fetching extended price data for {symbol}: {str(e)}")
#             continue
    
#     # Remove duplicates and sort by timestamp
#     unique_prices = {item[0]: item for item in all_prices}.values()
#     sorted_prices = sorted(unique_prices, key=lambda x: int(x[0]))
    
#     return sorted_prices



# """
#     interval: 60, "D"
#     from: integer from timestamp in seconds
#     limit: max size of 200
# """
# from config_strategy_api import session
# from config_strategy_api import timeframe
# import datetime
# import time

# """
#     interval: 60, "D"
#     from: integer from timestamp in seconds
#     limit: max size of 200
# """
# from config_strategy_api import session
# from config_strategy_api import timeframe
# import datetime
# import time

# # Get Historical Prices (klines) with extended range
# def get_price_klines(symbol, total_data_points=1000):
#     """
#     Fetch historical price data in batches to exceed the 200-point limit.
#     :param symbol: Trading pair symbol (e.g., "BTCUSDT")
#     :param total_data_points: Total number of data points to fetch
#     :return: List of klines
#     """
#     # Define constants
#     batch_size = 200  # Maximum allowed by the API
#     all_prices = []
    
#     # Calculate start and end times for the first batch
#     current_time = int(datetime.datetime.now().timestamp() * 1000)  # Current time in milliseconds
    
#     while len(all_prices) < total_data_points:
#         # Calculate start and end timestamps for the current batch
#         end_time = current_time
#         start_time = end_time - (batch_size * timeframe * 1000)
        
#         # Fetch data for the current batch
#         prices = session.get_mark_price_kline(
#             category="linear",  # Default to linear contracts
#             symbol=symbol,
#             interval=timeframe,
#             start=start_time,
#             end=end_time,
#             limit=batch_size
#         )
        
#         # Check if data is valid
#         if "result" not in prices or "list" not in prices["result"] or len(prices["result"]["list"]) == 0:
#             print(f"No more data available for {symbol}. Stopping.")
#             break
        
#         # Add the fetched data to the list
#         batch_prices = prices["result"]["list"]
#         all_prices = batch_prices + all_prices  # Prepend to maintain chronological order
        
#         # Log the batch details for debugging
#         print(f"Fetched {len(batch_prices)} points starting at {start_time}")
        
#         # Update the current time for the next batch
#         current_time = start_time
        
#         # Avoid hitting API rate limits
#         time.sleep(0.1)
        
#         # Stop if we've fetched enough data
#         if len(all_prices) >= total_data_points:
#             all_prices = all_prices[:total_data_points]  # Trim to the desired number of points
#             break
    
#     return all_prices




# import requests
# import json
# import pandas as pd
# import datetime as dt
# from config_strategy_api import session
# from config_strategy_api import timeframe
# import datetime
# import time


# import requests
# import json
# import pandas as pd
# import datetime as dt

# def get_bybit_bars(symbol, interval, startTime, endTime):
#     """
#     Fetch historical price data from Bybit API.
#     """
#     url = "https://api.bybit.com/v2/public/kline/list"
#     startTime = str(int(startTime.timestamp()))
#     endTime = str(int(endTime.timestamp()))
#     req_params = {"symbol": symbol, "interval": interval, "from": startTime, "to": endTime}
    
#     response = requests.get(url, params=req_params)
#     data = json.loads(response.text)
    
#     if "result" not in data or not data["result"]:
#         return None
    
#     df = pd.DataFrame(data["result"])
#     df.index = [dt.datetime.fromtimestamp(x) for x in df["open_time"].astype(float)]
#     return df

# def get_extended_price_klines(symbol, interval=60, max_points=4000):
#     """
#     Fetch extended historical price data by iterating over time intervals.
#     """
#     df_list = []
#     last_datetime = dt.datetime.now() - dt.timedelta(days=180)  # Start from 6 months ago
#     end_datetime = dt.datetime.now()
    
#     while True:
#         print(f"Fetching data from {last_datetime} to {end_datetime}")
#         new_df = get_bybit_bars(symbol, interval, last_datetime, end_datetime)
        
#         if new_df is None or len(new_df) == 0:
#             break
        
#         df_list.append(new_df)
#         last_datetime = max(new_df.index) + dt.timedelta(seconds=1)
        
#         # Stop if we've fetched enough data
#         if sum(len(df) for df in df_list) >= max_points:
#             break
        
#         # Avoid hitting API rate limits
#         time.sleep(0.2)
    
#     # Combine all chunks into a single DataFrame
#     df = pd.concat(df_list).sort_index()
#     return df.iloc[-max_points:]  # Return only the last `max_points` rows