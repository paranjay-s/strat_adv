#Store Price History for All Available Pairs
from func_price_klines import get_price_klines
import json
 
 
def store_price_history(symbols):
 
    #Get Prices and Store in DataFrame
    counts = 0
    price_history_dict = {}
    for sym in symbols:
        symbol_name = sym["symbol"] 
        price_history = get_price_klines(symbol_name)
        if len(price_history) > 0:
            price_history_dict[symbol_name] = price_history 
            counts += 1
            print (f"{counts} items stored")
        else:
            print ("item not stored")
    #Output prices to JSON 
    if len(price_history_dict) > 0:
        with open("1_price_list.json", "w") as fp:
            json.dump(price_history_dict, fp, indent=4)
        print("Prices saved successfully.")
 
    # Return output
    return

# # Store Price History for All Available Pairs
# from func_price_klines import get_price_klines
# import json



# from func_price_klines import get_extended_price_klines
# import json

# def store_price_history(symbols):
#     """
#     Store extended price history for all available pairs.
#     """
#     counts = 0
#     price_history_dict = {}
    
#     for sym in symbols:
#         symbol_name = sym["symbol"]
#         price_history = get_extended_price_klines(symbol_name, interval=1, max_points=4000)
        
#         if len(price_history) > 0:
#             price_history_dict[symbol_name] = price_history.to_dict(orient="records")
#             counts += 1
#             print(f"{counts} items stored: {symbol_name}")
#         else:
#             print(f"Item not stored: {symbol_name}")
    
#     # Save prices to JSON
#     if len(price_history_dict) > 0:
#         with open("1_price_list.json", "w") as fp:
#             json.dump(price_history_dict, fp, indent=4)
#         print("Prices saved successfully.")
    
#     return