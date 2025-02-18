"""
    API Documentation
    https://bybit-exchange.github.io/docs/linear/#t-introduction
"""

# API Imports
#API Imports
from pybit.unified_trading import HTTP

# CONFIG
testnet = False
timeframe = 60
kline_limit = 200
z_score_window = 21

# LIVE API
# api_key_mainnet = "enter your mainnet API key"
# api_secret_mainnet = "insert your key here"
api_key_mainnet = "XVPCOECHt1Cqi98ggY"
api_secret_mainnet = "LEa2cdI7bUfyZfuolZcO0SE0gB106noZSonw"

# TEST API
api_key_testnet = "enter your testnet API key"
api_secret_testnet = "insert your key here"

# SELECTED API
api_key = api_key_testnet if testnet == True else api_key_mainnet
api_secret = api_secret_testnet if testnet == True else api_secret_mainnet

#SELECTED URL
api_url = "https://api-testnet.bybit.com" if testnet == True else "https://api.bybit.com"

#SESSION Activation
session = HTTP(testnet=testnet)

# # Web Socket Connection
# subs = [
#     "candle.1.BTCUSDT"
# ]
# ws = WebSocket(
#     "wss://stream-testnet.bybit.com/realtime_public",
#     subscriptions=subs
# )
#
# while True:
#     data = ws.fetch(subs[0])
#     if data:
#         print(data)
