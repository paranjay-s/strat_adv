from config_strategy_api import session

def filter_liquid_symbols(min_volume=1000000, min_turns=50000):
    """Filter symbols by liquidity metrics with proper error handling"""
    liquid_symbols = []
    response = session.get_instruments_info(category="linear")
    
    if not response.get("result") or not response["result"].get("list"):
        return liquid_symbols

    for sym in response["result"]["list"]:
        try:
            # Safely get values with defaults for missing keys
            volume_24h = float(sym.get("volume24h", 0))
            turnover_24h = float(sym.get("turnover24h", 0))
            open_interest = float(sym.get("openInterest", 0))

            if all([
                sym.get("quoteCoin") == "USDT",
                sym.get("status") == "Trading",
                volume_24h > min_volume,
                turnover_24h > min_turns,
                open_interest > 0
            ]):
                liquid_symbols.append(sym)
                
        except (KeyError, ValueError, TypeError) as e:
            print(f"Skipping symbol {sym.get('symbol')} due to error: {str(e)}")
            continue

    return liquid_symbols
