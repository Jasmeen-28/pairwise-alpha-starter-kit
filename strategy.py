"""
Lunor Quest: PairWise Alpha Round 2 Strategy

"""

import pandas as pd
import numpy as np

def get_coin_metadata() -> dict:
    return {
        "targets": [
            {"symbol": "BONK", "timeframe": "1H"}
        ],
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"},
            {"symbol": "SOL", "timeframe": "1H"}
        ]
    }

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.merge(target_df, anchor_df, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)

        # Lagged returns for anchor coins
        df["btc_ret_lag1"] = df["close_BTC_1H"].pct_change().shift(1)
        df["eth_ret_lag2"] = df["close_ETH_1H"].pct_change().shift(2)
        df["sol_ret_lag3"] = df["close_SOL_1H"].pct_change().shift(3)

        # RSI and volatility for BONK
        df["rsi_bonk"] = compute_rsi(df["close_BONK_1H"])
        df["volatility"] = df["close_BONK_1H"].rolling(window=10).std()

        # Initialize signal logic
        signals = []
        sizes = []
        in_position = False
        entry_price = 0.0
        trailing_stop = 0.0

        for i in range(len(df)):
            price = df["close_BONK_1H"].iloc[i]
            rsi = df["rsi_bonk"].iloc[i]
            vol = df["volatility"].iloc[i]

            # Signal weights
            score = 0
            if pd.notna(df["btc_ret_lag1"].iloc[i]) and df["btc_ret_lag1"].iloc[i] > 0.01:
                score += 1
            if pd.notna(df["eth_ret_lag2"].iloc[i]) and df["eth_ret_lag2"].iloc[i] > 0.01:
                score += 1
            if pd.notna(df["sol_ret_lag3"].iloc[i]) and df["sol_ret_lag3"].iloc[i] > 0.01:
                score += 1

            if not in_position:
                if score >= 2 and pd.notna(price) and pd.notna(rsi) and rsi < 40 and (pd.isna(vol) or vol < 0.02):
                    signals.append("BUY")
                    sizes.append(0.7)
                    in_position = True
                    entry_price = price
                    trailing_stop = price * 0.97  # 3% trailing stop
                else:
                    signals.append("HOLD")
                    sizes.append(0.0)
            else:
                pnl = (price - entry_price) / entry_price if pd.notna(price) and entry_price > 0 else 0
                if price <= trailing_stop or pnl >= 0.06 or pnl <= -0.03 or (pd.notna(rsi) and rsi > 70):
                    signals.append("SELL")
                    sizes.append(0.0)
                    in_position = False
                    entry_price = 0.0
                    trailing_stop = 0.0
                else:
                    # Update trailing stop
                    if price > entry_price and price * 0.97 > trailing_stop:
                        trailing_stop = price * 0.97
                    signals.append("HOLD")
                    sizes.append(0.7)

        return pd.DataFrame({
            "timestamp": df["timestamp"],
            "symbol": "BONK",
            "signal": signals,
            "position_size": sizes
        })

    except Exception as e:
        raise RuntimeError(f"[Strategy Error] {e}")
