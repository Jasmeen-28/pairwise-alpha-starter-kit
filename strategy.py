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

        # RSI of BONK for entry filtering
        df["rsi_bonk"] = compute_rsi(df["close_BONK_1H"])

        # Entry and exit management
        signals = []
        sizes = []
        in_position = False
        entry_price = 0.0

        for i in range(len(df)):
            btc_signal = df["btc_ret_lag1"].iloc[i] > 0.015 if pd.notna(df["btc_ret_lag1"].iloc[i]) else False
            eth_signal = df["eth_ret_lag2"].iloc[i] > 0.015 if pd.notna(df["eth_ret_lag2"].iloc[i]) else False
            sol_signal = df["sol_ret_lag3"].iloc[i] > 0.015 if pd.notna(df["sol_ret_lag3"].iloc[i]) else False
            rsi = df["rsi_bonk"].iloc[i]
            price = df["close_BONK_1H"].iloc[i]

            if not in_position:
                if (btc_signal or eth_signal or sol_signal) and pd.notna(price) and pd.notna(rsi) and rsi < 40:
                    signals.append("BUY")
                    sizes.append(0.7)
                    in_position = True
                    entry_price = price
                else:
                    signals.append("HOLD")
                    sizes.append(0.0)
            else:
                if pd.notna(price) and entry_price > 0:
                    pnl = (price - entry_price) / entry_price
                    if pnl >= 0.06 or pnl <= -0.03:
                        signals.append("SELL")
                        sizes.append(0.0)
                        in_position = False
                        entry_price = 0.0
                    else:
                        signals.append("HOLD")
                        sizes.append(0.7)
                else:
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
