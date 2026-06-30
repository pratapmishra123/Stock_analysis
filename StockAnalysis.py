import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = 'AAPL'
PERIOD = '1y'
INTERVAL = '1d'

MA_SORT = 20
MA_LONG = 50
RSI_PERIOD = 14
BB_WINDOW = 20
BB_STD = 2

def download_data(ticker, period, interval):
    print(f"Downloading {ticker} ({period}, {interval}) ...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[['Open', 'High', 'Close', 'Volume']].copy()
    df.dropna(how='all', inplace=True)
    df.ffill(inplace=True)
    print(f"Loaded {len(df)} rows | {df.index[0].date()} -> {df.index[-1].date()}")
    return df

def add_moving_average(df):
    df[f"MA{MA_SORT}"] = df['Close'].rolling(MA_SORT).mean()
    df[f"MA{MA_LONG}"] = df['Close'].rolling(MA_LONG).mean()
    return df

def add_rsi(df, period=RSI_PERIOD):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_bollinger_bands(df):
    df['BB_mid'] = df['Close'].rolling(BB_WINDOW).mean()
    df['BB_std'] = df['Close'].rolling(BB_WINDOW).std()
    df['BB_upper'] = df['BB_mid'] + BB_STD * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - BB_STD * df['BB_std']  # FIX: was + (made upper==lower)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    return df

def add_all_indicators(df):
    df = add_moving_average(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    return df

def generate_signals(df):
    df = df.copy()
    df['MA_Signals'] = 0
    df.loc[df[f"MA{MA_SORT}"] > df[f"MA{MA_LONG}"], 'MA_Signals'] = 1
    df.loc[df[f"MA{MA_SORT}"] < df[f"MA{MA_LONG}"], 'MA_Signals'] = -1  # FIX: was '>' for both branches

    df['MA_Cross'] = df['MA_Signals'].diff()

    df['Signal'] = 0
    buy_cond = (df['MA_Cross'] == 2) & (df['RSI'] < 70)
    sell_cond = (df['MA_Cross'] == -2) & (df['RSI'] < 75)
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1

    df['Buy'] = buy_cond
    df['Sell'] = sell_cond
    print(f"Signals generated - BUY: {df['Buy'].sum()} | SELL: {df['Sell'].sum()}")
    return df

def backtesting(df, initial_capital=10_000.0):
    df = df.copy()
    cash = initial_capital
    share = 0
    portfolio = []

    for idx, row in df.iterrows():
        price = row['Close']
        if row['Buy'] and share == 0 and cash > 0:
            share = int(cash // price)
            cash -= share * price
        elif row['Sell'] and share > 0:
            cash += share * price
            share = 0
        portfolio.append(cash + share * price)
    df['Portfolio_Value'] = portfolio

    initial_price = df['Close'].iloc[0]
    bh_share = int(initial_capital // initial_price)
    df['BuyHold_Value'] = bh_share * df['Close'] + (initial_capital - bh_share * initial_price)

    final_strategy = df['Portfolio_Value'].iloc[-1]
    final_bh = df['BuyHold_Value'].iloc[-1]
    start_return = (final_strategy - initial_capital) / initial_capital * 100
    bh_return = (final_bh - initial_capital) / initial_capital * 100

    print(f"Strategy return: {start_return:+.2f}% | Buy-hold: {bh_return:+.2f}% | Alpha: {start_return - bh_return:+.2f}%")
    return df

def export_json(df, ticker, path):
    d = {}
    d['ticker'] = ticker
    d['dates'] = [dt.strftime('%Y-%m-%d') for dt in df.index]
    d['close'] = df['Close'].round(2).tolist()
    d['open'] = df['Open'].round(2).tolist()
    d['high'] = df['High'].round(2).tolist()
    d['volume'] = df['Volume'].astype(float).tolist()
    d['ma20'] = df[f"MA{MA_SORT}"].round(2).where(df[f"MA{MA_SORT}"].notna(), None).tolist()
    d['ma50'] = df[f"MA{MA_LONG}"].round(2).where(df[f"MA{MA_LONG}"].notna(), None).tolist()
    d['rsi'] = df['RSI'].round(2).where(df['RSI'].notna(), None).tolist()
    d['bb_upper'] = df['BB_upper'].round(2).where(df['BB_upper'].notna(), None).tolist()
    d['bb_lower'] = df['BB_lower'].round(2).where(df['BB_lower'].notna(), None).tolist()
    d['bb_width'] = df['BB_width'].round(4).where(df['BB_width'].notna(), None).tolist()
    d['portfolio_value'] = df['Portfolio_Value'].round(2).tolist()
    d['bh_value'] = df['BuyHold_Value'].round(2).tolist()

    buy_rows = df[df['Buy']]
    sell_rows = df[df['Sell']]
    d['buy_dates'] = [dt.strftime('%Y-%m-%d') for dt in buy_rows.index]
    d['buy_prices'] = buy_rows['Close'].round(2).tolist()
    d['sell_dates'] = [dt.strftime('%Y-%m-%d') for dt in sell_rows.index]
    d['sell_prices'] = sell_rows['Close'].round(2).tolist()

    last = df['Close'].iloc[-1]
    first = df['Close'].iloc[0]
    d['summary'] = {
        'last_price': round(float(last), 2),
        'price_change_pct': round(float((last - first) / first * 100), 2),
        'high_52w': round(float(df['Close'].max()), 2),
        'low_52w': round(float(df['Close'].min()), 2),
        'current_rsi': round(float(df['RSI'].iloc[-1]), 1),
        'current_ma20': round(float(df[f"MA{MA_SORT}"].iloc[-1]), 2),
        'current_ma50': round(float(df[f"MA{MA_LONG}"].iloc[-1]), 2),
        'current_bb_width': round(float(df['BB_width'].iloc[-1]), 4),
        'trend': 'BULLISH' if df[f"MA{MA_SORT}"].iloc[-1] > df[f"MA{MA_LONG}"].iloc[-1] else 'BEARISH',
        'final_strategy_value': round(float(df['Portfolio_Value'].iloc[-1]), 2),
        'final_bh_value': round(float(df['BuyHold_Value'].iloc[-1]), 2),
        'strategy_return_pct': round(float((df['Portfolio_Value'].iloc[-1] - 10000) / 10000 * 100), 2),
        'bh_return_pct': round(float((df['BuyHold_Value'].iloc[-1] - 10000) / 10000 * 100), 2),
        'buy_signals_count': int(df['Buy'].sum()),
        'sell_signals_count': int(df['Sell'].sum()),
    }
    d['summary']['alpha_pct'] = round(d['summary']['strategy_return_pct'] - d['summary']['bh_return_pct'], 2)

    with open(path, 'w') as f:
        json.dump(d, f)
    print(f"Exported -> {path}")

def main():
    df = download_data(TICKER, PERIOD, INTERVAL)
    df = add_all_indicators(df)
    df = generate_signals(df)
    df = backtesting(df, initial_capital=10_000)
    export_json(df, TICKER, 'stock_data.json')
    return df

if __name__ == '__main__':
    df = main()