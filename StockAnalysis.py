import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import yfinance as yf

#configuration

TICKER = 'AAPL'
PERIOD = '1y'
INTERVAL = '1d'

MA_SORT = 20
MA_LONG = 50
RSI_PERIOD = 14
BB_WINDOW = 20
BB_STD = 2

#data download and cleaning
def download_data(ticker:str,period:str,interval:str)->pd.DataFrame:
    print(f"\n📥  Downloading {ticker} ({period}, {interval}) ...")
    df = yf.download(ticker, period = period, interval=interval, auto_adjust=True,progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")
    
    #flatten multiindex column if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[['Open','High','Close','Volume']].copy()
    df.dropna(how='all',inplace=True)

    #forward_fill any isolated missing rows
    df.ffill(inplace=True)

    print(f"✅  Loaded {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df

#technical indicator
def add_moving_average(df: pd.DataFrame)->pd.DataFrame:
    df[f"MA{MA_SORT}"] = df['Close'].rolling(MA_SORT).mean()
    df[f"MA{MA_LONG}"] = df['Close'].rolling(MA_LONG).mean()
    return df

def add_rsi(df: pd.DataFrame, period:int = RSI_PERIOD)->pd.DataFrame:
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 /(1 + rs))
    return df

def add_bollinger_bands(df:pd.DataFrame)->pd.DataFrame:
    df['BB_mid'] = df['Close'].rolling(BB_WINDOW).mean()
    df['BB_std'] = df['Close'].rolling(BB_WINDOW).std()
    df['BB_upper'] = df['BB_mid'] + BB_STD * df['BB_std']
    df['BB_lower'] = df['BB_mid'] + BB_STD * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    return df

def add_all_indicators(df : pd.DataFrame) -> pd.DataFrame:
    df = add_moving_average(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    return df

#BUY / sell signal generation
def generate_signals(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df['MA_Signals'] = 0
    df.loc[df[f"MA{MA_SORT}"] > df[f"MA{MA_LONG}"], 'MA_Signals'] = 1
    df.loc[df[f"MA{MA_SORT}"] > df[f"MA{MA_LONG}"], 'MA_Signals'] = -1

    #crossover events (change in signals)
    df['MA_Cross'] = df['MA_Signals'].diff()

    #rsi filter
    df['Signal'] = 0
    buy_cond = (df['MA_Cross']==2) & (df['RSI'] < 70)
    sell_cond = (df['MA_Cross']== -2) & (df['RSI'] < 75)
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1

    #convenience boolean columns for plotting
    df['Buy'] = buy_cond
    df['Sell'] = sell_cond
    buy_count = df['Buy'].sum()
    sell_count = df['Sell'].sum()
    print(f"📊  Signals generated — BUY: {buy_count}  |  SELL: {sell_count}")
    return df

#bactesting
def backtesting(df: pd.DataFrame, initial_capital: float = 10_000.0)->pd.DataFrame:
    df = df.copy()
    cash   = initial_capital
    share  = 0
    portfolio = []

    for idx, row in df.iterrows():
        price = row['Close']

        if row['Buy'] and share == 0 and cash > 0:
            share = int(cash // price)
            cash -= share *price
        elif row['Sell'] and share > 0:
            cash += share * price
            share = 0

        portfolio.append(cash + share *price)
    df['Portfolio_Value'] = portfolio

    #buy and hold baseline
    initial_price = df['Close'].iloc[0]
    bh_share = int(initial_capital // initial_price)
    df['BuyHold_Value'] = bh_share * df['Close'] + (initial_capital - bh_share * initial_price)

    #summary state
    final_strategy = df['Portfolio_Value'].iloc[-1]
    final_bh = df['BuyHold_Value'].iloc[-1]
    start_return = (final_strategy - initial_capital) / initial_capital * 100
    bh_return = (final_bh  - initial_capital) / initial_capital * 100

    print(f"\n📈  Backtest Results (initial capital = ${initial_capital:,.0f})")
    print(f"    Strategy return  : ${final_strategy:>10,.2f}  ({start_return:+.2f}%)")
    print(f"    Buy-and-hold     : ${final_bh:>10,.2f}  ({bh_return:+.2f}%)")
    print(f"    Alpha (vs B&H)   : {start_return - bh_return:+.2f}%")

    return df

#visualization

def plot_analysis(df: pd.DataFrame, ticker: str, output_file: str = 'stock_analysis_chart.png'):
    fig = plt.figure(figsize=(16, 14), facecolor= "#0d1117")
    gs = gridspec.GridSpec(4,1, height_ratios=[4,2,1.5,2],hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex= ax1)
    ax3 = fig.add_subplot(gs[2], sharex= ax1)
    ax4 = fig.add_subplot(gs[3], sharex= ax1)

    axes = [ax1, ax2, ax3, ax4]
    for ax in axes:
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors = '#8b949e', labelsize = 9)
        ax.spines[['top','right','left','bottom']].set_color("#30363d")
        ax.yaxis.label.set_color('#8b949e')

    idx = df.index
    #panel 1:price

    ax1.fill_between(idx, df['BB_lower'], df['BB_upper'], alpha= 0.07, color = '#58a6ff',label='BB_Band')
    ax1.plot(idx, df['BB_upper'],color='#58a6ff',linewidth = 0.6, linestyle = '--',alpha= 0.5)
    ax1.plot(idx, df['BB_lower'],color='#58a6ff',linewidth = 0.6,linestyle= '--',alpha=0.5)
    ax1.plot(idx, df['Close'], color='#e6edf3', linewidth = 1.4, label = 'Close')
    ax1.plot(idx, df[f"MA{MA_SORT}"], color="#f78166", linewidth=1.2, linestyle="--", label=f"MA{MA_SORT}")
    ax1.plot(idx, df[f"MA{MA_LONG}"],  color="#ffa657", linewidth=1.2, linestyle="--", label=f"MA{MA_LONG}")

    #BUY SIGNALS
    buys = df[df['Buy']]
    ax1.scatter(buys.index, buys['Close'], marker='^', color = '#3fb950',s = 80,zorder = 5, label='Buy')
    #SELL SIGNALS
    sells = df[df['Sell']]
    ax1.scatter(sells.index, sells['Close'], marker='v', color='#f85149', s=80, zorder = 5, label = 'Sell')
    ax1.set_ylabel('Price (USD)', color='#8b949e')
    ax1.set_title(f"{ticker} - Technical Analysis", color='#e6edf3',fontsize=14,pad=12)
    leg = ax1.legend(loc="upper left",fontsize=8,facecolor='#161b22', edgecolor='#30363d')
    for text in leg.get_texts():
        text.set_color('#e6edf3')
    ax1.grid(color='#21262d',linewidth=0.5)

    #panel 2:RSI
    ax2.plot(idx, df['RSI'], color='#d2a8ff')
    ax2.axhline(70, color= '#f85149', linewidth=0.8, linestyle= "--", alpha=0.7)
    ax2.axhline(30, color='#3fb950',  linewidth=0.8, linestyle='--',  alpha=0.7)
    ax2.fill_between(idx, df['RSI'], 70, where=(df['RSI'] >= 70), alpha=0.15, color='#f85149')
    ax2.fill_between(idx, df['RSI'], 30, where=(df['RSI'] <= 30), alpha=0.15, color="#3fb950")
    ax2.set_ylim(0, 100)
    ax2.set_yticks([30, 50, 70])
    ax2.set_ylabel('RSI', color='#8b949e')
    ax2.text(idx[-1], 72, '70', color='#f85149', fontsize=7, ha='right')
    ax2.text(idx[-1], 28, '30', color='#3fb950', fontsize=7, ha='right')
    ax2.grid(color='#21262d', linewidth=0.5)

    # Panel 3: Volume 
    colors = np.where(df['Close'] >= df['Open'], '#3fb950', '#f85149')
    ax3.bar(idx, df['Volume'] / 1e6, color=colors, alpha=0.7, width=0.8)
    ax3.set_ylabel('Vol (M)', color='#8b949e')
    ax3.grid(color='#21262d', linewidth=0.5)

    # Panel 4: Portfolio
    ax4.plot(idx, df['Portfolio_Value'], color='#3fb950', linewidth=1.4, label='Strategy')
    ax4.plot(idx, df['BuyHold_Value'], color='#58a6ff', linewidth=1.4, linestyle='--', label='Buy & Hold')
    ax4.set_ylabel("Value (USD)", color='#8b949e')
    leg4 = ax4.legend(loc='upper left', fontsize=8, facecolor='#161b22', edgecolor='#30363d')
    for text in leg4.get_texts():
        text.set_color('#e6edf3')
    ax4.grid(color='#21262d', linewidth=0.5)

    # X-axis formating
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d '%y"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
    for ax in [ax1,ax2,ax3]:
        plt.setp(ax.xaxis.get_majorticklabels(), visible= False)
    
    plt.savefig(output_file,dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n Chart saved -> {output_file}")
    plt.show()

    #Summar Starts
def print_summary(df: pd.DataFrame):
    close = df['Close']
    last = close.iloc[-1]
    high = close.max()
    low = close.min()
    rsi = df['RSI'].iloc[-1]
    ma20 = df[f"MA{MA_SORT}"].iloc[-1]
    ma50 = df[f"MA{MA_LONG}"].iloc[-1]
    bb_w = df['BB_width'].iloc[-1]

    trend = 'BULLISH 🟢' if ma20 > ma50 else 'BEARISH 🔴'
    rsi_zone = ("Overbought ⚠️" if rsi> 70
                else "Oversold 💡" if rsi < 30
                else "Netural")
    
    print("\n" + "-" * 45)
    print(f"  {'symblo':<20} {TICKER}")
    print(f"  {'Last Close':<20} ${last:.2f}")
    print(f"  {'52w High / Low':<20} ${high:.2f} / ${low:.2f}")
    print(f"  {f'MA{MA_SORT}':<20} ${ma20:.2f}")
    print(f"  {f'MA{MA_LONG}':<20} ${ma50:.2f}")
    print(f"  {'Trend':<20} {trend}")
    print(f"  {'RSI (14)':<20} {rsi:.1f}  [{rsi_zone}]")
    print(f"  {'BB Width':<20} {bb_w:.3f}")
    print("─" * 45)
 
# MAIN
def main():
    #step1 Download
    df = download_data(TICKER, PERIOD, INTERVAL)

    #Step 2 indicator
    df = add_all_indicators(df)

    #Step 3 Signals
    df = generate_signals(df)

    #Step 4 Backtest
    df = backtesting(df, initial_capital=10_000)

    #Step 5 Summary
    print_summary(df)

    #Step 6 Chart
    plot_analysis(df, TICKER, output_file='stock_analysis_chart.png')

    return df

if __name__ == '__main__':
    df = main()