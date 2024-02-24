import pandas as pd

def calculate_pnl(signals_file):
    signals = pd.read_csv(signals_file)

    # Initialize variables
    position = 0  # 0 represents no position, 1 represents long position, -1 represents short position
    pnl = 0  # Profit and Loss
    buy_price = 0  # Price at which the last buy occurred

    # Iterate through signals
    for index, row in signals.iterrows():
        if row['signal'] == 'buy' and position == 0:
            # Open a long position
            position = 1
            buy_price = row['vwap']
        elif row['signal'] == 'sell' and position == 1:
            # Close the long position
            position = 0
            pnl += (row['vwap'] - buy_price)

    # If there's an open position at the end, close it using the last available price
    if position == 1:
        last_price = signals['vwap'].iloc[-1]
        pnl += (last_price - buy_price)

    pnl = pnl/signals['vwap'].iloc[-1]

    return pnl


# Calculate P&L for each ticker
for ticker in ['USDEUR', 'USDGBP', 'USDAUD', 'USDCNY', 'USDHKD', 'USDJPY', 'USDPLN']:
    signals_file = f"{ticker}_signals.csv"
    pnl = calculate_pnl(signals_file)
    print(f"P&L for {ticker}: ${pnl:.2f}")