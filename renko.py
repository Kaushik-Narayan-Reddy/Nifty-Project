import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
import mplfinance as mpl

df = pd.read_csv("NIFTY_data/NIFTY_2008.csv")
df = df[['Date', 'Time', 'Open', 'High', 'Low', 'Close']]
#print(df)

for i in range(len(df)):
    y = str(df.loc[i, 'Date'])
    z = str(df.loc[i, 'Time'])
    y = y[0:4]+'-'+y[4:6]+'-'+y[6:8]
    y = y + " " + z
    df.loc[i, 'Date'] = y
    #print(y)
    if df.loc[i, 'Date'] == None or df.loc[i, 'Time'] == None or df.loc[i, 'Open'] == None or df.loc[i, 'High'] == None or df.loc[i, 'Low'] == None or df.loc[i, 'Close'] == None:
        print(i)
        
print(df)

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(mpdates.date2num)
print(df['Date'])

quotes = [tuple(x)
          for x in df[['Date', 'Open', 'High', 'Low', 'Close']].values]

fig, ax = plt.subplots()

candlestick_ohlc(ax, quotes, width=0.6, colorup='green',
                 colordown='red', alpha=0.8)

ax.grid(True)

ax.set_xlabel('Date')
ax.set_ylabel('Price')

plt.title('Nifty data of the year 2008')

date_format = mpdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()
plt.autoscale(tight=True)

plt.show()

#mpl.plot(df, type = "renko")