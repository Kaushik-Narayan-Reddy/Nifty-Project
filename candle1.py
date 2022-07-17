import pandas as pd
import numpy as np
import mplfinance as mpl

df = pd.read_csv("C:/Users/kaush/Documents/Nifty Project/NIFTY_data/NIFTY_2008.csv")

for i in range(len(df)):
    y = str(df.loc[i, 'Date'])
    z = str(df.loc[i, 'Time'])
    y = y[0:4]+'-'+y[4:6]+'-'+y[6:8]
    y = y + " " + z
    df.loc[i, 'Date'] = y
    #print(y)
    if df.loc[i,'Open'] == None:
        df.loc[i,'Open']=df.loc[i-1,'Open']
    if df.loc[i,'High'] == None:
        df.loc[i,'High']=df.loc[i-1,'High']
    if df.loc[i,'Low'] == None:
        df.loc[i,'Low']=df.loc[i-1,'Low']
    if df.loc[i,'Close'] == None:
        df.loc[i,'Close']=df.loc[i-1,'Close']
    
df.pop('Time')
df.pop('Instrument')

df=df.set_index('Date')

df.to_csv('file1_2008.csv')

print(df)

#mpl.plot(df,type='candle', style='yahoo')