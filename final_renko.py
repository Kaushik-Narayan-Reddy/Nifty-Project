import datetime as dt
import pandas as pd
import numpy as np
import mplfinance as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas_ta as ta
import plotly as go
from sklearn import linear_model
from sklearn.model_selection import train_test_split

ticker_name = "NSE"

df = pd.read_csv('C:/Users/kaush/Documents/Nifty Project/file1_2009.csv', index_col=0,parse_dates=True)

def renko_val(df,bricks):
    dic = {'Renko_date':[],'Renko_brick':[],'Open':[],'High':[],'Low':[],'Close':[]}
    x=df.iloc[0]['Close']
    sign=[]
    for i in range(len(df)):
        if df.iloc[i]['Close'] >= x+bricks:
            br = (df.iloc[i]['Close']) - (x)
            br = br//bricks
            #print(br)
            for j in range(int(br)):
                x=x+10
                dic['Renko_date'].append(df.iloc[i]['Date'])
                dic['Renko_brick'].append(x)
                dic['Open'].append(df.iloc[i]['Open'])
                dic['High'].append(df.iloc[i]['High'])
                dic['Low'].append(df.iloc[i]['Low'])
                dic['Close'].append(df.iloc[i]['Close'])
                sign.append(1)
        elif df.iloc[i]['Close'] <= x-bricks:
            br = (x) - (df.iloc[i]['Close'])
            br = br//bricks
            #print(br)
            for j in range(int(br)):
                x=x-10
                dic['Renko_date'].append(df.iloc[i]['Date'])
                dic['Renko_brick'].append(x)
                dic['Open'].append(df.iloc[i]['Open'])
                dic['High'].append(df.iloc[i]['High'])
                dic['Low'].append(df.iloc[i]['Low'])
                dic['Close'].append(df.iloc[i]['Close'])
                sign.append(-1)
    rdf=pd.DataFrame.from_dict(dic)
    return rdf

def plot_rdf(df, rdf):
    sign=[]
    x=df.iloc[0]['Close']
    for i in range(len(df)):
        if df.iloc[i]['Close'] >= x+bricks:
            br = (df.iloc[i]['Close']) - (x)
            br = br//bricks
            for j in range(int(br)):
                x=x+10
                sign.append(1)
        elif df.iloc[i]['Close'] <= x-bricks:
            br = (x) - (df.iloc[i]['Close'])
            br = br//bricks
            for j in range(int(br)):
                x=x-10
                sign.append(-1)
    
    fig, ax = plt.subplots()
    for i in range(len(rdf)):
        if i==0:
            if rdf.iloc[i]['Renko_brick'] >= df.iloc[0]['Close'] +10:
                facecolor='green'
            else:
                facecolor='red'
            ax.add_patch(Rectangle(((i+1)*10,df.iloc[0]['Close']),10,10,edgecolor='black',facecolor=facecolor))
        elif i!=0:
            if sign[i-1] == 1:
                facecolor='green'
            elif sign[i-1] == -1:
                facecolor='red'
            ax.add_patch(Rectangle(((i+1)*10,rdf.iloc[i-1]['Renko_brick']),10,10,edgecolor='black',facecolor=facecolor))
    plt.xlim([0,((len(rdf)*10)+10)])
    plt.ylim([min(rdf['Renko_brick'])-100,max(rdf['Renko_brick'])+100])
    plt.title("Renko bricks of NSE 2009 data")
        
    plt.show()

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def plot_macd(prices, macd, signal, hist):
    ax2 = plt.subplot2grid((8,1), (0,0), rowspan = 7, colspan = 1)
    ax2.plot(macd, color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(signal, color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax2.bar(prices.index[i], hist[i], color = '#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], color = '#26a69a')
            
    ax2.set_title('Renko data 2009 MACD, Signal and Histogram')
    plt.legend(loc = 'lower right')
    
def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]

def plot_rsi(rdf):
    fig, ax = plt.subplots()
    ax = plt.subplot2grid((10,1), (0,0), rowspan = 5, colspan = 1)
    ax.plot(rdf['rsi'], color = 'orange', linewidth = 1.5)
    ax.axhline(30, linestyle = '--', linewidth = 1, color = 'grey')
    ax.axhline(70, linestyle = '--', linewidth = 1, color = 'grey')
    ax.set_title('Renko data 2009 RSI Index')
    plt.show()
    
def get_vol(rdf):
    dic={'Volume':[]}
    for i in range(len(rdf)):
        if i==0:
            dic['Volume'].append(abs((rdf.iloc[i+1]['Close']-rdf.iloc[i]['Close'])/0.05))
        else:
            dic['Volume'].append(abs((rdf.iloc[i]['Close']-rdf.iloc[i-1]['Close'])/0.05))
    df=pd.DataFrame.from_dict(dic)
    return df['Volume']

def vwap(df):
    v = df['Volume'].values
    tp = (df['Low'] + df['Close'] + df['High']).div(3).values
    return df.assign(vwap=(tp * v).cumsum() / v.cumsum())

def regression_predict(dates_train, dates_test, OHLC_train):
	lin_model = linear_model.LinearRegression()
	lin_model.fit(dates_train,OHLC_train)
	predicted_price = lin_model.predict(dates_test)
	return predicted_price

bricks=10    
rdf=renko_val(df, bricks)
rdf_macd = get_macd(rdf['Close'], 26, 12, 9)
rdf['Volume'] = get_vol(rdf)
rdf['rsi'] = get_rsi(rdf['Close'], 14)
rdf=vwap(rdf)

print("Renko data of "+ ticker_name +" of year 2009")
print(rdf)

print("MACD Renko data 2009")
print(rdf_macd)

price_high = rdf['High']
price_low = rdf['Low']
price_open = rdf['Open']
price_close = rdf['Close']
#from these prices calculate the OHLC average for the Google stocks
price_ave= (price_high[:]+price_low[:]+price_open[:]+price_close[:])/4

dates = rdf['Renko_date']
volume = rdf['Volume']
#convert date format to numeric format
dates = pd.to_datetime(rdf['Renko_date'])
dates = dates.map(dt.datetime.toordinal)
dates = np.array(dates).reshape((-1,1))

dates_train, dates_test, OHLC_train, OHLC_test = train_test_split(dates,price_ave,test_size=0.25)

pred_linear = regression_predict(dates_train, dates_test, OHLC_train)

d=list(rdf['Renko_date'])

chk=[]
for i in OHLC_test:
    chk.append(OHLC_test[OHLC_test == i].index[0])
reg_dates={'Date':[], 'Open':[], 'High':[], 'Close':[], 'Low':[]}
for i in chk:
    reg_dates['Date'].append(rdf.iloc[i]['Renko_date'])
    reg_dates['Open'].append(rdf.iloc[i]['Open'])
    reg_dates['High'].append(rdf.iloc[i]['High'])
    reg_dates['Low'].append(rdf.iloc[i]['Low'])
    reg_dates['Close'].append(rdf.iloc[i]['Close'])
lin_reg = pd.DataFrame.from_dict(reg_dates)
lin_reg['Pred_lin'] = pred_linear

print("Linear Regression values vs actual values")
print(lin_reg)

plt.figure(1)
ax = plt.subplot2grid((1,1),(0,0))
plt.scatter(dates_test, OHLC_test, color='orange', label='data')
plt.plot(dates_test,pred_linear,color='black', label='Linear Regression')
plt.xlabel('numeric dates')
plt.ylabel('target')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
plt.legend()
plt.show()

plot_rdf(df, rdf)
plot_macd(rdf['Close'], rdf_macd['macd'], rdf_macd['signal'], rdf_macd['hist'])
plot_rsi(rdf)