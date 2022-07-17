import datetime as dt
import pandas as pd
import numpy as np
import mplfinance as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

ticker_name = "NSE"

df = pd.read_csv('C:/Users/kaush/Documents/Nifty Project/file1_2009.csv', index_col=0,parse_dates=True)

def renko_val(df,bricks):
    dic = {'Renko_date':[],'Renko_brick':[],'Open':[],'High':[],'Low':[],'Close':[]}
    x=df.iloc[0]['Close']
    sign=[]
    for i in range(len(df)):
        if float(df.iloc[i]['Close']) >= float(x+bricks):
            br = (df.iloc[i]['Close']) - (x)
            br = br//bricks
            #print(br)
            for j in range(int(br)):
                x=x+10
                df.iloc[i]['Date']
                dic['Renko_date'].append(df.iloc[i]['Date'])
                dic['Renko_brick'].append(x)
                dic['Open'].append(df.iloc[i]['Open'])
                dic['High'].append(df.iloc[i]['High'])
                dic['Low'].append(df.iloc[i]['Low'])
                dic['Close'].append(df.iloc[i]['Close'])
                sign.append(1)
        elif float(df.iloc[i]['Close']) <= float(x-bricks):
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
    plt.title("Renko data of "+ ticker_name +" of year 2009")
        
    plt.show()
    return rdf

bricks=10    
rdf=renko_val(df, bricks)
print("Renko data of "+ ticker_name +" of year 2009")
print(rdf)
cvals = {}