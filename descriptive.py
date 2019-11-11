import pandas as pd
from sklearn.linear_model import LinearRegression

def Linear(data):
    x = np.array([i for i in range((len(data)))]).reshape(-1,1)
    y = np.array(data.values)
    model = LinearRegression()
    model.fit(x,y)
    Linear_trend = model.coef_ * x + model.intercept_ # Linear_trend = ax + c

def Add_column(data, Name, addition): #whole DataFrame
    Data.insert(len(list(data)), Name , addition)

def MA(data):
    n = input("Please specify in size of the moving window: ")
    MA = data.rolling(window = n).mean()
    return MA

def WMA(data):
    ""

def MACD(data):
    MACD_12 = data.ewm(span = 12, adjust = False).mean()
    MACD_16 = data.ewm(span = 16, adjust = False).mean()
    MACD = MACD_12 - MACD_16
    return MACD
