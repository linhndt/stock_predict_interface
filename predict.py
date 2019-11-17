from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("HistoricalPrices.csv")[:: -1]
df['Date'] = pd.to_datetime(df["Date"], format='%m/%d/%y')


# def add_int_date_col(df, col_name="Int Date"):
#
#     int_date_list = [matplotlib.dates.date2num(date) for date in df["Date"]]
#     int_date_series = pd.Series(int_date_list)
#     df[col_name] = int_date_series.values
#
#     return df
#
# def train(df):


lis_date = []
for date in df["Date"]:
    date = matplotlib.dates.date2num(date)
    lis_date.append(date)

int_date = pd.Series(lis_date)
df["Int Date"] = int_date.values

x_train = df["Int Date"].head(10).tolist()
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = df[" Close"].head(10)

model = LinearRegression()
model.fit(x_train, y_train)

date_input = "11/16/2016"

date_pred = matplotlib.dates.date2num(datetime.strptime(date_input, '%m/%d/%Y'))
x_pred = np.array(date_pred).reshape(-1, 1)
y_pred = model.predict(x_pred)
print(y_pred)

plt.figure(1, figsize=(16, 10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(x_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(x_train, model.predict(x_train), color='b', label='Pred')
# plt.plot(date_pred, y_pred, marker='o', markersize=10, color="red")
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()