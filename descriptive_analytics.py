import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# visualise the time series:
df = pd.read_csv("HistoricalPrices.csv")[:: -1]
df['Date'] = pd.to_datetime(df["Date"], format='%m/%d/%y')
df.index = df["Date"]
selected = df[" Close"]


def vis_ts(data):
    data.plot()
    plt.xlabel("Date")
    plt.ylabel("Closing price")
    plt.show()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
z = np.polyfit(range(0, 824), selected.values.flatten(), 1)
lis = []
for x in range(len(selected)):
    y = z[0] * x + z[1]
    lis.append(y)

linear_value = pd.Series(lis)
df["Linear Value"] = linear_value.values
plt.plot(df[" Close"])
plt.plot(df["Linear Value"])
plt.show()

