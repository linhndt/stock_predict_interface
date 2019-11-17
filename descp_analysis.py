import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# visualise the time series:
df = pd.read_csv("HistoricalPrices.csv")[:: -1]
df['Date'] = pd.to_datetime(df["Date"], format='%m/%d/%y')
df.index = df["Date"]
selected = df[" Close"]

z = np.polyfit(range(0, 824), selected.values.flatten(), 4)
lis = []
for x in range(len(selected)):
    y = z[0] * x**4 + z[1] * x**3 + z[2] * x**2 + z[3] * x + z[4]
    lis.append(y)

linear_value = pd.Series(lis)
df["Linear Value"] = linear_value.values
plt.plot(selected)
plt.plot(df["Linear Value"])
plt.show()

# visualise the