import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib
from sklearn import preprocessing
from Linh.data_gathering import gather_data

def count(df):
	"""
	Count the number of rows in a data frame
	Return the data frame counting number of values each column of df

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	Return:
	--------------
	df_count : data frame (1, N)
		A data frame stores number of values each column of df
	"""

	df_count_dict = dict()

	for i, col in enumerate(df.columns):
		df_count_dict[col] = df[col].count()

	df_count = pd.DataFrame(df_count_dict, index=['Count'])

	return df_count


def mean(df):
	"""
	Calculate mean value of data in each column in a data frame
	Return the data frame calculating the mean value of data in each column of df

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	Return:
	--------------
	df_mean : data frame (1, N)
		A data frame stores the mean value of data in each column of df
	"""

	df_mean_dict = dict()

	for i, col in enumerate(df.columns):
		df_mean_dict[col] = df[col].mean()

	df_mean = pd.DataFrame(df_mean_dict, index=['Mean'])
	pd.set_option('precision', 2)  # set output display precision in 2 decimal places

	return df_mean


def quantile(df):
	"""
	Calculate quantile values (25%, 50%, 75%) of data in each column in a data frame
	Return the data frame calculating the quantile values of data in each column of df

	Parameters:
	--------------
	df: data frame (M,N)
		Initial data frame

	Return:
	--------------
	df_quantile : data frame (3,N)
		A data frame stores the quantile values (25%, 50%, 75%) of data in each column of df
	"""

	df_quantile_dict = dict()

	for i, col in enumerate(df.columns):
		df_quantile_dict[col] = [df[col].quantile(0.25), df[col].quantile(0.5), df[col].quantile(0.75)]

	df_quantile = pd.DataFrame(df_quantile_dict, index=['Quantile (25%)', 'Quantile (50%)', 'Quantile (75%)'])
	pd.set_option('precision', 2)  # set output display precision in 2 decimal places

	return df_quantile


def range(df):
	"""
	Determine the maximum, minimum values and the range (max - min) value of data in each column in a data frame
	Return the data frame storing the max, min values and the range (max - min) value of data in each column of df

	Parameters:
	--------------
	df: data frame (M,N)
		Initial data frame

	Return:
	--------------
	df_range : data frame (3,N)
		A data frame stores the max, min values and the range (max - min) value of data in each column of df
	"""

	df_range_dict = dict()

	for i, col in enumerate(df.columns):
		df_range_dict[col] = [df[col].max(), df[col].min(), df[col].max() - df[col].min()]

	df_range = pd.DataFrame(df_range_dict, index=['Max Value', 'Min Value', 'Range (Max - Min)'])
	pd.set_option('precision', 2)  # set output display precision in 2 decimal places

	return df_range


def standard_variation(df):
	"""
	Calculate the standard deviation and variance of data in each column of a data frame
	Return the data frame storing the standard deviation and variance of data in each column of df

	Parameters:
	--------------
	df: data frame (M,N)
		Initial data frame

	Return:
	--------------
	df_sdv : data frame (2,N)
		A data frame stores the standard deviation and variance of data in each column of df
	"""

	df_sdv_dict = dict()

	for i, col in enumerate(df.columns):
		df_sdv_dict[col] = [df[col].std(), df[col].var()]

	df_sdv = pd.DataFrame(df_sdv_dict, index=['Standard Deviation', 'Variance'])
	pd.set_option('precision', 2)  # set output display precision in 2 decimal places

	return df_sdv


def coeff_variation(df):
	"""
	Calculate the coefficient of variation of data in each column of a data frame
	Return the data frame storing the coefficient of variation of data in each column of df

	Parameters:
	--------------
	df: data frame (M,N)
		Initial data frame

	Return:
	--------------
	df_coeff : data frame (1,N)
		A data frame stores the coefficient of variation of data in each column of df
	"""

	df_coeff_dict = dict()

	for i, col in enumerate(df.columns):
		df_coeff_dict[col] = [df[col].std() / df[col].mean()]

	df_coeff = pd.DataFrame(df_coeff_dict, index=['Coeefficient of Variation'])
	pd.set_option('precision', 2)  # set output display precision in 2 decimal places

	return df_coeff


def normalize_price_values(df):
	"""
	Normalize the coefficient of variation of data in each column of a data frame
	Return the data frame storing the coefficient of variation of data in each column of df

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	Return:
	--------------
	df_normalize : data frame (M, N)
		A data frame stores the coefficient of variation of data in each column of df
	"""

	df_normalize_dict = dict()

	for i, col in enumerate(df.columns):
		col_array = np.array(df[col])
		df_normalize_dict["Normalized" + col] = preprocessing.normalize([col_array])[0]

	df_normalize = pd.DataFrame(df_normalize_dict, index=df.index)
	pd.set_option('precision', 2)  # set output display precision in 2 decimal places

	return df_normalize


def visualize_trendline(df, close_price_col_name="Close", linear_value_col_name="Linear Value"):
	"""
	Visualize time series data from a data frame along with corresponding linear trend line.

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	close_price_col_name: string, default "Close"
		Name of column from df which contains time serie format (ie: Closing price)

	linear_value_col_name: string, default "Linear Value"
		Name of column added to df which contains linear values from Linear Regression model.
	"""

	# Find a linear model that fits data
	model_coff = np.polyfit(range(len(df.index)), df[close_price_col_name].values.flatten(), 1)

	linear_value_list = []

	for x in range(len(df[close_price_col_name])):
		y = model_coff[0] * x + model_coff[1]  # y = a * x + b
		linear_value_list.append(y)

	linear_value = pd.Series(linear_value_list)

	# Add a column storing linear values
	df[linear_value_col_name] = linear_value.values

	# Plot
	plt.plot(df[close_price_col_name], label="Closing price")
	plt.plot(df[linear_value_col_name], label="Linear trend")
	plt.title("Visualization of Linear Trend Line")
	plt.xlabel("Date")
	plt.ylabel("Closing price")
	plt.legend(loc='upper left')
	plt.show()


def candelstick(df, int_date_col_name="Int Date", volume_col_name="Volume"):
	"""
	Visualize candlestick chart for stock data from a data frame.

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	int_date_col_name: string, default "Int Date"
		Name of column in df which contains date values

	volume_col_name: string, default "Volume"
		Name of column in df which contains stock volume values
	"""

	# Convert each date value in Date column (index) into an integer value, store them in a list
	int_date_list = [matplotlib.dates.date2num(date) for date in df.index]

	# Insert integer date column into the data frame
	df.insert(0, int_date_col_name, int_date_list)

	# Plot
	ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
	candlestick_ohlc(ax1, quotes=df[[col for col in df.columns]].values, width=0.7,
					 colorup="indianred", colordown="olivedrab", alpha=0.7)
	plt.ylabel("Stock Price and Volume", fontsize="small", fontweight="bold", color="dimgray")
	ax1.xaxis_date()
	plt.setp(ax1.get_xticklabels(), visible=False)

	ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
	ax2.fill_between(df[int_date_col_name].values, df[volume_col_name].values, facecolor="steelblue", alpha=0.8, label="Volume")

	plt.suptitle("Candlestick Chart", color="dimgray", fontweight="bold")
	plt.xlabel("Date", fontsize="small", color="dimgray", fontweight="bold")
	plt.legend(loc="upper left")
	plt.show()

	del df[int_date_col_name]  # delete the Int Date column for re-graphing


def ma(df, close_price_col_name="Close", ma_col_name="MA"):
	"""
	Visualize moving average for stock data from a data frame.

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	close_price_col_name: string, default "Close"
		Name of column from df which contains time serie format (ie: Closing price)

	ma_col_name: string, default "MA"
		Name of column in df which contains MA values
	"""

	# Check N positive integer
	while True:

		N = input("Please input period for moving average model (a positive integer (recommend: 10, 20, 50, 100, or 200 )): ")

		try:
			if int(N) > 0:
				break

			elif "." in N:
				print("Please enter a positive integer, not a float ")
				continue

			elif int(N) < 0:
				print("Please enter a positive integer, not a negative one ")
				continue

		except ValueError:
			print("Please input a positive integer, not a string")
			continue

	# Add column to store value of MA
	df[ma_col_name] = df[close_price_col_name].rolling(window=int(N), min_periods=0).mean()

	# Plot
	plt.plot(df[close_price_col_name], label="Closing price")
	plt.plot(df[ma_col_name], label="Moving average " + N + " days")
	plt.title("Visualization of Moving Average " + N + " days")
	plt.xlabel("Date")
	plt.ylabel("Closing price")
	plt.legend(loc='upper left')
	plt.show()

	del df[ma_col_name]  # delete the MA column for re-graphing


def wma(df, close_price_col_name="Close", wma_col_name="WMA"):
	"""
	Visualize weighted moving average for stock data from a data frame.

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	close_price_col_name: string, default "Close"
		Name of column from df which contains time serie format (ie: Closing price)

	wma_col_name: string, default "WMA"
		Name of column in df which contains WMA values
	"""

	# Check N positive integer
	while True:

		N = input("Please input period for moving average model (a positive integer (recommend: 10, 20, 50, 100, or 200 )): ")

		try:
			if int(N) > 0:
				break

			elif "." in N:
				print("Please enter a positive integer, not a float ")
				continue

			elif int(N) < 0:
				print("Please enter a positive integer, not a negative one ")
				continue

		except ValueError:
			print("Please input a positive integer, not a string")
			continue

	# Add column to store value of WMA
	df[wma_col_name] = df[close_price_col_name].ewm(span=int(N)).mean()

	# Plot
	plt.plot(df[close_price_col_name], label="Closing price")
	plt.plot(df[wma_col_name], label="Exponential Weighted Moving Average " + N + " days")
	plt.title("Visualization of Exponential Weighted Moving Average " + N + " days")
	plt.xlabel("Date")
	plt.ylabel("Closing price")
	plt.legend(loc='upper left')
	plt.show()

	del df[wma_col_name]  # delete the WMA column for re-graphing


def macd(df, close_price_col_name="Close"):
	"""
	Visualize Moving Average Convergence/Divergence (MACD) for stock data from a data frame.

	Parameters:
	--------------
	df: data frame (M, N)
		Initial data frame

	close_price_col_name: string, default "Close"
		Name of column from df which contains time serie format (ie: Closing price)
	"""
	# Add column to store value of MACD
	df['Dif'] = df[close_price_col_name].ewm(span=12).mean() - df[close_price_col_name].ewm(span=26).mean()

	# Plot
	plt.plot(df[close_price_col_name], label="Closing price")
	plt.plot(df["Dif"], label="Moving Average Convergence/Divergence (MACD)")
	plt.title("Visualization of Moving Average Convergence/Divergence")
	plt.xlabel("Date")
	plt.ylabel("Closing price")
	plt.legend(loc='upper left')
	plt.show()

	del df["Dif"]  # delete the WMA column for re-graphing


def descriptive():
	# 0. Introduction

	print("{:-^50}".format('Welcome to descriptive analysis'))

	# 1. Data selection:
	# Users have to choose the period for descriptive analysis.
	stock_data = gather_data()

	while not stock_data.empty:

		# 2. Options Selection:
		# Users have to choose the options they want to do:

		choices = ["Count", "Mean", "Quantiles", "Max, Min and Range", "Standard Deviation and Variance", "Coefficient of Variation",
				   "Normalized Price Values", "Linear Trend Lines", "Candlestick Chart", "Moving Averages(MA)", "Exponential Weighted Moving Averages(EWMA)",
				   "Moving Average Convergence/Divergence(MACD)", "Choose Another Stock", "Quit"]

		select = -1

		while select != 13:

			print('/' + '{:-^40}'.format('Lists of choices') + '\\')

			for i, item in enumerate(choices):
				print_choice = '    {}. {}'.format(i, item)
				print('|' + '{:<40}'.format(print_choice) + "|")
			print('\\' + '-' * 40 + '/')

			# Select your choice

			while True:
				try:
					select = input('Select your choice from the above lists (only number) :')
					break
				except ValueError:
					print('{:*^30}'.format('WARNING'))
					print("Wrong input!! Please fill the number")
					continue
			select = int(select)

			if select == 0:
				# 2.0. Count number of days in period
				print(count(stock_data))

			elif select == 1:
				# 2.1. Calculate mean of data each column
				print(mean(stock_data))

			elif select == 2:
				# 2.2. Calculate quantile of data each column
				print(quantile(stock_data))

			elif select == 3:
				# 2.3. Calculate Max, Min, Range (Max - min) values of data each column
				print(range(stock_data))

			elif select == 4:
				# 2.4. Calculate Standard Deviation and Variance of data each column
				print(standard_variation(stock_data))

			elif select == 5:
				# 2.5. Calculate Coefficient of variation of data each column
				print(coeff_variation(stock_data))

			elif select == 6:
				# 2.6. Normalize stock prices
				print(normalize_price_values(stock_data))

			elif select == 7:
				# 2.7. Visualize linear trend lines
				visualize_trendline(stock_data)

			elif select == 8:
				# 2.8. Visualize candlestick chart
				candelstick(stock_data)

			elif select == 9:
				# 2.9. Visualize Moving Average
				ma(stock_data)

			elif select == 10:
				# 2.10. Visualize Exponential Weighted Moving Average
				wma(stock_data)

			elif select == 11:
				# 2.11. Visualize MACD
				macd(stock_data)

			elif select == 12:
				gather_data()
				select = int(select)

			elif select == 13:

				print("{:-^40}".format('Goodbye'))

			else:
				print("***Wrong choice***")

	print("There is no information on time given")


if __name__ == "__main__":

	descriptive()
