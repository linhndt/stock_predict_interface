import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib
from sklearn import preprocessing


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
	model_coff = np.polyfit(range(0, len(df.index)), df[close_price_col_name].values.flatten(), 1)

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


def macd(x):
	x['sema'] = x[' Close'].ewm(span = 12).mean()
	x['lema'] = x[' Close'].ewm(span = 26).mean()
	x['dif'] = x['sema'] - x['lema']
	x['dea'] = x['dif'].ewm(span = 9).mean()
	x['macd'] = 2*(x['dif']-x['dea'])
	plt.plot(x.index, x['dif'], color = 'indianred', label = 'dif')
	plt.plot(x.index, x['dea'], color = 'olivedrab', label = 'dea')
	plt.fill_between(x.index, x['macd'], alpha = 0.6, facecolor = 'steelblue')
	plt.xlabel('Date',fontsize = 'small',color = 'dimgray',fontweight = 'bold')
	plt.ylabel('MACD',fontsize = 'small',fontweight = 'bold',color = 'dimgray')
	plt.legend()
	plt.title('Moving Average Convergence/Divergence 12/26/9',color = 'dimgray',fontweight = 'bold')
	plt.show()
	#x.fillna(0,inplace = True)
print('1. Count\n2. Mean\n3. Quantiles\n4. Max, Min and Range\n5. Standard Variation and Standard Deviation\n6. Coefficient of Variation\n7. Normalized Price Values\
	\n8. Raw Time-Series and Linear Trend Lines\n9. Candlestick Chart\n10. Moving Averages(MA)\n11. Exponential Weighted Moving Averages(EWMA)\
	\n12. Moving Average Convergence/Divergence(MACD)\n13. Choose Another Stock\n14. Quit')
choice = input('Please choose the option(one at a time, such as 2): ')
while choice != '14':
	if choice == '1':
		count(descriptive)
	elif choice == '2':
		mean(descriptive)
	elif choice == '3':
		quantiles(descriptive)
	elif choice == '4':
		range(descriptive)
	elif choice == '5':
		standard_variation(descriptive)
	elif choice == '6':
		coefficient_variation(descriptive)
	elif choice == '7':
		normalized_price_values(descriptive)
	elif choice == '8':
		timeseries_trendline(descriptive)
	elif choice == '9':
		candlestick(descriptive)
	elif choice == '10':
		ma(descriptive)
	elif choice == '11':
		ewma(descriptive)
	elif choice == '12':
		macd(descriptive)
	else:
		print('Choose another stock')

	choice = input('Please choose the option(one at a time, such as 2): ')

