import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import datetime as dt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

descriptive = pd.read_csv('stockdata.csv', parse_dates = True, index_col = 0)

#statistics
def count(x):
	descriptive_count = {'Open':x[" Open"].count(),'High':x[" High"].count(),'Low':x[" Low"].count(),'Close':x[" Close"].count()}
	dcdf = pd.DataFrame(descriptive_count, index = ['Count'])
	print(dcdf)
def mean(x):
	descriptive_mean = {'Open':x[" Open"].mean(),'High':x[" High"].mean(),'Low':x[" Low"].mean(),'Close':x[" Close"].mean()}
	dmdf = pd.DataFrame(descriptive_mean,index = ['Mean'])
	pd.set_option('precision',2)
	print(dmdf)
def quantiles(x):
	descriptive_quantile = {'Open':[x[" Open"].quantile(0.25),x[" Open"].quantile(0.5),x[" Open"].quantile(0.75)],\
	'High':[x[" High"].quantile(0.25),x[" High"].quantile(0.5),x[" High"].quantile(0.75)],\
	'Low':[x[" Low"].quantile(0.25),x[" Low"].quantile(0.5),x[" Low"].quantile(0.75)],\
	'Close':[x[" Close"].quantile(0.25),x[" Low"].quantile(0.5),x[" Low"].quantile(0.75)]}
	dqdf = pd.DataFrame(descriptive_quantile,index = ['Quantile(25%)','Quantile(50%)','Quantile(75%)'])
	pd.set_option('precision',2)
	print(dqdf)
def range(x):
	descriptive_range = {'Open':[x[" Open"].max(),x[" Open"].min(),x[" Open"].max()-x[" Open"].min()],\
	'High':[x[" High"].max(),x[" High"].min(),x[" High"].max()-x[" High"].min()],\
	'Low':[x[" Low"].max(),x[" Low"].min(),x[" Low"].max()-x[" Low"].min()],\
	'Close':[x[" Close"].max(),x[" Low"].min(),x[" Close"].max()-x[" Low"].min()]}
	drdf = pd.DataFrame(descriptive_range,index = ['max','min','range'])
	pd.set_option('precision',2)
	print(drdf)
def standard_variation(x):
	descriptive_std = {'Open':[x[" Open"].std(),x[" Open"].var()],\
	'High':[x[" High"].std(),x[" High"].var()],\
	'Low':[x[" Low"].std(),x[" Low"].var()],\
	'Close':[x[" Close"].std(),x[" Close"].var()]}
	dsdf = pd.DataFrame(descriptive_std,index = ['Standard_Deviation','Standard_Variation'])
	pd.set_option('precision',2)
	print(dsdf)
def coefficient_variation(x):
	descriptive_cv = {'Open':x[" Open"].mean()/x[" Open"].var(),\
	'High':x[" High"].mean()/x[" High"].var(),\
	'Low':x[" Low"].mean()/x[" Low"].var(),\
	'Close':x[" Close"].mean()/x[" Close"].var()}
	dvdf = pd.DataFrame(descriptive_cv,index = ['Coefficient_of_Variation'])
	pd.set_option('precision',2)
	print(dvdf)
def normalized_price_values(x):
	open_array = np.array(x[' Open'])
	normalized_open = preprocessing.normalize([open_array])
	normalized_open_c = {'Normalized_Open':normalized_open[0]}
	normalized_value = pd.DataFrame(normalized_open_c,index = x.index)
	high_array = np.array(x[' High'])
	normalized_high = preprocessing.normalize([high_array])
	normalized_value['Normalized_High'] = normalized_high[0]
	low_array = np.array(x[' Low'])
	normalized_low = preprocessing.normalize([low_array])
	normalized_value['Normalized_Low'] = normalized_low[0]
	close_array = np.array(x[' Close'])
	normalized_close = preprocessing.normalize([close_array])
	normalized_value['Normalized_Close'] = normalized_close[0]
	normalized_value.to_csv('normalize.csv')
	print(normalized_value)
def timeseries_trendline(x):
	x.reset_index(inplace = True)
	x['Date'] = x['Date'].map(mdates.date2num)
	ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
	ax1v = ax1.twinx()
	ax1.plot(x['Date'],x[' Close'])
	ax1v.fill_between(x['Date'].values, x[' Volume'].values, alpha = 0.6)
	ax1.xaxis_date()
	plt.show()
def candlestick(x): 
	#style.use('ggplot')
	x.reset_index(inplace = True)
	x['Date'] = x['Date'].map(mdates.date2num)
	ax1 = plt.subplot2grid((6,1),(0,0),rowspan = 5, colspan = 1)
	candlestick_ohlc(ax1, quotes = x[['Date',' Open', ' High', ' Low', ' Close']].values,\
	width = 0.7, colorup = 'indianred', colordown = 'olivedrab', alpha = 0.7)
	plt.ylabel('Stock Price and Volume',fontsize = 'small',fontweight = 'bold',color = 'dimgray')
	ax1.xaxis_date()
	plt.setp(ax1.get_xticklabels(),visible=False)
	ax2 = plt.subplot2grid((6,1),(5,0),rowspan = 1, colspan = 1, sharex=ax1)
	ax2.fill_between(x['Date'].values, x[' Volume'].values,facecolor = 'steelblue', alpha = 0.8)
	plt.suptitle('Candlestick Chart',color = 'dimgray',fontweight = 'bold')
	plt.xlabel('Date',fontsize = 'small',color = 'dimgray',fontweight = 'bold')
	plt.show()
def ma(x):
	N = input("Please imput n you want for moving averages (a positve integer, such as 20): ")
	if float(N) > 0 and float(N)%1 == 0:
		x['MA'] = x[' Close'].rolling(window = int(N), min_periods=0).mean()
		ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
		ax1.plot(x.index,x['MA'], color = 'r')
		plt.ylabel('moving average',fontsize = 'small',fontweight = 'bold',color = 'dimgray')
		ax1v = ax1.twinx()
		ax1v.fill_between(x.index, x[' Volume'].values, facecolor = 'steelblue', alpha = 0.8)
		plt.ylabel('volume',fontsize = 'small',fontweight = 'bold',color = 'dimgray')
		plt.xlabel('Date',fontsize = 'small',color = 'dimgray',fontweight = 'bold')		
		plt.title('Moving Average',color = 'dimgray',fontweight = 'bold')
		plt.show()
	else:
		print("Sorry, the number is invalid.")
def ewma(x):
	N = input("Please imput n you want for moving averages (a positve integer, such as 20): ")
	if float(N) > 0 and float(N)%1 == 0:
		x['EWMA'] = x[' Close'].ewm(span = int(N)).mean()
		ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
		ax1.plot(x.index,x['EWMA'], color = 'r')
		plt.ylabel('exponential weighted moving average',fontsize = 'small',fontweight = 'bold',color = 'dimgray')
		ax1v = ax1.twinx()
		ax1v.fill_between(x.index, x[' Volume'].values, facecolor = 'steelblue', alpha = 0.6)
		plt.ylabel('volume',fontsize = 'small',fontweight = 'bold',color = 'dimgray')
		plt.xlabel('Date',fontsize = 'small',color = 'dimgray',fontweight = 'bold')		
		plt.title('Exponential Weighted Moving Average',color = 'dimgray',fontweight = 'bold')
		plt.show()
	else:
		print("Sorry, the number is invalid.")
#MACD
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



