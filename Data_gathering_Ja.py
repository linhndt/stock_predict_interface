import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression

# Load company list

def get_comp_list():
    COMP_LIST_URL = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
    comp_df = pd.read_csv(COMP_LIST_URL)
    comp_symbol = comp_df["Symbol"].tolist()
    return comp_symbol

def gather_data():

    #Ja edition
    SYMBOL_INPUT = input("Please choose a symbol input: ").upper().strip()
    while SYMBOL_INPUT not in get_comp_list():
        print("Sorry, the symbol is invalid.")
        SYMBOL_INPUT = input("Please choose a symbol input: ").upper().strip()

    start_date = input("Please choose a start date (MM/DD/YYYY): ")
    end_date = input("Please choose an end date: (MM/DD/YYYY): ")
    while start_date > end_date:
        print("Please choose the start date before the end date")
        start_date = input("Please choose a start date (MM/DD/YYYY): ")
        end_date = input("Please choose an end date: (MM/DD/YYYY): ")

    url = "https://quotes.wsj.com/" + SYMBOL_INPUT + "/historical-prices/download?MOD_VIEW=page&" \
          "num_rows=6299&range_days=6299&startDate=" + start_date + "&endDate=" + end_date

    data = pd.read_csv(url)

    #Amend the headings
    #data.dtypes
    #data.rename(columns={' Open':'Open', ' High':'High', ' Low':'Low', ' Close':'Close', ' Volumn':'Volumn'})
    column_name = []
    for i in range(len(data.columns)):
        column_name.append("")
        if i == 0: column_name[i] = data.columns[i]
        else: column_name[i] = data.columns[i][1:]
    data.columns = column_name

    data["Date"] = pd.to_datetime(data["Date"], format = "%m/%d/%y") #change data type from object to a date time object
    data = data.sort_values('Date')
    data = data.set_index('Date')

    return data, pd.to_datetime(start_date, format = "%m/%d/%Y"), pd.to_datetime(end_date, format = "%m/%d/%Y")

def prescriptive_analytics(data, start_date, end_date):

    Choice = ""
    while Choice != "8":
        print("\t1. Mean\n\t2. Quatiles\n\t3. Range\n\t4. Standard variation\
        \n\t5. Coefficient of variation\n\t6. Normalised price values\n\t7. Graphical visualisation\
        \n\t8. Quit")
        Choice = input("Please choose option: ")
        if Choice == "1": #Mean
            print(data.mean())
        elif Choice == "2": #Quatiles
            Quartile = float(input("Indicate a quartile (from 0 to 1): "))
            while Quartile > 1 or Quartile < 0: #if not in the range, an user has to fill the value again
                print("You filled the wrong value, please fill a value from 0 to 1")
                Quartile = float(input("Indicate a quartile (from 0 to 1): "))
            print(data.quantile(q = Quartile))
        elif Choice == "3": #Range
            print(data.max()-data.min())
        elif Choice == "4": #Standard variation
            print(data.std())
        elif Choice == "5": #Coefficient of variation
            print(data.std()/data.mean())
        elif Choice == "6": #Normalised price values
            print()
        elif Choice == "7": #graphically visualise data
            #lr = stats.linregress(data.Date, data.Open)
            #x = data.date
            #y = lr.intercept + lr.slope * x
            data_plot = data.plot(y = "Open", linestyle = '', marker = '.')
            data_plot.set_xlabel("Date")
            data_plot.set_ylabel("Total")
            #plt.plot(x, y, 'r') # 'r' = red
            plt.show(data_plot)
        elif Choice == "8": #Quit
            print("Thank you for using our programme")
        else:
            print("You filled the wrong choice, please select from the lists")
        input("Enter to continue: ")


def predictive_analytics(all_data_start_date,all_data_end_date):
    training_data_start_date = input("Please choose a start date of the modelling period (MM/DD/YYYY): ")
    training_data_end_date = input("Please choose an end date of the modelling period (MM/DD/YYYY): ")
    prediction_end_date = input("Please choose an end date of the prediction period (MM/DD/YYYY): ")

if __name__ == "__main__":

    [data, all_data_start_date, all_data_end_date] = gather_data()
    print(data)
    #print(start_date)
    #print(end_date)
    prescriptive_analytics(data, all_data_start_date, all_data_end_date)
    #prescriptive_analytics()
