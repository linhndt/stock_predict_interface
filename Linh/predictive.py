import pandas as pd
from data_gathering import gather_data
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def select_variable(df):
    """
    Select variable that users want to play with from the list of headings of the data frame.
    Returns an array contains the value of chosen variable and the name of chosen variable

    Parameters:
    --------------
    df: data frame
        Initial data frame

    Returns
    --------------
    variable_array: ndarray
        An array of values of chosen variable
    selected_variable: string
        Name of chosen variable
    """

    selected_number = -1

    # Print the choices and let a user to select a choice

    while float(selected_number) not in range(len(df.columns)):
        # Use float because we want to check if a user doesn't fill an interge (eg. 1.2)

        print('/' + '{:-^40}'.format('Lists of variable') + '\\')

        for i, item in enumerate(df.columns[: -1]):
             print_choice = '    {}. {}'.format(i, item)

             print('|' + '{:<40}'.format(print_choice) + "|")

        print('\\' + '-'*40 + '/')

        selected_number = input(" Select your choice from the above lists (only number) : ")

        if float(selected_number) not in range(len(df.columns)):
            print('***Wrong Input, please try again')

    selected_variable = df.columns[int(selected_number)]
    variable_array = df[selected_variable].values

    return variable_array, selected_variable


def convert_int_date(df):
    """
    Convert date value each row into an integer value, store them in an array.
    Returns an integer date array.

    Parameters:
    --------------
    df: data frame
        Initial data frame

    Returns
    --------------
    int_date_array: ndarray
        An array contains value of integer dates

    """
    # Convert each date value in Date column into an integer value, store them in a list
    int_date_list = [matplotlib.dates.date2num(date) for date in df.index]

    # Convert int_date_list into an array:
    int_date_array = np.array(int_date_list)

    return int_date_array


def find_model_coff(x, y):
    """
    Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
    Returns a vector of coefficients p that minimises the squared error in the order deg, deg-1, … 0.

    * Note: users need to input a degree of the model.

    Parameters:
    --------------
    x : array_like, shape (M,)
        x-coordinates of the M sample points (x[i], y[i]).
    y : array_like, shape (M,)
        y-coordinates of the sample points.

    Returns:
    --------------
    model_coff : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.
    """

    # Input degree of model

    degree = 0
    while degree < 1 or degree % 1 != 0:
        degree = input('Please select the degree of the model: ')
        degree = float(degree)
        if degree < 1 or degree % 1 != 0:
            print('The degree has to be an integer which is equal or more than 1')

    # Find the model
    model_coff = np.polyfit(x, y, int(degree))

    return model_coff


def model_predict(x, model_coff):
    """
    Return prediction value of f(x), with f(x) is polynomial model with model_coff cofficients.

    Parameters:
    --------------
    x : array_like, shape (1, 1) or (M,)
        x-coordinates of the M sample points (x[i], y[i]).
    model_coff: array_like, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.
    Returns:
    --------------
    y_hat: float or array_like shape (M,)
        Predicted value of f(x)
    """

    y_hat = x * 0

    for i in range(len(model_coff)):
        y_hat += model_coff[(len(model_coff) - 1) - i] * (x ** i)
        # Example: y_hat = model[0] * x^1 + model[1]
    return y_hat


def cal_MSE(y, y_hat):
    """
    Calculate mean square error(MSE) of training model
    https://en.wikipedia.org/wiki/Mean_squared_error

    Parameters:
    --------------
    y: array_like, shape (M,)
        Observed values of M sample points
    y_hat: array_like, shape (M,)
        Predicted values of M sample points based on the model.
    Returns:
    --------------
    error: float
        MSE value of the training model
    """

    n = len(y)
    diff_square = (y - y_hat)**2
    error = (1 / n) * sum(diff_square)
    return error


def cal_r_square(y, y_hat):
    """
    Calculate R-square(R^2) of training model
    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Parameters:
    --------------
    y: array_like, shape (M,)
        Observed values of M sample points
    y_hat: array_like, shape (M,)
        Predicted values of M sample points based on the model.
    Returns:
    --------------
    error: float
        R-square value of the training model
    """

    # Calculate SSE
    n = len(y)
    diff_square = (y - y_hat)**2
    SSE = sum(diff_square)

    # Calculate TSS
    y_bar = (1 / n) * sum(y)
    diff_square = (y - y_bar)**2
    TSS = sum(diff_square)

    # Calculate R_Square
    error = 1 - (SSE/TSS)
    return error


def visualize_training_model(df, selected_col, y_hat, pred_col_name="Predict"):
    """
    Visualize training model

    Parameters:
    --------------
    df: data frame
        An initial data frame
    selected_col: string
        Name of the column of df to visualize
    y_hat: array_like
        An array of predicted values based on training model
    pre_col_name: string, default "Predict"
        Name of the column storing value of y_hat
    """

    df.insert(len(df.columns), pred_col_name, y_hat, True)
    df[selected_col].plot(marker=".", linestyle="None")
    df[pred_col_name].plot()
    plt.show()


def predictive():

    # 0. Introduction

    print("{:-^50}".format('Welcome to predictive analysis'))

    # 1. Data selection:
    # Users have to choose the training period for predictive analysis.

    stock_data = gather_data()

    # 2. Variable Selection:
    # Users have to choose the variable they want to play with.

    while not stock_data.empty:

        while True:

            try:
                [y, selected_variable] = select_variable(stock_data)
                break

            except ValueError:
                print('{:*^30}'.format('WARNING'))
                print("Wrong input!! Please fill the number")
                continue

        # Build model

        int_date_array = convert_int_date(stock_data)

        while True:
            try:
                model_coff = find_model_coff(int_date_array, stock_data[selected_variable])
                break
            except ValueError:
                print('{:*^30}'.format('WARNING'))
                print("Wrong input!! Please fill the number")
                continue

        y_hat = model_predict(int_date_array, model_coff)

        # 3. Options Selection:
        # Users have to choose the options they want to do: Graph, Model Errors, Prediction, Quit.

        choices = ['Graph', 'Prediction errors', 'Forcasting', 'Quit']

        select = -1

        while select != 3:

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

                # 3.1. Visualisation

                visualize_training_model(stock_data, selected_variable, y_hat)

            elif select == 1:

                # 3.2. Calculate R^2 and RMSE

                # Solve model:

                # MSE = 1/n sum(from i=1 to n) (y(i)- y_hat(i))^2
                MSE = cal_MSE(y, y_hat)

                # RMSE = sqrt(MSE)
                RMSE = MSE ** (1 / 2)

                # R_Square = 1 - (SSE/TSS)
                R_Square = cal_r_square(y, y_hat)

                # Display
                print('/' + '{:-^30}'.format('Prediction errors') + '\\')
                print('|' + ' MSE  : {:<22}'.format(round(MSE, 3)) + '|')
                print('|' + ' RMSE : {:<22}'.format(round(RMSE, 3)) + '|')
                print('|' + ' R^2  : {:<22}'.format(round(R_Square, 3)) + '|')
                print('\\' + '-' * 30 + '/')

                input('Press \'Enter\' if you want to continue: ')
                del MSE, RMSE, R_Square  # del variable to save data storage

            elif select == 2:

                # 3.3 Forecast
                # 3.3.1 select date that you want to forecast

                while True:
                    try:
                        forecast_date = input('Please specify a date that you want to forecast ("MM/DD/YYYY"): ')
                        forecast_date = pd.to_datetime(forecast_date, format="%m/%d/%Y")
                        break
                    except ValueError:
                        print('{:*^30}'.format('WARNING'))
                        print("Wrong Format!! Please fill the date in MM/DD/YYYY format")
                        continue

                forecast_date_num = matplotlib.dates.date2num(forecast_date)

                # 3.3.2 Forcasting

                y_forecast = model_predict(forecast_date_num, model_coff)

                # 3.3.3 Display

                print("The forecast value is {:0.2f}.".format(y_forecast))
                input('Press \'Enter\' if you want to continue: ')

            elif select == 3:

                print("{:-^40}".format('Goodbye'))

            else:
                print("***Wrong choice***")

    print("There is no information on time given")


if __name__ == '__main__':

    predictive()