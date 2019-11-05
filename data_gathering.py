import pandas as pd
from datetime import datetime

# Load company list

COMP_LIST_URL = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
SYMBOL_INPUT = input("Please choose a symbol input: ")


def read_csv_file(url):
    df = pd.read_csv(url)
    return df


def get_comp_list():
    comp_df = read_csv_file(COMP_LIST_URL)
    comp_symbol = comp_df["Symbol"].values.tolist()
    return comp_symbol


def gather_data():

    if SYMBOL_INPUT in get_comp_list():

        start_date = input("Please choose a start date: ")  # date format: MM/DD/YYYY
        end_date = input("Please choose an end date: ")  # date format: MM/DD/YYYY

        # Check valid of input mm, dd

        if start_date <= end_date:
            url = "https://quotes.wsj.com/" + SYMBOL_INPUT + "/historical-prices/download?MOD_VIEW=page&" \
                  "num_rows=6299&range_days=6299&startDate=" + start_date + "&endDate=" + end_date

            symbol_df = read_csv_file(url)

        else:
            print("Please choose the start date before the end date")

    else:
        print("Sorry, the symbol is invalid. Please enter the valid symbol")

    return symbol_df


if __name__ == "__main__":

    a = gather_data()
    print(a)

