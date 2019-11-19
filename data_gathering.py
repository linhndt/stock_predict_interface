import pandas as pd

# Load company list


def get_comp_list():
    """
    Get the list of companies' symbols from Nasdaq url resource.
   "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"

    Return:
    ----------------
    comp_symbol: a list of companies' symbol
    """

    COMP_LIST_URL = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
    comp_df = pd.read_csv(COMP_LIST_URL)
    comp_symbol = comp_df["Symbol"].tolist()
    return comp_symbol


def remove_initial_space(df):
    """
    Remove the initial space of headings in a data frame

    Parameters:
    ----------------
    df: data frame
        A data frame whose each heading contains an initial space.

    Returns:
    ----------------
    df: data frame
        A data frame whose headings does not contain initial spaces.
    """

    column_name = []
    for i in range(len(df.columns)):
        column_name.append("")
        if i == 0:
            column_name[i] = df.columns[i]
        else:
            column_name[i] = df.columns[i][1:]
    df.columns = column_name

    return df


def gather_data():
    """
    Users need to input the following information:
    symbol_input: company symbol
    start_date: start date of the query period (format: MM/DD/YYYY)
    end_date: end date of the query period (format: MM/DD/YYYY)

    These information will be concatenated with Wall Street Journal API to create the url to download data.
    The url template is :
    "https://quotes.wsj.com/" + symbol_input + "/historical-prices/download?MOD_VIEW=page&" \
    "num_rows=6299&range_days=6299&startDate=" + start_date + "&endDate=" + end_date

    Returns:
    ----------------
    stock_data: data frame
        A data frame contains stock information of query period: Date, Volumn, Close price, Open price ...
    """

    symbol_input = input("Please choose a symbol input: ").upper().strip()

    while symbol_input not in get_comp_list():
        print("Sorry, the symbol is invalid.")
        symbol_input = input("Please choose a symbol input: ").upper().strip()

    while True:
        try:
            start_date = input("Please choose a start date (MM/DD/YYYY): ")
            start_date_check = pd.to_datetime(start_date, format="%m/%d/%Y")
            end_date = input("Please choose an end date: (MM/DD/YYYY): ")
            end_date_check = pd.to_datetime(end_date, format="%m/%d/%Y")
            break

        except ValueError:
            print('{:*^30}'.format('WARNING'))
            print("Wrong Format!! Please fill the date in MM/DD/YYYY format")
            continue

    while start_date > end_date:
        print("Please choose the start date before the end date")

        while True:
            try:
                start_date = input("Please choose a start date (MM/DD/YYYY): ")
                start_date = pd.to_datetime(start_date, format="%m/%d/%Y")
                end_date = input("Please choose an end date: (MM/DD/YYYY): ")
                end_date = pd.to_datetime(end_date, format="%m/%d/%Y")
                break

            except ValueError:
                print('{:*^30}'.format('WARNING'))
                print("Wrong Format!! Please fill the date in MM/DD/YYYY format")
                continue

    url = "https://quotes.wsj.com/" + symbol_input + "/historical-prices/download?MOD_VIEW=page&" \
          "num_rows=6299&range_days=6299&startDate=" + start_date + "&endDate=" + end_date

    stock_data = pd.read_csv(url)

    # Amend the headings: Remove the initial space from the headings

    stock_data = remove_initial_space(stock_data)

    # Change data type from object to a date time object

    stock_data["Date"] = pd.to_datetime(stock_data["Date"], format="%m/%d/%y")
    stock_data = stock_data.sort_values('Date')
    stock_data = stock_data.set_index('Date')

    return stock_data


if __name__ == "__main__":

    a = gather_data()
    print(a)