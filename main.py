
# import library

def main():

    print("Welcome to the Stock Analytics Interface.")
    print("Please read our menu")
    print(" 1. Stock Query \n 2. Stock Analytics \n 3. Stock Prediction \n 4. Quit")

    choice = int(input("Please select your choice: "))

    while choice != 4:

        # Using the while loops allows the customer to quit from the choice.
        # Continue the choice until customers choose 4. Quit.

        if choice == 1:
            # perform stock query

        elif choice == 2:
            # perform stock analytics

        elif choice == 3:
            # perform stock prediction

        else:
            print("Wrong choice. Please re-select.")

        choice = int(input("Please select your choice: "))

    print("Thank you for using our product.")

if __name__ == "__main__":

    main()
