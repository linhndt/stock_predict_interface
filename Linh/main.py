# import library
import predictive as pre
import descriptive as des


def main():

    print("Welcome to the Stock Analytics Interface.\nPlease read our menu")
    print("1. Stock Analytics \n2. Stock Prediction \n3. Quit")

    choice = int(input("Please select your choice: "))

    while choice != 3:

        # Using the while loops allows the customer to quit from the choice.
        # Continue the choice until customers choose 3. Quit.

        if choice == 1:
            # perform Stock Analytics:
            des.descriptive()

        elif choice == 2:
            # perform Stock Prediction:
            pre.predictive()

        else:
            print("Wrong choice. Please re-select.")

        print("Welcome to the Stock Analytics Interface.\nPlease read our menu")
        print("1. Stock Analytics \n2. Stock Prediction \n3. Quit")
        choice = int(input("Please select your choice: "))

    print("Thank you for using our product.")


if __name__ == "__main__":

    main()
