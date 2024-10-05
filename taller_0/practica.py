
def is_upper_age(age):
    if (age >= 18):
        return True
    return False

def is_odd_number(number):
    if (number % 2 == 0):
        return True
    return False

def pizza(pizzas):
     ingredients = {
             "vegan": ["Vegetales", "Napolitana"],
             "no-vegan": ["Pepperoni", "Jamon", "Ranchera"]
    }

    option = int(input("Vegan or not vegan menu: \n 1. Vegan \n 2. No Vegan\n"))
    if option == 1:
        print("Available pizzas: ", ingredients["vegan"])
    elif option == 2:
        print("Available pizzas: ", ingredients["no-vegan"])
    else:
        print("Invalid option")
        return



    pizza_selection = int(input("Select a pizza by number: "))

    pizzas.append((option, pizza_selection))
    
    another_one = input("Do you want another one? (yes/no): ").strip().lower()
    if another_one == "yes":
        return pizza(pizzas)

    print("Your order: ", pizzas)



   

   

def main():
    option = int(input("Select an option \n 1. Upper age \n 2. Odd Number \n 3. Pizza!\n"))
    if option == 1:
        age = int(input("Put an age: "))
        if is_upper_age(age):
            print("The user is upper age")
        else:
            print("The user is under age")
    elif option == 2:
        number = int(input("Put a number: "))
        if is_odd_number(number):
            print("The number is odd")
        else:
            print("The number is even")
    elif option == 3:
        pizza([])

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
