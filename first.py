def fibo(f1, f2, a, count):
    if count == a:
        return
    else:
        k = f1 + f2
        print(k)
        fibo(f2, k, a, count + 1)

# main program
a = int(input("Enter the number of terms: "))

# first two numbers
f1 = 0
f2 = 1
count = 0

if a <= 0:
    print("Please enter a positive integer")
elif a == 1:
    print(f1)
else:
    print(f1)
    print(f2)
    fibo(f1, f2, a - 2, count)  # subtract 2 because 0 and 1 already printed
