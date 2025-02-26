# **Portfolio part 1**

## Exercise 1 [0.5 marks]
#### Write a code for:
#### 1. Ask user to enter a number
#### 2. Check is the number is prime
#### 3. Print out the result (e.g. “This is a prime number” or “This is not a prime number”)

Code:
```python
n = int(input("Enter a number:"))
if n <= 1:
    print("not a prime number")
else:
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            print("not a prime number")
            break
        else:
            print("is a prime number")
```

```python
n = int(input("Enter a number:"))
if n <= 1:
    print("not a prime number")
```
This part of the code is to initially ask for the user's input and the check if the number is equal or less than 1. If it is, then the response will be that it's not a prime number.

```python
else:
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            print("not a prime number")
```
This section finds values up to the square root of the input number and checks if the input number is divisible by any of the numbers within the range. If it's divisible by any numbers the response will be that it's not a prime number.
