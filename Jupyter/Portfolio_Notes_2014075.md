# **Portfolio part 1**

## Exercise 1 [0.5 marks]
### Task:
#### Write a code for:
#### 1. Ask user to enter a number
#### 2. Check is the number is prime
#### 3. Print out the result (e.g. “This is a prime number” or “This is not a prime number”)


### Answer:
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

## Exercise 2 [0.5 marks]

### Task:
#### Write a code for printing out Fibonacci sequence. Ask a user to enter how many numbers in Fibonacci set to be calculated and displayed.

### Answer:
Code:
```python
n = int(input("Enter Number of Terms in sequence:"))
sequence = [0,1]
for x in range(2,n):
    Next_term= sequence[-1] + sequence[-2]
    sequence.append(Next_term)
print(sequence)
```

```python
n = int(input("Enter Number of Terms in sequence:"))
sequence = [0,1]
```
Code begins with asking input number for the amount of terms the fibonacci sequence will be caluclated up to. The first two terms of the sequence have to be defined to start the sequence. This is so that later in the code the next terms can be much easily calculated using the first two terms as reference.

```python
for x in range(2,n):
    Next_term= sequence[-1] + sequence[-2]
```
This part (above) sets up the loop that will calculate the next terms using the first two terms, that were earlier defined, as reference.
The 
```python 
for x in range(2,n)
``` 
function starts the calculation from the third term '2' and goes up to the nth term 'n'. Without this function the code would not loop up the nth term.

The equation used:
```python
Next_term= sequence[-1] + sequence[-2]
```
means that the next term is calculated by adding the last term (current term) and the second last term (previous term).
```python
    sequence.append(Next_term)
print(sequence)
```
Finally, the terms are compiled into a list with the append function that adds the next term to the list each time the loop runs. Once the list is complete it is then printed out.