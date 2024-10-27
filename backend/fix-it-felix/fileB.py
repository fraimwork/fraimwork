# fileB contains a bug, depended on by fileA

def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0:
        return 0  # Bug: should return 1 instead of 0 for factorial(0)
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result