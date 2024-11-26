# fileA: non-broken file that depends on fileB and fileC

from fileB import factorial
from fileC import validate_positive_integer

def combinations(n, r):
    if not (validate_positive_integer(n) and validate_positive_integer(r)):
        return "Inputs must be positive integers."

    if r > n:
        return "r must be less than or equal to n."

    # Calculate combinations using the formula n! / (r! * (n - r)!)
    try:
        return factorial(n) // (factorial(r) * factorial(n - r))
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Test case
print("Combinations of 5 choose 2:", combinations(5, 2))
print("Combinations of 5 choose 0:", combinations(5, 0))
print("Combinations of 5 choose 6:", combinations(5, 6))  # Should return an error
print("Combinations of -5 choose 2:", combinations(-5, 2))  # Should return an error
