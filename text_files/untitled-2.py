import random

def make_random_code():
    """returns four characters 'R', 'G', 'B', 'Y', 'O', or 'W' for Mastermind.
    """
    list = ['R', 'G', 'B', 'Y', 'O', 'W']
    code = []
    for n in range(0, 4):
           code += random.choice(list)
    return code
    
def count_exact_matches(code, guess):
    """Returns the number of places where two strings have the same letters
    at the same locations"""
    total = 0
    for x in code:
        if x in guess:
            total += 1
        else:
            total = total
    return total        
    
def count_letter_matches(code, guess):
    """Returns the number matches in the two strings no matter their location.
    """
    l1 = list(code.split(code))
    l2 = list(guess.split(guess))
    total = 0
    for x in code:
        if x in guess:
            for y in guess:
                if x == y:
                    total += 1
                    l1.remove(x)
                    l2.remove(y)
        else:
            total = total
    return total
                