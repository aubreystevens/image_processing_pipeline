import sys

def sum_of_key_values(dict, key1, key2):
    '''Return the sum of the values in the dictionary stored at key1 and key2.'''
    try:
        return dict[key1] + dict[key2]
    except KeyError:   
        quit()
        
dict = {'this': 'is', 'a': 'test'}

sum_of_key_values(dict, 'this', 'a')