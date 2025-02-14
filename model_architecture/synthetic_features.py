# In this file can extract the features vector values from the input file
import logging
import string
import numpy as np

def extract_syntactic_features(cell):
    if not isinstance(cell, str):
        cell = str(cell) 

    length = len(cell)
    if length == 0:
        return [0] * 39

    # Handle the digits in the cell
    num_digits = sum(c.isdigit() for c in cell)
    
    # Handle the uppercase characters in the cell
    num_uppercase = sum(c.isupper() for c in cell)
    
    # Handle the lowercase characters in the cell
    num_lowercase = sum(c.islower() for c in cell)
    
    # Handle the punctuation marks in the cell
    num_punctuation = sum(c in string.punctuation for c in cell)
    
    # Handle the whitespace characters in the cell
    num_whitespace = sum(c.isspace() for c in cell)
    
    # Handle the special characters in the cell
    num_special_chars = sum(not c.isalnum() and not c.isspace() for c in cell)
    
    # Calculate the proportions of each type of character
    proportion_uppercase = round(num_uppercase / length, 3) if length > 0 else 0
    
    # Calculate the proportions of each type of character
    proportion_digits = round(num_digits / length, 3) if length > 0 else 0
    
    # Calculate the proportions of each type of character
    proportion_punctuation = round(num_punctuation / length, 3) if length > 0 else 0
    
    # Calculate the proportions of each type of character
    proportion_whitespace = round(num_whitespace / length, 3) if length > 0 else 0
    
    # Calculate the proportions of each type of character
    words = cell.split()
    num_words = len(words)
    
    # Handle the number of characters in each word
    avg_word_length = round(sum(len(word) for word in words) / num_words, 3) if num_words > 0 else 0
    
    # Handle the number of words in each sentence
    longest_word_length = max((len(word) for word in words), default=0)
    
    # Handle the shortest word in the sentence
    shortest_word_length = min((len(word) for word in words), default=0)
    
    # Handle the proportion of words in the sentence
    proportion_words = round(num_words / length, 3) if length > 0 else 0
    
    # Handle the presence of at-symbol in the cell
    contains_email = "@" in cell
    
    # Handle the presence of URL in the cell
    contains_url = any(substr in cell for substr in ["http://", "https://", "www."])
    
    # Handle the presence of hashtag in the cell
    contains_hashtag = "#" in cell
    
    # Handle the dollar sign in the cell
    contains_at_symbol = "$" in cell 
    
    # Handle the presence of numeric values in the cell 
    is_numeric = cell.isdigit()
    
    # Handle the presence of alphabetic characters in the cell
    is_alpha = cell.isalpha()
    
    # Handle the presence of alphanumeric characters in the cell
    is_alphanumeric = cell.isalnum()
    
    # Handle the capitalization of the cell
    is_capitalized = cell.istitle()


    # this loop calculates the Shannon Entropy of the cell
    try:
        shannon_entropy = -sum(
            (cell.count(c) / length) * np.log2(cell.count(c) / length)
            for c in set(cell)
        )
        shannon_entropy = round(shannon_entropy, 3) if length > 1 else 0
    except ValueError:
        shannon_entropy = 0


    # Unique character ratio
    unique_chars = len(set(cell))
    
    # Proportion of vowels in the cell
    proportion_vowels = round(sum(c in "aeiouAEIOU" for c in cell) / length, 3) if length > 0 else 0
    
    
    # Check if the cell is a palindrome
    is_palindrome = cell == cell[::-1]
    
    # Check for repeating characters in the cell
    repeating_chars = sum(cell[i] == cell[i - 1] for i in range(1, length))
    
    # Check for repeating words in the cell
    repeating_words = sum(words[i] == words[i - 1] for i in range(1, len(words))) if num_words > 1 else 0
    
    # Calculate ASCII values of first and last characters
    first_char_type = ord(cell[0]) if length > 0 else 0
    
    # Calculate ASCII values of first and last characters
    last_char_type = ord(cell[-1]) if length > 0 else 0
    
    # Calculate the most frequent character in the cell
    most_frequent_char = max((cell.count(c) for c in set(cell)), default=0)
    
    # Calculate the least frequent character in the cell
    least_frequent_char = min((cell.count(c) for c in set(cell)), default=0)
    
    # Frequency-based features
    digit_frequency = round(num_digits / length, 3) if length > 0 else 0
    
    # punctuation characters frequency
    punctuation_frequency = round(num_punctuation / length, 3) if length > 0 else 0
    
    # whitespace characters frequency
    whitespace_frequency = round(num_whitespace / length, 3) if length > 0 else 0
    
    # Character diversity ratio
    char_diversity_ratio = round(unique_chars / length, 3) if length > 0 else 0
    
    # Number of alphabetic sequences in the cell
    num_alpha_sequences = sum(1 for part in words if part.isalpha())


    # it returns a list of features extracted from the cell
    return [
        length, num_digits, num_uppercase, num_lowercase, num_punctuation, 
        num_whitespace, num_special_chars, proportion_uppercase, 
        proportion_digits, proportion_punctuation, proportion_whitespace,
        num_words, avg_word_length, longest_word_length, shortest_word_length, 
        proportion_words, contains_email, contains_url, contains_hashtag, 
        contains_at_symbol, is_numeric, is_alpha, is_alphanumeric, is_capitalized, 
        shannon_entropy, unique_chars, proportion_vowels, is_palindrome, 
        repeating_chars, repeating_words, first_char_type, last_char_type, 
        most_frequent_char, least_frequent_char, digit_frequency, 
        punctuation_frequency, whitespace_frequency, char_diversity_ratio, 
        num_alpha_sequences
    ]
    
    
test_cases = [
"Hartin123!!!!!!!",  
"HELLO123!",      
"hello",      
"Hello World",  
"123456",     
"www.google.com",  
"email@domain.com", 
"",           
"A",         
"AAABBBCCC"   
]

for test in test_cases:
    print(f"\n Testing: {test}")
    features = extract_syntactic_features(test)
    print("First 10 features:", features[:10])