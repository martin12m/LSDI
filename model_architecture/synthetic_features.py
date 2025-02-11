
import logging
import string
import numpy as np

def extract_syntactic_features(cell):
    """
    Extracts 39 syntactic features dynamically from cell values.
    - Works for both string and numeric inputs.
    - Ensures correctness by handling special cases.
    - Aligns with research paper's approach.
    """
    if not isinstance(cell, str):
        cell = str(cell)  # Convert non-string inputs to string safely

    length = len(cell)
    
    # Handle edge case where length is 0
    if length == 0:
        return [0] * 39

    num_digits = sum(c.isdigit() for c in cell)
    num_uppercase = sum(c.isupper() for c in cell)
    num_lowercase = sum(c.islower() for c in cell)
    num_punctuation = sum(c in string.punctuation for c in cell)
    num_whitespace = sum(c.isspace() for c in cell)
    num_special_chars = sum(not c.isalnum() and not c.isspace() for c in cell)

    # Proportions (Ensure safe division)
    proportion_uppercase = round(num_uppercase / length, 3) if length > 0 else 0
    proportion_digits = round(num_digits / length, 3) if length > 0 else 0
    proportion_punctuation = round(num_punctuation / length, 3) if length > 0 else 0
    proportion_whitespace = round(num_whitespace / length, 3) if length > 0 else 0

    # Word-level features
    words = cell.split()
    num_words = len(words)
    avg_word_length = round(sum(len(word) for word in words) / num_words, 3) if num_words > 0 else 0
    longest_word_length = max((len(word) for word in words), default=0)
    shortest_word_length = min((len(word) for word in words), default=0)
    proportion_words = round(num_words / length, 3) if length > 0 else 0

    # Special patterns
    contains_email = "@" in cell
    contains_url = any(substr in cell for substr in ["http://", "https://", "www."])
    contains_hashtag = "#" in cell
    contains_at_symbol = "@" in cell  # Redundant but kept for research alignment
    is_numeric = cell.isdigit()
    is_alpha = cell.isalpha()
    is_alphanumeric = cell.isalnum()
    is_capitalized = cell.istitle()

    # Shannon entropy (Fixed for empty or single-character strings)
    try:
        shannon_entropy = -sum(
            (cell.count(c) / length) * np.log2(cell.count(c) / length)
            for c in set(cell)
        )
        shannon_entropy = round(shannon_entropy, 3) if length > 1 else 0
    except ValueError:
        shannon_entropy = 0  # Handles log(0) issue safely

    # Unique character ratio
    unique_chars = len(set(cell))
    proportion_vowels = round(sum(c in "aeiouAEIOU" for c in cell) / length, 3) if length > 0 else 0

    # Pattern-based features
    is_palindrome = cell == cell[::-1]
    repeating_chars = sum(cell[i] == cell[i - 1] for i in range(1, length))
    
    # Fix for repeating words in single-word inputs
    repeating_words = sum(words[i] == words[i - 1] for i in range(1, len(words))) if num_words > 1 else 0

    # Character types (ASCII values of first and last char)
    first_char_type = ord(cell[0]) if length > 0 else 0
    last_char_type = ord(cell[-1]) if length > 0 else 0

    # Most/Least frequent character occurrences
    most_frequent_char = max((cell.count(c) for c in set(cell)), default=0)
    least_frequent_char = min((cell.count(c) for c in set(cell)), default=0)

    # Frequency-based features
    digit_frequency = round(num_digits / length, 3) if length > 0 else 0
    punctuation_frequency = round(num_punctuation / length, 3) if length > 0 else 0
    whitespace_frequency = round(num_whitespace / length, 3) if length > 0 else 0
    char_diversity_ratio = round(unique_chars / length, 3) if length > 0 else 0
    num_alpha_sequences = sum(1 for part in words if part.isalpha())

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
"Martin123!!!!!!!",  # Mix of letters, digits, punctuation
"HELLO",      # All uppercase
"hello",      # All lowercase
"Hello World",  # Multi-word
"123456",     # All digits
"www.google.com",  # URL
"email@domain.com",  # Email
"",           # Empty string (edge case)
"A",          # Single-character string
"AAABBBCCC"   # Repeating characters
]

for test in test_cases:
    print(f"\n🔍 Testing: {test}")
    features = extract_syntactic_features(test)
    print("First 10 features:", features[:10])