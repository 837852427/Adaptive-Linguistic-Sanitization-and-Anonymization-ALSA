import random
import re

class PLRSCalculator:
    def __init__(self):
        """
        Initialize the PLRS calculator.
        """
        
    def calculate_plrs(self, sentence):
        """
        Calculate Privacy Leakage Risk Score (PLRS) for each word in the sentence.
        PLRS is randomly generated for demonstration purposes.

        :param sentence: Input sentence to calculate PLRS for each word.
        :return: A dictionary with words as keys and randomly generated PLRS as values.
        """
        # Split the sentence into words using regular expression to handle punctuation
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Generate a dictionary with words as keys and random PLRS values
        plrs_scores = {word: random.random() for word in words}
        
        return plrs_scores

# Example usage
if __name__ == "__main__":
    calculator = PLRSCalculator()
    sentence = "Your example sentence goes here."
    plrs_scores = calculator.calculate_plrs(sentence)
    
    print(f'plrs_scores: {plrs_scores}')
    # Print the results
    for word, score in plrs_scores.items():
        print(f"Word: {word}, PLRS: {score:.4f}")