import numpy as np
from sklearn.cluster import KMeans
import sys
import nltk
from nltk.corpus import wordnet as wn
import random
from transformers import pipeline

# Download the WordNet corpus
nltk.download('wordnet')

# Note: Potential memory leak risk when using k-means on Windows
import warnings
warnings.filterwarnings('ignore')

class CASMCalculator:
    def __init__(self, k=8, llm_model="gpt2"):
        """
        Initialize CASM calculator
        :param k: Number of clusters for k-means
        """
        self.k = k
        self.llm_model = llm_model
        self.llm = pipeline("text-generation", model=llm_model)

    def calculate_casm(self, word_metrics):
        """
        Calculate CASM actions for each word
        :param word_metrics: Dictionary of word metrics (PLRS, CIIS, TRS)
        :return: Dictionary of word actions
        """
        # Step 1: Determine assignment actions
        Retain_set, Replace_set, Encrypt_set, Delte_set = self.calculate_assignment_actions(word_metrics)

        # Step 2: Execute actions
        words = {}
        words.update(self.action_retain(Retain_set))
        words.update(self.action_replace(Replace_set))
        words.update(self.action_encrypt(Encrypt_set))
        words.update(self.action_delete(Delte_set))

        return words

    def action_retain(self, words):
        """Preserve words without modification"""
        return words
    
    def action_replace(self, words):
        """Replace words with synonyms"""
        replace_words = {}
        for word, value in words.items():
            replace_words[word] = self.get_random_synonym(word)
        return replace_words

    def action_encrypt(self, words):
        """Temporarily using WordNet for random replacement; to be updated later"""
        return self.action_replace(words)
    
    def action_delete(self, words):
        """Remove words from text"""
        delete_words = {}
        for word, value in words.items():
            delete_words[word] = ''
        return delete_words

    def calculate_assignment_actions(self, word_metrics):
        """
        Assign actions based on k-means clustering
        :param word_metrics: Dictionary of word metrics
        :return: Tuple of action sets
        """
        # Prepare feature vectors
        word_vectors = []
        for word, (plrs, ciis, trs) in word_metrics.items():
            word_vectors.append([plrs, ciis, trs])

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        clusters = kmeans.fit_predict(word_vectors)
        centroids = kmeans.cluster_centers_

        if __name__ == "__main__":
            print(f'clusters: {clusters}\n\ncentroids: {centroids}\n')

        print("\033[92m\033[1mK-means completed\033[0m\033[22m")

        # Initialize action sets
        Retain_set = {}
        Replace_set = {}
        Encrypt_set = {}
        Delte_set = {}
        actions = {}

        # Assign actions based on cluster centroids
        for i, word in enumerate(word_metrics):
            cluster_id = clusters[i]
            centroid = centroids[cluster_id]

            if self.check_retain(centroid, word_metrics[word]):
                Retain_set[word] = word
                actions[word] = "retain"
            elif self.check_replace(centroid, word_metrics[word]):
                Replace_set[word] = word
                actions[word] = "replace"
            elif self.check_encrypt(centroid, word_metrics[word]):
                Encrypt_set[word] = word
                actions[word] = "encrypt"
            elif self.check_delete(centroid, word_metrics[word]):
                Delte_set[word] = word
                actions[word] = "delete"
            else:
                print(f"\033[1;31mNo matching action for {word}\033[0m")
                print(f"Centroid: {centroid}, Metrics: {word_metrics[word]}")
                sys.exit(1)

        print(f'Retain: {Retain_set}\nReplace: {Replace_set}\nEncrypt: {Encrypt_set}\nDelete: {Delte_set}\n')

        if __name__ == "__main__":
            print("\033[92m\033[1mActions assigned\033[0m\033[22m")
            for word, action in actions.items():
                print(f"{word}: {action}")

        return Retain_set, Replace_set, Encrypt_set, Delte_set
    
    def check_retain(self, centroid, metrics):
        """Check retain conditions: low, high, low | low, low, high | low, high, high"""
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return (plrs < p_base and ciis >= c_base and trs < t_base) or \
               (plrs < p_base and ciis < c_base and trs >= t_base) or \
               (plrs < p_base and ciis >= c_base and trs >= t_base)
    
    def check_replace(self, centroid, metrics):
        """Check replace condition: high, low, high"""
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return plrs >= p_base and ciis < c_base and trs >= t_base
    
    def check_encrypt(self, centroid, metrics):
        """Check encrypt conditions: high, high, low | high, high, high"""
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return (plrs >= p_base and ciis >= c_base and trs < t_base) or \
               (plrs >= p_base and ciis >= c_base and trs >= t_base)
    
    def check_delete(self, centroid, metrics):
        """Check delete conditions: low, low, low | high, low, low"""
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return (plrs < p_base and ciis < c_base and trs < t_base) or \
               (plrs >= p_base and ciis < c_base and trs < t_base)

    def get_random_synonym(self, word):
        """Get random synonym from WordNet"""
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return random.choice(list(synonyms)) if synonyms else word
    
if __name__ == "__main__":
    # Example usage
    word_metrics = {
        "The": [0.1, 0.89, 0.2],
        "quick": [0.8, 0.1, 0.9],
        "brown": [0.2, 0.8, 0.28],
        "fox": [0.9, 0.26, 0.1],
        "jump": [0.33, 0.7, 0.3],
        "over": [0.7, 0.23, 0.7],
        "lazy": [0.4, 0.6, 0.4],
        "dog": [0.0, 0.0, 0.0],
        "clear": [0.18, 0.8, 0.8],
        "crazy": [0.9, 0.29, 0.9],
        "beauty": [0.7, 0.7, 0.677]
    }

    casm = CASMCalculator(k=2)
    actions = casm.calculate_casm(word_metrics)

    print("\n\033[1mCASM Actions:\033[0m")
    for word, action in actions.items():
        print(f"{word}: {action}")
