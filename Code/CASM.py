import numpy as np
from sklearn.cluster import KMeans
import sys
import nltk
from nltk.corpus import wordnet as wn
import random
from transformers import pipeline

# Download the WordNet corpus
# nltk.download('wordnet')

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

    def calculate(self, word_metrics):
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
        for (word, sentence), value in words.items():
            replace_words[(word, sentence)] = self.get_random_synonym(word)
        return replace_words

    def action_encrypt(self, words):
        """ Word Replacement"""
        def get_diff_word(word):
            prompt = f"Give a different English word than '{word}': "
            try:
                response = self.llm(
                    prompt,
                    temperature=0.9,  # 提高随机性
                    max_new_tokens=8
                ).strip().lower()
                
                # 基础清洗：取第一个单词
                result = response.split()[0] if response else word
                return result if result != word else f"{word}_alt"
                
            except:
                return f"{word}_alt"

        return {(word, sentence): get_diff_word(word) for (word, sentence) in words}

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
        for (word, sentence), (plrs, ciis, trs) in word_metrics.items():
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
        for i, (word, sentence) in enumerate(word_metrics):
            cluster_id = clusters[i]
            centroid = centroids[cluster_id]
            key = (word, sentence)

            if self.check_retain(centroid, word_metrics[key]):
                Retain_set[key] = word
                actions[key] = "retain"
            elif self.check_replace(centroid, word_metrics[key]):
                Replace_set[key] = word
                actions[key] = "replace"
            elif self.check_encrypt(centroid, word_metrics[key]):
                Encrypt_set[key] = word
                actions[key] = "encrypt"
            elif self.check_delete(centroid, word_metrics[key]):
                Delte_set[key] = word
                actions[key] = "delete"
            else:
                print(f"\033[1;31mNo matching action for {word}\033[0m")
                print(f"Centroid: {centroid}, Metrics: {word_metrics[word]}")
                sys.exit(1)

        print(f'Retain: {Retain_set}\nReplace: {Replace_set}\nEncrypt: {Encrypt_set}\nDelete: {Delte_set}\n')

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
        ('Jon', 'Jon applied for a loan using his credit card.'): [0.1, 0.89, 0.2],
        ('applied', 'Jon applied for a loan using his credit card.'): [0.8, 0.1, 0.9],
        ('for', 'Jon applied for a loan using his credit card.'): [0.2, 0.8, 0.28],
        ('a', 'Jon applied for a loan using his credit card.'): [0.9, 0.26, 0.1],
        ('loan', 'Jon applied for a loan using his credit card.'): [0.33, 0.7, 0.3],
        ('using', 'Jon applied for a loan using his credit card.'): [0.7, 0.23, 0.7],
        ('his', 'Jon applied for a loan using his credit card.'): [0.4, 0.6, 0.4],
        ('credit', 'Jon applied for a loan using his credit card.'): [0.0, 0.0, 0.0],
        ('card', 'Jon applied for a loan using his credit card.'): [0.18, 0.8, 0.8],
    }

    casm = CASMCalculator(k=2)
    actions = casm.calculate(word_metrics)

    print("\n\033[1mCASM Actions:\033[0m")
    print(f'\nwords_replace:\n{actions}\n')
    for word, action in actions.items():
        print(f"{word}: {action}")
