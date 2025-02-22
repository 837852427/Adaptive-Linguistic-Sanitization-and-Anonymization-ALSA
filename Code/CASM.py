import numpy as np
from sklearn.cluster import KMeans
import sys
import nltk
from nltk.corpus import wordnet as wn
import random
from transformers import pipeline  # Assuming you use Hugging Face for LLM models

# Download the WordNet corpus for the NLTK library
nltk.download('wordnet')

#windows使用k均值的时候会有内存泄漏的风险，不知道linux什么情况
#后续如果依然出现风险，可以尝试修改
import warnings
warnings.filterwarnings('ignore') 

class CASMCalculator:
    def __init__(self, k=8, llm_model="gpt2"):
        """
        Initialize the CASM calculator.
        :param k: Number of clusters for k-means.
        """
        self.k = k
        self.llm_model = llm_model
        self.llm = pipeline("text-generation", model=llm_model)

    def calculate_casm(self, word_metrics):
        """
        Calculate the CASM actions for each word.
        :param word_metrics: Dictionary of word metrics (PLRS, CIIS, TRS) for each word.
        :return: Word actions based on the CASM model.
        """
        # Step 1: Calculate the assignment actions
        Retain_set, Replace_set, Encrypt_set, Delte_set = self.calculate_assignment_actions(word_metrics)

        # Step 2: Perform the actions
        words = {}
        words.update(self.action_retain(Retain_set))
        words.update(self.action_replace(Replace_set))
        words.update(self.action_encrypt(Encrypt_set))
        words.update(self.action_delete(Delte_set))

        return words

    def action_retain(self, words):
        """
        Return the words that should be retained.
        """
        
        return words
    
    def action_replace(self, words):
        """
        Return the words that should be replaced.
        """
        
        replace_words = {}
        for word, value in words.items():
            replace_words[word] = self.get_random_synonym(word)
        
        return replace_words

    def action_encrypt(self, words):

        return self.action_replace(words) #暂时先用wordnet随机替换，后面再改
    
    def action_delete(self, words):
        """
        Return the words that should be deleted.
        """
        delete_words = {}
        for word, value in words.items():
            delete_words[word] = ''
        
        return delete_words


    # The following methods are used to calculate the assignment actions based on the CASM model
    def calculate_assignment_actions(self, word_metrics):
        """
        Calculate the assignment actions for each word based on the CASM model.
        :param word_metrics: Dictionary of word metrics (PLRS, CIIS, TRS) for each word.
        :return: Dictionary of actions for each word.
        """
        # Step 1: Prepare the feature vectors for clustering
        word_vectors = []
        for word, (plrs, ciis, trs) in word_metrics.items():
            # Each word will be represented by its (PLRS, CIIS, TRS) values as a vector
            word_vectors.append([plrs, ciis, trs])

        # Step 2: Perform k-means clustering
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        clusters = kmeans.fit_predict(word_vectors)

        # Step 3: Assign actions based on clustering and conditions in Table 1
        centroids = kmeans.cluster_centers_

        if __name__ == "__main__":
            print(f'clusters: {clusters}\n\nkmeans.cluster_centers_: {kmeans.cluster_centers_}\n')

        print("\033[92m\033[1mK_means completed\033[0m\033[22m")

        Retain_set = {}
        Replace_set = {}
        Encrypt_set = {}
        Delte_set = {}
        actions = {}

        for i, word in enumerate(word_metrics):
            # Get the centroid of the cluster that the word belongs to
            cluster_id = clusters[i]
            centroid = centroids[cluster_id]

            # Check conditions for each dimension (PLRS, CIIS, TRS)
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
                print(f"\033[1;31mWord {word} does not match any condition.\033[0m")
                print(f"Centroid: {centroid}, Word metrics: {word_metrics[word]}")
                sys.exit(1)

        print(f'Retain_set: {Retain_set}\nReplace_set: {Replace_set}\nEncrypt_set: {Encrypt_set}\nDelte_set: {Delte_set}\n')

        if __name__ == "__main__":
            print("\033[92m\033[1mActions assigned\033[0m\033[22m")
            for word, action in actions.items():
                print(f"{word}: {action}")

        return Retain_set, Replace_set, Encrypt_set, Delte_set
    
    def check_retain(self, centroid, word_metrics):
        """
        Check if the word should be retained based on the conditions in Table 1.
        """
        [plrs, ciis, trs] = word_metrics
        [p_base, c_base, t_base] = centroid

        if(plrs < p_base and ciis >= c_base and trs < t_base): # low, high, low
            return True
        if(plrs < p_base and ciis < c_base and trs >= t_base): # low, low, high
            return True
        if(plrs < p_base and ciis >= c_base and trs >= t_base): # low, high, high
            return True
        
        return False
    
    def check_replace(self, centroid, word_metrics):
        """
        Check if the word should be replaced based on the conditions in Table 1.
        """
        [plrs, ciis, trs] = word_metrics
        [p_base, c_base, t_base] = centroid

        if(plrs >= p_base and ciis < c_base and trs >= t_base): # high, low, high
            return True
        
        return False
    
    def check_encrypt(self, centroid, word_metrics):
        """
        Check if the word should be encrypted based on the conditions in Table 1.
        """
        [plrs, ciis, trs] = word_metrics
        [p_base, c_base, t_base] = centroid

        if(plrs >= p_base and ciis >= c_base and trs < t_base): # high, high, low
            return True
        if(plrs >= p_base and ciis >= c_base and trs >= t_base): # high, high, high
            return True
    
        return False
    
    def check_delete(self, centroid, word_metrics):
        """
        Check if the word should be deleted based on the conditions in Table 1.
        """
        [plrs, ciis, trs] = word_metrics
        [p_base, c_base, t_base] = centroid

        if(plrs < p_base and ciis < c_base and trs < t_base): # low, low, low
            return True
        if(plrs >= p_base and ciis < c_base and trs < t_base): # high, low, low
            return True
        
        return False

    # Function to get a random synonym from WordNet
    def get_random_synonym(self, word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return random.choice(list(synonyms))
    
if __name__ == "__main__":
    # Example usage of the CASM calculator
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

    casm_calculator = CASMCalculator(k=2, model_name="distilgpt2")
    words = casm_calculator.calculate_casm(word_metrics)

    print("\n\033[1mCASM Actions:\033[0m")
    for word, action in words.items():
        print(f"{word}: {action}")
    
    