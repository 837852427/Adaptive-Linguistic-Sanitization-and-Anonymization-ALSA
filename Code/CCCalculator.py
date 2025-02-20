import spacy
import torch
import numpy as np
from scipy.spatial.distance import mahalanobis

class CCCalculator:
    def __init__(self, model, tokenizer, alpha=0.5, beta=0.5, gamma=0.3, spacy_model="en_core_web_sm"):
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = gamma  # Function word weight constant (gamma)
        self.nlp = spacy.load(spacy_model)  # Load spaCy model for POS tagging and dependency parsing
        self.alpha = alpha
        self.beta = beta

    def calculate_cc(self, sentence):
        """
        Calculate the Contextual Coherence (CC) for each word in a sentence.
        Incorporates both semantic and positional differences.
        """
        pos_tags = self.get_pos_tags_spacy(sentence)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)  # Enable hidden states output

        # Use the last hidden state as the embedding
        embeddings = outputs.hidden_states[-1].squeeze(0)  # Shape: (sequence_length, hidden_size)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        words = [self.tokenizer.convert_tokens_to_string([t]) for t in tokens if t not in self.tokenizer.all_special_tokens]
        word_embeddings = embeddings[:len(words)].cpu().numpy()

        # Calculate the Dependency Matrix D_ij
        D = self.calculate_D_matrix(pos_tags)

        # Calculate pairwise distances using Mahalanobis distance
        T = np.zeros((len(words), len(words)))
        for i, v_i in enumerate(word_embeddings):
            for j, v_j in enumerate(word_embeddings):
                if i != j:
                    T[i, j] = self.calculate_mahalanobis_distance(v_i, v_j, D[i, j], self.alpha)

        # Calculate positional distances
        R = np.zeros((len(words), len(words)))
        for i in range(len(words)):
            for j in range(len(words)):
                if i != j:
                    R[i, j] = abs(i - j)

        # Final CC calculation
        cc_scores = {}
        epsilon = 1e-9  # Small value to avoid division by zero
        for i, word in enumerate(words):
            sum = 0
            for j in range(len(words)):
                if i != j:
                    sum += self.beta * R[i, j] + T[i, j] + epsilon
            cc_scores[word] = 1 / sum

        return cc_scores
    
    def calculate_D_matrix(self, pos_tags, embedding_dim=768):
        """
        Calculate the dependency matrix D_ij based on part-of-speech (POS) tags using spaCy.
        Each entry D_ij is a matrix representing the dependency between two words w_i and w_j.
        """
        # Initialize the matrix as a 2D array
        num_words = len(pos_tags)
        D = np.zeros((num_words, num_words), dtype=object)  # Create a matrix of objects (for storing matrices)

        for i in range(num_words):
            for j in range(num_words):
                # Get the POS tags for words i and j
                pos_i = pos_tags[i][1]  # POS tag for word i
                pos_j = pos_tags[j][1]  # POS tag for word j

                # Calculate the dependency weight based on whether the words are content or function words
                rho_i = 1.0 if self.is_content_word(pos_i) else self.gamma
                rho_j = 1.0 if self.is_content_word(pos_j) else self.gamma

                # Compute the dependency weight between word i and word j as a matrix
                # Create a scaled identity matrix of size (embedding_dim, embedding_dim)
                D[i, j] = rho_i * rho_j * np.eye(embedding_dim)  # Dependency weight (scaled identity matrix)

        return D

    def is_content_word(self, pos_tag):
        """
        Check if the word is a content word (noun, verb, adjective, adverb).
        """
        content_pos_tags = ['NN', 'VB', 'JJ', 'RB']  # Example content POS tags
        return pos_tag in content_pos_tags

    def calculate_mahalanobis_distance(self, v_i, v_j, D_ij, alpha=0.5):
        """
        Calculate the weighted Mahalanobis distance between two vectors with dependency weight Q_ij.
        """
        # Q_ij is a matrix, and D_ij is a scaled identity matrix (embedding_dim x embedding_dim)
        Q_ij = np.eye(v_i.shape[0]) + alpha * D_ij # Weighted dependency matrix of shape (embedding_dim, embedding_dim)
        diff = v_i - v_j
        return np.sqrt(diff.T @ Q_ij @ diff)

    def get_pos_tags_spacy(self, sentence):
        """
        Use spaCy to get POS tags.
        """
        doc = self.nlp(sentence)
        pos_tags = [(token.text, token.pos_) for token in doc]  # Return word and POS tag
        return pos_tags

if __name__ == "__main__":
    # Example usage
    from transformers import BertModel, BertTokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cc_calculator = CCCalculator(model, tokenizer)

    sentence = "The quick brown fox jumps over the lazy dog"
    cc_scores = cc_calculator.calculate_cc(sentence)
    print("Contextual Coherence scores:\n", cc_scores)