from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
import spacy

class IsolationTree:
    def __init__(self, height_limit, current_height=0):
        self.height_limit = height_limit
        self.current_height = current_height
        self.split_att = None   # Splitting feature index
        self.split_val = None   # Splitting threshold
        self.left = None        # Left subtree
        self.right = None       # Right subtree
        self.size = None        # Sample count at node

    def fit(self, X):
        """
        Build isolation tree recursively. Stop splitting when:
        1. Reaching height limit
        2. Node contains <=1 samples
        3. No valid split found
        """
        if self.current_height >= self.height_limit or len(X) <= 1:
            self.size = len(X)
            return

        self.split_att = np.random.randint(0, X.shape[1])
        min_val = np.min(X[:, self.split_att])
        max_val = np.max(X[:, self.split_att])

        if min_val == max_val:
            self.size = len(X)
            return

        self.split_val = np.random.uniform(min_val, max_val)
        left_idx = X[:, self.split_att] < self.split_val
        right_idx = ~left_idx

        self.left = IsolationTree(self.height_limit, self.current_height + 1)
        self.left.fit(X[left_idx])
        self.right = IsolationTree(self.height_limit, self.current_height + 1)
        self.right.fit(X[right_idx])

    def path_length(self, x):
        """Calculate path length for a sample through the tree"""
        if self.left is None and self.right is None:
            return self.current_height + c(self.size)
            
        if x[self.split_att] < self.split_val:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)

def build_isolation_forest(X, num_trees=8, sample_size=256):
    """
    Build isolation forest:
    1. Create multiple isolation trees
    2. Each tree uses random subsample
    3. Tree height limit: ceil(log2(sample_size))
    """
    trees = []
    height_limit = int(np.ceil(np.log2(sample_size)))

    for _ in range(num_trees):
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        tree = IsolationTree(height_limit)
        tree.fit(X_sample)
        trees.append(tree)

    return trees

def compute_average_depth(x, trees):
    """Calculate average path length across all trees"""
    depths = [tree.path_length(x) for tree in trees]
    return np.mean(depths)

def c(n):
    """
    Calculate expected path length adjustment term:
    c(n) = 2(ln(n-1) + γ) - 2(n-1)/n
    where γ is Euler-Mascheroni constant (~0.5772)
    """
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

def compute_outlier_score(avg_depth, sample_size):
    """
    Compute outlier score using formula:
    score = 2^(-avg_depth / c(sample_size))
    """
    c_val = c(sample_size)
    return 0.0 if c_val == 0 else 2 ** (-avg_depth / c_val)

class PLRSCalculator:
    def __init__(self, model, tokenizer, device, data_path='', spacy_model="en_core_web_sm"):
        self.data_path = data_path
        self.device = model.device
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.nlp = spacy.load(spacy_model)  # Load spaCy model for POS tagging and parsing

    def input_sentence(self):
        """
        Read the input sentence from a CSV file.
        """
        df = pd.read_csv(self.data_path)
        sentences = df['sentence'].tolist()

        return sentences
    
    def calculate(self):

        print('\n\033[1mCalculating PLRS Metrics\033[0m')

        sentences = self.input_sentence()

        # Calculate the CIIS scores for each sentence and merge them
        PLRS_scores = {}
        for sentence in sentences:
            PLRS_scores_sentence = self.calculate_PLRS(sentence)
            PLRS_scores.update(PLRS_scores_sentence)
        print("\n\033[1mCalculating CIIS completed\033[0m")


        print('\033[1;32mPLRS Completed\033[0m')
        return PLRS_scores
    
    def calculate_PLRS(self, sentence):
        words = self.get_words(sentence)
        embeddings = [self.get_word_embedding(word) for word in words]
        X = np.stack(embeddings)
        trees = build_isolation_forest(X, num_trees=8, sample_size=256)
        
        plrs_scors = {}
        for i, word in enumerate(words):
            x = X[i]
            avg_depth = compute_average_depth(x, trees)
            os_w = compute_outlier_score(avg_depth, sample_size=256)
            plrs_scors[(word, sentence)] = os_w
        
        return plrs_scors
    
    def get_word_embedding(self, word):
        """
        Generate word embedding using BERT:
        1. Tokenize input word
        2. Extract hidden states from BERT
        3. Handle multi-token words by averaging subword embeddings
        Returns: 768-dimensional numpy array
        """
        inputs = self.tokenizer(
            word, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # (sequence_length, hidden_size)
        if token_embeddings.shape[0] > 2:
            embedding = token_embeddings[1:-1].mean(dim=0)
        else:
            embedding = token_embeddings.mean(dim=0)
        return embedding.cpu().numpy()
    
    def get_words(self, sentence):
        """
        Implement identical token processing logic as CIIS
        """
        # Phase 1: SpaCy processing
        doc = self.nlp(sentence)
        pos_tags = [
            (token.text, token.pos_) 
            for token in doc 
            if token.pos_ != 'PUNCT'  # Exclude punctuation
        ]

        # Phase 2:  Merging
        merged_words = []
        for word, _ in pos_tags:
            merged_words.append(word)
            
        return merged_words

if __name__ == "__main__":
    model = "bert-base-uncased"
    plrs = PLRSCalculator(model=BertModel.from_pretrained(model),
                            tokenizer=BertTokenizer.from_pretrained(model),
                          data_path='data/ALSA.csv')
    ans = plrs.calculate()

    for (word, sentence), score in ans.items():
        print(f'{word}, {sentence}: {score}')