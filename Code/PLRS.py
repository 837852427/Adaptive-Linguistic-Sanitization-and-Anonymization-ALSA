import numpy as np
import random
import torch
from transformers import BertTokenizer, BertModel
import nltk

# =====================
# 0. Set Random Seeds
# =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Download NLTK word corpus (will download on first run)
nltk.download('words')
from nltk.corpus import words

# =============================
# 1. Random Word Selection
# =============================
# Filter to keep only alphabetic words
vocab = [w for w in words.words() if w.isalpha()]
# Randomly sample 500 words
selected_words = random.sample(vocab, 500)

# =============================
# 2. BERT Embedding Extraction 
# =============================
# Load pre-trained BERT model and tokenizer (using local model path)
model_path = "/root/autodl-tmp/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
model.eval()  # Set to evaluation mode

def get_word_embedding(word):
    """
    Generate word embedding using BERT:
    1. Tokenize input word
    2. Extract hidden states from BERT
    3. Handle multi-token words by averaging subword embeddings
    Returns: 768-dimensional numpy array
    """
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]  # (sequence_length, hidden_size)
    if token_embeddings.shape[0] > 2:
        embedding = token_embeddings[1:-1].mean(dim=0)
    else:
        embedding = token_embeddings.mean(dim=0)
    return embedding.numpy()

embeddings = [get_word_embedding(word) for word in selected_words]
X = np.stack(embeddings)  # Shape: (500, 768)

# ========================================
# 3. Isolation Forest Implementation
# ========================================
def c(n):
    """
    Calculate expected path length adjustment term:
    c(n) = 2(ln(n-1) + γ) - 2(n-1)/n
    where γ is Euler-Mascheroni constant (~0.5772)
    """
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

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

def compute_outlier_score(avg_depth, sample_size):
    """
    Compute outlier score using formula:
    score = 2^(-avg_depth / c(sample_size))
    """
    c_val = c(sample_size)
    return 0.0 if c_val == 0 else 2 ** (-avg_depth / c_val)

# =============================
# 4. Execution & Output
# =============================
trees = build_isolation_forest(X, num_trees=8, sample_size=256)

print("Word\tOutlier Score (OS)")
for i, word in enumerate(selected_words):
    x = X[i]
    avg_depth = compute_average_depth(x, trees)
    os_w = compute_outlier_score(avg_depth, sample_size=256)
    print(f"{word}\t{os_w:.4f}")
