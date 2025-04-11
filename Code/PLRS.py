from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
import spacy
from torch.cuda.amp import autocast

class IsolationTree:
    def __init__(self, height_limit, current_height=0):
        self.height_limit = height_limit
        self.current_height = current_height
        self.split_att = None   
        self.split_val = None   
        self.left = None        
        self.right = None       
        self.size = None        
    def fit(self, X):
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
        if self.left is None and self.right is None:
            return self.current_height + c(self.size)
        if x[self.split_att] < self.split_val:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)
def build_isolation_forest(X, num_trees=8, sample_size=256):
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
    depths = [tree.path_length(x) for tree in trees]
    return np.mean(depths)
def c(n):
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
def compute_outlier_score(avg_depth, sample_size):
    c_val = c(sample_size)
    return 0.0 if c_val == 0 else 2 ** (-avg_depth / c_val)
def get_word_embedding(self, word):
    inputs = self.tokenizer(
        word, 
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(self.device)
    with torch.no_grad():
        outputs = self.model(**inputs)
    token_embeddings = outputs.last_hidden_state[0]  
    if token_embeddings.shape[0] > 2:
        embedding = token_embeddings[1:-1].mean(dim=0)
    else:
        embedding = token_embeddings.mean(dim=0)
    return embedding.cpu().numpy()
class PLRSCalculator:
    def __init__(self, model, tokenizer, device, data_path='', spacy_model="en_core_web_sm"):
        self.data_path = data_path
        self.device = model.device
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.eval()  
        self.nlp = spacy.load(spacy_model)  
    def input_sentence(self):
        df = pd.read_csv(self.data_path)
        sentences = df['sentence'].tolist()
        return sentences
    def calculate(self, sentences=None):
        if sentences is None: 
            sentences = self.input_sentence()
        PLRS_scores = {}
        for sentence in sentences:
            doc = self.nlp(sentence)
            pos_tags = [
                (token.text, token.pos_) 
                for token in doc 
                if token.pos_ != 'PUNCT' 
            ]
            merged_words = []
            for word, _ in pos_tags:
                merged_words.append(word)
            if not merged_words:
                return {}
            inputs = self.tokenizer(sentence,return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=512).to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs,output_hidden_states=True).hidden_states[-1][0] 
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            clean  = [t.replace("##", "").lower() for t in tokens]
            emb_list = []
            search = 0
            for w in merged_words:
                sub = self.tokenizer.tokenize(w)[0].replace("##", "").lower()
                for idx in range(search, len(clean)):
                    if clean[idx] == sub:
                        emb_list.append(hidden[idx])
                        search = idx + 1
                        break
            if not emb_list:
                return {}
            X = torch.stack(emb_list).cpu().numpy()          
            trees = build_isolation_forest(X, num_trees=8, sample_size=256)
            plrs_scores = {}
            for i, w in enumerate(merged_words):
                avg_depth = compute_average_depth(X[i], trees)
                plrs_scores[(w, sentence)] = compute_outlier_score(avg_depth, 256)
            PLRS_scores.update(plrs_scores)
        return PLRS_scores
if __name__ == "__main__":
    model = "bert-base-uncased"
    plrs = PLRSCalculator(model=BertModel.from_pretrained(model),
                            tokenizer=BertTokenizer.from_pretrained(model),
                          data_path='data/ALSA.csv')
    ans = plrs.calculate()

    for (word, sentence), score in ans.items():
        print(f'{word}, {sentence}: {score}')