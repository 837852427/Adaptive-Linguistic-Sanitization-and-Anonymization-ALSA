import spacy
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
import math

class CCCalculator:
    def __init__(self, model, tokenizer, alpha=0.5, beta=0.5, gamma=0.3, nlp=spacy.load("en_core_web_sm")):
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = gamma 
        self.nlp = nlp  
        self.alpha = alpha
        self.beta = beta

    def calculate_cc(self, sentence):
        doc = self.nlp(sentence)
        pos_tags = [
            (token.text, token.pos_) 
            for token in doc 
            if token.pos_ != 'PUNCT' 
        ]
        words, word_emb = self.get_words_embedding(sentence, pos_tags) 
        if word_emb.numel() == 0:
            return {w: 0.0 for w, _ in pos_tags}
        N, dim = word_emb.shape
        device = word_emb.device
        dtype  = word_emb.dtype
        device = self.model.device 
        dtype = next(self.model.parameters()).dtype 
        num_words = len(pos_tags)
        Dmat = torch.zeros((num_words, num_words, dim, dim), 
                device=device, dtype=dtype)
        for i in range(num_words):
            for j in range(num_words):
                pos_i = pos_tags[i][1]
                pos_j = pos_tags[j][1]
                rho_i = 1.0 if self.is_content_word(pos_i) else self.gamma
                rho_j = 1.0 if self.is_content_word(pos_j) else self.gamma
                Dmat[i, j] = rho_i * rho_j * torch.eye(dim, device=device, dtype=dtype)    
        diff = word_emb.unsqueeze(1) - word_emb.unsqueeze(0)             
        Q    = torch.eye(dim, device=device, dtype=dtype) + self.alpha * Dmat
        diff_col = diff.unsqueeze(-1)                   
        mah_sq   = torch.matmul(torch.matmul(diff.unsqueeze(-2), Q), diff_col)  
        mah      = mah_sq.squeeze(-1).squeeze(-1).clamp(min=1e-9).sqrt()   
        idx = torch.arange(N, device=device, dtype=dtype)
        R   = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        sum_mat = self.beta * R + mah + 1e-9
        inv_sum = 1.0 / sum_mat.sum(dim=1)
        return {words[i]: float(inv_sum[i].cpu()) for i in range(N)}
    def calculate_D_matrix(self, pos_tags, dim=768):
        device = self.model.device 
        dtype = next(self.model.parameters()).dtype 
        num_words = len(pos_tags)
        Dmat = torch.zeros((num_words, num_words, dim, dim), 
                device=device, dtype=dtype)
        for i in range(num_words):
            for j in range(num_words):
                pos_i = pos_tags[i][1]
                pos_j = pos_tags[j][1]
                rho_i = 1.0 if self.is_content_word(pos_i) else self.gamma
                rho_j = 1.0 if self.is_content_word(pos_j) else self.gamma
                Dmat[i, j] = rho_i * rho_j * torch.eye(dim, device=device, dtype=dtype)
        return Dmat
    def is_content_word(self, pos_tag):
        content_pos_tags = ['NN', 'VB', 'JJ', 'RB'] 
        return pos_tag in content_pos_tags
    def calculate_mahalanobis_distance(self, v_i, v_j, D_ij, alpha=0.5):
        device = v_i.device
        dtype = v_i.dtype
        if isinstance(D_ij, np.ndarray):
            D_ij = torch.from_numpy(D_ij).to(device=device, dtype=dtype) 
        else:
            D_ij = D_ij.to(device=device, dtype=dtype)
        v_i = v_i.unsqueeze(0) if v_i.dim() == 1 else v_i
        v_j = v_j.unsqueeze(0) if v_j.dim() == 1 else v_j
        Q_ij = torch.eye(v_i.size(-1), device=device, dtype=dtype) + alpha * D_ij
        diff = v_i - v_j
        if diff.dtype != Q_ij.dtype:
            diff = diff.to(torch.float32)
            Q_ij = Q_ij.to(torch.float32)
        product = torch.matmul(diff, torch.matmul(Q_ij, diff.T))
        return torch.sqrt(product.clamp(min=1e-9)).item()
    def get_pos_tags_spacy(self, sentence):
        doc = self.nlp(sentence)
        pos_tags = [
            (token.text, token.pos_) 
            for token in doc 
            if token.pos_ != 'PUNCT' 
        ]
        return pos_tags
    def get_words_embedding(self, sentence, pos_tags):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, 
                                truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, 
            output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        clean_tokens = [
            token.replace("##", "").lower() 
            for token in tokens 
            if token not in self.tokenizer.all_special_tokens
        ]
        matched_words = []
        matched_embeddings = []
        search_start = 0 
        for original_word, pos in pos_tags:
            subwords = self.tokenizer.tokenize(original_word)
            if not subwords:
                continue
            target = subwords[0].replace("##", "").lower()
            for idx in range(search_start, len(clean_tokens)):
                if clean_tokens[idx] == target:
                    matched_words.append(original_word)
                    matched_embeddings.append(embeddings[idx])
                    search_start = idx + 1
                    break
        return matched_words, torch.stack(matched_embeddings) if matched_embeddings else torch.tensor([])
def input_sentence(csv_path):
    df = pd.read_csv(csv_path)
    sentences = df['sentence'].tolist()
    
    return sentences        
if __name__ == "__main__":
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cc_calculator = CCCalculator(model, tokenizer)
    sentence = input_sentence("data/ALSA.csv")
    cc_scores = cc_calculator.calculate_cc(sentence[0])
    print("CC scores:\n", cc_scores)
    print(f'cc_scores.shape: {len(cc_scores)}')
