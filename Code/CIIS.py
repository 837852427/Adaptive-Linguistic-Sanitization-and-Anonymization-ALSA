import CCCalculator as cc
import SDCalculator as sd
from transformers import BertTokenizer, BertModel
import pandas as pd
import spacy
import numpy as np
from nltk.corpus import wordnet as wn

import torch

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


class SDCalculator:
    def __init__(self, model, tokenizer, nlp=spacy.load("en_core_web_sm")):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = nlp  
    def calculate_sd(self, sentence, max_syn=3):
        pos_tags = self.get_pos_tags_spacy(sentence)
        words, word_embeddings = self.get_words_embedding(sentence, pos_tags)
        if len(words) == 0:
            return {}
        tokens = self.tokenizer.tokenize(sentence)
        clean_tokens = [t.replace("##", "").lower() for t in tokens]
        original_token_map = []
        search_idx = 0
        for word in words:
            subword = self.tokenizer.tokenize(word)[0].replace("##", "").lower()
            found = False
            for i in range(search_idx, len(clean_tokens)):
                if clean_tokens[i] == subword:
                    original_token_map.append(i)
                    search_idx = i + 1
                    found = True
                    break
            if not found:
                original_token_map.append(-1)
        sd_scores = {}
        batch_sents = []
        meta = []          
        for i, word in enumerate(words):
            syns = self.get_valid_synonyms(word)[:max_syn]
            for synonym in syns:
                new_sent, ok = self.replace_with_alignment(sentence, original_token_map[i], synonym)
                if ok:
                    batch_sents.append(new_sent)
                    meta.append(i)
        if not batch_sents:
            return {w: 0.0 for w in words}
        inputs = self.tokenizer(batch_sents, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=512).to(self.model.device)
        with torch.no_grad():
            all_emb = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
        sd_sum = [0.0] * len(words)
        cnt    = [0]   * len(words)
        for b, i_word in enumerate(meta):
            cur_emb = all_emb[b][original_token_map[i_word]]
            diff = 1 - torch.nn.functional.cosine_similarity(
                        word_embeddings[i_word], cur_emb, dim=0).item()
            sd_sum[i_word] += diff
            cnt[i_word]    += 1
        return {words[i]: round(sd_sum[i]/cnt[i], 2) if cnt[i] else 0.0
                for i in range(len(words))}
    def get_token_positions(self, sentence, words):
        tokens = self.tokenizer.tokenize(sentence)
        clean_tokens = [t.replace("##", "").lower() for t in tokens]
        original_token_map = []
        search_idx = 0
        for word in words:
            subword = self.tokenizer.tokenize(word)[0].replace("##", "").lower()
            found = False
            for i in range(search_idx, len(clean_tokens)):
                if clean_tokens[i] == subword:
                    original_token_map.append(i)
                    search_idx = i + 1
                    found = True
                    break
            if not found:
                original_token_map.append(-1)
        return original_token_map
    def replace_with_alignment(self, sentence, token_idx, synonym):
        tokens = self.tokenizer.tokenize(sentence)
        if token_idx >= len(tokens):
            return sentence, False
        synonym_tokens = self.tokenizer.tokenize(synonym)
        new_tokens = tokens[:token_idx] + synonym_tokens + tokens[token_idx+1:]
        try:
            modified_sentence = self.tokenizer.convert_tokens_to_string(new_tokens)
            return modified_sentence, True
        except:
            return sentence, False
    def get_valid_synonyms(self, word):
        if not wn._synset_from_pos_and_offset:
            wn.ensure_loaded()
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace('_', ' ')
                if self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(candidate)) != [self.tokenizer.unk_token_id]:
                    synonyms.add(candidate)
        return list(synonyms)
    def get_aligned_embeddings(self, sentence, original_pos_map):
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                            padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].squeeze(0)
        current_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        current_clean = [t.replace("##", "").lower() for t in current_tokens]
        aligned_embeddings = []
        for pos in original_pos_map:
            if pos < len(current_clean):
                aligned_embeddings.append(embeddings[pos])
            else:
                return None
        return torch.stack(aligned_embeddings)
    def calculate_semantic_difference(self, original_embedding, modified_embedding, gamma=0.1):
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarity = cos(original_embedding, modified_embedding)
        return 1 - similarity.item()
    def get_pos_tags_spacy(self, sentence):
        doc = self.nlp(sentence)
        pos_tags = [
            (token.text, token.pos_) 
            for token in doc 
            if token.pos_ != 'PUNCT'  
        ]
        return pos_tags
    def get_words_embedding(self, sentence, pos_tags):
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
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

class CIISCalculator:
    def __init__(self, model, tokenizer, lambda_1 = 0.4, lambda_2 = 0.6, alpha = 0.8, beta = 0.5, gamma = 0.3, spacy_model = "en_core_web_sm"):
        self.model = model;self.tokenizer = tokenizer
        self.lambda_1 = lambda_1;self.lambda_2 = lambda_2
        self.alpha = alpha;self.beta = beta;self.gamma = gamma
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.cc_calculator = cc.CCCalculator(self.model, self.tokenizer, self.alpha, 
                                                self.beta, self.gamma, self.nlp)
        self.sd_calculator = sd.SDCalculator(self.model, self.tokenizer, self.nlp)
    def calculate(self, data_or_list):
        if isinstance(data_or_list, list):       
            sentences = data_or_list
        else:
            df = pd.read_csv(csv_path)
            sentences = df['sentence'].tolist()
        ciis_scores = {}
        for sentence in sentences:
            ciis_scores_sentence = self.calculate_CIIS(sentence)
            ciis_scores.update(ciis_scores_sentence)
        return ciis_scores

    def calculate_CIIS(self, sentence):
        cc_scores = self.cc_calculator.calculate_cc(sentence)
        if __name__ == "__main__":
            print("\033[92m\033[1mCC calculation completed\033[0m\033[22m")
        sd_scores = self.sd_calculator.calculate_sd(sentence)
        if __name__ == "__main__":
            print("\033[92m\033[1mSD calculation completed\033[0m\033[22m")
        ciis_scores = {}
        if __name__ == "__main__":
            print(f'sd_scores:\n{sd_scores}\ncc_scores:\n{cc_scores}\n')
            print(f'sd_scores.type: {type(sd_scores)}    sd_scores.shape: {len(sd_scores)}')
            print(f'cc_scores.type: {type(cc_scores)}    cc_scores.shape: {len(cc_scores)}')
        device = self.model.device
        words = list(cc_scores.keys())
        cc_values = torch.tensor([cc_scores[w] for w in words], device=device, 
                                dtype=torch.float16)
        sd_values = torch.tensor([sd_scores[w] for w in words], device=device, 
                                dtype=torch.float16)
        ciis_values = self.lambda_1 * cc_values + self.lambda_2 * sd_values
        return {(word, sentence): ciis_values[i].item() 
              for i, word in enumerate(words)}

if __name__ == "__main__":
    tokenizer = "bert-base-uncased"
    model = "bert-base-uncased"
    ciis_calculator = CIISCalculator(BertModel.from_pretrained(model), BertTokenizer.from_pretrained(tokenizer))
    csv_path = "data/ALSA.csv"
    ciis_scores = ciis_calculator.calculate(csv_path)
    for (word, sentence), score in ciis_scores.items():
        print(f'{word} from "{sentence}" and score is {score}')