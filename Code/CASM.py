import numpy as np
import sys
import nltk
from nltk.corpus import wordnet as wn
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

class CASMCalculator:
    def __init__(self, k=8, llm_model=None):
        self.k = k
        if llm_model is None:
            raise ValueError("llm_model can't be None")
        self.llm_model = llm_model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model.name_or_path)
        self.llm_model.eval().requires_grad_(False)
        self.gen_kwargs = dict(
            max_new_tokens=8,
            do_sample=True,
            temperature=0.9,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )
    def calculate(self, word_metrics):
        Retain_set, Replace_set, Encrypt_set, Delte_set = self.calculate_assignment_actions(word_metrics)
        words = {}
        words.update(self.action_retain(Retain_set))
        words.update(self.action_replace(Replace_set))
        words.update(self.action_encrypt(Encrypt_set))
        words.update(self.action_delete(Delte_set))
        return words
    def action_retain(self, words):
        return words
    def action_replace(self, words):
        replace_words = {}
        for (word, sentence), value in words.items():
            replace_words[(word, sentence)] = self.get_random_synonym(word)
        return replace_words
    def get_random_synonym(self, word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return random.choice(list(synonyms)) if synonyms else word
    def action_encrypt(self, words):
        def get_diff_word(word):
            prompt = f"Provide an alternative word for '{word}' that obscures its original meaning."
            try:
                input_ids = self.llm_tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm_model.device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = self.llm_model.generate(input_ids, **self.gen_kwargs)
                text = self.llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()
                result = text.split()[-1] if text.split() else ""
                if not result or result.lower() == word.lower():
                    result = "*"
                return result if result == "*" else (result.capitalize() if word[0].isupper() else result.lower())
            except Exception:
                return "*"
        return {(word, sentence): get_diff_word(word) for (word, sentence) in words}
    def action_delete(self, words):
        delete_words = {}
        for word, value in words.items():
            delete_words[word] = ''
        return delete_words
    def calculate_assignment_actions(self, word_metrics):
        word_vectors = []
        for (word, sentence), (plrs, ciis, trs) in word_metrics.items():
            word_vectors.append([plrs, ciis, trs])
        vec = torch.tensor([v for v in word_metrics.values()],
                           device="cuda", dtype=torch.float32)   
        if vec.size(0) < self.k:
            repeat = self.k - vec.size(0)
            vec_init = torch.cat([vec, vec[:repeat]], dim=0)
        else:
            vec_init = vec[: self.k].clone()
        centroids = vec_init                                              
        for _ in range(10):               
            dist = torch.cdist(vec, centroids)       
            clusters = dist.argmin(dim=1)             
            for cid in range(self.k):
                mask = clusters == cid
                if mask.any():
                    centroids[cid] = vec[mask].mean(dim=0)
        centroids = centroids.cpu().numpy()
        clusters  = clusters.cpu().numpy()
        if __name__ == "__main__":
            print(f'clusters: {clusters}\n\ncentroids: {centroids}\n')
        Retain_set = {}
        Replace_set = {}
        Encrypt_set = {}
        Delte_set = {}
        actions = {}
        for i, (word, sentence) in enumerate(word_metrics):
            cluster_id = clusters[i]
            centroid = centroids[cluster_id]
            key = (word, sentence)
            plrs, ciis, trs = word_metrics[key]
            p_base, c_base, t_base = centroid
            if (plrs < p_base and ciis >= c_base and trs < t_base) or \
               (plrs < p_base and ciis < c_base and trs >= t_base) or \
               (plrs < p_base and ciis >= c_base and trs >= t_base):
                Retain_set[key] = word
                actions[key] = "retain"
            elif plrs >= p_base and ciis < c_base and trs >= t_base:
                Replace_set[key] = word
                actions[key] = "replace"
            elif (plrs >= p_base and ciis >= c_base and trs < t_base) or \
               (plrs >= p_base and ciis >= c_base and trs >= t_base):
                Encrypt_set[key] = word
                actions[key] = "encrypt"
            elif (plrs < p_base and ciis < c_base and trs < t_base) or \
               (plrs >= p_base and ciis < c_base and trs < t_base):
                Delte_set[key] = word
                actions[key] = "delete"
            else:
                sys.exit(1)
        return Retain_set, Replace_set, Encrypt_set, Delte_set
    def check_retain(self, centroid, metrics):
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return (plrs < p_base and ciis >= c_base and trs < t_base) or \
               (plrs < p_base and ciis < c_base and trs >= t_base) or \
               (plrs < p_base and ciis >= c_base and trs >= t_base)
    def check_replace(self, centroid, metrics):
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return plrs >= p_base and ciis < c_base and trs >= t_base
    def check_encrypt(self, centroid, metrics):
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return (plrs >= p_base and ciis >= c_base and trs < t_base) or \
               (plrs >= p_base and ciis >= c_base and trs >= t_base)
    def check_delete(self, centroid, metrics):
        plrs, ciis, trs = metrics
        p_base, c_base, t_base = centroid
        return (plrs < p_base and ciis < c_base and trs < t_base) or \
               (plrs >= p_base and ciis < c_base and trs < t_base)
if __name__ == "__main__":
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
    name = "meta-llama/Llama-2-7b-chat-hf"
    llm_model = llm_model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16  
    )
    llm_model = llm_model.eval().requires_grad_(False)
    casm = CASMCalculator(k=8, llm_model=llm_model)  
    actions = casm.calculate(word_metrics)

    print("\n\033[1mCASM Actions:\033[0m")
    for (word, sent), new_word in actions.items():
        print(f"{word:<10} -> {new_word:<15} | {sent}")

