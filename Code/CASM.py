import numpy as np
import sys
import nltk
from nltk.corpus import wordnet as wn
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Download the WordNet corpus
# nltk.download('wordnet')

# Note: Potential memory leak risk when using k-means on Windows
import warnings
warnings.filterwarnings('ignore')

class CASMCalculator:
    def __init__(self, k=8, llm_model=None):
        """
        Initialize CASM calculator
        :param k: Number of clusters for k-means
        """
        self.k = k
        # 如果外部传入已加载的 LLM (HF AutoModelForCausalLM)，就复用；否则懒加载
        if llm_model is None:
            raise ValueError("llm_model 不能为 None，需传入已加载好的 LLM 实例")
        self.llm_model = llm_model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model.name_or_path)
        self.llm_model.eval().requires_grad_(False)

        # 统一生成参数
        self.gen_kwargs = dict(
            max_new_tokens=8,
            do_sample=True,
            temperature=0.9,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )

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
            prompt = f"Provide an alternative word for '{word}' that obscures its original meaning."
            try:
                input_ids = self.llm_tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm_model.device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = self.llm_model.generate(input_ids, **self.gen_kwargs)
                text = self.llm_tokenizer.decode(output[0], skip_special_tokens=True).strip()
                # 从生成的文本中取最后一个单词作为替换
                result = text.split()[-1] if text.split() else ""
                if not result or result.lower() == word.lower():
                    result = f"{word}_alt"
                return result.capitalize() if word[0].isupper() else result.lower()
            except Exception:
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
        vec = torch.tensor([v for v in word_metrics.values()],
                           device="cuda", dtype=torch.float32)          # [N,3]

        # 初始化中心 = 前 k 个样本；若词数 < k，重复取
        if vec.size(0) < self.k:
            repeat = self.k - vec.size(0)
            vec_init = torch.cat([vec, vec[:repeat]], dim=0)
        else:
            vec_init = vec[: self.k].clone()

        centroids = vec_init                                                    # [k,3]
        for _ in range(10):                # 10 次迭代足够收敛
            dist = torch.cdist(vec, centroids)          # [N,k]
            clusters = dist.argmin(dim=1)               # [N]
            for cid in range(self.k):
                mask = clusters == cid
                if mask.any():
                    centroids[cid] = vec[mask].mean(dim=0)
        centroids = centroids.cpu().numpy()
        clusters  = clusters.cpu().numpy()

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
        if __name__ == "__main__":
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
    """
    测试用例说明  
    - Retain  : 低 PLRS + 高 CIIS + 低 TRS  →  (0.10, 0.80, 0.20)
    - Replace : 高 PLRS + 低 CIIS + 高 TRS  →  (0.85, 0.20, 0.90)
    - Encrypt : 高 PLRS + 高 CIIS + 低 TRS  →  (0.90, 0.85, 0.10)
    - Delete  : 低 PLRS + 低 CIIS + 低 TRS  →  (0.05, 0.10, 0.05)
    """
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
        device_map="auto",  # 自动分配设备（支持多GPU）
        torch_dtype=torch.float16  # 使用半精度
    )
    llm_model = llm_model.eval().requires_grad_(False)
    casm = CASMCalculator(k=4, llm_model=llm_model)  # 替换为已加载的模型实例亦可
    actions = casm.calculate(word_metrics)

    print("\n\033[1mCASM Actions:\033[0m")
    for (word, sent), new_word in actions.items():
        print(f"{word:<10} -> {new_word:<15} | {sent}")

