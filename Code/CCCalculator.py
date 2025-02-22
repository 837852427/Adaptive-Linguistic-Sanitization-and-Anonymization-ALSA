import spacy
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer

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
        words, word_embeddings = self.get_words_embedding(sentence, pos_tags)

        # Calculate the Dependency Matrix D_ij
        D = self.calculate_D_matrix(pos_tags)

        # Calculate pairwise distances using Mahalanobis distance
        T = np.zeros((len(pos_tags), len(pos_tags)))
        for i, v_i in enumerate(word_embeddings):
            for j, v_j in enumerate(word_embeddings):
                if i != j:
                    T[i, j] = self.calculate_mahalanobis_distance(v_i, v_j, D[i, j], self.alpha)

        # Calculate positional distances
        R = np.zeros((len(pos_tags), len(pos_tags)))
        for i in range(len(pos_tags)):
            for j in range(len(pos_tags)):
                if i != j:
                    R[i, j] = abs(i - j)

        # Final CC calculation
        cc_scores = {}
        epsilon = 1e-9  # Small value to avoid division by zero
        for i, (word, _) in enumerate(pos_tags):
            sum = 0
            for j in range(len(pos_tags)):
                if i != j:
                    sum += self.beta * R[i, j] + T[i, j] + epsilon
            cc_scores[word] = 1 / sum

        return cc_scores
    
    def calculate_D_matrix(self, pos_tags, embedding_dim=768):
        """
        修改后的D矩阵计算，返回PyTorch张量
        """
        device = self.model.device  # 获取模型所在设备
        num_words = len(pos_tags)
        D = torch.zeros((num_words, num_words, embedding_dim, embedding_dim), 
                    device=device, dtype=torch.float32)  # 使用PyTorch张量
        
        for i in range(num_words):
            for j in range(num_words):
                pos_i = pos_tags[i][1]
                pos_j = pos_tags[j][1]
                
                rho_i = 1.0 if self.is_content_word(pos_i) else self.gamma
                rho_j = 1.0 if self.is_content_word(pos_j) else self.gamma
                
                # 使用PyTorch创建单位矩阵
                D[i, j] = rho_i * rho_j * torch.eye(embedding_dim, device=device, dtype=torch.float32)
        
        return D

    def is_content_word(self, pos_tag):
        """
        Check if the word is a content word (noun, verb, adjective, adverb).
        """
        content_pos_tags = ['NN', 'VB', 'JJ', 'RB']  # Example content POS tags
        return pos_tag in content_pos_tags

    # 修改calculate_mahalanobis_distance函数
    def calculate_mahalanobis_distance(self, v_i, v_j, D_ij, alpha=0.5):
        """
        修改后的马氏距离计算, 统一使用PyTorch操作
        """
        # 确保输入为二维张量 [1, hidden_size]
        v_i = v_i.unsqueeze(0) if v_i.dim() == 1 else v_i  # 变为2D
        v_j = v_j.unsqueeze(0) if v_j.dim() == 1 else v_j
        
        # 转换D_ij为PyTorch张量（如果尚未转换）
        if isinstance(D_ij, np.ndarray):
            D_ij = torch.from_numpy(D_ij).to(device=v_i.device, dtype=torch.float32)
        
        # 计算Q_ij
        Q_ij = torch.eye(v_i.size(-1), device=v_i.device) + alpha * D_ij
        
        # 计算差值 [1, hidden_size]
        diff = v_i - v_j
        
        # 矩阵运算 (使用torch.matmul保持类型一致)
        product = torch.matmul(diff, torch.matmul(Q_ij, diff.T))  # [1,1]
        
        # 安全开平方
        return torch.sqrt(product.clamp(min=1e-9)).item()  # 返回标量数值

    
    def get_pos_tags_spacy(self, sentence):
        """
        使用spacy获取POS标签并过滤标点符号
        """
        doc = self.nlp(sentence)
        pos_tags = [
            (token.text, token.pos_) 
            for token in doc 
            if token.pos_ != 'PUNCT'  # 过滤标点符号
        ]
        return pos_tags

    def get_words_embedding(self, sentence, pos_tags):
        # Tokenize并获取embedding
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 获取embedding和原始token
        embeddings = outputs.hidden_states[-1].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        
        # 生成清洗后的比较用token列表（小写且去除##）
        clean_tokens = [
            token.replace("##", "").lower() 
            for token in tokens 
            if token not in self.tokenizer.all_special_tokens
        ]
        
        # 匹配逻辑
        matched_words = []
        matched_embeddings = []
        search_start = 0  # 维护搜索起始位置
        
        for original_word, pos in pos_tags:
            # 对原始词进行分词
            subwords = self.tokenizer.tokenize(original_word)
            if not subwords:
                continue
                
            # 获取第一个子词的清洗版本
            target = subwords[0].replace("##", "").lower()
            
            # 在token序列中顺序查找
            for idx in range(search_start, len(clean_tokens)):
                if clean_tokens[idx] == target:
                    # 保留原始词的大小写和形式
                    matched_words.append(original_word)
                    # 获取对应位置的embedding
                    matched_embeddings.append(embeddings[idx])
                    # 更新搜索起始位置避免重复匹配
                    search_start = idx + 1
                    break
        
        return matched_words, torch.stack(matched_embeddings) if matched_embeddings else torch.tensor([])

def input_sentence(csv_path):
    """
    Read the input sentence from a CSV file.
    """
    df = pd.read_csv(csv_path)
    sentences = df['sentence'].tolist()
    
    return sentences        
    
if __name__ == "__main__":
    # Example usage
    
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cc_calculator = CCCalculator(model, tokenizer)
    
    sentence = input_sentence("D:/论文/ALSA/test.csv")
    
    cc_scores = cc_calculator.calculate_cc(sentence[0])
    print("Contextual Coherence scores:\n", cc_scores)
    print(f'cc_scores.shape: {len(cc_scores)}')