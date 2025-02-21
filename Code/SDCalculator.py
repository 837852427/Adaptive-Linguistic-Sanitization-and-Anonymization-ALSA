import torch
from sklearn.metrics.pairwise import rbf_kernel
import nltk
import spacy
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

class SDCalculator:
    def __init__(self, model, tokenizer, spacy_model="en_core_web_sm"):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = spacy.load(spacy_model)  # Load spaCy model for POS tagging and dependency parsing

    def calculate_sd(self, sentence):
        pos_tags = self.get_pos_tags_spacy(sentence)
        words, word_embeddings = self.get_words_embedding(sentence, pos_tags)
        
        if len(words) == 0:
            return {}

        # 获取原始token位置映射
        original_token_map = self.get_token_positions(sentence, words)
        
        sd_scores = {}
        
        for i, word in enumerate(words):
            candidate_synonyms = self.get_valid_synonyms(word)  # 过滤有效同义词
            semantic_diff_sum = 0.0
            
            for synonym in candidate_synonyms:
                # 保持分词对齐的替换方式
                modified_sentence, success = self.replace_with_alignment(
                    sentence, 
                    original_token_map[i], 
                    synonym
                )
                if not success:
                    continue
                    
                # 获取对齐后的嵌入向量
                modified_embeddings = self.get_aligned_embeddings(
                    modified_sentence, 
                    original_token_map
                )
                
                if modified_embeddings is None:
                    continue
                    
                # 计算语义差异
                semantic_diff = self.calculate_semantic_difference(
                    word_embeddings[i].unsqueeze(0), 
                    modified_embeddings[i].unsqueeze(0)
                )
                semantic_diff_sum += semantic_diff
            
            # 计算结果
            if len(candidate_synonyms) > 0:
                sd = semantic_diff_sum / len(candidate_synonyms)
                sd_scores[word] = round(float(sd), 2)
            else:
                sd_scores[word] = 0.0
        
        return sd_scores

    # 新增辅助方法
    def get_token_positions(self, sentence, target_words):
        """获取每个目标词在token序列中的位置索引"""
        tokens = self.tokenizer.tokenize(sentence)
        clean_tokens = [t.replace("##", "").lower() for t in tokens]
        
        position_map = []
        search_idx = 0
        
        for word in target_words:
            subword = self.tokenizer.tokenize(word)[0].replace("##", "").lower()
            found = False
            for i in range(search_idx, len(clean_tokens)):
                if clean_tokens[i] == subword:
                    position_map.append(i)
                    search_idx = i + 1
                    found = True
                    break
            if not found:
                position_map.append(-1)
        
        return position_map

    def replace_with_alignment(self, sentence, token_idx, synonym):
        """保持分词对齐的替换方法"""
        tokens = self.tokenizer.tokenize(sentence)
        if token_idx >= len(tokens):
            return sentence, False
        
        # 生成替换后的token序列
        synonym_tokens = self.tokenizer.tokenize(synonym)
        new_tokens = tokens[:token_idx] + synonym_tokens + tokens[token_idx+1:]
        
        try:
            # 重建有效句子
            modified_sentence = self.tokenizer.convert_tokens_to_string(new_tokens)
            return modified_sentence, True
        except:
            return sentence, False

    def get_valid_synonyms(self, word):
        """获取模型词汇表中有效的同义词"""
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace('_', ' ')
                # 验证是否在tokenizer词汇表中
                if self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(candidate)) != [self.tokenizer.unk_token_id]:
                    synonyms.add(candidate)
        return list(synonyms)

    def get_aligned_embeddings(self, sentence, original_pos_map):
        """获取与原始位置对齐的嵌入向量"""
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        embeddings = outputs.hidden_states[-1].squeeze(0)
        
        # 获取当前句子的token位置映射
        current_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        current_clean = [t.replace("##", "").lower() for t in current_tokens]
        
        aligned_embeddings = []
        for pos in original_pos_map:
            if pos < len(current_clean):
                aligned_embeddings.append(embeddings[pos])
            else:
                return None  # 位置不匹配
        
        return torch.stack(aligned_embeddings)

    # 修改后的MMD计算
    def calculate_semantic_difference(self, original_embedding, modified_embedding, gamma=0.1):
        """
        使用余弦相似度计算语义差异
        """
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarity = cos(original_embedding, modified_embedding)
        return 1 - similarity.item()  # 返回差异度
    
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


if __name__ == "__main__":
    # Example usage of the SDCalculator class
    from transformers import BertModel, BertTokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sd_calculator = SDCalculator(model, tokenizer)

    sentence = "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
    sd_scores = sd_calculator.calculate_sd(sentence)
    print(sd_scores)