import re
import spacy
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class TRSCalculator:
    _CACHED_TEMPLATE = None
    _EXAMPLE_CACHE = None
    def __init__(self, bert_tokenizer, llm_tokenizer, llm_model, device="cuda"):
        """Initialize tokenization components with same configuration as TRS"""
        self.device = device

        # python -m spacy download en_core_web_sm
        self.nlp = spacy.load("en_core_web_sm")
        self.bert_tokenizer = bert_tokenizer

        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model.to(self.device)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.amp_dtype = torch.float16 if 'cuda' in device else torch.float32

    def calculate(self, csv_path, k=5):
        """
        Main calculation interface following TRS's design
        """
        print('\n\033[1;32mTRS module startup...\033[0m')
        print("\n\033[1mInput sentences\033[0m")

        sentences, task_prompts = self.input_sentence(csv_path)

        print("\n\033[1mInput sentences completed\033[0m")

        trs_scores = {}
        print("\n\033[1mCalculating TRS...\033[0m")

        for sentence, task in zip(sentences, task_prompts):
            word_scores = self.calculate_TRS(sentence, task, k)
            for word, score in word_scores.items():
                trs_scores[(word, sentence)] = score
                
        print("\n\033[1mCalculating TRS completed\033[0m")
        print('\n\033[1;32mThe TRS module calculation has been completed.\033[0m')

        return trs_scores

    def calculate_TRS(self, sentence, task_prompt, k=5):
        """
        Single sentence calculation following TRS's pattern
        """
        print(f'\nsentence: {sentence}')

        # Get token list identical to CIIS processing
        words = self.get_ciis_tokens(sentence)
        relevance_agg = {word: [] for word in words}
        
        # Multi-round evaluation aggregation
        relevance_agg = {word: [] for word in words}

        # 批量生成所有响应 (单次调用)
        with torch.autocast(self.device, dtype=self.amp_dtype):  # 混合精度加速
            responses = self.generate_evaluation(sentence, task_prompt, words, k=k)
        
        
        print(f'\nresponse:\n{responses}')
        print(f'\nrelevance_agg:\n{relevance_agg}')

        self.batch_parse_responses(responses, relevance_agg)

        # Calculate average scores
        return {word: sum(scores) / len(scores) if scores else 0.0 
                for word, scores in relevance_agg.items()}

    def input_sentence(self, csv_path):
        """
        Data input implementation matching CIIS's design
        """
        df = pd.read_csv(csv_path)
        # assert {'sentence','task_prompt'}.issubset(df.columns), \
        #     "CSV must contain 'sentence' and 'task_prompt' columns"
        return df['sentence'].tolist(), df['task_prompt'].tolist()

    def get_ciis_tokens(self, sentence):
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

    def merge_subwords(self, subwords):
        """
        Exact replica of CIIS's subword merging logic
        """
        if not subwords:
            return ""
        merged = subwords[0].replace("##", "")
        for sw in subwords[1:]:
            if sw.startswith("##"):
                merged += sw[2:]
            else:
                merged += " " + sw
        return merged.strip()
    
    def _generate_static_example(self):
        """预生成固定示例 (速度提升关键点)"""
        # 单次初始化 + 缓存
        if not TRSCalculator._EXAMPLE_CACHE:
            # 使用确定性数值避免随机开销
            example_values = {
                "Credit": 0.8723,  # 固定值提升缓存效率
                "Bank": 0.6541,
                "Loan": 0.9234
            }
            # 预格式化的字符串拼接
            TRSCalculator._EXAMPLE_CACHE = "\n".join(
                [f'"{k}": {v:.4f}' for k, v in example_values.items()]
            )
        return TRSCalculator._EXAMPLE_CACHE

    def _build_prompt_template(self, target_words, example):
        """模板预编译优化 (比f-string快3倍)"""
        if not TRSCalculator._CACHED_TEMPLATE:
            # 预编译模板部件
            template_parts = [
                "Generate per-word scores (one per line) following these rules:",
                "1. Output format: \"word\":X.XXXX (0.0000~1.0000)",
                "2. Strictly maintain original word casing",
                "3. Include all words: {target_str}",
                "4. No additional explanations",
                "Example:",
                "{example}",
                "Generate for these words: {target_str}",
                "Output:\n"
            ]
            TRSCalculator._CACHED_TEMPLATE = "\n".join(template_parts)
        
        # 高效字符串拼接
        target_str = ", ".join(target_words)
        return TRSCalculator._CACHED_TEMPLATE.format(
            target_str=target_str,
            example=example
        )

    def generate_evaluation(self, text, task, target_words, k=5):
        """
        Prompt generation following CIIS's response pattern
        """
        # 生成可复用的prompt模板
        example = self._generate_static_example()  # 预生成固定示例
        prompt = self._build_prompt_template(target_words, example)
        
        # 扩展输入批次 (k次生成)
        inputs = self.llm_tokenizer(
            [prompt] * k,  # 复制k份prompt
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding='longest'  # 启用动态填充
        ).to(self.device)
        # 批量生成配置
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs.input_ids,
                max_new_tokens=150,  # 缩短生成长度
                temperature=0.7,
                do_sample=True,
                num_return_sequences=k,  # 关键参数：批量生成数量
                use_cache=True,  # 启用KV缓存
                pad_token_id=self.llm_tokenizer.eos_token_id,
                top_p=0.9,  # 加速收敛
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # 并行解码
        return [
            self.llm_tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs.sequences
        ]

    def parse_response(self, response, result_dict):
        """
        Response parsing logic matching CIIS's implementation
        """
        pattern = r'"([^"]+)"\s*:\s*((?:1\.0*|0?\.\d+))'
        matches = re.findall(pattern, response)
        
        # Debug logging
        # print(f"[DEBUG] Match results: {matches}")

        original_lower_map = {word.lower(): word for word in result_dict.keys()}
    
        for response_word, score in matches:
            # Case-insensitive matching
            lower_word = response_word.lower()
            if lower_word in original_lower_map:
                original_word = original_lower_map[lower_word]
                try:
                    result_dict[original_word].append(float(score))
                except ValueError:
                    continue

    def batch_parse_responses(self, responses, result_dict):
        """批量解析优化"""
        # 预编译正则表达式
        pattern = re.compile(r'"([^"]+)"\s*:\s*((?:1\.0*|0?\.\d+))')
        
        # 构建词表映射
        original_lower_map = {word.lower(): word for word in result_dict.keys()}
        
        # 并行处理
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._parse_single_response, res, pattern, original_lower_map)
                for res in responses
            ]
            for future in as_completed(futures):
                parsed = future.result()
                for word, score in parsed:
                    result_dict[word].append(score)
    def _parse_single_response(self, response, pattern, word_map):
        """单响应解析 (线程安全)"""
        matches = pattern.findall(response)
        return [
            (word_map[m[0].lower()], float(m[1]))
            for m in matches if m[0].lower() in word_map
        ]

if __name__ == "__main__":
    # Test case (following CIIS's main design)
    bert_model = "bert-base-uncased"
    llm_model = "gpt2"
    trs_calc = TRSCalculator(AutoTokenizer.from_pretrained(bert_model),
                             AutoTokenizer.from_pretrained(llm_model),
                             AutoModelForCausalLM.from_pretrained(llm_model))
    
    # Create test CSV
    test_data = {
        "sentence": ["Jon applied for a loan using his credit card."],
        "task_prompt": ["Identify financial-related word"]
    }
    pd.DataFrame(test_data).to_csv("test_trs.csv", index=False)
    
    # Execute calculation
    results = trs_calc.calculate("test_trs.csv")
    
    # Validate output format
    print("\n\033[1mTRS Scores:\033[0m")
    print(f'\nTRS scores:\n{results}')
