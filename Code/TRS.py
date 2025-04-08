import re
import spacy
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

class TRSCalculator:
    _CACHED_TEMPLATE = None
    _EXAMPLE_CACHE = None
    
    def __init__(self, bert_tokenizer, llm_tokenizer, llm_model, device="cuda"):
        """Initialize tokenization components with same configuration as TRS"""
        self.device = torch.device(device) if isinstance(device, str) else device
        is_cuda = self.device.type == "cuda"

        # Load spaCy model (ensure it is installed: python -m spacy download en_core_web_sm)
        self.nlp = spacy.load("en_core_web_sm")
        self.bert_tokenizer = bert_tokenizer

        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model.to(self.device)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.amp_dtype = torch.float16 if is_cuda else torch.float32
        # 匹配模式：匹配范围在 0.0~1.0，且仅一位小数
        self._SCORE_RE = re.compile(r'\b([01]\.\d)\b')

    def calculate(self, data_or_path, k=5):
        """Main calculation interface following TRS's design"""
        print('\n\033[1;32mTRS module startup...\033[0m')
        if isinstance(data_or_path, list):
            sentences, task_prompts = zip(*data_or_path)
        else:
            sentences, task_prompts = self.input_sentence(data_or_path)

        trs_scores = {}
        for sentence, task in zip(sentences, task_prompts):
            word_scores = self.calculate_TRS(sentence, task, k)
            for word, score in word_scores.items():
                trs_scores[(word, sentence)] = score
        return trs_scores

    def calculate_TRS(self, sentence, task_prompt, k=1):
        """Single sentence calculation following TRS's pattern"""
        print(f'\nsentence: {sentence}')
        words = self.get_ciis_tokens(sentence)
        relevance_agg = {word: [] for word in words}

        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
            responses = self.generate_evaluation(sentence, task_prompt, words, k=k)
        
        self.batch_parse_responses(responses, relevance_agg, words)
        return {word: sum(scores) / len(scores) if scores else 0.0 for word, scores in relevance_agg.items()}

    def input_sentence(self, csv_path):
        """Data input implementation matching CIIS's design"""
        df = pd.read_csv(csv_path)
        return df['sentence'].tolist(), df['task_prompt'].tolist()

    def get_ciis_tokens(self, sentence):
        """Implement token processing logic similar to CIIS"""
        doc = self.nlp(sentence)
        tokens = [token.text for token in doc if token.pos_ != 'PUNCT']
        return tokens

    def _generate_static_example(self):
        """Pre-generate a static example to boost performance."""
        if not TRSCalculator._EXAMPLE_CACHE:
            example_scores = [0.8, 0.6, 0.9] 
            TRSCalculator._EXAMPLE_CACHE = " ".join(f"{score:.1f}" for score in example_scores)
        return TRSCalculator._EXAMPLE_CACHE

    def _build_prompt_template(self, target_words, example):
        """Build a concise prompt template to request scores without word labels."""
        if not TRSCalculator._CACHED_TEMPLATE:
            template_parts = [
                "Provide TRS scores for each word in order.",
                "Each score must be a number between 0.0 and 1.0 with one decimal place.",
                "Output only the scores separated by spaces.",
                "Words: {target_str}",
                "Example: {example}",
                "Output:"
            ]
            TRSCalculator._CACHED_TEMPLATE = "\n".join(template_parts)
        target_str = ", ".join(target_words)
        return TRSCalculator._CACHED_TEMPLATE.format(target_str=target_str, example=example)

    def generate_evaluation(self, text, task, target_words, k=5):
        """Generate k responses from the LLM according to the prompt template."""
        example = self._generate_static_example()
        prompt = self._build_prompt_template(target_words, example)
        prompt_list = [prompt] * k
        inputs = self.llm_tokenizer(
            prompt_list,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding='longest'
        ).to(self.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs.input_ids,
                max_new_tokens=32,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=k,
                use_cache=True,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True
            )
        return [self.llm_tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]

    def batch_parse_responses(self, responses, result_dict, word_list):
        """Parallel parse responses, mapping scores to each word."""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._parse_single_response, res, word_list) for res in responses]
            for future in as_completed(futures):
                parsed = future.result()
                for word, score in parsed:
                    result_dict[word].append(score)

    def _parse_single_response(self, response, word_list):
        """Parse a single response. Map each word to its corresponding score; if no score for a word, use 0.5."""
        tokens = self._SCORE_RE.findall(response)
        results = []
        for i, word in enumerate(word_list):
            if i < len(tokens):
                score = float(tokens[i])
            else:
                score = 0.5
            results.append((word, score))
        return results

if __name__ == "__main__":
    bert_model = "bert-base-uncased"
    llm_model = "gpt2"
    trs_calc = TRSCalculator(
        AutoTokenizer.from_pretrained(bert_model),
        AutoTokenizer.from_pretrained(llm_model),
        AutoModelForCausalLM.from_pretrained(llm_model)
    )
    test_data = {
        "sentence": ["Jon applied for a loan using his credit card."],
        "task_prompt": ["Identify financial-related word"]
    }
    pd.DataFrame(test_data).to_csv("test_trs.csv", index=False)
    results = trs_calc.calculate("test_trs.csv")
    print("\nTRS Scores:")
    print(results)
