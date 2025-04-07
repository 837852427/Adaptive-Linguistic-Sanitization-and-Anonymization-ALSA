import re
import spacy
import pandas as pd
import inflect
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

class TRSCalculator:
    def __init__(self, bert_tokenizer, llm_tokenizer, llm_model, device="cuda"):
        """Initialize tokenization components with same configuration as TRS"""
        self.device = device

        # python -m spacy download en_core_web_sm
        self.nlp = spacy.load("en_core_web_sm")
        self.bert_tokenizer = bert_tokenizer
        
        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model.to(self.device)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

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
        
        # Multi-round evaluation aggregation
        relevance_agg = {word: [] for word in words}
        for _ in range(k):
            print(f'\033[1mThe {inflect.engine().ordinal(_ + 1)} iteration\033[0m')
            response = self.generate_evaluation(sentence, task_prompt, words)
            self.parse_response(response, relevance_agg)
        
        print(f'\nresponse:\n{response}')
        print(f'\nrelevance_agg:\n{relevance_agg}')

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

    def generate_evaluation(self, text, task, target_words):
        """
        Prompt generation following CIIS's response pattern
        """
        example = "\n".join([f'"{word}": {round(random.uniform(0, 1), 4):.4f}' 
                       for word in ["Credit", "Bank", "Loan"]])
    
        prompt = f"""Generate per-word scores (one per line) following these rules:
    1. Output format: "word":X.XXXX (0.0000~1.0000)
    2. Strictly maintain original word casing
    3. Include all words: {", ".join(target_words)}
    4. No additional explanations
    Example:
    {example}
    Generate for these words: {", ".join(target_words)}
    Output:\n"""
        
        inputs = self.llm_tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        outputs = self.llm_model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

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
