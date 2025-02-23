import PLRS
import CIIS
import TRS
import CASM
import pandas as pd
from transformers import BertModel, BertTokenizer
from collections import defaultdict

class ALSA:
    """ALSA Framework: Comprehensive Text Analysis System"""
    def __init__(self, model, tokenizer, csv_path, llm_model="gpt2", k=8, bert_model="bert-base-uncased"):
        """
        Initialize ALSA components
        :param model: Pretrained language model
        :param tokenizer: Text tokenizer
        :param csv_path: Input data path
        :param llm_model: Large Language Model name
        """
        self.model = model
        self.tokenizer = tokenizer
        self.llm_model = llm_model
        self.csv_path = csv_path
        self.k = k
        self.bert_model = bert_model
        self.PLRS = PLRS.PLRSCalculator()
        self.CIIS = CIIS.CIISCalculator(model, tokenizer)
        self.TRS = TRS.TRSCalculator(bert_model=self.bert_model, llm_model=llm_model)
        self.CASM = CASM.CASMCalculator(k=self.k, llm_model=self.llm_model) 
    
    def calculate(self):
        """Execute complete ALSA analysis pipeline"""
        triple_metrics = self.calculate_part1()
        replacement_dict = self.calculate_part2(triple_metrics)
        self.calculate_part3(replacement_dict, self.csv_path)

    def calculate_part1(self):
        """Execute first stage metrics calculation"""
        print('\033[1;32mStarting Part 1 Calculations...\033[0m')

        # PLRS Calculation
        PLRS_metrics = {}
        PLRS_metrics = self.PLRS.calculate()
        print(f'\nPLRS Results:\n{PLRS_metrics}')
        
        # CIIS Calculation
        CIIS_metrics = {}
        CIIS_metrics = self.CIIS.calculate(self.csv_path)
        print(f'\nCIIS Results:\n{CIIS_metrics}')
        
        # TRS Calculation
        TRS_metrics = {}
        TRS_metrics = self.TRS.calculate(self.csv_path)
        print(f'\nTRS Results:\n{TRS_metrics}')
        
        
        # CASM Aggregation
        print('\n\033[1mGenerating CASM Metrics\033[0m')
        CASM_metrics = {}
        
        # Aggregate metrics
        for key, score in PLRS_metrics.items():
            CASM_metrics[key] = [score, 0.0, 0.0]
        
        for key, score in CIIS_metrics.items():
            if key in CASM_metrics:
                CASM_metrics[key][1] = score
            else:  # This may need adjustment
                CASM_metrics[key] = [0.0, score, 0.0]
        
        for key, score in TRS_metrics.items():
            if key in CASM_metrics:
                CASM_metrics[key][2] = score
            else:
                CASM_metrics[key] = [0.0, 0.0, score]
        
        print(f'\nCASM Aggregation:\n{CASM_metrics}')
        print('\n\033[1mCASM Aggregation Completed\033[0m')
        print('\n\033[1;32mPart 1 Finished\033[0m')

        return CASM_metrics
    
    def calculate_part2(self, triple_metrics):
        """Execute second stage action determination"""
        print('\n\033[1;32mStarting Part 2 Calculations...\033[0m')

        words_metrics = self.CASM.calculate(triple_metrics)
        print(f'\nwords_metrics:\n{words_metrics}')

        print('\n\033[1;32mPart 2 Completed\033[0m')
        return words_metrics

    def calculate_part3(self, part2_output, csv_path):
        """
        Ultimate Replacement Logic: Precise Replacement Based on (word, sentence)
        :param part2_output: Dictionary structure {(word, sentence): r_word}
        :param csv_path: Path to the original CSV file
        """
        print('\n\033[1;32mStarting Part 3...\033[0m')
        
        # Read the original data
        df = pd.read_csv(csv_path)
        
        # Build a three-level nested dictionary structure (optimized for query speed)
        sentence_map = defaultdict(lambda: defaultdict(dict))
        
        for (word, sent), r_word in part2_output.items():
            # Generate sentence hash fingerprints (to address performance issues with long texts as keys)
            sent_hash = hash(sent)
            # Store the original word form (preserving case sensitivity)
            sentence_map[sent_hash][word.lower()][word] = r_word
        
        # Process each sentence
        for index in df.index:
            original_sent = df.at[index, 'sentence']
            sent_hash = hash(original_sent)
            
            # Get the replacement rules for the current sentence
            if sent_hash not in sentence_map:
                continue
                
            word_dict = sentence_map[sent_hash]
            tokens = original_sent.split()  # Use word segmentation to ensure order
            
            # Generate replacement index mapping
            replace_map = {}
            for i, token in enumerate(tokens):
                lower_token = token.lower()
                if lower_token in word_dict:
                    # Exact match the original word's case form
                    if token in word_dict[lower_token]:
                        replace_map[i] = word_dict[lower_token][token]
                    # Compatible with variants such as capitalized first letters
                    elif token.title() in word_dict[lower_token]:
                        replace_map[i] = word_dict[lower_token][token.title()]
            
            # Perform index replacement
            for i, r_word in replace_map.items():
                tokens[i] = r_word
            
            df.at[index, 'sentence'] = ' '.join(tokens)
        
        # Save the result
        df.to_csv('output.csv', index=False)
        print("\033[1;32mPart3 completed. Saved to output.csv\033[0m")

if __name__ == "__main__":
    # Initialize components
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Execute ALSA
    alsa = ALSA(
        model, 
        tokenizer, 
        csv_path="data/ALSA.csv",
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    alsa.calculate()

    print("\n\033[1;32mALL COMPLETED\033[0m")

