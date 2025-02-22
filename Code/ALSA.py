import PLRS
import CIIS
import TRS
import CASM
import pandas as pd
import re
import pandas as pd
from transformers import BertModel, BertTokenizer

class ALSA:
    """ALSA Framework: Comprehensive Text Analysis System"""
    def __init__(self, model, tokenizer, csv_path, llm_model="gpt2"):
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
        self.PLRS = PLRS.PLRSCalculator()
        self.CIIS = CIIS.CIISCalculator(model, tokenizer)
        self.TRS = TRS.TRSCalculator(llm_model=llm_model)
        self.CASM = CASM.CASMCalculator() 
    
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
        for word, score in PLRS_metrics.items():
            CASM_metrics[word] = [score, 0.0, 0.0]
        
        for word, score in CIIS_metrics.items():
            if word in CASM_metrics:
                CASM_metrics[word][1] = score
            else:  # This may need adjustment
                CASM_metrics[word] = [0.0, score, 0.0]
        
        for word, score in TRS_metrics.items():
            if word in CASM_metrics:
                CASM_metrics[word][2] = score
            else:
                CASM_metrics[word] = [0.0, 0.0, score]
        
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
        Execute text replacement and save to output.csv
        :param part2_output: Dictionary from part2 {word: replacement}
        :param csv_path: Path to original CSV file
        """
        print('\n\033[1;32mStarting Part 3...\033[0m')
        # Read original CSV
        df = pd.read_csv(csv_path)
        
        # Create replacement patterns (handle word boundaries)
        sorted_words = sorted(part2_output.keys(), 
                            key=lambda x: len(x), 
                            reverse=True)
        
        patterns = {
            word: re.compile(r'\b{}\b'.format(re.escape(word)))
            for word in sorted_words
        }

        # Process each sentence
        for index in df.index:
            original = df.at[index, 'sentence']
            modified = original
            
            # Apply replacements in descending length order
            for word in sorted_words:
                modified = patterns[word].sub(part2_output[word], modified)
            
            df.at[index, 'sentence'] = modified

        # Save with original structure
        df.to_csv('output.csv', index=False)
        print("\033[1;32mPart3 completed. Saved to output.csv\033[0m")

if __name__ == "__main__":
    # Initialize components
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare test data
    test_data = {
        "sentence": ["Jon applied for a loan using his credit card."],
        "task_prompt": ["Identify financial-related word"]
    }
    pd.DataFrame(test_data).to_csv("test_trs.csv", index=False)
    
    # Execute ALSA
    alsa = ALSA(
        model, 
        tokenizer, 
        csv_path="test_trs.csv",
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    alsa.calculate()

    print("\n\033[1;32mALL COMPLETED\033[0m")

