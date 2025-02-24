import os
import subprocess
import pandas as pd
import argparse
from transformers import BertModel, BertTokenizer, pipeline
from collections import defaultdict
import PLRS
import CIIS
import TRS
import CASM
from datasets import load_dataset
import sys

class ALSA:
    """ALSA Framework: Comprehensive Text Analysis System"""
    def __init__(self, model, tokenizer, csv_path, llm_model, k, bert_model,
                 lambda_1=0.4, lambda_2=0.6, alpha=0.5, beta=0.5, gamma=0.3, spacy_model="en_core_web_sm"):
        """
        Initialize ALSA components
        :param model: Pretrained language model
        :param tokenizer: Text tokenizer
        :param csv_path: Input data path
        :param llm_model: Large Language Model name
        """
        # Check if models are local or need to be downloaded
        self.model = BertModel.from_pretrained(model) if os.path.isdir(model) else BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer) if os.path.isdir(tokenizer) else BertTokenizer.from_pretrained(tokenizer)
        
        # Initialize the LLM model pipeline
        self.llm_pipeline = pipeline("text-generation", model=llm_model)

        # Load CSV data (can be local or from HuggingFace datasets)
        if csv_path.startswith("huggingface"):
            dataset_name = csv_path.split("/")[1]
            dataset = load_dataset(dataset_name)
            self.csv_data = pd.DataFrame(dataset["train"])  # You can change this depending on the dataset split
        else:
            self.csv_data = pd.read_csv(csv_path)
        
        self.csv_path = csv_path
        self.k = k
        self.bert_model = bert_model
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spacy_model = spacy_model

        self.PLRS = PLRS.PLRSCalculator()
        self.CIIS = CIIS.CIISCalculator(self.model, self.tokenizer, lambda_1, lambda_2, alpha, beta, gamma, spacy_model)
        self.TRS = TRS.TRSCalculator(bert_model=self.bert_model, llm_model=self.llm_pipeline)
        self.CASM = CASM.CASMCalculator(k=self.k, llm_model=self.llm_pipeline)

    def calculate(self):
        """Execute complete ALSA analysis pipeline"""
        triple_metrics = self.calculate_part1()
        replacement_dict = self.calculate_part2(triple_metrics)
        self.calculate_part3(replacement_dict, self.csv_path)

    def calculate_part1(self):
        """Execute first stage metrics calculation"""
        print('\033[1;32mStarting Part 1 Calculations...\033[0m')

        # PLRS Calculation
        PLRS_metrics = self.PLRS.calculate()
        print(f'\nPLRS Results:\n{PLRS_metrics}')
        
        # CIIS Calculation
        CIIS_metrics = self.CIIS.calculate(self.csv_path)
        print(f'\nCIIS Results:\n{CIIS_metrics}')
        
        # TRS Calculation
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
            else:
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
        df = self.csv_data
        
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

def install_requirements():
    """Install necessary dependencies"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "pandas", "spacy", "datasets"])

if __name__ == "__main__":
    install_requirements()

    parser = argparse.ArgumentParser(description='Run ALSA Text Analysis')
    parser.add_argument('--csv_path', type=str, default="data/ALSA.csv", help='Path to CSV dataset or HuggingFace dataset')
    parser.add_argument('--bert_model', type=str, default="bert-base-uncased", help='BERT model')
    parser.add_argument('--llm_model', type=str, default="gpt2", help='Large Language Model')
    parser.add_argument('--k', type=int, default=8, help='Parameter k for CASM')
    parser.add_argument('--lambda_1', type=float, default=0.4, help='Lambda 1 for CIIS')
    parser.add_argument('--lambda_2', type=float, default=0.6, help='Lambda 2 for CIIS')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for CASM')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for CASM')
    parser.add_argument('--gamma', type=float, default=0.3, help='Gamma parameter for CASM')
    parser.add_argument('--spacy_model', type=str, default="en_core_web_sm", help='spaCy model')
    parser.add_argument('--output_path', type=str, default="output.csv", help='Path to save the output')
    
    args = parser.parse_args()

    alsa = ALSA(
        model=args.bert_model, 
        tokenizer=args.bert_model, 
        csv_path=args.csv_path,
        llm_model=args.llm_model,
        k=args.k,
        bert_model=args.bert_model,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        spacy_model=args.spacy_model
    )
    alsa.calculate()

    print("\n\033[1;32mALL COMPLETED\033[0m")
