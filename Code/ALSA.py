import sys
import subprocess
import pandas as pd
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from collections import defaultdict
import PLRS
import CIIS
import TRS
import CASM
from datasets import load_dataset

class ALSA:
    """ALSA Framework: Comprehensive Text Analysis System"""
    def __init__(self, model, tokenizer, data_path, llm_model, k_means,
                 lambda_1=0.4, lambda_2=0.6, alpha=0.5, beta=0.5, gamma=0.3, spacy_model="en_core_web_sm"):
        """
        Initialize ALSA components
        :param model: Pretrained language model (Llama model)
        :param tokenizer: Text tokenizer
        :param data_path: Input data path or HuggingFace dataset
        :param llm_model: Large Language Model (used for text generation)
        :param k_means: K-means clustering parameter
        :param lambda_1: CIIS parameter
        :param lambda_2: CIIS parameter
        :param alpha: CASM parameter
        :param beta: CASM parameter
        :param gamma: CASM parameter
        :param spacy_model: spaCy model for NLP tasks
        """
        # Load Llama model and tokenizer
        self.model = LlamaForCausalLM.from_pretrained(model)
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer)

        # Initialize the LLM model pipeline
        self.llm_pipeline = pipeline("text-generation", model=llm_model)

        # Load dataset
        if data_path.startswith("huggingface"):
            dataset_name = data_path.split("/")[1]
            dataset = load_dataset(dataset_name)
            self.data_path = pd.DataFrame(dataset["train"])  # You can change this depending on the dataset split
        else:
            self.data_path = pd.read_csv(data_path)
        
        self.data_path = data_path
        self.k_means = k_means
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spacy_model = spacy_model

        # Initialize other components
        self.PLRS = PLRS.PLRSCalculator()
        self.CIIS = CIIS.CIISCalculator(self.model, self.tokenizer, lambda_1, lambda_2, alpha, beta, gamma, spacy_model)
        self.TRS = TRS.TRSCalculator(bert_model=self.tokenizer, llm_model=self.llm_pipeline)
        self.CASM = CASM.CASMCalculator(k=self.k_means, llm_model=self.llm_pipeline)

    def calculate(self):
        """Execute complete ALSA analysis pipeline"""
        triple_metrics = self.calculate_part1()
        replacement_dict = self.calculate_part2(triple_metrics)
        self.calculate_part3(replacement_dict, self.data_path)

    def calculate_part1(self):
        """Execute first stage metrics calculation"""
        print('\033[1;32mStarting Part 1 Calculations...\033[0m')

        # PLRS Calculation
        PLRS_metrics = self.PLRS.calculate()
        print(f'\nPLRS Results:\n{PLRS_metrics}')
        
        # CIIS Calculation
        CIIS_metrics = self.CIIS.calculate(self.data_path)
        print(f'\nCIIS Results:\n{CIIS_metrics}')
        
        # TRS Calculation
        TRS_metrics = self.TRS.calculate(self.data_path)
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

    def calculate_part3(self, part2_output, data_path):
        """
        Ultimate Replacement Logic: Precise Replacement Based on (word, sentence)
        :param part2_output: Dictionary structure {(word, sentence): r_word}
        :param data_path: Path to the original CSV file
        """
        print('\n\033[1;32mStarting Part 3...\033[0m')
        
        original_sentences = self.dataset['sentence']
        
        # Construct a three-level nested dictionary structure (to optimize query speed)
        sentence_map = defaultdict(lambda: defaultdict(dict))
        
        for (word, sent), r_word in part2_output.items():
            # Generate Sentence Hash Fingerprint (Solve the Performance Issue of Long Text as Key)
            sent_hash = hash(sent)
            # Store the original word form (retain case sensitivity)
            sentence_map[sent_hash][word.lower()][word] = r_word
        
        processed_sentences = []
        for original_sent in original_sentences:
            sent_hash = hash(original_sent)
            
            # Obtain the replacement rules for the current sentence
            if sent_hash not in sentence_map:
                processed_sentences.append(original_sent)
                continue
                
            word_dict = sentence_map[sent_hash]
            tokens = original_sent.split()  # Ensure the sequence by using word segmentation
            
            # Generate Replacement Index Mapping
            replace_map = {}
            for i, token in enumerate(tokens):
                lower_token = token.lower()
                if lower_token in word_dict:
                    # Match the original word exactly in terms of both capitalization and spelling.
                    if token in word_dict[lower_token]:
                        replace_map[i] = word_dict[lower_token][token]
                    # Compatible with variations such as capitalization of the first letter
                    elif token.title() in word_dict[lower_token]:
                        replace_map[i] = word_dict[lower_token][token.title()]
            
            # Perform Index Replacement
            for i, r_word in replace_map.items():
                tokens[i] = r_word
            
            processed_sentences.append(' '.join(tokens))
        
        # Save as TXT file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_sentences))
        
        print("\033[1;32mPart3 completed. Saved to output.txt\033[0m")

def install_requirements():
    """Install necessary dependencies"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "pandas", "spacy", "datasets"])

if __name__ == "__main__":
    install_requirements()

    parser = argparse.ArgumentParser(description='Run ALSA Text Analysis')
    parser.add_argument('--data_path', type=str, default="data/ALSA.csv", help='Path to CSV dataset or HuggingFace dataset')
    parser.add_argument('--bert_model', type=str, default="bert-base-uncased", help='BERT model')
    parser.add_argument('--llm_model', type=str, default="decapoda-research/llama-7b-hf", help='Llama model')
    parser.add_argument('--k_means', type=int, default=8, help='K-means clustering parameter')
    parser.add_argument('--lambda_1', type=float, default=0.4, help='Lambda 1 for CIIS')
    parser.add_argument('--lambda_2', type=float, default=0.6, help='Lambda 2 for CIIS')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for CASM')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for CASM')
    parser.add_argument('--gamma', type=float, default=0.3, help='Gamma parameter for CASM')
    parser.add_argument('--spacy_model', type=str, default="en_core_web_sm", help='spaCy model')
    parser.add_argument('--output_path', type=str, default="output.txt", help='Path to save the output')
    
    args = parser.parse_args()

    alsa = ALSA(
        model=args.llm_model, 
        tokenizer=args.bert_model, 
        data_path=args.data_path,
        llm_model=args.llm_model,
        k_means=args.k_means,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        spacy_model=args.spacy_model
    )
    alsa.calculate()

    print("\n\033[1;32mALL COMPLETED\033[0m")
