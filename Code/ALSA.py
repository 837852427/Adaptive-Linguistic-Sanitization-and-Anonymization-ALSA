import sys
import subprocess
import pandas as pd
import argparse
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import PLRS
import CIIS
import TRS
import CASM
import torch
from datasets import load_dataset

import time

class ALSA:
    def __init__(self, bert_model, bert_tokenizer, data_path, output_path,  llm_model, k_means,
                 lambda_1=0.4, lambda_2=0.6, alpha=0.5, beta=0.5, gamma=0.3, 
                 spacy_model="en_core_web_sm"):
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.bert_model_name = bert_model
        self.bert_tokenizer_tokenizer = bert_tokenizer
        self.llm_model_name = llm_model
        if data_path.startswith("huggingface"):
            dataset_name = data_path.split("/")[1]
            dataset = load_dataset(dataset_name)
            self.dataset = pd.DataFrame(dataset["train"])
        else:
            self.dataset = pd.read_csv(data_path)
            self.sent_list = self.dataset["sentence"].tolist()
        self.data_path = data_path
        self.output_path = output_path
        self.k_means = k_means
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spacy_model = spacy_model
        self.bert_model = BertModel.from_pretrained(
            bert_model,
            torch_dtype=torch.float16,
        ).to(self.device) 
        if self.device.type == 'cuda':
            self.bert_model = self.bert_model.half()
            torch.backends.cudnn.benchmark = True 
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map="auto",  
            torch_dtype=torch.float16  
        ).eval().requires_grad_(False)
        self.PLRS = PLRS.PLRSCalculator(
            self.bert_model, 
            self.bert_tokenizer,
            device=self.device,  
            data_path=data_path,
            spacy_model=spacy_model
        )
        self.CIIS = CIIS.CIISCalculator(self.bert_model, self.bert_tokenizer, 
                                        lambda_1, lambda_2, alpha, beta, gamma, spacy_model)
        self.TRS = TRS.TRSCalculator(bert_tokenizer=self.bert_tokenizer,
                                      llm_tokenizer=self.llm_tokenizer, 
                                      llm_model=self.llm_model, device=self.device)   
        self.CASM = CASM.CASMCalculator(k=self.k_means, llm_model=self.llm_model)
    def calculate(self):
        PLRS_metrics = self.PLRS.calculate(self.sent_list)
        CIIS_metrics = self.CIIS.calculate(self.sent_list)
        TRS_metrics = self.TRS.calculate(list(zip(self.sent_list, 
                                        self.dataset["task_prompt"].tolist())))
        triple_metrics = {}
        for key, score in PLRS_metrics.items():
            triple_metrics[key] = [score, 0.0, 0.0]
        for key, score in CIIS_metrics.items():
            if key in triple_metrics:
                triple_metrics[key][1] = score
            else:
                triple_metrics[key] = [0.0, score, 0.0]
        for key, score in TRS_metrics.items():
            if key in triple_metrics:
                triple_metrics[key][2] = score
            else:
                triple_metrics[key] = [0.0, 0.0, score]
        replacement_dict = self.CASM.calculate(triple_metrics)
        original_sentences = self.dataset['sentence']
        sentence_map = defaultdict(lambda: defaultdict(dict))
        for (word, sent), r_word in replacement_dict.items():
            sent_hash = hash(sent)
            sentence_map[sent_hash][word.lower()][word] = r_word
        processed_sentences = []
        for original_sent in original_sentences:
            sent_hash = hash(original_sent)
            if sent_hash not in sentence_map:
                processed_sentences.append(original_sent)
                continue
            word_dict = sentence_map[sent_hash]
            tokens = original_sent.split()  
            replace_map = {}
            for i, token in enumerate(tokens):
                lower_token = token.lower()
                if lower_token in word_dict:
                    if token in word_dict[lower_token]:
                        replace_map[i] = word_dict[lower_token][token]
                    elif token.title() in word_dict[lower_token]:
                        replace_map[i] = word_dict[lower_token][token.title()]
            for i, r_word in replace_map.items():
                tokens[i] = r_word
            processed_sentences.append(' '.join(tokens))
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "pandas", "spacy", "datasets"])

if __name__ == "__main__":
    # install_requirements()
    parser = argparse.ArgumentParser(description='Run ALSA Text Analysis')
    parser.add_argument('--data_path', type=str, default="data/ALSA.csv", help='Path to CSV dataset or HuggingFace dataset')
    parser.add_argument('--bert_model', type=str, default="bert-base-uncased", help='BERT model')
    parser.add_argument('--llm_model', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Llama model')
    parser.add_argument('--k_means', type=int, default=8, help='K-means clustering parameter')
    parser.add_argument('--lambda_1', type=float, default=0.4, help='Lambda 1 for CIIS')
    parser.add_argument('--lambda_2', type=float, default=0.6, help='Lambda 2 for CIIS')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for CASM')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for CASM')
    parser.add_argument('--gamma', type=float, default=0.3, help='Gamma parameter for CASM')
    parser.add_argument('--spacy_model', type=str, default="en_core_web_sm", help='spaCy model')
    parser.add_argument('--output_path', type=str, default="output.txt", help='Path to save the output file')
    args = parser.parse_args()
    alsa = ALSA(
        bert_model=args.bert_model, bert_tokenizer=args.bert_model, 
        data_path=args.data_path,output_path=args.output_path,
        llm_model=args.llm_model,
        k_means=args.k_means,
        lambda_1=args.lambda_1,lambda_2=args.lambda_2,
        alpha=args.alpha,beta=args.beta,gamma=args.gamma,
        spacy_model=args.spacy_model
    )
    alsa.calculate()