import CCCalculator as cc
import SDCalculator as sd
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np

class CIISCalculator:
    def __init__(self, model, tokenizer, lambda_1 = 0.4, lambda_2 = 0.6, alpha = 0.8, beta = 0.5, gamma = 0.3, space_model = "en_core_web_sm"):
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.space_model = space_model
        self.cc_calculator = cc.CCCalculator(self.model, self.tokenizer, self.alpha, self.beta, self.gamma, self.space_model)
        self.sd_calculator = sd.SDCalculator(self.model, self.tokenizer)


    def calculate(self, csv_path):
        """
        Calculate the CIIS scores for each word in a sentence.
        """
        print('\n\033[1;32mCIIS module startup...\033[0m')
        print("\n\033[1mInput sentences\033[0m")

        sentences = self.input_sentence(csv_path)

        print("\n\033[1mInput sentences completed\033[0m")

        ciis_scores = {}

        # Calculate the CIIS scores for each sentence and merge them
        print("\n\033[1mCalculating CIIS...\033[0m")
        for sentence in sentences:
            ciis_scores_sentence = self.calculate_CIIS(sentence)
            ciis_scores.update(ciis_scores_sentence)
        print("\n\033[1mCalculating CIIS completed\033[0m")


        print('\n\033[1;32mThe CIIS module calculation has been completed.\033[0m')
        return ciis_scores

    def calculate_CIIS(self, sentence):
        """
        Calculate the Contextual Importance of Individual Sentences (CIIS) for each word in a sentence.
        """

        # Calculate the Contextual Coherence (CC)
        cc_scores = self.cc_calculator.calculate_cc(sentence)
        print(f'\ncc_scores: {cc_scores}')

        if __name__ == "__main__":
            print("\033[92m\033[1mCC calculation completed\033[0m\033[22m")

        # Calculate the Semantic Difference (SD)
        sd_scores = self.sd_calculator.calculate_sd(sentence)
        print(f'\nsd_scores: {sd_scores}')

        if __name__ == "__main__":
            print("\033[92m\033[1mSD calculation completed\033[0m\033[22m")

        ciis_scores = {}

        if __name__ == "__main__":
            print(f'sd_scores:\n{sd_scores}\ncc_scores:\n{cc_scores}\n')
            print(f'sd_scores.type: {type(sd_scores)}    sd_scores.shape: {len(sd_scores)}')
            print(f'cc_scores.type: {type(cc_scores)}    cc_scores.shape: {len(cc_scores)}')
        
        # Merge SD and CC
        for word in cc_scores:
            if __name__ == "__main__":
                print(f'word: {word}, cc_scores[word]: {cc_scores[word]}, sd_scores[word]: {sd_scores[word]}')
            
            ciis_scores[(word, sentence)] = self.lambda_1 * cc_scores[word] + self.lambda_2 * sd_scores[word]
        
        return ciis_scores

    def input_sentence(self, csv_path):
        """
        Read the input sentence from a CSV file.
        """
        df = pd.read_csv(csv_path)
        sentences = df['sentence'].tolist()
        
        return sentences

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    ciis_calculator = CIISCalculator(model, tokenizer)
    csv_path = "D:/论文/ALSA/test.csv"
    ciis_scores = ciis_calculator.calculate(csv_path)
    for (word, sentence), score in ciis_scores.items():
        print(f'\n{word} from "{sentence}" and score is {score}')