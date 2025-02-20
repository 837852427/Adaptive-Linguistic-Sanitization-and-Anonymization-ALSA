
import CCCalculator as cc
import SDCalculator as sd
from transformers import BertTokenizer, BertModel

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

    def calculate_CIIS(self, sentence):
        """
        Calculate the Contextual Importance of Individual Sentences (CIIS) for each word in a sentence.
        """

        # Calculate the Contextual Coherence (CC)
        cc_scores = self.cc_calculator.calculate_cc(sentence)
        print(f'cc_scores: {cc_scores}')
        print("\033[92m\033[1mCC calculation completed\033[0m\033[22m")

        # Calculate the Semantic Difference (SD)
        sd_scores = self.sd_calculator.calculate_sd(sentence)
        print(f'sd_scores: {sd_scores}')
        print("\033[92m\033[1mSD calculation completed\033[0m\033[22m")

        ciis_scores = {}
        for word in cc_scores:
            ciis_scores[word] = self.lambda_1 * cc_scores[word] + self.lambda_2 * sd_scores[word]
        
        return ciis_scores

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    ciis_calculator = CIISCalculator(model, tokenizer)
    sentence = "The quick brown fox jumps over the lazy dog"
    ciis_scores = ciis_calculator.calculate_CIIS(sentence)
    print(f'CIIS scores: {ciis_scores}')
    print("\033[92m\033[1mCIIS calculation completed\033[0m\033[22m")