import torch
from sklearn.metrics.pairwise import rbf_kernel
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
class SDCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def calculate_sd(self, sentence):
        """
        Calculate the Semantic Distinctiveness (SD) for each word in the sentence.
        """
        # Tokenize the sentence and get the embeddings
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Use the last hidden state as the embedding
        embeddings = outputs.hidden_states[-1].squeeze(0)  # shape: (sequence_length, hidden_size)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        words = [self.tokenizer.convert_tokens_to_string([t]) for t in tokens if t not in self.tokenizer.all_special_tokens]
        word_embeddings = embeddings[:len(words)].cpu().numpy()

        sd_scores = {}

        # For each word in the sentence, calculate its SD
        for i, word in enumerate(words):
            candidate_synonyms = self.get_synonyms(word)  # Get synonyms for the word
            semantic_diff_sum = 0.0

            for synonym in candidate_synonyms:
                # Generate the modified sentence with the synonym
                modified_sentence = self.replace(sentence, i, synonym)
                modified_inputs = self.tokenizer(modified_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
                with torch.no_grad():
                    modified_outputs = self.model(**modified_inputs, output_hidden_states=True)

                # Get the embeddings for the modified sentence
                modified_embeddings = modified_outputs.hidden_states[-1].squeeze(0)
                modified_word_embeddings = modified_embeddings[:len(words)].cpu().numpy()

                # Calculate the semantic difference (cosine similarity)
                semantic_diff = self.calculate_semantic_difference(word_embeddings, modified_word_embeddings)
                semantic_diff_sum += semantic_diff
            
            # Average the semantic differences for the word
            if len(candidate_synonyms) == 0:
                sd_scores[word] = 0.0
            else:
                sd_scores[word] = round(semantic_diff_sum / len(candidate_synonyms), 2)

        return sd_scores

    def replace(self, sentence, i, word):
        replace_list = sentence.split()
        replace_list[i] = word
        replace_sentence = ' '.join(replace_list)
        return replace_sentence
        

    def get_synonyms(self, word):
        """
        Retrieve synonyms for a word from WordNet or another source.
        This method should return a list of words (synonyms).
        """
        # Get the synonyms from WordNet
        synonyms = set()  # 使用集合避免重复
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')  # 替换下划线为空格
                synonyms.add(synonym)

        #输出WordNet中的同义词
        if __name__ == "__main__":
            print(f'word: {word}\nsynonyms: {synonyms}\n')

        
        return list(synonyms)
        

    def calculate_semantic_difference(self, original_embedding, modified_embedding, gamma=1.0):
        """
        Calculate the semantic difference using Maximum Mean Discrepancy (MMD).
        """
        # Ensure the embeddings are in the correct shape
        original_embedding = original_embedding.reshape(1, -1)
        modified_embedding = modified_embedding.reshape(1, -1)

        # Compute the RBF kernel for the original and modified embeddings
        kernel_original = rbf_kernel(original_embedding, original_embedding, gamma=gamma)
        kernel_modified = rbf_kernel(modified_embedding, modified_embedding, gamma=gamma)
        kernel_cross = rbf_kernel(original_embedding, modified_embedding, gamma=gamma)

        # Compute MMD
        mmd = kernel_original.mean() + kernel_modified.mean() - 2 * kernel_cross.mean()

        return mmd

if __name__ == "__main__":
    # Example usage of the SDCalculator class
    from transformers import BertModel, BertTokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sd_calculator = SDCalculator(model, tokenizer)

    sentence = "The quick brown fox jumps over the lazy dog"
    sd_scores = sd_calculator.calculate_sd(sentence)
    print(sd_scores)