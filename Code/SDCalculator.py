import torch
from sklearn.metrics.pairwise import rbf_kernel
import nltk
import spacy
from nltk.corpus import wordnet as wn
# nltk.download('wordnet')

class SDCalculator:
    def __init__(self, model, tokenizer, nlp=spacy.load("en_core_web_sm")):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = nlp  # Load spaCy model for POS tagging and dependency parsing

    def calculate_sd(self, sentence, max_syn=3):
        pos_tags = self.get_pos_tags_spacy(sentence)
        words, word_embeddings = self.get_words_embedding(sentence, pos_tags)
        
        if len(words) == 0:
            return {}

        # Obtain original token position mapping
        original_token_map = self.get_token_positions(sentence, words)
        
        sd_scores = {}
        
        batch_sents = []
        meta = []            

        for i, word in enumerate(words):
            syns = self.get_valid_synonyms(word)[:max_syn]
            for syn in syns:
                new_sent, ok = self.replace_with_alignment(sentence, original_token_map[i], syn)
                if ok:
                    batch_sents.append(new_sent)
                    meta.append(i)

        if not batch_sents:
            return {w: 0.0 for w in words}

        inputs = self.tokenizer(batch_sents, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=512).to(self.model.device)
        with torch.no_grad():
            all_emb = self.model(**inputs, output_hidden_states=True).hidden_states[-1]

        sd_sum = [0.0] * len(words)
        cnt    = [0]   * len(words)

        for b, i_word in enumerate(meta):
            cur_emb = all_emb[b][original_token_map[i_word]]
            diff = 1 - torch.nn.functional.cosine_similarity(
                        word_embeddings[i_word], cur_emb, dim=0).item()
            sd_sum[i_word] += diff
            cnt[i_word]    += 1

        return {words[i]: round(sd_sum[i]/cnt[i], 2) if cnt[i] else 0.0
                for i in range(len(words))}

    # Helper methods
    def get_token_positions(self, sentence, target_words):
        """Map target words to their positions in token sequence"""
        tokens = self.tokenizer.tokenize(sentence)
        clean_tokens = [t.replace("##", "").lower() for t in tokens]
        
        position_map = []
        search_idx = 0
        
        for word in target_words:
            subword = self.tokenizer.tokenize(word)[0].replace("##", "").lower()
            found = False
            for i in range(search_idx, len(clean_tokens)):
                if clean_tokens[i] == subword:
                    position_map.append(i)
                    search_idx = i + 1
                    found = True
                    break
            if not found:
                position_map.append(-1)
        
        return position_map

    def replace_with_alignment(self, sentence, token_idx, synonym):
        """Token-aligned replacement method"""
        tokens = self.tokenizer.tokenize(sentence)
        if token_idx >= len(tokens):
            return sentence, False
        
        # Generate modified token sequence
        synonym_tokens = self.tokenizer.tokenize(synonym)
        new_tokens = tokens[:token_idx] + synonym_tokens + tokens[token_idx+1:]
        
        try:
            # Reconstruct valid sentence
            modified_sentence = self.tokenizer.convert_tokens_to_string(new_tokens)
            return modified_sentence, True
        except:
            return sentence, False

    def get_valid_synonyms(self, word):
        """Retrieve synonyms present in model's vocabulary"""
        if not wn._synset_from_pos_and_offset:
            wn.ensure_loaded()
            
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace('_', ' ')
                # Verify existence in tokenizer's vocabulary
                if self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(candidate)) != [self.tokenizer.unk_token_id]:
                    synonyms.add(candidate)
        return list(synonyms)

    def get_aligned_embeddings(self, sentence, original_pos_map):
        """Retrieve embeddings aligned with original positions"""
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        embeddings = outputs.hidden_states[-1].squeeze(0)
        
        # Create current token position mapping
        current_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        current_clean = [t.replace("##", "").lower() for t in current_tokens]
        
        aligned_embeddings = []
        for pos in original_pos_map:
            if pos < len(current_clean):
                aligned_embeddings.append(embeddings[pos])
            else:
                return None  # Position mismatch
        
        return torch.stack(aligned_embeddings)

    def calculate_semantic_difference(self, original_embedding, modified_embedding, gamma=0.1):
        """
        Compute semantic difference using cosine similarity
        Returns: 1 - similarity score (difference metric)
        """
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarity = cos(original_embedding, modified_embedding)
        return 1 - similarity.item()

    def get_pos_tags_spacy(self, sentence):
        """
        Extract POS tags with spaCy, excluding punctuation
        """
        doc = self.nlp(sentence)
        pos_tags = [
            (token.text, token.pos_) 
            for token in doc 
            if token.pos_ != 'PUNCT'  # Exclude punctuation
        ]
        return pos_tags

    def get_words_embedding(self, sentence, pos_tags):
        # Tokenize and extract embeddings
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract embeddings and original tokens
        embeddings = outputs.hidden_states[-1].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        
        # Generate cleaned tokens for comparison
        clean_tokens = [
            token.replace("##", "").lower() 
            for token in tokens 
            if token not in self.tokenizer.all_special_tokens
        ]
        
        # Token matching logic
        matched_words = []
        matched_embeddings = []
        search_start = 0  # Track search start position
        
        for original_word, pos in pos_tags:
            # Tokenize original word
            subwords = self.tokenizer.tokenize(original_word)
            if not subwords:
                continue
                
            # Get cleaned version of first subword
            target = subwords[0].replace("##", "").lower()
            
            # Sequential search in token sequence
            for idx in range(search_start, len(clean_tokens)):
                if clean_tokens[idx] == target:
                    # Preserve original word casing
                    matched_words.append(original_word)
                    # Get corresponding embedding
                    matched_embeddings.append(embeddings[idx])
                    # Update search start to prevent duplicates
                    search_start = idx + 1
                    break
        
        return matched_words, torch.stack(matched_embeddings) if matched_embeddings else torch.tensor([])


if __name__ == "__main__":
    # Usage demonstration
    from transformers import BertModel, BertTokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sd_calculator = SDCalculator(model, tokenizer)

    sentence = "Unions representing workers at Turner Newall say they are 'disappointed' after talking with stricken parent firm Federal Mogul."
    sd_scores = sd_calculator.calculate_sd(sentence)
    print(sd_scores)
