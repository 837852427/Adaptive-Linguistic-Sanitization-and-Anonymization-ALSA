import torch
import torch.nn.functional as F
import re
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
class TRSCalculator:
    """
    Class to compute the Privacy Leakage Risk Score (PLRS) for text.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def calculate_trs(self, text, task_prompt, k=5):
        """
        Calculate average task relevance by running generate_response k times.
        """
        relevance_aggregator = {}
        for _ in range(k):
            assistant_response = self.generate_task_relevance_response(text, task_prompt)
            task_relevance = self.extract_task_relevance(assistant_response)
            for word, relevance in task_relevance.items():
                relevance_aggregator.setdefault(word, []).append(relevance)

        # print the relevance_aggregator
        print("Relevance Aggregator:\n", relevance_aggregator)
        
        average_task_relevance = {
            word: round(sum(relevances) / len(relevances), 2)
            for word, relevances in relevance_aggregator.items()
        }
        # print("TR:\n",average_task_relevance)
        return average_task_relevance

    def generate_task_relevance_response(self, text, task_prompt):
        """
        Generate the assistant's response for task relevance.
        """
        prompt = f"""
I have the following text:

{text}

And it is within the task:

{task_prompt}

Please evaluate the task relevance of each word phrase (word group) w_i in the text based on the task above:

Instructions:

- For each word w_i in the text, compute its task relevance T(w_i) on a scale from 0.00 to 1.00, where:
  - 0.00 means the word is irrelevant to the task.
  - 1.00 means the word is highly relevant to the task.
- Ensure that the most accurate and likely task relevance is computed for each word or phrase, considering context and meaning.
- Ensure that named entities (e.g., person names, place names, organization names, specialized phrases, etc.) are kept as a whole with the same task relevance.
- Present the results with the task relevance value for each word or phrase rounded to at least two decimal places.

Example:

text: John applied for a loan using his credit card.
task output:
1. applied for: 0.82
2. loan: 0.93
3. credit card: 0.88
...
"""
        conversation_history = [("user", prompt)]
        assistant_response = self.generate_response(conversation_history)
        # print("Assistant Response:\n", assistant_response)
        return assistant_response

    def extract_task_relevance(self, assistant_response):
        """
        Extract task relevance T(w_i) from the assistant's response.
        Supports various formats, such as:
        - 1. Word: 0.75 - explanation text
        - 1. "Word Phrase" - Task relevance: 0.85
        - 1. Word - Task relevance: 0.75 (explanation)
        """
        # print("Assistant Response:")
        # print(assistant_response)
        # Regular expression pattern to match different formats
        pattern = r'^\d+\.\s*(?:"([^"]+)"|([\w\s\.\'\-]+?))\s*(?:-|:)\s*(?:Task relevance:)?\s*([\d\.]+)'
        matches = re.findall(pattern, assistant_response, re.MULTILINE)

        task_relevance = {}
        if matches:
            try:
                for match in matches:
                    word = match[0] or match[1]
                    word = word.strip()
                    relevance = float(match[2])
                    task_relevance[word] = relevance
            except ValueError:
                print("Failed to convert task relevance values to float.")
        else:
            print("Task relevance data not found in assistant response.")

        return task_relevance
    
    def generate_response(self, conversation_history):
        """
        Generate the assistant's reply based on the conversation history.
        """
        prompt = self.format_conversation(conversation_history)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        # Check if the tokenized input exceeds the model's max length (1024 tokens for GPT-2)
        if inputs["input_ids"].size(1) > 1024:
            # If the length exceeds, truncate the tokens to the first 1024 tokens
            inputs["input_ids"] = inputs["input_ids"][:, :1024]
            inputs["attention_mask"] = inputs["attention_mask"][:, :1024]

        # Ensure the model input is within the valid token length
        inputs = inputs.to(self.model.device)  # Move to the correct device (e.g., GPU)

        # Generate response from the model
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,  # Adjust max_new_tokens to limit the output length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode the response (excluding the prompt part)
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

        return generated_text.strip()


    def format_conversation(self, conversation_history):
        """
        Format the conversation history to build the model input.
        """
        prompt = ""
        for i, (speaker, text) in enumerate(conversation_history):
            if speaker == "user":
                prompt += f"<s>[INST] {text.strip()} [/INST]"
            elif speaker == "assistant":
                prompt += f" {text.strip()}"
        return prompt

if __name__ == "__main__":
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # 添加以下两行设置pad_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    trs_calculator = TRSCalculator(model, tokenizer)
    text = "John applied for a loan using his credit card."
    task_prompt = "Identify the key terms in the text."
    trs = trs_calculator.calculate_trs(text, task_prompt)
    print(trs)