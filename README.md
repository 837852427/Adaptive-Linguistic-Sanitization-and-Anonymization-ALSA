# ALSA:Context-Sensitive Prompt Privacy Preservation in Large Language Models

<img src="https://github.com/837852427/Adaptive-Linguistic-Sanitization-and-Anonymization-ALSA/blob/main/ALSA%20Framework.png?raw=true" style="zoom:50%;" />

The ALSA framework comprises four modules, which integrate the results of Privacy Leakage Risk Assessment, Contextual Information Importance Assessment, and Task Relevance Assessment into Clustering and Action Selection. This process then generates a privacy-preserved prompt for a cloud-based black-box LLM to execute.

<h2>Dependencies </h2>

* Python >= 3.9
* PyTorch >= 2.6.0


<h2>Usage </h2>

**Step1**: Navigate to the main code directory within the current repository, which contains `ALSA.py` and the core framework of ALSA.

```bash
cd Code 
```

**Step2**: Execute the code.

ALSA can be executed with its default configuration using the following command:

```bash
python ALSA.py
```

Alternatively, users may modify its configuration to fit specific requirements, as shown below:

```bash
python ALSA.py \
--data_path "huggingface:imdb" \
--bert_model "bert-base-uncased" \
--llm_model "decapoda-research/llama-7b-hf" \
--k_means 8 \
--lambda_1 0.5 \
--lambda_2 0.7 \
--alpha 0.6 \
--beta 0.4 \
--gamma 0.2 \
--spacy_model "en_core_web_sm" \
--output_path "output.csv"
```
#### configuration Explanation:

1. **`--data_path`**  
   Path to the dataset. This can either be a local path (e.g., `data/ALSA.csv`) .

2. **`--bert_model`**  
   The name of the BERT model. It supports pre-trained BERT models from HuggingFace (e.g., `bert-base-uncased`). 

3. **`--llm_model`**  
   The name of the backbone LLM. It supports pre-trained Llama models from HuggingFace (e.g., `decapoda-research/llama-7b-hf`). 

4. **`--k_means`**  
   The `k` value for K-means clustering.

5. **`--lambda_1`**  
   The first lambda parameter for CIIS calculation. It controls the weight of the first hyperparameter in CIIS.

6. **`--lambda_2`**  
   The second lambda parameter for CIIS calculation. It controls the weight of the second hyperparameter in CIIS.

7. **`--alpha`**  
   In CIIS, the alpha hyperparameter is used in the Contextual Coherence (CC) computation, directly affecting the matrix $Q_{ij}$.

8. **`--beta`**  
   In CIIS, the beta hyperparameter is used in the Contextual Coherence (CC) computation, directly affecting the $r^*_{ij}$.

9. **`--gamma`**  
   In CIIS, the gamma hyperparameter is used in the Contextual Coherence (CC) computation, directly affecting the matrix $D_{ij}$.

10.  **`--spacy_model`**  
    The spaCy model to be used for natural language processing tasks (e.g., `en_core_web_sm`).

11. **`--output_path`**  
    The path to save the final privacy-preserved prompt. The processed results will be saved to this file.


<h2>Project Structure </h2>

```bash

├── Code
│   ├── data
│   │   └── ALSA.csv
│   ├── ALSA.py
│   ├── CASM.py
│   ├── CCCalculator.py
│   ├── CIIS.py
│   ├── output.csv
│   ├── PLRS.py
│   ├── SDCalculator.py
│   └── TRS.py
├── ALSA Framework.png
├── README.md
```
