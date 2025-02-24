# Adaptive Linguistic Sanitization and Anonymization (ALSA)

<img src="https://github.com/837852427/Adaptive-Linguistic-Sanitization-and-Anonymization-ALSA/blob/main/ALSA%20Framework.png?raw=true" style="zoom:50%;" />

The ALSA framework comprises four modules, which integrate the results of Privacy Leakage Risk Assessment, Contextual Information Importance Assessment, and Task Relevance Assessment into Clustering and Action Selection. This process then generates a privacy-preserved prompt for a cloud-based black-box LLM to execute.

<h2>Dependencies </h2>

* Python >= 3.9.21
* [PyTorch](https://pytorch.org/) >= 2.6.0+cu118


<h2>Usage </h2>

```bash
cd Code 
python ALSA.py
```

Step1：Switch to the main code directory of the project (including ALSA.py and model code)

Step2：Execution code

The following parameters are set

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
#### Parameter Explanation:

1. **`--data_path`**  
   Path to the dataset. This can either be a local path (e.g., `data/ALSA.csv`) .

2. **`--bert_model`**  
   The name of the BERT model. It supports pre-trained BERT models from HuggingFace (e.g., `bert-base-uncased`). This model is used for CIIS (Contextual Information Integration System) calculations.

3. **`--llm_model`**  
   The name of the Llama model. It supports pre-trained Llama models from HuggingFace (e.g., `decapoda-research/llama-7b-hf`). This model is used for TRS (Text Replacement System) calculations.

4. **`--k_means`**  
   The `k` value for K-means clustering, used in the CASM (Comprehensive Action-based Sentence Metrics) calculation.

5. **`--lambda_1`**  
   The first lambda parameter for CIIS (Contextual Information Integration System) calculation. It controls the weight of the first metric in CIIS.

6. **`--lambda_2`**  
   The second lambda parameter for CIIS calculation. It controls the weight of the second metric in CIIS.

7. **`--alpha`**  
   The alpha parameter for the CASM calculation. It influences the weight of the first metric in the aggregation of results.

8. **`--beta`**  
   The beta parameter for the CASM calculation. It influences the weight of the second metric in the aggregation of results.

9. **`--gamma`**  
   The gamma parameter for the CASM calculation. It influences the weight of the third metric in the aggregation of results.

10.  **`--spacy_model`**  
    The spaCy model to be used for natural language processing tasks (e.g., `en_core_web_sm`).

11. **`--output_path`**  
    The path to save the final output file. The processed results will be saved to this file.


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
