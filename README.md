# Adaptive Linguistic Sanitization and Anonymization (ALSA)

<img src="https://github.com/837852427/Adaptive-Linguistic-Sanitization-and-Anonymization-ALSA/blob/main/ALSA%20Framework.png?raw=true" style="zoom:50%;" />

The ALSA framework comprises four modules, which integrate the results of Privacy Leakage Risk Assessment, Contextual Information Importance Assessment, and Task Relevance Assessment into Clustering and Action Selection. This process then generates a privacy-preserved prompt for a cloud-based black-box LLM to execute.

<h2>Dependencies </h2>

* Python >= 3.9.21
* [PyTorch](https://pytorch.org/) >= 2.6.0+cu118


<h2>Basic Usage </h2>

```bash
cd Code 
python ALSA.py
```

Step1：Switch to the main code directory of the project (including ALSA.py and model code)
Step2：Execution code

The following parameters are set

```bash
python ALSA.py \
--csv_path "huggingface:imdb" \
--bert_model "bert-base-uncased" \
--llm_model "EleutherAI/gpt-neo-1.3B" \
--k 8 \
--lambda_1 0.5 \
--lambda_2 0.7 \
--alpha 0.6 \
--beta 0.4 \
--gamma 0.2 \
--spacy_model "en_core_web_sm" \
--output_path "output.csv"
```

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
