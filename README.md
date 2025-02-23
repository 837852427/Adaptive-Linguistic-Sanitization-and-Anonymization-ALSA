# Adaptive Linguistic Sanitization and Anonymization (ALSA)

<img src="https://github.com/837852427/Adaptive-Linguistic-Sanitization-and-Anonymization-ALSA/blob/main/ALSA%20Framework.png?raw=true" style="zoom:50%;" />

The ALSA framework comprises four modules, which integrate the results of Privacy Leakage Risk Assessment, Contextual Information Importance Assessment, and Task Relevance Assessment into Clustering and Action Selection. This process then generates a privacy-preserved prompt for a cloud-based black-box LLM to execute.

<h2>Dependencies </h2>

* Python >= 3.9.21
* [PyTorch](https://pytorch.org/) >= 2.6.0+cu118


<h2>Basic Usage </h2>

```bash
cd Code
python ./ALSA.py
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