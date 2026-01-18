# RLShield

This repo contains the code for **RLShield**. For algorithmic details, please refer to the paper.

## Pipeline Overview (see paper for details)

1) **Dynamic Reference Selection**  
   Retrieve and rewrite safe/unsafe references for each prompt.  

2) **Critical Parameter Localization**  
   locate safety layers, select critical parameters and compute harmfulness score.  

3) **RL-Driven Adaptive Thresholding**  
   Train a SAC policy to output a **prompt-specific threshold** for final binary detection.  



---

## Code Layout

```
RLShield/
├── Dynamic_Reference_Selection/
│   ├── retrieve.py          # retrieve safe/unsafe references
│   └── rewrite.py           # rewrite references with vLLM
│
├── Critical_Parameter_Localization/
│   ├── safety_layer/
│   │   ├── location.py      # locate safety layers
│   │   └── utils/           # helper utilities
│   │
│   ├── safety_parament_and_score/
│   │   ├── fixed.py         # fixed reference filter
│   │   ├── dynamic.py       # dynamic reference filter
│   │   └── score.py         # compute harmfulness score
│   │
│   └── templates/           # prompt templates
│
└── RL-Driven_Adaptive_Thresholding/
    ├── train_sac.py         # train SAC threshold policy
    └── evaluate_sac.py      # evaluate
```

---

## Data and Models (download required)

Datasets and model checkpoints are **not included** in this repository. Please download them from Hugging Face before running:

**Datasets**
- XSTest: https://huggingface.co/datasets/walledai/XSTest
- ToxicChat: https://huggingface.co/datasets/lmsys/toxic-chat
- WildGuardMix: https://huggingface.co/datasets/allenai/wildguardmix

**Models**
- Llama-2-7b-chat-hf: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- Llama-3.1-8B-Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---


