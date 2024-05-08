# Enhancing Text Summarization of Biomedical Journal Papers with Domain-Specific Knowledge Integration in State-of-the-Art NLP Models

### Nilay Bhatt, Tom Shin, Luning Yang

## Abstract
The exponential growth of biomedical literature over the past decade has not just created a need, but a pressing need for efficient summarization tools. These tools are crucial for researchers to stay informed about recent developments in their field. 
As the volume and complexity of scientific papers increase, automated summarization has become indispensable for researchers aiming to distill key information rapidly. 
Although modern Natural Language Processing (NLP) models like BERT and GPT have shown promising results in text summarization, they often need help to fully capture the nuances and domain-specific language inherent in biomedical texts. This results in summaries that lack accuracy or comprehensiveness, posing a significant challenge for researchers.
To address these challenges, this project not only leverages state-of-the-art NLP models, including BART, T5, BioGPT, and LED, but also supplements them with domain-specific biomedical knowledge. 
This unique approach is designed to enhance the summarization quality of biomedical journal papers.
By integrating specialized knowledge with these advanced models, we aim to not just improve the accuracy and conciseness of summaries, but also make them contextually relevant. 
This will enable researchers to navigate the rapidly expanding scientific literature more effectively. 
Our experimental design involves in-domain and cross-domain summarization tasks to rigorously assess and refine our models. 
Ultimately, our goal is to establish new benchmarks for summarization in this specialized field, a significant step towards advancing biomedical literature summarization.

## Navigation

```bash
/Text_Summarization_LLMs
├── Data
│   ├── Experiment_1
│   │   ├── eLife_val_50_articles.jsonl
│   │   ├── model_BART
│   │   │   └── bart_plos_50_articles.txt
│   │   ├── model_BIOGPT
│   │   │   └── biogpt_plos_50_articles.txt
│   │   ├── model_LED
│   │   │   └── led_plos_50_articles.txt
│   │   ├── model_T5
│   │   │   └── t5_plos_50_articles.txt
│   │   └── PLOS_val_50_articles.jsonl
│   └── Experiment_2
│       ├── data
│       │   ├── ground_truth
│       │   │   └── arxiv_pubmed_test.jsonl
│       │   └── sub_prediction
│       │       ├── FT-AP
│       │       │   └── long-ft-arxiv_pubmed-grt-AP_final.txt
│       │       ├── FT-Arvix
│       │       │   └── long-ft-arvix-grt-AP_final.txt
│       │       └── FT-Pubmed
long-ft-pubmed-grt-AP_final.txt
│       ├── data_Arvix
│       │   ├── ground_truth
│       │   │   └── arvix_test_50.jsonl
│       │   └── sub_prediction
│       │       ├── FT-AP
│       │       │   └── long-ft-arvix_pubmed-grt-AP_f50.txt
│       │       ├── FT-Arvix
│       │       │   └── long-ft-arvix-grt-AP_f50.txt
│       │       └── FT-Pubmed
│       │           └── long-ft-pubmed-grt-AP_f50.txt
│       └── data_Pubmed
│           ├── ground_truth
│           │   └── pubmed_test_50.jsonl
│           └── sub_prediction
│               ├── FT-AP
│               │   └── long-ft-arvix_pubmed-grt-AP_l50.txt
│               ├── FT-Arvix
│               │   └── long-ft-arvix-grt-AP_l50.txt
│               └── FT-Pubmed
│                   └── long-ft-pubmed-grt-AP_l50.txt
├── Generated
│   ├── final
│   │   ├── First_50
│   │   │   ├── long-ft-arvix-grt-AP_f50.txt
│   │   │   ├── long-ft-arxiv_pubmed-grt-AP_f50.txt
│   │   │   └── long-ft-pubmed-grt-AP_f50.txt
│   │   ├── Last_50
│   │   │   ├── long-ft-arvix-grt-AP_l50.txt
│   │   │   ├── long-ft-arxiv_pubmed-grt-AP_l50.txt
│   │   │   └── long-ft-pubmed-grt-AP_l50.txt
│   │   ├── long-ft-arvix-grt-AP_final.txt
│   │   ├── long-ft-arxiv_pubmed-grt-AP_final.txt
│   │   └── long-ft-pubmed-grt-AP_final.txt
│   ├── long-ft-arvix-grt-AP.txt
│   ├── long-ft-arxiv_pubmed-grt-AP
│   └── long-ft-pubmed-grt-AP.txt
├── Results
│   ├── Experimental_figure1.png
│   ├── Experimental_figure2.png
│   ├── experimental_results.csv
│   ├── long-ft-arvix-grt-AP-eval.txt
│   ├── long-ft-arvix-grt-AP_f50-eval.txt
│   ├── long-ft-arvix-grt-AP_l50-eval.txt
│   ├── long-ft-arvix_pubmed-grt-AP-eval.txt
│   ├── long-ft-arvix_pubmed-grt-AP_f50-eval.txt
│   ├── long-ft-arvix_pubmed-grt-AP_l50-eval.txt
│   ├── long-ft-pubmed-grt-AP-eval.txt
│   ├── long-ft-pubmed-grt-AP_f50-eval.txt
│   └── long-ft-pubmed-grt-AP_l50-eval.txt
└── Scripts
    ├── Experiment_1
    │   └── evaluate.py
    ├── Experiment_2
    │   ├── evaluate_A_AP.py
    │   ├── evaluate_AP_AP-2.py
    │   └── evaluate_P_AP-2.py
    ├── finetune_bart.py
    ├── finetune_biogpt.py
    ├── finetune_long_2-2.py
    ├── finetune_long-2.py
    ├── finetune_long_3.py
    ├── finetune_t5 (1).py
    ├── generate_bart.py
    ├── generate_biogpt.py
    ├── generate_long-2-2.py
    ├── generate_long-3.py
    ├── generate_long-4.py
    └── generate_t5 (1).py

