# Active Learning in Machine Reading Comprehension
Implementation of the machine reading comprehension architecture with active learning.

## Installation

```bash
git clone https://github.com/huggingface/transformers.git
python -m venv .env #install torch
source .env/bin/activate
pip install transformers-3.4
```

## Model architectures

1. Active Learning in Machine Reading Comprehension
	- Sampling Strategies: query_strategies
	- Framework: abl_run_squad_L2.py
	- Experiments: abl_squad.sh for albation study, /legacy/examples for experiments 
```bash
RES_DIR=/al_transformers/output

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_squad_L2.py \
  --thresh 0.1 \
  --num_round 2 \
  --flag_sub 0 \
  --init 1000 \
  --query 6000 \
  --query_learnrate 1.1 \
  --al_method beyond \
  --version 7 \
  --do_lower_case \
  --al_incremental \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_active \
  --overwrite_output_dir \
  --lambda_parameter 1\
  --per_gpu_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --dataset squad \
  --output_dir $RES_DIR
```

2. Define function compute_prob in the transformers environment, e.g.
```bash
.local/lib/python3.8/site-packages/transformers/data/metrics/squad_metrics.py
```
Function compute_prob is provided in /src/transformers/

3. Rewrite read_squad_examples in the transformers environment, e.g.
```bash
.local/lib/python3.8/site-packages/transformers/data/processors/squad.py
```
Function read_squad_examples is provided in /src/transformers/


## Citation

We now have a [paper](https://openreview.net/forum?id=QaDevCcmcg) you can cite:
```bibtex
@inproceedings{activeMRC,
    title = "Uncertainty-Based Active Learning for Reading Comprehension",
    author = "Jing Wang and Jie Shen and Xiaofei Ma and Andrew Arnold",
    booktitle = "Transactions on Machine Learning Research",
    year = "2022",
    url = "https://openreview.net/forum?id=QaDevCcmcg",
}
```
