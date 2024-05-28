# OwLore

This repo contains the pre-release version of OwLore algorithm, proposed by [OwLore: Outlier-weighed Layerwise Sampled Low-Rank Projection for Memory-Efficient LLM Fine-tuning](https://github.com/pixeli99/OwLore).

Outlier-weighed Layerwise Sampled Low-Rank Projection (OwLore) is a novel memory-efficient LLM fine-tuning approach, enhances fine-tuning performance by using layerwise sampling and gradient low-rank training.

<div align="center">
  <img src="https://github.com/pixeli99/OwLore/assets/46072190/fb60054b-7af1-4aa0-9cc8-329c0f96d093" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>

## Abstract

The rapid advancements in Large Language Models (LLMs) have revolutionized various natural language processing tasks. However, the substantial size of LLMs presents significant challenges in training or fine-tuning. While parameter-efficient approaches such as low-rank adaptation (LoRA) have gained popularity, they often compromise performance compared to full-rank fine-tuning. In this paper, we propose Outlier-weighed Layerwise Sampled Low-Rank Projection (OwLore), a new memory-efficient fine-tuning approach, inspired by the layerwise outlier distribution of LLMs, which dynamically samples pre-trained layers to fine-tune instead of adding additional adaptors. We first interpret the outlier phenomenon through the lens of Heavy-Tailed Self-Regularization theory (HT-SR), discovering that layers with more outliers tend to be more heavy-tailed and consequently better trained. Inspired by this finding, OwLore strategically assigns higher sampling probabilities to layers with more outliers to better leverage the knowledge stored in pre-trained LLMs. To further mitigate the memory demands of fine-tuning, we integrate gradient low-rank projection into our approach, which facilitates each layer to be efficiently trained in a low-rank manner. By incorporating the efficient characteristics of low-rank and optimal layerwise sampling, OwLore significantly improves the memory-performance trade-off in LLM pruning. Our extensive experiments across various architectures, including LLaMa2, LLaMa3, and Mistral, demonstrate that OwLore consistently outperforms baseline approaches, including full fine-tuning. Specifically, it achieves up to a 1.1% average accuracy gain on the Commonsense Reasoning benchmark, a 3.0% improvement on MMLU, and a notable 10% boost on MT-Bench, while being more memory efficient. OwLore allows us to fine-tune LLaMa2-7B with only 21GB of memory.

## Quick Start

### Setup

Our repository is built on top of [LMFlow](https://github.com/OptimalScale/LMFlow). You can configure the environment using the following command lines:
```bash
conda create -n owlore python=3.9 -y
conda activate owlore
conda install mpi4py
bash install.sh
pip install peft
```

### Prepare Dataset

You can download our processed datasets from Hugging Face [here](https://huggingface.co/datasets/pengxiang/OwLore_Dataset).

### Finetuning Examples

--- 
We provide a quick overview of the arguments:
- `--model_name_or_path`: The identifier for the model on the Hugging Face model hub.
- `--lisa_activated_layers`: Specifies the number of layers to activate at each step during training.
- `--lisa_interval_steps`: Indicates the number of steps after which resampling occurs.
- `--lisa_prob_mode`: Defines the method used to determine the sampling probability, which can include options such as `uniform`, `owl`, `decrease`, `increase`, etc.
- `--galore`: Indicates whether to use GaLore as the optimizer.

#### Commonsense Reasoning
The script will run LISA on the `Commonsense Reasoning` dataset.
```bash
bash owlore_scripts/run_lisa.sh merge # LISA
```
The script will run OwLore on the `Commonsense Reasoning` dataset.
```bash
bash owlore_scripts/run_owlore_low_rank.sh merge # OwLore
```

#### MMLU
The script will run LISA on the `MMLU` dataset.
```bash
bash owlore_scripts/run_lisa.sh mmlu # LISA
```
The script will run OwLore on the `MMLU` dataset.
```bash
bash owlore_scripts/run_owlore_low_rank.sh mmlu # OwLore
```

#### GSM8K
The script will run LISA on the `GSM8k` dataset.
```bash
bash owlore_scripts/run_lisa.sh gsm # LISA
```
The script will run OwLore on the `GSM8k` dataset.
```bash
bash owlore_scripts/run_owlore_low_rank.sh gsm # OwLore
```

### Evaluation

We use [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to obtain evaluation results. Please refer to its installation instructions to configure `lm_eval`. The steps are as follows:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

After setting up the environment, use the following command to run the evaluation:

#### MMLU
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=/path/to/model\
    --tasks mmlu \
    --output_path mmlu_results \
    --num_fewshot 5 \
    --batch_size auto \
    --cache_requests true
```
#### Commonsense Reasoning
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=/path/to/model\
    --tasks boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
    --output_path qa_results \
    --num_fewshot 5 \
    --batch_size auto \
    --cache_requests true
```
#### GSM8K
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=/path/to/model\
    --tasks gsm8k \
    --output_path math_results \
    --batch_size auto \
    --cache_requests true
```

### Acknowledgement
This repository is build upon the [LMFlow](https://github.com/OptimalScale/LMFlow) and [OWL](https://github.com/luuyin/OWL) repositories. Thanks for their great work!

## Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```

```