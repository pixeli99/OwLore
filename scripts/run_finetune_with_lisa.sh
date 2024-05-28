#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses arguments
model_name_or_path=meta-llama/Llama-2-7b-hf
dataset_path=data/alpaca/train
output_dir=output_models/finetune_lisa
lisa_activated_layers=1
lisa_interval_steps=20
lisa_prob_mode="uniform"
galore=False

# Other optional arguments that can improve memory saving
gradient_checkpointing=True
use_flash_attention=1
gradient_accumulation_steps=1
block_size=512
per_device_train_batch_size=1
seed=0

# Enable model parallelism for multiple gpus, modify this if you prefer
# customized deepspeed zero-redundancy optimization settings
num_gpu=$(python -c "import torch; print(torch.cuda.device_count())")
ds_config_file=configs/ds_config_zero0_no_offload.json
if [ ${num_gpu} -ge 2 ]; then
  ds_config_file=configs/ds_config_zero2_no_offload.json
fi

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --lisa_activated_layers)
      lisa_activated_layers="$2"
      shift
      ;;
    --lisa_interval_steps)
      lisa_interval_steps="$2"
      shift
      ;;
    --lisa_prob_mode)
      lisa_prob_mode="$2"
      shift
      ;;
    --galore)
      galore="$2"
      shift
      ;;
    --gradient_checkpointing)
      gradient_checkpointing="$2"
      shift
      ;;
    --deepspeed)
      ds_config_file="$2"
      shift
      ;;
    --use_flash_attention)
      use_flash_attention="$2"
      shift
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      shift
      ;;
    --block_size)
      block_size="$2"
      shift
      ;;
    --per_device_train_batch_size|--batch_size)
      per_device_train_batch_size="$2"
      shift
      ;;
    --run_name)
      run_name="$2"
      shift
      ;;
    --seed)
      seed="$2"
      shift
      ;;
    --learning_rate)
      learning_rate="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done
echo "LISA Probability Mode: ${lisa_prob_mode}"
# Finetune
exp_id=finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

accelerate launch --main_process_port=29500 examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1.0 \
    --learning_rate ${learning_rate} \
    --disable_group_texts 0 \
    --block_size ${block_size} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --bf16 \
    --torch_dtype bfloat16 \
    --run_name ${run_name}_${learning_rate} \
    --optim adamw_hf \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 50000000000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing ${gradient_checkpointing} \
    --use_flash_attention ${use_flash_attention} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --use_lisa 1 \
    --lisa_activated_layers ${lisa_activated_layers} \
    --lisa_interval_steps ${lisa_interval_steps} \
    --lisa_prob_mode ${lisa_prob_mode} \
    --seed ${seed} \
    --galore ${galore} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
