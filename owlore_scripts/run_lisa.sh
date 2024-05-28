export CUDA_VISIBLE_DEVICES=0

lisa_activated_layers=2
MODE=uniform
data_dir=$1
seed=111
interval=3
learning_rate=3e-4

MODEL_PATH="/llama2-7b"
DATASET_PATH="data/${data_dir}"
OUTPUT_MODEL_PATH="/output_models/finetuned_llama2_${MODE}_${data_dir}_seed_${seed}_inter${interval}_${learning_rate}_${lisa_activated_layers}"

./scripts/run_finetune_with_lisa.sh \
  --model_name_or_path ${MODEL_PATH} \
  --dataset_path ${DATASET_PATH} \
  --output_model_path ${OUTPUT_MODEL_PATH} \
  --lisa_activated_layers ${lisa_activated_layers} \
  --lisa_interval_steps ${interval} \
  --lisa_prob_mode ${MODE} \
  --per_device_train_batch_size 16 \
  --seed ${seed} \
  --gradient_accumulation_steps 1 \
  --run_name ${seed}_${MODE}_LISA_${data_dir}_llama_inter${interval}_${lisa_activated_layers} \
  --learning_rate ${learning_rate}
