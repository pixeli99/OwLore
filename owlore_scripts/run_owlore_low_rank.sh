export CUDA_VISIBLE_DEVICES=0

lisa_activated_layers=5
data_dir=$1
seed=111
model_type=llama2
learning_rate=3e-4

MODEL_PATH="/llama2-7b"
DATASET_PATH="data/${data_dir}"
OUTPUT_MODEL_PATH="/output_models/finetuned_${model_type}_owlore_${data_dir}_owl_seed_${seed}_inter${lisa_activated_layers}_${learning_rate}"

./scripts/run_finetune_with_lisa.sh \
  --model_name_or_path ${MODEL_PATH} \
  --dataset_path ${DATASET_PATH} \
  --output_model_path ${OUTPUT_MODEL_PATH} \
  --lisa_activated_layers ${lisa_activated_layers} \
  --lisa_interval_steps 50 \
  --lisa_prob_mode owl \
  --per_device_train_batch_size 16 \
  --galore True \
  --seed ${seed} \
  --gradient_accumulation_steps 1 \
  --run_name ${seed}_OwLore_${data_dir}_${model_type}_${lisa_activated_layers} \
  --learning_rate ${learning_rate}
