python process_glue_dataset.py \
    --task_name "cola" \
    --output_dir "glue_datasets"

python tokenize_glue_dataset.py \
    --task_name "rte" \
    --output_dir "glue_datasets" \
    --model_name_or_path NousResearch/Llama-2-7b-hf

for TASK in mnli qnli qqp
do
  echo "Processing task: $TASK"
  python process_glue_dataset.py --task_name "$TASK" --output_dir "glue_datasets"
done

python intrinsic_dims.py --task_name "wnli"

for TASK in rte cola mrpc stsb wnli
do
  echo "Processing task: $TASK"
  python tokenize_glue_dataset.py --task_name "$TASK" --output_dir "glue_datasets" --model_name_or_path NousResearch/Llama-2-7b-hf
done

python finetune.py \
    --task_name "wnli" \
    --output_dir "glue_datasets" \
    --model_name_or_path NousResearch/Llama-2-7b-hf

torchrun --nproc_per_node=2 finetune.py \
    --task_name "wnli" \
    --output_dir "glue_datasets" \
    --model_name_or_path NousResearch/Llama-2-7b-hf
