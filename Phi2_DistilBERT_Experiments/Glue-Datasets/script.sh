python process_glue_dataset.py \
    --task_name "rte" \
    --output_dir "glue_datasets"

python tokenize_glue_dataset.py \
    --task_name "rte" \
    --output_dir "glue_datasets" \
    --model_name_or_path microsoft/deberta-v3-base