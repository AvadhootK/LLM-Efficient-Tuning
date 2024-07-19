## Help for running this directory

This directoy contains modified files from SoRA repository. 
Steps to run experiments:
1. Clone SoRA github directory and replace the files form this repo with respective files.
2. Download the dependencies
  - what worked for me is downloading the modified requirements.txt first.
  - then download the requirements2.txt
  - update datasets and transformers

3. Example script to run experiments:
  - python -u run_glue_updated.py --do_eval --do_predict --do_train --task_name rte --eval_steps 1000 --evaluation_strategy steps --greater_is_better true --learning_rate 1.2e-3 --max_grad_norm 0.1 --load_best_model_at_end --logging_steps 100 --max_steps -1 --model_name_or_path microsoft/deberta-v3-base --num_train_epochs 1 --output_dir results/rte-lambda2_0_7e-4_lambda_0.001_epoch_1_seed_48_1 --overwrite_output_dir --per_device_eval_batch_size 1 --per_device_train_batch_size 1 --save_steps 1000 --save_strategy steps --save_total_limit 1 --warmup_ratio 0.06 --warmup_steps 0 --weight_decay 0.1 --disable_tqdm false --sparse_lambda 0.001 --sparse_lambda_2 0 --seed 48 --lora_r 8 --max_seq_length 256 --max_lambda 7e-4 --lambda_schedule linear --lambda_num 1 --train_sparse --lora_method sora 

for data samples range:
--max_train_samples 4
--max_val_samples 2
--max_pred_samples 3

for lora:
--lora_method lora --apply_lora

for sora:
--lora_method sora --train_sparse
