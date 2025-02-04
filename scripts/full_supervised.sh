python3 full_supervised.py \
    --model_name_or_path t5-small \
    --tokenizer_name_or_path t5-small \
    --max_input_length 32 \
    --max_target_length 32 \
    --task_type T2A \
    --absa_tuple cp \
    --annotations_text_type gas_extraction_style \
    --text_column text \
    --annotations_column annotation \
    --annotations_raw_format cp \
    --annotation_flag \
    --dataset_path /home/b.kabongo/datasets/mams_acsa/data.csv \
    --exp_dir /home/b.kabongo/exps/cycle_absa/mams_acsa/full_supervised_t2a/t5_small \
    --train_size 0.8 \
    --val_size 0.1 \
    --test_size 0.1 \
    --lang en \
    --verbose \
    --verbose_every 1 \
    --random_state 42 \
    --n_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --save_every 1 \
    --train_flag \
    --save_eval_results \
    --lower_flag; \
python3 full_supervised.py \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    --max_input_length 32 \
    --max_target_length 32 \
    --task_type T2A \
    --absa_tuple cp \
    --annotations_text_type gas_extraction_style \
    --text_column text \
    --annotations_column annotation \
    --annotations_raw_format cp \
    --annotation_flag \
    --dataset_path /home/b.kabongo/datasets/mams_acsa/data.csv \
    --exp_dir /home/b.kabongo/exps/cycle_absa/mams_acsa/full_supervised_t2a/t5_base \
    --train_size 0.8 \
    --val_size 0.1 \
    --test_size 0.1 \
    --lang en \
    --verbose \
    --verbose_every 1 \
    --random_state 42 \
    --n_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --save_every 1 \
    --train_flag \
    --save_eval_results \
    --lower_flag; \
python3 full_supervised.py \
    --model_name_or_path t5-small \
    --tokenizer_name_or_path t5-small \
    --max_input_length 32 \
    --max_target_length 32 \
    --task_type A2T \
    --absa_tuple cp \
    --annotations_text_type gas_extraction_style \
    --text_column text \
    --annotations_column annotation \
    --annotations_raw_format cp \
    --annotation_flag \
    --dataset_path /home/b.kabongo/datasets/mams_acsa/data.csv \
    --exp_dir /home/b.kabongo/exps/cycle_absa/mams_acsa/full_supervised_a2t/t5_small \
    --train_size 0.8 \
    --val_size 0.1 \
    --test_size 0.1 \
    --lang en \
    --verbose \
    --verbose_every 1 \
    --random_state 42 \
    --n_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --save_every 1 \
    --train_flag \
    --save_eval_results \
    --lower_flag; \
python3 full_supervised.py \
    --model_name_or_path t5-base \
    --tokenizer_name_or_path t5-base \
    --max_input_length 32 \
    --max_target_length 32 \
    --task_type A2T \
    --absa_tuple cp \
    --annotations_text_type gas_extraction_style \
    --text_column text \
    --annotations_column annotation \
    --annotations_raw_format cp \
    --annotation_flag \
    --dataset_path /home/b.kabongo/datasets/mams_acsa/data.csv \
    --exp_dir /home/b.kabongo/exps/cycle_absa/mams_acsa/full_supervised_a2t/t5_base \
    --train_size 0.8 \
    --val_size 0.1 \
    --test_size 0.1 \
    --lang en \
    --verbose \
    --verbose_every 1 \
    --random_state 42 \
    --n_epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --save_every 1 \
    --train_flag \
    --save_eval_results \
    --lower_flag; \