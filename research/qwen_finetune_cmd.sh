export 

python punctuator/training/llm_finetune.py --model_name Qwen/Qwen2-1.5B-Instruct \
    --tokenizer_name Qwen/Qwen2-1.5B-Instruct \
    --dataset_dir "data/llm_datasets/special_token" \
    --model_type QWEN2 \
    --additional_special_tokens "{\"additional_special_tokens\": [\"@P\"]}" \
    --additional_tokinezer_config "{\"padding_side\": \"left\", \"truncation_side\": \"left\"}" \
    --deepspped  research/qwen_finetune_dp_config.json \
    --report_to tensorboard \
    --logging_dir runs/llm_finetune/qwen2/special_token \
    --output_dir models/llm_finetune/qwen2/special_token \
    --load_best_model_at_end \
    --eval_strategy steps \
    --save_strategy steps \
    --num_ 50000 \
    --batch_eval_metrics \
    
