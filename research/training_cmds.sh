#!/bin/bash

# Bert_Kan
python research/bert_kan_2_finetune_focal_loss.py \
    --training_data_file_path data/IWSLT/formatted/train2012 \
    --validation_data_file_path data/IWSLT/formatted/dev2012 \
    --min_sequence_length 32 --max_sequence_length 160 \
    --model_weight_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --model BERT_KAN_FOCAL_LOSS \
    --batch_size 32 \
    --model_storage_dir models/pykan_bert_base_focal_loss/0612 \
    --tensorboard_log_dir runs/pykan_bert_base_focal_loss/0612 \
    --label2id "{\"O\": 0, \"COMMA\": 1, \"PERIOD\": 2, \"QUESTION\": 3}" \
    --additional_model_config "{\"dropout\": 0.3, \"attention_dropout\": 0.3}" \
    --epoch 60 \
    --early_stop_count 10 \
    --warm_up_steps 500 \
    --use_class_weight true 2&>1 | tee logs/pykan_bert_base_focal_loss/0612.txt


# 
# --additional_model_config "{\"rotary_value\": false, \"embedding_size\": 768, \"max_position_embeddings": 42}" \