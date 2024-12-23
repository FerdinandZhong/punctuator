#!/bin/bash

python research/evaluation.py \
    --evaluation_data_file_path data/IWSLT/formatted/test2011 \
    --min_sequence_length 32 --max_sequence_length 160 \
    --model_weight_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --model BERT_KAN_FOCAL_LOSS \
    --batch_size 32 \
    --label2id "{\"O\": 0, \"COMMA\": 1, \"PERIOD\": 2, \"QUESTION\": 3}" 2&>1 | tee logs/pykan_bert_base_focal_loss/0612.txt


python research/evaluation.py \
    --evaluation_data_file_path data/newly_code_switching/test.txt \
    --min_sequence_length 512 --max_sequence_length 512 \
    --model_weight_name models/focal_loss/modernbert_focal_loss/1222 \
    --tokenizer_name answerdotai/ModernBERT-large \
    --model MODERNBERT_FOCAL_LOSS \
    --batch_size 8 \
    --label2id "{\"O\": 0, \"COMMA\": 1, \"PERIOD\": 2, \"QUESTIONMARK\": 3, \"EXCLAMATIONMARK\": 4}" 2>&1 | tee logs/focal_loss/modernbert_focal_loss/eval_logs_1222.log
