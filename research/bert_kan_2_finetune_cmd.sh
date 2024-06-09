#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 research/bert_kan_2_finetune_multiprocessing.py \
    --training_data_file_path data/IWSLT/formatted/train2012 \
    --validation_data_file_path data/IWSLT/formatted/dev2012 \
    --min_sequence_length 64 --max_sequence_length 160 \
    --model_weight_name bert-large-uncased \
    --tokenizer_name bert-large-uncased \
    --model BERT_KAN_2 \
    --batch_size 64 \
    --model_storage_dir models/pykan_bert_large_2/0609 \
    --tensorboard_log_dir runs/pykan_bert_large_2/0609 \
    --label2id "{\"O\": 0, \"COMMA\": 1, \"PERIOD\": 2, \"QUESTION\": 3}" \
    --additional_model_config "{\"dropout\": 0.3, \"attention_dropout\": 0.3}" \
    --epoch 60 \
    --early_stop_count 5 \
