# Distilbert-punctuator

## Introduction
Distilbert-punctuator is a python package provides a bert-based punctuator (fine-tuned model of `pretrained huggingface DistilBertForTokenClassification`) with following three components:

* **data process**: funcs for processing user's data to prepare for training. If user perfer to fine-tune the model with his/her own data.
* **training**: training pipeline. User can fine-tune his/her own punctuator with the pipeline
* **inference**: easy-to-use interface for user to use trained punctuator. If user doesn't want to train a punctuator himself/herself, a pre-fined-tuned model from huggingface model hub `Qishuai/distilbert_punctuator_en` can be used when launching the inference


## Data Process
Component for pre-processing the training data. To use this component, please install as `pip install distilbert-punctuator[data_process]`

The package is providing a simple pipeline for you to generate `NER` format training data.

### Example
`examples/data_sample.py`

## Train
Component for providing a training pipeline for fine-tuning a pretrained `DistilBertForTokenClassification` model from `huggingface`.

### Example
`examples/train_sample.py`

### Training_arguments:
Arguments required for the training pipeline.

`data_file_path(str)`: path of training data
`model_name(str)`: name or path of pre-trained model
`tokenizer_name(str)`: name of pretrained tokenizer
`split_rate(float)`: train and validation split rate
`sequence_length(int)`: sequence length of one sample
`epoch(int)`: number of epoch
`batch_size(int)`: batch size
`model_storage_path(str)`: fine-tuned model storage path
`tag2id_storage_path(str)`: tag2id storage path
`addtional_model_config(Optional[Dict])`: additional configuration for model

## Inference
Component for providing an inference interface for user to use punctuator.

### Architecture
```
 +----------------------+              (child process)
 |   user application   |             +-------------------+
 +                      + <---------->| punctuator server |
 |   +inference object  |             +-------------------+
 +----------------------+
```

The punctuator will be deployed in a child process which communicates with main process through pipe connection.
Therefore user can initialize an inference object and call its `punctuation` function when needed. The punctuator will never block the main process unless doing punctuation.
There is a `graceful shutdown` methodology for the punctuator, hence user dosen't need to worry about the shutting-down.

### Example
`examples/inference_sample.py`

### Inference_arguments
Arguments required for the inference pipeline.

`model_name_or_path(str)`: name or path of pre-trained model
`tokenizer_name(str)`: name of pretrained tokenizer
`tag2id_storage_path(Optional[str])`: tag2id storage path. If None, DEFAULT_TAG_ID will be used.

`DEFAULT_TAG_ID`: {"E": 0, "O": 1, "P": 2, "C": 3, "Q": 4}