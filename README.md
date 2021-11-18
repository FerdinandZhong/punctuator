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