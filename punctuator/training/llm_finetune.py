import json
import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.data import DataCollatorForSeq2Seq
from torch.nn import CrossEntropyLoss

from punctuator.utils import Models, model_type

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)

logger = logging.getLogger(__name__)


_VALID_DICT_FIELDS = [
    "additional_special_tokens",
    "additional_tokenizer_config",
    "additional_model_config",
]


def _convert_str_dict(passed_value: dict):
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            # First check for bool and convert
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            # Check for digit
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                passed_value[key] = float(value)

    return passed_value


class MetricsAccumulator:
    def __init__(self):
        self.reset()

    def update(self, new_metrics):
        for key, value in new_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

    def compute_summary(self):
        summary = {}
        for key, values in self.metrics.items():
            summary[key] = sum(values) / len(values)  # Average for simplicity
        return summary

    def reset(self):
        self.metrics = {}


def compute_metrics(
    eval_pred,
    compute_result=True,
    metrics_accumulator=None,
    specific_tokens_ids=[],
    specific_tokens=[],
    tokenizer=None,
):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)  # Convert logits to predicted ids

    # Flatten the outputs and labels for simpler comparison
    flat_predictions = predictions.cpu().numpy().flatten()
    flat_labels = labels.cpu().numpy().flatten()

    reduce_ignored = flat_labels >= 0
    true_labels = flat_labels[reduce_ignored]  # remove ignored -100
    true_preds = flat_predictions[reduce_ignored]

    accuracy = accuracy_score(flat_labels, flat_predictions)

    report = classification_report(
        true_labels,
        true_preds,
        labels=specific_tokens_ids,
        digits=4,
        target_names=specific_tokens,
        zero_division=1,
        output_dict=True,
    )
    
    pred_text = tokenizer.batch_decode(true_preds, skip_special_tokens=False)
    label_text = tokenizer.batch_decode(true_labels, skip_special_tokens=False)
    
    min_len = min(len(pred_text), len(label_text))
    
    # Calculate the number of matching words in the range of the shorter sentence
    matches = sum(1 for w1, w2 in zip(pred_text[:min_len], label_text[:min_len]) if w1 == w2)
    
    # Compute accuracy based on the shorter sentence
    text_accuracy = matches / min_len if min_len else 0
    
    if np.random.rand() < 0.005:  # Roughly once per 200 calls
        print(
            "Text Preds:", "".join(pred_text)
        )
        print(
            "Text Labels:", "".join(label_text)
        )
        print(
            f"Shape of labels: {true_labels.shape} ---- Shape of preds: {true_preds.shape}"
        )
        print("validation report: \n %s", report)

    result = {}
    for token in specific_tokens:
        metrics_details = report[token]
        for key, value in metrics_details.items():
            if key in ["precision", "recall", "f1-score"]:
                result[f"{token}_{key}"] = value

    # result.update(
    #     {f"micro_avg_{key}": value for key, value in report["micro avg"].items()}
    # )
    result["text_accuracy"] = text_accuracy

    result["overal_accuracy"] = accuracy

    if not compute_result:
        metrics_accumulator.update(result)
    else:
        result = metrics_accumulator.compute_summary()
        metrics_accumulator.reset()
    return result


@dataclass
class BasicArguments:
    """
    Basic arguments for the training

    Parameters:
        model_name (str)
        tokenizer_name (str)
        dataset_dir (str)
        model_type (str)
        additional_special_tokens (dict)
        additional_tokenizer_config (dict)
        additional_model_config (dict)
        specific_tokens (list)
        compute_loss_in_chunk (bool)
    """

    model_name: str = field(
        metadata={
            "help": "Either the remote pretrainded model name or local model dir"
        },
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer name or dir"},
    )
    dataset_dir: str = field(metadata={"help": "Dataset directory containing fields"})
    model_type: str = field(metadata={"help": "Model type defined in model zoo"})
    additional_special_tokens: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={"help": "additional special tokens to attach to tokenizer"},
    )
    additional_tokenizer_config: Optional[Union[dict, str]] = field(
        default_factory=dict, metadata={"help": "additional config for tokenizer"}
    )
    additional_model_config: Optional[Union[dict, str]] = field(
        default_factory=dict, metadata={"help": "additional config for model"}
    )
    specific_tokens: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "specific tokens for evaluation metrics"},
    )
    compute_loss_in_chunk : bool = field(
        default=False,
        metadata={"help": "Whether to compute loss in chunk"},
    )

    def __post_init__(self):
        for field in _VALID_DICT_FIELDS:
            passed_value = getattr(self, field)
            # We only want to do this if the str starts with a bracket to indiciate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, field, loaded_dict)


llm_instructions = {
    Models.QWEN2.value: (
        # "<|im_start|>system\n"
        # + "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.<|im_end|>\n"  # noqa E501
        "{instruction}"
        + "<|im_start|>user\n{input}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
}


def shift_labels(sample, launched_tokenizer, llm_instruction):
    full_input = llm_instruction.format(
        instruction=sample["instruction"], input=sample["input"]
    )
    tokenized_input = launched_tokenizer(full_input)
    input_attention_mask = tokenized_input["attention_mask"]
    prompt_input_ids = tokenized_input["input_ids"]
    full_output = sample["output"] + launched_tokenizer.eos_token
    tokenized_output = launched_tokenizer(full_output)
    output_ids = tokenized_output["input_ids"]
    output_attention_mask = tokenized_output["attention_mask"]
    sample["input_ids"] = prompt_input_ids + output_ids
    sample["labels"] = [-100] * len(prompt_input_ids) + output_ids
    sample["attention_mask"] = input_attention_mask + output_attention_mask
    sample.pop("instruction")
    sample.pop("input")
    sample.pop("output")
    return sample


class CustomTrainer(Trainer):
    def _chunk_loss(self, logits, labels):
        _, seq_length = labels.shape # dim 0 is the batch_size

        # Prepare to calculate loss every 5 tokens
        loss_fct = CrossEntropyLoss()
        total_loss = 0.0

        # Calculate loss for every 5-token segment in each sequence in the batch
        step = 5
        for i in range(0, seq_length - step + 1, step):
            # Only consider the segment if it's full (i.e., has 'step' tokens)
            if i + step <= seq_length:
                # Slice to get the segment of logits and corresponding labels
                logits_segment = logits[:, i:i+step, :].contiguous()
                labels_segment = labels[:, i:i+step].contiguous()

                # Reshape for loss calculation
                logits_flat = logits_segment.view(-1, logits_segment.size(-1))
                labels_flat = labels_segment.view(-1)

                # Calculate and accumulate the loss
                segment_loss = loss_fct(logits_flat, labels_flat)
                total_loss += segment_loss

        # Average the loss over the number of segments
        total_loss /= (seq_length // step)

        return total_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        logits = outputs.logits
        labels = inputs["labels"]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        chunk_loss = self._chunk_loss(logits, labels)
        
        total_loss = chunk_loss + loss
        return (total_loss, outputs) if return_outputs else total_loss
    

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, BasicArguments))
    training_args, basic_args = parser.parse_args_into_dataclasses()

    model_collection = model_type(basic_args.model_type).value

    tokenizer = model_collection.tokenizer.from_pretrained(
        basic_args.tokenizer_name, **basic_args.additional_tokenizer_config
    )
    model = model_collection.model.from_pretrained(
        basic_args.tokenizer_name, **basic_args.additional_model_config
    )
    if basic_args.additional_special_tokens:
        tokenizer.add_special_tokens(basic_args.additional_special_tokens)
        model.resize_token_embeddings(len(tokenizer))

        index_of_dot = tokenizer.convert_tokens_to_ids(".")
        for new_token in basic_args.additional_special_tokens:
            index_of_new = tokenizer.convert_tokens_to_ids(new_token)
            with torch.no_grad():
                model.model.embed_tokens.weight[index_of_new] = (
                    model.model.embed_tokens.weight[index_of_dot].clone()
                )

    model.to(training_args.device)

    dataset = load_dataset("json", data_dir=basic_args.dataset_dir)
    
    shifted_label_dataset = dataset.map(
        shift_labels,
        fn_kwargs={
            "launched_tokenizer": tokenizer,
            "llm_instruction": llm_instructions[model_collection],
        },
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")

    specific_tokens_ids = []
    for token in basic_args.specific_tokens:
        specific_tokens_ids.append(tokenizer.convert_tokens_to_ids(token))

    print(specific_tokens_ids)
    print(basic_args.specific_tokens)

    metrics_accumulator = MetricsAccumulator()
    if basic_args.compute_loss_in_chunk:
        trainer_cls = CustomTrainer
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=shifted_label_dataset["train"],
        eval_dataset=shifted_label_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics,
            metrics_accumulator=metrics_accumulator,
            specific_tokens_ids=specific_tokens_ids,
            specific_tokens=basic_args.specific_tokens,
            tokenizer=tokenizer
        ),
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
