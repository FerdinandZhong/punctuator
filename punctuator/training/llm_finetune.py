import json
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union

import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments
from transformers.data import DataCollatorForSeq2Seq

from punctuator.utils import Models, model_type

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


def compute_precision_recall(predictions, labels, token_id):
    # Convert to NumPy arrays if not already
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Identify where predictions and labels are equal to the specific token
    pred_token_positions = predictions == token_id
    label_token_positions = labels == token_id

    # True positives: The token is correctly predicted
    true_positives = np.sum(pred_token_positions & label_token_positions)

    # Predicted positives (where the model predicted the specific token)
    predicted_positives = np.sum(pred_token_positions)

    # Actual positives (where the true label is the specific token)
    actual_positives = np.sum(label_token_positions)

    # Calculate precision and recall
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    return precision, recall


def compute_metrics_for_position(
    eval_pred, compute_result=True, specific_token_id=None
):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)  # Convert logits to predicted ids

    # Flatten the outputs and labels for simpler comparison
    flat_predictions = predictions.detach().cpu().numpy().flatten()
    flat_labels = labels.detach().cpu().numpy().flatten()

    # Compute precision and recall for the specific token
    precision, recall = compute_precision_recall(
        flat_predictions, flat_labels, specific_token_id
    )

    return {"precision": precision, "recall": recall}


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


logger = logging.getLogger(__name__)


llm_instructions = {
    Models.QWEN2.value: (
        "<|im_start|>system\n"
        + "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.<|im_end|>\n"  # noqa E501
        + "{instruction}"
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


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, BasicArguments))
    training_args, basic_args = parser.parse_args_into_dataclasses()

    model_collection = model_type(basic_args.model_type).value

    tokenizer = model_collection.tokenizer.from_pretrained(
        basic_args.tokenizer_name, **basic_args.additional_tokenizer_config
    )
    if basic_args.additional_special_tokens:
        tokenizer.add_special_tokens(basic_args.additional_special_tokens)
    model = model_collection.model.from_pretrained(
        basic_args.tokenizer_name, **basic_args.additional_model_config
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shifted_label_dataset["train"],
        eval_dataset=shifted_label_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics_for_position,
            specific_token_id=tokenizer.additional_special_tokens_ids[0],
        ),
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
