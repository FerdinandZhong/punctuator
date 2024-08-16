import json
import logging
import warnings
from dataclasses import dataclass, field
from functools import partial

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import HfArgumentParser, Trainer, TrainingArguments
from transformers.data import DataCollatorForSeq2Seq

from punctuator.utils import model_type

## TODO: may include the evaluation of the fine-tuned results

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


@dataclass
class BasicArguments:
    """
    Basic arguments for the training

    Parameters:
        model_path (str)
        tokenizer_name (str)
        dataset_dir (str)
    """

    model_path: str = field(
        metadata={
            "help": "Either the remote pretrainded model name or local model dir"
        },
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer name or dir"},
    )
    dataset_dir: str = field(metadata={"help": "Dataset directory containing fields"})

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


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, BasicArguments))
    training_args, basic_args = parser.parse_args_into_dataclasses()

    model_collection = model_type(basic_args.model_type).value

    tokenizer = model_collection.tokenizer.from_pretrained(
        basic_args.tokenizer_name,
        padding_side="left",
        truncation_side="left",
        **basic_args.additional_tokenizer_config,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                model.model.embed_tokens.weight[
                    index_of_new
                ] = model.model.embed_tokens.weight[index_of_dot].clone()

    if basic_args.use_peft:
        with open(basic_args.peft_config, "r") as config_file:
            peft_config_json = json.load(config_file)
        peft_config = LoraConfig(**peft_config_json)
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    model.to(training_args.device)

    dataset = load_dataset("json", data_dir=basic_args.dataset_dir)

    shifted_label_dataset = dataset.map(
        shift_labels,
        fn_kwargs={
            "launched_tokenizer": tokenizer,
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
            tokenizer=tokenizer,
        ),
        data_collator=data_collator,
    )
    trainer.chunk_size = basic_args.chunk_size

    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
