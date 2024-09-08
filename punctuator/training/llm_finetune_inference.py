import json
import logging
import warnings
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from punctuator.utils import model_type

## TODO: may include the evaluation of the fine-tuned results

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)

logger = logging.getLogger(__name__)


_VALID_DICT_FIELDS = []


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
        batch_size (int)
        max_new_tokens (int)
    """

    model_path: str = field(
        metadata={
            "help": "Either the remote pretrainded model name or local model dir"
        },
    )
    model_type: str = field(metadata={"help": "Model type defined in model zoo"})

    tokenizer_name: str = field(
        metadata={"help": "Tokenizer name or dir"},
    )
    output_path: str = field(
        metadata={"help": "Output file path"},
    )
    dataset_dir: str = field(
        metadata={"help": "Dataset directory containing fields"},
        default="data/llm_datasets/special_token_#_new_all_lower/test.jsonl",
    )
    
    batch_size: int = field(metadata={"help": "Batch size"}, default=4)
    max_new_tokens: int = field(metadata={"help": "max new tokens"}, default=1024)

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


chat_messages = [
    {
        "role": "system",
        "content": "Insert {token} after each English word or Chinese character where punctuation is required. Keep all other tokens unchanged.",
    },
]


def process_inputs(sample, launched_tokenizer):
    full_input = launched_tokenizer.apply_chat_template(
        sample.pop("chat_messages"), tokenize=False, add_generation_prompt=True
    )
    tokenized_input = launched_tokenizer(full_input)
    prompt_input_ids = tokenized_input["input_ids"]
    sample["input_ids"] = prompt_input_ids
    sample.pop("output")
    return sample


if __name__ == "__main__":
    parser = HfArgumentParser((BasicArguments,))
    basic_args = parser.parse_args_into_dataclasses()

    model_collection = model_type(basic_args.model_type).value

    tokenizer = model_collection.tokenizer.from_pretrained(
        basic_args.tokenizer_name,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model_collection.model.from_pretrained(
        basic_args.tokenizer_name,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)

    model.to(device)
    model.eval()

    dataset = load_dataset("json", data_dir=basic_args.dataset_dir)

    processed_dataset = dataset.map(
        process_inputs,
        fn_kwargs={
            "launched_tokenizer": tokenizer,
        },
    )

    data_collator = DataLoader(processed_dataset, batch_size=basic_args.batch_size)

    file_writer = open(basic_args.output_file_path, "w", encoding="utf-8")

    with tqdm(total=len(data_collator)) as pbar:
        for batch in data_collator:
            padded_batch = pad_without_fast_tokenizer_warning(
                tokenizer, batch, padding="longest"
            )

            outputs = model.generate(
                input_ids=padded_batch["input_ids"].to(device),
                max_new_tokens=basic_args.max_new_tokens,
            )
            decoded_outputs = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )
            for output in decoded_outputs:
                file_writer.write(repr(output) + "\n")

            pbar.update(1)

        file_writer.close()
