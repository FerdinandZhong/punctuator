import os
from punctuator.llm_evaluation.pipeline import (
    generate_llm_results_bert_hint, 
    evaluate_llm_output, 
    generate_llm_results_directly
)

from punctuator.llm_evaluation.constants import PUNCTSPECIAL2LABLE, LLM_CHAT_MESSAGES_BERT_OUTPUT_5_EN_SAMPLE, LLM_CHAT_MESSAGES_RAW_5_EN_SAMPLE, LLM_CHAT_MESSAGES_RAW
from punctuator.llm_evaluation.dataset_utils import clean_up_data_from_txt, normalize_puncs
from functools import partial
from punctuator.utils import chinese_split, remove_brackets_text
from punctuator.llm_evaluation.pipeline import evaluate_llm_output
from openai import AsyncOpenAI
import asyncio

from punctuator.llm_evaluation.dataset_utils import (
    read_data_to_w_special_token,
)

pure_tokens_gt, _, pure_labels_gt = read_data_to_w_special_token(
    source_data = "/Users/qishuai.zhong/Projects/punctuator/data/newly_code_switching/test.txt",
    bert_output="/Users/qishuai.zhong/Projects/punctuator/data/new_code_switching_binary/roberta_binary_output_test.txt",
    min_sequence_length=512,
    max_sequence_length=512,
    punct_special_token="#",
)

openai_client = AsyncOpenAI(
    api_key=os.environ["api_key"]
)

PUNCT2LABEL = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTIONMARK",
    "!": "EXCLAMATIONMARK",
    "#": "O",
}

CROSS_LANG_PUNCT_MAPPINGS = {
    ":": ",",
    "？": "?",
    "！": "!",
    "。": ".",
    "、": ",",
    "，": ",",
    ": “": ",",
    "（": "(",
    "）": ")",
}

def process_results_gpt(raw_file, output_file):
    with open(raw_file, "r", encoding="utf-8") as results_file:
        generated_sentences = results_file.readlines()

    processed_results = clean_up_data_from_txt(
        generated_sentences,
        output_file,
        target_punctuations=PUNCT2LABEL.keys(),
        additional_to_keep=["'", "-", "—"],
        additional_to_remove=["℃", "|", "♫"],
        special_cleaning_funcs=[
            partial(normalize_puncs, normalization=CROSS_LANG_PUNCT_MAPPINGS),
            chinese_split,
            remove_brackets_text,
        ],
    )
    return processed_results

async def main():
    # _, pure_labels_gt = await generate_llm_results_directly(
    #     source_file_path="/Users/qishuai.zhong/Projects/punctuator/data/newly_code_switching/test.txt",
    #     bert_output="/Users/qishuai.zhong/Projects/punctuator/data/new_code_switching_binary/roberta_binary_output_test.txt",
    #     target_model="gpt-4o",
    #     openai_client=openai_client,
    #     min_sequence_length=512,
    #     max_sequence_length=512,
    #     raw_output_file_path="/Users/qishuai.zhong/Projects/punctuator/data/llm_results/gpt_4o_direct_raw.txt",
    #     processed_output_file_path="/Users/qishuai.zhong/Projects/punctuator/data/llm_results/gpt_4o_direct_processed.txt",
    #     chunk_size=20,
    #     chat_messages=LLM_CHAT_MESSAGES_RAW
    # )

    gpt_4o_direct_processed_results = process_results_gpt(
        "/Users/qishuai.zhong/Projects/punctuator/data/llm_results/gpt_4o_direct_raw.txt",
        "/Users/qishuai.zhong/Projects/punctuator/data/llm_results/gpt_4o_direct_processed_new.txt"
    )
    evaluate_llm_output(
        processed_results=gpt_4o_direct_processed_results,
        pure_tokens_gt=pure_tokens_gt, 
        pure_labels_gt=pure_labels_gt,
        target_model="gpt-4o"
    )
    

if __name__ == "__main__":
    asyncio.run(main())