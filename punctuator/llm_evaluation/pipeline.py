import logging
from functools import partial

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from punctuator.data_process.data_cleanning import text_lines_cleaning
from punctuator.utils import ALL_PUNCS, NORMAL_TOKEN_TAG, chinese_split

from .constants import (
    CROSS_LANG_PUNCT_MAPPINGS,
    LABEL2ID,
    LLM_CHAT_MESSAGES,
    PUNCT2LABEL,
    PUNCT_SPECIAL_TOKEN,
)
from .dataset_utils import (
    generate_dataset,
    normalize_puncs,
    process_line,
    read_data_to_w_special_token,
)
from .requests import openai, query_server_in_chunk

EMAIL_TOKEN = "email"
URL_TOKEN = "url"
additional_to_remove = ["℃", "|", "♫"]
logger = logging.getLogger(__name__)


async def generate_llm_results_bert_hint(
    source_file_path,
    bert_output,
    target_model,
    min_sequence_length=32,
    max_sequence_length=160,
    processed_output_file_path=None,
):
    wo_punt, w_bert_output, pure_tokens_gt = read_data_to_w_special_token(
        source_file_path,
        bert_output,
        min_sequence_length,
        max_sequence_length,
        PUNCT_SPECIAL_TOKEN,
    )
    chat_messages_list_bert = generate_dataset(
        chat_messages=LLM_CHAT_MESSAGES,
        input_list=w_bert_output,
        token=PUNCT_SPECIAL_TOKEN,
    )
    logger.info("chat message sample: %s", chat_messages_list_bert[0])

    chat_completion_sample = await openai.chat.completions.create(
        model=target_model, messages=chat_messages_list_bert[0], temperature=0.1
    )

    logger.info(
        "sample output: %s",
        chat_completion_sample.choices[0].message.content.split("\n")[0],
    )

    generated_sentences = await query_server_in_chunk(
        chat_messages_list_bert,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        chunk_size=40,
    )

    kept_punctuations = [ord(p) for p in set(PUNCT2LABEL.keys())]
    removed_punctuations = [p for p in ALL_PUNCS if p not in kept_punctuations] + [
        ord(p) for p in additional_to_remove
    ]

    processed_results = list(
        text_lines_cleaning(
            generated_sentences,
            kept_punctuations,
            removed_punctuations,
            *[
                partial(normalize_puncs, normalization=CROSS_LANG_PUNCT_MAPPINGS),
                chinese_split,
            ],
        )
    )

    if processed_output_file_path is not None:
        with open(processed_output_file_path, "w", encoding="utf-8") as results_file:
            for sentence in processed_results:
                results_file.write(sentence + "\n")

    return processed_results, pure_tokens_gt


def evaluate_llm_output(processed_results, pure_tokens_gt, target_model):
    results_tokens_list = []
    results_labels_list = []
    for line in tqdm(processed_results):
        result_tokens, result_labels = process_line(line, ner_mapping=PUNCT2LABEL)
        results_tokens_list.append(result_tokens)
        results_labels_list.append(result_labels)

    length_not_matched_index = []
    all_gt_labels = []
    all_result_labels = []
    matched_gt_labels = []
    matched_result_labels = []

    for list_index, (test_label, result_labels) in enumerate(
        zip(pure_tokens_gt, results_labels_list)
    ):
        gt_label_ids = [LABEL2ID[label] for label in test_label]
        result_label_ids = [LABEL2ID[label] for label in result_labels]
        all_gt_labels.extend(gt_label_ids)
        all_result_labels.extend(result_label_ids)
        if len(test_label) != len(result_labels):
            length_not_matched_index.append(list_index)
            if np.random.rand() < 0.002:
                logger.info(
                    "correct length: %s | predicted length: %s",
                    len(test_label),
                    len(result_labels),
                )
        else:
            matched_gt_labels.extend(gt_label_ids)
            matched_result_labels.extend(result_label_ids)

    logger.info(
        "not matched percentage: %s",
        round(len(length_not_matched_index / len(pure_tokens_gt))),
    )

    tested_labels = []
    target_names = []
    for label, label_id in LABEL2ID.items():
        if label != NORMAL_TOKEN_TAG:
            tested_labels.append(label_id)
            target_names.append(label)

    report_for_matched = classification_report(
        matched_gt_labels,
        matched_result_labels,
        labels=tested_labels,
        digits=4,
        target_names=target_names,
        zero_division=1,
    )

    logger.info(
        "validation report for %s with bert result matched: \n %s",
        target_model,
        report_for_matched,
    )
