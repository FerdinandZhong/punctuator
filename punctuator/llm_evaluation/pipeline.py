import logging
import random
from functools import partial

import torch
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from punctuator.utils import NORMAL_TOKEN_TAG, chinese_split, remove_brackets_text

from .constants import (
    CROSS_LANG_PUNCT_MAPPINGS,
    LABEL2ID,
    LLM_CHAT_MESSAGES_BERT_OUTPUT,
    LLM_CHAT_MESSAGES_FIND_POSITION,
    LLM_CHAT_MESSAGES_REPETATION,
    LLM_CHAT_MESSAGES_RAW,
    PUNCT2LABEL,
    PUNCT_SPECIAL_TOKEN,
)
from .dataset_utils import (
    clean_up_data_from_txt,
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


class SimilarityEngine:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def get_sentence_embedding(self, tokens):
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True,
            truncation=False,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Average the token embeddings to get a sentence embedding
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def compute_similarity(self, list1, list2):
        # Get embeddings for both lists
        embedding1 = self.get_sentence_embedding(list1)
        embedding2 = self.get_sentence_embedding(list2)

        # Compute cosine similarity
        similarity = cosine_similarity(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        )[0][0]
        return similarity


class BatchSimilarityCalculator:
    def __init__(self, max_length: int = 100, threshold=0.5):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.threshold = threshold

    def get_batch_embeddings(self, token_lists):
        all_chunks = []
        chunk_mappings = []
        
        for token_list in token_lists:
            batch_chunk_indexes = []
            for index_of_chunk in range(0, len(token_list), self.max_length):
                index_in_batch = len(all_chunks)
                chunk = token_list[index_of_chunk:index_of_chunk+self.max_length]
                all_chunks.append(chunk)
                batch_chunk_indexes.append(index_in_batch)
            
            chunk_mappings.append(batch_chunk_indexes)

        inputs = self.tokenizer(
            all_chunks,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True,
            truncation=False, 
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        chunk_embeddings = outputs.last_hidden_state.mean(dim=1)

        sequence_embeddings = []
        for indexes in chunk_mappings:
            sequence_embedding = torch.mean(chunk_embeddings[indexes], dim=0)
            sequence_embeddings.append(sequence_embedding)

        return sequence_embeddings

    def compute_batch_similarity(self, batch1, batch2):
        if len(batch1) != len(batch2):
            raise ValueError("Both batches must have the same number of elements")

        # Get embeddings for both batches
        embeddings1 = self.get_batch_embeddings(batch1)
        embeddings2 = self.get_batch_embeddings(batch2)

        # Compute pairwise cosine similarity for each pair in the batches
        similarities = []
        for batch_index, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
            similarity = cosine_similarity(emb1.unsqueeze(0).cpu(), emb2.unsqueeze(0).cpu())[0][0]
            if similarity < self.threshold:
                logger.info("Very low similarity! llm result: %s \n original: %s", " ".join(batch1[batch_index]), " ".join(batch2[batch_index]))
            similarities.append(similarity)

        return similarities

    def get_overall_similarity(
        self,
        full_llm_result,
        full_groundtruth,
        batch_size: int = 10,
        ner_mapping=PUNCT2LABEL,
    ):
        full_llm_results_list = [
            process_line(result_line.strip("'"), ner_mapping=ner_mapping)[0]
            for result_line in full_llm_result
        ]

        assert len(full_groundtruth) == len(full_llm_results_list)
        all_similarities = []
        starting_index = 0
        ending_index = starting_index + batch_size
        pbar = tqdm(total=len(full_groundtruth))
        while ending_index <= len(full_groundtruth):
            batch_llm = full_llm_results_list[starting_index:ending_index]
            batch_groundtruth = full_groundtruth[starting_index:ending_index]
            all_similarities.extend(
                self.compute_batch_similarity(batch_llm, batch_groundtruth)
            )
            starting_index = ending_index
            ending_index += batch_size
            pbar.update(
                batch_size
                if ending_index <= len(full_groundtruth)
                else (batch_size - (ending_index - len(full_groundtruth)))
            )

        pbar.close()
        logger.info(
            "avg similarity score: %.3f",
            sum(all_similarities) / len(all_similarities),
        )
        return all_similarities


async def generate_llm_results_bert_hint(
    source_file_path,
    bert_output,
    target_model,
    min_sequence_length=32,
    max_sequence_length=160,
    raw_output_file_path=None,
    processed_output_file_path=None,
    chunk_size=40,
    chat_messages=LLM_CHAT_MESSAGES_BERT_OUTPUT
):
    _, w_bert_output, pure_tokens_gt = read_data_to_w_special_token(
        source_file_path,
        bert_output,
        min_sequence_length,
        max_sequence_length,
        PUNCT_SPECIAL_TOKEN,
    )
    logger.info("random sample length: %s", len(random.choice(w_bert_output).split()))
    chat_messages_list_bert = generate_dataset(
        chat_messages=chat_messages,
        input_list=w_bert_output,
        token=PUNCT_SPECIAL_TOKEN,
    )
    logger.info("chat message sample: %s", chat_messages_list_bert[0])

    chat_completion_sample = await openai.chat.completions.create(
        model=target_model, messages=chat_messages_list_bert[0], temperature=0.1
    )

    logger.info(
        "sample output: %s",
        chat_completion_sample.choices[0].message.content,
    )

    generated_sentences = await query_server_in_chunk(
        chat_messages_list_bert,
        model_name=target_model,
        chunk_size=chunk_size,
    )

    if raw_output_file_path is not None:
        with open(raw_output_file_path, "w", encoding="utf-8") as results_file:
            for sentence in generated_sentences:
                results_file.write(sentence + "\n")

    processed_results = clean_up_data_from_txt(
        generated_sentences,
        processed_output_file_path,
        target_punctuations=PUNCT2LABEL.keys(),
        additional_to_keep=["'", "-"],
        additional_to_remove=["℃", "|", "♫"],
        special_cleaning_funcs=[
            partial(normalize_puncs, normalization=CROSS_LANG_PUNCT_MAPPINGS),
            chinese_split,
            remove_brackets_text,
        ],
    )

    return processed_results, pure_tokens_gt


async def generate_llm_results_directly(
    source_file_path,
    bert_output,
    target_model,
    min_sequence_length=32,
    max_sequence_length=160,
    raw_output_file_path=None,
    processed_output_file_path=None,
    chunk_size=40,
    chat_messages=LLM_CHAT_MESSAGES_RAW
):
    raw_input_list, _, pure_labels_gt = read_data_to_w_special_token(
        source_file_path,
        bert_output,
        min_sequence_length,
        max_sequence_length,
        PUNCT_SPECIAL_TOKEN,
        is_split_into_words=False,
    )
    chat_messages_list = generate_dataset(
        chat_messages=chat_messages,
        input_list=raw_input_list,
        token=None,
    )
    logger.info("chat message sample: %s", chat_messages_list[0])

    chat_completion_sample = await openai.chat.completions.create(
        model=target_model, messages=chat_messages_list[0], temperature=0.1
    )

    logger.info(
        "sample output: %s",
        chat_completion_sample.choices[0].message.content,
    )

    generated_sentences = await query_server_in_chunk(
        chat_messages_list,
        model_name=target_model,
        chunk_size=chunk_size,
    )

    if raw_output_file_path is not None:
        with open(raw_output_file_path, "w", encoding="utf-8") as results_file:
            for sentence in generated_sentences:
                results_file.write(sentence + "\n")

    processed_results = clean_up_data_from_txt(
        generated_sentences,
        processed_output_file_path,
        target_punctuations=PUNCT2LABEL.keys(),
        additional_to_keep=["'", "-"],
        additional_to_remove=["℃", "|", "♫"],
        special_cleaning_funcs=[
            partial(normalize_puncs, normalization=CROSS_LANG_PUNCT_MAPPINGS),
            chinese_split,
            remove_brackets_text,
        ],
    )

    return processed_results, pure_labels_gt


async def generate_llm_results_punct_positions(
    source_file_path,
    bert_output,
    target_model,
    min_sequence_length=32,
    max_sequence_length=160,
    raw_output_file_path=None,
    processed_output_file_path=None,
    chunk_size=40,
):
    raw_input_list, _, pure_labels_gt = read_data_to_w_special_token(
        source_file_path,
        bert_output,
        min_sequence_length,
        max_sequence_length,
        PUNCT_SPECIAL_TOKEN,
        is_split_into_words=False,
    )
    chat_messages_list = generate_dataset(
        chat_messages=LLM_CHAT_MESSAGES_FIND_POSITION,
        input_list=raw_input_list,
        token=PUNCT_SPECIAL_TOKEN,
    )
    logger.info("chat message sample: %s", chat_messages_list[0])

    chat_completion_sample = await openai.chat.completions.create(
        model=target_model, messages=chat_messages_list[0], temperature=0.1
    )

    logger.info(
        "sample output: %s",
        chat_completion_sample.choices[0].message.content,
    )

    generated_sentences = await query_server_in_chunk(
        chat_messages_list,
        model_name=target_model,
        chunk_size=chunk_size,
    )

    if raw_output_file_path is not None:
        with open(raw_output_file_path, "w", encoding="utf-8") as results_file:
            for sentence in generated_sentences:
                results_file.write(sentence + "\n")

    processed_results = clean_up_data_from_txt(
        generated_sentences,
        processed_output_file_path,
        target_punctuations=PUNCT2LABEL.keys(),
        additional_to_keep=["'", "-"],
        additional_to_remove=["℃", "|", "♫"],
        special_cleaning_funcs=[
            partial(normalize_puncs, normalization=CROSS_LANG_PUNCT_MAPPINGS),
            chinese_split,
            remove_brackets_text,
        ],
    )

    return processed_results, pure_labels_gt


async def generate_llm_results_repeat_sequence(
    source_file_path,
    bert_output,
    target_model,
    min_sequence_length=32,
    max_sequence_length=160,
    raw_output_file_path=None,
    processed_output_file_path=None,
    chunk_size=40,
):
    raw_input_list, _, pure_labels_gt = read_data_to_w_special_token(
        source_file_path,
        bert_output,
        min_sequence_length,
        max_sequence_length,
        PUNCT_SPECIAL_TOKEN,
        is_split_into_words=False,
    )
    chat_messages_list = generate_dataset(
        chat_messages=LLM_CHAT_MESSAGES_REPETATION,
        input_list=raw_input_list,
        token=None,
    )
    logger.info("chat message sample: %s", chat_messages_list[0])

    chat_completion_sample = await openai.chat.completions.create(
        model=target_model, messages=chat_messages_list[0], temperature=0.1
    )

    logger.info(
        "sample output: %s",
        chat_completion_sample.choices[0].message.content,
    )

    generated_sentences = await query_server_in_chunk(
        chat_messages_list,
        model_name=target_model,
        chunk_size=chunk_size,
    )

    if raw_output_file_path is not None:
        with open(raw_output_file_path, "w", encoding="utf-8") as results_file:
            for sentence in generated_sentences:
                results_file.write(sentence + "\n")

    processed_results = clean_up_data_from_txt(
        generated_sentences,
        processed_output_file_path,
        target_punctuations=PUNCT2LABEL.keys(),
        additional_to_keep=["'", "-"],
        additional_to_remove=["℃", "|", "♫"],
        special_cleaning_funcs=[
            partial(normalize_puncs, normalization=CROSS_LANG_PUNCT_MAPPINGS),
            chinese_split,
            remove_brackets_text,
        ],
    )

    return processed_results, pure_labels_gt


def evaluate_llm_output(
    # similarity_engine: SimilarityEngine,
    processed_results,
    pure_tokens_gt,
    pure_labels_gt,
    target_model,
    label2id=LABEL2ID,
    ner_mapping=PUNCT2LABEL,
):
    really_not_matched = []

    all_gt_labels = []
    all_result_labels = []

    # similarity_list = []
    for result_index, result_line in enumerate(tqdm(processed_results)):
        result_tokens, result_labels = process_line(
            result_line.strip("'"), ner_mapping=ner_mapping
        )
        gt_tokens = pure_tokens_gt[result_index]
        gt_labels = pure_labels_gt[result_index]

        gt_label_ids = [label2id.get(label, 1) for label in gt_labels]
        result_label_ids = [label2id.get(label, 1) for label in result_labels]
        if len(gt_label_ids) == len(result_label_ids):
            all_gt_labels.extend(gt_label_ids)
            all_result_labels.extend(result_label_ids)
            continue
        # is_subset, subset_indices = is_sublist_and_get_indices(gt_tokens, result_tokens)
        # if is_subset:
        #     result_label_ids = [
        #         label_id
        #         for label_index, label_id in enumerate(result_label_ids)
        #         if label_index in subset_indices
        #     ]
        #     # result_label_ids = [label_id for label_id in result_label_ids if label_id >= 0]
        #     all_gt_labels.extend(gt_label_ids)
        #     all_result_labels.extend(result_label_ids)
        # else:
        token_index = 0
        first_token = gt_tokens[token_index]
        first_index_in_prediction = None
        while token_index < len(gt_tokens):
            try:
                first_index_in_prediction = result_tokens.index(first_token)
                break
            except ValueError:
                # logger.warning(str(ve))
                # logger.warning(gt_tokens)
                # really_not_matched.append(result_index)
                # simialrity_score = similarity_engine.compute_similarity(
                #     result_tokens, gt_tokens
                # )
                # similarity_list.append(simialrity_score)
                # if np.random.rand() < 0.002:
                #     logger.info(
                #         "original list length: %s | predicted list length: %s | similarity: %s",
                #         len(gt_tokens),
                #         len(result_tokens),
                #         simialrity_score,
                #     )
                token_index += 1
                if token_index >= len(gt_tokens):
                    break
                first_token = gt_tokens[token_index]
                continue
        if first_index_in_prediction is None:
            really_not_matched.append(result_index)
        else:
            pred_labels = result_label_ids[
                first_index_in_prediction : min(
                    len(result_label_ids), len(gt_label_ids)
                )
            ]
            all_result_labels.extend(pred_labels)

            all_gt_labels.extend(gt_label_ids[: len(pred_labels)])
        assert len(all_gt_labels) == len(all_result_labels)

    logger.info(
        "total not matched index: %s",
        len(really_not_matched),
    )

    tested_labels = []
    target_names = []
    for label, label_id in label2id.items():
        if label != NORMAL_TOKEN_TAG:
            tested_labels.append(label_id)
            target_names.append(label)

    report_for_matched = classification_report(
        all_gt_labels,
        all_result_labels,
        labels=tested_labels,
        digits=4,
        target_names=target_names,
        zero_division=1,
    )

    logger.info(
        "validation report for %s result matched: \n %s",
        target_model,
        report_for_matched,
    )
    # logger.info(
    #     "not matched avg similarity score: %.3f",
    #     sum(similarity_list) / len(similarity_list),
    # )


