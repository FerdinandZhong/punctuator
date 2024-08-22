from punctuator.llm_evaluation.pipeline import (
    generate_llm_results_bert_hint, 
    evaluate_llm_output, 
    generate_llm_results_directly,
    generate_llm_results_punct_positions,
    BatchSimilarityCalculator
)
from punctuator.llm_evaluation.constants import PUNCTSPECIAL2ID, PUNCTSPECIAL2LABLE
from punctuator.llm_evaluation.constants import (
    PUNCT_SPECIAL_TOKEN, PUNCTSPECIAL2ID
)
from punctuator.llm_evaluation.dataset_utils import (
    read_data_to_w_special_token,
)
import logging

logger = logging.getLogger(__name__)

similarity_engine = BatchSimilarityCalculator(max_length=160)


with open("data/llm_results/llama3.1_8b_with_roberta_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_llama8b_bert_results = result_file.readlines()

with open("data/llm_results/llama3.1_8b_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_llama8b_direct_restoration_results = result_file.readlines()

with open("data/llm_results/llama3.1_70b_with_roberta_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_llama70b_bert_results = result_file.readlines()

with open("data/llm_results/llama3.1_70b_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_llama70b_direct_restoration_results = result_file.readlines()

with open("data/llm_results/lama3.1_405b_with_roberta_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_llama405b_bert_results = result_file.readlines()

with open("data/llm_results/lama3.1_405b_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_llama405b_direct_restoration_results = result_file.readlines()

with open("data/llm_results/qwen2_72b_with_roberta_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_qwen72b_bert_results = result_file.readlines()

with open("data/llm_results/qwen2_72b_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_qwen72b_direct_restoration_results = result_file.readlines()

with open("data/llm_results/qwen2_72b_punct_position_output_processed.txt", "r", encoding="utf-8") as result_file:
    processed_qwen72b_punct_position = result_file.readlines()


pure_tokens_gt, _, pure_labels_gt = read_data_to_w_special_token(
    source_data = "data/newly_code_switching/test.txt",
    bert_output="data/new_code_switching_binary/roberta_binary_output_test.txt",
    min_sequence_length=512,
    max_sequence_length=512,
    punct_special_token=PUNCT_SPECIAL_TOKEN,
)


logger.info("SIMILARITY FOR LLAMA 8B")
similarity_engine.get_overall_similarity(processed_llama8b_direct_restoration_results, pure_tokens_gt, batch_size=40)

similarity_engine.get_overall_similarity(processed_llama70b_direct_restoration_results, pure_tokens_gt, batch_size=40)

similarity_engine.get_overall_similarity(processed_llama405b_direct_restoration_results, pure_tokens_gt, batch_size=40)

similarity_engine.get_overall_similarity(processed_qwen72b_direct_restoration_results, pure_tokens_gt, batch_size=40)