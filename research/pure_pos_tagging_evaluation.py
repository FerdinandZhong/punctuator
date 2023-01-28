from punctuator.training import process_data
from punctuator.training.pos_evalute import (
    PosEvaluationArguments,
    PosEvaluationPipeline,
)
from punctuator.utils import Models

test_data_file_path = "data/IWSLT/formatted/test2011"

with open(test_data_file_path, "r") as file:
    test_data = file.readlines()

# must be exact same as model's config
label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
evalution_corpus, evaluation_tags = process_data(test_data, 256, 256)
evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]
evaluation_args = PosEvaluationArguments(
    evaluation_corpus=evalution_corpus,
    evaluation_tags=evaluation_tags,
    model=Models.BERT_TOKEN_CLASSIFICATION,
    model_weight_name="models/iwslt_pure_pos_tagging",
    tokenizer_name="bert-large-uncased",
    batch_size=16,
    gpu_device=0,
    label2id=label2id,
    test_pos_tagging_path="data/IWSLT/formatted/postagging_test",
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
)

evaluation_pipeline = PosEvaluationPipeline(evaluation_args)
evaluation_pipeline.run()


# logger = logging.getLogger(__name__)
# register_logger(logger)

# DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP = {
#     "O": ("", False),
#     "COMMA": (",", False),
#     "PERIOD": (".", True),
#     "QUESTION": ("?", True),
# }

# args = InferenceArguments(
#     model=Models.BERT,
#     model_weight_name="models/iwslt_bert_finetune",
#     tokenizer_name="bert-large-uncased",
#     tag2punctuator=DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP,
#     use_gpu=False
# )

# inference = InferencePipeline(args, verbose=True)


# test_texts_1 = [
#     "how are you its been ten years since we met in shanghai i'm really happy to meet you again whats your current phone number",  # noqa: E501
#     "my number is 82732212",
# ]

# inference.punctuation(test_texts_1)
