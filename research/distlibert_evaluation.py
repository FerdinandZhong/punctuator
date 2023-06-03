from punctuator.training import EvaluationArguments, EvaluationPipeline, process_data

test_data_file_path = "data/IWSLT/formatted/test2011"

with open(test_data_file_path, "r", encoding="ISO-8859-1") as file:
    test_data = file.readlines()

# must be exact same as model's config
label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
evalution_corpus, evaluation_tags = process_data(test_data, 128)
evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]
evaluation_args = EvaluationArguments(
    evaluation_corpus=evalution_corpus,
    evaluation_tags=evaluation_tags,
    model_weight_name="models/iwslt_distilbert",
    tokenizer_name="distilbert-base-uncased",
    batch_size=64,
    gpu_device=0,
    label2id=label2id,
)

evaluation_pipeline = EvaluationPipeline(evaluation_args)
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
#     model_weight_name="models/iwslt_distilbert",
#     tokenizer_name="Qishuai/distilbert_punctuator_en",
#     tag2punctuator=DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP,
#     gpu_device=0,
# )

# inference = InferencePipeline(args, verbose=True)


# test_texts_1 = [
#     "how are you its been ten years since we met in shanghai i'm really happy to meet you again whats your current phone number",  # noqa: E501
#     "my number is 82732212",
# ]

# inference.punctuation(test_texts_1)
