import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup

from punctuator.focal_loss.focal_loss import FocalLoss
from punctuator.utils import NORMAL_TOKEN_TAG, Models, model_type, str2bool

from .finetuning_data_process import read_data_after_step1

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 0.1
DEFAULT_LABEL2ID = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}


# TODO: adding additional information from step 1
class Step2EncodingDataset:
    def __init__(self, encodings, labels, step1_features):
        self.encodings = encodings
        self.labels = labels
        self.step1_features = step1_features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["step1_features"] = torch.tensor(self.step1_features[idx])
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    inputs, has_punctuation = zip(*batch)
    return {"inputs": inputs, "has_punctuation": torch.stack(has_punctuation)}


class Step2NERTrainingArguments(BaseModel):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        # basic arguments
        training_corpus(List[List[str]]): list of sequences for training, longest sequence should be no longer than pretrained LM # noqa: E501
        validation_corpus(List[List[str]]): list of sequences for validation, longest sequence should be no longer than pretrained LM # noqa: E501
        training_tags(List[List[int]]): tags(int) for training
        validation_tags(List[List[int]]): tags(int) for validation
        model(Optional(enum)): model selected from Enum Models, default is "DISTILBERT"
        model_weight_name(str): name or path of pre-trained model weight
        tokenizer_name(str): name of pretrained tokenizer

        # training arguments
        epoch(int): number of epoch
        batch_size(int): batch size
        model_storage_dir(str): fine-tuned model storage path
        label2id(Dict): the tags label and id mapping
        early_stop_count(int): after how many epochs to early stop training if valid loss not become smaller. default 3 # noqa: E501
        use_gpu(Optional[bool]): whether to use gpu for training, default is "True"
        gpu_device(Optional[int]): specific gpu card index, default is the CUDA_VISIBLE_DEVICES from environ
        warm_up_steps(int): warm up steps.
        r_drop(bool): whether to train with r-drop
        r_alpha(int): alpha value for kl divengence in the loss, default is 0
        plot_steps(int): record training status to tensorboard among how many steps
        tensorboard_log_dir(Optional[str]): the tensorboard logs output directory, default is "runs"

        # model arguments
        additional_model_config(Optional[Dict]): additional configuration for model
    """

    # basic args
    training_corpus: Union[List[List[str]], List[str]]
    validation_corpus: Union[List[List[str]], List[str]]
    training_tags: List[List[int]]
    validation_tags: List[List[int]]
    training_step1_features: List[List[int]]
    validation_step1_features: List[List[int]]
    model_weight_name: str
    tokenizer_name: str
    model: Optional[Models] = Models.DISTILBERT
    load_backbone_only: bool = True

    # training ars
    epoch: int
    batch_size: int
    model_storage_dir: str
    intermediate_persist_step: int = 5
    label2id: Dict
    early_stop_count: Optional[int] = 3
    use_gpu: Optional[bool] = True
    gpu_device: Optional[int] = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    warm_up_steps: int = 1000
    r_drop: bool = False
    r_alpha: float = 0
    plot_steps: int = 50
    tensorboard_log_dir: Optional[str] = "runs"
    use_class_weight: bool = True
    log_class_weight: bool = True
    is_split_into_words: bool = True

    # model args
    additional_model_config: Optional[Dict]
    additional_tokenizer_config: Optional[Dict] = {}
    additional_classifier_kwargs: Optional[Dict] = {}
    label_at_start: bool = True

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:

        # Basic arguments
        parser.add_argument(
            "--training_data_file_path",
            type=str,
            required=True,
            help="Path to training corpus file",
        )
        parser.add_argument(
            "--validation_data_file_path",
            type=str,
            required=True,
            help="Path to validation corpus file",
        )
        parser.add_argument(
            "--training_step1_result_file_path",
            type=str,
            required=True,
            help="Path to training corpus file",
        )
        parser.add_argument(
            "--validation_step1_result_file_path",
            type=str,
            required=True,
            help="Path to validation corpus file",
        )
        parser.add_argument(
            "--punct_special_token",
            type=str,
            default="PUNCT",
            help="Special punctuation for processing step1 result",
        )
        parser.add_argument(
            "--min_sequence_length",
            type=int,
            required=True,
            default=32,
            help="Minimum sequence length (count of the words) in each sample.",
        )
        parser.add_argument(
            "--max_sequence_length",
            type=int,
            required=True,
            default=128,
            help="Maximum sequence length (count of the words) in each sample.",
        )
        parser.add_argument(
            "--load_backbone_only",
            type=str2bool,
            default=True,
            help="Load the backbone model only.",
        )
        parser.add_argument(
            "--model_weight_name",
            type=str,
            required=True,
            help="Path or name of pre-trained model weight",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            required=True,
            help="Name of pretrained tokenizer",
        )
        parser.add_argument(
            "--model",
            type=str,
            choices=[m.name for m in Models],
            help="Model to use",
        )

        # Training arguments
        parser.add_argument("--epoch", type=int, required=True, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
        parser.add_argument(
            "--model_storage_dir",
            type=str,
            required=True,
            help="Path to store the fine-tuned model",
        )
        parser.add_argument(
            "--label2id", type=str, required=True, help="Label to ID mapping"
        )
        parser.add_argument(
            "--early_stop_count",
            type=int,
            default=3,
            help="Epochs to stop training if no improvement",
        )
        parser.add_argument(
            "--use_gpu",
            type=str2bool,
            default=True,
            help="Whether to use GPU for training",
        )
        parser.add_argument(
            "--gpu_device", type=int, default=0, help="Local rank of the process"
        )
        parser.add_argument(
            "--warm_up_steps", type=int, default=1000, help="Number of warm-up steps"
        )
        parser.add_argument(
            "--r_drop",
            type=str2bool,
            default=False,
            help="Whether to train with R-Drop",
        )
        parser.add_argument(
            "--r_alpha",
            type=float,
            default=0.5,
            help="Alpha value for KL divergence in R-Drop",
        )
        parser.add_argument(
            "--plot_steps",
            type=int,
            default=50,
            help="Steps interval to log status to TensorBoard",
        )
        parser.add_argument(
            "--tensorboard_log_dir",
            type=str,
            default="runs",
            help="TensorBoard logs directory",
        )
        parser.add_argument(
            "--intermediate_persist_step",
            type=int,
            default=5,
            help="Intermediate step to persist the model weights",
        )

        parser.add_argument(
            "--use_class_weight",
            type=str2bool,
            default=True,
            help="Whether to assign weights to classes",
        )
        parser.add_argument(
            "--log_class_weight",
            type=str2bool,
            default=True,
            help="Whether to assign weights to classes",
        )

        # Model-specific arguments
        parser.add_argument(
            "--additional_model_config",
            type=str,
            help="JSON string of additional model config",
        )
        parser.add_argument(
            "--additional_tokenizer_config",
            type=str,
            default="{}",
            help="JSON string of additional model config",
        )
        parser.add_argument(
            "--is_split_into_words",
            type=str2bool,
            default=True,
            help="Whether the input is split into words",
        )
        parser.add_argument(
            "--additional_classifier_kwargs",
            type=str,
            default="{}",
            help="JSON string of additional classifier kwargs",
        )
        parser.add_argument(
            "--label_at_start",
            type=str2bool,
            default=True,
            help="Whether have the label at the start of the word",
        )
        return parser

    @staticmethod
    def generate_corpus(args: argparse.Namespace):
        """
        Generates the training and validation corpus along with their corresponding tags and step1 features from raw data files.

        This method reads the training and validation data files, processes them according to the specifications defined in args,
        and prepares the datasets for model training. It also handles the conversion of labels to numerical IDs based on the label2id mapping provided.

        Args:
            args (argparse.Namespace): Command line arguments containing paths to training and validation data files, model parameters, and other configurations.

        Returns:
            tuple: A tuple containing:
                - training_corpus (List[List[str]]: Tokenized training corpus.
                - validation_corpus (List[List[str]]: Tokenized validation corpus.
                - training_tags (List[List[int]]: Numerical tags for training data.
                - validation_tags (List[List[int]]: Numerical tags for validation data.
                - training_step1_features (List[List[int]]: Features from step 1 for training data.
                - validation_step1_features (List[List[int]]: Features from step 1 for validation data.
                - label2id (Dict[str, int]): Mapping of labels to numerical IDs.
        """
        with open(args.training_data_file_path, "r", encoding="utf-8") as file:
            training_raw = file.readlines()

        with open(args.validation_data_file_path, "r", encoding="utf-8") as file:
            val_raw = file.readlines()

        with open(args.training_step1_result_file_path, "r", encoding="utf-8") as file:
            training_step1_result = file.readlines()

        with open(
            args.validation_step1_result_file_path, "r", encoding="utf-8"
        ) as file:
            val_step1_result = file.readlines()

        (
            training_corpus,
            _,
            training_tags,
            training_step1_features,
        ) = read_data_after_step1(
            training_raw,
            training_step1_result,
            args.min_sequence_length,
            args.max_sequence_length,
            args.punct_special_token,
            args.is_split_into_words,
        )

        (
            validation_corpus,
            _,
            validation_tags,
            validation_step1_features,
        ) = read_data_after_step1(
            val_raw,
            val_step1_result,
            args.min_sequence_length,
            args.max_sequence_length,
            args.punct_special_token,
            args.is_split_into_words,
        )

        try:
            label2id = json.loads(args.label2id)
        except json.JSONDecodeError:
            label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
        training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
        validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

        sample = training_corpus[0]
        logger.info("Corpus Sample: %s", sample)
        logger.info("Step 1 result: %s", training_step1_features[0])

        return (
            training_corpus[:100],
            validation_corpus[:100],
            training_tags[:100],
            validation_tags[:100],
            training_step1_features[:100],
            validation_step1_features[:100],
            label2id,
        )

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace,
        training_corpus: List[List[str]],
        validation_corpus: List[List[str]],
        training_tags: List[List[int]],
        validation_tags: List[List[int]],
        training_step1_features: List[List[int]],
        validation_step1_features: List[List[int]],
        label2id: Dict,
    ):
        try:
            additional_model_config = json.loads(args.additional_model_config)
        except (json.JSONDecodeError, TypeError):
            additional_model_config = {}
        try:
            additional_tokenizer_config = json.loads(args.additional_tokenizer_config)
        except (json.JSONDecodeError, TypeError):
            additional_tokenizer_config = {}
        try:
            additional_classifier_kwargs = json.loads(args.additional_classifier_kwargs)
        except (json.JSONDecodeError, TypeError):
            additional_classifier_kwargs = {}

        # Set the attributes from the parsed arguments.
        training_pipeline_args = cls(
            training_corpus=training_corpus,
            validation_corpus=validation_corpus,
            training_tags=training_tags,
            validation_tags=validation_tags,
            training_step1_features=training_step1_features,
            validation_step1_features=validation_step1_features,
            model=model_type(args.model),
            load_backbone_only=args.load_backbone_only,
            model_weight_name=args.model_weight_name,
            tokenizer_name=args.tokenizer_name,
            epoch=args.epoch,
            batch_size=args.batch_size,
            model_storage_dir=args.model_storage_dir,
            intermediate_persist_step=args.intermediate_persist_step,
            additional_model_config=additional_model_config,
            additional_classifier_kwargs=additional_classifier_kwargs,
            gpu_device=args.gpu_device,
            warm_up_steps=args.warm_up_steps,
            r_drop=args.r_drop,
            r_alpha=args.r_alpha,
            tensorboard_log_dir=args.tensorboard_log_dir,
            plot_steps=args.plot_steps,
            label2id=label2id,
            early_stop_count=args.early_stop_count,
            use_class_weight=args.use_class_weight,
            log_class_weight=args.log_class_weight,
            additional_tokenizer_config=additional_tokenizer_config,
            is_split_into_words=args.is_split_into_words,
            label_at_start=args.label_at_start
        )

        return training_pipeline_args


class Step2NERTrainingPipeline:
    def __init__(self, training_arguments):
        """Training pipeline for fine-tuning the distilbert token classifier for punctuation

        Args:
            training_arguments (TrainingArguments): arguments passed to training pipeline
        """
        self.arguments = training_arguments
        logger.info("cuda available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            self.world_size = torch.cuda.device_count()
        else:
            self.world_size = 1

        self.label2id = training_arguments.label2id
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.tensorboard_writter = SummaryWriter(training_arguments.tensorboard_log_dir)

        model_collection = training_arguments.model.value

        self.num_labels = len(self.id2label)
        self.model_config = model_collection.config.from_pretrained(
            training_arguments.model_weight_name,
            label2id=self.label2id,
            id2label=self.id2label,
            num_labels=self.num_labels,
            **training_arguments.additional_model_config,
        )

        self.tokenizer = model_collection.tokenizer.from_pretrained(
            training_arguments.tokenizer_name,
            **training_arguments.additional_tokenizer_config,
        )
        logger.info("loaded tokenizer: %s", self.tokenizer)
        logger.info("start loading model")
        if training_arguments.load_backbone_only:
            backbone_model = model_collection.backbone_model.from_pretrained(
                training_arguments.model_weight_name,
                # config=self.model_config,
            )
            self.classifier = model_collection.model(
                self.model_config,
                backbone_model=backbone_model,
                **training_arguments.additional_classifier_kwargs,
            )

        else:
            self.classifier = model_collection.model.from_pretrained(
                training_arguments.model_weight_name,
                config=self.model_config,
                **training_arguments.additional_classifier_kwargs,
            )

        self.model_class = self.classifier.__class__.__name__
        self.classifier.set_loss_fct(FocalLoss())
        logger.info("model loaded")

        if torch.cuda.is_available() and training_arguments.use_gpu:
            if torch.cuda.device_count() > 1:
                self.classifier = torch.nn.DataParallel(self.classifier)
                self.classifier.cuda()
                self.device = torch.device("cuda")
                self.is_parallel = True
            else:
                self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
                self.classifier.to(self.device)
                self.is_parallel = False

        else:
            self.device = torch.device("cpu")
            self.is_parallel = False

        self.total_steps = 0
        self.class_weights = None
        self.train_encoded_tags = None
        self.validation_encoded_tags = None
        self.train_encoded_step1_features = None
        self.validation_encoded_step1_features = None
        self.train_encodings = None
        self.val_encodings = None
        self.training_dataset = None
        self.val_dataset = None
        self.best_state_dict = None
        self.best_acc_state_dict = None

    def tokenize(self):
        """
        Tokenizes the training and validation corpora using the specified tokenizer.

        This method prepares the text data for further processing by converting it into a format that the model can understand.
        It also handles splitting the text into words where necessary and ensures that the resulting tokens are padded appropriately.
        """  # noqa E501
        logger.info("tokenize data")

        logger.info("splited into words: %s", self.arguments.is_split_into_words)

        self.train_encodings = self.tokenizer(
            self.arguments.training_corpus,
            is_split_into_words=self.arguments.is_split_into_words,
            return_offsets_mapping=True,
            padding=True,
        )
        self.val_encodings = self.tokenizer(
            self.arguments.validation_corpus,
            is_split_into_words=self.arguments.is_split_into_words,
            return_offsets_mapping=True,
            padding=True,
        )

        all_ner_tag_ids = [
            tag_id
            for sen_tag_ids in self.arguments.training_tags
            + self.arguments.validation_tags
            for tag_id in sen_tag_ids
        ]
        unique_tag_ids = set(all_ner_tag_ids)

        logger.info("unique tag ids: %s, id2label: %s", unique_tag_ids, self.id2label)

        if self.arguments.use_class_weight:
            if self.arguments.log_class_weight:
                weights = [
                    weight if weight > 0 else DEFAULT_LABEL_WEIGHT
                    for weight in np.log(
                        class_weight.compute_class_weight(
                            "balanced",
                            classes=np.array(list(unique_tag_ids)),
                            y=all_ner_tag_ids,
                        )
                    )
                ] * torch.cuda.device_count()
            else:
                weights = (
                    class_weight.compute_class_weight(
                        "balanced",
                        classes=np.array(list(unique_tag_ids)),
                        y=all_ner_tag_ids,
                    ).tolist()
                    * torch.cuda.device_count()
                )
            logger.info(
                "class weights: %s, id2label: %s",
                ", ".join([f"{round(weight, 2)}" for weight in weights]),
                self.id2label,
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float).to(
                self.device
            )
            logger.info("class weights tensor: %s", self.class_weights)
        else:
            self.class_weights = None

        self.train_encoded_tags = self._encode_tags(
            self.arguments.training_tags,
            self.train_encodings,
            self.arguments.training_corpus,
        )
        self.validation_encoded_tags = self._encode_tags(
            self.arguments.validation_tags,
            self.val_encodings,
            self.arguments.validation_corpus,
        )

        self.train_encoded_step1_features = self._encode_step1_features(
            self.arguments.training_step1_features,
            self.train_encodings,
            self.arguments.training_corpus,
        )

        self.validation_encoded_step1_features = self._encode_step1_features(
            self.arguments.validation_step1_features,
            self.val_encodings,
            self.arguments.validation_corpus,
        )

        return self

    def generate_dataset(self):
        """
        Generates datasets for training and validation from tokenized data.

        Removes the 'offset_mapping' from the encodings and creates instances of EncodingDataset for training and validation datasets.

        Returns:
            self: The instance of the class itself for method chaining.
        """  # noqa E 501
        logger.info("generate dataset from tokenized data")
        self.train_encodings.pop("offset_mapping")
        self.val_encodings.pop("offset_mapping")
        self.training_dataset = Step2EncodingDataset(
            self.train_encodings,
            self.train_encoded_tags,
            self.train_encoded_step1_features,
        )
        self.val_dataset = Step2EncodingDataset(
            self.val_encodings,
            self.validation_encoded_tags,
            self.validation_encoded_step1_features,
        )

        return self

    def fine_tune(self):
        logger.info("start fine tune")

        train_loader = DataLoader(
            self.training_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        optim = AdamW(self.classifier.parameters(), lr=1e-5)

        scheduler = get_constant_schedule_with_warmup(
            optim,
            num_warmup_steps=self.arguments.warm_up_steps,
        )

        best_val_loss = 100
        best_val_acc = 0
        no_improvement_count = 0

        with tqdm(total=self.arguments.epoch) as pbar:
            for epoch in range(self.arguments.epoch):
                pbar.set_description(f"Processing epoch: {epoch + 1}")

                start_time = time.time()

                self.classifier.train()

                train_loss, train_acc = self._train(train_loader, optim, scheduler)
                val_loss, val_acc = self._train(val_loader, optim, scheduler, True)

                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/train", train_loss, epoch + 1
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/valid", val_loss, epoch + 1
                )

                self.tensorboard_writter.add_scalar(
                    "Epoch acc/train", train_acc, epoch + 1
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch acc/valid", val_acc, epoch + 1
                )

                end_time = time.time()

                epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

                logger.info(
                    "Epoch: %02d | Epoch Time: %dm %ds",
                    epoch + 1,
                    epoch_mins,
                    epoch_secs,
                )
                logger.info(
                    "\tTrain Loss: %.3f | Train Acc: %.2f%%",
                    train_loss,
                    train_acc * 100,
                )
                logger.info(
                    "\t Val. Loss: %.3f |  Val. Acc: %.2f%%", val_loss, val_acc * 100
                )

                pbar.update(1)

                if (epoch + 1) % self.arguments.intermediate_persist_step == 0:
                    logger.info(
                        "Save the intermediate checkpoint for epoch: %d", epoch + 1
                    )

                    self._intermediate_persist(epoch + 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.is_parallel:
                        self.best_state_dict = self.classifier.module.state_dict()
                    else:
                        self.best_state_dict = self.classifier.state_dict()
                    no_improvement_count = 0
                elif val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if self.is_parallel:
                        self.best_acc_state_dict = self.classifier.module.state_dict()
                    else:
                        self.best_acc_state_dict = self.classifier.state_dict()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if (
                        self.arguments.early_stop_count > 0
                        and no_improvement_count >= self.arguments.early_stop_count
                    ):
                        logger.info(
                            "No improvement for past %s epochs, early stop training.",
                            no_improvement_count,
                        )  # noqa E501
                        return self

            logger.info("fine-tune finished")

        self.tensorboard_writter.flush()

        return self

    def _intermediate_persist(self, epoch_index):
        if self.is_parallel:
            persist_moodel = self.classifier.module
        else:
            persist_moodel = self.classifier
        torch.save(
            persist_moodel.state_dict(),
            os.path.join(
                self.arguments.model_storage_dir,
                f"epoch_{epoch_index}_pytorch_model.bin",
            ),
        )

    def persist(self):
        """Persist this model into the passed directory."""

        logger.info("persist fine-tuned model")

        # self.classifier.load_state_dict(self.best_state_dict)
        # self.classifier.save_pretrained(self.arguments.model_storage_dir)

        self.model_config.architectures = [self.model_class]
        self.model_config.save_pretrained(self.arguments.model_storage_dir)
        torch.save(
            self.best_state_dict,
            os.path.join(self.arguments.model_storage_dir, "pytorch_model.bin"),
        )
        torch.save(
            self.best_acc_state_dict,
            os.path.join(
                self.arguments.model_storage_dir, "pytorch_model_best_acc.bin"
            ),
        )

        logger.info("fine-tuned model stored to %s", self.arguments.model_storage_dir)

    def _encode_tags(self, tags, encodings, corpus):
        logger.info("encoding tags")
        encoded_labels = []
        with tqdm(total=len(tags)) as pbar:
            for doc_labels, doc_offset, doc in zip(
                tags, encodings.offset_mapping, corpus,
            ):
                new_labels = []
                try:
                    for label, word in zip(doc_labels, doc):
                        tokens = self.tokenizer.tokenize(word)
                        if len(tokens) > 1:
                            if self.arguments.label_at_start:
                                new_labels.extend([label] + [-100]*(len(tokens)-1))
                            else:
                                new_labels.extend([-100]*(len(tokens)-1) + [label])
                        else:
                            new_labels.append(label)
                    
                    # create an empty array of -100
                    doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
                    arr_offset = np.array(doc_offset)

                    # set labels whose first offset position is 0 and the second is not 0
                    doc_enc_labels[
                        ~np.all(arr_offset == 0, axis=1)
                    ] = new_labels
                    encoded_labels.append(doc_enc_labels.tolist())
                except ValueError as e:
                    logger.warning("error encoding: %s", str(e))
                    logger.warning("tags: %s", doc_labels)
                    logger.warning("sample: %s", doc)
                    logger.warning("doc offset: %s", doc_offset)
                    raise e
                pbar.update(1)

        return encoded_labels

    def _encode_step1_features(self, all_step1_features, encodings, corpus):
        logger.info("encoding features")
        encoded_labels = []
        with tqdm(total=len(all_step1_features)) as pbar:
            for step1_features, doc_offset, sample in zip(
                all_step1_features, encodings.offset_mapping, corpus
            ):
                new_labels = []
                try:
                    for label, word in zip(step1_features, sample):
                        tokens = self.tokenizer.tokenize(word)
                        if len(tokens) > 1:
                            if self.arguments.label_at_start:
                                new_labels.extend([label] + [-100]*(len(tokens)-1))
                            else:
                                new_labels.extend([-100]*(len(tokens)-1) + [label])
                        else:
                            new_labels.append(label)

                    # create an empty array of -100
                    doc_enc_labels = np.ones(len(doc_offset), dtype=int) * 0
                    arr_offset = np.array(doc_offset)

                    # set labels whose first offset position is 0 and the second is not 0
                    doc_enc_labels[
                         ~np.all(arr_offset == 0, axis=1)
                    ] = new_labels
                    encoded_labels.append(doc_enc_labels.tolist())
                except ValueError as e:
                    logger.warning("error encoding: %s", str(e))
                    logger.warning("tags: %s", step1_features)
                    logger.warning("sample: %s", sample)
                    raise e
                pbar.update(1)

        return encoded_labels

    def _compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1),
            F.softmax(p, dim=-1),
            reduction="none",
        )

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        # # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def _train(self, iterator, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_val:
            self.classifier.train(False)
        else:
            self.classifier.train()

        in_epoch_steps = 0

        with tqdm(total=len(iterator)) as pbar:
            total_preds = []
            total_labels = []
            for batch in iterator:
                in_epoch_steps += 1
                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                step1_features = batch["step1_features"].to(self.device)

                if self.arguments.r_drop:
                    outputs_1 = self.classifier(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        token_type_ids=step1_features,
                    )
                    logits_1 = outputs_1.logits
                    if in_epoch_steps == 1:
                        logger.info("logits shape %s", logits_1.size())
                    logits_1_viewed = logits_1.view(-1, self.num_labels)
                    if in_epoch_steps == 1:
                        logger.info("viewed logits shape %s", logits_1_viewed.size())
                    loss_1 = F.cross_entropy(
                        logits_1_viewed,
                        labels.view(-1),
                        weight=self.class_weights,
                        reduction="mean",
                    )

                    outputs_2 = self.classifier(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    logits_2 = outputs_2.logits

                    logits_2_viewed = logits_2.view(-1, self.num_labels)
                    loss_2 = F.cross_entropy(
                        logits_2_viewed,
                        labels.view(-1),
                        weight=self.class_weights,
                        reduction="mean",
                    )

                    # cross entropy loss for classifier
                    ce_loss = 0.5 * (loss_1 + loss_2)
                    kl_loss = self._compute_kl_loss(logits_1, logits_2)

                    # carefully choose hyper-parameters
                    loss = ce_loss + self.arguments.r_alpha * kl_loss
                    logits = logits_1.add(logits_2) / 2  # average over two logits

                else:

                    outputs = self.classifier(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        class_weights=self.class_weights,
                        token_type_ids=step1_features,
                    )
                    logits = outputs.logits
                    loss = outputs.loss

                if self.is_parallel:
                    loss = loss.mean()

                if loss is not None:
                    if not is_val:
                        loss.backward()
                        optim.step()
                        if scheduler:
                            scheduler.step()
                        if self.total_steps % self.arguments.plot_steps == 0:
                            self.tensorboard_writter.add_scalar(
                                "Step Loss/train", loss, self.total_steps
                            )
                    else:
                        true_preds, true_labels = self._post_process(
                            logits, labels, attention_mask
                        )
                        total_preds.extend(true_preds)
                        total_labels.extend(true_labels)

                    epoch_loss += loss.item()
                    epoch_acc += self._accuracy(logits, attention_mask, labels)

                self.total_steps += 1

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.3f}",
                        "Avg_cum_loss": f"{epoch_loss/in_epoch_steps:.3f}",
                    }
                )

            if is_val:
                tested_labels = []
                target_names = []
                for label, label_id in self.label2id.items():
                    if label != NORMAL_TOKEN_TAG:
                        tested_labels.append(label_id)
                        target_names.append(label)
                report = classification_report(
                    total_labels,
                    total_preds,
                    labels=tested_labels,
                    digits=4,
                    target_names=target_names,
                    zero_division=1,
                )
                logger.info("validation report: \n %s", report)
            torch.cuda.empty_cache()

        return epoch_loss / in_epoch_steps, epoch_acc / in_epoch_steps

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _post_process(self, logits, labels, attention_mask):
        if self.device.type == "cuda":
            max_preds = logits.argmax(dim=-1).detach().cpu().numpy().flatten()
            flattened_labels = labels.detach().cpu().numpy().flatten()
            flattened_attention = attention_mask.detach().cpu().numpy().flatten()
        else:
            max_preds = logits.argmax(dim=-1).detach().numpy().flatten()
            flattened_labels = labels.detach().numpy().flatten()
            flattened_attention = attention_mask.detach().numpy().flatten()
        not_padding_labels = flattened_labels[flattened_attention == 1]
        not_padding_preds = max_preds[flattened_attention == 1]
        reduce_ignored = not_padding_labels >= 0
        true_labels = not_padding_labels[reduce_ignored]  # remove ignored -100
        true_preds = not_padding_preds[reduce_ignored]

        return true_preds, true_labels

    def _accuracy(self, logits, attention_mask, labels):
        if self.device.type == "cuda":
            max_preds = logits.argmax(dim=-1).detach().cpu().numpy().flatten()
            flattened_labels = labels.detach().cpu().numpy().flatten()
            flattened_attention = attention_mask.detach().cpu().numpy().flatten()
        else:
            max_preds = logits.argmax(dim=-1).detach().numpy().flatten()
            flattened_labels = labels.detach().numpy().flatten()
            flattened_attention = attention_mask.detach().numpy().flatten()

        not_padding_labels = flattened_labels[flattened_attention == 1]
        not_padding_preds = max_preds[flattened_attention == 1]
        reduce_ignored = not_padding_labels >= 0
        true_labels = not_padding_labels[reduce_ignored]  # remove ignored -100
        true_preds = not_padding_preds[reduce_ignored]

        if true_preds.shape[0] == true_labels.shape[0]:
            return np.sum(true_preds == true_labels) / true_preds.shape[0]

        return 0

    def run(self):
        self.tokenize().generate_dataset().fine_tune().persist()
