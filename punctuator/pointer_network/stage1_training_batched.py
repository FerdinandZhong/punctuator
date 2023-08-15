import logging
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.optim.adamw as adamw
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from punctuator.utils import Models

from .model import PointerPunctuator

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 1


def data_collator(batch):
    assert all("input_sequences" in x for x in batch)
    assert all("tags" in x for x in batch)
    return {
        "input_sequences": [x["input_sequences"] for x in batch],
        "tags": [x["tags"] for x in batch],
    }


class EncodingDataset(Dataset):
    def __init__(self, input_seqeunces, tags):
        self.input_seqeunces = input_seqeunces
        self.tags = tags

    def __getitem__(self, idx):
        item = {"input_sequences": self.input_seqeunces[idx], "tags": self.tags[idx]}
        return item

    def __len__(self):
        return len(self.tags)


class Stage1TrainingBatchedArguments(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    # basic args
    training_corpus: List[List[str]]
    validation_corpus: List[List[str]]
    training_tags: List[List[int]]
    validation_tags: List[List[int]]
    pretrained_model: Optional[Models] = Models.BART
    load_local_model: bool = False
    model_config_name: str
    tokenizer_name: str
    model_name: str

    # training ars
    epoch: int
    model_storage_dir: str
    batch_size: int = 16
    pointer_threshold: float = 0.5
    pointer_tolerance: int = 10
    backward_count: int = 1
    early_stop_count: Optional[int] = 3
    use_gpu: Optional[bool] = True
    gpu_device: Optional[int] = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    warm_up_steps: int = 1000
    plot_steps: int = 50
    tensorboard_log_dir: Optional[str] = "runs"
    intermediate_persist_step: int = 5

    # model args
    additional_model_config: Optional[Dict] = {}
    additional_tokenizer_config: Optional[Dict] = {}


class Stage1TrainingBatchedPipeline:
    """Two steps model:

    * punctuation position detection based on pointer network.
    * punctuation prediction at predicted positions.

    """

    def __init__(self, training_arguments, verbose=False):
        """Training pipeline

        Args:
            training_arguments (TrainingArguments): arguments passed to training pipeline
        """
        self.arguments = training_arguments

        self.tensorboard_writter = SummaryWriter(training_arguments.tensorboard_log_dir)

        model_collection = training_arguments.pretrained_model.value

        if training_arguments.load_local_model:
            config = model_collection.config.from_pretrained(
                os.path.join(training_arguments.model_config_name, "model_config.json"),
                **training_arguments.additional_model_config,
            )
            logger.info(config)
            self.model = PointerPunctuator(config)
            self.model.load_state_dict(
                torch.load(
                    os.path.join(training_arguments.model_name, "whole_model.bin")
                )
            )
        else:
            pretrained_model = model_collection.model.from_pretrained(
                training_arguments.model_name
            )
            config = model_collection.config.from_pretrained(
                training_arguments.model_config_name,
                **training_arguments.additional_model_config,
            )
            self.model = PointerPunctuator(config, pretrained_model)
            self.encoder = self.model.encoder
        if verbose:
            logger.info(f"Full model config: {self.model.config}")

        self.tokenizer = model_collection.tokenizer.from_pretrained(
            training_arguments.tokenizer_name,
            **training_arguments.additional_tokenizer_config,
        )

        if torch.cuda.is_available() and training_arguments.use_gpu:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.encoder = torch.nn.DataParallel(self.encoder)
                self.model.cuda()
                self.device = torch.device("cuda")
                self.is_parallel = True
            else:
                self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
                self.model.to(self.device)
                self.is_parallel = False
        else:
            self.device = torch.device("cpu")
            self.is_parallel = False

        if verbose:
            logger.info(f"Full model architecture: {self.model}")

        self.training_dataset = EncodingDataset(
            self.arguments.training_corpus, self.arguments.training_tags
        )
        self.val_dataset = EncodingDataset(
            self.arguments.validation_corpus, self.arguments.validation_tags
        )

    def _generate_mask(self, offset_mapping):
        """Generate mask based on the offset mapping for pointer network
        To avoid pointing to the special token e.g. <s> and the non-ending token of a word.

        Args:
            offset_mapping (tensor): Offset mapping generated by the tokenizer

        Returns:
            tensor: The mapping mask. (Only valuable tokens have a mask as 1 rest are -100)
        """
        try:
            # create an empty array of -100
            doc_mask = torch.zeros(offset_mapping.shape[0], dtype=int)
            arr_offset = np.array(offset_mapping)

            # set labels whose first offset position is 0 and the second is not 0
            doc_mask[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 1
        except ValueError as e:
            logger.error(f"error encoding: {str(e)}")

        return doc_mask

    def _generate_batch_mask(self, batched_offset_mapping):
        """Generate mask based on the offset mapping for pointer network
        To avoid pointing to the special token e.g. <s> and the non-ending token of a word.

        Args:
            batched_offset_mapping (tensor): Offset mapping generated by the tokenizer

        Returns:
            tensor: The mapping mask. (Only valuable tokens have a mask as 1 rest are -100)
        """
        try:
            # create an empty array of -100
            batched_doc_mask = []
            for offset_mapping in batched_offset_mapping:
                batched_doc_mask.append(self._generate_mask(offset_mapping))
        except ValueError as e:
            logger.error(f"error encoding: {str(e)}")

        return torch.stack(batched_doc_mask)

    def _generate_label(self, offset_mapping, correct_tags):
        """Generate tags based on the offset mapping for loss

        Args:
            offset_mapping (tensor): Offset mapping generated by the tokenizer

        Returns:
            tensor: The mapping mask. (Only valuable tokens have a mask as 1 rest are -100)
        """
        try:
            # create an empty array of -100
            doc_mask = torch.zeros(offset_mapping.shape[0], dtype=int)
            arr_offset = np.array(offset_mapping)

            # logger.debug(f"offset_mapping shape: {offset_mapping.shape}, tags: {len(correct_tags)}")
            # logger.debug(f"check: {doc_mask[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)], doc_mask[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)].shape}")
            # set labels whose first offset position is 0 and the second is not 0
            doc_mask[
                (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
            ] = correct_tags.clone().detach()
        except ValueError as e:
            logger.error(f"error encoding: {str(e)}")

        return doc_mask

    def _generate_batch_label(self, batched_offset_mapping, batched_correct_tags):
        try:
            # create an empty array of -100
            batched_labels = []
            for offset_mapping, correct_tags in zip(
                batched_offset_mapping, batched_correct_tags
            ):
                batched_labels.append(
                    self._generate_label(offset_mapping, correct_tags)
                )
        except ValueError as e:
            logger.error(f"error encoding: {str(e)}")

        return torch.stack(batched_labels)

    def _generate_padding_indices(self, batched_input_ids):
        padding_startings = []
        for batched_input_id in batched_input_ids:
            padding_starting = (
                (batched_input_id == self.tokenizer.pad_token_id)
                .nonzero()
                .squeeze()
                .tolist()
            )
            if isinstance(padding_starting, list) and len(padding_starting) > 0:
                padding_startings.append(padding_starting[0])
            else:
                padding_startings.append(len(batched_input_id) - 1)

        return padding_startings

    def _tokenize(self, input_sequence: Union[List[str], List[List[str]]]):
        """Tokenize at sequence level. (not batch)

        Args:
            input_sequence(List[str]): List of tokens to be tokenized

        Returns:
            Dict: Tokenized input (input_ids, attention_mask and offset_mapping)
        """
        tokenized_input = self.tokenizer(
            input_sequence,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        return tokenized_input

    def _epoch_train(
        self, iterator, optim, scheduler=None, criterion=None, is_val=False
    ):
        epoch_loss = 0
        epoch_total_acc = 0
        if is_val:
            self.model.train(False)
        else:
            self.model.train()

        in_epoch_steps = 1
        with tqdm(total=len(iterator)) as pbar:
            for batch in iterator:
                # TODO: solve the data loader issue

                groundtruth_tags = torch.tensor(batch["tags"])
                input_sequences = batch["input_sequences"]
                # print(f"batch input_sequences: {input_sequences}")
                batch_size = len(input_sequences)

                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                output_tags = [[]] * batch_size
                batch_step = 0
                batch_total_loss = 0
                remaining_indexes = list(range(batch_size))
                while batch_size > 0:
                    batch_step += 1
                    # every time need to do the tokenization (for padding)
                    tokenized_inputs = self._tokenize(input_sequences)
                    tokenized_input_ids = tokenized_inputs["input_ids"]
                    tokenized_attention_masks = tokenized_inputs["attention_mask"]
                    pointer_mask = self._generate_batch_mask(
                        tokenized_inputs["offset_mapping"]
                    )
                    correct_tags = self._generate_batch_label(
                        tokenized_inputs["offset_mapping"], groundtruth_tags
                    )
                    padding_startings = self._generate_padding_indices(
                        tokenized_input_ids
                    )
                    # TODO: remove the input that reaching the padding stage or over the length
                    encoder_outputs_lhs = self.encoder(
                        input_ids=tokenized_input_ids,
                        attention_mask=tokenized_attention_masks,
                    ).last_hidden_state
                    model_output = self.model(
                        decoder_input_ids=tokenized_input_ids.to(self.device),
                        decoder_input_index=0,
                        pointer_mask=pointer_mask.to(self.device),
                        encoder_outputs_last_hidden_state=encoder_outputs_lhs,
                    )
                    # for batching, every step counts
                    predicted_position_list = model_output.argmax(dim=-1).cpu().numpy()
                    nearest_boundary = correct_tags.argmax(-1)

                    step_loss = criterion(
                        model_output,
                        nearest_boundary.to(self.device),
                        # correct_tags.float().to(self.device)
                    )
                    if self.is_parallel:
                        step_loss = step_loss.mean()
                    batch_total_loss += step_loss.cpu().item()
                    if not is_val:
                        step_loss.backward()  # batch's loss
                        optim.step()
                        if scheduler:
                            scheduler.step()
                        optim.zero_grad()

                    # to generate the new batch
                    new_input_sequences = []
                    new_groundtruth_tags = []
                    new_remaining_indexes = []
                    new_index = 0
                    for (
                        index,
                        predicted_position,
                        padding_starting,
                        input_sequence,
                        groundtruth_tag,
                    ) in zip(
                        remaining_indexes,
                        predicted_position_list,
                        padding_startings,
                        input_sequences,
                        groundtruth_tags,
                    ):
                        if predicted_position >= padding_starting:
                            output_tags[index].extend([0] * (padding_starting + 1))
                        else:
                            output_tags[index].extend([0] * predicted_position + [1])
                            remaining_tag = correct_tags[new_index][predicted_position + 1 :]
                            if 1 in remaining_tag:
                                new_remaining_indexes.append(index)
                                current_tokenized_input_ids = tokenized_input_ids[new_index]
                                current_tokenized_input_ids = current_tokenized_input_ids[
                                    predicted_position + 1 :
                                ]
                                current_tokenized_attention_mask = (
                                    tokenized_attention_masks[new_index]
                                )
                                current_tokenized_attention_mask = (
                                    current_tokenized_attention_mask[
                                        predicted_position + 1 :
                                    ]
                                )
                                current_pointer_mask = pointer_mask[new_index]

                                original_position = torch.sum(
                                    current_pointer_mask[: predicted_position + 1].squeeze()
                                    > 0
                                ).item()
                                # TODO: generate new correct tags
                                new_input_sequences.append(
                                    input_sequence[original_position + 1 :]
                                )
                                new_groundtruth_tags.append(
                                    groundtruth_tag[original_position + 1 :]
                                )
                        new_index += 1

                    input_sequences = new_input_sequences
                    groundtruth_tags = new_groundtruth_tags
                    remaining_indexes = new_remaining_indexes
                    batch_size = len(input_sequences)

                    if in_epoch_steps == 1:
                        logger.info(f"remaining batch size: {batch_size}")
                        logger.info(
                            f"remaining indexes: {remaining_indexes}"
                        )
                        logger.info(f"boundaries: {nearest_boundary}")

                self.total_steps += 1
                batch_loss = batch_total_loss / batch_step

                in_epoch_steps += 1
                epoch_loss += batch_loss
                last_batch_accuracy = self._accuracy(output_tags, batch["tags"])
                epoch_total_acc += last_batch_accuracy

                pbar.set_postfix(
                    {
                        "Last_loss": f"{batch_loss:.2f}",
                        "Avg_cum_loss": f"{epoch_loss/in_epoch_steps:.2f}",
                        "Last_acc": f"{last_batch_accuracy:.4f}"
                    }
                )
                pbar.update(1)

                if self.total_steps % self.arguments.plot_steps == 0:
                    self.tensorboard_writter.add_scalar(
                        "Step Loss/train", batch_loss, self.total_steps
                    )
                    step_acc = epoch_total_acc / self.total_steps
                    self.tensorboard_writter.add_scalar(
                        "Step Accuracy/train", step_acc, self.total_steps
                    )
                    logger.info(f"current accuracy of epoch: {step_acc:.2f}")

        return {
            "epoch_loss": epoch_loss / in_epoch_steps,
            "epoch_acc": epoch_total_acc / in_epoch_steps,
        }

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _accuracy(self, prediction_logits, labels):
        true_preds = np.array(prediction_logits).flatten()
        true_labels = np.array(labels).flatten()
        if true_preds.shape[0] == true_labels.shape[0]:
            print(np.sum(true_preds == true_labels))
            return np.sum(true_preds == true_labels) / true_labels.shape[0]

        return 0

    def train(self):
        logger.info("start training")

        train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.arguments.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.arguments.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

        optim = adamw.AdamW(self.model.parameters(), lr=1e-6)
        # criterion = torch.nn.BCE()
        criterion = torch.nn.CrossEntropyLoss()

        scheduler = get_constant_schedule_with_warmup(
            optim,
            num_warmup_steps=self.arguments.warm_up_steps,
        )

        best_valid_loss = float("inf")
        no_improvement_count = 0
        self.total_steps = 0

        with tqdm(total=self.arguments.epoch) as pbar:
            for epoch in range(self.arguments.epoch):
                pbar.set_description(f"Processing epoch: {epoch + 1}")

                start_time = time.time()
                train_epoch_outputs = self._epoch_train(
                    iterator=train_loader,
                    optim=optim,
                    scheduler=scheduler,
                    criterion=criterion,
                )
                val_epoch_outputs = self._epoch_train(
                    iterator=val_loader,
                    optim=optim,
                    scheduler=scheduler,
                    criterion=criterion,
                    is_val=True,
                )

                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/train", train_epoch_outputs["epoch_loss"], epoch + 1
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/valid", val_epoch_outputs["epoch_loss"], epoch + 1
                )

                self.tensorboard_writter.add_scalar(
                    "Epoch acc/train",
                    train_epoch_outputs["epoch_acc"],
                    epoch + 1,
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch acc/valid",
                    val_epoch_outputs["epoch_acc"],
                    epoch + 1,
                )

                end_time = time.time()

                epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

                logger.info(
                    f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
                )
                logger.info(f"Train Loss: {train_epoch_outputs['epoch_loss']:.3f}")
                logger.info(f"Val. Loss: { val_epoch_outputs['epoch_loss']:.3f}")
                logger.info(f"Train Acc: {train_epoch_outputs['epoch_acc'] * 100:.2f}%")
                logger.info(f"Val. Acc: {val_epoch_outputs['epoch_acc'] * 100:.2f}%")

                pbar.update(1)

                if (epoch + 1) % self.arguments.intermediate_persist_step == 0:
                    logger.info(
                        f"Save the intermediate checkpoint for epoch: {epoch + 1}"
                    )
                    self._intermediate_persist(epoch + 1)

                if val_epoch_outputs["epoch_loss"] < best_valid_loss:
                    best_valid_loss = val_epoch_outputs["epoch_loss"]
                    if self.is_parallel:
                        self.best_state_dict = self.model.module.state_dict()
                    else:
                        self.best_state_dict = self.model.state_dict()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if (
                        self.arguments.early_stop_count > 0
                        and no_improvement_count >= self.arguments.early_stop_count
                    ):
                        logger.info(
                            f"No improvement for past {no_improvement_count} epochs, early stop training."
                        )
                        return self

            logger.info("fine-tune finished")

        self.tensorboard_writter.flush()

        return self

    def _intermediate_persist(self, epoch_index):
        if self.is_parallel:
            persist_moodel = self.model.module
        else:
            persist_moodel = self.model
        torch.save(
            persist_moodel.state_dict(),
            os.path.join(
                self.arguments.model_storage_dir, f"epoch_{epoch_index}_whole_model.bin"
            ),
        )

    def persist(self):
        """Persist this model into the passed directory."""

        logger.info("persist fine-tuned model")

        # self.classifier.load_state_dict(self.best_state_dict)
        # self.classifier.save_pretrained(self.arguments.model_storage_dir)

        output_config_file = f"{self.arguments.model_storage_dir}/model_config.json"
        self.model.config.to_json_file(output_config_file, use_diff=True)

        if self.is_parallel:
            self.model.module.load_state_dict(self.best_state_dict)
            torch.save(
                self.model.module.lm_model.state_dict(),
                os.path.join(self.arguments.model_storage_dir, "whole_model.bin"),
            )
        else:
            self.model.load_state_dict(self.best_state_dict)
            torch.save(
                self.model.lm_model.state_dict(),
                os.path.join(self.arguments.model_storage_dir, "whole_model.bin"),
            )

        logger.info(f"model stored to {self.arguments.model_storage_dir}")
