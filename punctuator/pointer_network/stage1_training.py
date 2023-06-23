import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as f
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup

from punctuator.utils import Models

from .model import PointerPunctuator

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 1


class Stage1TrainingArguments(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    # basic args
    training_corpus: List[str]
    validation_corpus: List[str]
    training_tags: List[int]
    validation_tags: List[int]
    pretrained_model: Optional[Models] = Models.BART
    load_local_model: bool = False
    model_config_name: str
    tokenizer_name: str
    model_name: str

    # training ars
    epoch: int
    model_storage_dir: str
    max_input_length: int = 320
    min_input_length: int = 160
    pointer_threshold: float = 0.5
    pointer_tolerance: int = 10
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


class Stage1TrainingPipeline:
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
            self.model = PointerPunctuator.from_pretrained(
                training_arguments.model_name
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
        if verbose:
            logger.info(f"Full model config: {self.model.config}")

        self.tokenizer = model_collection.tokenizer.from_pretrained(
            training_arguments.tokenizer_name,
            **training_arguments.additional_tokenizer_config,
        )

        if torch.cuda.is_available() and training_arguments.use_gpu:
            # if torch.cuda.device_count() > 1:
            #     self.model = torch.nn.DataParallel(self.model)
            #     self.model.cuda()
            #     self.device = torch.device("cuda")
            #     self.is_parallel = True
            # else: # FIXME: currently no parallel available
            self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
            self.model.to(self.device)
            self.is_parallel = False
        else:
            self.device = torch.device("cpu")
            self.is_parallel = False

        if verbose:
            logger.info(f"Full model architecture: {self.model}")

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
            print(f"error encoding: {str(e)}")

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
            print(f"error encoding: {str(e)}")

        return torch.stack(batched_doc_mask)

    def _generate_label(self, offset_mapping, current_tags):
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

            # logger.debug(f"offset_mapping shape: {offset_mapping.shape}, tags: {len(current_tags)}")
            # logger.debug(f"check: {doc_mask[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)], doc_mask[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)].shape}")
            # set labels whose first offset position is 0 and the second is not 0
            doc_mask[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = torch.tensor(
                current_tags
            )
        except ValueError as e:
            print(f"error encoding: {str(e)}")

        return doc_mask

    def _tokenize(self, input_sequence: List[str]):
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

    def _epoch_train(self, corpus, tags, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_val:
            self.model.train(False)
        else:
            self.model.train()

        in_epoch_steps = 0
        starting_index = 0
        ending_index = self.arguments.max_input_length
        encoder_outputs_lhs = None
        input_tokens = []
        input_tags = []
        tokenized_input_ids = None
        pointer_mask = None
        correct_tags = None

        # FIXME: implement a proper batching-based method, currently it's in a single sequence style
        # TODO: change to every time re-compute the encoder
        with tqdm(total=len(corpus)) as pbar:
            while True:
                output_tags = []
                groundtruth_tags = []
                loss = 0
                input_tokens += corpus[starting_index:ending_index]
                input_tags += tags[starting_index:ending_index]
                new_tokenized_input = self._tokenize(corpus[starting_index:ending_index])
                tokenized_input_ids = (
                    new_tokenized_input.input_ids
                    if tokenized_input_ids is None
                    else torch.cat(
                        [tokenized_input_ids, new_tokenized_input.input_ids], dim=1
                    )
                )
                pointer_mask = (
                    self._generate_mask(new_tokenized_input.offset_mapping.squeeze())
                    if pointer_mask is None
                    else torch.cat(
                        [
                            pointer_mask,
                            self._generate_mask(
                                new_tokenized_input.offset_mapping.squeeze()
                            ),
                        ]
                    )
                )
                correct_tags = (
                    self._generate_label(
                        new_tokenized_input.offset_mapping.squeeze(),
                        tags[starting_index:ending_index],
                    )
                    if correct_tags is None
                    else torch.cat(
                        [
                            correct_tags,
                            self._generate_label(
                                new_tokenized_input.offset_mapping.squeeze(),
                                tags[starting_index:ending_index],
                            ),
                        ]
                    )
                )
                new_encoder_outputs_lhs = self.model.encoder(
                    input_ids=new_tokenized_input.input_ids.to(self.device),
                    attention_mask=new_tokenized_input.attention_mask.to(self.device),
                ).last_hidden_state
                if encoder_outputs_lhs is None:
                    encoder_outputs_lhs = new_encoder_outputs_lhs
                else:
                    encoder_outputs_lhs = torch.cat(
                        [encoder_outputs_lhs, new_encoder_outputs_lhs], dim=1
                    )

                starting_index = ending_index
                in_epoch_steps += 1
                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                decoder_step = 0
                batch_step = 0
                while len(input_tokens) >= self.arguments.min_input_length:
                    batch_step += 1

                    decoder_step = (
                        decoder_step + 1
                        if tokenized_input_ids.squeeze()[0].item()
                        in self.tokenizer.all_special_ids
                        else decoder_step
                    )
                    decoder_ids = tokenized_input_ids[:, : decoder_step + 1]
                    model_output = self.model.detect_boundary(
                        decoder_input_ids=decoder_ids.to(self.device),
                        decoder_input_index=decoder_step,
                        pointer_mask=pointer_mask.to(self.device),
                        encoder_outputs_last_hidden_state=encoder_outputs_lhs,
                    )
                    predicted_position = model_output.argmax(dim=-1).squeeze().item()
                    possibility = model_output[:, predicted_position].squeeze().item()
                    loss += f.binary_cross_entropy(
                        model_output.squeeze()[:predicted_position+1],
                        correct_tags[:predicted_position+1].float().to(self.device),
                    )  # loss computed till predicted_position. noqa 401

                    # only start a new batch if either higher than threshold or reach the tolerance
                    if (
                        possibility >= self.arguments.pointer_threshold
                        or batch_step >= self.arguments.pointer_tolerance
                    ):
                        tokenized_input_ids = tokenized_input_ids[
                            :, predicted_position + 1 :
                        ]
                        correct_tags = correct_tags[predicted_position + 1 :]
                        encoder_outputs_lhs = encoder_outputs_lhs[
                            :, predicted_position + 1 :, :
                        ]
                        original_position = torch.sum(
                            pointer_mask[: predicted_position + 1].squeeze() > 0
                        ).item()
                        pointer_mask = pointer_mask[predicted_position + 1 :]
                        output_tags += [0] * original_position + [1]
                        groundtruth_tags.extend(input_tags[: original_position + 1])
                        input_tokens = input_tokens[original_position + 1 :]
                        input_tags = input_tags[original_position + 1 :]
                        pbar.update(original_position + 1)
                    elif (
                        ending_index - (len(corpus) - 1)
                    ) > self.arguments.min_input_length and decoder_step >= (
                        len(input_tokens)
                    ):
                        output_tags.extend([0] * len(input_tokens))
                        pbar.update(len(input_tokens))
                    else:
                        decoder_step += 1

                required_length = self.arguments.max_input_length - len(input_tokens)
                ending_index = starting_index + required_length
                if (
                    starting_index >= len(corpus) - 1
                ):  # means already loop over all corpus
                    break

                if not is_val:
                    loss.backward(retain_graph=True)  # batch's loss
                    optim.step()
                    if scheduler:
                        scheduler.step()
                    if self.total_steps % self.arguments.plot_steps == 0:
                        self.tensorboard_writter.add_scalar(
                            "Step Loss/train", loss, self.total_steps
                        )

                self.total_steps += 1

                epoch_loss += loss.item()

                epoch_acc += self._accuracy(output_tags, groundtruth_tags)

                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.2f}",
                        "Avg_cum_loss": f"{epoch_loss/in_epoch_steps:.2f}",
                    }
                )

        return {
            "epoch_loss": epoch_loss / in_epoch_steps,
            "epoch_acc": epoch_acc / in_epoch_steps,
        }

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _accuracy(self, predictions, labels):
        if len(predictions) == len(labels):
            return np.sum(predictions == labels) / len(labels)

        return 0

    def train(self):
        logger.info("start training")

        optim = AdamW(self.model.parameters(), lr=1e-5)

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
                    corpus=self.arguments.training_corpus,
                    tags=self.arguments.training_tags,
                    optim=optim, 
                    scheduler=scheduler
                )
                val_epoch_outputs = self._epoch_train(
                    corpus=self.arguments.validation_corpus,
                    tags=self.arguments.validation_tags,
                    optim=optim, 
                    scheduler=scheduler,
                    is_val=True
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

        output_config_file = (
            f"{self.arguments.model_storage_dir}/model_config.json"
        )
        self.model.config.to_json_file(output_config_file, use_diff=True)

        torch.save(
            self.best_state_dict,
            os.path.join(self.arguments.model_storage_dir, "whole_model.bin"),
        )

        logger.info(f"model stored to {self.arguments.model_storage_dir}")
