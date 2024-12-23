from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from fastkan import FastKAN as KAN
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel, BertForTokenClassification, BertEncoder
from transformers.models.roberta.modeling_roberta import *
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead, ModernBertModel, ModernBertForTokenClassification


class BertFocalLossForTokenClassification(BertForTokenClassification):
    def __init__(self, config, backbone_model: BertModel = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.bert = backbone_model
        else:
            self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
        self._loss_fct = None

    def set_loss_fct(self, focal_loss):
        self._loss_fct = focal_loss.to(self.device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self._loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1), class_weights
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaFocalLossForTokenClassification(RobertaForTokenClassification):
    def __init__(self, config, backbone_model: RobertaModel = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.roberta = backbone_model
        else:
            self.roberta = RobertaModel(config, add_pooling_layer=False)

        if config.type_vocab_size >= 2:
            self.roberta.embeddings.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
            self.roberta.embeddings.token_type_embeddings.weight.data.normal_(
                mean=0.0, std=config.initializer_range
            )

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
        self._loss_fct = None

    def set_loss_fct(self, focal_loss):
        self._loss_fct = focal_loss.to(self.device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self._loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1), class_weights
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ModernBertFocalLossForTokenClassification(ModernBertForTokenClassification):
    def __init__(self, config, backbone_model: ModernBertModel = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.mobilebert = backbone_model
        else:
            self.mobilebert = ModernBertModel(config)

        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
        self._loss_fct = None

    def set_loss_fct(self, focal_loss):
        self._loss_fct = focal_loss.to(self.device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        self._maybe_set_compile()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)


        loss = None
        if labels is not None:
            loss = self._loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1), class_weights
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertEmbeddingsStep2(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.punct_positions_embedding = nn.Embedding(
            2, config.hidden_size
        )  # num of features == 2

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        punct_positions: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        punct_positions_embeddings = self.punct_positions_embedding(punct_positions)

        embeddings = inputs_embeds + token_type_embeddings + punct_positions_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelStep2(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = BertEmbeddingsStep2(config)


class BertFocalLossForTokenClassificationStep2(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModelStep2(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
        self._loss_fct = None

    def set_loss_fct(self, focal_loss):
        self._loss_fct = focal_loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self._loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1), class_weights
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaEmbeddingsStep2(nn.Module):
    def __init__(self, config, roberta_embedding):
        super().__init__()
        self.punct_positions_embedding = nn.Embedding(
            2, config.hidden_size
        )  # num of features == 2
        self.punct_positions_embedding.weight.data.normal_(
            mean=0.0, std=config.initializer_range
        )
        self.roberta_embedding = roberta_embedding

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        original_embeddings = self.roberta_embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        punct_positions_embeddings = self.punct_positions_embedding(token_type_ids)

        embeddings = original_embeddings + punct_positions_embeddings

        return embeddings


class RobertaModelStep2(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = RobertaEmbeddings(config)


# class RobertaFocalLossForTokenClassificationStep2(RobertaForTokenClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.roberta = RobertaModelStep2(config, add_pooling_layer=False)

#         classifier_dropout = (
#             config.classifier_dropout
#             if config.classifier_dropout is not None
#             else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.post_init()
#         self._loss_fct = None

#     def set_loss_fct(self, focal_loss):
#         self._loss_fct = focal_loss

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         class_weights: Optional[torch.Tensor] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)

#         loss = None
#         if labels is not None:
#             loss = self._loss_fct(
#                 logits.view(-1, self.num_labels), labels.view(-1), class_weights
#             )

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class MLPStep2Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        output = self.dense(hidden_states)
        output = self.activation(output)
        output = self.output(output)
        return output


class FocalLossForTokenClassificationStep2(BertFocalLossForTokenClassification):
    def __init__(
        self,
        config,
        backbone_model: BertFocalLossForTokenClassification = None,
        freeze_encoder: bool = False,
        use_kan: bool = False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.bert = backbone_model.bert
        else:
            self.bert = BertModel(config, add_pooling_layer=False)

        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if not use_kan:
            self.classifier = MLPStep2Classifier(config)
        else:
            self.classifier = KAN(
                [
                    config.hidden_size,
                    config.hidden_size // 2,
                    config.num_labels,
                ]
            )
        self.post_init()
        self._loss_fct = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if labels is not None:
            labels[token_type_ids == 0] = -100
        else:
            mean_loss = None

        # Create a tensor to store restored logits
        restored_logits = torch.full(
            (token_type_ids.size(0), token_type_ids.size(1), self.num_labels),
            float(-1),
            device=token_type_ids.device,
        ).to(token_type_ids.device)

        # Set the first class to have the maximum probability by default (logit value of 0)
        restored_logits[:, :, 0] = 0

        # Loop through the batch
        for batch_index in range(
            token_type_ids.size(0)
        ):  # Loop over the batch dimension

            mask = token_type_ids[batch_index] > 0
            selected_hiddenstates = sequence_output[batch_index][mask]
            if selected_hiddenstates.size(0) > 0:
                logits = self.classifier(selected_hiddenstates.cuda())
                restored_logits[batch_index][mask] = logits

        if labels is not None:
            mean_loss = self._loss_fct(
                restored_logits.view(-1, self.num_labels),
                labels.view(-1),
                class_weights,
            )

        if not return_dict:
            output = (restored_logits,) + outputs[2:]
            return ((mean_loss,) + output) if mean_loss is not None else output

        return TokenClassifierOutput(
            loss=mean_loss,
            logits=restored_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaFocalLossForTokenClassificationStep2(
    RobertaFocalLossForTokenClassification
):
    def __init__(
        self,
        config,
        backbone_model: BertFocalLossForTokenClassification = None,
        freeze_encoder: bool = False,
        use_kan: bool = False,
        positonal_importance_alpha: int = 2,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.roberta = backbone_model.roberta
        else:
            self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.roberta.embeddings.token_type_embeddings = nn.Embedding(
            2, config.hidden_size
        )
        self.roberta.embeddings.token_type_embeddings.weight.data.normal_(
            mean=0.0, std=config.initializer_range
        )

        # self.roberta.embeddings = RobertaEmbeddingsStep2(config, self.roberta.embeddings)

        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if not use_kan:
            # self.classifier = MLPStep2Classifier(config)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = KAN(
                [
                    config.hidden_size,
                    config.hidden_size // 2,
                    config.num_labels,
                ]
            )
        self.post_init()
        self._loss_fct = None
        self._positonal_importance_alpha = float(positonal_importance_alpha)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # positional_importances = torch.full(
        #     token_type_ids.shape, float(1), device=token_type_ids.device
        # )

        # positional_importances[token_type_ids == 1] = self._positonal_importance_alpha

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss = self._loss_fct(
        #         logits.view(-1, self.num_labels),
        #         labels.view(-1),
        #         alpha=class_weights,
        #         positional_importances=positional_importances.view(-1),
        #     )

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((mean_loss,) + output) if loss is not None else output

        # return TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        if labels is not None:
            labels[token_type_ids == 0] = -100
        else:
            mean_loss = None

        # Create a tensor to store restored logits
        restored_logits = torch.full(
            (token_type_ids.size(0), token_type_ids.size(1), self.num_labels),
            float(-1),
            device=token_type_ids.device,
        ).to(token_type_ids.device)

        # Set the first class to have the maximum probability by default (logit value of 0)
        restored_logits[:, :, 0] = 0

        # Loop through the batch
        for batch_index in range(
            token_type_ids.size(0)
        ):  # Loop over the batch dimension

            mask = token_type_ids[batch_index] > 0
            selected_hiddenstates = sequence_output[batch_index][mask]
            if selected_hiddenstates.size(0) > 0:
                logits = self.classifier(selected_hiddenstates.cuda())
                restored_logits[batch_index][mask] = logits

        if labels is not None:
            mean_loss = self._loss_fct(
                restored_logits.view(-1, self.num_labels),
                labels.view(-1),
                class_weights,
            )

        if not return_dict:
            output = (restored_logits,) + outputs[2:]
            return ((mean_loss,) + output) if mean_loss is not None else output

        return TokenClassifierOutput(
            loss=mean_loss,
            logits=restored_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
