from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from fastkan import FastKAN as KAN
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import *


class BertKanForTokenClassification(BertPreTrainedModel):
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
        self.classifier = KAN(
            [
                config.hidden_size,
                config.num_labels,
            ]
        )
        # Initialize weights and apply final processing
        self.post_init()
        self._loss_fct = None

    def set_loss_fct(self, focal_loss):
        self._loss_fct = focal_loss

    # TODO: add r-drop
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
        batch_size, sequence_length, hidden_size = sequence_output.shape

        kan_input = sequence_output.reshape(batch_size * sequence_length, hidden_size)
        kan_output = self.classifier(kan_input)
        logits = kan_output.view(batch_size, sequence_length, self.num_labels)

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


class BertLayerKan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # 0616 ==================================================================
        # self.kan = KAN(
        #     [
        #         config.hidden_size,
        #         config.hidden_size,
        #     ]
        # )
        # 0616 ==================================================================
        # 0615 ==================================================================
        # self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # self.kan = KAN(
        #     [
        #         config.hidden_size // 2,
        #         config.hidden_size * 2,
        #         config.hidden_size // 2,
        #     ]
        # )
        # self.dense_2 = nn.Linear(config.hidden_size // 2, config.hidden_size)
        # 0615 ==================================================================
        # 0611 ==================================================================
        if config.hidden_size >= 1024:
            self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
            self.kan = KAN(
                [
                    config.hidden_size // 2,
                    config.hidden_size * 2,
                    config.hidden_size // 2,
                ]
            )
            self.dense_2 = nn.Linear(config.hidden_size // 2, config.hidden_size)
        else:
            self.dense_1, self.dense_2 = None, None
            self.kan = KAN(
                [
                    config.hidden_size,
                    config.hidden_size * 4,
                    config.hidden_size,
                ]
            )
        # 0611 ==================================================================
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        if self.dense_1 is not None and self.dense_2 is not None:
            hidden_states = self.dense_1(hidden_states)
            kan_output = self.dense_2(self.kan(hidden_states))
        else:
            kan_output = self.kan(hidden_states)
        # 0616 ==================================================================
        # kan_output = self.kan(hidden_states)
        # 0616 ==================================================================
        kan_output = self.dropout(kan_output)
        return self.LayerNorm(kan_output + input_tensor)


class BertKanLayer(BertLayer):
    """Bert with Kan layer. Attention still follows the original architecture.

    But replace every encoder layer's output MLPs with KANs
    """

    def __init__(self, config, bert_layer: BertLayer = None):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
        if bert_layer is None:
            self.attention = BertAttention(config)
            if self.add_cross_attention:
                self.crossattention = BertAttention(
                    config, position_embedding_type="absolute"
                )
        else:
            self.attention = bert_layer.attention
            if self.add_cross_attention:
                self.crossattention = bert_layer.crossattention
        self.output = BertLayerKan(config)

    def feed_forward_chunk(self, attention_output):
        layer_output = self.output(attention_output, attention_output)
        return layer_output


class BertKanEncoder(BertEncoder):
    def __init__(self, config, layer_list: nn.ModuleList = None):
        super().__init__(config)
        self.config = config
        if layer_list is None:
            self.layer = nn.ModuleList(
                [BertKanLayer(config) for _ in range(config.num_hidden_layers)]
            )
        else:
            self.layer = nn.ModuleList(
                [BertKanLayer(config, bert_layer) for bert_layer in layer_list]
            )
        self.gradient_checkpointing = False


class BertKanModel(BertModel):
    def __init__(
        self, config, add_pooling_layer=True, backbone_model: BertModel = None
    ):
        super().__init__(config)

        if backbone_model is not None:
            self.embeddings = backbone_model.embeddings
            self.encoder = BertKanEncoder(config, backbone_model.encoder.layer)
            self.pooler = backbone_model.pooler
        else:
            self.embeddings = BertEmbeddings(config)
            self.encoder = BertKanEncoder(config)

            self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        self.post_init()


class BertKanForTokenClassification2(BertKanForTokenClassification):
    def __init__(self, config, backbone_model: BertModel = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.bert = BertKanModel(
                config, add_pooling_layer=False, backbone_model=backbone_model
            )
        else:
            self.bert = BertKanModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = KAN(
            [
                config.hidden_size,
                config.hidden_size // 2,
                config.hidden_size // 4,
                config.hidden_size // 8,
                config.num_labels,
            ]
        )
        # Initialize weights and apply final processing
        self.post_init()

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
        batch_size, sequence_length, hidden_size = sequence_output.shape

        kan_input = sequence_output.reshape(batch_size * sequence_length, hidden_size)
        kan_output = self.classifier(kan_input)
        logits = kan_output.view(batch_size, sequence_length, self.num_labels)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weights, reduction="mean")
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertKanForTokenClassificationFocalLoss(BertKanForTokenClassification):
    def __init__(self, config, backbone_model: BertModel = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        if backbone_model is not None:
            self.bert = BertKanModel(
                config, add_pooling_layer=False, backbone_model=backbone_model
            )
        else:
            self.bert = BertKanModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = KAN(
            [
                config.hidden_size,
                config.hidden_size // 2,
                config.hidden_size // 4,
                config.hidden_size // 8,
                config.num_labels,
            ]
        )
        # 0615 ==================================================================
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 0615 ==================================================================
        # Initialize weights and apply final processing
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
        batch_size, sequence_length, hidden_size = sequence_output.shape

        kan_input = sequence_output.reshape(batch_size * sequence_length, hidden_size)
        kan_output = self.classifier(kan_input)
        logits = kan_output.view(batch_size, sequence_length, self.num_labels)

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
