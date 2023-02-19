import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertLMPredictionHead,
    BertModel,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class BertForSpanPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    span_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertLMSpanPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.span_decoder = nn.Linear(config.hidden_size, out_features=2)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.span_decoder(hidden_states)
        return hidden_states


class BertOnlySpanMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.span_predictions = BertLMSpanPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        span_prediction_scores = self.span_predictions(sequence_output)
        return span_prediction_scores


class BertMLMHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.span_predictions = BertLMSpanPredictionHead(config)
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        span_prediction_scores = self.span_predictions(sequence_output)
        prediction_scores = self.predictions(sequence_output)
        return span_prediction_scores, prediction_scores


class BertForSpanMaskedLM(BertPreTrainedModel):
    def __init__(self, config, span_only=True):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.span_only = span_only
        if span_only:
            self.cls = BertOnlySpanMLMHead(config)
        else:
            self.cls = BertMLMHeads(config)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        span_labels: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForSpanPreTrainingOutput]:
        """One input_ids, two kinds of labels.

        Args:
            input_ids (Optional[torch.Tensor]): _description_
            span_labels (Optional[torch.Tensor]): _description_
            labels (Optional[torch.Tensor], optional): _description_. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): _description_. Defaults to None.
            position_ids (Optional[torch.Tensor], optional): _description_. Defaults to None.
            head_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            inputs_embeds (Optional[torch.Tensor], optional): _description_. Defaults to None.
            encoder_hidden_states (Optional[torch.Tensor], optional): _description_. Defaults to None.
            encoder_attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            output_attentions (Optional[bool], optional): _description_. Defaults to None.
            output_hidden_states (Optional[bool], optional): _description_. Defaults to None.
            return_dict (Optional[bool], optional): _description_. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor], BertForSpanPreTrainingOutput]: _description_
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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        loss_fct = nn.CrossEntropyLoss()

        if self.span_only:
            span_masked_lm_loss = None
            span_prediction_scores = self.cls(sequence_output)
            span_masked_lm_loss = loss_fct(
                span_prediction_scores.view(-1, 2), span_labels.view(-1)
            )
            if not return_dict:
                output = (span_prediction_scores,) + outputs[2:]
                return (
                    ((span_masked_lm_loss,) + output)
                    if masked_lm_loss is not None
                    else output
                )

            return BertForSpanPreTrainingOutput(
                loss=span_masked_lm_loss,
                span_prediction_logits=span_prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        else:
            masked_lm_loss, span_masked_lm_loss = None, None
            span_prediction_scores, prediction_scores = self.cls(sequence_output)
            span_masked_lm_loss = loss_fct(
                span_prediction_scores.view(-1, 2), span_labels.view(-1)
            )
            if labels is not None:
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
                )

            final_masked_lm_loss = 0.5 * span_masked_lm_loss + 0.5 * masked_lm_loss
            if not return_dict:
                output = (span_prediction_scores, prediction_scores) + outputs[2:]
                return (
                    ((final_masked_lm_loss,) + output)
                    if masked_lm_loss is not None
                    else output
                )

            return BertForSpanPreTrainingOutput(
                loss=final_masked_lm_loss,
                prediction_logits=prediction_scores,
                span_prediction_logits=span_prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
