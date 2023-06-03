import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class SpanPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    span_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMSpanHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.span_decoder = nn.Linear(config.hidden_size, out_features=2)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.span_decoder(hidden_states)
        return hidden_states


class OnlySpanMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.span_predictions = LMSpanHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        span_prediction_scores = self.span_predictions(sequence_output)
        return span_prediction_scores


class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.transform = PredictionHeadTransform(config)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.transform(features)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class MLMHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.span_predictions = LMSpanHead(config)
        self.predictions = LMHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        span_prediction_scores = self.span_predictions(sequence_output)
        prediction_scores = self.predictions(sequence_output)
        return span_prediction_scores, prediction_scores


class SpanMaskedLM(BertPreTrainedModel):
    def __init__(self, config, lm_model, span_only=True):
        super().__init__(config)

        # self.encoder = BertModel(config, add_pooling_layer=False)
        self.lm_model = lm_model
        self.span_only = span_only
        if span_only:
            self.cls = OnlySpanMLMHead(config)
        else:
            self.cls = MLMHeads(config)

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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanPreTrainingOutput]:
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

        outputs = self.lm_model(
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
            past_key_values=past_key_values,
            use_cache=use_cache,
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

            return SpanPreTrainingOutput(
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

            return SpanPreTrainingOutput(
                loss=final_masked_lm_loss,
                prediction_logits=prediction_scores,
                span_prediction_logits=span_prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
