from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class MultiLabelNextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `next_sentence_label` is provided):
            Next sequence prediction (classification) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, len(labels))`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MultiLabelNSPHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class MultiLabelNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config, plm, r_drop=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.r_drop = r_drop

        self.plm = plm
        self.cls = MultiLabelNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def bert_cross_entropy(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        labels,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):

        outputs = self.plm(
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

        pooled_output = outputs[
            1
        ]  # Last layer hidden-state of the first token of the sequence (pooled_output)
        logits = self.cls(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits, outputs

    def bert_kl(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        labels,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):

        logits_list = []
        outputs_list = []
        for _ in range(2):
            outputs = self.plm(
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

            pooled_output = outputs[1]
            logits = self.cls(pooled_output)

            logits_list.append(logits)
            outputs_list.append(outputs)

        loss = None
        alpha = 1.0
        for logits in logits_list:
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    if loss:
                        loss += alpha * loss_fct(logits.view(-1), labels.view(-1))
                    else:
                        loss = alpha * loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    if loss:
                        loss += alpha * loss_fct(
                            logits.view(-1, self.num_labels), labels.view(-1)
                        )
                    else:
                        loss = alpha * loss_fct(
                            logits.view(-1, self.num_labels), labels.view(-1)
                        )

        if loss is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss += 1.0 * loss_fct(
                    logits_list[0].view(-1), logits_list[-1].view(-1)
                )
            else:
                p = torch.log_softmax(logits_list[0].view(-1, self.num_labels), dim=-1)
                p_tec = torch.softmax(logits_list[0].view(-1, self.num_labels), dim=-1)
                q = torch.log_softmax(logits_list[-1].view(-1, self.num_labels), dim=-1)
                q_tec = torch.softmax(logits_list[-1].view(-1, self.num_labels), dim=-1)

                kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction="none").sum()
                reverse_kl_loss = torch.nn.functional.kl_div(
                    q, p_tec, reduction="none"
                ).sum()
                loss += 0.5 * (kl_loss + reverse_kl_loss) / 2.0

        return loss, logits_list[0], outputs_list[0]

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
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MultiLabelNextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:
            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        Returns:
        Example:
        ```python
        >>> from transformers import BertTokenizer, BertForNextSentencePrediction
        >>> import torch
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if not self.r_drop:
            loss, logits, outputs = self.bert_cross_entropy(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
        else:
            loss, logits, outputs = self.bert_kl(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiLabelNextSentencePredictorOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
