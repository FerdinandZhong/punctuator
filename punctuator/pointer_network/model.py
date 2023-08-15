import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BartModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-45
) -> torch.Tensor:
    """
    Apply softmax to x with masking. Masking is for recognizing padding.

    Adapted from allennlp by allenai:
      https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    Args:
      x - Tensor of arbitrary shape to apply softmax over.
      mask - Binary mask of same shape as x where "False" indicates elements
        to disregard from operation.
      dim - Dimension over which to apply operation.
      eps - Stability constant for log operation. Added to mask to avoid NaN
        values in log.
    Outputs:
      Tensor with same dimensions as x.
    """

    x = x + (mask.float() + eps).log()
    return torch.nn.functional.softmax(x, dim=dim)


# TODO: refering to question answering way of prediction next boundary
class PointerNetwork(nn.Module):
    """
    From "Pointer Networks" by Vinyals et al. (2017)

    Adapted from pointer-networks-pytorch by ast0414:
      https://github.com/ast0414/pointer-networks-pytorch

    Args:
      n_hidden: The number of features to expect in the inputs.
    """

    def __init__(self, n_hidden: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.w1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.w2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.v = nn.Linear(n_hidden, 1, bias=False)

    def forward(
        self,
        x_decoder: torch.Tensor,
        x_encoder: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-16,
    ) -> torch.Tensor:
        """
        Args:
          x_decoder: Encoding over the output tokens.
          x_encoder: Encoding over the input tokens.
          mask: Binary mask over the softmax input.
        Shape:
          x_decoder: (B, Ne, C)
          x_encoder: (B, Nd, C)
          mask: (B, Nd, Ne)
        """

        # (B, Nd, Ne, C) <- (B, Ne, C)
        encoder_transform = (
            self.w1(x_encoder).unsqueeze(1).expand(-1, x_decoder.shape[1], -1, -1)
        )
        # (B, Nd, 1, C) <- (B, Nd, C)
        decoder_transform = self.w2(x_decoder).unsqueeze(2)
        # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
        prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        # (B, Nd, Ne) <- (B, Nd, Ne)
        # mask = mask.unsqueeze(1) # for batched
        log_score = masked_softmax(prod, mask, dim=-1, eps=eps)
        return log_score


class PointerPunctuator(PreTrainedModel):
    def __init__(self, config, bart_model: BartModel = None):
        super().__init__(config)
        config = copy.deepcopy(config)
        config.is_decoder = False
        config.is_encoder_decoder = True

        if bart_model is None:
            self.bart_model = BartModel(config)
        else:
            self.bart_model = bart_model
        self.encoder = self.bart_model.get_encoder()
        self.decoder = self.bart_model.get_decoder()
        self.ptr_model = PointerNetwork(n_hidden=config.d_model)

        self.punctuator_head = nn.Linear(config.hidden_size, config.num_labels)
        self.boundary_head = nn.Linear(
            config.hidden_size, 1
        )  # predict nearest boundary once

        self.post_init()

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        decoder_input_index: int,
        pointer_mask: torch.LongTensor,
        input_ids: torch.LongTensor = None,
        encoder_outputs_last_hidden_state: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """_summary_

        Args:
            decoder_input_ids (torch.LongTensor): Generated decoder input ids for each step
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            decoder_attention_mask (Optional[torch.LongTensor], optional): _description_. Defaults to None.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs_last_hidden_state is None:
            encoder_outputs_last_hidden_state = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        boundary_score = self.ptr_model(
            x_decoder=decoder_outputs.last_hidden_state,
            x_encoder=encoder_outputs_last_hidden_state,
            mask=pointer_mask,
        )  # B, Nd, Ne

        return boundary_score[:, decoder_input_index, :]

    def detect_boundary_2(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )
        # outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=encoder_outputs_last_hidden_state,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     past_key_values=past_key_values,
        #     inputs_embeds=decoder_inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        outputs = self.bart_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        logits = self.boundary_head(sequence_output)

        return logits.squeeze(-1).contiguous()

    def punctuator_generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Decoder only for generation

        Args:
            input_ids (torch.LongTensor, optional): _description_. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            encoder_hidden_states (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            encoder_attention_mask (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            head_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            cross_attn_head_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            past_key_values (Optional[List[torch.FloatTensor]], optional): _description_. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            labels (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            use_cache (Optional[bool], optional): _description_. Defaults to None.
            output_attentions (Optional[bool], optional): _description_. Defaults to None.
            output_hidden_states (Optional[bool], optional): _description_. Defaults to None.
            return_dict (Optional[bool], optional): _description_. Defaults to None.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.punctuator_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
