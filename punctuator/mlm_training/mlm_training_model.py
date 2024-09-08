from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from fastkan import FastKAN as KAN
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import *



class RobertaMLMForTokenClassification(
    RobertaForTokenClassification
):

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
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

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

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(restored_logits.device)
            loss = loss_fct(
                restored_logits.view(-1, self.num_labels),
                labels.view(-1),
                class_weights,
            )
        

        if not return_dict:
            output = (restored_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=restored_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
