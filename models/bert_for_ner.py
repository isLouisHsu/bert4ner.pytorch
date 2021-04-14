import torch
from torch import nn
from torchtyping import TensorType
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from modules.crf import CRF
from modules.span import Span

class BertCrfForNer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        label:      TensorType["batch_size", "sequence_length"]=None,
        input_len:  TensorType["batch_size"]=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        logits = self.linear(sequence_output)

        loss = None
        if label is not None:
            loss = -1 * self.crf(
                emissions=logits, tags=label, mask=attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertSpanForNer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span = Span(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions:TensorType["batch_size", "sequence_length"]=None,
        end_positions:  TensorType["batch_size", "sequence_length"]=None,
        input_len:      TensorType["batch_size"]=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        start_logits, end_logits = self.span(sequence_output, start_positions)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            num_labels = start_logits.size(-1)
            loss_mask: TensorType["batch_size * sequence_length"] \
                = attention_mask.view(-1) == 1
            active_start_logits = start_logits.view(-1, num_labels)[loss_mask]
            active_end_logits   = end_logits.view(-1, num_labels)[loss_mask]
            active_start_labels = start_positions.view(-1)[loss_mask]
            active_end_labels   = end_positions.view(-1)[loss_mask]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TokenClassifierOutput(
            loss=total_loss,
            logits=(start_logits, end_logits,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
