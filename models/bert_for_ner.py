import torch
from torch import nn
from torchtyping import TensorType
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from modules.crf import CRF
from modules.span import Span, SpanLoss
from modules.ssd import SSD, SSDLoss, decode
from modules.biaffine import BiAffineParser

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
            loss_fct = SpanLoss()
            total_loss, start_loss, end_loss = loss_fct(
                attention_mask, start_logits, end_logits, start_positions, end_positions)

        if not return_dict:
            output = (start_logits, end_logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TokenClassifierOutput(
            loss=total_loss,
            logits=(start_logits, end_logits,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertSsdForNer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ssd = SSD(config.hidden_size, config.num_labels,
            config.max_position_embeddings, config.anchor_size,
            conf_thresh=config.conf_thresh, 
            nms_thresh=config.nms_thresh, 
            tag_o=config.tag_o
        )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        label:          TensorType["num_entities", "3(type, begin, end)"]=None,
        index:          TensorType["num_entities", "3(type, begin, end)"]=None,
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
        conf_logits, cls_logits, reg_logits = self.ssd(sequence_output)

        total_loss = conf_loss = cls_loss = reg_loss = None
        if label is not None and index is not None:
            loss_fct = SSDLoss(
                weight_conf=self.config.weight_conf, 
                weight_cls=self.config.weight_cls, 
                weight_reg=self.config.weight_reg, 
                neg_sample_rate=self.config.neg_sample_rate,
            )
            total_loss, conf_loss, cls_loss, reg_loss = loss_fct(
                input_len, conf_logits, cls_logits, reg_logits, 
                label, index, self.ssd.anchors, 
                self.config.iou_thresh_pos, 
                self.config.iou_thresh_neg,
            )

        if not return_dict:
            output = (conf_logits, cls_logits, reg_logits,) + outputs[2:]
            return ((total_loss, conf_loss, cls_loss, reg_loss,) + output) if total_loss is not None else output

        return TokenClassifierOutput(
            loss=(total_loss, conf_loss, cls_loss, reg_loss,),
            logits=(conf_logits, cls_logits, reg_logits,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertBiAffineForNer(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.biaffine = BiAffineParser(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels:     TensorType["batch_size", "sequence_length", "sequence_length"]=None,
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
        logits = self.biaffine(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            mask = torch.ones_like(labels).triu().bool().view(-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss[mask].mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    