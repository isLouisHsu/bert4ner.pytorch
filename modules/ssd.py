import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchtyping import TensorType


def iou(a: TensorType["num_a", "2(left, right)"], 
        b: TensorType["num_b", "2(left, right)"]):
    a = a.unsqueeze(1); b = b.unsqueeze(0)
    la, ra = a[:, :, 0], a[:, :, 1]
    lb, rb = b[:, :, 0], b[:, :, 1]

    i: TensorType["num_a", "num_b"] = \
        torch.clamp(torch.min(ra, rb) - torch.max(la, lb), min=0)
    u: TensorType["num_a", "num_b"] = \
        (ra - la) + (rb - lb) - i
    return i / u


class SSD(nn.Module):

    def __init__(self, hidden_size:int, num_labels:int, max_seq_length: int, 
            anchor_size: List[int], iou_thresh_pos: float, iou_thresh_neg: float):
        super().__init__()

        self.iou_thresh_pos = iou_thresh_pos
        self.iou_thresh_neg = iou_thresh_neg
        anchors = self._get_anchors(max_seq_length, anchor_size)
        self.register_buffer("anchors", anchors)

        self.num_anchors = len(anchor_size)
        self.cls_head = nn.Linear(hidden_size, self.num_anchors * num_labels)
        self.reg_head = nn.Linear(hidden_size, self.num_anchors * 2)

    def _get_anchors(self, max_seq_length: int, anchor_size: List[int]):
        position = torch.arange(max_seq_length)
        anchors = []
        for size in anchor_size:
            anchors.append(torch.stack([position, 
                torch.ones_like(position) * size], dim=-1))
        anchors: TensorType["max_seq_length", "num_anchors", "2(center, size)"] \
            = torch.stack(anchors, dim=1)
        return anchors
    
    def forward(self,
        hidden_states:  TensorType["batch_size", "sequence_length", "hidden_size"]=None, 
    ):
        batch_size, sequence_length, _ = hidden_states.size()
        cls_output: TensorType["batch_size", "sequence_length", "num_anchors", "num_labels"
            ] = self.cls_head(hidden_states).view(batch_size, sequence_length, self.num_anchors, -1)
        reg_output: TensorType["batch_size", "sequence_length", "num_anchors", "2(center, size)"
            ] = self.reg_head(hidden_states).view(batch_size, sequence_length, self.num_anchors, -1)
        return cls_output, reg_output


class SSDLoss(nn.Module):

    IGNORE_INDEX = -100

    def __init__(self, tag_o=1):
        super().__init__()
        self.tag_o = tag_o
    
    def forward(self,
        input_len:  TensorType["batch_size"]=None,
        cls_output: TensorType["batch_size", "sequence_length", "num_anchors", "num_labels"]=None,
        reg_output: TensorType["batch_size", "sequence_length", "num_anchors", "2(center, size)"]=None,
        label:      TensorType["num_entities", "3(type, left, right)"]=None,
        index:      TensorType["num_entities",]=None,
        anchors:    TensorType["max_seq_length", "num_anchors", "2(center, size)"]=None,
        iou_thresh_pos: float=None,
        iou_thresh_neg: float=None,
    ):
        batch_size, sequence_length, num_anchors, num_labels = cls_output.size()

        # calculate iou
        anchor_lr = torch.stack([
            anchors[..., 0] - anchors[..., 1] / 2.,
            anchors[..., 0] + anchors[..., 1] / 2.,
        ], dim=-1)

        # generate labels
        cls_label = torch.zeros(batch_size, sequence_length, num_anchors, 
            device=label.device, dtype=torch.long).fill_(self.IGNORE_INDEX)
        reg_label = torch.zeros_like(reg_output, 
            device=label.device).fill_(self.IGNORE_INDEX)
        for b in range(batch_size):
            # get batch
            len_   = input_len[b]
            mask_ = index == b
            label_ = label[mask_]
            # calculate iou
            anchor_ = anchor_lr[:len_].view(-1, 2)
            ious_ = iou(label_[:, 1:] + 1, anchor_) \
                .view(-1, len_, num_anchors)                    # `+1` for [CLS]
            ious_max_, iou_argmax_ = ious_.max(dim=0)           # shape(len_, num_anchors)
            # get ground truth
            cls_label[b, :len_] = torch.where(ious_max_ < iou_thresh_neg, 
                self.tag_o, cls_label[b, :len_])
            cls_label[b, :len_] = torch.where(ious_max_ > iou_thresh_pos, 
                label_[iou_argmax_, 0], cls_label[b, :len_])    # `iou_argmax_` is the index for label_
            # reg_label[b, :len_] = # TODO:
            print()

        return
    