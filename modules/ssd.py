import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torchtyping import TensorType

# TODO: 
# 实体位置坐标确认
# 读取数据那边，实体坐标要改
# x x x x x x x x
#  | | |
# 实体长度和anchor尺寸间iou的关系

fcenter = lambda left, right: (left + right) / 2.
fsize = lambda left, right: right - left
fleft = lambda center, size: center - size / 2.
fright = lambda center, size: center + size / 2. 

def iou(a: TensorType["num_a", "2(left, right)"], 
        b: TensorType["num_b", "2(left, right)"]):
    a = a.unsqueeze(1); b = b.unsqueeze(0)
    la, ra = a[:, :, 0], a[:, :, 1]
    lb, rb = b[:, :, 0], b[:, :, 1]

    i: TensorType["num_a", "num_b"] = torch.clamp(
        fsize(torch.max(la, lb), torch.min(ra, rb)), min=0)
    u: TensorType["num_a", "num_b"] = fsize(la, ra) + fsize(lb, rb) - i
    return i / u

def nms(dets: TensorType["num_dets", "3(left, right, confidence)"], thresh: float):
    if dets.size(0) == 0:
        return torch.tensor([])
    scores = dets[:, -1]
    regions = dets[:, :-1]
    order = scores.argsort(descending=True)

    keep = []
    while order.size(0) > 0:
        index = order[0]
        keep.append(index)
        iou_val = iou(
            regions[index].unsqueeze(0), 
            regions[order[1:]],
        )[0]
        mask = iou_val <= thresh
        order = order[1:][mask]
    keep = torch.stack(keep)
    return keep

def encode(batch_size, sequence_length,
    input_len: TensorType["batch_size"],
    label: TensorType["num_entities", "3(type, left, right)"],
    index: TensorType["num_entities",],
    anchors: TensorType["max_seq_length", "num_anchors", "2(center, size)"],
    iou_thresh_pos: float,
    iou_thresh_neg: float,
    ignore_index: int,
    tag_o: int,
):
    num_anchors = anchors.size(1)
    anchors_lr = torch.stack([fleft(anchors[..., 0], anchors[..., 1]), 
        fright(anchors[..., 0], anchors[..., 1])], dim=-1)
    conf_label = torch.zeros(batch_size, sequence_length, num_anchors, 
        device=label.device, dtype=torch.long).fill_(0)
    cls_label = torch.zeros(batch_size, sequence_length, num_anchors, 
        device=label.device, dtype=torch.long).fill_(ignore_index)
    reg_label = torch.zeros(batch_size, sequence_length, num_anchors, 2, 
        device=label.device, dtype=torch.float)
    for b in range(batch_size):
        # get batch
        mask_ = index == b
        if mask_.sum() == 0: continue
        len_   = input_len[b]
        label_ = label[mask_]
        anchors_lr_ = anchors_lr[:len_].view(-1, 2)
        ious_ = iou(label_[:, 1:], anchors_lr_) \
            .view(-1, len_, num_anchors)                                # shape(num_label_, len_, num_anchors)
        # match labels and anchors
        label_ious_max_, label_index_ = ious_.max(dim=0)                # shape(len_, num_anchors)
                                                                        # `label_index_` is the matched label index for each anchor
        # get ground truth for confidence & classification
        neg_mask_ = label_ious_max_ <= iou_thresh_neg
        pos_mask_ = label_ious_max_ >= iou_thresh_pos
        conf_label[b, :len_] = torch.where(pos_mask_, 1, conf_label[b, :len_])
        cls_label[b, :len_] = torch.where(neg_mask_, tag_o, cls_label[b, :len_])
        cls_label[b, :len_] = torch.where(pos_mask_, label_[label_index_, 0].long(), cls_label[b, :len_])
        # get ground truth for regression
        anchors_cs_ = anchors[:len_].view(-1, 2)                        # shape(len_ * num_anchors, 2)
        ac_ = anchors_cs_[:, 0].unsqueeze(0)
        as_ = anchors_cs_[:, 1].unsqueeze(0)                            # shape(1, len_ * num_anchors)
        lc_ = fcenter(label_[:, 2], label_[:, 1]).unsqueeze(1)
        ls_ = fsize(label_[:, 1], label_[:, 2]).unsqueeze(1)            # shape(num_label_, 1)
        offset_c_ = (lc_ - ac_) / as_
        # offset_s_ = torch.log(ls_ / as_)                                # shape(num_label_, len_ * num_anchors)
        offset_s_ = ls_ / as_                                           # shape(num_label_, len_ * num_anchors)
        offset_c_ = offset_c_.gather(0, label_index_.view(-1).unsqueeze(0))
        offset_s_ = offset_s_.gather(0, label_index_.view(-1).unsqueeze(0))
        reg_label_ = torch.stack([offset_c_, offset_s_], dim=-1
            ).view(len_, num_anchors, 2)                                # shape(len_ * num_anchors, 2)
        reg_label[b, :len_] = reg_label_
    return conf_label.float(), cls_label, reg_label

def decode(
    input_len:  TensorType["batch_size"],
    conf_logits:TensorType["batch_size", "sequence_length", "num_anchors"],
    cls_logits: TensorType["batch_size", "sequence_length", "num_anchors", "num_labels"],
    reg_logits: TensorType["batch_size", "sequence_length", "num_anchors", "2(offset_center, offset_size)"],
    anchors:    TensorType["max_seq_length", "num_anchors", "2(center, size)"],
    conf_thresh: float,
    nms_thresh: float,
    tag_o: int,
):
    batch_size, sequence_length, num_anchors, num_labels = cls_logits.size()
    anchors_cz = anchors[:sequence_length]
    # decode output
    conf_probas = conf_logits.sigmoid()
    cls_logits = cls_logits.softmax(dim=-1)
    cls_probas, cls_labels = cls_logits.max(dim=-1)
    anchor_c = anchors_cz[..., 0].unsqueeze(0)
    anchor_s = anchors_cz[..., 1].unsqueeze(0)
    reg_logits[..., 0] = reg_logits[..., 0] * anchor_s + anchor_c
    # reg_logits[..., 1] = torch.exp(reg_logits[..., 1] * anchor_s)
    reg_logits[..., 1] = reg_logits[..., 1] * anchor_s
    reg_logits[..., 0], reg_logits[...,1] = \
        fleft (reg_logits[..., 0], reg_logits[..., 1]), \
        fright(reg_logits[..., 0], reg_logits[..., 1]),
    # decode predictions
    predicts = []; index = []
    for b in range(batch_size):
        len_ = input_len[b]
        # filter predictions according to probability
        conf_ = conf_probas[b, :len_].view(-1)
        mask_ = conf_ >= conf_thresh
        if mask_.sum() == 0: continue
        conf_ = conf_[mask_].unsqueeze(-1)
        labels_ = cls_labels[b, :len_].view(-1)[mask_].unsqueeze(-1).float()
        reg_ = reg_logits[b, :len_].view(-1, 2)[mask_]
        # do nms
        keep_ = nms(torch.cat([reg_, conf_], dim=-1), nms_thresh)
        predicts_ = torch.cat([reg_, conf_, labels_], dim=-1)[keep_]
        # save predictions
        index_ = [b for i in range(len(keep_))]
        predicts.append(predicts_); index.extend(index_)
    if len(predicts) == 0:
        dets = torch.zeros(0, 3)
        tags = torch.zeros(0,)
        index = torch.zeros(0,)
        return dets, tags, index
    # parse predicts
    predicts: TensorType["num_entities", "4(left, right, confident, label)"] = torch.cat(predicts, dim=0)
    index: TensorType["num_entities"] = torch.tensor(
        index, dtype=torch.long, device=predicts.device)
    dets: TensorType["num_entities", "3(left, right, confident)"] = predicts[:, :-1].float()
    tags: TensorType["num_entities",] = predicts[:, -1].long()
    return dets, tags, index


class SSD(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int, max_seq_length: int, anchor_size: List[int], 
        conf_thresh: float=0.7, nms_thresh: float=0.6, tag_o: int=1):
        super().__init__()

        self.register_buffer("anchors", self._get_anchors(max_seq_length, anchor_size))

        self.num_anchors = len(anchor_size)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.tag_o = tag_o
        self.conf_head = nn.Linear(hidden_size, self.num_anchors)
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
        hidden_states:  TensorType["batch_size", "sequence_length", "hidden_size"], 
    ):
        batch_size, sequence_length, _ = hidden_states.size()
        conf_logits: TensorType["batch_size", "sequence_length", "num_anchors"
            ] = self.conf_head(hidden_states).view(batch_size, sequence_length, self.num_anchors)
        cls_logits: TensorType["batch_size", "sequence_length", "num_anchors", "num_labels"
            ] = self.cls_head(hidden_states).view(batch_size, sequence_length, self.num_anchors, -1)
        reg_logits: TensorType["batch_size", "sequence_length", "num_anchors", "2(center, size)"
            ] = self.reg_head(hidden_states).view(batch_size, sequence_length, self.num_anchors, -1)
        return conf_logits, cls_logits, reg_logits

    def decode(self,
        input_len:  TensorType["batch_size"],
        conf_logits:TensorType["batch_size", "sequence_length", "num_anchors"],
        cls_logits: TensorType["batch_size", "sequence_length", "num_anchors", "num_labels"],
        reg_logits: TensorType["batch_size", "sequence_length", "num_anchors", "2(offset_center, offset_size)"],
    ):
        return decode(
            input_len=input_len,
            conf_logits=conf_logits,
            cls_logits=cls_logits,
            reg_logits=reg_logits,
            anchors=self.anchors,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            tag_o=self.tag_o,
        )


class SSDLoss(nn.Module):

    IGNORE_INDEX = -100

    def __init__(self, weight_conf: float=1.0, weight_cls: float=1.0, weight_reg: float=1.0, tag_o: int=1):
        super().__init__()
        self.tag_o = tag_o

        # weight_sum = weight_conf + weight_cls + weight_reg
        # self.weight_conf = weight_conf / weight_sum
        # self.weight_cls = weight_cls / weight_sum
        # self.weight_reg = weight_reg / weight_sum
        self.weight_conf = weight_conf
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
    
    def forward(self,
        input_len:  TensorType["batch_size"],
        conf_logits:TensorType["batch_size", "sequence_length", "num_anchors"],
        cls_logits: TensorType["batch_size", "sequence_length", "num_anchors", "num_labels"],
        reg_logits: TensorType["batch_size", "sequence_length", "num_anchors", "2(offset_center, offset_size)"],
        label:      TensorType["num_entities", "3(type, left, right)"],
        index:      TensorType["num_entities",],
        anchors:    TensorType["max_seq_length", "num_anchors", "2(center, size)"],
        iou_thresh_pos: float,
        iou_thresh_neg: float,
    ):
        batch_size, sequence_length, num_anchors, num_labels = cls_logits.size()

        # encode labels
        conf_label, cls_label, reg_label = encode(batch_size, sequence_length, 
            input_len, label, index, anchors, iou_thresh_pos, iou_thresh_neg,
            ignore_index=self.IGNORE_INDEX, tag_o=self.tag_o)

        # dets, tags, index = decode(input_len, conf_logits, cls_logits, reg_logits, anchors,
        #     conf_thresh=0.7, nms_thresh=0.7, tag_o=1)
        
        # focal loss
        weight = None

        # calculate loss
        conf_loss = nn.BCEWithLogitsLoss(weight=weight)(
            conf_logits.view(-1), conf_label.view(-1))
        cls_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=self.IGNORE_INDEX)(
            cls_logits.view(-1, num_labels), cls_label.view(-1))
        reg_loss = nn.MSELoss(reduction='none')(reg_logits, reg_label)
        reg_loss = reg_loss[conf_label == 1].mean()
        total_loss = conf_loss * self.weight_conf + \
            cls_loss * self.weight_cls + reg_loss * self.weight_reg

        return total_loss, conf_loss, cls_loss, reg_loss

def test():
    hidden_size, num_labels, max_seq_length, anchor_size, iou_thresh_pos, iou_thresh_neg = 768, 17, 512, [1, 3, 5, 7], 0.6, 0.3
    ssd = SSD(hidden_size, num_labels, max_seq_length, anchor_size)
    batch_size, sequence_length, num_anchors = 4, 38, 4
    loss_fct = SSDLoss()
    input_len = torch.tensor([24, 36, 12, 29])
    conf_logits = torch.rand(batch_size, sequence_length, num_anchors)
    cls_logits = torch.rand(batch_size, sequence_length, num_anchors, num_labels)
    reg_logits = torch.rand(batch_size, sequence_length, num_anchors, 2)
    label = torch.tensor([
        [0, 1, 4],
        [0, 3, 5],
        [0, 6, 12],
        [1, 1, 4],
        [1, 6, 9],
        [1, 18, 24],
        [2, 1, 6],
        [3, 3, 6],
        [3, 9, 19],
    ])
    index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3, 3])
    anchors = ssd.anchors
    loss_fct(input_len, conf_logits, cls_logits, reg_logits, label, index, anchors, iou_thresh_pos, iou_thresh_neg)
if __name__ == "__main__":
    test()
