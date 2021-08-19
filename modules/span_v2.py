from modules.ssd import decode
from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

# from allennlp.nn.util import batched_index_select
def batched_index_select(target: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = target.size()
    _, num_spans = indices.size()
    indexed = []
    for b in range(batch_size):
        target_b = target[b]
        indices_b = indices[b]
        indexed_b = []
        for i in range(num_spans):
            indexed_b.append(target_b[indices_b[i]])
        indexed_b = torch.stack(indexed_b, dim=0)
        indexed.append(indexed_b)
    indexed = torch.stack(indexed, dim=0)
    return indexed

class SpanV2(nn.Module):
    
    def __init__(self, hidden_size, num_labels, max_span_length, width_embedding_dim):
        super(SpanV2, self).__init__()

        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + width_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, hidden_states, spans):
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(hidden_states, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(hidden_states, spans_end)
        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)
        spans_embedding: TensorType["batch_size", "num_spans", "num_features"] = torch.cat([
            spans_start_embedding, 
            spans_end_embedding, 
            spans_width_embedding
        ], dim=-1)

        logits = self.classifier(spans_embedding)
        return logits
        
    def decode_batch(self,
        batch:      TensorType["batch_size", "num_spans", "num_labels"], 
        spans:      TensorType["batch_size", "num_spans", "3"],
        span_mask:  TensorType["batch_size", "num_spans",],
        is_logits:  bool=True,
    ):
        decodeds = []
        if is_logits:
            labels = batch.argmax(dim=-1)
        else:
            labels = batch
        for labels_, spans_, span_mask_ in zip(labels, spans, span_mask):
            span_mask_ = span_mask_ == 1.
            labels_ = labels_[span_mask_].cpu().numpy().tolist()
            spans_ = spans_[span_mask_].cpu().numpy().tolist()

            decoded_ = []
            for t, s in zip(labels_, spans_):
                decoded_.append([t, s[0] - 1, s[1] - 1])
            decodeds.append(decoded_)
        
        return decodeds

class SpanV2Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, 
        logits: TensorType["batch_size", "num_spans", "num_labels"], 
        label:  TensorType["batch_size", "num_spans"], 
        mask:   TensorType["batch_size", "num_spans"]
    ):
        num_labels = logits.size(-1)
        loss_mask = mask.view(-1) == 1
        loss = self.loss_fct(logits.view(-1, num_labels), label.view(-1))
        loss = loss[loss_mask].mean()
        return loss
    

