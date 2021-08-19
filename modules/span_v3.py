import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class SpanV3(nn.Module):
    """ Implementation for paper ``Named Entity Recognition as Dependency Parsing``

    """
    def __init__(self, hidden_size, num_labels, max_span_length, width_embedding_dim):
        super().__init__()

        self.max_span_length = max_span_length
        self.width_embedding_dim = width_embedding_dim
        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + width_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )
    
    def forward(self, 
        hidden_states: TensorType["batch_size", "sequence_length", "hidden_size"]
    ):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        
        # span length embedding
        range_vec = torch.arange(sequence_length).to(hidden_states.device)
        range_mat = range_vec.repeat(sequence_length).view(sequence_length, sequence_length)
        span_length_mat = range_mat - torch.t(range_mat) + 1
        span_valid_mask = ~((span_length_mat <= 0) | (span_length_mat > self.max_span_length))
        span_length_mat = torch.where(span_valid_mask, span_length_mat, torch.zeros_like(span_length_mat))
        span_length_embed = self.width_embedding(span_length_mat.view(1, -1)) \
            .view(1, sequence_length, sequence_length, -1)

        logits: TensorType["batch_size", "sequence_length", "sequence_length", "num_labels"] = \
            self.classifier(torch.cat([
            hidden_states.unsqueeze(2).expand(batch_size, sequence_length, sequence_length, hidden_size), 
            hidden_states.unsqueeze(1).expand(batch_size, sequence_length, sequence_length, hidden_size),
            span_length_embed.expand(batch_size, sequence_length, sequence_length, self.width_embedding_dim),
        ], dim=-1))
        mask = span_valid_mask.unsqueeze(0).expand(batch_size, sequence_length, sequence_length).contiguous()

        return logits, mask
    
    def decode_batch(self, 
        batch: TensorType["batch_size", "sequence_length", "sequence_length", "num_labels"],
        mask: TensorType["batch_size", "sequence_length", "sequence_length"] = None,
        is_logits: bool=True,
    ):
        decodeds = []
        for i in range(batch.size(0)):
            b = batch[i]
            if is_logits:
                prob, label = b.softmax(dim=-1).max(dim=-1)
            else:
                label = b
                prob = torch.ones_like(b, dtype=torch.float)
            
            start_, end_ = torch.where(~torch.isnan(label))
            start_ = start_ - 1; end_ = end_ - 1
            decoded = torch.stack([label.view(-1), prob.view(-1), 
                start_, end_], dim=-1)
            if mask is None:
                valid_mask = (start_ >= 0) & (end_ >= 0) & (start_ <= end_)
            else:
                valid_mask = mask[i].view(-1)
            decoded = decoded[valid_mask].cpu().numpy().tolist()
            decoded = sorted(decoded, key=lambda x: (x[1], x[-2] - x[-1]))
            
            decoded_ = []
            for t, p, s, e in decoded:
                is_covered = False
                for t_, s_, e_ in decoded_:
                    if (t == t_) and (min(e, e_) - max(s, s_) >= 0):
                        is_covered = True
                        break
                if is_covered: continue
                decoded_.append([int(t), int(s), int(e)])
            decodeds.append(decoded_)
        
        return decodeds


if __name__ == "__main__":
    m = SpanV3(768, 10, 10, 128)
    x = torch.rand(8, 128, 768)
    m(x)