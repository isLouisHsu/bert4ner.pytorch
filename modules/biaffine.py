import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class BiAffineParser(nn.Module):
    """ Implementation for paper ``Named Entity Recognition as Dependency Parsing``

    """
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.num_labels = num_labels

        # self.start_fc = nn.Linear(hidden_size, hidden_size)
        # self.end_fc = nn.Linear(hidden_size, hidden_size)
        self.start_fc = nn.Identity()
        self.end_fc = nn.Identity()

        # self.Um = nn.Parameter(torch.Tensor(hidden_size, num_labels, hidden_size))
        # bound = 1 / math.sqrt(hidden_size)
        # nn.init.uniform_(self.Um, -bound, bound)
        # # nn.init.zeros_(self.Um)

        # self.ffn = nn.Linear(2 * hidden_size, num_labels)
        self.ffn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels),
        )
    
    def forward(self, 
        hidden_states: TensorType["batch_size", "sequence_length", "hidden_size"]
    ):
        batch_size, sequence_length, hidden_size = hidden_states.size()

        x_start = self.start_fc(hidden_states)
        x_end = self.end_fc(hidden_states)
        
        # item1: TensorType["batch_size", "sequence_length", "sequence_length", "num_labels"] = \
        #     torch.matmul(x_start.view(-1, hidden_size), self.Um.view(hidden_size, -1)) \
        #     .view(batch_size, -1, hidden_size) \
        #     .bmm(x_end.permute(0, 2, 1)) \
        #     .view(batch_size, sequence_length, self.num_labels, sequence_length) \
        #     .permute(0, 1, 3, 2) \
        #     .contiguous()
        item2: TensorType["batch_size", "sequence_length", "sequence_length", "num_labels"] = \
            self.ffn(torch.cat([
            x_start.unsqueeze(2).expand(batch_size, sequence_length, sequence_length, hidden_size), 
            x_end.unsqueeze(1).expand(batch_size, sequence_length, sequence_length, hidden_size),
        ], dim=-1))

        # logits = item1 + item2
        logits = item2
        return logits
    
    def decode_batch(self, 
        batch: TensorType["batch_size", "sequence_length", "sequence_length", "num_labels"],
        is_logits: bool=True,
    ):
        decodeds = []
        for b in batch:
            label = b.max(dim=-1)[1] if is_logits else b
            start_, end_ = torch.where(~torch.isnan(label))
            start_ = start_ - 1; end_ = end_ - 1
            decoded = torch.stack([label.view(-1), start_, end_], dim=-1)
            valid_mask = (start_ >= 0) & (end_ >= 0) & (start_ <= end_)
            decoded = decoded[valid_mask]
            decodeds.append(decoded.cpu().numpy().tolist())
        return decodeds


if __name__ == "__main__":

    m = BiAffineParser(768, 16)
    x = torch.rand(4, 256, 768)
    y = m(x)