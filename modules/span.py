import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(PoolerStartLogits, self).__init__()
        self.dense = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states)
        return x

class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(PoolerEndLogits, self).__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, num_labels)

    def forward(self, 
        hidden_states: TensorType["batch_size", "sequence_length", "hidden_size"], 
        start_positions=None, 
        p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x)
        return x

class Span(nn.Module):

    def __init__(self, hidden_size, num_labels, soft_label=True):
        super().__init__()
        self.num_labels = num_labels
        self.soft_label = soft_label
        self.start_fc = PoolerStartLogits(hidden_size, num_labels)
        self.end_fc = PoolerEndLogits(
            hidden_size + num_labels if self.soft_label else 1, num_labels)
    
    def forward(self, 
        hidden_states:      TensorType["batch_size", "sequence_length", "hidden_size"]=None, 
        start_positions:    TensorType["batch_size", "sequence_length"]=None, 
    ):
        start_logits: TensorType["batch_size", "sequence_length", "num_labels"] \
            = self.start_fc(hidden_states)
        if start_positions is not None and self.training:
            batch_size, seq_len, _ = hidden_states.size()
            if self.soft_label:
                label_logits = torch.FloatTensor(
                    batch_size, seq_len, self.num_labels).to(hidden_states.device)
                label_logits.zero_()
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits: TensorType["batch_size", "sequence_length", "num_labels"] \
            = self.end_fc(hidden_states, label_logits)

        return start_logits, end_logits
    
    def decode_logits(self, 
        start_logits:   TensorType["sequence_length", "num_labels"], 
        end_logits:     TensorType["sequence_length", "num_labels"],
    ):
        start_positions = torch.argmax(start_logits, -1)
        end_positions   = torch.argmax(end_logits,   -1)
        decoded = self.decode_positions(start_positions, end_positions)
        return decoded

    def decode_positions(self, 
        start_positions:   TensorType["sequence_length"], 
        end_positions:     TensorType["sequence_length"],
    ):
        start_positions = start_positions.cpu().numpy().tolist()
        end_positions   = end_positions.cpu().numpy().tolist()
        decoded = []
        for i, s_l in enumerate(start_positions):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_positions[i:]):
                if s_l == e_l:
                    decoded.append((s_l, i, i + j))
                    break
        return decoded

    def decode_logits_batch(self, 
        start_logits:   TensorType["batch_size", "sequence_length", "num_labels"], 
        end_logits:     TensorType["batch_size", "sequence_length", "num_labels"],
    ):
        return [self.decode_logits(s, e) for s, e in zip(start_logits, end_logits)]

    def decode_positions_batch(self, 
        start_positions:   TensorType["batch_size", "sequence_length"], 
        end_positions:     TensorType["batch_size", "sequence_length"],
    ):
        return [self.decode_positions(s, e) for s, e in zip(start_positions, end_positions)]


class SpanLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, 
        start_logits:   TensorType["batch_size", "sequence_length", "num_labels"], 
        end_logits:     TensorType["batch_size", "sequence_length", "num_labels"],
        start_positions:   TensorType["batch_size", "sequence_length"], 
        end_positions:     TensorType["batch_size", "sequence_length"],
    ):
        num_labels = start_logits.size(-1)
        loss_mask: TensorType["batch_size * sequence_length"] \
            = attention_mask.view(-1) == 1
        active_start_logits = start_logits.view(-1, num_labels)[loss_mask]
        active_end_logits   = end_logits.view(-1, num_labels)[loss_mask]
        active_start_labels = start_positions.view(-1)[loss_mask]
        active_end_labels   = end_positions.view(-1)[loss_mask]

        start_loss = self.loss_fct(active_start_logits, active_start_labels)
        end_loss = self.loss_fct(active_end_logits, active_end_labels)
        total_loss = (start_loss + end_loss) / 2

        return total_loss, start_loss, end_loss
