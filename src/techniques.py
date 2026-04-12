import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPooler(nn.Module):
    """Takes the absolute highest similarity score (The strict approach)."""
    def forward(self, sim):
        return sim.max(dim=1).values

class MeanPooler(nn.Module):
    """Takes the average of all topic words (The broad approach)."""
    def forward(self, sim):
        valid_mask = (sim != float("-inf")).float()
        safe_sim = torch.where(sim == float("-inf"), torch.zeros_like(sim), sim)
        return safe_sim.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-8)

class TopKPooler(nn.Module):
    """Averages the top K strongest matches (The sweet spot)."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, sim):
        k_actual = min(self.k, sim.size(1))
        top_k_values = torch.topk(sim, k=k_actual, dim=1).values
        return top_k_values.mean(dim=1)
    
class Conv1DPooler(nn.Module):
    """Slides a window across the topic words to find matching phrases."""
    def __init__(self, window_size=2):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=window_size)

    def forward(self, sim):
        batch_size, topic_len, text_len = sim.shape
        
        if topic_len < self.window_size:
            return sim.max(dim=1).values
            
        safe_sim = torch.where(sim == float("-inf"), torch.tensor(-1.0, device=sim.device), sim)

        reshaped_sim = safe_sim.transpose(1, 2).reshape(batch_size * text_len, 1, topic_len)
        
        phrase_scores = self.conv(reshaped_sim)
        
        phrase_scores = phrase_scores.reshape(batch_size, text_len, -1).transpose(1, 2)
        
        return phrase_scores.max(dim=1).values

TECHNIQUE_FACTORIES = {
    "max": lambda: MaxPooler(),
    "mean": lambda: MeanPooler(),
    "top3": lambda: TopKPooler(k=3),
    "top5": lambda: TopKPooler(k=5),
    "conv2": lambda: Conv1DPooler(window_size=2),
    "conv3": lambda: Conv1DPooler(window_size=3)
}