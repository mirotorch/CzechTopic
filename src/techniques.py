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


def build_text_spans(text_positions, max_span_length, device):
    start_positions = []
    end_positions = []
    span_lengths = []

    for start_offset in range(text_positions.numel()):
        max_end_offset = min(start_offset + max_span_length, text_positions.numel())
        for end_offset in range(start_offset, max_end_offset):
            start_positions.append(text_positions[start_offset].item())
            end_positions.append(text_positions[end_offset].item())
            span_lengths.append(end_offset - start_offset + 1)

    span_indices = torch.tensor(
        list(zip(start_positions, end_positions)),
        device=device,
        dtype=torch.long,
    )
    span_lengths = torch.tensor(
        span_lengths,
        device=device,
        dtype=torch.long,
    )
    return span_indices, span_lengths


def project_span_scores_to_tokens(span_scores, span_indices, seq_len, device):
    token_positions = torch.arange(seq_len, device=device).unsqueeze(0)
    span_starts = span_indices[:, 0].unsqueeze(1)
    span_ends = span_indices[:, 1].unsqueeze(1)
    span_cover_mask = (token_positions >= span_starts) & (token_positions <= span_ends)
    expanded_span_scores = span_scores.unsqueeze(1).expand(-1, seq_len)
    covered_scores = expanded_span_scores.masked_fill(~span_cover_mask, float("-inf"))
    return covered_scores.max(dim=0).values


def compute_span_mean_vectors(hidden_states, span_indices):
    span_means = []
    for start_pos, end_pos in span_indices.tolist():
        span_means.append(hidden_states[start_pos : end_pos + 1].mean(dim=0))
    return torch.stack(span_means, dim=0)

class SpanExtractionPooler(nn.Module):
    """Scores candidate text spans against a topic vector and projects them back to tokens."""

    def __init__(self, hidden_size=768, max_span_length=4, length_embedding_size=32):
        super().__init__()
        self.max_span_length = max_span_length
        self.length_embedding = nn.Embedding(max_span_length + 1, length_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + length_embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states, topic_vector, text_mask):
        """
        hidden_states: [batch_size, sequence_length, hidden_size]
        topic_vector: [batch_size, hidden_size]
        text_mask: [batch_size, sequence_length]
        """
        batch_size, seq_len, _ = hidden_states.size()
        token_scores = hidden_states.new_full((batch_size, seq_len), float("-inf"))
        span_scores_per_batch = []
        span_indices_per_batch = []

        for batch_idx in range(batch_size):
            text_positions = torch.nonzero(text_mask[batch_idx], as_tuple=False).flatten()
            if text_positions.numel() == 0:
                span_scores_per_batch.append(hidden_states.new_empty((0,)))
                span_indices_per_batch.append(hidden_states.new_empty((0, 2), dtype=torch.long))
                continue

            span_indices, span_lengths = build_text_spans(
                text_positions,
                self.max_span_length,
                hidden_states.device,
            )

            start_vectors = hidden_states[batch_idx, span_indices[:, 0], :]
            end_vectors = hidden_states[batch_idx, span_indices[:, 1], :]
            repeated_topic = topic_vector[batch_idx].unsqueeze(0).expand_as(start_vectors)
            length_vectors = self.length_embedding(span_lengths)

            span_features = torch.cat(
                [repeated_topic, start_vectors, end_vectors, length_vectors],
                dim=-1,
            )
            span_scores = self.mlp(span_features).squeeze(-1)
            token_scores[batch_idx] = project_span_scores_to_tokens(
                span_scores,
                span_indices,
                seq_len,
                hidden_states.device,
            )

            span_scores_per_batch.append(span_scores)
            span_indices_per_batch.append(span_indices)

        token_scores = token_scores.masked_fill(~text_mask, float("-inf"))
        return {
            "token_scores": token_scores,
            "span_scores": span_scores_per_batch,
            "span_indices": span_indices_per_batch,
        }


class SpanMaxPooler(nn.Module):
    """Encodes spans into vectors and compares them against topic token embeddings."""

    def __init__(self, hidden_size=768, max_span_length=4, length_embedding_size=32):
        super().__init__()
        self.max_span_length = max_span_length
        self.length_embedding = nn.Embedding(max_span_length + 1, length_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3 + length_embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden_states, topic_hidden, topic_mask, text_mask):
        batch_size, seq_len, _ = hidden_states.size()
        token_scores = hidden_states.new_full((batch_size, seq_len), float("-inf"))
        span_scores_per_batch = []
        span_indices_per_batch = []

        for batch_idx in range(batch_size):
            text_positions = torch.nonzero(text_mask[batch_idx], as_tuple=False).flatten()
            if text_positions.numel() == 0:
                span_scores_per_batch.append(hidden_states.new_empty((0,)))
                span_indices_per_batch.append(hidden_states.new_empty((0, 2), dtype=torch.long))
                continue

            span_indices, span_lengths = build_text_spans(
                text_positions,
                self.max_span_length,
                hidden_states.device,
            )

            sample_hidden = hidden_states[batch_idx]
            start_vectors = sample_hidden[span_indices[:, 0], :]
            end_vectors = sample_hidden[span_indices[:, 1], :]
            mean_vectors = compute_span_mean_vectors(sample_hidden, span_indices)
            length_vectors = self.length_embedding(span_lengths)

            span_features = torch.cat(
                [start_vectors, end_vectors, mean_vectors, length_vectors],
                dim=-1,
            )
            span_vectors = self.mlp(span_features)

            valid_topic_vectors = topic_hidden[batch_idx][topic_mask[batch_idx]]
            valid_topic_vectors = F.normalize(valid_topic_vectors, p=2, dim=-1)
            span_vectors = F.normalize(span_vectors, p=2, dim=-1)
            similarity_matrix = torch.matmul(valid_topic_vectors, span_vectors.transpose(0, 1))
            span_scores = similarity_matrix.max(dim=0).values

            token_scores[batch_idx] = project_span_scores_to_tokens(
                span_scores,
                span_indices,
                seq_len,
                hidden_states.device,
            )

            span_scores_per_batch.append(span_scores)
            span_indices_per_batch.append(span_indices)

        token_scores = token_scores.masked_fill(~text_mask, float("-inf"))
        return {
            "token_scores": token_scores,
            "span_scores": span_scores_per_batch,
            "span_indices": span_indices_per_batch,
        }

def apply_nms(spans, scores, threshold=0.85):
    """
    Filters overlapping predictions, keeping only the best ones.
    spans: List of tuples (start_idx, end_idx)
    scores: List of floats (cosine similarity scores)
    """
    valid_predictions = [(span, score) for span, score in zip(spans, scores) if score >= threshold]
    valid_predictions.sort(key=lambda x: x[1], reverse=True)

    approved_spans = []

    for current_span, current_score in valid_predictions:
        overlap = False
        current_start, current_end = current_span

        for approved_span, _ in approved_spans:
            app_start, app_end = approved_span
            if current_start <= app_end and current_end >= app_start:
                overlap = True
                break

        if not overlap:
            approved_spans.append((current_span, current_score))

    return approved_spans

TECHNIQUE_FACTORIES = {
    "max": lambda **_: MaxPooler(),
    "mean": lambda **_: MeanPooler(),
    "top3": lambda **_: TopKPooler(k=3),
    "top5": lambda **_: TopKPooler(k=5),
    "conv2": lambda **_: Conv1DPooler(window_size=2),
    "conv3": lambda **_: Conv1DPooler(window_size=3),
    "span": lambda hidden_size=768, max_span_length=4, **_: SpanExtractionPooler(
        hidden_size=hidden_size,
        max_span_length=max_span_length,
    ),
    "span-max": lambda hidden_size=768, max_span_length=4, **_: SpanMaxPooler(
        hidden_size=hidden_size,
        max_span_length=max_span_length,
    ),
}
