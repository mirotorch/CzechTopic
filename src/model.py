"""Cross-Encoder model for CzechTopic.

Architecture:
- [CLS] topic_name topic_description [SEP] text [SEP] (BERT)
- <s> topic_name topic_description </s></s> text </s> (RoBERTa)
- Single BERT/RoBERTa forward pass
- Dot product similarity between topic and text tokens
- Max pooling over topic tokens
- Sigmoid activation
"""

import torch
from transformers import PreTrainedModel, AutoModel
import torch.nn.functional as F

from .config import CrossEncoderConfig
from .techniques import TECHNIQUE_FACTORIES


def get_segment_masks(input_ids, attention_mask, sep_token_id):
    """Create topic and text masks based on SEP token positions.
    
    BERT format: [CLS] topic [SEP] text [SEP]
        - 2 SEPs total, text is between first and second SEP
    
    RoBERTa format: <s> topic </s></s> text </s>
        - 3 SEPs total, text starts after second SEP
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    topic_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    text_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for b in range(batch_size):
        sep_positions = (input_ids[b] == sep_token_id).nonzero(as_tuple=True)[0]
        num_seps = len(sep_positions)
        
        topic_indices = torch.arange(seq_len, device=device)
        
        if num_seps == 0:
            pass
        elif num_seps == 1:
            first_sep = sep_positions[0].item()
            topic_mask[b] = (topic_indices > 0) & (topic_indices < first_sep) & (attention_mask[b] == 1)
            text_mask[b] = (topic_indices > first_sep) & (attention_mask[b] == 1)
        elif num_seps == 2:
            first_sep = sep_positions[0].item()
            second_sep = sep_positions[1].item()
            topic_mask[b] = (topic_indices > 0) & (topic_indices < first_sep) & (attention_mask[b] == 1)
            text_mask[b] = (topic_indices > first_sep) & (topic_indices < second_sep) & (attention_mask[b] == 1)
        else:
            first_sep = sep_positions[0].item()
            second_sep = sep_positions[1].item()
            last_sep = sep_positions[-1].item()
            topic_mask[b] = (topic_indices > 0) & (topic_indices < first_sep) & (attention_mask[b] == 1)
            text_mask[b] = (topic_indices > second_sep) & (topic_indices < last_sep) & (attention_mask[b] == 1)
    
    return topic_mask, text_mask


class TopicCrossEncoder(PreTrainedModel):
    def __init__(
        self,
        config: CrossEncoderConfig,
        technique: str = "max",
        encoder=None,
        max_span_length: int = 4,
    ):
        super().__init__(config)
        if encoder is not None:
            self.bert = encoder
        else:
            self.bert = AutoModel.from_config(config)
        self.technique = technique
        self.pooler = TECHNIQUE_FACTORIES[technique](
            hidden_size=self.bert.config.hidden_size,
            max_span_length=max_span_length,
        )
        
        self.temperature = 10.0
        self.bias = 0.5
        
        self.sep_token_id = self._get_sep_token_id()
    
    def _get_sep_token_id(self):
        if hasattr(self.config, 'sep_token_id') and self.config.sep_token_id is not None:
            return self.config.sep_token_id
        if hasattr(self.bert.config, 'sep_token_id'):
            return self.bert.config.sep_token_id
        elif hasattr(self.bert.config, 'eos_token_id') and self.bert.config.eos_token_id is not None:
            return self.bert.config.eos_token_id
        elif hasattr(self.bert.config, 'decoder_start_token_id') and self.bert.config.decoder_start_token_id is not None:
            return self.bert.config.decoder_start_token_id
        elif hasattr(self.bert.config, 'pad_token_id') and self.bert.config.pad_token_id is not None:
            return self.bert.config.pad_token_id + 1
        return 2

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state

        topic_mask, text_mask = get_segment_masks(input_ids, attention_mask, self.sep_token_id)

        topic_hidden = hidden * topic_mask.unsqueeze(-1).float()
        text_hidden = hidden * text_mask.unsqueeze(-1).float()

        if self.technique == "span":
            topic_vector = hidden[:, 0, :]
            pooled = self.pooler(
                hidden_states=hidden,
                topic_vector=topic_vector,
                text_mask=text_mask,
            )
            sim_agg = pooled["token_scores"]
        else:
            topic_hidden_norm = F.normalize(topic_hidden, p=2, dim=-1)
            text_hidden_norm = F.normalize(text_hidden, p=2, dim=-1)

            sim = torch.matmul(topic_hidden_norm, text_hidden_norm.transpose(-2, -1))

            topic_mask_2d = topic_mask.unsqueeze(-1).float()
            text_mask_2d = text_mask.unsqueeze(-1).float()
            sim_mask = torch.matmul(topic_mask_2d, text_mask_2d.transpose(-2, -1))

            sim = sim.masked_fill(sim_mask == 0, float("-inf"))

            sim_agg = self.pooler(sim)
        scores = torch.sigmoid((sim_agg - self.bias) * self.temperature)

        scores = torch.clamp(scores, min=1e-7, max=1.0 - 1e-7)

        scores = scores * text_mask.float()

        result = {"logits": scores, "text_mask": text_mask}
        if self.technique == "span":
            result["span_scores"] = [
                torch.clamp(
                    torch.sigmoid((span_scores - self.bias) * self.temperature),
                    min=1e-7,
                    max=1.0 - 1e-7,
                )
                for span_scores in pooled["span_scores"]
            ]
            result["span_indices"] = pooled["span_indices"]
        return result
