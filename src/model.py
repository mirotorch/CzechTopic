"""Cross-Encoder model for CzechTopic.

Architecture:
- [CLS] topic_name topic_description [SEP] text [SEP]
- Single BERT forward pass
- Dot product similarity between topic and text tokens
- Max pooling over topic tokens
- Sigmoid activation
"""

import torch
from transformers import BertModel, PreTrainedModel
import torch.nn.functional as F

from .config import CrossEncoderConfig


class TopicCrossEncoder(PreTrainedModel):
    def __init__(self, config: CrossEncoderConfig):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state

        topic_mask = (token_type_ids == 0) & (attention_mask == 1)
        text_mask = (token_type_ids == 1) & (attention_mask == 1)

        topic_hidden = hidden * topic_mask.unsqueeze(-1).float()
        text_hidden = hidden * text_mask.unsqueeze(-1).float()

        topic_hidden_norm = F.normalize(topic_hidden, p=2, dim=-1)
        text_hidden_norm = F.normalize(text_hidden, p=2, dim=-1)

        sim = torch.matmul(topic_hidden_norm, text_hidden_norm.transpose(-2, -1))

        topic_mask_2d = topic_mask.unsqueeze(-1).float()
        text_mask_2d = text_mask.unsqueeze(-1).float()
        sim_mask = torch.matmul(topic_mask_2d, text_mask_2d.transpose(-2, -1))

        sim = sim.masked_fill(sim_mask == 0, float("-inf"))

        sim_max = sim.max(dim=1).values
        scores = torch.sigmoid(sim_max)

        scores = scores * text_mask.float()

        return {"logits": scores, "text_mask": text_mask}
