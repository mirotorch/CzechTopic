"""Cross-Encoder model for Topic Localization in Czech Texts.

Runs BERT once on [CLS] topic [SEP] text [SEP] sequence,
then computes similarity between topic and text token representations.
"""

import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedModel

from .config import CrossEncoderConfig
from .similarity_head import SimilarityMatrixHead


class TopicCrossEncoder(PreTrainedModel):
    def __init__(self, config: CrossEncoderConfig):
        super().__init__(config)
        self.bert = BertModel(config)

        self.topic_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.topic_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        self.sim_head = SimilarityMatrixHead(
            topic_size=config.topic_hidden_size,
            text_size=config.hidden_size,
            aggregation=config.similarity_aggregation,
            dropout=config.dropout,
        )

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        topic_mask=None,
        text_mask=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state
        
        topic_mask_local = (token_type_ids == 0).long() * attention_mask
        text_mask_local = (token_type_ids == 1).long() * attention_mask
        
        topic_hidden = self.topic_encoder(hidden)
        
        text_hidden = hidden
        
        logits = self.sim_head(topic_hidden, text_hidden, topic_mask_local, text_mask_local)
        
        return {"logits": logits}
