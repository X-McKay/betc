"""
model.py - BERT-based email classification model implementation.
"""
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoModel

from betc.model.config import ModelConfig


class EmailClassifier(nn.Module):
    """
    BERT-based model for email classification.

    This model uses a BERT variant as the backbone and adds a classification head
    for multi-class email classification.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the email classifier.

        Args:
            config (ModelConfig): Configuration object containing model parameters
        """
        super().__init__()

        # Load pre-trained BERT model and configuration
        self.config = config
        model_config = AutoConfig.from_pretrained(config.model_name, num_labels=config.num_labels)

        # Initialize BERT backbone
        self.bert = AutoModel.from_pretrained(config.model_name)

        # Classification head
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input sequences
            attention_mask (torch.Tensor): Attention mask for input sequences
            labels (Optional[torch.Tensor]): Ground truth labels

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing loss (if labels provided) and logits
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Use pooled output for classification
        pooled_output = outputs.pooler_output

        # Apply layer normalization and dropout
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Get logits
        logits = self.classifier(pooled_output)

        # Prepare output dictionary
        result = {"logits": logits}

        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            result["loss"] = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return result

    def save_pretrained(self, save_path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            save_path (str): Path to save the model
        """
        torch.save(
            {"model_state_dict": self.state_dict(), "config": self.config.dict()},
            save_path,
        )

    @classmethod
    def from_pretrained(cls, save_path: str) -> "EmailClassifier":
        """
        Load a model from the specified path.

        Args:
            save_path (str): Path to load the model from

        Returns:
            EmailClassifier: Loaded model instance
        """
        checkpoint = torch.load(save_path)
        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
