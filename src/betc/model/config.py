"""
config_models.py - Configuration classes for model and training parameters.
"""
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ModelConfig(BaseModel):
    """Configuration for the BERT model architecture and parameters."""

    model_name: str = Field(..., description="Name of the BERT variant to use (e.g., 'bert-base-uncased')")
    num_labels: int = Field(6, description="Number of classification labels")
    max_length: int = Field(512, description="Maximum sequence length for tokenization")
    dropout_rate: float = Field(0.1, description="Dropout rate for classification head")
    hidden_size: int = Field(768, description="Size of hidden layers")

    class Config:
        arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    """Configuration for training parameters and settings."""

    learning_rate: float = Field(2e-5, description="Learning rate for optimization")
    batch_size: int = Field(16, description="Batch size for training")
    num_epochs: int = Field(3, description="Number of training epochs")
    warmup_steps: int = Field(500, description="Number of warmup steps for learning rate scheduler")
    weight_decay: float = Field(0.01, description="Weight decay for AdamW optimizer")
    gradient_accumulation_steps: int = Field(1, description="Number of steps for gradient accumulation")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm for clipping")
    evaluation_steps: int = Field(100, description="Number of steps between evaluations")
    save_steps: int = Field(1000, description="Number of steps between model checkpoints")

    # MLflow configuration
    experiment_name: str = Field("email-classification", description="Name of MLflow experiment")
    run_name: Optional[str] = Field(None, description="Name of MLflow run")

    # Early stopping configuration
    early_stopping_patience: int = Field(3, description="Number of evaluations to wait for improvement")
    early_stopping_threshold: float = Field(1e-4, description="Minimum change to qualify as improvement")

    # Mixed precision training
    fp16: bool = Field(True, description="Whether to use mixed precision training")

    class Config:
        arbitrary_types_allowed = True


class ExperimentConfig(BaseModel):
    """Configuration for hyperparameter search experiments."""

    model_names: list[str] = Field(..., description="List of BERT variant model names to try")
    param_grid: dict[str, list[Any]] = Field(
        default_factory=lambda: {
            "learning_rate": [1e-5, 2e-5, 3e-5],
            "batch_size": [16, 32],
            "dropout_rate": [0.1, 0.2],
        },
        description="Grid of hyperparameters to search",
    )
    n_trials: int = Field(10, description="Number of trials for hyperparameter search")
    metric_for_best_model: str = Field("eval_f1", description="Metric to optimize for")

    class Config:
        arbitrary_types_allowed = True
