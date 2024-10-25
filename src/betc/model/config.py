"""
config_models.py - Comprehensive configuration models for email classification system.
"""
import urllib.parse
from enum import Enum
from typing import Any
from typing import Optional
from typing import Union

from pydantic import AnyHttpUrl
from pydantic import BaseModel
from pydantic import Field
from pydantic import IPvAnyAddress
from pydantic import validator


class ServiceMode(str, Enum):
    """Deployment modes for services."""

    LOCAL = "local"  # Services run locally via docker-compose
    REMOTE = "remote"  # Services running on remote servers
    DISABLED = "disabled"  # Service is disabled


class MetricType(str, Enum):
    """Supported metrics for model evaluation."""

    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    LOSS = "loss"


class LRSchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class ServiceConfig(BaseModel):
    """Base configuration for services."""

    mode: ServiceMode = Field(ServiceMode.LOCAL, description="Deployment mode for the service")
    host: Optional[Union[AnyHttpUrl, IPvAnyAddress, str]] = Field(
        None, description="Host address for the service"
    )
    port: Optional[int] = Field(None, description="Port for the service")

    @property
    def uri(self) -> Optional[str]:
        """Generate URI for the service."""
        if self.mode == ServiceMode.DISABLED:
            return None

        if self.mode == ServiceMode.LOCAL:
            host = self.host or "localhost"
            port = self.port or self.default_port
        else:
            if not self.host:
                raise ValueError(f"Host must be specified for remote {self.__class__.__name__}")
            port = self.port or self.default_port

        # Handle case where host is already a complete URL
        if isinstance(self.host, str) and "://" in self.host:
            parsed = urllib.parse.urlparse(self.host)
            return f"{parsed.scheme}://{parsed.netloc}"

        return f"http://{host}:{port}"

    @property
    def default_port(self) -> int:
        """Default port for the service. Override in subclasses."""
        raise NotImplementedError


class MLflowConfig(ServiceConfig):
    """MLflow service configuration."""

    tracking_uri: Optional[str] = Field(
        None, description="MLflow tracking URI (e.g., sqlite:///mlflow.db for local backend)"
    )
    artifact_location: Optional[str] = Field(None, description="Base location for storing artifacts")
    registry_uri: Optional[str] = Field(None, description="URI for MLflow model registry")
    experiment_name: Optional[str] = Field(None, description="Default experiment name")

    @property
    def default_port(self) -> int:
        return 5000

    @validator("tracking_uri", pre=True, always=True)
    def validate_tracking_uri(cls, v, values):
        """Set default tracking URI based on mode."""
        if v is None and values.get("mode") == ServiceMode.LOCAL:
            return "sqlite:///mlflow.db"
        return v


class TensorBoardConfig(ServiceConfig):
    """TensorBoard service configuration."""

    logdir: Optional[str] = Field(None, description="Directory for TensorBoard logs")
    reload_interval: int = Field(5, description="Reload interval in seconds")

    @property
    def default_port(self) -> int:
        return 6006

    @validator("logdir", pre=True, always=True)
    def validate_logdir(cls, v, values):
        """Set default logdir based on mode."""
        if v is None and values.get("mode") == ServiceMode.LOCAL:
            return "runs"
        return v


class ServicesConfig(BaseModel):
    """Configuration for all services."""

    mlflow: MLflowConfig = Field(default_factory=MLflowConfig, description="MLflow service configuration")
    tensorboard: TensorBoardConfig = Field(
        default_factory=TensorBoardConfig, description="TensorBoard service configuration"
    )


class ModelConfig(BaseModel):
    """Configuration for the BERT model architecture and parameters."""

    base_model_name: str = Field(..., description="Name of the BERT variant to use")
    num_labels: int = Field(..., description="Number of classification labels")
    max_length: int = Field(..., description="Maximum sequence length for tokenization")
    dropout_rate: float = Field(0.1, description="Dropout rate for classification head")
    hidden_size: int = Field(..., description="Size of hidden layers")
    attention_probs_dropout_prob: float = Field(0.1, description="Attention dropout rate")
    hidden_dropout_prob: float = Field(0.1, description="Hidden layer dropout rate")
    use_cache: bool = Field(True, description="Whether to use cached key/value attention states")
    gradient_checkpointing: bool = Field(False, description="Use gradient checkpointing to save memory")

    class Config:
        arbitrary_types_allowed = True


class TrainingConfig(BaseModel):
    """Configuration for training parameters and settings."""

    experiment_name: str = Field("betc_classifier", description="Name of the experiment")
    # Basic training parameters
    optimizer_type: str = Field("adamw", description="Type of optimizer to use")
    learning_rate: float = Field(2e-5, description="Learning rate for optimization")
    weight_decay: float = Field(0.01, description="Weight decay for AdamW optimizer")
    adam_beta1: float = Field(0.9, description="Beta1 for Adam optimizer")
    adam_beta2: float = Field(0.999, description="Beta2 for Adam optimizer")
    adam_epsilon: float = Field(1e-8, description="Epsilon for Adam optimizer")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm for clipping")
    batch_size: int = Field(32, description="Batch size for training")
    eval_batch_size: int = Field(64, description="Batch size for evaluation")
    num_epochs: int = Field(3, description="Number of training epochs")
    max_steps: Optional[int] = Field(
        None, description="Maximum number of training steps (overrides num_epochs if set)"
    )

    # Optimizer configuration
    optimizer_type: str = Field("adamw", description="Type of optimizer to use")
    learning_rate: float = Field(2e-5, description="Learning rate for optimization")
    weight_decay: float = Field(0.01, description="Weight decay for AdamW optimizer")
    adam_beta1: float = Field(0.9, description="Beta1 for Adam optimizer")
    adam_beta2: float = Field(0.999, description="Beta2 for Adam optimizer")
    adam_epsilon: float = Field(1e-8, description="Epsilon for Adam optimizer")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm for clipping")

    # Scheduler configuration
    scheduler_type: LRSchedulerType = Field(
        LRSchedulerType.LINEAR, description="Type of learning rate scheduler"
    )
    warmup_steps: int = Field(0, description="Number of warmup steps")
    warmup_ratio: float = Field(0.1, description="Ratio of total steps to use for warmup")
    num_cycles: Optional[int] = Field(None, description="Number of cycles for cyclic schedulers")

    # Training optimizations
    gradient_accumulation_steps: int = Field(1, description="Number of steps for gradient accumulation")
    fp16: bool = Field(True, description="Whether to use mixed precision training")
    fp16_opt_level: str = Field("O1", description="Mixed precision optimization level")
    train_batch_size: Optional[int] = Field(
        None, description="Effective batch size for training (batch_size * gradient_accumulation_steps)"
    )

    # Evaluation and saving parameters
    evaluation_strategy: str = Field("steps", description="When to evaluate during training")
    eval_steps: int = Field(100, description="Number of steps between evaluations")
    save_strategy: str = Field("steps", description="When to save checkpoints")
    save_steps: int = Field(1000, description="Number of steps between saving checkpoints")
    save_total_limit: Optional[int] = Field(3, description="Maximum number of checkpoints to keep")
    metric_for_best_model: MetricType = Field(
        MetricType.F1, description="Metric to use for selecting best model"
    )
    greater_is_better: bool = Field(True, description="Whether higher metric values are better")

    # Early stopping parameters
    early_stopping_patience: int = Field(3, description="Number of evaluations to wait for improvement")
    early_stopping_threshold: float = Field(1e-4, description="Minimum change to qualify as improvement")

    # Logging parameters
    logging_strategy: str = Field("steps", description="When to log training metrics")
    logging_steps: int = Field(50, description="Number of steps between logging")
    logging_first_step: bool = Field(True, description="Whether to log metrics for first step")

    # Services configuration
    services: ServicesConfig = Field(
        default_factory=ServicesConfig, description="Services configuration for training"
    )

    # Data processing
    dataloader_num_workers: int = Field(4, description="Number of dataloader workers")
    dataloader_pin_memory: bool = Field(True, description="Whether to pin memory in dataloaders")

    # Additional training features
    use_data_parallel: bool = Field(True, description="Whether to use DataParallel for multi-GPU training")
    ddp_find_unused_parameters: bool = Field(False, description="Find unused parameters in DDP")
    seed: int = Field(42, description="Random seed for reproducibility")

    @validator("train_batch_size", pre=True, always=True)
    def validate_train_batch_size(cls, v, values):
        """Calculate effective training batch size."""
        if v is None and "batch_size" in values and "gradient_accumulation_steps" in values:
            return values["batch_size"] * values["gradient_accumulation_steps"]
        return v

    class Config:
        arbitrary_types_allowed = True


class ExperimentConfig(BaseModel):
    """Configuration for hyperparameter search experiments."""

    base_model_names: list[str] = Field(..., description="List of BERT variant model names to try")
    param_grid: dict[str, list[Any]] = Field(
        default_factory=lambda: {
            "learning_rate": [1e-5, 2e-5, 3e-5],
            "batch_size": [16, 32],
            "dropout_rate": [0.1, 0.2],
            "warmup_ratio": [0.1, 0.2],
            "weight_decay": [0.01, 0.1],
        },
        description="Grid of hyperparameters to search",
    )
    n_trials: int = Field(10, description="Number of trials for hyperparameter search")
    metric_for_best_model: MetricType = Field(MetricType.F1, description="Metric to optimize for")
    cross_validation_folds: Optional[int] = Field(
        None, description="Number of cross-validation folds (if None, uses train/val split)"
    )

    # Experiment tracking
    log_code: bool = Field(True, description="Whether to log code to MLflow")
    log_model_artifacts: bool = Field(True, description="Whether to log model artifacts")

    # Resource management
    max_concurrent_trials: Optional[int] = Field(None, description="Maximum number of concurrent trials")
    gpu_per_trial: float = Field(1.0, description="Number of GPUs per trial")

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
