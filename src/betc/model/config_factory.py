"""
config_factory.py - Factory for creating and optimizing configurations.
"""
import itertools
import logging
import math
from typing import Any
from typing import Optional

import psutil
import torch
from transformers import AutoConfig

from betc.model.config import ExperimentConfig
from betc.model.config import LRSchedulerType
from betc.model.config import MetricType
from betc.model.config import MLflowConfig
from betc.model.config import ModelConfig
from betc.model.config import ServiceMode
from betc.model.config import ServicesConfig
from betc.model.config import TensorBoardConfig
from betc.model.config import TrainingConfig
from betc.model.dataset import DatasetManager

logger = logging.getLogger(__name__)


class ConfigFactory:
    """Factory for creating configurations based on dataset and model."""

    def __init__(self, dataset_name: str, base_model_name: str, cache_dir: Optional[str] = None):
        """
        Initialize the configuration factory.

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            base_model_name: Base model name to use
            cache_dir: Cache directory for models and datasets
        """
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir

        # Initialize dataset manager
        self.data_manager = DatasetManager(
            dataset_name=dataset_name, base_model_name=base_model_name, cache_dir=cache_dir
        )

        # Get model config
        self.model_config = AutoConfig.from_pretrained(base_model_name)

    def create_model_config(self, dropout_rate: float = 0.1, max_length: Optional[int] = None) -> ModelConfig:
        """
        Create model configuration with dynamic settings.

        Args:
            dropout_rate: Dropout rate for the model
            max_length: Maximum sequence length (uses recommended if None)

        Returns:
            ModelConfig: Configuration for the model
        """
        # Use recommended max length if not provided
        if max_length is None:
            max_length = self.data_manager.recommended_max_length
            logger.info(f"Using recommended max_length: {max_length}")

        return ModelConfig(
            base_model_name=self.base_model_name,
            num_labels=self.data_manager.num_labels,
            max_length=max_length,
            dropout_rate=dropout_rate,
            hidden_size=self.model_config.hidden_size,
            attention_probs_dropout_prob=dropout_rate,
            hidden_dropout_prob=dropout_rate,
            use_cache=not self._should_use_gradient_checkpointing(),
            gradient_checkpointing=self._should_use_gradient_checkpointing(),
        )

    def create_training_config(self, experiment_name: str, **kwargs: Any) -> TrainingConfig:
        """
        Create training configuration with dynamic settings.

        Args:
            experiment_name: Name of the experiment
            **kwargs: Override default training configuration parameters

        Returns:
            TrainingConfig: Configuration for training
        """
        # Calculate steps based on dataset size
        num_examples = self.data_manager.dataset_info["num_train_examples"]
        default_batch_size = self._calculate_optimal_batch_size()
        steps_per_epoch = math.ceil(num_examples / default_batch_size)

        # Set default number of epochs based on dataset size
        default_epochs = self._calculate_optimal_epochs(num_examples)

        # Calculate evaluation and logging steps
        eval_steps = min(steps_per_epoch // 2, 100)  # Evaluate twice per epoch or every 100 steps
        logging_steps = min(steps_per_epoch // 10, 50)  # Log 10 times per epoch or every 50 steps

        # Calculate warmup steps
        total_steps = steps_per_epoch * default_epochs
        warmup_steps = int(0.1 * total_steps)

        # Create services config
        services_config = kwargs.get(
            "services",
            ServicesConfig(
                mlflow=MLflowConfig(mode=ServiceMode.LOCAL),
                tensorboard=TensorBoardConfig(mode=ServiceMode.LOCAL),
            ),
        )

        # Create default config dict
        config_dict = {
            # Basic training parameters
            "batch_size": default_batch_size,
            "eval_batch_size": default_batch_size * 2,
            "num_epochs": default_epochs,
            "max_steps": kwargs.get("max_steps", None),
            # Optimizer
            "optimizer_type": "adamw",
            "learning_rate": kwargs.get("learning_rate", 2e-5),
            "weight_decay": kwargs.get("weight_decay", 0.01),
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            # Scheduler
            "scheduler_type": LRSchedulerType.LINEAR,
            "warmup_steps": warmup_steps,
            "warmup_ratio": 0.1,
            "num_cycles": None,
            # Training optimizations
            "gradient_accumulation_steps": self._calculate_gradient_accumulation_steps(),
            "fp16": torch.cuda.is_available(),
            "fp16_opt_level": "O1",
            # Evaluation and saving parameters
            "evaluation_strategy": "steps",
            "eval_steps": eval_steps,
            "save_strategy": "steps",
            "save_steps": steps_per_epoch,
            "save_total_limit": 3,
            "metric_for_best_model": MetricType.F1,
            "greater_is_better": True,
            # Early stopping parameters
            "early_stopping_patience": 3,
            "early_stopping_threshold": 1e-4,
            # Logging parameters
            "logging_strategy": "steps",
            "logging_steps": logging_steps,
            "logging_first_step": True,
            # Services configuration
            "services": services_config,
            # Data processing
            "dataloader_num_workers": self._calculate_optimal_workers(),
            "dataloader_pin_memory": torch.cuda.is_available(),
            # Additional training features
            "use_data_parallel": torch.cuda.device_count() > 1,
            "ddp_find_unused_parameters": False,
            "seed": 42,
            # Override with any provided parameters
            **kwargs,
        }

        return TrainingConfig(**config_dict)

    def create_experiment_config(
        self, base_model_names: Optional[list[str]] = None, param_grid: Optional[dict[str, list[Any]]] = None
    ) -> ExperimentConfig:
        """
        Create experiment configuration for hyperparameter search.

        Args:
            base_model_names: List of model names to try
            param_grid: Parameter grid for search

        Returns:
            ExperimentConfig: Configuration for experiment
        """
        # Default model variants if none provided
        if base_model_names is None:
            base_name = self.base_model_name.split("-")[0]
            base_model_names = [
                self.base_model_name,
                f"distil{self.base_model_name}" if base_name == "bert" else f"distil{base_name}-base",
                "roberta-base",
            ]

        # Default parameter grid if none provided
        if param_grid is None:
            # Calculate optimal batch sizes
            base_batch_size = self._calculate_optimal_batch_size()
            batch_sizes = [base_batch_size // 2, base_batch_size]

            # Calculate warmup ratios based on dataset size
            num_examples = self.data_manager.dataset_info["num_train_examples"]
            default_warmup = 0.1 if num_examples > 10000 else 0.2

            param_grid = {
                "learning_rate": [1e-5, 2e-5, 3e-5],
                "batch_size": batch_sizes,
                "dropout_rate": [0.1, 0.2],
                "warmup_ratio": [default_warmup],
                "weight_decay": [0.01, 0.1],
                "gradient_accumulation_steps": [self._calculate_gradient_accumulation_steps()],
                "num_epochs": [self._calculate_optimal_epochs(num_examples)],
                "fp16": [torch.cuda.is_available()],
            }

        # Get GPU information for resource allocation
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            max_concurrent = max(1, int(gpu_memory // 8))  # Assume 8GB per trial
        else:
            max_concurrent = 1

        return ExperimentConfig(
            base_model_names=base_model_names,
            param_grid=param_grid,
            n_trials=len(base_model_names) * len(self._generate_param_combinations(param_grid)),
            metric_for_best_model=MetricType.F1,
            cross_validation_folds=None,
            log_code=True,
            log_model_artifacts=True,
            max_concurrent_trials=max_concurrent,
            gpu_per_trial=1.0 if max_concurrent == 1 else 1.0 / max_concurrent,
        )

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on model size and available resources."""
        if not torch.cuda.is_available():
            return 8  # Conservative batch size for CPU

        # Get GPU memory in GB
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Base batch size on model size
        if self.model_config.hidden_size <= 768:  # BERT-base size
            base_size = 32
        else:  # BERT-large size
            base_size = 16

        # Adjust based on sequence length
        max_length = self.data_manager.recommended_max_length
        if max_length <= 128:
            multiplier = 1
        elif max_length <= 256:
            multiplier = 0.5
        else:
            multiplier = 0.25

        # Adjust based on GPU memory
        if gpu_memory >= 16:
            memory_multiplier = 2
        elif gpu_memory >= 8:
            memory_multiplier = 1
        else:
            memory_multiplier = 0.5

        return max(8, int(base_size * multiplier * memory_multiplier))

    def _calculate_gradient_accumulation_steps(self) -> int:
        """Calculate gradient accumulation steps based on available memory."""
        if not torch.cuda.is_available():
            return 2

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if gpu_memory >= 16:
            return 1
        elif gpu_memory >= 8:
            return 2
        else:
            return 4

    def _calculate_optimal_epochs(self, num_examples: int) -> int:
        """Calculate optimal number of epochs based on dataset size."""
        if num_examples < 1000:
            return 10  # Small dataset needs more epochs
        elif num_examples < 10000:
            return 5
        elif num_examples < 100000:
            return 3
        else:
            return 2  # Large dataset needs fewer epochs

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for data loading."""
        return min(4, psutil.cpu_count() // 2)

    def _should_use_gradient_checkpointing(self) -> bool:
        """Determine if gradient checkpointing should be used."""
        if not torch.cuda.is_available():
            return False

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return gpu_memory < 8

    def _generate_param_combinations(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Generate all possible combinations of parameters."""
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations
