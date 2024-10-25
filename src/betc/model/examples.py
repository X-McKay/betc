"""
Example script demonstrating different ways to configure and run experiments.
"""
from pathlib import Path

from betc.model.config import MetricType
from betc.model.config import MLflowConfig
from betc.model.config import ServiceMode
from betc.model.config import ServicesConfig
from betc.model.config import TensorBoardConfig
from betc.model.experimentation import ExperimentRunner


def run_local_experiment():
    """Run experiment with local MLflow and TensorBoard services."""
    # Configure local services
    services_config = ServicesConfig(
        mlflow=MLflowConfig(mode=ServiceMode.LOCAL, host="localhost", port=5000),
        tensorboard=TensorBoardConfig(mode=ServiceMode.LOCAL, host="localhost", port=6006),
    )

    # Initialize experiment runner
    runner = ExperimentRunner(
        dataset_name="dair-ai/emotion",
        base_model_name="bert-base-uncased",
        experiment_name="emotion-classification-local",
        cache_dir=Path("./cache"),
        output_dir=Path("./experiments"),
        services_config=services_config,
    )

    # Run experiments with custom configuration
    runner.run_experiments(
        base_model_names=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
        param_grid={
            "learning_rate": [1e-5, 2e-5, 3e-5],
            "batch_size": [16, 32],
            "dropout_rate": [0.1, 0.2],
            "warmup_ratio": [0.1],
            "num_epochs": [3],
            "weight_decay": [0.01],
            "metric_for_best_model": [MetricType.F1],
        },
    )


def run_remote_experiment():
    """Run experiment with remote MLflow and TensorBoard services."""
    # Configure remote services
    services_config = ServicesConfig(
        mlflow=MLflowConfig(
            mode=ServiceMode.REMOTE,
            host="http://mlflow.example.com",
            port=5000,
            tracking_uri="postgresql://user:pass@db.example.com/mlflow",
        ),
        tensorboard=TensorBoardConfig(
            mode=ServiceMode.REMOTE,
            host="http://tensorboard.example.com",
            port=6006,
            logdir="gs://my-bucket/tensorboard-logs",
        ),
    )

    runner = ExperimentRunner(
        dataset_name="dair-ai/emotion",
        base_model_name="bert-base-uncased",
        experiment_name="emotion-classification-remote",
        cache_dir=Path("./cache"),
        output_dir=Path("./experiments"),
        services_config=services_config,
    )

    runner.run_experiments(
        base_model_names=["bert-base-uncased"],  # Single model for remote testing
        param_grid={"learning_rate": [2e-5], "batch_size": [32], "dropout_rate": [0.1]},
    )


def run_hybrid_experiment():
    """Run experiment with local MLflow and remote TensorBoard."""
    # Configure hybrid services
    services_config = ServicesConfig(
        mlflow=MLflowConfig(mode=ServiceMode.LOCAL, host="localhost", port=5000),
        tensorboard=TensorBoardConfig(
            mode=ServiceMode.REMOTE,
            host="http://tensorboard.example.com",
            port=6006,
            logdir="gs://my-bucket/tensorboard-logs",
        ),
    )

    runner = ExperimentRunner(
        dataset_name="dair-ai/emotion",
        base_model_name="bert-base-uncased",
        experiment_name="emotion-classification-hybrid",
        cache_dir=Path("./cache"),
        output_dir=Path("./experiments"),
        services_config=services_config,
    )

    runner.run_experiments()  # Use default configurations


def run_minimal_experiment():
    """Run experiment with minimal configuration."""
    # Use default local services configuration
    runner = ExperimentRunner(
        dataset_name="dair-ai/emotion",
        base_model_name="bert-base-uncased",
        experiment_name="emotion-classification-minimal",
    )

    runner.run_experiments()  # Use default configurations


def run_custom_experiment():
    """Run experiment with custom dataset and extensive parameter search."""
    services_config = ServicesConfig(
        mlflow=MLflowConfig(mode=ServiceMode.LOCAL), tensorboard=TensorBoardConfig(mode=ServiceMode.LOCAL)
    )

    runner = ExperimentRunner(
        dataset_name="dair-ai/emotion",
        base_model_name="bert-base-uncased",
        experiment_name="emotion-classification-custom",
        services_config=services_config,
    )

    # Extensive parameter grid
    param_grid = {
        "learning_rate": [1e-5, 2e-5, 3e-5],
        "batch_size": [16, 32],
        "dropout_rate": [0.1, 0.2],
        "warmup_ratio": [0.05, 0.1],
        "num_epochs": [3, 5],
        "weight_decay": [0.01, 0.1],
        "gradient_accumulation_steps": [1, 2],
        "metric_for_best_model": [MetricType.F1, MetricType.ACCURACY],
        "early_stopping_patience": [2, 3],
    }

    runner.run_experiments(
        base_model_names=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"],
        param_grid=param_grid,
    )


if __name__ == "__main__":
    # Choose which experiment to run
    import sys

    experiment_types = {
        "local": run_local_experiment,
        "remote": run_remote_experiment,
        "hybrid": run_hybrid_experiment,
        "minimal": run_minimal_experiment,
        "custom": run_custom_experiment,
    }

    if len(sys.argv) != 2 or sys.argv[1] not in experiment_types:
        print(f"Usage: python {sys.argv[0]} <experiment_type>")
        print(f"Available experiment types: {', '.join(experiment_types.keys())}")
        sys.exit(1)

    experiment_types[sys.argv[1]]()
