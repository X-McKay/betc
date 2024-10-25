"""
run_experiments.py - Script to run experiments across multiple BERT variants and hyperparameters.
"""
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import mlflow
import torch
from tqdm.auto import tqdm

from betc.model.base import EmailClassifier
from betc.model.config import ExperimentConfig
from betc.model.config import ModelConfig
from betc.model.config import TrainingConfig
from betc.model.trainer import EmailClassifierTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_dataset() -> tuple[datasets.Dataset, datasets.Dataset]:
    """
    Load and prepare the email dataset.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: Training and evaluation datasets
    """
    # Load dataset from Hugging Face (replace with your dataset)
    dataset = datasets.load_dataset("dair-ai/emotion")

    # Split dataset
    train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
    return train_test["train"], train_test["test"]


def create_experiment_config() -> ExperimentConfig:
    """
    Create experiment configuration with models and parameter grid.

    Returns:
        ExperimentConfig: Configuration for the experiment
    """
    return ExperimentConfig(
        model_names=["distilbert-base-uncased", "roberta-base", "bert-base-uncased"],
        param_grid={
            "learning_rate": [1e-5, 2e-5, 3e-5],
            "batch_size": [16, 32],
            "dropout_rate": [0.1, 0.2],
            "hidden_size": [768],  # Same for all models for compatibility
            "warmup_steps": [100, 500],
            "weight_decay": [0.01, 0.1],
        },
        metric_for_best_model="eval_f1",
    )


def generate_configs(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Generate all possible combinations of hyperparameters.

    Args:
        param_grid (Dict[str, List[Any]]): Grid of parameters to search

    Returns:
        List[Dict[str, Any]]: List of parameter combinations
    """
    keys = param_grid.keys()
    values = param_grid.values()
    configs = []

    for instance in itertools.product(*values):
        config = dict(zip(keys, instance))
        configs.append(config)

    return configs


def run_single_experiment(
    model_name: str,
    param_config: dict[str, Any],
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    experiment_name: str,
) -> dict[str, float]:
    """
    Run a single experiment with given model and parameters.

    Args:
        model_name (str): Name of the BERT variant to use
        param_config (Dict[str, Any]): Configuration parameters
        train_dataset (datasets.Dataset): Training dataset
        eval_dataset (datasets.Dataset): Evaluation dataset
        experiment_name (str): Name of the experiment

    Returns:
        Dict[str, float]: Results of the experiment
    """
    try:
        # Create configs
        model_config = ModelConfig(
            model_name=model_name,
            dropout_rate=param_config["dropout_rate"],
            hidden_size=param_config["hidden_size"],
        )

        training_config = TrainingConfig(
            learning_rate=param_config["learning_rate"],
            batch_size=param_config["batch_size"],
            warmup_steps=param_config["warmup_steps"],
            weight_decay=param_config["weight_decay"],
            experiment_name=experiment_name,
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Initialize model and trainer
        model = EmailClassifier(model_config)
        trainer = EmailClassifierTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            model_config=model_config,
        )

        # Train and get results
        results = trainer.train()

        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        logger.error(f"Error in experiment with {model_name}: {e!s}")
        return {"error": str(e)}


def main():
    """Main function to run the experiments."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Create experiment directory
    experiment_dir = Path("experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    train_dataset, eval_dataset = load_dataset()

    # Get experiment configuration
    experiment_config = create_experiment_config()

    # Save experiment configuration
    with open(experiment_dir / "experiment_config.json", "w") as f:
        json.dump(experiment_config.dict(), f, indent=2)

    # Generate all parameter combinations
    param_configs = generate_configs(experiment_config.param_grid)

    # Setup MLflow experiment
    experiment_name = f"email_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)

    # Track best results
    best_results = {
        "model_name": None,
        "params": None,
        "metrics": None,
        "f1_score": float("-inf"),
    }

    # Run experiments
    total_experiments = len(experiment_config.model_names) * len(param_configs)
    logger.info(f"Starting {total_experiments} experiments...")

    with open(experiment_dir / "results.json", "w") as results_file:
        for model_name in experiment_config.model_names:
            for param_config in tqdm(param_configs, desc=f"Running {model_name} experiments"):
                # Run experiment
                results = run_single_experiment(
                    model_name=model_name,
                    param_config=param_config,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    experiment_name=experiment_name,
                )

                # Log results
                experiment_result = {
                    "model_name": model_name,
                    "params": param_config,
                    "metrics": results,
                }
                json.dump(experiment_result, results_file)
                results_file.write("\n")

                # Update best results
                if results.get("eval_f1", float("-inf")) > best_results["f1_score"]:
                    best_results = {
                        "model_name": model_name,
                        "params": param_config,
                        "metrics": results,
                        "f1_score": results["eval_f1"],
                    }

    # Save best results
    with open(experiment_dir / "best_results.json", "w") as f:
        json.dump(best_results, f, indent=2)

    logger.info(f"Best model: {best_results['model_name']}")
    logger.info(f"Best parameters: {best_results['params']}")
    logger.info(f"Best metrics: {best_results['metrics']}")


if __name__ == "__main__":
    main()
