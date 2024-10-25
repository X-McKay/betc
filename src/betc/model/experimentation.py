"""
run_experiments.py - Orchestrates experiments across multiple models and hyperparameters.
"""
import itertools
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

import mlflow
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TimeElapsedColumn
from rich.table import Table

from betc.model.base import EmailClassifier
from betc.model.config import MLflowConfig
from betc.model.config import ServiceMode
from betc.model.config import ServicesConfig
from betc.model.config import TensorBoardConfig
from betc.model.config_factory import ConfigFactory
from betc.model.service_manager import ServiceManager
from betc.model.trainer import EmailClassifierTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])

logger = logging.getLogger(__name__)
console = Console()


class ExperimentRunner:
    """Manages and runs experiments across multiple models and hyperparameters."""

    def __init__(
        self,
        dataset_name: str,
        base_model_name: str,
        experiment_name: str,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        services_config: Optional[ServicesConfig] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            base_model_name: Base model to use
            experiment_name: Name of the experiment
            cache_dir: Directory for caching models and datasets
            output_dir: Directory for experiment outputs
            services_config: Configuration for MLflow and TensorBoard services
        """
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.experiment_name = experiment_name
        self.cache_dir = cache_dir or Path("./cache")
        self.output_dir = output_dir or Path("./experiments")

        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize configuration factory
        self.config_factory = ConfigFactory(
            dataset_name=dataset_name, base_model_name=base_model_name, cache_dir=str(self.cache_dir)
        )

        # Setup services configuration
        self.services_config = services_config or self._create_default_services_config()
        self.service_manager = ServiceManager(config=self.services_config, project_dir=self.output_dir)

        # Track best results
        self.best_results = {"base_model_name": None, "params": None, "metrics": None, "score": float("-inf")}

        # Setup experiment directory
        self.experiment_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir.mkdir(parents=True)

    def _create_default_services_config(self) -> ServicesConfig:
        """Create default local services configuration."""
        return ServicesConfig(
            mlflow=MLflowConfig(mode=ServiceMode.LOCAL, host="localhost", port=5000),
            tensorboard=TensorBoardConfig(mode=ServiceMode.LOCAL, host="localhost", port=6006),
        )

    def _generate_configs(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Generate all possible combinations of parameters."""
        keys = param_grid.keys()
        values = param_grid.values()
        return [dict(zip(keys, instance)) for instance in itertools.product(*values)]

    def _save_experiment_config(self, config: dict, services_info: dict[str, Any]) -> None:
        """
        Save experiment configuration and services info to file.

        Args:
            config: Experiment configuration
            services_info: Information about service endpoints
        """
        save_data = {
            "experiment_config": config,
            "services": services_info,
            "dataset": self.dataset_name,
            "base_model": self.base_model_name,
            "timestamp": datetime.now().isoformat(),
        }

        config_path = self.experiment_dir / "experiment_config.json"
        with config_path.open("w") as f:
            json.dump(save_data, f, indent=2)

    def _save_results(self, results: dict) -> None:
        """Save experiment results to file."""
        results_path = self.experiment_dir / "results.jsonl"
        with results_path.open("a") as f:
            f.write(json.dumps(results) + "\n")

    def _save_best_results(self) -> None:
        """Save best results to file."""
        best_path = self.experiment_dir / "best_results.json"
        with best_path.open("w") as f:
            json.dump(self.best_results, f, indent=2)

    def _display_experiment_info(
        self, model_variants: list[str], param_configs: list[dict[str, Any]]
    ) -> None:
        """Display experiment information in a rich format."""
        info_table = Table(title="Experiment Information")

        info_table.add_column("Setting", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Dataset", self.dataset_name)
        info_table.add_row("Base Model", self.base_model_name)
        info_table.add_row("Number of Model Variants", str(len(model_variants)))
        info_table.add_row("Number of Parameter Configs", str(len(param_configs)))
        info_table.add_row("Total Experiments", str(len(model_variants) * len(param_configs)))

        console.print(Panel(info_table))

    def _run_single_experiment(
        self,
        base_model_name: str,
        param_config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a single experiment with given model and parameters.

        Args:
            base_model_name: Name of the BERT variant to use
            param_config: Configuration parameters

        Returns:
            Dict containing experiment results
        """
        try:
            # Create model configuration
            model_config = self.config_factory.create_model_config(
                dropout_rate=param_config.get("dropout_rate", 0.1)
            )
            model_config.base_model_name = base_model_name

            # Create training configuration with services
            training_config = self.config_factory.create_training_config(
                experiment_name=self.experiment_name, services=self.services_config, **param_config
            )

            # Prepare dataset
            train_dataset, val_dataset = self.config_factory.data_manager.prepare_dataset(
                max_length=model_config.max_length
            )

            # Initialize model and trainer
            model = EmailClassifier(model_config)
            trainer = EmailClassifierTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                training_config=training_config,
                model_config=model_config,
            )

            # Train and get results
            run_name = f"{base_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # End any existing run before starting a new one
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run()

            with mlflow.start_run(run_name=run_name) as run:
                try:
                    # Log parameters
                    mlflow.log_params(
                        {
                            "base_model_name": base_model_name,
                            **param_config,
                            "dataset": self.dataset_name,
                            "num_train_examples": len(train_dataset),
                            "num_val_examples": len(val_dataset),
                        }
                    )

                    # Train model and get results
                    results = trainer.train()

                    # Log metrics
                    mlflow.log_metrics(results)

                    # Save best model
                    if results[f"eval_{training_config.metric_for_best_model}"] > self.best_results["score"]:
                        model_save_path = self.experiment_dir / f"best_model_{run.info.run_id}.pt"
                        trainer.save_checkpoint(model_save_path, results)
                        mlflow.log_artifact(str(model_save_path))

                    return {
                        "base_model_name": base_model_name,
                        "params": param_config,
                        "metrics": results,
                        "run_id": run.info.run_id,
                        "status": "success",
                    }

                except Exception as e:
                    mlflow.end_run()
                    logger.exception(f"Error during training run: {e!s}")
                    raise

                finally:
                    # Ensure run is ended
                    if mlflow.active_run():
                        mlflow.end_run()

        except Exception as e:
            logger.exception(f"Error in experiment with {base_model_name}")
            return {
                "base_model_name": base_model_name,
                "params": param_config,
                "error": str(e),
                "status": "failed",
            }

        finally:
            # Clean up
            del model
            del trainer
            torch.cuda.empty_cache()

    def _display_final_results(self) -> None:
        """Display final results in a rich format."""
        results_table = Table(title="Best Model Results")

        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Model", self.best_results["base_model_name"])

        for param, value in self.best_results["params"].items():
            results_table.add_row(f"Parameter: {param}", str(value))

        for metric, value in self.best_results["metrics"].items():
            results_table.add_row(f"Metric: {metric}", f"{value:.4f}")

        console.print(Panel(results_table))

        # Print locations of results
        console.print("\n[bold cyan]Results saved to:[/]")
        console.print(f"  • Config: {self.experiment_dir / 'experiment_config.json'}")
        console.print(f"  • Results: {self.experiment_dir / 'results.jsonl'}")
        console.print(f"  • Best Results: {self.experiment_dir / 'best_results.json'}")

    def run_experiments(
        self, base_model_names: Optional[list[str]] = None, param_grid: Optional[dict[str, list[Any]]] = None
    ) -> None:
        """
        Run experiments across multiple models and hyperparameters.

        Args:
            base_model_names: List of model names to try
            param_grid: Grid of parameters to search
        """
        try:
            # Start services
            with self.service_manager.service_context() as service_manager:
                # Get service URIs
                service_info = {
                    "mlflow_uri": service_manager.get_mlflow_uri(),
                    "tensorboard_logdir": service_manager.get_tensorboard_logdir(),
                }

                # Get experiment configuration
                experiment_config = self.config_factory.create_experiment_config(
                    base_model_names=base_model_names, param_grid=param_grid
                )

                # Save configurations
                self._save_experiment_config(experiment_config.dict(), service_info)

                # Generate parameter configurations
                param_configs = self._generate_configs(experiment_config.param_grid)

                # Display experiment information
                self._display_experiment_info(experiment_config.base_model_names, param_configs)

                # Setup MLflow experiment
                mlflow.set_tracking_uri(service_info["mlflow_uri"])

                # End any existing run
                if mlflow.active_run():
                    mlflow.end_run()

                # Set or create experiment
                try:
                    # Try to set experiment
                    mlflow.set_experiment(self.experiment_name)
                except Exception as e:
                    logger.warning(f"Error setting experiment: {e!s}. Creating new experiment.")
                    experiment = mlflow.create_experiment(self.experiment_name)
                    mlflow.set_experiment(experiment.experiment_id)

                # Calculate total experiments
                total_experiments = len(experiment_config.base_model_names) * len(param_configs)
                logger.info(f"Starting {total_experiments} experiments...")

                # Run experiments
                with Progress(
                    SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console
                ) as progress:
                    task = progress.add_task("Running experiments", total=total_experiments)

                    for base_model_name in experiment_config.base_model_names:
                        for param_config in param_configs:
                            # Run experiment
                            results = self._run_single_experiment(
                                base_model_name=base_model_name, param_config=param_config
                            )

                            # Save results
                            self._save_results(results)

                            # Update best results if successful
                            if (
                                results["status"] == "success"
                                and results["metrics"][f"eval_{experiment_config.metric_for_best_model}"]
                                > self.best_results["score"]
                            ):
                                self.best_results = {
                                    "base_model_name": base_model_name,
                                    "params": param_config,
                                    "metrics": results["metrics"],
                                    "score": results["metrics"][
                                        f"eval_{experiment_config.metric_for_best_model}"
                                    ],
                                }

                            progress.advance(task)

                # Save best results
                self._save_best_results()

                # Display final results
                self._display_final_results()

        except Exception as e:
            logger.exception(f"Error running experiments: {e!s}")
            raise

        finally:
            # Always ensure any active run is ended
            if mlflow.active_run():
                mlflow.end_run()

            # Cleanup any temporary files
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)


def main():
    """Command line interface for running experiments."""
    from pathlib import Path
    from typing import Optional

    import typer

    app = typer.Typer()

    @app.command()
    def run(
        dataset_name: str = typer.Option("dair-ai/emotion", help="Name of the dataset on HuggingFace Hub"),
        base_model_name: str = typer.Option("bert-base-uncased", help="Base BERT model to use"),
        experiment_name: str = typer.Option("emotion-classification", help="Name for the experiment"),
        model_variants: Optional[list[str]] = typer.Option(
            None, help="List of model variants to try (default: base model + distil variants)"
        ),
        cache_dir: Optional[Path] = typer.Option(None, help="Directory for caching models and datasets"),
        output_dir: Optional[Path] = typer.Option(None, help="Directory for experiment outputs"),
        mlflow_port: int = typer.Option(5000, help="Port for MLflow server"),
        tensorboard_port: int = typer.Option(6006, help="Port for TensorBoard server"),
    ):
        """Run experiments with specified configuration."""
        # Configure services
        services_config = ServicesConfig(
            mlflow=MLflowConfig(mode=ServiceMode.LOCAL, host="localhost", port=mlflow_port),
            tensorboard=TensorBoardConfig(mode=ServiceMode.LOCAL, host="localhost", port=tensorboard_port),
        )

        try:
            # Initialize and run experiments
            runner = ExperimentRunner(
                dataset_name=dataset_name,
                base_model_name=base_model_name,
                experiment_name=experiment_name,
                cache_dir=cache_dir,
                output_dir=output_dir,
                services_config=services_config,
            )

            runner.run_experiments(base_model_names=model_variants)

            console.print("[green]Experiments completed successfully![/]")

        except Exception as e:
            console.print(f"[red]Error running experiments: {e!s}[/]")
            raise typer.Exit(code=1)

    @app.command()
    def clean(
        output_dir: Path = typer.Option(
            Path("experiments"), help="Directory containing experiment outputs to clean"
        ),
        force: bool = typer.Option(False, "--force", "-f", help="Force cleanup without confirmation"),
    ):
        """Clean up experiment directories and cached files."""
        if output_dir.exists():
            if not force:
                confirm = typer.confirm(f"Are you sure you want to delete all contents in {output_dir}?")
                if not confirm:
                    raise typer.Abort()

            shutil.rmtree(output_dir)
            console.print(f"[green]Cleaned up {output_dir}[/]")

    @app.command()
    def list_experiments(
        output_dir: Path = typer.Option(Path("experiments"), help="Directory containing experiment outputs")
    ):
        """List all completed experiments and their results."""
        if not output_dir.exists():
            console.print("[yellow]No experiments found.[/]")
            return

        experiments_table = Table(title="Completed Experiments")
        experiments_table.add_column("Timestamp", style="cyan")
        experiments_table.add_column("Experiment Name", style="green")
        experiments_table.add_column("Best Model", style="blue")
        experiments_table.add_column("Best Score", style="magenta")

        for exp_dir in sorted(output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            best_results_path = exp_dir / "best_results.json"
            config_path = exp_dir / "experiment_config.json"

            if best_results_path.exists() and config_path.exists():
                with best_results_path.open() as f:
                    best_results = json.load(f)
                with config_path.open() as f:
                    config = json.load(f)

                experiments_table.add_row(
                    exp_dir.name,
                    config.get("experiment_config", {}).get("experiment_name", "N/A"),
                    best_results.get("base_model_name", "N/A"),
                    f"{best_results.get('score', 0):.4f}",
                )

        console.print(experiments_table)

    app()


if __name__ == "__main__":
    main()
