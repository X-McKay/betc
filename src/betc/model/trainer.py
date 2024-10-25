"""
trainer.py - Trainer class for email classification model with comprehensive logging and metrics tracking.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

import datasets
import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from betc.model.base import EmailClassifier
from betc.model.config import ModelConfig
from betc.model.config import TrainingConfig

logger = logging.getLogger(__name__)


class EmailClassifierTrainer:
    """
    Trainer class for email classification model with comprehensive logging and evaluation.
    Supports mixed precision training, early stopping, and gradient accumulation.
    """

    def __init__(
        self,
        model: EmailClassifier,
        train_dataset: datasets.Dataset,
        eval_dataset: datasets.Dataset,
        training_config: TrainingConfig,
        model_config: ModelConfig,
    ):
        """
        Initialize the trainer with model and datasets.

        Args:
            model (EmailClassifier): Model to train
            train_dataset (datasets.Dataset): Training dataset
            eval_dataset (datasets.Dataset): Evaluation dataset
            training_config (TrainingConfig): Training configuration
            model_config (ModelConfig): Model configuration

        Raises:
            ValueError: If datasets are empty or configurations are invalid
        """
        # Validate inputs
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            raise ValueError("Training and evaluation datasets cannot be empty")

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.model_config = model_config

        # Setup device and move model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.base_model_name,
            use_fast=True,  # Use fast tokenizer for better performance
        )

        # Setup data loaders
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False)

        # Setup optimizer with weight decay fix (exclude bias and LayerNorm)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Setup learning rate scheduler
        num_training_steps = len(self.train_dataloader) * training_config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=training_config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Setup tensorboard
        self.tensorboard_dir = Path("runs") / training_config.experiment_name
        self.writer = SummaryWriter(self.tensorboard_dir)

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if training_config.fp16 else None

        # Initialize tracking variables
        self.best_metric = float("-inf")
        self.no_improvement_count = 0
        self.global_step = 0

    def _create_dataloader(self, dataset: datasets.Dataset, shuffle: bool) -> DataLoader:
        """
        Create a DataLoader with proper collation and preprocessing.

        Args:
            dataset (datasets.Dataset): Dataset to create loader for
            shuffle (bool): Whether to shuffle the data

        Returns:
            DataLoader: Configured data loader
        """
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
        )

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate function for processing batch data.

        Args:
            batch (List[Dict[str, Any]]): Batch of examples

        Returns:
            Dict[str, torch.Tensor]: Processed batch
        """
        texts = [example["text"] for example in batch]
        labels = [example["label"] for example in batch]

        # Tokenize all texts in batch
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model_config.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions (np.ndarray): Model predictions
            labels (np.ndarray): True labels

        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
        }

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_eval_loss = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        # Compute metrics
        metrics = self._compute_metrics(np.array(all_predictions), np.array(all_labels))

        # Add average loss
        metrics["loss"] = total_eval_loss / len(self.eval_dataloader)

        return metrics

    def train(self) -> dict[str, float]:
        """
        Train the model with full lifecycle management.

        Returns:
            Dict[str, float]: Final evaluation metrics

        Raises:
            RuntimeError: If training fails
        """
        try:
            # Start MLflow run
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run()
            run_name = f"{self.model_config.base_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                mlflow.log_params(
                    {
                        "base_model_name": self.model_config.base_model_name,
                        "learning_rate": self.training_config.learning_rate,
                        "batch_size": self.training_config.batch_size,
                        "num_epochs": self.training_config.num_epochs,
                        "warmup_steps": self.training_config.warmup_steps,
                        "weight_decay": self.training_config.weight_decay,
                    }
                )

                best_model_path = None

                # Training loop
                for epoch in range(self.training_config.num_epochs):
                    logger.info(f"Starting epoch {epoch + 1}/{self.training_config.num_epochs}")

                    self.model.train()
                    total_train_loss = 0
                    optimizer_steps = 0

                    # Training steps
                    for step, batch in enumerate(tqdm(self.train_dataloader, desc="Training")):
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                        # Forward pass with mixed precision
                        with torch.cuda.amp.autocast(enabled=bool(self.scaler)):
                            outputs = self.model(**batch)
                            loss = outputs["loss"] / self.training_config.gradient_accumulation_steps

                        # Backward pass with gradient scaling
                        if self.scaler:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        total_train_loss += loss.item()

                        # Update weights if gradient accumulation steps reached
                        if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                            if self.scaler:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.training_config.max_grad_norm,
                                )
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.training_config.max_grad_norm,
                                )
                                self.optimizer.step()

                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            optimizer_steps += 1

                            # Log training loss
                            if optimizer_steps % 10 == 0:
                                self.writer.add_scalar(
                                    "train/loss",
                                    total_train_loss / optimizer_steps,
                                    self.global_step,
                                )

                        self.global_step += 1

                        # Evaluate and log metrics
                        if self.global_step % self.training_config.evaluation_steps == 0:
                            metrics = self._evaluate()

                            # Log metrics
                            for metric_name, metric_value in metrics.items():
                                self.writer.add_scalar(
                                    f"eval/{metric_name}",
                                    metric_value,
                                    self.global_step,
                                )
                                mlflow.log_metric(
                                    f"eval_{metric_name}",
                                    metric_value,
                                    self.global_step,
                                )

                            # Check for improvement
                            current_metric = metrics[self.training_config.metric_for_best_model]
                            if current_metric > self.best_metric:
                                self.best_metric = current_metric
                                self.no_improvement_count = 0

                                # Save best model
                                if best_model_path:
                                    Path(best_model_path).unlink(missing_ok=True)
                                best_model_path = f"best_model_step_{self.global_step}.pt"
                                self.model.save_pretrained(best_model_path)
                                mlflow.log_artifact(best_model_path)
                            else:
                                self.no_improvement_count += 1

                            # Early stopping check
                            if self.no_improvement_count >= self.training_config.early_stopping_patience:
                                logger.info("Early stopping triggered")
                                break

                            # Switch back to training mode
                            self.model.train()

                    # End of epoch evaluation
                    metrics = self._evaluate()
                    logger.info(f"End of epoch {epoch + 1} metrics: {metrics}")

                # Load best model for final evaluation
                if best_model_path:
                    self.model = EmailClassifier.from_pretrained(best_model_path)
                    self.model.to(self.device)

                # Final evaluation
                final_metrics = self._evaluate()
                logger.info(f"Final metrics: {final_metrics}")

                return final_metrics

        except Exception as e:
            logger.error(f"Training failed: {e!s}")
            raise RuntimeError(f"Training failed: {e!s}")

        finally:
            # Cleanup
            self.writer.close()

    def save_checkpoint(self, path: str, metrics: Optional[dict[str, float]] = None) -> None:
        """
        Save a training checkpoint.

        Args:
            path (str): Path to save checkpoint
            metrics (Optional[Dict[str, float]]): Current metrics to save
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "model_config": self.model_config.dict(),
                "training_config": self.training_config.dict(),
            },
            "metrics": metrics,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "no_improvement_count": self.no_improvement_count,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load a training checkpoint.

        Args:
            path (str): Path to load checkpoint from

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        self.no_improvement_count = checkpoint["no_improvement_count"]
