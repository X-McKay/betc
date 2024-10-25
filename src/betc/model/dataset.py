"""
data_manager.py - Dataset loading and preprocessing functionality with proper type handling.
"""
import json
import logging
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset loading, preprocessing, and analysis."""

    def __init__(
        self,
        dataset_name: str,
        base_model_name: str,
        cache_dir: Optional[str] = None,
        train_split: str = "train",
        val_split: str = "test",
    ):
        """
        Initialize the dataset manager.

        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            base_model_name: Name of the model to use (for tokenizer)
            cache_dir: Directory to cache the dataset
            train_split: Name of the training split
            val_split: Name of the validation split
        """
        self.dataset_name = dataset_name
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir
        self.train_split = train_split
        self.val_split = val_split

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load dataset
        self.dataset = self._load_dataset()

        # Analyze dataset
        self.dataset_info = self._analyze_dataset()

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def _compute_class_distribution(self) -> dict[str, dict[str, int]]:
        """Compute class distribution statistics."""

        def get_distribution(split_data):
            labels = split_data["label"]
            unique, counts = np.unique(labels, return_counts=True)
            return {str(label): int(count) for label, count in zip(unique, counts)}

        return {
            "train": get_distribution(self.dataset[self.train_split]),
            "validation": get_distribution(self.dataset[self.val_split]),
        }

    def _compute_sequence_length_stats(self) -> dict[str, float]:
        """Compute sequence length statistics for the dataset."""
        logger.info("Computing sequence length statistics...")

        def get_lengths(examples):
            tokenized = self.tokenizer(examples["text"], truncation=False, padding=False)
            return [len(ids) for ids in tokenized["input_ids"]]

        # Compute lengths for both splits
        train_lengths = get_lengths(self.dataset[self.train_split])
        val_lengths = get_lengths(self.dataset[self.val_split])
        all_lengths = train_lengths + val_lengths

        return {
            "max_length": float(np.max(all_lengths)),
            "min_length": float(np.min(all_lengths)),
            "mean_length": float(np.mean(all_lengths)),
            "median_length": float(np.median(all_lengths)),
            "95th_percentile": float(np.percentile(all_lengths, 95)),
            "99th_percentile": float(np.percentile(all_lengths, 99)),
        }

    def _get_unique_labels(self) -> list[str]:
        """Get sorted list of unique labels from the dataset."""
        train_labels = set(self.dataset[self.train_split]["label"])
        val_labels = set(self.dataset[self.val_split]["label"])

        # Ensure labels are consistent across splits
        if train_labels != val_labels:
            logger.warning("Label sets differ between training and validation splits!")

        return sorted(list(train_labels.union(val_labels)))

    def _analyze_dataset(self) -> dict:
        """Analyze the dataset and compute important statistics."""
        logger.info("Analyzing dataset...")

        # Get label information
        label_list = self._get_unique_labels()
        num_labels = len(label_list)

        # Compute sequence length statistics
        length_stats = self._compute_sequence_length_stats()

        # Get class distribution
        class_distribution = self._compute_class_distribution()

        # Create dataset info with converted types
        dataset_info = {
            "num_labels": num_labels,
            "label_list": label_list,
            "label_to_id": {str(label): i for i, label in enumerate(label_list)},
            "id_to_label": {str(i): str(label) for i, label in enumerate(label_list)},
            "sequence_length_stats": length_stats,
            "class_distribution": class_distribution,
            "num_train_examples": len(self.dataset[self.train_split]),
            "num_val_examples": len(self.dataset[self.val_split]),
        }

        # Convert all values to serializable types
        dataset_info = self._convert_to_serializable(dataset_info)

        # Save dataset info
        self._save_dataset_info(dataset_info)

        return dataset_info

    def _save_dataset_info(self, info: dict) -> None:
        """Save dataset information to a JSON file."""
        save_path = Path("dataset_info.json")
        try:
            # Convert to serializable format and save
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            logger.info(f"Dataset info saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving dataset info: {e!s}")

    def _load_dataset(self) -> DatasetDict:
        """Load and prepare the dataset."""
        logger.info(f"Loading dataset: {self.dataset_name}")
        try:
            dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)

            # Ensure we have the required splits
            if self.train_split not in dataset:
                raise ValueError(f"Training split '{self.train_split}' not found in dataset")
            if self.val_split not in dataset:
                raise ValueError(f"Validation split '{self.val_split}' not found in dataset")

            return dataset

        except Exception as e:
            logger.error(f"Error loading dataset: {e!s}")
            raise

    def prepare_dataset(
        self, max_length: Optional[int] = None, batch_size: int = 1000
    ) -> tuple[Dataset, Dataset]:
        """
        Prepare the dataset for training.

        Args:
            max_length: Maximum sequence length (defaults to 95th percentile if None)
            batch_size: Batch size for preprocessing

        Returns:
            Tuple of processed training and validation datasets
        """
        logger.info("Preparing dataset for training...")

        # Determine max length if not provided
        if max_length is None:
            max_length = int(self.dataset_info["sequence_length_stats"]["95th_percentile"])
            logger.info(f"Using 95th percentile as max_length: {max_length}")

        def preprocess_function(examples):
            # Tokenize texts
            tokenized = self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=max_length
            )

            # Convert labels to ids
            tokenized["labels"] = [
                self.dataset_info["label_to_id"][str(label)] for label in examples["label"]
            ]

            return tokenized

        # Process datasets
        train_dataset = self.dataset[self.train_split].map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset[self.train_split].column_names,
            desc="Processing training dataset",
        )

        val_dataset = self.dataset[self.val_split].map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=self.dataset[self.val_split].column_names,
            desc="Processing validation dataset",
        )

        return train_dataset, val_dataset

    @property
    def num_labels(self) -> int:
        """Get number of labels in the dataset."""
        return self.dataset_info["num_labels"]

    @property
    def recommended_max_length(self) -> int:
        """Get recommended maximum sequence length (95th percentile)."""
        return int(self.dataset_info["sequence_length_stats"]["95th_percentile"])
