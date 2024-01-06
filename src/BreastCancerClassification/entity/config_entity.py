from dataclasses import dataclass
from pathlib import Path

"""
The variable-names starting with 'config' are set it config_entity.yaml.
The variable-names starting with 'param' are set in param.yaml.
"""


@dataclass(frozen=True)
class DataIngestionConfig:
    config_root_dir: Path
    config_compressed_file: Path
    config_extract_dir: Path


@dataclass(frozen=True)
class DataSplitConfig:
    source_data_dir:Path
    config_root_dir: Path
    config_train_dir: Path
    config_test_dir: Path
    config_val_dir: Path
    param_zoom_factor: str
    param_train_test_val_ratio: list


@dataclass(frozen=True)
class ModelConfig:
    config_root_dir: Path
    config_model: Path
    param_feature_extractor: str
    param_collaborator: str
    param_image_size: list
    param_num_target_class: int
    param_weights: str
    param_optimizer: str
    param_initial_learning_rate: float
    param_loss: str
    param_metrics: list

@dataclass(frozen=True)
class TrainingConfig:
    config_root_dir: Path
    param_batch_size: int
    param_epochs: int
    param_decay_rate: float
    param_decay_epoch: int
@dataclass(frozen=True)
class EvaluationConfig:
    all_params: dict
    model_path: Path
    test_data_dir: Path
    test_image_size: list
    params_batch_size: int
    class_mode: int
    mlflow_uri: str

