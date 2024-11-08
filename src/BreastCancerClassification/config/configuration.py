from pathlib import Path
from BreastCancerClassification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, MLFLOW_FILE_PATH
from BreastCancerClassification.utils.common import read_yaml, create_directories
from BreastCancerClassification.entity.config_entity import DataIngestionConfig, DataSplitConfig, ModelConfig, \
    TrainingConfig, EvaluationConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH,
                 mlflow_filepath=MLFLOW_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.mlflow = read_yaml(mlflow_filepath)

        """
        Create data_root_dir. dataset and Training result will be stored inside it.
        """
        self.data_root_dir = Path(self.config.data_root_dir)
        create_directories([self.data_root_dir])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([self.data_root_dir / config.root_dir])

        data_ingestion_config = DataIngestionConfig(config_root_dir=self.data_root_dir / config.root_dir,
                                                    config_compressed_file=Path(config.compressed_file),
                                                    config_extract_dir=self.data_root_dir / config.extract_dir
                                                    )
        return data_ingestion_config

    def get_data_split_config(self) -> DataSplitConfig:
        data_split_config = self.config.data_split
        data_ingestion_config = self.config.data_ingestion
        param = self.params.data
        create_directories([self.data_root_dir / data_split_config.root_dir])

        ret_data_split_config = DataSplitConfig(source_data_dir=self.data_root_dir / data_ingestion_config.extract_dir,
                                                config_root_dir=self.data_root_dir / data_split_config.root_dir,
                                                config_train_dir=self.data_root_dir / data_split_config.root_dir / data_split_config.train_dir,
                                                config_test_dir=self.data_root_dir / data_split_config.root_dir / data_split_config.test_dir,
                                                config_val_dir=self.data_root_dir / data_split_config.root_dir / data_split_config.val_dir,
                                                param_zoom_factor=str(param.zoom_factor),
                                                param_train_test_val_ratio=list(param.train_test_val_ratio)
                                                )
        return ret_data_split_config

    def get_model_config(self) -> ModelConfig:
        config = self.config.models
        param = self.params.models

        create_directories([Path(config.root_dir)])

        model_config = ModelConfig(config_root_dir=Path(config.root_dir),
                                   config_model=Path(config.root_dir) / config.model,
                                   param_feature_extractor=str(param.feature_extractor),
                                   param_collaborator=str(param.collaborator),
                                   param_image_size=list(param.image_size),
                                   param_num_target_class=int(param.num_target_class),
                                   param_weights=str(param.weights),
                                   param_optimizer=str(param.optimizer),
                                   param_initial_learning_rate=float(param.initial_learning_rate),
                                   param_loss=str(param.loss),
                                   param_metrics=list(param.metrics),
                                   )
        return model_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        param = self.params.training
        create_directories([self.data_root_dir / config.root_dir])

        training_config = TrainingConfig(config_root_dir=self.data_root_dir / config.root_dir,
                                         config_trained_model=Path(self.config.models.root_dir) / config.trained_model,
                                         param_batch_size=int(param.batch_size),
                                         param_epochs=int(param.epochs),
                                         param_decay_rate=float(param.decay_rate),
                                         param_decay_epoch=int(param.decay_epoch),
                                         )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(all_params=self.params,
                                       model_path=Path(self.config.models.root_dir) / self.config.training.trained_model,
                                       test_data_dir=self.data_root_dir / self.config.data_split.root_dir / self.config.data_split.test_dir,
                                       test_image_size=self.params.models.image_size,
                                       params_batch_size=self.params.evaluation.batch_size,
                                       class_mode=self.params.models.num_target_class,
                                       MLFLOW_TRACKING_URI=str(self.mlflow.MLFLOW_TRACKING_URI),
                                       MLFLOW_TRACKING_USERNAME=str(self.mlflow.MLFLOW_TRACKING_USERNAME),
                                       MLFLOW_TRACKING_PASSWORD=str(self.mlflow.MLFLOW_TRACKING_PASSWORD)

                                       )
        return eval_config
