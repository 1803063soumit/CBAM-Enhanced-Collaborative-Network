import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from BreastCancerClassification.entity.config_entity import EvaluationConfig
from BreastCancerClassification.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def data_generator(self):

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        if self.config.class_mode == 1:
            class_mode = "binary"
        else:
            class_mode = 'categorical'
        self.test_generator = test_datagen.flow_from_directory(
            self.config.test_data_dir,
            target_size=tuple(self.config.test_image_size[:2]),
            batch_size=self.config.params_batch_size,
            class_mode=class_mode,
            shuffle=False
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self.data_generator()
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.MLFLOW_TRACKING_URI)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "models", registered_model_name="ColabAttentionNet")
            else:
                mlflow.keras.log_model(self.model, "models")