import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
import time
from pathlib import Path
from BreastCancerClassification.entity.config_entity import TrainingConfig,ModelConfig,DataSplitConfig
from BreastCancerClassification.utils.common import create_directories
from datetime import datetime

class Training:
    def __init__(self, training_config: TrainingConfig,model_config:ModelConfig,data_split_config:DataSplitConfig):
        self.training_config = training_config
        self.model_config = model_config
        self.data_split_config = data_split_config

    def get_model(self):
        self.model = tf.keras.models.load_model(self.model_config.config_model)

    def create_result_dir(self):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.train_result_dir = self.training_config.config_root_dir / formatted_datetime
        create_directories([self.train_result_dir])
    def create_callbacks(self):
        self.create_result_dir()

        filepath = self.train_result_dir / 'best_model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
        filepath = self.train_result_dir / 'training.csv'
        csv_logger = CSVLogger(filepath)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

        def lr_scheduler(epoch, lr):
            if epoch % self.training_config.param_decay_epoch == 0 and epoch > 0:
                lr = lr * self.training_config.param_decay_rate
            return lr

        lr_callback = LearningRateScheduler(lr_scheduler)

        self.callbacks = [checkpoint, csv_logger, lr_callback, early_stop]

    def data_generate(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        if self.model_config.param_num_target_class == 1:
            class_mode = 'binary'
        else:
            class_mode = 'categorical'
        self.train_generator = train_datagen.flow_from_directory(
            self.data_split_config.config_train_dir,
            target_size=tuple(self.model_config.param_image_size[:2]),
            batch_size=self.training_config.param_batch_size,
            class_mode=class_mode,
            shuffle=True
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_split_config.config_val_dir,
            target_size=tuple(self.model_config.param_image_size[:2]),
            batch_size=self.training_config.param_batch_size,
            class_mode=class_mode,
            shuffle=False
        )

        # image_shape = self.train_generator.image_shape
        # image_shape = self.val_generator.image_shape
        # tp = self.data_split_config.config_train_dir
        # isz = self.model_config.param_image_size
        # print(image_shape)
        # print(tp)
        # print(isz)
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.data_generate()
        self.get_model()
        self.create_callbacks()


        print("GPU available: ", "Yes" if tf.config.list_physical_devices('GPU') else "No")

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.training_config.param_batch_size,
            epochs=self.training_config.param_epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.training_config.param_batch_size,
            callbacks=self.callbacks,
            class_weight={0: 1.5961098398169336, 1: 0.7280793319415448}
        )

        self.save_model(
            path=self.training_config.config_trained_model,
            model=self.model
        )
