import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Multiply, Concatenate, Conv2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121, InceptionResNetV2, EfficientNetB0, InceptionV3, VGG19, ResNet101
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.optimizers import Adam,SGD
import tensorflow.keras.backend as K
from pathlib import Path
from BreastCancerClassification.entity.config_entity import ModelConfig
from BreastCancerClassification import logger

class PrepareModel:
    def __init__(self, config: ModelConfig,augmentation=True):
        self.config = config
        self.augmentation = augmentation

        self.collaborator = None
        #input layer
        self.input_shape = self.config.param_image_size
        self.input_layer = Input(shape=self.input_shape)

    def channel_attention(self,x, ratio=8):
        _, height, width, channels = x.shape

        # Shared layers
        shared_layer1 = Dense(channels // ratio, activation="relu", use_bias=False)
        shared_layer2 = Dense(channels, use_bias=False)

        # Global Average Pooling
        x1 = GlobalAveragePooling2D()(x)
        x1 = shared_layer1(x1)
        x1 = shared_layer2(x1)

        # Global Max Pooling
        x2 = GlobalMaxPooling2D()(x)
        x2 = shared_layer1(x2)
        x2 = shared_layer2(x2)

        feats = x1 + x2
        feats = Activation("sigmoid")(feats)
        feats = Multiply()([x, feats])

        return feats

    def spatial_attention(self,x, ratio=8):
        # Average Pooling
        x1 = K.mean(x, axis=-1, keepdims=True)

        # Global Max Pooling
        x2 = K.max(x, axis=-1, keepdims=True)

        feats = Concatenate()([x1, x2])
        feats = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(feats)
        feats = Multiply()([x, feats])

        return feats

    def cbam(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

    def get_augmentation_layers(self,inputs):
        if self.augmentation:
            augmentation_layer = preprocessing.RandomFlip("horizontal")(inputs)
            augmentation_layer = preprocessing.RandomFlip("vertical")(augmentation_layer)
            augmentation_layer = preprocessing.RandomRotation(0.2)(augmentation_layer)
            return augmentation_layer
        return inputs

    def get_feature_extractor(self, inputs):
        if self.config.param_feature_extractor == 'None':
            return None

        model_mapping = {
            'DenseNet121': DenseNet121,
            'InceptionResNetV2': InceptionResNetV2,
            'EfficientNetB0': EfficientNetB0,
            'InceptionV3': InceptionV3,
            'VGG19': VGG19,
            'ResNet101': ResNet101
        }

        feature_extractor_base = model_mapping.get(self.config.param_feature_extractor)

        if feature_extractor_base:
            feature_extractor_base = feature_extractor_base(weights=self.config.param_weights,
                                                            include_top=False,
                                                            input_shape=self.input_shape,
                                                            input_tensor=inputs)
            for layer in feature_extractor_base.layers:
                layer.trainable = False
            return GlobalAveragePooling2D()(self.cbam(feature_extractor_base.output))

        return None
    def get_collaborator(self, inputs):
        if self.config.param_collaborator == 'None':
            return None

        model_mapping = {
            'DenseNet121': DenseNet121,
            'InceptionResNetV2': InceptionResNetV2,
            'EfficientNetB0': EfficientNetB0,
            'InceptionV3': InceptionV3,
            'VGG19': VGG19,
            'ResNet101': ResNet101
        }

        collaborator_base = model_mapping.get(self.config.param_collaborator)

        if collaborator_base:
            collaborator_base = collaborator_base(weights=self.config.param_weights,
                                                            include_top=False,
                                                            input_shape=self.input_shape,
                                                            input_tensor=inputs)
            for layer in collaborator_base.layers:
                layer.trainable = True
            return GlobalAveragePooling2D()(self.cbam(collaborator_base.output))

        return None

    def compile_model(self):
        optim_mapping = {
            'Adam': Adam,
            'SGD': SGD,
        }

        optimizer = optim_mapping.get(self.config.param_optimizer)
        if optimizer:
            optimizer = optimizer(learning_rate=self.config.param_initial_learning_rate)
        else:
            optimizer = Adam(learning_rate=self.config.param_initial_learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.config.param_loss,
            metrics=self.config.param_metrics,
        )

    def build_model(self):

        inputs_ = self.get_augmentation_layers(self.input_layer)
        print(inputs_.shape)
        feb = self.get_feature_extractor(inputs_)
        cb = self.get_collaborator(inputs_)

        concat = None
        if feb == None and cb == None:
            logger.info("Failed Model Creation.")
            return None
        elif feb == None:
            logger.info("Model does not have the Feature Extraction branch")
            concat = cb
        elif cb == None:
            logger.info("Model does not have the Collaborative branch")
            concat = feb
        else:
            concat = Concatenate(name="conc_layer")([feb, cb])
        if concat == None:
            return None
        # Dense layers with dropout
        dense_layer = Dense(1024, activation='relu')(concat)
        dense_layer = Dropout(0.5)(dense_layer)
        dense_layer = Dense(512, activation='relu')(dense_layer)
        dense_layer = Dropout(0.3)(dense_layer)

        # Output layer
        output_layer = Dense(self.config.param_num_target_class, activation='sigmoid')(dense_layer)

        # Create the model
        self.model = Model(inputs=inputs_, outputs=output_layer)
        logger.info("Model Created")
        self.compile_model()
        self.save_model(path=self.config.config_model, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
