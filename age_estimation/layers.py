"""Custom Keras layers."""
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class SoftArgmaxLayer(tf.keras.layers.Layer):
    """
    Custom Layer tính Tuổi kỳ vọng (Expected Age):
    E[y] = sum(k * P(k)) với k trong khoảng [0, max_age - 1]
    """

    def __init__(self, max_age=101, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age

    def build(self, input_shape):
        self.age_vector = tf.constant(np.arange(0, self.max_age), dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: Tensor có dạng (batch_size, max_age) chứa xác suất Softmax
        return tf.reduce_sum(inputs * self.age_vector, axis=-1, keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update({"max_age": self.max_age})
        return config
