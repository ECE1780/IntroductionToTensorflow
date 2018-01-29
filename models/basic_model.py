import tensorflow as tf


class BasicModel:

    def __init__(self, resolution, channels):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None] + resolution + [channels],
                                                name='image')

        x = tf.reshape(self.input_placeholder, shape=[-1, resolution[0]*resolution[1]*channels])

        x = tf.layers.dense(inputs=x,
                            units=1000,
                            activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x,
                            units=1000,
                            activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x,
                            units=10)

        self.predictions = x
