__author__ = 'yunbo'

import tensorflow as tf
from layers.TensorLayerNorm import tensor_layer_norm

<<<<<<< HEAD
class GHU():
=======
class GHU(tf.keras.layers.Layer):
>>>>>>> merge from verde
    def __init__(self, layer_name, filter_size, num_features, tln=False,
                 initializer=0.001):
        """Initialize the Gradient Highway Unit.
        """
<<<<<<< HEAD
=======
        super(GHU, self).__init__()
>>>>>>> merge from verde
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_features = num_features
        self.layer_norm = tln
<<<<<<< HEAD
=======
        
>>>>>>> merge from verde
        if initializer == -1:
            self.initializer = None
        else:
            self.initializer = tf.random_uniform_initializer(-initializer,initializer)
<<<<<<< HEAD

=======
        
        self.conv2d_z = tf.keras.layers.SeparableConv2D(
                                                        self.num_features*2,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='state_to_state')
        self.conv2d_x = tf.keras.layers.SeparableConv2D(
                                                        self.num_features*2,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='input_to_state')
    
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'layer_name': self.layer_name,
            'filter_size': self.filter_size,
            'num_features': self.num_features,
            'layer_norm': self.layer_norm,
            'initializer': self.initializer,
            'conv2d_z': self.conv2d_z,
            'conv2d_x': self.conv2d_x,
        })
        
        return config
    
    @tf.function
>>>>>>> merge from verde
    def init_state(self, inputs, num_features):
        dims = inputs.get_shape().ndims
        if dims == 4:
            batch = inputs.get_shape()[0]
            height = inputs.get_shape()[1]
            width = inputs.get_shape()[2]
        else:
            raise ValueError('input tensor should be rank 4.')
        return tf.zeros([batch, height, width, num_features], dtype=tf.float32)
<<<<<<< HEAD

    def __call__(self, x, z):
        if z is None:
            z = self.init_state(x, self.num_features)
        with tf.compat.v1.variable_scope(self.layer_name):
            z_concat = tf.keras.layers.Conv2D(
                self.num_features*2,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='state_to_state')(z)
            if self.layer_norm:
                z_concat = tensor_layer_norm(z_concat, 'state_to_state')

            x_concat = tf.keras.layers.Conv2D(
                self.num_features*2,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='input_to_state')(x)
            if self.layer_norm:
                x_concat = tensor_layer_norm(x_concat, 'input_to_state')

            gates = tf.add(x_concat, z_concat)
            p, u = tf.split(gates, 2, 3)
            p = tf.nn.tanh(p)
            u = tf.nn.sigmoid(u)
            z_new = u * p + (1-u) * z
            return z_new
=======
    
    def __call__(self, x, z):
        if z is None:
            z = self.init_state(x, self.num_features)
        z_concat = self.conv2d_z(z)
        if self.layer_norm:
            z_concat = tensor_layer_norm(z_concat, 'state_to_state')

        x_concat = self.conv2d_x(x)
        if self.layer_norm:
            x_concat = tensor_layer_norm(x_concat, 'input_to_state')

        gates = tf.add(x_concat, z_concat)
        p, u = tf.split(gates, 2, 3)
        p = tf.nn.tanh(p)
        u = tf.nn.sigmoid(u)
        z_new = u * p + (1-u) * z
        return z_new
>>>>>>> merge from verde

