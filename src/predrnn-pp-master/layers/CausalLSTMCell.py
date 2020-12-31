__author__ = 'yunbo'

import tensorflow as tf
from layers.TensorLayerNorm import tensor_layer_norm

class CausalLSTMCell(tf.keras.layers.Layer):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape, forget_bias=1.0, tln=False, initializer=0.001):
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        super(CausalLSTMCell, self).__init__()
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.batch = seq_shape[0]
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self.layer_norm = tln
        self._forget_bias = forget_bias
        self.initializer = tf.random_uniform_initializer(-initializer,initializer)
        self.conv2d_h = tf.keras.layers.SeparableConv2D(
                                                        self.num_hidden*4,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='temporal_state_transition')
        self.conv2d_c = tf.keras.layers.SeparableConv2D(
                                                        self.num_hidden*3,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='temporal_memory_transition')
        self.conv2d_m = tf.keras.layers.SeparableConv2D(
                                                        self.num_hidden*3,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='spatial_memory_transition')
        self.conv2d_x = tf.keras.layers.SeparableConv2D(
                                                        self.num_hidden*7,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='input_to_state')
        self.conv2d_c2m = tf.keras.layers.SeparableConv2D(
                                                        self.num_hidden*4,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='c2m')
        self.conv2d_m2o = tf.keras.layers.SeparableConv2D(
                                                        self.num_hidden,
                                                        self.filter_size, 1, padding='same',
                                                        kernel_initializer=self.initializer,
                                                        name='m_to_o')
        self.conv2d_mem = tf.keras.layers.SeparableConv2D(self.num_hidden, 1, 1,
                                                        padding='same', name='memory_reduce')
    
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'layer_name': self.layer_name,
            'filter_size': self.filter_size,
            'num_hidden_in': self.num_hidden_in,
            'num_hidden': self.num_hidden,
            'batch': self.batch,
            'height': self.height,
            'width': self.width,
            'layer_norm': self.layer_norm,
            '_forget_bias': self._forget_bias,
            'initializer': self.initializer,
            'conv2d_h': self.conv2d_h,
            'conv2d_c': self.conv2d_c,
            'conv2d_m': self.conv2d_m,
            'conv2d_x': self.conv2d_x,
            'conv2d_c2m': self.conv2d_c2m,
            'conv2d_m2o': self.conv2d_m2o,
            'conv2d_mem': self.conv2d_mem,
        })
        
        return config
        

    def __call__(self, x, h, c, m):
        if h is None:
            h = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if c is None:
            c = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if m is None:
            m = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden_in],
                         dtype=tf.float32)

        h_cc = self.conv2d_h(h)
        c_cc = self.conv2d_c(c)
        m_cc = self.conv2d_m(m)
        if self.layer_norm:
            h_cc = tensor_layer_norm(h_cc, 'h2c')
            c_cc = tensor_layer_norm(c_cc, 'c2c')
            m_cc = tensor_layer_norm(m_cc, 'm2m')

        i_h, g_h, f_h, o_h = tf.split(h_cc, 4, 3)
        i_c, g_c, f_c = tf.split(c_cc, 3, 3)
        i_m, f_m, m_m = tf.split(m_cc, 3, 3)

        if x is None:
            i = tf.sigmoid(i_h + i_c)
            f = tf.sigmoid(f_h + f_c + self._forget_bias)
            g = tf.tanh(g_h + g_c)
        else:
            x_cc = self.conv2d_x(x)
            if self.layer_norm:
                x_cc = tensor_layer_norm(x_cc, 'x2c')

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = tf.split(x_cc, 7, 3)

            i = tf.sigmoid(i_x + i_h + i_c)
            f = tf.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = tf.tanh(g_x + g_h + g_c)

        c_new = f * c + i * g

        c2m = self.conv2d_c2m(c_new)
        if self.layer_norm:
            c2m = tensor_layer_norm(c2m, 'c2m')

        i_c, g_c, f_c, o_c = tf.split(c2m, 4, 3)

        if x is None:
            ii = tf.sigmoid(i_c + i_m)
            ff = tf.sigmoid(f_c + f_m + self._forget_bias)
            gg = tf.tanh(g_c)
        else:
            ii = tf.sigmoid(i_c + i_x_ + i_m)
            ff = tf.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = tf.tanh(g_c + g_x_)

        m_new = ff * tf.tanh(m_m) + ii * gg

        o_m = self.conv2d_m2o(m_new)
        if self.layer_norm:
            o_m = tensor_layer_norm(o_m, 'm2o')

        if x is None:
            o = tf.tanh(o_h + o_c + o_m)
        else:
            o = tf.tanh(o_x + o_h + o_c + o_m)

        cell = tf.concat([c_new, m_new],-1)
        cell = self.conv2d_mem(cell)

        h_new = o * tf.tanh(cell)

        return h_new, c_new, m_new


