import os.path
import tensorflow as tf

import parflow_nn
from parflow_nn.layers.GradientHighwayUnit import GHU as ghu
from parflow_nn.layers.CausalLSTMCell import CausalLSTMCell as cslstm


class PredPP(tf.keras.layers.Layer):
    def __init__(self, shape, output_channels, num_layers,
                 num_hidden, filter_size, tln=True,
                 stride=1, init_cond = False, static_shape = None):
        super(PredPP, self).__init__()
        self.lstm = []
        self.cell = []
        self.hidden = []

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.tln = tln
        self.init_cond = init_cond
        self.static_shape = static_shape

        self.gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)
        self.conv2d = tf.keras.layers.SeparableConv2D(filters=output_channels,
                                                      kernel_size=1,
                                                      strides=1,
                                                      padding='same',
                                                      name="back_to_pixel")

        self.conv2d_pre = tf.keras.layers.SeparableConv2D(filters=3,
                                                          kernel_size=5,
                                                          strides=1,
                                                          padding='same',
                                                          name="before_MobileNET")

        self.conv2d_post = tf.keras.layers.SeparableConv2D(filters=num_hidden[num_layers - 1],
                                                           kernel_size=5,
                                                           strides=1,
                                                           padding='same',
                                                           name="after_MobileNET")

    def build(self, input_shape):
        self.shape = input_shape[0]
        
        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers - 1]
            else:
                num_hidden_in = self.num_hidden[i - 1]
            self.new_cell = cslstm('lstm_' + str(i + 1),
                                   self.filter_size,
                                   num_hidden_in,
                                   self.num_hidden[i],
                                   self.shape,
                                   tln=self.tln)
            self.lstm.append(self.new_cell)
            self.cell.append(None)
            self.hidden.append(None)
        
        if self.static_shape:
            if self.static_shape[2] < 32:
                self.mob_width_mul = np.ceil(32/(self.static_shape[2])).astype(np.int)
                self.mob_width = self.static_shape[2]*self.mob_width_mul
            else:
                self.mob_width = self.static_shape[2]
                self.mob_width_mul = 1

            if self.static_shape[3] < 32:
                self.mob_height_mul = np.ceil(32/(self.static_shape[3])).astype(np.int)
                self.mob_height = self.static_shape[3]*self.mob_height_mul
            else:
                self.mob_height = self.static_shape[3]
                self.mob_height_mul = 1

            if self.init_cond:
                self.mobile_net = tf.keras.applications.MobileNet(
                    input_shape=(self.mob_width, self.mob_height, 3),
                    include_top=False,
                    weights=os.path.join(os.path.dirname(parflow_nn.__file__), 'data', 'mobilenet_1_0_224_tf_no_top.h5')
                )

                for layer in self.mobile_net.layers:
                    layer.trainable = False

    def call(self, inputs):
        images = inputs[0]
        init_mem = inputs[1]
        seq_length = images.shape[1]
        gen_images = []
        if self.init_cond:
            tmp_x = tf.keras.layers.UpSampling2D(size = (self.mob_width_mul, self.mob_height_mul))(init_mem)
            tmp_x = self.conv2d_pre(tmp_x)
            for layer in self.mobile_net.layers:
                tmp_x = layer(tmp_x)
                if layer.name == 'conv_pw_5_relu':
                    break
            tmp_x = tf.keras.layers.UpSampling2D(size=(int(self.mob_width / 5), int(self.mob_height / 5)))(tmp_x)
            tmp_x = tf.keras.layers.MaxPooling2D(pool_size = (self.mob_width/(self.shape[2]), self.mob_height/(self.shape[3])))(tmp_x)
            mem = self.conv2d_post(tmp_x)
        else:
            mem = None

        z_t = None
        for t in range(seq_length - 1):
            # print(t)
            inputs = images[:, t]

            self.hidden[0], self.cell[0], mem = self.lstm[0](inputs, self.hidden[0], self.cell[0], mem)
            z_t = self.gradient_highway(self.hidden[0], z_t)
            self.hidden[1], self.cell[1], mem = self.lstm[1](z_t, self.hidden[1], self.cell[1], mem)

            for i in range(2, self.num_layers):
                self.hidden[i], self.cell[i], mem = self.lstm[i](self.hidden[i - 1], self.hidden[i], self.cell[i], mem)

            x_gen = self.conv2d(self.hidden[self.num_layers - 1])
            gen_images.append(x_gen)

        gen_images = tf.stack(gen_images)
        gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
        return gen_images

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lstm': self.lstm,
            'cell': self.cell,
            'hidden': self.hidden,
            'shape': self.shape,
            'num_layers': self.num_layers,
            'gradient_highway': self.gradient_highway,
            'conv2d': self.conv2d,
        })

        return config