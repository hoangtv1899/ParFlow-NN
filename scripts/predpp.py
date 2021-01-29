import sys
sys.path.append('../src/predrnn-pp-master')

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm


class PredPP(tf.keras.layers.Layer):
    def __init__(self, shape, output_channels, num_layers,
                 num_hidden, filter_size, seq_length, tln=True,
                 stride=1, init_mem=None):
        super(PredPP, self).__init__()
        self.lstm = []
        self.cell = []
        self.hidden = []
        # self.shape = shape

        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.filter_size = filter_size
        self.tln = tln
        self.init_mem = init_mem

        # for i in range(num_layers):
        #     if i == 0:
        #         num_hidden_in = num_hidden[num_layers - 1]
        #     else:
        #         num_hidden_in = num_hidden[i - 1]
        #     self.new_cell = cslstm('lstm_' + str(i + 1),
        #                            filter_size,
        #                            num_hidden_in,
        #                            num_hidden[i],
        #                            shape,
        #                            tln=tln)
        #     self.lstm.append(self.new_cell)
        #     self.cell.append(None)
        #     self.hidden.append(None)

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
        # self.mobile_net = tf.keras.applications.MobileNet(input_shape=(shape[2], shape[3], 3),
        #                                                   include_top=False,
        #                                                   weights='mobilenet_1_0_224_tf_no_top.h5')
        # for layer in self.mobile_net.layers:
        #     layer.trainable = False
        self.conv2d_post = tf.keras.layers.SeparableConv2D(filters=num_hidden[num_layers - 1],
                                                           kernel_size=5,
                                                           strides=1,
                                                           padding='same',
                                                           name="after_MobileNET")

    def build(self, input_shape):
        self.shape = input_shape
        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers - 1]
            else:
                num_hidden_in = self.num_hidden[i - 1]
            self.new_cell = cslstm('lstm_' + str(i + 1),
                                   self.filter_size,
                                   num_hidden_in,
                                   self.num_hidden[i],
                                   input_shape,
                                   tln=self.tln)
            self.lstm.append(self.new_cell)
            self.cell.append(None)
            self.hidden.append(None)

        self.mobile_net = tf.keras.applications.MobileNet(input_shape=(input_shape[2], input_shape[3], 3),
                                                          include_top=False,
                                                          weights='mobilenet_1_0_224_tf_no_top.h5')
        for layer in self.mobile_net.layers:
            layer.trainable = False

    def call(self, input):
        images = input
        init_mem = self.init_mem
        gen_images = []
        if init_mem is not None:
            tmp_x = self.conv2d_pre(init_mem)
            ####Mobile Net
            # block1 = tf.keras.layers.UpSampling2D(size=(int(self.shape[2] / 5), int(self.shape[3] / 5)))(
            #     self.mobile_net.get_layer("conv_pw_5_relu").output)
            # block1 = tf.image.resize_with_pad(block1, self.shape[2], self.shape[3])
            # tmp_model = tf.keras.models.Model(self.mobile_net.input, block1)
            # tmp_x = tmp_model(tmp_x)
            for layer in self.mobile_net.layers:
                tmp_x = layer(tmp_x)
                if layer.name == 'conv_pw_5_relu':
                    break
            tmp_x = tf.keras.layers.UpSampling2D(size=(int(self.shape[2] / 5), int(self.shape[3] / 5)))(tmp_x)
            tmp_x = tf.image.resize_with_pad(tmp_x, self.shape[2], self.shape[3])
            ####
            mem = self.conv2d_post(tmp_x)
        else:
            mem = None

        z_t = tf.zeros((1, 41, 41, 16))
        for t in range(self.seq_length - 1):
            inputs = images[:, t]

            self.hidden[0], self.cell[0], mem = self.lstm[0](inputs, self.hidden[0], self.cell[0], mem)
            z_t = self.gradient_highway(self.hidden[0], z_t)
            self.hidden[1], self.cell[1], mem = self.lstm[1](z_t, self.hidden[1], self.cell[1], mem)

            for i in range(2, self.num_layers):
                self.hidden[i], self.cell[i], mem = self.lstm[i](self.hidden[i - 1], self.hidden[i], self.cell[i], mem)

            x_gen = self.conv2d(self.hidden[self.num_layers - 1])
            gen_images.append(x_gen)

        gen_images = tf.stack(gen_images)
        # [batch_size, seq_length, height, width, channels]
        gen_images = tf.transpose(gen_images, [1, 0, 2, 3, 4])
        return gen_images

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lstm': self.lstm,
            'cell': self.cell,
            #'hidden': self.hidden,
            'shape': self.shape,
            'seq_length': self.seq_length,
            'num_layers': self.num_layers,
            'gradient_highway': self.gradient_highway,
            'conv2d': self.conv2d,
        })

        return config