__author__ = 'yunbo'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm

class PredPP(tf.keras.layers.Layer):
    def __init__(self, shape, output_channels, num_layers, 
                 num_hidden, filter_size, seq_length, tln=True,
                 stride=1):
        super(PredPP, self).__init__()
        self.lstm = []
        self.cell = []
        self.hidden = []
        
        self.seq_length = seq_length
        self.num_layers = num_layers
        
        for i in range(num_layers):
            if i == 0:
                num_hidden_in = num_hidden[num_layers-1]
            else:
                num_hidden_in = num_hidden[i-1]
            self.new_cell = cslstm('lstm_'+str(i+1),
                              filter_size,
                              num_hidden_in,
                              num_hidden[i],
                              shape,
                              tln=tln)
            self.lstm.append(self.new_cell)
            self.cell.append(None)
            self.hidden.append(None)

        self.gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)
        self.conv2d = tf.keras.layers.SeparableConv2D(filters=output_channels,
                                         kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         name="back_to_pixel")
    def call(self, images):
        gen_images = []
        mem = None
        z_t = None
        for t in range(self.seq_length - 1):
            inputs = images[:,t]

            self.hidden[0], self.cell[0], mem = self.lstm[0](inputs, self.hidden[0], self.cell[0], mem)
            z_t = self.gradient_highway(self.hidden[0], z_t)
            self.hidden[1], self.cell[1], mem = self.lstm[1](z_t, self.hidden[1], self.cell[1], mem)

            for i in range(2, self.num_layers):
                self.hidden[i], self.cell[i], mem = self.lstm[i](self.hidden[i-1], self.hidden[i], self.cell[i], mem)

            x_gen = self.conv2d(self.hidden[self.num_layers-1])
            gen_images.append(x_gen)

        gen_images = tf.stack(gen_images)
        # [batch_size, seq_length, height, width, channels]
        gen_images = tf.transpose(gen_images, [1,0,2,3,4])
        return gen_images
    
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'lstm': self.lstm,
            'cell': self.cell,
            'hidden': self.hidden,
            'seq_length': self.seq_length,
            'num_layers': self.num_layers,
            'gradient_highway': self.gradient_highway,
            'conv2d': self.conv2d,
        })
        
        return config