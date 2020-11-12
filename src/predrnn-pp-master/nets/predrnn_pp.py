__author__ = 'yunbo'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.CausalLSTMCell import CausalLSTMCell as cslstm

def rnn(images, targets, num_layers, num_hidden, filter_size, seq_length, 
        stride=1, tln=True):

    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = targets.get_shape().as_list()[-1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in range(seq_length-1):
        reuse = bool(gen_images)
        with tf.compat.v1.variable_scope('predrnn_pp', reuse=reuse):
            inputs = images[:,t]

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)

            x_gen = tf.keras.layers.Conv2D(filters=output_channels,
                                             kernel_size=1,
                                             strides=1,
                                             padding='same',
                                             name="back_to_pixel")(hidden[num_layers-1])
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    if len(tf.unique(tf.compat.v1.layers.flatten(targets)[0,:])) == 1:
        return [gen_images, 0]
    else:
        loss = tf.nn.l2_loss(gen_images - targets[:,1:])
        #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
        return [gen_images, loss]

