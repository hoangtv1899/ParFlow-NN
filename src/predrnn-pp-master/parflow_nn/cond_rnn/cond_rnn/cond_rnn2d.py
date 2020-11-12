import tensorflow as tf


class ConditionalRNN2D(tf.keras.layers.Layer):

    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units, kernel, *args, **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.final_states = None
        self.init_state = None
        self.rnn = tf.keras.layers.ConvLSTM2D(filters = units, kernel_size = kernel,
                                                data_format="channels_last",
                                                padding = "same",
                                                *args, **kwargs)

        # multi cond
        max_num_conditions = 50
        self.multi_cond_to_init_state_Conv = []
        for i in range(max_num_conditions):
            self.multi_cond_to_init_state_Conv.append(tf.keras.layers.Conv2D(filters = 1, kernel_size= kernel,
                                                                    padding = "same",
                                                                     data_format="channels_first"))
        self.multi_cond_p = tf.keras.layers.ConvLSTM2D(filters = units, kernel_size = kernel,
                                                        padding = "same",
                                                        data_format="channels_last")


    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert (isinstance(inputs, list) or isinstance(inputs, tuple)) and len(inputs) >= 2
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_Conv[ii](c))
            multi_cond_state = self.multi_cond_p(tf.stack(init_state_list, axis=-1))
            self.init_state = multi_cond_state
        else:
            print('need to be multiple condition')
            return None
        out = self.rnn(x, initial_state = [self.init_state, self.init_state],
                         *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out