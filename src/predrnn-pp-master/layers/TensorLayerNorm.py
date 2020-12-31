import tensorflow as tf

EPSILON = 0.00001

@tf.function
def tensor_layer_norm(x, state_name):
    x_shape = x.get_shape()
    dims = x_shape.ndims
    params_shape = x_shape[-1:]
    if dims == 4:
        m, v = tf.nn.moments(x, [1,2,3], keepdims=True)
    elif dims == 5:
        m, v = tf.nn.moments(x, [1,2,3,4], keepdims=True)
    else:
        raise ValueError('input tensor for layer normalization must be rank 4 or 5.')
    b = tf.zeros(params_shape)
    s = tf.ones(params_shape)
    x_tln = tf.nn.batch_normalization(x, m, v, b, s, EPSILON)
    return x_tln
