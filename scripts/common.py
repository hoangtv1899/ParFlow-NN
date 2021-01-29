import tensorflow as tf


def train_step(model, input, init_stat, target, learning_rate):
    # prediction = model(input, training=True)

    loss_func = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as ae_tape:
        # prediction = model({'images':input, 'init_mem':init_stat})
        prediction = model(input)
        # Calculate loss
        loss = loss_func(target[:, 1:], prediction)
    # Get the encoder and decoder variables
    trainable_vars = model.trainable_variables
    # Calculate gradient
    ae_grads = ae_tape.gradient(loss, trainable_vars)
    # And then apply the gradient to change the weights
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ae_optimizer.apply_gradients(zip(ae_grads, trainable_vars))

    # Loss is returned to monitor it while training
    return loss, ae_optimizer
