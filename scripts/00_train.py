import os.path
import numpy as np
import xarray as xr
import time
import tensorflow as tf

from parflow_nn.preprocess_PF import create_feature_or_target_da
from parflow_nn.predpp import PredPP


def train_step(model, input, target, learning_rate):
    # prediction = model(input, training=True)

    loss_func = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as ae_tape:
        prediction = model(input)
        
        # Calculate loss
        loss = loss_func(target[:, 1:], prediction)
        assert loss.ndim > 0, "Model predicts NaN"
    # Get the encoder and decoder variables
    trainable_vars = model.trainable_variables
    # Calculate gradient
    ae_grads = ae_tape.gradient(loss, trainable_vars)
    # And then apply the gradient to change the weights
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ae_optimizer.apply_gradients(zip(ae_grads, trainable_vars))

    # Loss is returned to monitor it while training
    return loss, ae_optimizer


@tf.function
def reshape_patch(img_tensor, patch_size):
    batch_size = tf.shape(img_tensor)[0]
    seq_length = tf.shape(img_tensor)[1]
    img_height = tf.shape(img_tensor)[2]
    img_width = tf.shape(img_tensor)[3]
    num_channels = tf.shape(img_tensor)[4]
    a = tf.reshape(img_tensor, [batch_size, seq_length,
                                int(img_height/patch_size), patch_size,
                                int(img_width/patch_size), patch_size,
                                num_channels])
    b = tf.transpose(a, [0,1,2,4,3,5,6])
    patch_tensor = tf.reshape(b, [batch_size, seq_length,
                                  int(img_height/patch_size),
                                  int(img_width/patch_size),
                                  patch_size*patch_size*num_channels])
    return patch_tensor


def normalize_feature_da(feature_da, feature_names=None):
    """Normalize feature arrays, and optionally target array
    Args:
        feature_da: feature Dataset
        feature_names: Feature name strings
    Returns:
        da: Normalized DataArray
    """
    if feature_names is not None: # static inputs
        con_stats_norm = []
        for feati in feature_da:
            if len(np.unique(feati)) == 1:
                con_stats_norm.append(feati)
            else:
                meani = np.ma.mean(feati, axis = (2, 3))
                stdi = np.ma.std(feati, axis = (2, 3))
                meani[stdi == 0] = 0
                stdi[stdi == 0] = 1
                # broadcast back stdi
                stdi_broadcast = np.tile(stdi, (1, feati.shape[2], feati.shape[3], 1))
                stdi_broadcast = np.swapaxes(stdi_broadcast, 2, 3)
                stdi_broadcast = np.swapaxes(stdi_broadcast, 1, 2)
                # broadcast back meani
                meani_broadcast = np.tile(meani, (1, feati.shape[2], feati.shape[3], 1))
                meani_broadcast = np.swapaxes(meani_broadcast, 2, 3)
                meani_broadcast = np.swapaxes(meani_broadcast, 1, 2)
                feati_norm = (feati - meani_broadcast) / stdi_broadcast
                feati_norm = tf.conver_to_tensor(feati_norm, dtype = tf.float32)
                con_stats_norm.append(feati_norm)
        return con_stats_norm
    else: # forcing inputs and target
        forcing_mean = np.ma.mean(feature_da, axis = (0, 1, 2, 3))
        forcing_std = np.ma.std(feature_da, axis = (0, 1, 2, 3))
        forcing_mean[forcing_std == 0] = 0
        forcing_std[forcing_std == 0] = 1
        # broadcast back
        mean_broadcast = np.tile(forcing_mean, (1, feature_da.shape[1], feature_da.shape[2],
                                feature_da.shape[3], 1))
        std_broadcast = np.tile(forcing_std, (1, feature_da.shape[1], feature_da.shape[2],
                                feature_da.shape[3], 1))
        out_arr = (feature_da - mean_broadcast) / std_broadcast
        return tf.convert_to_tensor(out_arr, dtype = tf.float32)


if __name__ == '__main__':

    # --------------------------------------------------
    is_clm = True
    NC_DIR = 'nc_files'
    static_input = xr.open_dataset(os.path.join(NC_DIR, 'static.nc'))
    forcing_input = xr.open_dataset(os.path.join(NC_DIR, 'forcings.nc'))
    target_press_input_xr = xr.open_dataset(os.path.join(NC_DIR, 'press.nc'))
    target_satur_input_xr = xr.open_dataset(os.path.join(NC_DIR, 'satur.nc'))
    if is_clm:
        target_clm_input_xr = xr.open_dataset(os.path.join(NC_DIR, 'clm.nc'))

    num_hidden = [16, 16, 32, 32, 64, 64, 128, 128, 128, 48]
    num_layers = len(num_hidden)
    delta = 0.00002
    base = 0.99998
    eta = 1
    filter_size = 5
    reverse_input = True
    # --------------------------------------------------

    # TODO: The second argument is simply first_argument.data_vars.keys()
    static_feature_da, static_feature_names = create_feature_or_target_da(
        static_input,
        ['prev_press', 'slope_x', 'slope_y', 'perm', 'poros',
         'rel_perm_alpha', 'rel_perm_N',
         'satur_alpha', 'satur_N', 'satur_sres', 'satur_ssat',
         'tensor_x', 'tensor_y', 'tensor_z', 'spec_storage', 'mannings'],
        0,
        'feature',
        flx_same_dt=True
    )

    one_layer_feats = ['slope_x', 'slope_y', 'spec_storage', 'mannings',
                       'tensor_x', 'tensor_y', 'tensor_z']
    new_static_feature_da = []
    new_static_names = []
    for ii, fname in enumerate(static_feature_names.data):
        if fname.split('_lev')[0] in one_layer_feats:
            if int(fname[-2:]) == 0:
                new_static_feature_da.append(static_feature_da[:, ii, :, :])
                new_static_names.append(fname)
            else:
                continue
        else:
            new_static_feature_da.append(static_feature_da[:, ii, :, :])
            new_static_names.append(fname)

    new_static_feature_da = np.stack(new_static_feature_da, axis=0)
    new_static_feature_da = np.swapaxes(new_static_feature_da, 0, 1)
    new_static_feature_da = np.swapaxes(new_static_feature_da, 1, 2)
    new_static_feature_da = np.swapaxes(new_static_feature_da, 2, 3)

    # ---------------------------------------------
    # FORCING
    # ---------------------------------------------

    forcing_feature_da, forcing_feature_names = create_feature_or_target_da(
        forcing_input,
        ['forcings'],
        0,
        'feature',
        flx_same_dt=True
    )

    # Add channel dimension
    if is_clm:
        forcing_feature_da = forcing_feature_da.data[:]
        forcing_feature_da = np.swapaxes(forcing_feature_da, 1, 2)
        forcing_feature_da = np.swapaxes(forcing_feature_da, 2, 3)
        forcing_feature_da = np.repeat(forcing_feature_da,
                                       repeats=[2] + [1] * (forcing_feature_da.shape[0] - 1),
                                       axis=0)  # duplicate the first row
        forcing_feature_da = forcing_feature_da[np.newaxis, ...]
    else:
        forcing_feature_da = forcing_feature_da.data[:, 0, :, :]
        forcing_feature_da = forcing_feature_da[..., np.newaxis]
        forcing_feature_da = forcing_feature_da[np.newaxis, ...]

    # ---------------------------------------------
    # TARGETS
    # ---------------------------------------------

    if is_clm:
        target_clm = np.repeat(target_clm_input_xr.clm,
                               repeats=[2] + [1] * (target_clm_input_xr.clm.shape[0] - 1),
                               axis=0)  # duplicate the first row
        target_da = np.concatenate([target_press_input_xr.press,
                                    target_satur_input_xr.satur,
                                    target_clm], axis=1)
        target_da = target_da[np.newaxis, ...]
    else:
        target_dataset = target_press_input_xr.merge(target_satur_input_xr)
        target_da, target_names = create_feature_or_target_da(
            target_dataset,
            ['press', 'satur'],
            0,
            'target',
            1,
            flx_same_dt=True
        )

        target_da = target_da.data[np.newaxis, ...]

    target_da = np.swapaxes(target_da, 2, 3)
    target_da = np.swapaxes(target_da, 3, 4)
    print(target_da.shape)  # 1, 8761, 41, 41, 123

    list_days = np.arange(0, 365, 30)
    forcings = []
    targets = []
    for ii, timei in enumerate(list_days[:-1]):
        forcings.append(forcing_feature_da[0, 24*timei: 24*list_days[ii + 1], :, :, :])
        targets.append(target_da[0, 24*timei: 24*list_days[ii + 1], :, :, :])

    # forcing_feature_train = np.stack(forcings)
    # target_train = np.stack(targets)
    TRAIN_HOURS = 24 * 2
    forcing_feature_train = forcing_feature_da[:, :TRAIN_HOURS, :, :, :]
    target_train = target_da[:, :TRAIN_HOURS, :, :, :]

    forcing_norm_train = normalize_feature_da(forcing_feature_train)
    target_norm_train = normalize_feature_da(target_train)

    t0 = time.time()
    patch_size = tf.Variable(1)
    ims = reshape_patch(forcing_norm_train, patch_size)
    tars = reshape_patch(target_norm_train, patch_size)
    tars = tars[:, :, :, :, :50]
    t1 = time.time()
    print('reshape time: ' + str(t1 - t0))

    # --------------------------------------------------
    # OPTIMIZER AND LOSS FUNCTION
    # --------------------------------------------------
    # Optimizer and loss function
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    # MSE works here best
    loss_func = tf.keras.losses.MeanSquaredError()

    model = tf.keras.models.Sequential()
    mylayer = PredPP(ims.get_shape().as_list(), tars.shape[4],
                     num_layers, num_hidden,
                     filter_size,
                     ims.shape[1],
                     True,
                     init_mem=new_static_feature_da
                     )

    model.add(mylayer)
    model.compile(optimizer=ae_optimizer, loss=loss_func, metrics='mse')

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    t0 = time.time()
    for ii in range(51):
        loss, ae_optimizer = train_step(model, ims, tars, 1e-2)

        if reverse_input:
            ims_rev = ims[:, ::-1]
            tars_rev = tars[:, ::-1]
            tmp_loss, _ = train_step(model, ims_rev, tars_rev, 1e-2)
            loss += tmp_loss
            loss = loss / 2
        if ii % 1 == 0:
            t1 = time.time()
            elapsed_time = t1 - t0
            t0 = time.time()
            print("loss {:1.6f}, time step {:1.0f}, elapsed_time {:2.4f} s".format(loss.numpy(), ii, elapsed_time))

    model.save_weights('saved')
