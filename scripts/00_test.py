import os.path
import numpy as np
import xarray as xr
import time

import tensorflow.compat.v1 as tfv1
#tfv1.disable_eager_execution()

import tensorflow as tf
import matplotlib.pyplot as plt
from parflow_nn.preprocess_PF import create_feature_or_target_da
from parflow_nn.predpp import PredPP

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

@tf.function
def reshape_patch_back(patch_tensor, patch_size):
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = int(channels / (patch_size*patch_size))
    a = tf.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = tf.transpose(a, [0,1,2,4,3,5,6])
    img_tensor = tf.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor

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
    NC_DIR = '/home/hvtran/washita_clm/nc_files'
    static_input = xr.open_dataset(os.path.join(NC_DIR, 'washita_clm_static.nc'))
    forcing_input = xr.open_dataset(os.path.join(NC_DIR, 'washita_clm_forcings.nc'))
    target_flow_input_xr = xr.open_dataset(os.path.join(NC_DIR, 'washita_clm_flow.nc'))
    #target_wtd_input_xr = xr.open_dataset(os.path.join(NC_DIR, 'washita_clm_wtd.nc'))

    # --------------------------------------------------

    num_hidden = [1028]*8
    num_layers = len(num_hidden)
    delta = 0.00002
    base = 0.99998
    eta = 1
    reverse_input = True
    filter_size = 5
    
    # --------------------------------------------------

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
    
    # --------------------------------------------------

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

    # --------------------------------------------------

    target_da = np.concatenate([target_flow_input_xr.flow], axis=1)
    target_da = target_da[np.newaxis, ...]


    target_da = np.swapaxes(target_da, 2, 3)
    target_da = np.swapaxes(target_da, 3, 4)
    print(target_da.shape)  # 1, 8761, 41, 41, 123

    # --------------------------------------------------

    # Trim to get dimension of 40 by 40
    n_sample = 3
    n_days = 30
    TRAIN_HOURS = 24 * n_days * n_sample
    forcing_feature_train = forcing_feature_da[:, :TRAIN_HOURS, :40, :40, [2,3,6,7]]
    target_train = target_da[:, :TRAIN_HOURS, :40, :40, :]
    new_static_feature_da = new_static_feature_da[:, :40, :40, :]

    # Reshape based on number of samples
    forcing_feature_train = np.reshape(forcing_feature_train, (n_sample, 24 * n_days, forcing_feature_train.shape[2], forcing_feature_train.shape[3],
                                                               forcing_feature_train.shape[4]))
    target_train = np.reshape(target_train, (n_sample, 24 * n_days, target_train.shape[2], target_train.shape[3],
                                                               target_train.shape[4]))

    forcing_norm_train = normalize_feature_da(forcing_feature_train)
    target_norm_train = normalize_feature_da(target_train)

    t0 = time.time()
    patch_size = tf.Variable(20)
    ims = reshape_patch(forcing_norm_train, patch_size)
    tars = reshape_patch(target_norm_train, patch_size)
    # tars = tars[:, :, :, :, :50]
    t1 = time.time()
    print('reshape time: ' + str(t1 - t0))
    
    # --------------------------------------------------

    forcing_mean = np.mean(forcing_feature_train,axis=(2,3,))
    plt.plot(forcing_mean[0,:,1],'b')
    plt.plot(forcing_mean[1,:,1],'r')
    plt.plot(forcing_mean[2,:,1],'m')

    # --------------------------------------------------

    ae_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    # MSE works here best
    loss_func = tf.keras.losses.MeanSquaredError()

    model = tf.keras.models.Sequential()
    mylayer = PredPP(ims.get_shape().as_list(), tars.shape[4],
                     num_layers, num_hidden,
                     filter_size,
                     tln=True,
                     )

    model.add(mylayer)
    model.compile(optimizer=ae_optimizer, loss=loss_func, metrics='mse')

    save_name = '3_samples_2_weeks_8_layers_weights'

    _ = model.call(ims[0,:][np.newaxis,...])
    model.load_weights(save_name)
    model.summary()
    
    # --------------------------------------------------

    i = 1
    t0 = time.time()
    predict = model(ims[i,:][np.newaxis,...])
    t1 = time.time()
    print('predict time: '+str(t1-t0))

    predict = reshape_patch_back(predict.numpy(), 20)

    # --------------------------------------------------

    # de-normalization
    target_mean = np.ma.mean(target_train[i,:], axis = (0, 1, 2))
    target_std = np.ma.std(target_train[i,:], axis = (0, 1, 2))
    target_mean[target_std == 0] = 0
    target_std[target_std == 0] = 1
    # broadcast back
    mean_broadcast = np.tile(target_mean, (target_train.shape[1], target_train.shape[2],
                            target_train.shape[3], 1))
    std_broadcast = np.tile(target_std, (target_train.shape[1], target_train.shape[2],
                            target_train.shape[3], 1))
    denorm_pred = (predict[0,:] * std_broadcast[1:,:]) + mean_broadcast[1:,:]
    denorm_pred = denorm_pred[np.newaxis,...]

    # --------------------------------------------------

    # evaluate loss
    loss_func = tf.keras.losses.MeanSquaredError()
    loss_val = loss_func(predict,target_norm_train[i,1:][np.newaxis, ...])
    print(loss_val)

    # --------------------------------------------------

    fig, axs = plt.subplots(1,2, figsize = (18, 18))

    ax0 = axs[0]
    im0 = ax0.imshow(denorm_pred[0, 600, :, :, 0])
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(im0, cax = cax, orientation = 'vertical')

    ax1 = axs[1]
    im1 = ax1.imshow(target_train[0, 601, :, :, 0])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(im1, cax = cax, orientation = 'vertical')

    plt.show()

    # --------------------------------------------------

    plt.plot(denorm_pred[0,:,31,39,0],'b')
    plt.plot(target_train[i,:,31,39,0],'r')
    plt.show()