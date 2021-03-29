import os.path
import numpy as np
import xarray as xr
import time
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from parflow_nn.daily_average import daily_average
from parflow_nn.write_nc import generate_nc_files, generate_nc_file_stream
from parflow_nn.preprocess_PF import create_feature_or_target_da
from parflow_nn.predpp import PredPP

def train_step(model, input, target, static, learning_rate, weights = None):
    # prediction = model(input, training=True)

    loss_func = tf.keras.losses.MeanSquaredError()
    n_samples = input.shape[0]

    with tf.GradientTape() as ae_tape:
        total_loss = 0
        for j in range(n_samples):
            prediction = model([input[j, :][np.newaxis, ...], static[j, :][np.newaxis, ...]], training = True)
            targeti = target[j, 1:][np.newaxis, ...]
            # Calculate loss
            if weights is not None:
                prediction = prediction * weights[None, None, None, None, :]
                targeti = targeti * weights[None, None, None, None, :]
            loss = loss_func(prediction, targeti)
            # print(loss)
            total_loss += loss
    # Get the encoder and decoder variables
    trainable_vars = model.trainable_variables
    # Calculate gradient
    ae_grads = ae_tape.gradient(total_loss, trainable_vars)
    # And then apply the gradient to change the weights
    ae_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    ae_optimizer.apply_gradients(zip(ae_grads, trainable_vars))

    # Loss is returned to monitor it while training
    return total_loss, ae_optimizer

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
        out_arrs = []
        n_samples = feature_da.shape[0]
        for i in range(n_samples):
            forcing_mean = np.ma.mean(feature_da[i,:], axis = (0, 1, 2))
            forcing_std = np.ma.std(feature_da[i,:], axis = (0, 1, 2))
            forcing_mean[forcing_std == 0] = 0
            forcing_std[forcing_std == 0] = 1
            # broadcast back
            mean_broadcast = np.tile(forcing_mean, (feature_da.shape[1], feature_da.shape[2],
                                    feature_da.shape[3], 1))
            std_broadcast = np.tile(forcing_std, (feature_da.shape[1], feature_da.shape[2],
                                    feature_da.shape[3], 1))
            out_arr = (feature_da[i,:] - mean_broadcast) / std_broadcast
            out_arrs.append(out_arr[np.newaxis,...])
        out_arrs = np.vstack(out_arrs)
        return tf.convert_to_tensor(out_arrs, dtype = tf.float32)

if __name__ == '__main__':
    # ------------------ Model architecture --------------------------------
    
    num_hidden = [1028]*8
    num_layers = len(num_hidden)
    delta = 0.00002
    base = 0.99998
    eta = 1
    reverse_input = False
    filter_size = 5
    
    # --------------------- Static ------------------------------
    
    is_clm = True
    NC_DIR = '/home/hvtran/taylor/nc_files'
    static_input = xr.open_dataset(os.path.join(NC_DIR, 'taylor_1983_static.nc'))
    static_feature_da, static_feature_names = create_feature_or_target_da(
            static_input,
            ['slope_x', 'slope_y', 'perm', 'poros',
             'spec_storage', 'mannings'],
            0,
            'feature',
            flx_same_dt=True
        )

    one_layer_feats = ['slope_x', 'slope_y', 'spec_storage', 'mannings']
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

    # ----------------------- Daily Variables ---------------------------
    
    NC_DIR = '/home/hvtran/taylor/daily_nc_files'
    years = [1983, 1985, 1989]
    
    ## ---------------------- Pressure ----------------------------------
    
    total_press_feature_da = []
    for year in years:
        press_input = xr.open_dataset(os.path.join(NC_DIR, 'taylor_'+str(year)+'_press.nc'))
        press_feature_da, press_feature_names = create_feature_or_target_da(
                press_input,
                ['press'],
                0,
                'feature',
                flx_same_dt=True
            )

        # Add channel dimension
        press_feature_da = press_feature_da.data[:]
        press_feature_da = np.swapaxes(press_feature_da, 1, 2)
        press_feature_da = np.swapaxes(press_feature_da, 2, 3)
        press_feature_da = press_feature_da[np.newaxis, ...]
        press_feature_da = press_feature_da[:, :365, :, :, :]
        # Append
        total_press_feature_da.append(press_feature_da)

    total_press_feature_da = np.vstack(total_press_feature_da)
    
    ## ---------------------- Forcings ----------------------------------
    
    total_forcing_feature_da = []
    for year in years:
        forcing_input = xr.open_dataset(os.path.join(NC_DIR, 'taylor_'+str(year)+'_forcings.nc'))
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

        forcing_feature_da = forcing_feature_da[:, :365, :, :, :]
        # Append
        total_forcing_feature_da.append(forcing_feature_da)

    total_forcing_feature_da = np.vstack(total_forcing_feature_da)
    
    ## ------------------------- Targets -----------------------------------
    
    total_target_da = []
    for year in years:
        target_flow_input = xr.open_dataset(os.path.join(NC_DIR, 'taylor_'+str(year)+'_flow.nc'))
        target_wtd_input = xr.open_dataset(os.path.join(NC_DIR, 'taylor_'+str(year)+'_wtd.nc'))
        target_clm_input = xr.open_dataset(os.path.join(NC_DIR, 'taylor_'+str(year)+'_clm.nc'))
        SWE_input = target_clm_input.clm[:,10:11,:,:]
        target_da = np.concatenate([target_flow_input.flow,
                                   target_wtd_input.wtd,
                                   SWE_input], axis=1)
        # target_da = np.concatenate([target_flow_input.flow], axis=1)
        target_da = target_da[np.newaxis, ...]
        target_da = np.swapaxes(target_da, 2, 3)
        target_da = np.swapaxes(target_da, 3, 4)
        target_da = target_da[:, :365, :, :, :]
        # Append
        total_target_da.append(target_da)

    total_target_da = np.vstack(total_target_da)
    
    ## ------------------------- Reshaping -----------------------------------
    
    # Trim to get dimension of 45 by 45
    forcing_feature_train = total_forcing_feature_da[:, :, :45, :45, [2,3,6,7]]
    target_train = total_target_da[:, :, :45, :45, :]
    new_static_feature_train = new_static_feature_da[:, :45, :45, :]
    press_feature_train = total_press_feature_da[:, 0:1, :45, :45, :]

    # Tile static
    n_sample = total_press_feature_da.shape[0]
    tile_static = np.tile(new_static_feature_train[np.newaxis,...], [n_sample, 1, 1, 1, 1])
    static_train = np.concatenate([press_feature_train, tile_static], axis = 4)
    
    ## ------------------------- Normalizing -----------------------------------
    
    forcing_norm_train = normalize_feature_da(forcing_feature_train)
    target_norm_train = normalize_feature_da(target_train)
    static_norm_train = normalize_feature_da(static_train)

    t0 = time.time()
    patch_size = tf.Variable(15)
    ims = reshape_patch(forcing_norm_train, patch_size)
    tars = reshape_patch(target_norm_train, patch_size)
    stas = static_norm_train
    # tars = tars[:, :, :, :, :50]
    t1 = time.time()
    print('reshape time: ' + str(t1 - t0))
    
    # --------------------------- Define Model ----------------------------------
    
    ae_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
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
    
    weights = np.array([3, 1, 2])
    weights = np.tile(weights.reshape(1,-1).T,patch_size**2).ravel()
    
    save_name = 'taylor_3_samples_daily_3_vars_8_layers_weights'
    
    try:
        _ = model([ims[0,:][np.newaxis,...]])
    except:
        print('continue')
    
    # ------------------------------ Training ---------------------------------
    
    t0 = time.time()
    lr = 1e-3
    curr_loss = 8
    for ii in range(300):
        loss, ae_optimizer = train_step(model, ims, tars, stas, lr, weights)
        if loss < curr_loss:
            print('save loss: '+str(loss))
            model.save_weights(save_name)
            curr_loss = loss

        if ii % 2 == 0:
            t1 = time.time()
            elapsed_time = t1 - t0
            t0 = time.time()
            print("loss {:1.6f}, time step {:1.0f}, elapsed_time {:2.4f} s".format(loss, ii, elapsed_time))


