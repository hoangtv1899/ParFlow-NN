import sys
import os.path
import numpy as np
import tensorflow as tf
import keras
import xarray as xr
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, TimeDistributed, Flatten, Dense, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint

from parflow_nn.losses import log_loss, rmse, var_loss, var_ratio, metrics
from parflow_nn.preprocess_PF import create_feature_or_target_da
from parflow_nn.write_nc import generate_nc_files, config as c


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python nn.py <run_dir>')
        sys.exit(0)

    run_dir, = sys.argv[1:]
    out_dir = generate_nc_files(run_dir)

    # Set GPU usage
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    run_name = os.path.basename(run_dir)
    static_file = os.path.join(out_dir, f'{run_name}_static.nc')
    precip_file = os.path.join(out_dir, f'{run_name}_precip.nc')
    prev_press_file = os.path.join(out_dir, f'{run_name}_prev_press.nc')
    target_satur_file = os.path.join(out_dir, f'{run_name}_satur.nc')
    target_press_file = os.path.join(out_dir, f'{run_name}_press.nc')

    # Forcing data
    forcing_input = xr.open_dataset(precip_file)
    forcing_feature_da, forcing_feature_names = create_feature_or_target_da(
        forcing_input,
        ['precip'],
        0,
        'feature',
        flx_same_dt=True
    )

    # Add channel dimension
    forcing_feature_da = forcing_feature_da.data[:, 0, :, :]
    forcing_feature_da = forcing_feature_da[..., np.newaxis]
    forcing_feature_da = forcing_feature_da[np.newaxis, ...]

    # Static inputs
    static_input_xr = xr.open_dataset(static_file)

    static_feature_da, static_feature_names = create_feature_or_target_da(
        static_input_xr,
        ['slope_x', 'slope_y', 'perm', 'poros', 'rel_perm_alpha', 'rel_perm_N', 'satur_alpha', 'satur_N',
            'satur_sres', 'satur_ssat', 'tensor_x', 'tensor_y', 'tensor_z', 'spec_storage', 'mannings'],
        0,
        'feature',
        flx_same_dt=True
    )

    # Reduce input
    one_layer_feats = ['slope_x', 'slope_y', 'spec_storage', 'mannings', 'tensor_x', 'tensor_y', 'tensor_z']
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
    new_static_feature_da = np.tile(new_static_feature_da, (forcing_feature_da.shape[1], 1, 1, 1))
    new_static_feature_da = new_static_feature_da[np.newaxis, ...]

    # Previous pressure level
    prev_press_input = xr.open_dataset(prev_press_file)
    prev_press_feature_da, prev_press_feature_names = create_feature_or_target_da(
        prev_press_input,
        ['prev_press'],
        0,
        'feature',
        flx_same_dt=True
    )
    prev_press_feature_da = np.swapaxes(prev_press_feature_da.data, 1, 2)
    prev_press_feature_da = np.swapaxes(prev_press_feature_da, 2, 3)
    prev_press_feature_da = prev_press_feature_da[np.newaxis, ...]

    target_press_input_xr = xr.open_dataset(target_press_file)
    target_satur_input_xr = xr.open_dataset(target_satur_file)
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

    batch_norm = c.nn.batch_norm
    pooling = c.nn.pooling
    l2 = c.nn.l2
    dr = c.nn.dr
    activation = c.nn.activation

    n_sample, n_timestep, nlat, nlon, n_static_feat = new_static_feature_da.shape
    static_nodes = [int(n_static_feat / 8), 48]
    _, n_timestep, nlat, nlon, nlev_press = prev_press_feature_da.shape
    _, _, nlat, nlon, nlev_forc = forcing_feature_da.shape
    dynamic_nodes = [16, 48]
    n_sample, n_timestep, nlat, nlon, target_number = target_da.shape

    lr = 1e-4
    loss_dict = {
        'mae': 'mae',
        'mse': 'mse',
        'log_loss': log_loss
    }

    # First model for static data
    model0 = Sequential()
    model0.add(
        ConvLSTM2D(
            # In convLSTM, #filters defines the output space dimensions & the capacity of the network.
            # Similar to #units in a LSTM
            filters=static_nodes[0],
            data_format='channels_last',
            kernel_size=(3, 3),
            padding='same',
            input_shape=(None, nlat, nlon, n_static_feat),
            return_sequences=True,
        )
    )
    if batch_norm:
        model0.add(BatchNormalization())
    if pooling:
        model0.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    model0.add(TimeDistributed(Flatten()))
    model0.add(Dense(static_nodes[-1]))

    # convLSTM model for previous press
    model1 = Sequential()
    model1.add(
        ConvLSTM2D(
            # In convLSTM, #filters defines the output space dimensions & the capacity of the network.
            # Similar to #units in a LSTM
            filters=dynamic_nodes[0],
            data_format='channels_last',
            kernel_size=(3, 3),
            padding='same',
            input_shape=(None, nlat, nlon, nlev_press),
            return_sequences=True,
        )
    )
    if batch_norm:
        model1.add(BatchNormalization())
    if pooling:
        model1.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    if len(dynamic_nodes) > 1:
        for h in dynamic_nodes[1:]:
            model1.add(ConvLSTM2D(
                filters=h,
                data_format='channels_last',
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True)
            )
            if batch_norm:
                model1.add(BatchNormalization())
            if pooling:
                model1.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    model1.add(TimeDistributed(Flatten()))
    model1.add(Dense(static_nodes[-1]))

    # convLSTM model for forcing
    model2 = Sequential()
    model2.add(
        ConvLSTM2D(
            # In convLSTM, #filters defines the output space dimensions & the capacity of the network.
            # Similar to #units in a LSTM
            filters=dynamic_nodes[0],
            data_format='channels_last',
            kernel_size=(3, 3),
            padding='same',
            input_shape=(None, nlat, nlon, nlev_forc),
            return_sequences=True,
        )
    )
    if batch_norm:
        model2.add(BatchNormalization())
    if pooling:
        model2.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    if len(dynamic_nodes) > 1:
        for h in dynamic_nodes[1:]:
            model2.add(
                ConvLSTM2D(
                    filters=h,
                    data_format='channels_last',
                    kernel_size=(3, 3),
                    padding='same',
                    return_sequences=True
                )
            )
            if batch_norm:
                model2.add(BatchNormalization())
            if pooling:
                model2.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model2.add(TimeDistributed(Flatten()))
    model2.add(Dense(static_nodes[-1]))

    # Combine models
    combined = concatenate([model0.output, model1.output, model2.output])
    z = Dense(nlat * nlon * target_number, activation="linear")(combined)
    final_model = Model(inputs=[model0.input, model1.input, model2.input], outputs=z)
    final_model.compile(Adam(lr), loss='mse', metrics=metrics)

    # Define the checkpoint
    filepath = c.nn.output
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Fit model
    t = c.nn.train_timesteps
    subset_target = target_da[:, :t, :, :, :]
    final_model.fit(
            x=[new_static_feature_da[:, :t, :, :, :],
               prev_press_feature_da[:, :t, :, :, :],
               forcing_feature_da[:, :t, :, :, :]],
            y=np.reshape(subset_target, (subset_target.shape[0], subset_target.shape[1], -1)),
            epochs=5, batch_size=nlat * nlon,
            callbacks=callbacks_list)

    # define the checkpoint
    new_model = keras.models.load_model(
        filepath,
        custom_objects={
            "tf": tf,
            "rmse": rmse,
            'log_loss': log_loss,
            "var_ratio": var_ratio,
            "var_loss": var_loss
        }
    )

    # fit the model
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    subset_target = target_da[:, :t, :, :, :]
    new_model.fit(
        x=[
            new_static_feature_da[:, :t, :, :, :],
            prev_press_feature_da[:, :t, :, :, :],
            forcing_feature_da[:, :t, :, :, :]
        ],
        y=np.reshape(subset_target, (subset_target.shape[0], subset_target.shape[1], -1)),
        epochs=c.nn.train_epochs,
        batch_size=nlat * nlon,
        callbacks=callbacks_list
    )

    t = c.nn.pred_timesteps
    pred = new_model.predict([
        new_static_feature_da[:, :t, :, :, :],
        prev_press_feature_da[:, :t, :, :, :],
        forcing_feature_da[:, :t, :, :, :]
    ])

    keras.utils.plot_model(new_model, to_file='model.png', show_shapes=True, show_layer_names=True)
