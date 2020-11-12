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
from parflow_nn.write_nc import generate_nc_files
from parflow_nn.config import Config
from parflow_nn.cond_rnn.cond_rnn.cond_rnn2d import ConditionalRNN
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
run_name, out_dir, is_clm = 'washita_clm', '/home/hoang/work/ssd_NN/washita_clm/nc_files', True

static_file = os.path.join(out_dir, f'{run_name}_static.nc')
forcing_file = os.path.join(out_dir, f'{run_name}_forcings.nc')

# Forcing data
forcing_input = xr.open_dataset(forcing_file)

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

# Static inputs
static_input_xr = xr.open_dataset(static_file)

con_stats = []
con_stat_names = []
for var in static_input_xr.data_vars:
    tmp_stat = static_input_xr[var].data
    if var in ['slope_x', 'slope_y', 'spec_storage', 'mannings', 'tensor_x', 'tensor_y', 'tensor_z']:
        tmp_stat = tmp_stat[:, -1, :, :]
        tmp_stat = tmp_stat[np.newaxis, ...]
        tmp_stat = np.swapaxes(tmp_stat, 0, 1)
    con_stat_names.append(var)
    con_stats.append(tmp_stat)

kernel = 5
max_num_conditions = 50
multi_cond_to_init_state_Conv = []
for i in range(max_num_conditions):
    multi_cond_to_init_state_Conv.append(tf.keras.layers.Conv2D(filters = 1, kernel_size= kernel,
                                                                    padding = "same",
                                                                     data_format="channels_first"))

multi_cond_p = tf.keras.layers.ConvLSTM2D(filters = 8, kernel_size = kernel,
                                                        padding = "same",
                                                        data_format="channels_last")

init_state_list = []
for ii, c in enumerate(con_stats):
    init_state_list.append(multi_cond_to_init_state_Conv[ii](c))

multi_cond_state = multi_cond_p(tf.stack(init_state_list, axis=-1))

forcing_feature_da1 = forcing_feature_da[:, :24*30, :, :, :]

rnn = tf.keras.layers.ConvLSTM2D(filters = 8, kernel_size = 3,
                                 data_format="channels_last",
                                 padding = "same",
                                 input_shape = (None, None, 41, 41, 8))

out = rnn(forcing_feature_da1, initial_state = [multi_cond_state, multi_cond_state])










