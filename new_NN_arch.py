# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from neural_net.cbrain.imports import *
from neural_net.cbrain.data_generator import DataGenerator
from neural_net.cbrain.models import *
from neural_net.cbrain.legacy.losses import *
from tensorflow.keras.callbacks import LearningRateScheduler
from preprocess.preprocess_PF import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.callbacks import ModelCheckpoint
import time
#from cond_rnn import ConditionalRNN


# %%
## set GPU usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# %%
static_file = 'nc_file/LW_static.nc'
precip_file = 'nc_file/LW_precip.nc'
prev_press_file = 'nc_file/LW_prev_press.nc'
target_satur_file = 'nc_file/LW_satur.nc'
target_press_file = 'nc_file/LW_press.nc'


# %%
## forcing data
t5 = time.time()
forcing_input = xr.open_dataset(precip_file)
forcing_feature_da, forcing_feature_names = create_feature_or_target_da(
            forcing_input,
            ['precip'],
            0,
            'feature',
            flx_same_dt=True
    )

#adding channel dimension
forcing_feature_da = forcing_feature_da.data[:,0,:,:]
forcing_feature_da = forcing_feature_da[...,np.newaxis]
forcing_feature_da = forcing_feature_da[np.newaxis,...]
#merge_feature_da = merge_feature_da.data[np.newaxis,...]
t6 = time.time()
print('time to read forcing input data: '+str(np.around(t6-t5,3))+' s')


# %%
### static inputs
t1 = time.time()
static_input_xr = xr.open_dataset(static_file)
"""
fig,axs = plt.subplots(3,5,figsize=(16,8))
for ii,keyi in enumerate(static_input_xr.data_vars):
    ax = axs[ii//5,ii%5]
    tmp_im = ax.imshow(static_input_xr[keyi][0,-1,:,:])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(tmp_im, cax=cax, orientation='vertical')
    ax.set_title(keyi)
"""
"""
time_array = precip_input_xr['time'].data
repeat_static_input = xr.Dataset(coords={'lat':static_input_xr['lat'].data,
                                         'lon':static_input_xr['lon'].data,
                                         'lev':static_input_xr['lev'].data,'time':time_array})
for k in static_input_xr.data_vars:
    repeat_static_input[k] = (['time','lev','lat','lon'],np.tile(static_input_xr.data_vars[k],(time_array.shape[0],1,1,1)))
"""
static_feature_da, static_feature_names = create_feature_or_target_da(
            static_input_xr,
            ['slope_x','slope_y','perm','poros',
             'rel_perm_alpha','rel_perm_N',
             'satur_alpha','satur_N','satur_sres','satur_ssat',
             'tensor_x','tensor_y','tensor_z','spec_storage','mannings'],
            0,
            'feature',
            flx_same_dt=True
    )

t2 = time.time()
####reshape feature
#static_feature_da = reshape_da(static_feature_da)
print('time to read static data: '+str(np.around(t2-t1,3))+' s')


# %%
#reduce input
t3 = time.time()
one_layer_feats = ['slope_x','slope_y','spec_storage','mannings',
                  'tensor_x','tensor_y','tensor_z']
new_static_feature_da = []
new_static_names = []
for ii,fname in enumerate(static_feature_names.data):
    if fname.split('_lev')[0] in one_layer_feats:
        if int(fname[-2:]) == 0:
            new_static_feature_da.append(static_feature_da[:,ii,:,:])
            new_static_names.append(fname)
        else:
            continue
    else:
        new_static_feature_da.append(static_feature_da[:,ii,:,:])
        new_static_names.append(fname)

new_static_feature_da = np.stack(new_static_feature_da,axis=0)
new_static_feature_da = np.swapaxes(new_static_feature_da,0,1)
new_static_feature_da = np.swapaxes(new_static_feature_da,1,2)
new_static_feature_da = np.swapaxes(new_static_feature_da,2,3)
new_static_feature_da = np.tile(new_static_feature_da,(forcing_feature_da.shape[1],1,1,1))
new_static_feature_da = new_static_feature_da[np.newaxis,...]
t4 = time.time()
print('time to reduce static input data: '+str(np.around(t4-t3,3))+' s')


# %%
new_static_feature_da.shape


# %%
#previous pressure level
t7 = time.time()
prev_press_input = xr.open_dataset(prev_press_file)
prev_press_feature_da, prev_press_feature_names = create_feature_or_target_da(
            prev_press_input,
            ['prev_press'],
            0,
            'feature',
            flx_same_dt=True
    )
prev_press_feature_da = np.swapaxes(prev_press_feature_da.data,1,2)
prev_press_feature_da = np.swapaxes(prev_press_feature_da,2,3)
prev_press_feature_da = prev_press_feature_da[np.newaxis,...]
t8 = time.time()
print('time to read previous press input data: '+str(np.around(t8-t7,3))+' s')


# %%
## read target files
t6 = time.time()
target_press_input_xr = xr.open_dataset(target_press_file)
target_satur_input_xr = xr.open_dataset(target_satur_file)
target_dataset = target_press_input_xr.merge(target_satur_input_xr)
target_da, target_names = create_feature_or_target_da(
                                                    target_dataset,
                                                    ['press','satur'],
                                                    0,
                                                    'target',
                                                    1,
                                                    flx_same_dt=True
                                                )
#target_da = reshape_da(target_da)
target_da = target_da.data[np.newaxis,...]
target_da = np.swapaxes(target_da,2,3)
target_da = np.swapaxes(target_da,3,4)
t7 = time.time()
print('time to read and prepare target data: '+str(np.around(t7-t6,3))+' s')


# %%
batch_norm = True
pooling = True
l2=None
dr=None
activation = 'relu'
n_sample,n_timestep,nlat,nlon,n_static_feat = new_static_feature_da.shape
static_nodes = [int(n_static_feat/8),48]
_,n_timestep,nlat,nlon,nlev_press = prev_press_feature_da.shape
_,_,nlat,nlon,nlev_forc = forcing_feature_da.shape
dynamic_nodes = [16,48]
n_sample,n_timestep,nlat,nlon,target_number = target_da.shape

lr = 1e-4
loss_dict = {
    'mae': 'mae',
    'mse': 'mse',
    'log_loss': log_loss
}


# %%
#define the model architecture
## first model for static data
time8 = time.time()
model0 = Sequential()
model0.add(
    ConvLSTM2D(filters=static_nodes[0],  # in convLSTM, #filters defines the output space dimensions & the capacity of the network. Similar to #units in a LSTM
               data_format='channels_last',
               kernel_size=(3,3), 
               padding='same',
                input_shape = (None,nlat,nlon,n_static_feat),
               return_sequences=True,
    )
)
if batch_norm:
    model0.add(BatchNormalization())
if pooling:
    model0.add(MaxPooling3D(pool_size=(1, 2, 2), 
                    padding='same', 
                    data_format='channels_last'))

model0.add(TimeDistributed(Flatten()))
model0.add(Dense(static_nodes[-1]))
t9 = time.time()
print('time to prepare model for static data: '+str(np.around(t9-t8,3))+' s')


# %%
model0.output


# %%
model0.summary()


# %%
##convLSTM model for previous press
t10 = time.time()
model1 = Sequential()
model1.add(
    ConvLSTM2D(filters=dynamic_nodes[0],  # in convLSTM, #filters defines the output space dimensions & the capacity of the network. Similar to #units in a LSTM
               data_format='channels_last',
               kernel_size=(3,3), 
               padding='same',
                input_shape = (None,nlat,nlon,nlev_press),
               return_sequences=True,
    )
)
if batch_norm:
    model1.add(BatchNormalization())
if pooling:
    model1.add(MaxPooling3D(pool_size=(1, 2, 2), 
                    padding='same', 
                    data_format='channels_last'))
if len(dynamic_nodes) >1:
    for h in dynamic_nodes[1:]:
        model1.add(ConvLSTM2D(filters=h,
                        data_format='channels_last',
                       kernel_size=(3,3), 
                       padding='same',
                       return_sequences=True))
        if batch_norm:
            model1.add(BatchNormalization())
        if pooling:
            model1.add(MaxPooling3D(pool_size=(1, 2, 2), 
                    padding='same', 
                    data_format='channels_last'))


model1.add(TimeDistributed(Flatten()))
model1.add(Dense(static_nodes[-1]))
#model1.add(Reshape(target_shape=(1,-1)))
t11 = time.time()
print('time to prepare model for prev press data: '+str(np.around(t11-t10,3))+' s')


# %%
model1.output


# %%
model1.summary()


# %%
##convLSTM model for forcing
t12 = time.time()
model2 = Sequential()
model2.add(
    ConvLSTM2D(filters=dynamic_nodes[0],  # in convLSTM, #filters defines the output space dimensions & the capacity of the network. Similar to #units in a LSTM
               data_format='channels_last',
               kernel_size=(3,3), 
               padding='same',
                input_shape = (None,nlat,nlon,nlev_forc),
               return_sequences=True,
    )
)
if batch_norm:
    model2.add(BatchNormalization())
if pooling:
    model2.add(MaxPooling3D(pool_size=(1, 2, 2), 
                    padding='same', 
                    data_format='channels_last'))
if len(dynamic_nodes) >1:
    for h in dynamic_nodes[1:]:
        model2.add(ConvLSTM2D(filters=h,
                        data_format='channels_last',
                       kernel_size=(3,3), 
                       padding='same',
                       return_sequences=True))
        if batch_norm:
            model2.add(BatchNormalization())
        if pooling:
            model2.add(MaxPooling3D(pool_size=(1, 2, 2), 
                    padding='same', 
                    data_format='channels_last'))
model2.add(TimeDistributed(Flatten()))
model2.add(Dense(static_nodes[-1]))
t13 = time.time()
print('time to prepare model for forcing data: '+str(np.around(t13-t12,3))+' s')


# %%
model2.output


# %%
model2.summary()


# %%
#combine models
combined = concatenate([model0.output, model1.output, model2.output])
#z = LSTM(nlat*nlon*10, activation="relu")(combined)
z = Dense(static_nodes[-1]*6,activation="relu")(combined)
z = Dense(nlat*nlon*target_number,activation="linear")(combined)
#z = Reshape(target_shape = (-1,nlat,nlon,target_number))(z)
final_model = Model(inputs=[model0.input, model1.input, model2.input], outputs=z)
final_model.compile(Adam(lr), loss='mse', metrics=metrics)
#z = Dense(48, activation="relu")(combined)
#z = Dense(nlat*nlon*target_number, activation="linear")(z)
#z = Reshape(target_shape=(n_timestep,nlat,nlon,target_number))(z)


# %%
final_model.output


# %%
final_model.summary()


# %%
# define the checkpoint
filepath = "saved_models/lstm_model_004.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# %%
tf.config.list_physical_devices('GPU')
new_static_feature_da.shape


# %%
#fit model
time_step = 30
subset_target = target_da[:,:time_step,:,:,:]
final_model.fit(
        x=[new_static_feature_da[:,:time_step,:,:,:],
           prev_press_feature_da[:,:time_step,:,:,:],
           forcing_feature_da[:,:time_step,:,:,:]], 
        y=np.reshape(subset_target,(subset_target.shape[0],subset_target.shape[1],-1)),
        epochs=5, batch_size=nlat*nlon,
        callbacks=callbacks_list)


# %%
#pred = final_model.predict(new_merg_feature_da[:,:10,:,:,:])
fig,axs = plt.subplots(4,5,figsize=(16,8))
for jj,ii in enumerate([0,50,51,52,53,103,153,203,253,303,353,403,453,454,455,456,457]):
    ax = axs[jj//5,jj%5]
    ax.imshow(new_merg_feature_da[0,5,ii,:,:])
    #ax.set_title(new_merg_names[ii])
#plt.imshow(new_merg_feature_da[0,5,0,:,:])
#plt.colorbar()


# %%
# define the checkpoint
filepath = "/glade/scratch/hoangtran/ssd_NN/NN/saved_models/lstm_model_003.h5"
new_model = keras.models.load_model(filepath, custom_objects={"tf": tf,
                                                                    "rmse":rmse,
                                                                     'log_loss': log_loss,
                                                                     "var_ratio":var_ratio,
                                                                     "var_loss":var_loss})
# fit the model
checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
subset_target = target_da[:,:time_step,:,:,:]
new_model.fit(
        x=[new_static_feature_da[:,:time_step,:,:,:],
           prev_press_feature_da[:,:time_step,:,:,:],
           forcing_feature_da[:,:time_step,:,:,:]], 
        y=np.reshape(subset_target,(subset_target.shape[0],subset_target.shape[1],-1)),
        epochs=200, batch_size=nlat*nlon,
        callbacks=callbacks_list)


# %%
pred = new_model.predict([new_static_feature_da[:,:100,:,:,:],
           prev_press_feature_da[:,:100,:,:,:],
           forcing_feature_da[:,:100,:,:,:]])


# %%
pred1 = np.reshape(pred,(pred.shape[0],pred.shape[1],nlat,nlon,target_number),'C')
fig,axs = plt.subplots(1,2,figsize=(17,8))
im0 = axs[0].imshow(pred1[0,3,:,:,49])
im1 = axs[1].imshow(target_da[0,3,:,:,49])
#np.unique(pred1[0,0,:,:,:])
axs[0].set_title('Prediction')
axs[1].set_title('Truth')
cb0 = fig.colorbar(im0, ax=axs[0], orientation='horizontal')
cb1 = fig.colorbar(im1, ax=axs[1], orientation='horizontal')


# %%
plt.imshow(pred[0,:2,1000:1020])
plt.colorbar()


# %%
subset_tar_flat = np.reshape(subset_target,(subset_target.shape[0],subset_target.shape[1],-1))
plt.imshow(subset_tar_flat[0,:2,100:120])
plt.colorbar()


# %%
keras.utils.vis_utils.plot_model(new_model,to_file='model2.png', show_shapes=True, show_layer_names=True)


# %%



