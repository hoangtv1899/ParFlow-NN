# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from parsePF import *
import time
import matplotlib.pyplot as plt
import os
os.getcwd()


# %%
#get initial information
in_dir = '../washita/tcl_scripts/Outputs1'
in_file = '../washita/tcl_scripts/LW_Test.tcl'
nx,ny,nz,dx,dy,dz,dz_scale,time_arrays,lat0,lon0,lev0,var_outs = init_arrays(in_dir, in_file)
precip_arrs = var_outs['precip']
init_press = var_outs['prev_press']
del var_outs['precip']
#del var_outs['prev_press']


# %%
plt.imshow(var_outs['poros'][-2,:,:])
plt.colorbar()


# %%
new_precip_arr = np.array([]).reshape(0,ny,nx)
for p_arr in precip_arrs:
    new_precip_arr = np.vstack([new_precip_arr,p_arr[-1,:,:][np.newaxis,...]])
plt.plot(np.mean(new_precip_arr,axis=(1,2)))


# %%
var_outs['perm'].shape


# %%
#check directory structure
os.getcwd()
outdirs = os.listdir('..')

if 'ncfile' not in outdirs :
    os.makedirs("../nc_file")


# %%
#write precip nc file

out_nc = '../nc_file/LW_precip.nc'
write_nc(out_nc,nx,ny,1,lat0,lon0,np.array([0]),
         time_arrays,{'precip':new_precip_arr.reshape(-1,1,ny,nx)},
        islev=True)


# %%
#write static nc file
out_nc = '../nc_file/LW_static.nc'
write_nc(out_nc,nx,ny,nz,lat0,lon0,lev0,[datetime(1982, 10, 1, 6, 0)],var_outs,islev=True)


# %%
t1 = time.time()
target_arrs = init_arrays_with_press(in_dir, in_file)
t2 = time.time()
print('load target files '+str(t2-t1))


# %%
#write press nc file
out_nc = '../nc_file/LW_press.nc'
write_nc(out_nc,nx,ny,nz,lat0,lon0,lev0,time_arrays,{'press':target_arrs['press']},islev=True)


# %%
#write satur nc file
out_nc = '../nc_file/LW_satur.nc'
write_nc(out_nc,nx,ny,nz,lat0,lon0,lev0,time_arrays,{'satur':target_arrs['satur']},islev=True)


# %%
#write previous press nc file. Only for trainning
out_nc = '../nc_file/LW_prev_press.nc'
write_nc(out_nc,nx,ny,nz,lat0,lon0,lev0,[x-timedelta(hours=1) for x in time_arrays],
        {'prev_press':[init_press]+target_arrs['press'][:-1]},islev=True)

