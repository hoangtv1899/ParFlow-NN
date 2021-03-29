import os.path
import numpy as np
import xarray as xr
import time
from datetime import datetime, timedelta
from parflow_nn.parsePF import write_nc

def daily_average(input_dir, run_name, year, output_dir):
    hourly_var = ['forcings', 'flow', 'press', 'wtd', 'clm']
    for vari in hourly_var:
        nc_arr = xr.open_dataset(os.path.join(input_dir, run_name+'_'+str(year)+'_'+vari+'.nc'))
        daily_arr = []
        n_hour, nlev, ny, nx = nc_arr[vari].shape
        time_arrays = nc_arr.time.values.astype('datetime64[s]').tolist()
        hour_steps = np.arange(0, n_hour+1, 24)
        for i in range(len(hour_steps)-1):
            tmp_sel_hourly = nc_arr[vari][hour_steps[i]:hour_steps[i+1],:,:,:]
            tmp_daily = np.mean(tmp_sel_hourly, axis = 0)
            daily_arr.append(tmp_daily)
        daily_arr = np.stack(daily_arr)
        out_file = os.path.join(output_dir, run_name+'_'+str(year)+'_'+vari+'.nc')
        if os.path.isfile(out_file):
            os.remove(out_file)
        print(time_arrays[0])
        write_nc(out_file, nx, ny, nlev, nc_arr.lat, nc_arr.lon, range(nlev),np.array(time_arrays)[hour_steps[:-1]], {vari: daily_arr}, t_start0 = time_arrays[0].strftime('%Y-%m-%d'))
    