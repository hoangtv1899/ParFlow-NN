import sys
import os
from datetime import datetime, timedelta
import numpy as np
from .parsePF import init_arrays, init_arrays_with_press, write_nc


def generate_nc_files(run_dir, tcl_file):
    print('Generating nc files ...')
    out_dir = os.path.join(run_dir, 'nc_files')

    if os.path.exists(out_dir):
        print(f'Output Folder {out_dir} already exists. Exiting.')
        return out_dir
    os.makedirs(out_dir)

    run_name = os.path.basename(run_dir)
    nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs = init_arrays(run_dir, tcl_file)
    precip_arrs = var_outs['precip']
    init_press = var_outs['prev_press']
    del var_outs['precip']

    new_precip_arr = np.array([]).reshape(0, ny, nx)
    for p_arr in precip_arrs:
        new_precip_arr = np.vstack(
            [new_precip_arr, p_arr[-1, :, :][np.newaxis, ...]]
        )

    out_nc = os.path.join(out_dir, f'{run_name}_precip.nc')
    write_nc(out_nc, nx, ny, 1, lat0, lon0, np.array([0]), time_arrays,
             {'precip': new_precip_arr.reshape(-1, 1, ny, nx)}, islev=True)

    out_nc = os.path.join(out_dir, f'{run_name}_static.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, [datetime(1982, 10, 1, 6, 0)], var_outs, islev=True)

    target_arrs = init_arrays_with_press(run_dir, tcl_file)

    out_nc = os.path.join(out_dir, f'{run_name}_press.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'press': target_arrs['press']}, islev=True)

    out_nc = os.path.join(out_dir, f'{run_name}_satur.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'satur': target_arrs['satur']}, islev=True)

    out_nc = os.path.join(out_dir, f'{run_name}_prev_press.nc')
    write_nc(
        out_nc, nx, ny, nz, lat0, lon0, lev0,
        [x - timedelta(hours=1) for x in time_arrays],
        {'prev_press': [init_press] + target_arrs['press'][:-1]}, islev=True
    )

    return out_dir
