import sys
import os
from datetime import datetime, timedelta
from parsePF import init_arrays, init_arrays_with_press, write_nc
import numpy as np


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage python write_nc.py <run_dir> <tcl_file>')
        sys.exit(0)

    IN_DIR, IN_FILE = sys.argv[1:]
    OUT_DIR = os.path.join(IN_DIR, 'nc_files')

    if os.path.exists(OUT_DIR):
        print(f'Output Folder {OUT_DIR} already exists. Exiting.')
        sys.exit(1)
    os.makedirs(OUT_DIR)

    run_name = os.path.basename(IN_DIR)
    nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs \
        = init_arrays(IN_DIR, IN_FILE)
    precip_arrs = var_outs['precip']
    init_press = var_outs['prev_press']
    del var_outs['precip']

    new_precip_arr = np.array([]).reshape(0, ny, nx)
    for p_arr in precip_arrs:
        new_precip_arr = np.vstack(
            [new_precip_arr,p_arr[-1, :, :][np.newaxis, ...]]
        )

    out_nc = os.path.join(OUT_DIR, f'{run_name}_precip.nc')
    write_nc(out_nc, nx, ny, 1, lat0, lon0, np.array([0]), time_arrays,
             {'precip': new_precip_arr.reshape(-1, 1, ny, nx)}, islev=True)

    out_nc = os.path.join(OUT_DIR, f'{run_name}_static.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0,
             [datetime(1982, 10, 1, 6, 0)], var_outs, islev=True)

    target_arrs = init_arrays_with_press(IN_DIR, IN_FILE)

    out_nc = os.path.join(OUT_DIR, f'{run_name}_press.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays,
             {'press': target_arrs['press']}, islev=True)

    out_nc = os.path.join(OUT_DIR, f'{run_name}_satur.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays,
             {'satur': target_arrs['satur']}, islev=True)

    out_nc = os.path.join(OUT_DIR, f'{run_name}_prev_press.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0,
             [x-timedelta(hours=1) for x in time_arrays],
             {'prev_press': [init_press]+target_arrs['press'][:-1]}, islev=True
    )
