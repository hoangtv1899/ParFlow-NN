import sys
import os
import shutil
from datetime import timedelta
import numpy as np

from parflow_nn.parsePF import init_arrays, init_arrays_with_pfbs, write_nc
from parflow_nn import config


def generate_nc_files(run_dir, overwrite=False):
    print('Generating nc files ...')
    out_dir = os.path.join(run_dir, 'nc_files')

    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            print(f'Output Folder {out_dir} already exists. Exiting.')
            return out_dir
    os.makedirs(out_dir)

    run_name = os.path.basename(run_dir)
    metadata_file = os.path.join(run_dir, f'{run_name}.out.pfmetadata')
    nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs = init_arrays(metadata_file)

    new_forcing_arr = np.stack([p[-1, ...] for p in var_outs['forcings']])
    del var_outs['forcings']

    out_nc = os.path.join(out_dir, f'{run_name}_forcings.nc')
    write_nc(out_nc, nx, ny, 1, lat0, lon0, np.array([0]), time_arrays,
             {'forcings': new_forcing_arr.reshape(-1, 1, ny, nx)})

    out_nc = os.path.join(out_dir, f'{run_name}_static.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, [config.init.t0], var_outs)

    target_arrs = init_arrays_with_pfbs(metadata_file)

    out_nc = os.path.join(out_dir, f'{run_name}_press.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'press': target_arrs['pressure']})

    out_nc = os.path.join(out_dir, f'{run_name}_satur.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'satur': target_arrs['saturation']})

    out_nc = os.path.join(out_dir, f'{run_name}_clm.nc')
    write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'clm': target_arrs['clm_output']})

    out_nc = os.path.join(out_dir, f'{run_name}_prev_press.nc')
    write_nc(
        out_nc, nx, ny, nz, lat0, lon0, lev0,
        [x - timedelta(hours=1) for x in time_arrays],
        {'prev_press': [var_outs['prev_press']] + target_arrs['pressure'][:-1]}
    )

    return out_dir
