import sys
import os
import shutil
from datetime import datetime, timedelta
import numpy as np

from parflow_nn.parsePF import init_arrays, init_arrays_with_pfbs, write_nc
from parflow_nn import config

from pfspinup.common import calculate_water_table_depth, \
    calculate_evapotranspiration, calculate_overland_flow
from pfspinup.pfmetadata import PFMetadata
from parflowio.pyParflowio import PFData

def pfread(pfbfile):
    # print(pfbfile)
    """
    Read a pfb file and return data as an ndarray
    :param pfbfile: path to pfb file
    :return: An ndarray of ndim=3

    TODO: parflowio seems to read arrays such that the rows (i.e. axis=1) are reversed w.r.t what pfio gives us
    Hence the np.flip
    """
    pfb_data = PFData(pfbfile)
    pfb_data.loadHeader()
    pfb_data.loadData()
    arr = pfb_data.moveDataArray()
    pfb_data.close()
    assert arr.ndim == 3, 'Only 3D arrays are supported'
    return np.flip(arr, axis=1)


def generate_nc_files(run_dir, t0 = None, overwrite=False, is_clm=True,year=''):
    print('Generating nc files ...')
    out_dir = os.path.join(run_dir, 'nc_files')

    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        else:
            print(f'Output Folder {out_dir} already exists...')    

    run_name = os.path.basename(run_dir)
    metadata_file = os.path.join(run_dir, f'{run_name}.out.pfmetadata')
    if t0 is not None:
        nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs = init_arrays(metadata_file, t_start0 = t0)
    else:
        nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs = init_arrays(metadata_file)

    out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_forcings.nc')

    if is_clm:
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        if t0 is not None:
            write_nc(out_nc, nx, ny, 8, lat0, lon0, range(8), time_arrays, {'forcings': var_outs['forcings']}, t_start0 = t0.strftime('%Y-%m-%d'))
        else:
            write_nc(out_nc, nx, ny, 8, lat0, lon0, range(8), time_arrays, {'forcings': var_outs['forcings']})
    else:
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        new_forcing_arr = np.stack([p[-1, ...] for p in var_outs['forcings']])
        if t0 is not None:
            write_nc(out_nc, nx, ny, 1, lat0, lon0, np.array([0]), time_arrays,
                     {'forcings': new_forcing_arr.reshape(-1, 1, ny, nx)}, t_start0 = t0.strftime('%Y-%m-%d'))
        else:
            write_nc(out_nc, nx, ny, 1, lat0, lon0, np.array([0]), time_arrays,
                     {'forcings': new_forcing_arr.reshape(-1, 1, ny, nx)})

    del var_outs['forcings']
    del var_outs['prev_press']
    del var_outs['rel_perm_alpha']
    del var_outs['rel_perm_N']
    del var_outs['satur_alpha']
    del var_outs['satur_N']
    del var_outs['satur_sres']
    del var_outs['satur_ssat']
    del var_outs['tensor_x']
    del var_outs['tensor_y']
    del var_outs['tensor_z']

    out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_static.nc')
    if os.path.isfile(out_nc):
        os.remove(out_nc)
    if t0 is not None:
        write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, [config.init.t0], var_outs, t_start0 = t0.strftime('%Y-%m-%d'))
    else:
        write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, [config.init.t0], var_outs)

    target_arrs = init_arrays_with_pfbs(metadata_file)

    if is_clm:
        out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_press.nc')
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        if t0 is not None:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays + [time_arrays[0] + timedelta(hours = len(time_arrays))],
                     {'press': target_arrs['pressure']}, t_start0 = t0.strftime('%Y-%m-%d')) # add the last time step
        else:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays + [time_arrays[0] + timedelta(hours = len(time_arrays))],
                     {'press': target_arrs['pressure']})

        out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_satur.nc')
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        if t0 is not None:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays + [time_arrays[0] + timedelta(hours=len(time_arrays))],
                     {'satur': target_arrs['saturation']}, t_start0 = t0.strftime('%Y-%m-%d'))  # add the last time step
        else:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays + [time_arrays[0] + timedelta(hours=len(time_arrays))],
                     {'satur': target_arrs['saturation']})

        out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_clm.nc')
        clm_array = target_arrs['clm_output']
        nlev_clm = clm_array.shape[1]
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        if t0 is not None:
            write_nc(out_nc, nx, ny, nlev_clm, lat0, lon0, range(nlev_clm), [x + timedelta(hours = 1) for x in time_arrays],
                     {'clm': clm_array}, t_start0 = t0.strftime('%Y-%m-%d'))  # shift the time step by 1 hour
        else:
            write_nc(out_nc, nx, ny, nlev_clm, lat0, lon0, range(nlev_clm), [x + timedelta(hours = 1) for x in time_arrays],
                     {'clm': clm_array})
    else:
        out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_press.nc')
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        if t0 is not None:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'press': target_arrs['pressure']}, t_start0 = t0.strftime('%Y-%m-%d'))
        else:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'press': target_arrs['pressure']})

        out_nc = os.path.join(out_dir, f'{run_name}_'+year+'_satur.nc')
        if os.path.isfile(out_nc):
            os.remove(out_nc)
        if t0 is not None:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'satur': target_arrs['saturation']}, t_start0 = t0.strftime('%Y-%m-%d'))
        else:
            write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, {'satur': target_arrs['saturation']})



    return out_dir

def generate_nc_file_stream(RUN_DIR, RUN_NAME, t0 = None, overwrite = False, year=''):
    print('Generating nc files ...')
    out_dir = os.path.join(RUN_DIR, 'nc_files')

    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        else:
            print(f'Output Folder {out_dir} already exists...')
            # return out_dir
    
    ## Get Forcing and Static .nc using old method
    
    metadata_file = os.path.join(RUN_DIR, f'{RUN_NAME}.out.pfmetadata')
    if t0 is not None:
        nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs = init_arrays(metadata_file, t_start0 = t0)
    else:
        nx, ny, nz, dx, dy, dz, dz_scale, time_arrays, lat0, lon0, lev0, var_outs = init_arrays(metadata_file)

    # out_nc = os.path.join(out_dir, f'{RUN_NAME}_forcings.nc')

    # write_nc(out_nc, nx, ny, 8, lat0, lon0, range(8), time_arrays, {'forcings': var_outs['forcings']})
    
    # del var_outs['forcings']

    # out_nc = os.path.join(out_dir, f'{RUN_NAME}_static.nc')
    # write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, [config.init.t0], var_outs)

    ## Get Streamflow and WTD .nc files using PF_simulation
    metadata = PFMetadata(f'{RUN_DIR}/{RUN_NAME}.out.pfmetadata')
    
    # Resolution
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    # Thickness of each layer, bottom to top
    dz = metadata.dz()

    # Extent
    nx = metadata['ComputationalGrid.NX']
    ny = metadata['ComputationalGrid.NY']
    nz = metadata['ComputationalGrid.NZ']
    
    # Origin
    x0 = metadata['ComputationalGrid.Lower.X']
    y0 = metadata['ComputationalGrid.Lower.Y']
    
    # Latitude and Longitude
    lat0 = np.arange(y0, y0 + ny * dy, dy)
    lon0 = np.arange(x0, x0 + nx * dx, dx)

    # ------------------------------------------
    # Get numpy arrays from metadata
    # ------------------------------------------

    # ------------------------------------------
    # Time-invariant values
    # ------------------------------------------
    # porosity = metadata.input_data('porosity')
    mask = metadata.input_data('mask')
    # Note that only time-invariant ET flux values are supported for now
    
    slopex = metadata.slope_x()  # shape (ny, nx)
    slopey = metadata.slope_y()  # shape (ny, nx)
    mannings = metadata.get_single_domain_value('Mannings')  # scalar value
    
    # ------------------------------------------
    # Time-variant values
    # ------------------------------------------
    # Get as many pressure files as are available, while also getting their corresponding index IDs and timing info
    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    # We're typically interested in the first value of the returned 3-tuple.
    # Note that if we were interested in specific time steps, we can specify these as the `index_list` parameter.
    # examples:
    #   files, _, _ = metadata.output_files('pressure', index_list=range(0, 31, 10))
    #   files, _, _ = metadata.output_files('pressure', index_list=[10, 30])

    # By explicitly passing in the index_list that we obtained in the call below,
    # we insist that all saturation files corresponding to the pressure files be present.
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    # no. of time steps
    nt = len(index_list)

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    # Arrays for total values (across all layers), with time as the first axis
    wtd = np.zeros((nt, ny, nx))
    overland_flow = np.zeros((nt, ny, nx))

    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfread(pressure_file)
        saturation = pfread(saturation_file)

        wtd[i, ...] = calculate_water_table_depth(pressure, saturation, dz)

        overland_flow[i, ...] = calculate_overland_flow(mask, pressure, slopex, slopey, mannings, dx, dy, kinematic=False)
     
    out_nc = os.path.join(out_dir, f'{RUN_NAME}_'+year+'_flow.nc')
    if os.path.isfile(out_nc):
        os.remove(out_nc)
    if t0 is not None:
        write_nc(out_nc, nx, ny, 1, lat0, lon0, range(1), time_arrays, {'flow': overland_flow}, t_start0 = t0.strftime('%Y-%m-%d'))
    else:
        write_nc(out_nc, nx, ny, 1, lat0, lon0, range(1), time_arrays, {'flow': overland_flow})
    
    out_nc = os.path.join(out_dir, f'{RUN_NAME}_'+year+'_wtd.nc')
    if os.path.isfile(out_nc):
        os.remove(out_nc)
    if t0 is not None:
        write_nc(out_nc, nx, ny, 1, lat0, lon0, range(1), time_arrays, {'wtd': wtd}, t_start0 = t0.strftime('%Y-%m-%d'))
    else:
        write_nc(out_nc, nx, ny, 1, lat0, lon0, range(1), time_arrays, {'wtd': wtd})
    



