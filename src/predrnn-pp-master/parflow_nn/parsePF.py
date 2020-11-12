import os
from types import SimpleNamespace
from datetime import timedelta
from glob import glob
import numpy as np
import netCDF4 as nc4
from parflowio.pyParflowio import PFData

from parflow_nn.pfmetadata import PFMetadata
from parflow_nn import config


def pfread(pfbfile):
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
    arr = pfb_data.getDataAsArray()
    pfb_data.close()
    assert arr.ndim == 3, 'Only 3D arrays are supported'
    return np.flip(arr, axis=1)


def parse_metadata(metadata_file):

    p = PFMetadata(metadata_file)

    #add output data files
    outputs = p.get_outputs()
    output_data = {}
    for key in outputs:
        output_data[key], time_range = p.get_key_files(key)

    ic_pressure = p.icpressure_value() if p['ICPressure.Type'] == 'HydroStaticPatch' else p.icpressure_filename()

    return SimpleNamespace(
        x0=p['ComputationalGrid.Lower.X'],
        y0=p['ComputationalGrid.Lower.Y'],
        z0=p['ComputationalGrid.Lower.Z'],
        dx=p['ComputationalGrid.DX'],
        dy=p['ComputationalGrid.DY'],
        dz=p['ComputationalGrid.DZ'],
        dz_scale=p.dz_scale(),
        nx=p['ComputationalGrid.NX'],
        ny=p['ComputationalGrid.NY'],
        nz=p['ComputationalGrid.NZ'],
        indi_file=p.indicator_file(),
        icp_type=p['ICPressure.Type'],
        icp_ref_path=p.icpressure_refpatch(),
        icp_file=ic_pressure,
        slope_x_file=p.get_absolute_path(p['TopoSlopesX.FileName']),
        slope_y_file=p.get_absolute_path(p['TopoSlopesY.FileName']),
        indicator_geom_indices=p.indicator_geom_values(),

        geom_data=dict(
            perm=p.permeability(),
            poros=p.porosity(),
            rel_perm_alpha=p.phase_geom_values('RelPerm', 'Alpha'),
            rel_perm_N=p.phase_geom_values('RelPerm', 'N'),
            satur_alpha=p.phase_geom_values('Saturation', 'Alpha'),
            satur_N=p.phase_geom_values('Saturation', 'N'),
            satur_sres=p.phase_geom_values('Saturation', 'SRes'),
            satur_ssat=p.phase_geom_values('Saturation', 'SSat'),
            tensor_x=p.geom_tensors('Perm', 'X'),
            tensor_y=p.geom_tensors('Perm', 'Y'),
            tensor_z=p.geom_tensors('Perm', 'Z'),
            spec_storage=p.get_values_by_geom('SpecificStorage'),

            # TODO: Why do we have 'Geom.domain.SpecificStorage.Value'
            # but 'Mannings.Geom.domain.Value' (i.e. reversed)?
            mannings=p.get_values_by_geom('Mannings', is_reversed=True)
        )

    ), output_data, time_range, p


def init_arrays(metadata_file):
    m, output_data, time_range, p = parse_metadata(metadata_file)

    indi_arr = pfread(m.indi_file)

    indi_dict = m.indicator_geom_indices
    geom_data = {}
    for key, value in m.geom_data.items():
        domain_val = value['domain']
        tmp_arr = np.ones((m.nz, m.ny, m.nx)) * domain_val
        for k in value.keys():
            if k != 'domain':
                tmp_arr[indi_arr == indi_dict[k]] = value[k]
        geom_data[key] = tmp_arr
    
    # slopes
    geom_data['slope_x'] = np.tile(pfread(m.slope_x_file), (m.nz, 1, 1))
    geom_data['slope_y'] = np.tile(pfread(m.slope_y_file), (m.nz, 1, 1))

    # initial pressure
    dz_scale = m.dz_scale
    if m.icp_type == 'HydroStaticPatch':
        icp_arr = np.ones((m.nz, m.ny, m.nx)) * -99
        sum_dz = 0
        if ('bottom' in m.icp_ref_path) or ('lower' in m.icp_ref_path):
            for zi in range(m.nz):
                if m.dz_scale is None:  # uniform dz
                    icp_arr[zi, np.where(indi_arr[zi, :, :] > 0)] = m.icp_file - m.dz * (2 * zi + 1) / 2
                else:  # non linear dz
                    dz_scale[-1] = 0
                    sum_dz += (dz_scale[zi]+dz_scale[zi-1]) * 0.5 * m.dz
                    icp_arr[zi, np.where(indi_arr[zi, :, :] > 0)] = m.icp_file - sum_dz
        else:
            for zi in range(m.nz):
                if dz_scale is None:  # uniform dz
                    icp_arr[zi, np.where(indi_arr[zi, :, :] > 0)] = -1 * (m.icp_file - m.dz * (2 * zi + 1) / 2)
                else:  # non linear dz
                    dz_scale[-1] = 0
                    sum_dz += (dz_scale[zi] + dz_scale[zi-1]) * 0.5 * m.dz
                    icp_arr[zi, np.where(indi_arr[zi, :, :] > 0)] = -1 * (m.icp_file - sum_dz)
    else:
        icp_arr = pfread(m.icp_file)

    t_start0 = config.init.t0
    lat0 = np.arange(m.y0, m.y0 + m.ny * m.dy, m.dy)
    lon0 = np.arange(m.x0, m.x0 + m.nx * m.dx, m.dx)
    if (dz_scale is None) or (len(dz_scale) == 1):
        lev0 = np.arange(m.z0, m.z0 + m.nz * m.dz, m.dz)
    else:
        dz_scale.pop(-1, None)
        lev0 = [0]
        sum_depth = 0
        for levi in sorted(dz_scale.keys())[1:]:
            sum_depth += dz_scale[levi - 1] * m.dz
            lev0.append(sum_depth)
        lev0 = np.array(lev0)

    NLDAS_vars = ['DSWR', 'DLWR', 'APCP', 'Temp',
                  'UGRD', 'VGRD', 'Press', 'SPFH']

    # get clm_forcing
    var_forc_arrays = []
    time_arrays = []
    if p.is_clm():
        istart, ts_in_file, clm_name, clm_path, forcing_3d = p.get_clm_info()
        if forcing_3d == True:
            starti = ts_in_file * round(istart / ts_in_file) + 1
            endi = ts_in_file * (round(time_range[-1] / ts_in_file) - 1) + 1
            for NLDAS_vari in NLDAS_vars:
                timei_arr = []
                for timei in range(starti, endi + 1, 24):
                    timei_arr.append(
                        pfread('%s/%s.%s.%06d_to_%06d.pfb' % (clm_path, clm_name, NLDAS_vari, timei, timei + 23)))
                timei_arr = np.vstack(timei_arr)
                var_forc_arrays.append(timei_arr)
            var_forc_arrays = np.stack(var_forc_arrays)
            var_forc_arrays = np.swapaxes(var_forc_arrays, 0, 1)
            assert forcing_3d == True, 'Only 3D forcing are supported'
            time_arrays = [t_start0 + timedelta(hours = x) for x in range(var_forc_arrays.shape[0])]
    else:
        rain_len = p['Cycle.rainrec.rain.Length'],
        rec_len = p['Cycle.rainrec.rec.Length'],
        rain_val = p['Patch.z-upper.BCPressure.rain.Value']
        unit_rain_rec_len = [1] * int(m.rain_len) + [0] * int(m.rec_len)
        # get precip value
        output_files = output_data['pressure']
        for cci, filei in enumerate(output_files):
            deltai = int(os.path.basename(filei).split('.')[-2])
            if unit_rain_rec_len[deltai % len(unit_rain_rec_len)] == 1:
                tmp_arr_forc = np.zeros((m.nz, m.ny, m.nx))
                tmp_arr_forc[-1, :, :] = np.ones((m.ny, m.nx)) * m.rain_val
            else:
                tmp_arr_forc = np.zeros((m.nz, m.ny, m.nx))
            var_forc_arrays.append(tmp_arr_forc)
            time_arrays.append(t_start0 + timedelta(hours=deltai))

    var_outs = {'forcings': var_forc_arrays, 'prev_press': icp_arr}
    var_outs.update(geom_data)

    return m.nx, m.ny, m.nz, m.dx, m.dy, m.dz, m.dz_scale, time_arrays, lat0, lon0, lev0, var_outs


def init_arrays_with_pfbs(metadata_file):
    _, output_data, _, _ = parse_metadata(metadata_file)
    #return {k: [pfread(file) for file in sorted(glob(pfb_dir + '/*.out.press.*.pfb'))] for k in keys}
    return {key: np.stack([pfread(file) for file in output_data[key]]) for key in output_data}


def write_nc(out_nc, nx, ny, nz, lat0, lon0, lev0, time_arrays, var_outs, islev=True):

    with nc4.Dataset(out_nc, 'w', format=config.netcdf.format) as f:

        # create dimensions
        f.createDimension('lat', ny)
        f.createDimension('lon', nx)
        f.createDimension('time', len(time_arrays))
        f.createDimension('lev', nz)

        # create dimension variables
        lat = f.createVariable('lat', np.float64, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lon = f.createVariable('lon', np.float64, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lev = f.createVariable('lev', np.float64, ('lev',))
        #lev.units = 'depth from the surface (m)'
        lev.long_name = 'level'
        time = f.createVariable('time', np.float64, ('time',))
        time.units = 'hours since ' + config.init.t0.strftime('%Y-%m-%d')
        time.long_name = 'time'

        # writing data
        lat[:] = lat0
        lon[:] = lon0
        lev[:] = lev0
        time[:] = [(x-time_arrays[0]).total_seconds()/3600. for x in time_arrays]

        units = {
            'precip': 'm/h',
            'perm': 'm/h',
            'press': 'm',
            'prev_press': 'm'
        }

        for k, v in var_outs.items():
            if islev:
                # note: unlimited dimension is leftmost
                var = f.createVariable(k, np.float64, ('time', 'lev', 'lat', 'lon'))
            else:
                # note: unlimited dimension is leftmost
                var = f.createVariable(k, np.float64, ('time', 'lat', 'lon'))

            if k in units:
                var.units = units[k]
            var[:] = v
