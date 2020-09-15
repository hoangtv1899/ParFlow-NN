import netCDF4 as nc4
from netCDF4 import Dataset
import sys
import os
import numpy as np
import pfio
from glob import glob
from datetime import datetime, timedelta
from parflowio.pyParflowio import PFData


def pfread(pfbfile):
    """
    Read a pfb file and return data as an ndarray
    :param pfbfile: path to pfb file
    :return: An ndarray of ndim=3

    Note: parflowio seems to read arrays such that the rows (i.e. axis=1)
    This may need investigation
    """
    pfb_data = PFData(pfbfile)
    pfb_data.loadHeader()
    pfb_data.loadData()
    return np.flip(pfb_data.getDataAsArray(), axis=1)


def parse_tcl(infile, output_dir):
    with open(infile,'r') as fi:
        content = fi.read()
    content = content.split('\n')
    dz_scales = None
    for line in content:
        if "GeomInput.Names" in line:
            geom_name = line.split()[-1]
            geom_name = geom_name.replace('"','')
        elif "ComputationalGrid.Lower.X" in line:
            x0 = float(line.split()[-1])
        elif "ComputationalGrid.Lower.Y" in line:
            y0 = float(line.split()[-1])
        elif "ComputationalGrid.Lower.Z" in line:
            z0 = float(line.split()[-1])
        elif "ComputationalGrid.DX" in line:
            dx = float(line.split()[-1])
        elif "ComputationalGrid.DY" in line:
            dy = float(line.split()[-1])
        elif "ComputationalGrid.DZ" in line:
            dz = float(line.split()[-1])
        elif "ComputationalGrid.NX" in line:
            nx = int(line.split()[-1])
        elif "ComputationalGrid.NY" in line:
            ny = int(line.split()[-1])
        elif "ComputationalGrid.NZ" in line:
            nz = int(line.split()[-1])
        elif "TopoSlopesX.FileName" in line:
            slope_x_file = line.split()[-1].replace('"','')
            slope_x_file = output_dir+'/'+slope_x_file
        elif "TopoSlopesY.FileName" in line:
            slope_y_file = line.split()[-1].replace('"','')
            slope_y_file = output_dir+'/'+slope_y_file
        elif "Cycle.rainrec.rain.Length" in line:
            rain_len = float(line.split()[-1])
        elif "Cycle.rainrec.rec.Length" in line:
            rec_len = float(line.split()[-1])
        elif "BCPressure.rain.Value" in line:
            rain_val = float(line.split()[-1])
        elif "ICPressure.Type" in line:
            icp_type = line.split()[-1]
        elif "ICPressure.RefPatch" in line:
            icp_ref_path = line.split()[-1]
        elif "ICPressure.Value" in line:
            icp_val = float(line.split()[-1])
        elif "ICPressure.FileName" in line:
            icp_file = line.split()[-1]
            if icp_file[0] == '$':
                for line1 in content:
                    if (icp_file[1:] in line1) and ('set' in line1) and ('pfset' not in line1):
                        line1 = line1.strip()
                        if '#' == line1[0]:
                            continue
                        icp_file1 = line1.split()[-1]
                        icp_file1 = icp_file1.replace('"','')
                        icp_file1 = icp_file1.replace('/','')
                icp_file = icp_file1
            icp_file = output_dir+'/'+icp_file
        elif "Solver.Nonlinear.VariableDz" in line:
            nonlinear_dz = line.split()[-1]
            if nonlinear_dz == 'True':
                for line1 in content:
                    if "dzScale.nzListNumber" in line1:
                        nz_list = int(line1.split()[-1])
                dz_scales = {}
                for line1 in content:
                    for celli in range(nz_list):
                        if "Cell."+str(celli)+".dzScale.Value" in line1:
                            dz_scales[celli] = float(line1.split()[-1])
    for line in content:
        if "GeomInput."+geom_name+".GeomNames" in line:
            list_indis = line.split()
            list_indis = [x.replace('"','') for x in list_indis if x not in ['pfset',
                                                                    'GeomInput.'+geom_name+'.GeomNames']]
            list_indis = list(filter(None,list_indis))
        elif "Geom."+geom_name+".FileName" in line:
            indi_file = line.split()[-1]
            indi_file = indi_file.replace('"','')
            indi_file = output_dir+'/'+indi_file
            #indi_arr = pfio.pfread(indi_file)
            indi_dict = {}
            for indii in list_indis:
                for line in content:
                    if 'GeomInput.'+indii+'.Value' in line:
                        tmp_value = int(line.split()[-1])
                        indi_dict[indii] = tmp_value
            list_indis.append('domain')
            perm_dict = {}
            poros_dict = {}
            rel_perm_alpha_dict = {}
            rel_perm_N_dict = {}
            satur_alpha_dict = {}
            satur_N_dict = {}
            satur_sres_dict = {}
            satur_ssat_dict = {}
            tensor_x_dict = {}
            tensor_y_dict = {}
            tensor_z_dict = {}
            spec_storage_dict = {}
            manning_dict = {}
            for indii in list_indis:
                for line in content:
                    if 'Geom.'+indii+'.Perm.Value' in line:
                        perm_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.Porosity.Value' in line:
                        poros_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.RelPerm.Alpha' in line:
                        rel_perm_alpha_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.RelPerm.N' in line:
                        rel_perm_N_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.Saturation.Alpha' in line:
                        satur_alpha_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.Saturation.N' in line:
                        satur_N_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.Saturation.SRes' in line:
                        satur_sres_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.Saturation.SSat' in line:
                        satur_ssat_dict[indii] = float(line.split()[-1])
                    elif 'Geom.'+indii+'.Perm.TensorValX' in line:
                        tensor_x_dict[indii] = float(line.split()[-1].replace('d0',''))
                    elif 'Geom.'+indii+'.Perm.TensorValY' in line:
                        tensor_y_dict[indii] = float(line.split()[-1].replace('d0',''))
                    elif 'Geom.'+indii+'.Perm.TensorValZ' in line:
                        tensor_z_dict[indii] = float(line.split()[-1].replace('d0',''))
                    elif 'Geom.'+indii+'.SpecificStorage.Value' in line:
                        spec_storage_dict[indii] = float(line.split()[-1].replace('d0',''))
                    elif 'Mannings.Geom.'+indii+'.Value' in line:
                        manning_dict[indii] = float(line.split()[-1].replace('d0',''))
            list_dicts = [indi_dict,perm_dict,poros_dict,rel_perm_alpha_dict,
                            rel_perm_N_dict,satur_alpha_dict,satur_N_dict,
                            satur_sres_dict,satur_ssat_dict,tensor_x_dict,
                            tensor_y_dict,tensor_z_dict,spec_storage_dict,manning_dict]
    if icp_type == 'HydroStaticPatch':
        retVal = x0,y0,z0,dx,dy,dz,dz_scales,nx,ny,nz,rain_len,rec_len,rain_val,indi_file,\
                icp_type,icp_ref_path,icp_val,\
                slope_x_file,slope_y_file,list_dicts
    elif icp_type == 'PFBFile':
        retVal = x0,y0,z0,dx,dy,dz,dz_scales,nx,ny,nz,rain_len,rec_len,rain_val,indi_file,\
                icp_type,icp_ref_path,icp_file,\
                slope_x_file,slope_y_file,list_dicts
    else:
        print("Error finding initial condition")
        retVal = None

    return retVal

def init_arrays(output_dir, tcl_file):
    x0,y0,z0,dx,dy,dz,dz_scale,nx,ny,nz,rain_len,rec_len,rain_val,indi_file,\
                    icp_type,icp_ref_path,icp_file,\
                    slope_x_file,slope_y_file,list_dicts  = parse_tcl(tcl_file,output_dir)
    
    indi_arr = pfio.pfread(indi_file)
    data = pfread(indi_file)
    assert np.allclose(data, indi_arr)

    indi_dict = list_dicts[0]
    list_array = []
    for dd in list_dicts[1:]:
        domain_val = dd['domain']
        tmp_arr = np.ones((nz,ny,nx))*domain_val
        for k in dd.keys():
            if k != 'domain':
                tmp_arr[indi_arr==indi_dict[k]]=dd[k]
        #print(np.unique(tmp_arr))
        list_array.append(tmp_arr)
    
    #slopes
    for slope_file in [slope_x_file,slope_y_file]:
        tmp_slope = pfio.pfread(slope_file)
        data = pfread(slope_file)
        assert np.allclose(data, tmp_slope)
        list_array.append(np.tile(data,(nz,1,1)))
    
    #initial pressure
    if icp_type == 'HydroStaticPatch':
        icp_arr = np.ones((nz,ny,nx))*-99
        sum_dz = 0
        if ('bottom' in icp_ref_path) or ('lower' in icp_ref_path):
             for zi in range(nz):
                if dz_scale == None: #uniform dz
                    icp_arr[zi,np.where(indi_arr[zi,:,:]>0)] = icp_file - dz*(2*zi+1)/2
                else: #non linear dz
                    dz_scale[-1] = 0
                    sum_dz += (dz_scale[zi]+dz_scale[zi-1])*0.5*dz
                    icp_arr[zi,np.where(indi_arr[zi,:,:]>0)] = icp_file - sum_dz
        else:
            for zi in range(nz):
                if dz_scale == None: #uniform dz
                    icp_arr[zi,np.where(indi_arr[zi,:,:]>0)] = -1*(icp_file - dz*(2*zi+1)/2)
                else: #non linear dz
                    dz_scale[-1] = 0
                    sum_dz += (dz_scale[zi]+dz_scale[zi-1])*0.5*dz
                    icp_arr[zi,np.where(indi_arr[zi,:,:]>0)] = -1*(icp_file - sum_dz)
    else:
        icp_arr = pfio.pfread(icp_file)
        data = pfread(icp_file)
        assert np.allclose(data, icp_arr)
    
    unit_rain_rec_len = [1]*int(rain_len) + [0]*int(rec_len)
    t_start0 = datetime(1982,10,1,6) #hard-coded
    lat0 = np.arange(y0,y0+ny*dy,dy)
    lon0 = np.arange(x0,x0+nx*dx,dx)
    if (dz_scale == None) or (len(dz_scale) == 1):
        lev0 = np.arange(z0,z0+nz*dz,dz)
    else:
        dz_scale.pop(-1,None)
        lev0 = [0]
        sum_depth = 0
        for levi in sorted(dz_scale.keys())[1:]:
            sum_depth += dz_scale[levi-1]*dz
            lev0.append(sum_depth)
        lev0 = np.array(lev0)
    #get precip value
    var_forc_arrays =[]
    time_arrays = []
    output_files = sorted(glob(output_dir+'/*.out.press.*.pfb'))
    for cci,filei in enumerate(output_files):
        deltai = int(os.path.basename(filei).split('.')[-2])
        if unit_rain_rec_len[deltai%len(unit_rain_rec_len)] ==1:
            tmp_arr_forc = np.zeros((nz,ny,nx))
            tmp_arr_forc[-1,:,:] = np.ones((ny,nx))*rain_val
        else:
            tmp_arr_forc = np.zeros((nz,ny,nx))
        var_forc_arrays.append(tmp_arr_forc)
        time_arrays.append(t_start0+timedelta(hours=deltai))
    var_outs = {'precip':var_forc_arrays, 'perm':list_array[0], 'poros':list_array[1], 
                'rel_perm_alpha':list_array[2], 'rel_perm_N':list_array[3],
                'satur_alpha':list_array[4], 'satur_N':list_array[5], 
                'satur_sres':list_array[6],'satur_ssat':list_array[7],
                'tensor_x':list_array[8],'tensor_y':list_array[9], 'tensor_z':list_array[10],
                'spec_storage':list_array[11],'mannings':list_array[12],
                'slope_x':list_array[13],'slope_y':list_array[14],'prev_press':icp_arr}
    return nx,ny,nz,dx,dy,dz,dz_scale,time_arrays,lat0,lon0,lev0,var_outs

def init_arrays_with_press(output_dir, tcl_file):
    x0,y0,z0,dx,dy,dz,dz_scale,nx,ny,nz,rain_len,rec_len,rain_val,indi_file,\
                    icp_type,icp_ref_path,icp_file,\
                    slope_x_file,slope_y_file,list_dicts  = parse_tcl(tcl_file,output_dir)
    
    var_out = ['press','satur']
    var_out_arrays = []
    for vari in var_out:
        var_arrays = []
        output_files = sorted(glob(output_dir+'/*.out.'+vari+'.*.pfb'))
        for cci,filei in enumerate(output_files):
            var_array = pfio.pfread(filei)
            data = pfread(filei)
            assert np.allclose(data, var_array)
            var_arrays.append(var_array)
        var_out_arrays.append(var_arrays)
    
    var_outs = {'press':var_out_arrays[0], 'satur':var_out_arrays[1]}
    return var_outs

def write_nc_series(out_nc,nx,ny,nz,list_t,lat0,lon0,lev0,time_arrays,var_outs):
    for ii,dt in enumerate(list_t):
        input_name = out_nc.replace('.nc','')+'_'+dt.strftime('%Y%m%d')+'.nc'
        if os.path.isfile(input_name):
            os.remove(input_name)
        input_f = nc4.Dataset(input_name,'w', format='NETCDF4')
        #create dimensions
        lat_dim = input_f.createDimension('lat',ny)
        lon_dim = input_f.createDimension('lon',nx)
        time_dim = input_f.createDimension('time',len(time_arrays[ii]))
        lev_dim = input_f.createDimension('lev',nz)
        #create dimension variables
        lat = input_f.createVariable('lat', np.float64, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lon = input_f.createVariable('lon', np.float64, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lev = input_f.createVariable('lev', np.float64, ('lev',))
        lev.units = 'depth from the surface (m)'
        lev.long_name = 'level'
        time = input_f.createVariable('time', np.float64, ('time',))
        time.units = 'hours since '+dt.strftime('%Y-%m-%d')
        time.long_name = 'time'
        
        #writing data
        lat[:] = lat0
        lon[:] = lon0
        lev[:] = lev0
        time[:] = time_arrays[ii]
        
        for keyi in var_outs:
            #create variables
            tmpi = input_f.createVariable(keyi,np.float64,('time','lev','lat','lon')) # note: unlimited dimension is leftmost
            if keyi in ['precip','perm']:
                tmpi.units = 'm/h'
            elif keyi in ['press','prev_press']:
                tmpi.units = 'm'
            #writing data
            tmpi[:] = var_outs[keyi]

        input_f.close()

def write_nc(out_nc,nx,ny,nz,lat0,lon0,lev0,time_arrays,var_outs,islev=False):
    input_name = out_nc
    if os.path.isfile(input_name):
        os.remove(input_name)
    input_f = nc4.Dataset(input_name,'w', format='NETCDF4')
    #create dimensions
    lat_dim = input_f.createDimension('lat',ny)
    lon_dim = input_f.createDimension('lon',nx)
    time_dim = input_f.createDimension('time',len(time_arrays))
    lev_dim = input_f.createDimension('lev',nz)
    #create dimension variables
    lat = input_f.createVariable('lat', np.float64, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = input_f.createVariable('lon', np.float64, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    lev = input_f.createVariable('lev', np.float64, ('lev',))
    lev.units = 'depth from the surface (m)'
    lev.long_name = 'level'
    time = input_f.createVariable('time', np.float64, ('time',))
    time.units = 'hours since 1982-10-1'
    time.long_name = 'time'
    
    #writing data
    lat[:] = lat0
    lon[:] = lon0
    lev[:] = lev0
    time[:] = [(x-time_arrays[0]).total_seconds()/3600. for x in time_arrays]
    
    for keyi in var_outs:
        #print(islev)
        #create variables
        if islev:
            tmpi = input_f.createVariable(keyi,np.float64,('time','lev','lat','lon')) # note: unlimited dimension is leftmost
        else:
            tmpi = input_f.createVariable(keyi,np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
        if keyi in ['precip','perm']:
            tmpi.units = 'm/h'
        elif keyi in ['press','prev_press']:
            tmpi.units = 'm'
        #writing data
        tmpi[:] = var_outs[keyi]
    input_f.close()