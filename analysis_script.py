#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis to the change in precipitation due to forest loss
Sensitivity studies included as variants:
    Number of years at start and end of analysis
    Temporal length of analysis
    Size of nearest neighbour grid
    Limited by similar climate
"""

# Read in packages
import numpy as np
import pandas as pd
import xarray as xr

# resolutions that the analysis is run at
resolutions = ['0.05', '0.1', '0.25', '0.5', '1.0', '2.0']
# temporal length that the analysis is run at, either long (2003-2020) or short (2003-2017)
lengths = ['long', 'short']
# size of nearest neighbour grid
grid_size = ['3x3', '5x5']
# number of average years at start and end of analysis period
num_avg = ['3yr', '5yr']
# whether the climate is constrained to being similar, or not
clim_sim = ['similar', 'not_similar']

for res in resolutions:
    for length in lengths:
        for grid in grid_size:
            for num in num_avg:
                for clim in clim_sim:

                    # describe the set up
                    print('res: ', res, '. Time series length: ', length, '. Grid size: ', grid, '. Average years: ', num, '. Climate: ', clim)

                    file_path = 'file_path'
                    ds = xr.open_dataset(file_path + 'precip_ds_' + res + 'deg_' + length + '.nc')

                    # for each variable in ds, calculate the n year averages for start and finish of time period
                    if num == '3yr':
                        ds_start = ds.sel(time=slice('2003-01-01', '2005-12-01')).mean(dim='time')
                        ds_end   = ds.sel(time=slice('2015-01-01', '2017-12-01')).mean(dim='time')
                    elif:
                        ds_start = ds.sel(time=slice('2003-01-01', '2007-12-01')).mean(dim='time')
                        ds_end   = ds.sel(time=slice('2013-01-01', '2017-12-01')).mean(dim='time')

                    ds = ds_end - ds_start

                    ds_list = []
                    ds_names = []

                    for i in ds.data_vars:
                        # print(ds[i].name)
                        ds_names.append(ds[i].name)
                        ds_list.append((ds[i]))

                    # read in tree cover change, calculated from hansen data at a range of resolutions
                    tree_path = ('/nfs/b0122/ee13c2s/precip_analysis/make_vars_pp/')
                    cc = xr.open_dataset(tree_path + 'treeCoverChange_tropics_' + res + '_2017_v1.9_bilinear.nc')
                    # limit to tropics
                    cc = cc.sel(latitude=slice(-30, 30))
                    cc = np.negative(cc)
                    lat, lon = cc.indexes.values()

                    # convert cover change to array
                    cc_arr = cc['tree']

                    # create inds where canopy cover change is > 0, this informs us where tree cover change has occurred
                    inds = np.where(cc_arr > 0)
                    inds = [inds]

                    # create canopy cover masked array
                    cc_arr = cc_arr.to_masked_array()

                    # make a df with the lats and lons prescribed, we fill this later with the output
                    df_presc = ds.to_dataframe().iloc[:, 0:0]
                    # make a df that contains the 'difference' col for each ds
                    df_diff = ds.to_dataframe().iloc[:, 0:0]
                    df_defP = pd.DataFrame()
                    df_contP = pd.DataFrame()
                    df_def_cont = pd.DataFrame()

                    # analysis loop
                    # Compare the P change value of a pixel to the surrounding pixels
                    print('analysing...')
                    counts = []
                    error_counter = 0
                    for ind in inds:
                        for n in range(len(ds_names)):
                            count = 0
                            ds = (ds_list[n]).to_masked_array()
                            name = ds_names[n]
                            def_delta_vals = []
                            def_delta_vals_all_time = []
                            def_cc_vals = []
                            control_delta_vals = []
                            control_cc_vals = []
                            cc_delta_vals = []
                            def_lats = []
                            def_lons = []

                            for coord1 in range(len(ind[0])):
                                yy = ind[0][coord1]
                                xx = ind[1][coord1]
                                delta_val = ds[yy, xx]
                                delta_val_all_time = ds_all_time[yy, xx]

                                if delta_val == delta_val:
                                    if grid == '3x3':
                                        yy_list = [yy+1, yy+1, yy+1, yy, yy, yy-1, yy-1, yy-1]
                                        xx_list = [xx-1, xx, xx+1, xx-1, xx+1, xx-1, xx, xx+1]
                                    elif:
                                        yy_list = [yy+2, yy+2, yy+2, yy+2, yy+2,
                                                    yy+1, yy+1, yy+1, yy+1, yy+1,
                                                    yy, yy, yy, yy,
                                                    yy-1, yy-1, yy-1, yy-1, yy-1,
                                                    yy-2, yy-2, yy-2, yy-2, yy-2]
                                        xx_list = [xx-2, xx-1, xx, xx+1, xx+2,
                                                    xx-2, xx-1, xx, xx+1, xx+2,
                                                    xx-2, xx-1, xx+1, xx+2,
                                                    xx-2, xx-1, xx, xx+1, xx+2,
                                                    xx-2, xx-1, xx, xx+1, xx+2]

                                    temp_list = []
                                    temp_control_cc_vals = []
                                    CC_difference_values = []
                                    def_count = 0
                                    for_count = 0

                                    for coord2 in range(len(xx_list)):
                                        yy1 = yy_list[coord2]
                                        xx1 = xx_list[coord2]

                                        # Check whether any surrounding pixels have a smaller canopyLoss value
                                        try:
                                            DEF_CanopyLoss = cc_arr[yy, xx]
                                            CONTROL_CanopyLoss = cc_arr[yy1, xx1]
                                            if CONTROL_CanopyLoss < DEF_CanopyLoss:
                                                control_delta_val = ds[yy1, xx1]
                                                if control_delta_val == control_delta_val:
                                                    temp_list.append(control_delta_val)
                                                    temp_control_cc_vals.append(CONTROL_CanopyLoss)
                                                    CC_difference_values.append(DEF_CanopyLoss-CONTROL_CanopyLoss)
                                            else:
                                                continue
                                        except IndexError:
                                            continue
                                            error_counter += 1

                                    if len(temp_list) > 0:
                                        # variable change value over deforested
                                        def_delta_vals.append(delta_val)

                                        def_delta_vals_all_time.append(delta_val_all_time)
                                        # canopy change value over deforested
                                        def_cc_vals.append(DEF_CanopyLoss)
                                        # variable change value over control
                                        # find where in list the difference in cc between def and control is greatest
                                        i = np.where(CC_difference_values==np.max(CC_difference_values))
                                        #control_delta_vals.append(temp_list[i[0]])
                                        control_delta_vals.append(np.array(temp_list)[i].mean())
                                        # canopy change value over control
                                        control_cc_vals.append(np.array(temp_control_cc_vals)[i].mean())
                                        # difference in canopy cover between deforested and control
                                        cc_delta_vals.append(np.array(CC_difference_values)[i].mean())
                                        # lats and lons
                                        def_lats.append(lat[yy])
                                        def_lons.append(lon[xx])
                                        count += 1


                            df = pd.DataFrame()
                            diff = pd.DataFrame()
                            dcc_df = pd.DataFrame()
                            cc_diff_df = pd.DataFrame()

                            dcc_df['def_cc_vals'] = def_cc_vals
                            cc_diff_df['cc_difference_vals'] = cc_delta_vals
                            df['lat'] = def_lats
                            df['lon'] = def_lons

                            # calculate the background climate and test to see whether pixel is similar, if not throw out.
                            if clim == 'similar':
                                # make a column with the percentage difference between control and def mean P
                                # remove row if diff greater than say 10%
                                df['meanP_percent_diff'] = ((df['def_meanP'] - df['control_meanP']) / ((df['def_meanP'] + df['control_meanP']) / 2)) * 100
                                # remove pixels with greater than xx% mean P difference
                                threshold = 10
                                df.drop(df[df.meanP_percent_diff > threshold].index, inplace=True)
                                df.drop(df[df.meanP_percent_diff < -(threshold)].index, inplace=True)

                            elif:
                                continue

                            df['control_delta_vals'] = control_delta_vals
                            df['control_cc_vals'] = control_cc_vals
                            df['def_delta_vals'] = def_delta_vals
                            df['def_cc_vals'] = def_cc_vals
                            df['difference'] = df['def_delta_vals'] - df['control_delta_vals']
                            df['cc_difference_vals'] = cc_delta_vals
                            # calculate the difference in P over the difference in canopy cover
                            df[name] = df['difference']/df['cc_difference_vals']
                            # drop rows where difference in canopy cover change is < 0.1%
                            df.drop(df[df.cc_difference_vals < 0.1].index, inplace=True)
                            # print(df)

                            # get the difference cols from df into 'diff'. hopefully this should be the constrained cols version
                            diff['lat'] = def_lats
                            diff['lon'] = def_lons
                            diff[name] = df['difference']

                            selected_cols = df[["lat","lon", 'control_delta_vals', 'def_delta_vals']]
                            df_def_cont_ind = selected_cols.copy()
                            df_def_cont_ind = (df_def_cont_ind.rename(columns={"control_delta_vals": name+'_contP', "def_delta_vals": name+'_defP'})).set_index(['lat', 'lon'])
                            df_def_cont[name+'_contP'] = df_def_cont_ind[name+'_contP']
                            df_def_cont[name+'_defP'] = df_def_cont_ind[name+'_defP']
                            # print(df)

                            # create new df with lat and lon as index and varibale named as dataset
                            ds = (df[['lat', 'lon', name]].set_index(['lat', 'lon']))
                            # print(ds)
                            # add each df to the master df containing the lats and lons
                            df_presc[name] = ds[name]
                            # print(df_presc)

                            ds_diff = diff.set_index(['lat', 'lon'])
                            df_diff[name] = ds_diff[name]

                    # Option to output as csvs
                    # convert 'diff' to csv
                    df_diff.to_csv(file_path + 'diff_'+res+'_'+num+'_'+grid+'_'+length+'.csv')

                    df_def_cont.to_csv(file_path + 'def_cont_'+res+'_'+num+'_'+grid+'_'+length+'.csv')

                    dcc_df.to_csv(file_path + 'dcc_'+res+'_'+num+'_'+grid+'_'+length+'.csv')
                    cc_diff_df.to_csv(file_path + 'CC_diff_'+res+'_'+num+'_'+grid+'_'+length+'.csv')

                    # convert df to ds
                    ds_all = df_presc.to_xarray()

                    df_dropna = df_presc.dropna()

                    # print(ds_all)
                    # save as netcdf
                    print('saving output...')
                    resub_file_path = 'file_path'
                    ds_all.to_netcdf(resub_file_path + 'delP_by_delCC_'+res+'_'+num+'_'+grid+'_'+length+'.nc')
