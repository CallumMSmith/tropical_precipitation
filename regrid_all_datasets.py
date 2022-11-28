#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to regrid the precipitation datasets from their original resolution to their analysis resolutions
From a max of 0.05degrees to 2.0 degrees, depending on the dataset's highest available res.
"""

import numpy as np
import xarray as xr
import xesmf as xe
import time


start = time.process_time()

#  read in dataset
file_path = 'file_path'
file = 'file'

print('reading in precip dataset...')
ds = xr.open_dataset(file_path +  file, chunks={"time": 30}).sel(
        datetime=('2003-01-01')).sel(lat=slice(30, -30)).sel(lon=slice(-130, 160))


# define the output grid as an xarray dataset
print('define grids to regrid to...')
ds_out_005 = xr.Dataset({'lat': (['lat'], np.arange(-89.95, 90.05, 0.05)),
                     'lon': (['lon'], np.arange(-180, 180, 0.05)),})
ds_out_01 = xr.Dataset({'lat': (['lat'], np.arange(-89.9, 90.1, 0.1)),
                     'lon': (['lon'], np.arange(-180, 180, 0.1)),})
ds_out_025 = xr.Dataset({'lat': (['lat'], np.arange(-89.75, 90.25, 0.25)),
                     'lon': (['lon'], np.arange(-180, 180, 0.25)),})
ds_out_05 = xr.Dataset({'lat': (['lat'], np.arange(-89.5, 90.5, 0.5)),
                     'lon': (['lon'], np.arange(-180, 180, 0.5)),})
ds_out_1 = xr.Dataset({'lat': (['lat'], np.arange(-89, 91, 1.0)),
                     'lon': (['lon'], np.arange(-180, 180, 1.0)),})
ds_out_2 = xr.Dataset({'lat': (['lat'], np.arange(-88, 92, 2.0)),
                     'lon': (['lon'], np.arange(-180, 180, 2.0)),})


# make the regridder using: xe.Regridder(grid_in, grid_out, method), where bilinear is mostly sufficient
print('make the regridders using XE...')
regridder_005 = xe.Regridder(ds, ds_out_005, 'bilinear')
regridder_01 = xe.Regridder(ds, ds_out_01, 'bilinear')
regridder_025 = xe.Regridder(ds, ds_out_025, 'bilinear')
regridder_05 = xe.Regridder(ds, ds_out_05, 'bilinear')
regridder_1 = xe.Regridder(ds, ds_out_1, 'bilinear')
regridder_2 = xe.Regridder(ds, ds_out_2, 'bilinear')


# open dataset to regrid
print('opening dataset to regrid...')
ds = xr.open_dataset(file_path + file, chunks={"time": 30}).sel(
        datetime=slice('2003-01-01', '2020-12-31')).sel(lat=slice(30, -30)).sel(lon=slice(-130, 160))



# xesmf regridding process, utilising dask delayed
print('regridding with dask delayed...')
dr_out_005 = regridder_005(ds)
dr_out_01 = regridder_01(ds)
dr_out_025 = regridder_025(ds)
dr_out_05  = regridder_05(ds)
dr_out_1   = regridder_1(ds)
dr_out_2   = regridder_2(ds)


# compute dask regridding
print('computing regridding using dask...')
dr_out_005['precip'] = dr_out_005['precip'].compute()
dr_out_01['precip'] = dr_out_01['precip'].compute()
dr_out_025['precip'] = dr_out_025['precip'].compute()
dr_out_05['precip'] = dr_out_05['precip'].compute()
dr_out_1['precip'] = dr_out_1['precip'].compute()
dr_out_2['precip'] = dr_out_2['precip'].compute()


# add units
print('adding unit attributes...')
dr_out_005['precip'] = dr_out_005.precip.assign_attrs(units='mm/month')
dr_out_01['precip'] = dr_out_01.precip.assign_attrs(units='mm/month')
dr_out_025['precip'] = dr_out_025.precip.assign_attrs(units='mm/month')
dr_out_05['precip'] = dr_out_05.precip.assign_attrs(units='mm/month')
dr_out_1['precip'] = dr_out_1.precip.assign_attrs(units='mm/month')
dr_out_2['precip'] = dr_out_2.precip.assign_attrs(units='mm/month')



# save as new netcdf
print('saving new datasets, 0.05 degree.........')
dr_out_005.to_netcdf('file.nc')

print('saving new datasets, 0.1 degree.........')
dr_out_01.to_netcdf('file.nc')

print('saving new datasets, 0.25 degree.........')
dr_out_025.to_netcdf('file.nc')

print('saving new datasets, 0.5 degree.........')
dr_out_05.to_netcdf('file.nc')

print('saving new datasets, 1.0 degree.........')
dr_out_1.to_netcdf('file.nc')

print('saving new datasets, 2.0 degree.........')
dr_out_2.to_netcdf('file.nc')

print('******************** finished regridding ************************* ')


print('time taken (s): ', time.process_time() - start)






"""
Script to regrid the land cover from original resolution to analysis resolutions
From a max of 0.05degrees to 2.0 degrees
"""


start = time.process_time()

#  read in dataset
file_path = 'file_path'
file = 'file'

print('reading in precip dataset...')
ds = xr.open_dataset(file_path +  file, chunks={"time": 30}).sel(
        datetime=('2003-01-01')).sel(lat=slice(30, -30)).sel(lon=slice(-130, 160))


# define the output grid as an xarray dataset
print('define grids to regrid to...')
ds_out_005 = xr.Dataset({'lat': (['lat'], np.arange(-89.95, 90.05, 0.05)),
                     'lon': (['lon'], np.arange(-180, 180, 0.05)),})
ds_out_01 = xr.Dataset({'lat': (['lat'], np.arange(-89.9, 90.1, 0.1)),
                     'lon': (['lon'], np.arange(-180, 180, 0.1)),})
ds_out_025 = xr.Dataset({'lat': (['lat'], np.arange(-89.75, 90.25, 0.25)),
                     'lon': (['lon'], np.arange(-180, 180, 0.25)),})
ds_out_05 = xr.Dataset({'lat': (['lat'], np.arange(-89.5, 90.5, 0.5)),
                     'lon': (['lon'], np.arange(-180, 180, 0.5)),})
ds_out_1 = xr.Dataset({'lat': (['lat'], np.arange(-89, 91, 1.0)),
                     'lon': (['lon'], np.arange(-180, 180, 1.0)),})
ds_out_2 = xr.Dataset({'lat': (['lat'], np.arange(-88, 92, 2.0)),
                     'lon': (['lon'], np.arange(-180, 180, 2.0)),})


# make the regridder using: xe.Regridder(grid_in, grid_out, method), where nearest_s2d is mostly sufficient
print('make the regridders using XE...')
regridder_005 = xe.Regridder(ds, ds_out_005, 'nearest_s2d')
regridder_01 = xe.Regridder(ds, ds_out_01, 'nearest_s2d')
regridder_025 = xe.Regridder(ds, ds_out_025, 'nearest_s2d')
regridder_05 = xe.Regridder(ds, ds_out_05, 'nearest_s2d')
regridder_1 = xe.Regridder(ds, ds_out_1, 'nearest_s2d')
regridder_2 = xe.Regridder(ds, ds_out_2, 'nearest_s2d')


# open dataset to regrid
print('opening dataset to regrid...')
ds = xr.open_dataset(file_path + file, chunks={"time": 30}).sel(
        datetime=slice('2003-01-01', '2020-12-31')).sel(lat=slice(30, -30)).sel(lon=slice(-130, 160))


# xesmf regridding process, utilising dask delayed
print('regridding with dask delayed...')
dr_out_005 = regridder_005(ds)
dr_out_01 = regridder_01(ds)
dr_out_025 = regridder_025(ds)
dr_out_05  = regridder_05(ds)
dr_out_1   = regridder_1(ds)
dr_out_2   = regridder_2(ds)


# compute dask regridding
print('computing regridding using dask...')
dr_out_005['land_cover'] = dr_out_005['land_cover'].compute()
dr_out_01['land_cover'] = dr_out_01['land_cover'].compute()
dr_out_025['land_cover'] = dr_out_025['land_cover'].compute()
dr_out_05['land_cover'] = dr_out_05['land_cover'].compute()
dr_out_1['land_cover'] = dr_out_1['land_cover'].compute()
dr_out_2['land_cover'] = dr_out_2['land_cover'].compute()




# save as new netcdf
print('saving new datasets, 0.05 degree.........')
dr_out_005.to_netcdf('file.nc')

print('saving new datasets, 0.1 degree.........')
dr_out_01.to_netcdf('file.nc')

print('saving new datasets, 0.25 degree.........')
dr_out_025.to_netcdf('file.nc')

print('saving new datasets, 0.5 degree.........')
dr_out_05.to_netcdf('file.nc')

print('saving new datasets, 1.0 degree.........')
dr_out_1.to_netcdf('file.nc')

print('saving new datasets, 2.0 degree.........')
dr_out_2.to_netcdf('file.nc')

print('******************** finished regridding ************************* ')


print('time taken (s): ', time.process_time() - start)






"""
Script to regrid the land cover from original resolution to analysis resolutions
From a max of 0.05degrees to 2.0 degrees
"""


start = time.process_time()

#  read in dataset
file_path = 'file_path'
file = 'file'

print('reading in precip dataset...')
ds = xr.open_dataset(file_path +  file, chunks={"time": 30}).sel(
        datetime=('2003-01-01')).sel(lat=slice(30, -30)).sel(lon=slice(-130, 160))


# define the output grid as an xarray dataset
print('define grids to regrid to...')
ds_out_005 = xr.Dataset({'lat': (['lat'], np.arange(-89.95, 90.05, 0.05)),
                     'lon': (['lon'], np.arange(-180, 180, 0.05)),})
ds_out_01 = xr.Dataset({'lat': (['lat'], np.arange(-89.9, 90.1, 0.1)),
                     'lon': (['lon'], np.arange(-180, 180, 0.1)),})
ds_out_025 = xr.Dataset({'lat': (['lat'], np.arange(-89.75, 90.25, 0.25)),
                     'lon': (['lon'], np.arange(-180, 180, 0.25)),})
ds_out_05 = xr.Dataset({'lat': (['lat'], np.arange(-89.5, 90.5, 0.5)),
                     'lon': (['lon'], np.arange(-180, 180, 0.5)),})
ds_out_1 = xr.Dataset({'lat': (['lat'], np.arange(-89, 91, 1.0)),
                     'lon': (['lon'], np.arange(-180, 180, 1.0)),})
ds_out_2 = xr.Dataset({'lat': (['lat'], np.arange(-88, 92, 2.0)),
                     'lon': (['lon'], np.arange(-180, 180, 2.0)),})


# make the regridder using: xe.Regridder(grid_in, grid_out, method), where bilinear is mostly sufficient
print('make the regridders using XE...')
regridder_005 = xe.Regridder(ds, ds_out_005, 'bilinear')
regridder_01 = xe.Regridder(ds, ds_out_01, 'bilinear')
regridder_025 = xe.Regridder(ds, ds_out_025, 'bilinear')
regridder_05 = xe.Regridder(ds, ds_out_05, 'bilinear')
regridder_1 = xe.Regridder(ds, ds_out_1, 'bilinear')
regridder_2 = xe.Regridder(ds, ds_out_2, 'bilinear')


# open dataset to regrid
print('opening dataset to regrid...')
ds = xr.open_dataset(file_path + file, chunks={"time": 30}).sel(
        datetime=slice('2003-01-01', '2020-12-31')).sel(lat=slice(30, -30)).sel(lon=slice(-130, 160))



# xesmf regridding process, utilising dask delayed
print('regridding with dask delayed...')
dr_out_005 = regridder_005(ds)
dr_out_01 = regridder_01(ds)
dr_out_025 = regridder_025(ds)
dr_out_05  = regridder_05(ds)
dr_out_1   = regridder_1(ds)
dr_out_2   = regridder_2(ds)


# compute dask regridding
print('computing regridding using dask...')
dr_out_005['forest_cover'] = dr_out_005['forest_cover'].compute()
dr_out_01['forest_cover'] = dr_out_01['forest_cover'].compute()
dr_out_025['forest_cover'] = dr_out_025['forest_cover'].compute()
dr_out_05['forest_cover'] = dr_out_05['forest_cover'].compute()
dr_out_1['forest_cover'] = dr_out_1['forest_cover'].compute()
dr_out_2['forest_cover'] = dr_out_2['forest_cover'].compute()




# save as new netcdf
print('saving new datasets, 0.05 degree.........')
dr_out_005.to_netcdf('file.nc')

print('saving new datasets, 0.1 degree.........')
dr_out_01.to_netcdf('file.nc')

print('saving new datasets, 0.25 degree.........')
dr_out_025.to_netcdf('file.nc')

print('saving new datasets, 0.5 degree.........')
dr_out_05.to_netcdf('file.nc')

print('saving new datasets, 1.0 degree.........')
dr_out_1.to_netcdf('file.nc')

print('saving new datasets, 2.0 degree.........')
dr_out_2.to_netcdf('file.nc')

print('******************** finished regridding ************************* ')


print('time taken (s): ', time.process_time() - start)
