#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot P onto maps

Maps showing changes in P for satellite (col1), station (col2) and reanalysis (col3)
products over 2003-2007 (row1), 2013-2017 (row2) and the difference (row3)
For 2.0 degrees
"""

import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.util import add_cyclic_point
import string

shp_path = 'file_path'
# read in shapefiles
amazon_shp = list(shpreader.Reader(shp_path + 'Amazon_biome/amazonia.shp').geometries())
congo_shp = list(shpreader.Reader(shp_path + 'congo_shp_sept21.shp').geometries())
sea_shp = list(shpreader.Reader(shp_path + 'sea_with_png.shp').geometries())

data_path = 'file_path'

# colourbar min and max
cbmax = 300
cbmin = 50

satellite_ds = ['pr_chirps', 'pr_cmorph', 'pr_gpcp', 'pr_gpm', 'pr_per_ccs', 'pr_per_ccscdr', 'pr_per_cdr', 'pr_per_now', 'pr_per_per', 'pr_trmm']
station_ds = ['pr_cpc', 'pr_cru', 'pr_gpcc', 'pr_udel']
reanalysis_ds = ['pr_era', 'pr_jra', 'pr_merra', 'pr_noaa']

ds_list = []
# read in datasets
ds = xr.open_dataset(data_path + 'precip_ds_2.0deg_03_17.nc').sel(lon=slice(-100,160)).sel(lat=slice(-30,30))

# satellite ds medians...
sat_start_ds = ds[satellite_ds].sel(time=slice('2003-01-01', '2007-12-31')).median(dim='time')
median = sat_start_ds.to_array(dim='new').median('new')
sat_start_ds = sat_start_ds.assign(start=median)
sat_start_ds = sat_start_ds['start']
mean_val_sat_start = str(round(np.nanmean(sat_start_ds), 2))

sat_end_ds = ds[satellite_ds].sel(time=slice('2013-01-01', '2017-12-31')).median(dim='time')
median = sat_end_ds.to_array(dim='new').median('new')
sat_end_ds = sat_end_ds.assign(end=median)
sat_end_ds = sat_end_ds['end']
mean_val_sat_end = str(round(np.nanmean(sat_end_ds), 2))

sat_diff_ds = sat_end_ds - sat_start_ds
mean_val_sat_diff = str(round(np.nanmean(sat_diff_ds), 2))

# station ds medians...
stat_start_ds = ds[station_ds].sel(time=slice('2003-01-01', '2007-12-31')).median(dim='time')
median = stat_start_ds.to_array(dim='new').median('new')
stat_start_ds = stat_start_ds.assign(start=median)
stat_start_ds = stat_start_ds['start']
mean_val_stat_start = str(round(np.nanmean(stat_start_ds), 2))

stat_end_ds = ds[station_ds].sel(time=slice('2013-01-01', '2017-12-31')).median(dim='time')
median = stat_end_ds.to_array(dim='new').median('new')
stat_end_ds = stat_end_ds.assign(end=median)
stat_end_ds = stat_end_ds['end']
mean_val_stat_end = str(round(np.nanmean(stat_end_ds), 2))

stat_diff_ds = stat_end_ds - stat_start_ds
mean_val_stat_diff = str(round(np.nanmean(stat_diff_ds), 2))

# reanalysis ds medians...
rean_start_ds = ds[reanalysis_ds].sel(time=slice('2003-01-01', '2007-12-31')).median(dim='time')
median = rean_start_ds.to_array(dim='new').median('new')
rean_start_ds = rean_start_ds.assign(start=median)
rean_start_ds = rean_start_ds['start']
mean_val_rean_start = str(round(np.nanmean(rean_start_ds), 2))

rean_end_ds = ds[reanalysis_ds].sel(time=slice('2013-01-01', '2017-12-31')).median(dim='time')
median = rean_end_ds.to_array(dim='new').median('new')
rean_end_ds = rean_end_ds.assign(end=median)
rean_end_ds = rean_end_ds['end']
mean_val_rean_end = str(round(np.nanmean(rean_end_ds), 2))

rean_diff_ds = rean_end_ds - rean_start_ds
mean_val_rean_diff = str(round(np.nanmean(rean_diff_ds), 2))

# add the ds to a list
ds_list = [sat_start_ds, sat_end_ds, stat_start_ds, stat_end_ds, rean_start_ds, rean_end_ds, sat_diff_ds, stat_diff_ds, rean_diff_ds]


# make the plot
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(14,6), constrained_layout=True)
axs=axs.flatten()

for i,data in enumerate(ds_list):
    if i <6:
        cs_1 = axs[i].pcolormesh(data['lon'], data['lat'], data, vmin=50, vmax=250, cmap='Blues', alpha=1, shading='auto')
    else:
        cs_2 = axs[i].pcolormesh(data['lon'], data['lat'], data, vmin=-20, vmax=20, cmap='bwr_r', alpha=1, shading='auto')

    # calc mean and add to title of subplot
    mean_val = str(round(np.nanmean(data), 2))
    axs[i].text(0, -42, 'mean = ' + mean_val)

    # add shapefiles outlines
    axs[i].add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.7)
    axs[i].add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.7)
    axs[i].add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.7)

    # add text titles
    if i == 0:
        axs[i].set_title('Satellite', fontsize=14)
        axs[i].text(-115, -28, '2003-2007', rotation=90, size=12)
    if i == 1:
        axs[i].set_title('Station', fontsize=14)
    if i == 2:
        axs[i].set_title('Reanalysis', fontsize=14)
    if i == 3:
        axs[i].text(-115, -28, '2013-2017', rotation=90, size=12)
    if i == 6:
        axs[i].text(-115, -28, 'End-Start', rotation=90, size=12)

    # coastines
    axs[i].coastlines()

    # turn off bounding box
    axs[i].spines['geo'].set_edgecolor('white')

    # add plot labels
    for n, ax in enumerate(axs):
        ax.text(0.9, 0.75, string.ascii_lowercase[n] + ')', transform=ax.transAxes,
                        size=13, weight='normal')

# Add a vertical colorbar axis to diff axis
# format x1,y1,x2,y2
cbar_ax = fig.add_axes([1, 0.4, 0.01, 0.46])
fig.colorbar(cs_1, ax=axs[5], cax=cbar_ax)
cbar_ax.set_ylabel('mm month$^{-1}$', rotation=90)

cbar_ax = fig.add_axes([1, 0.15, 0.01, 0.19])
fig.colorbar(cs_2, ax=axs[8], cax=cbar_ax)
cbar_ax.set_ylabel('mm month$^{-1}$', rotation=90)

plt.subplots_adjust(wspace=-0.8,hspace=-0.8)
plt.tight_layout()

out_path = 'file_path'
fname = 'map_P_start_end_diff_3x3_ED_plot.jpg'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
