#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot change in forest cover at each resolution onto maps
"""

# read in packages
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import xarray as xr
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature

# read in shapefiles
shp_path = 'file_path'
amazon_shp = list(shpreader.Reader(shp_path + 'Amazon_biome/amazonia.shp').geometries())
congo_shp = list(shpreader.Reader(shp_path + 'congo_shp_sept21.shp').geometries())
sea_shp = list(shpreader.Reader(shp_path + 'sea_with_png.shp').geometries())

# read in forest cover change at different resolutions
data_path = 'file_path'
ds_005 = xr.open_dataset(data_path + 'treeCoverChange_tropics_0.05_2017_v1.9_bilinear.nc')
ds_01 = xr.open_dataset(data_path + 'treeCoverChange_tropics_0.1_2017_v1.9_bilinear.nc')
ds_025 = xr.open_dataset(data_path + 'treeCoverChange_tropics_0.25_2017_v1.9_bilinear.nc')
ds_05 = xr.open_dataset(data_path + 'treeCoverChange_tropics_0.5_2017_v1.9_bilinear.nc')
ds_1 = xr.open_dataset(data_path + 'treeCoverChange_tropics_1.0_2017_v1.9_bilinear.nc')
ds_2 = xr.open_dataset(data_path + 'treeCoverChange_tropics_2.0_2017_v1.9_bilinear.nc')

# get lat and longitudes into lists
longitudes_005 = ds_005['longitude']
latitudes_005 = ds_005['latitude']
longitudes_01 = ds_01['longitude']
latitudes_01 = ds_01['latitude']
longitudes_025 = ds_025['longitude']
latitudes_025 = ds_025['latitude']
longitudes_05 = ds_05['longitude']
latitudes_05 = ds_05['latitude']
longitudes_1 = ds_1['longitude']
latitudes_1 = ds_1['latitude']
longitudes_2 = ds_2['longitude']
latitudes_2 = ds_2['latitude']

# get the data arrays from the datasets
ds_005 = ds_005['tree']
ds_01 = ds_01['tree']
ds_025 = ds_025['tree']
ds_05 = ds_05['tree']
ds_1 = ds_1['tree']
ds_2 = ds_2['tree']

# plotting the maps
# Define the figure and each axis for the 3 rows and 3 columns
nrows = 3
ncols = 2
# colour bar min and max
cbmax = 0
cbmin = -30

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-100, 160, -30, 30], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.6)
ax1.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.6)
ax1.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.6)
varplot = ax1.pcolormesh(longitudes_005, latitudes_005, ds_005, vmax=cbmax, vmin=cbmin, cmap='gist_heat', alpha=1)
ax1.spines['geo'].set_edgecolor('white')
ax1.annotate('a)', xy=(0.9, 0.8), xycoords="axes fraction")

ax2 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent([-100, 160, -30, 30], crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax2.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.6)
ax2.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.6)
ax2.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.6)
varplot = ax2.pcolormesh(longitudes_01, latitudes_01, ds_01, vmax=cbmax, vmin=cbmin, cmap='gist_heat', alpha=1)
ax2.spines['geo'].set_edgecolor('white')
ax2.annotate('b)', xy=(0.9, 0.8), xycoords="axes fraction")

ax3 = fig.add_subplot(3, 2, 3, projection=ccrs.PlateCarree())
ax3.set_extent([-100, 160, -30, 30], crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax3.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.6)
ax3.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.6)
ax3.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.6)
varplot = ax3.pcolormesh(longitudes_025, latitudes_025, ds_025, vmax=cbmax, vmin=cbmin, cmap='gist_heat', alpha=1)
ax3.spines['geo'].set_edgecolor('white')
ax3.annotate('c)', xy=(0.9, 0.8), xycoords="axes fraction")

ax4 = fig.add_subplot(3, 2, 4, projection=ccrs.PlateCarree())
ax4.set_extent([-100, 160, -30, 30], crs=ccrs.PlateCarree())
ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax4.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.6)
ax4.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.6)
ax4.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.6)
varplot = ax4.pcolormesh(longitudes_05, latitudes_05, ds_05, vmax=cbmax, vmin=cbmin, cmap='gist_heat', alpha=1)
ax4.spines['geo'].set_edgecolor('white')
ax4.annotate('d)', xy=(0.9, 0.8), xycoords="axes fraction")

ax5 = fig.add_subplot(3, 2, 5, projection=ccrs.PlateCarree())
ax5.set_extent([-100, 160, -30, 30], crs=ccrs.PlateCarree())
ax5.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax5.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.6)
ax5.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.6)
ax5.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.6)
varplot = ax5.pcolormesh(longitudes_1, latitudes_1, ds_1, vmax=cbmax, vmin=cbmin, cmap='gist_heat', alpha=1)
ax5.spines['geo'].set_edgecolor('white')
ax5.annotate('e)', xy=(0.9, 0.8), xycoords="axes fraction")

ax6 = fig.add_subplot(3, 2, 6, projection=ccrs.PlateCarree())
ax6.set_extent([-100, 160, -30, 30], crs=ccrs.PlateCarree())
ax6.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax6.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8,  linewidth=0.6)
ax6.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='purple',  fc='none', alpha=0.8, linewidth=0.6)
ax6.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='purple', fc='none', alpha=0.8, linewidth=0.6)
varplot = ax6.pcolormesh(longitudes_2, latitudes_2, ds_2, vmax=cbmax, vmin=cbmin, cmap='gist_heat', alpha=1)
ax6.spines['geo'].set_edgecolor('white')
ax6.annotate('f)', xy=(0.9, 0.8), xycoords="axes fraction")
# ax6.set_yticklabels()

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0, 0.6, 0.02])

# Draw the colorbar
cbar=fig.colorbar(varplot, cax=cbar_ax,orientation='horizontal', extend='min')
cbar.set_label('Forest cover change (%)')  # cax == cb.ax

plt.tight_layout()

print('saving map')
out_path = 'file_path'
fname = 'map_DCC_03_17_all_res_unboxed.jpeg'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
