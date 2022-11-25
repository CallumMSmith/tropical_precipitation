#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make multipanel plot of GCAM tree cover loss data and how that impacts P in future
Completed at 2.0 degree resolution only
Using 3x3 grid and 5yr analysis and shorter time series output
"""

# read in packages
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy
import rioxarray as rio
import geopandas as gpd
from numpy import median
from numpy import mean
from matplotlib.offsetbox import AnchoredText
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

# paths to files
gcam_path = 'file_path'
p_path = 'file_path'
fig_path = 'file_path'
shp_path = 'file_path'

# select resolution
res = '2.0'
# read in files
ds_gcam = xr.open_dataset(gcam_path + 'gcam_' + res + 'deg_loss_year_notnan_gainnan.nc')
ds_p    = xr.open_dataset(p_path + 'delP_by_delCC_' + res + 'deg_5yr_3x3_03_17.nc')

# change gcam tree cover loss to be positive (i.e. canopy cover loss)
ds_gcam = np.negative(ds_gcam)

# multiply the datasets to find change in P with time
# read in median satellite P value for each reigon, amazon, congo, sea and tropics from a table...
# then apply to df using dictionary
# sort and find median of P ds
# find median for each dataset category
if res == '2.0':
    satellite_med = ds_p[['pr_cmorph' ,'pr_per_ccs' ,'pr_per_ccscdr' ,'pr_per_cdr' ,'pr_per_now' ,'pr_per_per' ,'pr_chirps' ,'pr_gpcp' ,'pr_gpm' ,'pr_trmm']].to_array(dim='new').median('new')
    ds_p = ds_p.assign(satellite=satellite_med)
else:
    satellite_med = ds_p[['pr_per_ccs' ,'pr_per_ccscdr' ,'pr_per_now'  ,'pr_chirps']].to_array(dim='new').median('new')
    ds_p = ds_p.assign(satellite=satellite_med)


# convert ds to df
df_p = ds_p['satellite'].to_dataframe().reset_index().rename(columns={"level_0": "lat"})

# convert df to geo dataframe using lat and lons
gdf_p = gpd.GeoDataFrame(df_p, geometry=gpd.points_from_xy(df_p.lon, df_p.lat))

# read in shapefiles to define the areas
amazon_shp = gpd.read_file(shp_path + 'Amazon_biome/amazonia.shp')
congo_shp = gpd.read_file(shp_path + 'congo_shp_sept21.shp')
sea_shp = gpd.read_file(shp_path + 'sea_with_png.shp')

# clip the regions using the shapefiles
gdf_amazon = gdf_p.clip(amazon_shp)
gdf_congo = gdf_p.clip(congo_shp)
gdf_sea = gdf_p.clip(sea_shp)

# assign the regions a column 'region' value
gdf_p['region'] = 'tropics'
gdf_amazon['region'] = 'amazon'
gdf_congo['region'] = 'congo'
gdf_sea['region'] = 'sea'

# join the gdfs together into one, reset index and remove nan rows
gdf_p = pd.concat([gdf_p, gdf_amazon, gdf_congo, gdf_sea]).reset_index().dropna()


# find the median value for each region and save the number to a variable
median_p = (gdf_p.groupby(['region'])['satellite'].median().reset_index().set_index('region')['satellite'])

# mask the gcam tree loss layer to just be over the evergreen broadleaf area
tree_path = 'file_path'
ds_tc = xr.open_dataset(tree_path + 'treeCoverChange_tropics_'+res+'_2017_v1.9_bilinear.nc')
# change latitude to lat etc.
ds_tc = ds_tc.rename({'longitude': 'lon','latitude': 'lat'})

mask_eb = np.ones((ds_tc.dims['lat'], ds_tc.dims['lon'])) * np.isfinite(ds_tc.tree)
mask_eb = mask_eb.where(mask_eb != 0)

# enact the mask on the tree cover layer
ds_gcam = ds_gcam * mask_eb


# now read in gcam, constrain by tree cover netcdf, categorise into regions, then apply the multiplication factor calcd above
# convert ds to df
df_gcam = ds_gcam.to_dataframe().reset_index().rename(columns={"level_0": "lat"})
# print(df_gcam)
# convert df to geo dataframe using lat and lons
gdf_gcam = gpd.GeoDataFrame(df_gcam, geometry=gpd.points_from_xy(df_gcam.lon, df_gcam.lat))

# read in shapefiles to define the areas
amazon_shp = gpd.read_file(shp_path + 'Amazon_biome/amazonia.shp')
congo_shp = gpd.read_file(shp_path + 'congo_shp_sept21.shp')
sea_shp = gpd.read_file(shp_path + 'sea_with_png.shp')

# clip the regions using the shapefiles
gdf_amazon = gdf_gcam.clip(amazon_shp)
gdf_congo = gdf_gcam.clip(congo_shp)
gdf_sea = gdf_gcam.clip(sea_shp)

# assign the regions a column 'region' value
gdf_gcam['region'] = 'tropics'
gdf_amazon['region'] = 'amazon'
gdf_congo['region'] = 'congo'
gdf_sea['region'] = 'sea'

# join the gdfs together into one
gdf_gcam = pd.concat([gdf_gcam, gdf_amazon, gdf_congo, gdf_sea]).reset_index()

# multiply GCAM tree cover loss by regional median P value
# map median p value to dataframe
gdf_gcam['region_val'] = gdf_gcam['region'].map(median_p)

# these lines are necessary if capping the analysis at 30%
# make a new df that contains the precip projections which are capped to 30
gdf_gcam_p_proj = gdf_gcam.copy()
# where PFT4 > 30, set to 30
# gdf_gcam_p_proj['PFT4'].values[gdf_gcam_p_proj['PFT4'] > 30] = 30
# gdf_gcam['PFT4'].values[gdf_gcam['PFT4'] > 30] = 30

# multiply P change region val by P 'PFT4' (GCAM tree cover loss)
gdf_gcam_p_proj['p_fut'] = gdf_gcam_p_proj['PFT4'] * gdf_gcam_p_proj['region_val']

# find what the median value of delP should be in 2100
# print('median 2100 PFT', gdf_gcam_p_proj.groupby(['time','region'])['PFT4'].mean())
# print('median 2100 p fut', gdf_gcam_p_proj.groupby(['time','region'])['p_fut'].mean())

# create dfs for plotting
gdf_gcam_plot = gdf_gcam_p_proj.loc[gdf_gcam['time'] == '2100-01-01'].drop(columns=['index', 'geometry', 'PFT4', 'region_val', 'time'])
tropics_df = gdf_gcam_plot.loc[gdf_gcam_plot['region'] == 'tropics']
amazon_df = gdf_gcam_plot.loc[((gdf_gcam_plot['region'] == 'amazon'))]
congo_df = gdf_gcam_plot.loc[((gdf_gcam_plot['region'] == 'congo'))]
sea_df = gdf_gcam_plot.loc[((gdf_gcam_plot['region'] == 'sea'))]

# now makes into ds
tropics_df = tropics_df.set_index(["lat", "lon"]).drop(columns=['region'])
# could add all lats and lons from tropics df to regions df, this might solve congo bug
amazon_df = amazon_df.set_index(["lat", "lon"]).drop(columns=['region'])
congo_df = congo_df.set_index(["lat", "lon"]).drop(columns=['region'])
sea_df = sea_df.set_index(["lat", "lon"]).drop(columns=['region'])

tropics_ds = xr.Dataset.from_dataframe(tropics_df)
amazon_ds = xr.Dataset.from_dataframe(amazon_df)
congo_ds = xr.Dataset.from_dataframe(congo_df)
sea_ds = xr.Dataset.from_dataframe(sea_df)

# convert to dataarray for plotting and limit to tropics
tropics_da = tropics_ds.p_fut.sel(lon=slice(-100,160)).sel(lat=slice(-30,30))
amazon_da = amazon_ds.p_fut.sel(lon=slice(-100,160)).sel(lat=slice(-30,30))
congo_da = congo_ds.p_fut.sel(lon=slice(-100,160)).sel(lat=slice(-30,30))
sea_da = sea_ds.p_fut.sel(lon=slice(-100,160)).sel(lat=slice(-30,30))

# make one multipanel labeled plot
# tree cover change over time, P change over time, map of tree cover 2100, map of p 2100
# panel a, tree cover loss vs time

labels = ['Tropics', 'Amazon', 'Congo', 'SEA']

# preamble
fs = 12
font = {'weight' : 'normal',
        'size'   : fs,
        'serif': 'Arial'}

plt.rc('font', **font)
plt.rc('axes', linewidth=1.5)

sns.set_style("white")

fig = plt.figure(figsize=(12,9))
ncols = 30
gs = gridspec.GridSpec(nrows=3, ncols=ncols, figure=fig)
yr1 = pd.to_datetime('2018', format='%Y')
yr2 = pd.to_datetime('2102', format='%Y')

ax1 = fig.add_subplot(gs[0, 0:12])
g = sns.lineplot(data=gdf_gcam, x='time', y='PFT4', hue='region', estimator=mean,
                 ci=68, ax=ax1, legend=False,
                 palette=['crimson', 'deepskyblue', 'darkorange', 'mediumorchid'])
plt.xlabel('')
plt.ylabel('Forest cover loss (%)', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)


# panel b, P change vs time
ax2 = fig.add_subplot(gs[0, 15:-3])

g = sns.lineplot(data=gdf_gcam_p_proj, x='time', y='p_fut', hue='region',
                 estimator=mean, ci=68, ax=ax2, legend=True,
                 palette=['crimson', 'deepskyblue', 'darkorange', 'mediumorchid'])

h, l = ax2.get_legend_handles_labels()
plt.legend(handles=h, labels = labels, frameon=False, loc='lower left', fontsize=fs)
plt.xticks(fontsize=fs)
plt.xlabel('')
plt.ylabel('∆P (mm month$^{-1}$)', fontsize=fs)
plt.yticks(fontsize=fs)


# panel c, map of tree cover loss in 2100
ax3 = fig.add_subplot(gs[1, 0:-1], projection=ccrs.PlateCarree())
gcam_100 = ds_gcam.PFT4.sel(time='2100-01-01').sel(lon=slice(-100,160)).sel(lat=slice(-30,30))
ax3.add_feature(cfeature.LAND, linewidth=0)
ax3.add_feature(cfeature.OCEAN, linewidth=0, zorder=1)
ax3.add_feature(cfeature.COASTLINE, linewidth=0)
ax3.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='black', fc='none', alpha=0.6,  linewidth=0.5)
ax3.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='black',  fc='none', alpha=0.6, linewidth=0.5)
ax3.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='black', fc='none', alpha=0.6, linewidth=0.5)
gcam_100.plot(ax=ax3, transform=ccrs.PlateCarree(),
         vmin=0, vmax=100, cmap='Reds', cbar_kwargs={'shrink': 0.85, 'label':'Forest cover loss (%)'})
ax3.set_xticklabels([])
ax3.set_yticklabels([])
plt.title('')


# panel d, map of P change in 2100
cbmin=-40
cbmax=0
ax4 = fig.add_subplot(gs[2, 0:-1], projection=ccrs.PlateCarree())
# ax4.set_extent([-100, 160, -30, 30])
ax4.add_geometries(amazon_shp, ccrs.PlateCarree(), edgecolor='black', fc='none', alpha=0.6,  linewidth=0.5)
ax4.add_geometries(congo_shp, ccrs.PlateCarree(), edgecolor='black',  fc='none', alpha=0.6, linewidth=0.5)
ax4.add_geometries(sea_shp, ccrs.PlateCarree(), edgecolor='black', fc='none', alpha=0.6, linewidth=0.5)
tropics_da.plot(ax=ax4, transform=ccrs.PlateCarree(), vmin=cbmin, vmax=cbmax, cmap='Reds_r', cbar_kwargs={'shrink': 0.85, 'label':'∆P (mm month$^{-1}$)'})
amazon_da.plot(ax=ax4, transform=ccrs.PlateCarree(), vmin=cbmin, vmax=cbmax, cmap='Reds_r', add_colorbar=False)
congo_da.plot(ax=ax4, transform=ccrs.PlateCarree(), vmin=cbmin, vmax=cbmax, cmap='Reds_r', add_colorbar=False)
sea_da.plot(ax=ax4, transform=ccrs.PlateCarree(), vmin=cbmin, vmax=cbmax, cmap='Reds_r', add_colorbar=False)
ax4.add_feature(cfeature.OCEAN, linewidth=0, zorder=1)
ax4.add_feature(cfeature.COASTLINE, linewidth=0)
ax4.add_feature(cfeature.LAND, linewidth=0)
plt.title('')
ax4.set_xticklabels([])
ax4.set_yticklabels([])


# add fig annotations
at = AnchoredText("a)", prop=dict(size=fs), frameon=False, loc='upper left')
ax1.add_artist(at)
at = AnchoredText("b)", prop=dict(size=fs), frameon=False, loc='upper left')
ax2.add_artist(at)
at = AnchoredText("c)", prop=dict(size=fs), frameon=False, loc='upper right')
ax3.add_artist(at)
at = AnchoredText("d)", prop=dict(size=fs), frameon=False, loc='upper right')
ax4.add_artist(at)

plt.savefig(fig_path + 'gcam_unlimit_5yr_3x3_03_17_mean.pdf', bbox_inches='tight', dpi=600)
sns.reset_orig()
