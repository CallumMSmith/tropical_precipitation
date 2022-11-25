#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a series of bars comparing wet dry and annual mean values for each resolution.
Absolute change in P for grid 3x3, average years 5yr and shorter time series
"""

# read in packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import median
from numpy import mean
import geopandas as gpd
import xarray as xr


# categorise the datasets
def f(row):
    val = []
    if (row['Dataset'] == 'pr_cmorph' or row['Dataset'] == 'pr_per_ccs' or row['Dataset'] == 'pr_per_ccscdr' or row['Dataset'] == 'pr_per_cdr' or row['Dataset'] == 'pr_per_now'
        or row['Dataset'] == 'pr_per_per' or row['Dataset'] == 'pr_chirps' or row['Dataset'] == 'pr_gpcp' or row['Dataset'] == 'pr_gpm' or row['Dataset'] == 'pr_trmm'):
        val = 'Satellite'
    elif row['Dataset'] == 'pr_cpc' or row['Dataset'] == 'pr_cru' or row['Dataset'] == 'pr_gpcc' or row['Dataset'] == 'pr_udel':
        val = 'Station'
    elif row['Dataset'] == row['Dataset'] == 'pr_era' or row['Dataset'] == 'pr_merra' or row['Dataset'] == 'pr_ncep' or row['Dataset'] == 'pr_noaa' or row['Dataset'] == 'pr_jra':
        val = 'Reanalysis'
    return val


# read in datasets
data_path = 'file_path'

# annual season

# 0.05 degree
ds_005 = xr.open_dataset(data_path + 'delP_by_delCC_0.05deg_5yr_3x3_03_17.nc').rename({'lat':'latitude','lon':'longitude'}).to_dataframe().stack().reset_index()
df_005 = ds_005.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.05')
df_005['Cat'] = df_005.apply(f, axis=1)

# 0.1 degree
ds_01 = xr.open_dataset(data_path + 'delP_by_delCC_0.1deg_5yr_3x3_03_17.nc').rename({'lat':'latitude','lon':'longitude'}).to_dataframe().stack().reset_index()
df_01 = ds_01.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.1')
df_01['Cat'] = df_01.apply(f, axis=1)

# 0.25 degree
ds_025 = xr.open_dataset(data_path + 'delP_by_delCC_0.25deg_5yr_3x3_03_17.nc').rename({'lat':'latitude','lon':'longitude'}).to_dataframe().stack().reset_index()
df_025 = ds_025.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.25')
df_025['Cat'] = df_025.apply(f, axis=1)

# 0.5 degree
ds_05 = xr.open_dataset(data_path + 'delP_by_delCC_0.5deg_5yr_3x3_03_17.nc').rename({'lat':'latitude','lon':'longitude'}).to_dataframe().stack().reset_index()
df_05 = ds_05.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.5')
df_05['Cat'] = df_05.apply(f, axis=1)

# 1.0 degree
ds_1 = xr.open_dataset(data_path + 'delP_by_delCC_1.0deg_5yr_3x3_03_17.nc').rename({'lat':'latitude','lon':'longitude'}).to_dataframe().stack().reset_index()
df_1 = ds_1.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='1.0')
df_1['Cat'] = df_1.apply(f, axis=1)

# 2.0 degree
ds_2 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_17.nc').rename({'lat':'latitude','lon':'longitude'}).to_dataframe().stack().reset_index()
df_2 = ds_2.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='2.0')
df_2['Cat'] = df_2.apply(f, axis=1)

print('concatentating dfs... ')

# concat all dfs together into one
df_trop_annual = pd.concat([df_005, df_01, df_025, df_05, df_1, df_2])


# wet season
# 0.05 degree
ds_005 = xr.open_dataset(data_path + 'delP_by_delCC_0.05deg_5yr_3x3_03_17_wet.nc').to_dataframe().stack().reset_index()
df_005 = ds_005.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.05')
df_005['Cat'] = df_005.apply(f, axis=1)

# 0.1 degree
ds_01 = xr.open_dataset(data_path + 'delP_by_delCC_0.1deg_5yr_3x3_03_17_wet.nc').to_dataframe().stack().reset_index()
df_01 = ds_01.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.1')
df_01['Cat'] = df_01.apply(f, axis=1)

# 0.25 degree
ds_025 = xr.open_dataset(data_path + 'delP_by_delCC_0.25deg_5yr_3x3_03_17_wet.nc').to_dataframe().stack().reset_index()
df_025 = ds_025.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.25')
df_025['Cat'] = df_025.apply(f, axis=1)

# 0.5 degree
ds_05 = xr.open_dataset(data_path + 'delP_by_delCC_0.5deg_5yr_3x3_03_17_wet.nc').to_dataframe().stack().reset_index()
df_05 = ds_05.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.5')
df_05['Cat'] = df_05.apply(f, axis=1)

# 1.0 degree
ds_1 = xr.open_dataset(data_path + 'delP_by_delCC_1.0deg_5yr_3x3_03_17_wet.nc').to_dataframe().stack().reset_index()
df_1 = ds_1.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='1.0')
df_1['Cat'] = df_1.apply(f, axis=1)

# 2.0 degree
ds_2 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_17_wet.nc').to_dataframe().stack().reset_index()
df_2 = ds_2.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='2.0')
df_2['Cat'] = df_2.apply(f, axis=1)

# concat all dfs together into one
df_trop_wet = pd.concat([df_005, df_01, df_025, df_05, df_1, df_2])


# dry season
# 0.05 degree
ds_005 = xr.open_dataset(data_path + 'delP_by_delCC_0.05deg_5yr_3x3_03_17_dry.nc').to_dataframe().stack().reset_index()
df_005 = ds_005.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.05')
df_005['Cat'] = df_005.apply(f, axis=1)

# 0.1 degree
ds_01 = xr.open_dataset(data_path + 'delP_by_delCC_0.1deg_5yr_3x3_03_17_dry.nc').to_dataframe().stack().reset_index()
df_01 = ds_01.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.1')
df_01['Cat'] = df_01.apply(f, axis=1)

# 0.25 degree
ds_025 = xr.open_dataset(data_path + 'delP_by_delCC_0.25deg_5yr_3x3_03_17_dry.nc').to_dataframe().stack().reset_index()
df_025 = ds_025.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.25')
df_025['Cat'] = df_025.apply(f, axis=1)

# 0.5 degree
ds_05 = xr.open_dataset(data_path + 'delP_by_delCC_0.5deg_5yr_3x3_03_17_dry.nc').to_dataframe().stack().reset_index()
df_05 = ds_05.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.5')
df_05['Cat'] = df_05.apply(f, axis=1)

# 1.0 degree
ds_1 = xr.open_dataset(data_path + 'delP_by_delCC_1.0deg_5yr_3x3_03_17_dry.nc').to_dataframe().stack().reset_index()
df_1 = ds_1.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='1.0')
df_1['Cat'] = df_1.apply(f, axis=1)

# 2.0 degree
ds_2 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_17_dry.nc').to_dataframe().stack().reset_index()
df_2 = ds_2.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='2.0')
df_2['Cat'] = df_2.apply(f, axis=1)

# concat all dfs together into one
df_trop_dry = pd.concat([df_005, df_01, df_025, df_05, df_1, df_2])


# trans season
# 0.05 degree
ds_005 = xr.open_dataset(data_path + 'delP_by_delCC_0.05deg_5yr_3x3_03_17_trans.nc').to_dataframe().stack().reset_index()
df_005 = ds_005.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.05')
df_005['Cat'] = df_005.apply(f, axis=1)

# 0.1 degree
ds_01 = xr.open_dataset(data_path + 'delP_by_delCC_0.1deg_5yr_3x3_03_17_trans.nc').to_dataframe().stack().reset_index()
df_01 = ds_01.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.1')
df_01['Cat'] = df_01.apply(f, axis=1)

# 0.25 degree
ds_025 = xr.open_dataset(data_path + 'delP_by_delCC_0.25deg_5yr_3x3_03_17_trans.nc').to_dataframe().stack().reset_index()
df_025 = ds_025.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.25')
df_025['Cat'] = df_025.apply(f, axis=1)

# 0.5 degree
ds_05 = xr.open_dataset(data_path + 'delP_by_delCC_0.5deg_5yr_3x3_03_17_trans.nc').to_dataframe().stack().reset_index()
df_05 = ds_05.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.5')
df_05['Cat'] = df_05.apply(f, axis=1)

# 1.0 degree
ds_1 = xr.open_dataset(data_path + 'delP_by_delCC_1.0deg_5yr_3x3_03_17_trans.nc').to_dataframe().stack().reset_index()
df_1 = ds_1.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='1.0')
df_1['Cat'] = df_1.apply(f, axis=1)

# 2.0 degree
ds_2 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_17_trans.nc').to_dataframe().stack().reset_index()
df_2 = ds_2.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='2.0')
df_2['Cat'] = df_2.apply(f, axis=1)

# concat all dfs together into one
df_trop_trans = pd.concat([df_005, df_01, df_025, df_05, df_1, df_2])


# apply season col to each df
df_trop_annual['season'] = 'Annual'
df_trop_wet['season'] = 'Wet'
df_trop_dry['season'] = 'Dry'
df_trop_trans['season'] = 'Transition'



# clip areas by shapefiles into regions
# read in shapefile
shp_path = 'file_path'

amazon_shp = gpd.read_file(shp_path + 'amazonia.shp')
congo_shp = gpd.read_file(shp_path + 'congo_shp_sept21.shp')
sea_shp = gpd.read_file(shp_path + 'sea_with_png.shp')

# convert pandas df to geopandas
gdf_annual = gpd.GeoDataFrame(df_trop_annual, geometry=gpd.points_from_xy(df_trop_annual.longitude, df_trop_annual.latitude))
gdf_wet = gpd.GeoDataFrame(df_trop_wet, geometry=gpd.points_from_xy(df_trop_wet.longitude, df_trop_wet.latitude))
gdf_dry = gpd.GeoDataFrame(df_trop_dry, geometry=gpd.points_from_xy(df_trop_dry.longitude, df_trop_dry.latitude))
gdf_trans = gpd.GeoDataFrame(df_trop_trans, geometry=gpd.points_from_xy(df_trop_trans.longitude, df_trop_trans.latitude))

# clip geopandas to regions
gdf_amazon_annual = gdf_annual.clip(amazon_shp)
gdf_congo_annual = gdf_annual.clip(congo_shp)
gdf_sea_annual = gdf_annual.clip(sea_shp)

gdf_amazon_wet = gdf_wet.clip(amazon_shp)
gdf_congo_wet = gdf_wet.clip(congo_shp)
gdf_sea_wet = gdf_wet.clip(sea_shp)

gdf_amazon_dry = gdf_dry.clip(amazon_shp)
gdf_congo_dry = gdf_dry.clip(congo_shp)
gdf_sea_dry = gdf_dry.clip(sea_shp)

gdf_amazon_trans = gdf_trans.clip(amazon_shp)
gdf_congo_trans = gdf_trans.clip(congo_shp)
gdf_sea_trans = gdf_trans.clip(sea_shp)


# concat the regions together and plot as one
gdf_annual['Region'] = 'Tropics'
gdf_amazon_annual['Region'] = 'Amazon'
gdf_congo_annual['Region'] = 'Congo'
gdf_sea_annual['Region'] = 'SEA'

gdf_wet['Region'] = 'Tropics'
gdf_amazon_wet['Region'] = 'Amazon'
gdf_congo_wet['Region'] = 'Congo'
gdf_sea_wet['Region'] = 'SEA'

gdf_dry['Region'] = 'Tropics'
gdf_amazon_dry['Region'] = 'Amazon'
gdf_congo_dry['Region'] = 'Congo'
gdf_sea_dry['Region'] = 'SEA'

gdf_trans['Region'] = 'Tropics'
gdf_amazon_trans['Region'] = 'Amazon'
gdf_congo_trans['Region'] = 'Congo'
gdf_sea_trans['Region'] = 'SEA'


df_all_annual = pd.concat([gdf_annual, gdf_amazon_annual, gdf_congo_annual, gdf_sea_annual])
df_all_wet = pd.concat([gdf_wet, gdf_amazon_wet, gdf_congo_wet, gdf_sea_wet])
df_all_dry = pd.concat([gdf_dry, gdf_amazon_dry, gdf_congo_dry, gdf_sea_dry])
df_all_trans = pd.concat([gdf_trans, gdf_amazon_trans, gdf_congo_trans, gdf_sea_trans])

# create one df for plotting
df_all = pd.concat([df_all_annual, df_all_wet, df_all_dry, df_all_trans])

# select only satellite datasets
df_all = df_all.loc[df_all['Cat'] == 'Satellite']

# subset df_all for plotting
# ax1
ax1 = df_all.loc[df_all['Region'] == 'Tropics']
# ax1
ax2 = df_all.loc[df_all['Region'] == 'Amazon']
# ax1
ax3 = df_all.loc[df_all['Region'] == 'Congo']
# ax1
ax4 = df_all.loc[df_all['Region'] == 'SEA']

# set up figure
sns.set_context("paper", font_scale=3.0)
fs = 15
fs_t = 15

inner = [['innerA'],
         ['innerB']]
outer = [['upper left',  'upper right'],
          [inner, 'lower right']]

palette ={"Annual": "limegreen", "Dry": "darkorange", "Wet": "dodgerblue", "Transition": "purple"}

# use matplotlib mosaic plot
fig, axd = plt.subplot_mosaic(outer, constrained_layout=True, figsize=(20,20))

g = sns.barplot(ax = axd['upper left'], x="Res", y="P", hue='season', data=ax1, palette=palette, ci=68, estimator=median, order=['0.05', '0.1', '0.25', '0.5', '1.0', '2.0'])
g.legend_.remove()
g.axhline(0, alpha=0.4, c='k')
g.annotate('a)', xy=(0.05, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-0.9, 0.35))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)
g.set_title('Tropics')


g = sns.barplot(ax = axd['upper right'], x="Res", y="P", hue='season', data=ax2, palette=palette, ci=68, estimator=median, order=['0.05', '0.1', '0.25', '0.5', '1.0', '2.0'])
g.legend_.remove()
g.axhline(0, alpha=0.4, c='k')
g.annotate('b)', xy=(0.05, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-0.9, 0.35))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)
g.set_title('Amazon')


ax_a = sns.barplot(ax = axd['innerA'], x="Res", y="P", hue='season', data=ax3, palette=palette, ci=68, estimator=median, order=['0.05', '0.1', '0.25', '0.5', '1.0', '2.0'])
ax_a.axhline(0, alpha=0.4, c='k')
ax_a.legend_.remove()
ax_a.annotate('c)', xy=(0.05, 0.9), xycoords="axes fraction")
ax_a.set(xticklabels=[])
ax_a.set(xlabel=None)
ax_a.set(ylabel=None)
ax_a.set(ylim=(-0.75, 0.95))
sns.despine(right=True, top=True, bottom=True)
ax_a.set_title('Congo')
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax_a.transAxes, color='k', clip_on=False)
ax_a.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

ax_b = sns.barplot(ax = axd['innerB'], x="Res", y="P", hue='season', data=ax3, palette=palette, ci=68, estimator=median, order=['0.05', '0.1', '0.25', '0.5', '1.0', '2.0'])
ax_b.axhline(0, alpha=0.4, c='k')
ax_b.legend_.remove()
ax_b.set(xlabel=None)
ax_b.set(ylabel=None)
ax_b.set(ylim=(-3, -1.3))
sns.despine(right=True, top=True)
kwargs.update(transform=ax_b.transAxes)  # switch to the bottom axes
ax_b.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal


g = sns.barplot(ax = axd['lower right'], x="Res", y="P", hue='season', data=ax4, palette=palette, ci=68, estimator=median, order=['0.05', '0.1', '0.25', '0.5', '1.0', '2.0'])
g.axhline(0, alpha=0.4, c='k')
g.annotate('d)', xy=(0.05, 0.9), xycoords="axes fraction")
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-0.9, 0.35))
g.tick_params(left=False)
sns.despine(right=True, top=True)
g.set_title('SEA')
sns.move_legend(g, 'lower left', frameon=False, title='')


# common axis labels
fig.supylabel('∆P (mm month$^{-1}$ %$^{-1}$)')
fig.supxlabel('Resolution (degrees)')

out_path = 'file_path'
fname = 'precip_barplot_absolute_5yr_3x3_broken_axis.pdf'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
