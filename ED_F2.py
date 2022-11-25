#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Barplot showing median change in Precipitation due to forest loss for all datasets
Displayed for all resolutions, regions and datasets
Analysis from 3x3 grid, 5yr averages and shorter time series
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
import matplotlib


# categorise datasets
def f(row):
    val = []
    if row['Dataset'] == 'pr_cmorph' or row['Dataset'] == 'pr_per_ccs' or row['Dataset'] == 'pr_per_ccscdr' or row['Dataset'] == 'pr_per_cdr' or row['Dataset'] == 'pr_per_now' or row['Dataset'] == 'pr_per_per' or row['Dataset'] == 'pr_chirps' or row['Dataset'] == 'pr_gpcp' or row['Dataset'] == 'pr_gpm' or row['Dataset'] == 'pr_trmm':
        val = 'Satellite'
    elif row['Dataset'] == 'pr_cpc' or row['Dataset'] == 'pr_cru' or row['Dataset'] == 'pr_gpcc' or row['Dataset'] == 'pr_udel':
        val = 'Station'
    elif row['Dataset'] == row['Dataset'] == 'pr_era' or row['Dataset'] == 'pr_merra' or row['Dataset'] == 'pr_ncep' or row['Dataset'] == 'pr_noaa' or row['Dataset'] == 'pr_jra':
        val = 'Reanalysis'
    return val


data_path = ('file_path')

# 0.05 degree
ds_005 = xr.open_dataset(data_path + 'delP_by_delCC_0.05deg_5yr_3x3_03_17.nc').sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_005 = ds_005.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.05')
df_005['Cat'] = df_005.apply(f, axis=1)

# 0.1 degree
ds_01 = xr.open_dataset(data_path + 'delP_by_delCC_0.1deg_5yr_3x3_03_17.nc').sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_01 = ds_01.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.1')
df_01['Cat'] = df_01.apply(f, axis=1)

# 0.25 degree
ds_025 = xr.open_dataset(data_path + 'delP_by_delCC_0.25deg_5yr_3x3_03_17.nc').sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_025 = ds_025.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.25')
df_025['Cat'] = df_025.apply(f, axis=1)

# 0.5 degree
ds_05 = xr.open_dataset(data_path + 'delP_by_delCC_0.5deg_5yr_3x3_03_17.nc').sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_05 = ds_05.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='0.5')
df_05['Cat'] = df_05.apply(f, axis=1)

# 1.0 degree
ds_1 = xr.open_dataset(data_path + 'delP_by_delCC_1.0deg_5yr_3x3_03_17.nc').sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_1 = ds_1.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='1.0')
df_1['Cat'] = df_1.apply(f, axis=1)

# 2.0 degree
ds_2 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_17.nc').sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_2 = ds_2.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Res='2.0')
df_2['Cat'] = df_2.apply(f, axis=1)

# concat all dfs together into one
df_trop = pd.concat([df_005, df_01, df_025, df_05, df_1, df_2])


# clip area to amazon
# convert pandas df to geopandas
gdf = gpd.GeoDataFrame(df_trop, geometry=gpd.points_from_xy(df_trop.lon, df_trop.lat))

# read in shapefile
shp_path = 'file_path'

amazon_shp = gpd.read_file(shp_path + 'amazonia.shp')
congo_shp = gpd.read_file(shp_path + 'congo_shp_sept21.shp')
sea_shp = gpd.read_file(shp_path + 'sea_with_png.shp')

gdf_amazon = gdf.clip(amazon_shp)
gdf_congo = gdf.clip(congo_shp)
gdf_sea = gdf.clip(sea_shp)

# concat the regions together and plot as one
gdf['Region'] = 'Tropics'
gdf_amazon['Region'] = 'Amazon'
gdf_congo['Region'] = 'Congo'
gdf_sea['Region'] = 'SEA'

df_all = pd.concat([gdf, gdf_amazon, gdf_congo, gdf_sea])

x1 = {'P':0, 'Dataset':'pr_cmorph'}
x2 = {'P':0, 'Dataset':'pr_cpc'}
x3 = {'P':0, 'Dataset':'pr_cru'}
x4 = {'P':0, 'Dataset':'pr_era'}
x5 = {'P':0, 'Dataset':'pr_gpcc'}
x6 = {'P':0, 'Dataset':'pr_gpcp'}
x7 = {'P':0, 'Dataset':'pr_gpm'}
x8 = {'P':0, 'Dataset':'pr_jra'}
x9 = {'P':0, 'Dataset':'pr_merra'}
x10 = {'P':0, 'Dataset':'pr_noaa'}
x11 = {'P':0, 'Dataset':'pr_per_cdr'}
x12 = {'P':0, 'Dataset':'pr_per_per'}
x13 = {'P':0, 'Dataset':'pr_trmm'}
x14 = {'P':0, 'Dataset':'pr_udel'}

x005_list = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]
x01_list = [x1,x2,x3,x5,x6,x8,x9,x10,x11,x12,x13,x14]
x025_list = [x2,x3,x6,x8,x9,x10,x11,x12,x13]
x05_list = [x9,x10]

# ax1...
ax1 = df_all.loc[(df_all['Region'] == 'Tropics') & (df_all['Res'] == '0.05')]
ax2 = df_all.loc[(df_all['Region'] == 'Tropics') & (df_all['Res'] == '0.1')]
ax3 = df_all.loc[(df_all['Region'] == 'Tropics') & (df_all['Res'] == '0.25')]
ax4 = df_all.loc[(df_all['Region'] == 'Tropics') & (df_all['Res'] == '0.5')]
ax5 = df_all.loc[(df_all['Region'] == 'Tropics') & (df_all['Res'] == '1.0')]
ax6 = df_all.loc[(df_all['Region'] == 'Tropics') & (df_all['Res'] == '2.0')]
ax7 = df_all.loc[(df_all['Region'] == 'Amazon') & (df_all['Res'] == '0.05')]
ax8 = df_all.loc[(df_all['Region'] == 'Amazon') & (df_all['Res'] == '0.1')]
ax9 = df_all.loc[(df_all['Region'] == 'Amazon') & (df_all['Res'] == '0.25')]
ax10 = df_all.loc[(df_all['Region'] == 'Amazon') & (df_all['Res'] == '0.5')]
ax11 = df_all.loc[(df_all['Region'] == 'Amazon') & (df_all['Res'] == '1.0')]
ax12 = df_all.loc[(df_all['Region'] == 'Amazon') & (df_all['Res'] == '2.0')]
ax13 = df_all.loc[(df_all['Region'] == 'Congo') & (df_all['Res'] == '0.05')]
ax14 = df_all.loc[(df_all['Region'] == 'Congo') & (df_all['Res'] == '0.1')]
ax15 = df_all.loc[(df_all['Region'] == 'Congo') & (df_all['Res'] == '0.25')]
ax16 = df_all.loc[(df_all['Region'] == 'Congo') & (df_all['Res'] == '0.5')]
ax17 = df_all.loc[(df_all['Region'] == 'Congo') & (df_all['Res'] == '1.0')]
ax18 = df_all.loc[(df_all['Region'] == 'Congo') & (df_all['Res'] == '2.0')]
ax19 = df_all.loc[(df_all['Region'] == 'SEA') & (df_all['Res'] == '0.05')]
ax20 = df_all.loc[(df_all['Region'] == 'SEA') & (df_all['Res'] == '0.1')]
ax21 = df_all.loc[(df_all['Region'] == 'SEA') & (df_all['Res'] == '0.25')]
ax22 = df_all.loc[(df_all['Region'] == 'SEA') & (df_all['Res'] == '0.5')]
ax23 = df_all.loc[(df_all['Region'] == 'SEA') & (df_all['Res'] == '1.0')]
ax24 = df_all.loc[(df_all['Region'] == 'SEA') & (df_all['Res'] == '2.0')]

# groupby to achieve the strip plot variant
ax1 = (ax1.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax2 = (ax2.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax3 = (ax3.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax4 = (ax4.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax5 = (ax5.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax6 = (ax6.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax7 = (ax7.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax8 = (ax8.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax9 = (ax9.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax10 = (ax10.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax11 = (ax11.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax12 = (ax12.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax13 = (ax13.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax14 = (ax14.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax15 = (ax15.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax16 = (ax16.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax17 = (ax17.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax18 = (ax18.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax19 = (ax19.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax20 = (ax20.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax21 = (ax21.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax22 = (ax22.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax23 = (ax23.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()
ax24 = (ax24.groupby(['Dataset', 'Cat'])['P'].median()).reset_index()

# define colour palette
palette = {'pr_chirps' : "coral", 'pr_per_ccs' : "coral", 'pr_per_ccscdr' : "coral", 'pr_per_cdr'  : "coral", 'pr_per_now' : "coral",
           'pr_per_per' : "coral",  'pr_gpm' : "coral",  'pr_trmm' : "coral",  'pr_cmorph' : "coral",  'pr_gpcp' : "coral",
           'pr_cpc' : "#F7D238",  'pr_cru' : "#F7D238",  'pr_gpcc' : "#F7D238",  'pr_udel' : "#F7D238",  'pr_era'  : "mediumaquamarine",
           'pr_merra'  : "mediumaquamarine", 'pr_noaa'  : "mediumaquamarine", 'pr_jra'  : "mediumaquamarine"}

# define dataset order on x axis
x_order = ['pr_chirps', 'pr_per_ccs', 'pr_per_ccscdr', 'pr_per_cdr', 'pr_per_now', 'pr_per_per',  'pr_gpm',  'pr_trmm',  'pr_cmorph',  'pr_gpcp',
           'pr_cpc',  'pr_cru',  'pr_gpcc',  'pr_udel',  'pr_era', 'pr_merra', 'pr_noaa', 'pr_jra']

# plotting
sns.set_context("paper", font_scale=3.0)

fig = plt.figure(figsize=(25,20))

g = fig.add_subplot(4, 6, 1)
g = sns.barplot(x="Dataset", y="P", data=ax1, palette=palette, ci=68, estimator=median, order=x_order)
sat_patch = matplotlib.patches.Patch(color='coral', label='Satellite')
stat_patch = matplotlib.patches.Patch(color='#F7D238', label='Station')
rean_patch = matplotlib.patches.Patch(color='mediumaquamarine', label='Reanalysis')
plt.legend(handles=[sat_patch, stat_patch, rean_patch],
           ncol=3, bbox_to_anchor=(0.8, 1.35), frameon=False,
           fontsize='medium')
g.axhline(0, alpha=0.4, c='k')
g.annotate('a)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 2)
g = sns.barplot(x="Dataset", y="P", data=ax2, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('b)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 3)
g = sns.barplot(x="Dataset", y="P", data=ax3, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('c)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 4)
g = sns.barplot(x="Dataset", y="P", data=ax4, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('d)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 5)
g = sns.barplot(x="Dataset", y="P", data=ax5, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('e)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 6)
g = sns.barplot(x="Dataset", y="P", data=ax6, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('f)', xy=(0.1, 0.9), xycoords="axes fraction")
g.annotate('Tropics', xy=(1.1, 0.3), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 7)
g = sns.barplot(x="Dataset", y="P", data=ax7, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('g)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
# g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)

g = fig.add_subplot(4, 6, 8)
g = sns.barplot(x="Dataset", y="P", data=ax8, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('h)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 9)
g = sns.barplot(x="Dataset", y="P", data=ax9, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('i)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 10)
g = sns.barplot(x="Dataset", y="P", data=ax10, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('j)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 11)
g = sns.barplot(x="Dataset", y="P", data=ax11, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('k)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 12)
g = sns.barplot(x="Dataset", y="P", data=ax12, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('l)', xy=(0.1, 0.9), xycoords="axes fraction")
g.annotate('Amazon', xy=(1.1, 0.3), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 13)
g = sns.barplot(x="Dataset", y="P", data=ax13, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('m)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
# g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 14)
g = sns.barplot(x="Dataset", y="P", data=ax14, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('n)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 15)
g = sns.barplot(x="Dataset", y="P", data=ax15, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('o)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 16)
g = sns.barplot(x="Dataset", y="P", data=ax16, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('p)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 17)
g = sns.barplot(x="Dataset", y="P", data=ax17, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('q)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 18)
g = sns.barplot(x="Dataset", y="P", data=ax18, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('r)', xy=(0.1, 0.9), xycoords="axes fraction")
g.annotate('Congo', xy=(1.1, 0.35), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 19)
g = sns.barplot(x="Dataset", y="P", data=ax19, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('s)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], fontsize=14)
# [1::2] means start from the second element in the list and get every other element
for tick in g.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
g.set(xlabel='0.05')
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 20)
g = sns.barplot(x="Dataset", y="P", data=ax20, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('t)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], fontsize=14)
# [1::2] means start from the second element in the list and get every other element
for tick in g.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
g.set(yticklabels=[])
g.set(xlabel='0.1')
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 21)
g = sns.barplot(x="Dataset", y="P", data=ax21, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('u)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], fontsize=14)
# [1::2] means start from the second element in the list and get every other element
for tick in g.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
g.set(yticklabels=[])
g.set(xlabel='0.25')
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 22)
g = sns.barplot(x="Dataset", y="P", data=ax22, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('v)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], fontsize=14)
# [1::2] means start from the second element in the list and get every other element
for tick in g.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
g.set(yticklabels=[])
g.set(xlabel='0.5')
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 23)
g = sns.barplot(x="Dataset", y="P", data=ax23, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('w)', xy=(0.1, 0.9), xycoords="axes fraction")
g.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], fontsize=14)
# [1::2] means start from the second element in the list and get every other element
for tick in g.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
g.set(yticklabels=[])
g.set(xlabel='1.0')
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 24)
g = sns.barplot(x="Dataset", y="P", data=ax24, palette=palette, ci=68, estimator=median, order=x_order)
g.axhline(0, alpha=0.4, c='k')
g.annotate('x)', xy=(0.1, 0.9), xycoords="axes fraction")
g.annotate('SEA', xy=(1.1, 0.4), xycoords="axes fraction", rotation=90, size=40)
g.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], fontsize=14)
# [1::2] means start from the second element in the list and get every other element
for tick in g.xaxis.get_major_ticks()[1::2]:
    tick.set_pad(20)
g.set(yticklabels=[])
g.set(xlabel='2.0')
g.set(ylabel=None)
g.set(ylim=(-1.1,1.1))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


# common axis labels
fig.supylabel('∆P (mm month$^{-1}$ %$^{-1}$)', x=0.07)
fig.supxlabel('Resolution (degrees)', y=0.02)
plt.tight_layout()

out_path = 'file_path'
fname = 'precip_barplot_all_datasets_5yr_3x3_03_17_numbered.jpg'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
