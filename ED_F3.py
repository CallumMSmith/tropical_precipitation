#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6*4 plot synthesising the supplementary barplots into one plot for the ED.
Only showing results at 2 degrees
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from itertools import cycle
from numpy import median
from numpy import mean
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
import matplotlib
import warnings
warnings.filterwarnings('ignore')
warnings.warn('CRS mismatch between the CRS of left geometries and the CRS of right geometries.')
warnings.warn('Do not show this message')


# define colour palette
palette = {"Satellite": "coral", "Station": "#F7D238", "Reanalysis": "mediumaquamarine"}

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

# read in shapefile
shp_path = 'file_path'

amazon_shp = gpd.read_file(shp_path + 'amazonia.shp')
congo_shp = gpd.read_file(shp_path + 'congo_shp_sept21.shp')
sea_shp = gpd.read_file(shp_path + 'sea_with_png.shp')

print('Reading in Amazon, Congo, SEA and Tropics Precip datasets...')

data_path = ('file_path')
out_path = 'file_path'

# 03-17 3yr 3x3
ds_03_17_3yr_3x3 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_3yr_3x3_03_17.nc').sel(lat=slice(-30,30)).sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_03_17_3yr_3x3 = ds_03_17_3yr_3x3.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Type='03_17_3yr_3x3')
df_03_17_3yr_3x3['Cat'] = df_03_17_3yr_3x3.apply(f, axis=1)

# 03-17 3yr 5x5
ds_03_17_3yr_5x5 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_3yr_5x5_03_17.nc').sel(lat=slice(-30,30)).sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_03_17_3yr_5x5 = ds_03_17_3yr_5x5.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Type='03_17_3yr_5x5')
df_03_17_3yr_5x5['Cat'] = df_03_17_3yr_5x5.apply(f, axis=1)

# 03-17 5yr 3x3
ds_03_17_5yr_3x3 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_17.nc').sel(lat=slice(-30,30)).sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_03_17_5yr_3x3 = ds_03_17_5yr_3x3.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Type='03_17_5yr_3x3')
df_03_17_5yr_3x3['Cat'] = df_03_17_5yr_3x3.apply(f, axis=1)

# 03-17 5yr 5x5
ds_03_17_5yr_5x5 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_5x5_03_17.nc').sel(lat=slice(-30,30)).sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_03_17_5yr_5x5 = ds_03_17_5yr_5x5.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Type='03_17_5yr_5x5')
df_03_17_5yr_5x5['Cat'] = df_03_17_5yr_5x5.apply(f, axis=1)

# 03-20 3yr 3x3
ds_03_20_3yr_3x3 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_3yr_3x3_03_20.nc').sel(lat=slice(-30,30)).sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_03_20_3yr_3x3 = ds_03_20_3yr_3x3.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Type='03_20_3yr_3x3')
df_03_20_3yr_3x3['Cat'] = df_03_20_3yr_3x3.apply(f, axis=1)

# 03-20 5yr 3x3
ds_03_20_5yr_3x3 = xr.open_dataset(data_path + 'delP_by_delCC_2.0deg_5yr_3x3_03_20.nc').sel(lat=slice(-30,30)).sel(lon=slice(-100,160)).to_dataframe().stack().reset_index()
df_03_20_5yr_3x3 = ds_03_20_5yr_3x3.rename(columns={'level_2': "Dataset", 0: "P"}).assign(Type='03_20_5yr_3x3')
df_03_20_5yr_3x3['Cat'] = df_03_20_5yr_3x3.apply(f, axis=1)

df_list = [df_03_17_3yr_3x3, df_03_17_3yr_5x5, df_03_17_5yr_3x3, df_03_17_5yr_5x5, df_03_20_3yr_3x3, df_03_20_5yr_3x3]

print('processing data...')
# convert pandas df to geopandas
gdf_list = []
for df in df_list:
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    type = str(gdf['Type'][0])

    # clip areas to shapefiles
    gdf_amazon = gdf.clip(amazon_shp)
    gdf_congo = gdf.clip(congo_shp)
    gdf_sea = gdf.clip(sea_shp)

    # concat the regions together and plot as one
    gdf['Region'] = 'Tropics'
    gdf_amazon['Region'] = 'Amazon'
    gdf_congo['Region'] = 'Congo'
    gdf_sea['Region'] = 'SEA'

    gdf_concat = pd.concat([gdf, gdf_amazon, gdf_congo, gdf_sea])
    gdf_concat = gdf_concat.drop(['lat', 'lon', 'geometry'], axis=1)
    gdf_list.append(gdf_concat)

gdf = pd.concat(gdf_list)

# find median values incase they need to be printed somewhere...
supp_table = gdf.groupby(['Cat', 'Type', 'Region'])['P'].median()
# print(supp_table)
supp_table.to_csv(out_path + 'precip_combined_ED_table_values.csv')


# make data for plotting
ax1 = gdf.loc[(gdf['Region'] == 'Tropics') & (gdf['Type'] == '03_17_3yr_3x3')]
ax2 = gdf.loc[(gdf['Region'] == 'Tropics') & (gdf['Type'] == '03_17_3yr_5x5')]
ax3 = gdf.loc[(gdf['Region'] == 'Tropics') & (gdf['Type'] == '03_17_5yr_3x3')]
ax4 = gdf.loc[(gdf['Region'] == 'Tropics') & (gdf['Type'] == '03_17_5yr_5x5')]
ax5 = gdf.loc[(gdf['Region'] == 'Tropics') & (gdf['Type'] == '03_20_3yr_3x3')]
ax6 = gdf.loc[(gdf['Region'] == 'Tropics') & (gdf['Type'] == '03_20_5yr_3x3')]

ax7 = gdf.loc[(gdf['Region'] == 'Amazon') & (gdf['Type'] == '03_17_3yr_3x3')]
ax8 = gdf.loc[(gdf['Region'] == 'Amazon') & (gdf['Type'] == '03_17_3yr_5x5')]
ax9 = gdf.loc[(gdf['Region'] == 'Amazon') & (gdf['Type'] == '03_17_5yr_3x3')]
ax10 = gdf.loc[(gdf['Region'] == 'Amazon') & (gdf['Type'] == '03_17_5yr_5x5')]
ax11 = gdf.loc[(gdf['Region'] == 'Amazon') & (gdf['Type'] == '03_20_3yr_3x3')]
ax12 = gdf.loc[(gdf['Region'] == 'Amazon') & (gdf['Type'] == '03_20_5yr_3x3')]

ax13 = gdf.loc[(gdf['Region'] == 'Congo') & (gdf['Type'] == '03_17_3yr_3x3')]
ax14 = gdf.loc[(gdf['Region'] == 'Congo') & (gdf['Type'] == '03_17_3yr_5x5')]
ax15 = gdf.loc[(gdf['Region'] == 'Congo') & (gdf['Type'] == '03_17_5yr_3x3')]
ax16 = gdf.loc[(gdf['Region'] == 'Congo') & (gdf['Type'] == '03_17_5yr_5x5')]
ax17 = gdf.loc[(gdf['Region'] == 'Congo') & (gdf['Type'] == '03_20_3yr_3x3')]
ax18 = gdf.loc[(gdf['Region'] == 'Congo') & (gdf['Type'] == '03_20_5yr_3x3')]

ax19 = gdf.loc[(gdf['Region'] == 'SEA') & (gdf['Type'] == '03_17_3yr_3x3')]
ax20 = gdf.loc[(gdf['Region'] == 'SEA') & (gdf['Type'] == '03_17_3yr_5x5')]
ax21 = gdf.loc[(gdf['Region'] == 'SEA') & (gdf['Type'] == '03_17_5yr_3x3')]
ax22 = gdf.loc[(gdf['Region'] == 'SEA') & (gdf['Type'] == '03_17_5yr_5x5')]
ax23 = gdf.loc[(gdf['Region'] == 'SEA') & (gdf['Type'] == '03_20_3yr_3x3')]
ax24 = gdf.loc[(gdf['Region'] == 'SEA') & (gdf['Type'] == '03_20_5yr_3x3')]


print('plotting barplot...')
sns.set_context("paper", font_scale=3.0)

fig = plt.figure(figsize=(25,20))

g = fig.add_subplot(4, 6, 1)
g = sns.barplot(x="Cat", y="P", data=ax1, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])
g.axhline(0, alpha=0.4, c='k')
g.annotate('a)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
plt.yticks([-1,0,1])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)
g.set_title('     03-17, 3yr, 3x3', fontsize=26)


g = fig.add_subplot(4, 6, 2)
g = sns.barplot(x="Cat", y="P", data=ax2, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('b)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)
g.set_title('     03-17, 3yr, 5x5', fontsize=26)


g = fig.add_subplot(4, 6, 3)
g = sns.barplot(x="Cat", y="P", data=ax3, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('c)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)
g.set_title('     03-17, 5yr, 3x3', fontsize=26)


g = fig.add_subplot(4, 6, 4)
g = sns.barplot(x="Cat", y="P", data=ax4, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])
g.axhline(0, alpha=0.4, c='k')
g.annotate('d)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)
g.set_title('     03-17, 5yr, 5x5', fontsize=26)


g = fig.add_subplot(4, 6, 5)
g = sns.barplot(x="Cat", y="P", data=ax5, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])
g.axhline(0, alpha=0.4, c='k')
g.annotate('e)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)
g.set_title('     03-20, 3yr, 3x3', fontsize=26)


g = fig.add_subplot(4, 6, 6)
g = sns.barplot(x="Cat", y="P", data=ax6, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('f)', xy=(0.1, 0.8), xycoords="axes fraction")
g.annotate('Tropics', xy=(1.1, 0.3), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)
g.set_title('     03-20, 5yr, 3x3', fontsize=26)


g = fig.add_subplot(4, 6, 7)
g = sns.barplot(x="Cat", y="P", data=ax7, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('g)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
plt.yticks([-1,0,1])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 8)
g = sns.barplot(x="Cat", y="P", data=ax8, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('h)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 9)
g = sns.barplot(x="Cat", y="P", data=ax9, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('i)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 10)
g = sns.barplot(x="Cat", y="P", data=ax10, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('j)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 11)
g = sns.barplot(x="Cat", y="P", data=ax11, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('k)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 12)
g = sns.barplot(x="Cat", y="P", data=ax12, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('l)', xy=(0.1, 0.8), xycoords="axes fraction")
g.annotate('Amazon', xy=(1.1, 0.3), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 13)
g = sns.barplot(x="Cat", y="P", data=ax13, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('m)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
plt.yticks([-1,0,1])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 14)
g = sns.barplot(x="Cat", y="P", data=ax14, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('n)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 15)
g = sns.barplot(x="Cat", y="P", data=ax15, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('o)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 16)
g = sns.barplot(x="Cat", y="P", data=ax16, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('p)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 17)
g = sns.barplot(x="Cat", y="P", data=ax17, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('q)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 18)
g = sns.barplot(x="Cat", y="P", data=ax18, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('r)', xy=(0.1, 0.8), xycoords="axes fraction")
g.annotate('Congo', xy=(1.1, 0.35), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.set(xlabel=None)
g.set(ylabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 19)
g = sns.barplot(x="Cat", y="P", data=ax19, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])
g.axhline(0, alpha=0.4, c='k')
g.annotate('s)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
plt.yticks([-1,0,1])
# g.set_xlabel('     03-17, 3yr, 3x3', fontsize=26)
g.set(ylabel=None)
g.set(xlabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False)
sns.despine(right=True, top=True)
sat_patch = matplotlib.patches.Patch(color='coral', label='Satellite')
stat_patch = matplotlib.patches.Patch(color='#F7D238', label='Station')
rean_patch = matplotlib.patches.Patch(color='mediumaquamarine', label='Reanalysis')
plt.legend(handles=[sat_patch, stat_patch, rean_patch],
           ncol=3, bbox_to_anchor=(3.8, 0), frameon=False,
           fontsize='medium')


g = fig.add_subplot(4, 6, 20)
g = sns.barplot(x="Cat", y="P", data=ax20, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('t)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_xlabel('     03-17, 3yr, 5x5', fontsize=26)
g.set(ylabel=None)
g.set(xlabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 21)
g = sns.barplot(x="Cat", y="P", data=ax21, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('u)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_xlabel('     03-17, 5yr, 3x3', fontsize=26)
g.set(ylabel=None)
g.set(xlabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 22)
g = sns.barplot(x="Cat", y="P", data=ax22, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('v)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_xlabel('     03-17, 5yr, 5x5', fontsize=26)
g.set(ylabel=None)
g.set(xlabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 23)
g = sns.barplot(x="Cat", y="P", data=ax23, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])

g.axhline(0, alpha=0.4, c='k')
g.annotate('w)', xy=(0.1, 0.8), xycoords="axes fraction")
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_xlabel('     03-20, 3yr, 3x3', fontsize=26)
g.set(ylabel=None)
g.set(xlabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


g = fig.add_subplot(4, 6, 24)
g = sns.barplot(x="Cat", y="P", data=ax24, palette=palette, ci=68, estimator=median, order=['Satellite', 'Station', 'Reanalysis'])
g.axhline(0, alpha=0.4, c='k')
g.annotate('x)', xy=(0.1, 0.8), xycoords="axes fraction")
g.annotate('SEA', xy=(1.1, 0.4), xycoords="axes fraction", rotation=90, size=40)
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_xlabel('     03-20, 5yr, 3x3', fontsize=26)
g.set(ylabel=None)
g.set(xlabel=None)
g.set(ylim=(-1.4, 1.4))
g.tick_params(bottom=False, left=False)
sns.despine(right=True, top=True)


# common axis labels
fig.supylabel('∆P (mm month$^{-1}$ %$^{-1}$)', x=0.07)
plt.tight_layout()

fname = 'precip_barplot_combined_ED.jpg'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
