"""
plot scatter of dP (mm/month) vs dCC

USING EQUAL BINS METHOD: order the data by CC loss, then split into 5 with bin 1 would have the first 20% of the data, bin 2 the second 20% etc.
Then plot dP (mm/month) against the median CC loss of each bin.
want to plot the median of those bins, so, do pandas groupby median
For satellite datasets
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from itertools import cycle
import glob
from numpy import median
import xarray as xr
import geopandas as gpd
import string

# with 20% bin width
def bin_method(df, width):
    bin = []
    if width == '20':
        for row in df['tree']:
            # if more than a value,
            if row > 80:
                bin.append(90)
            elif row > 60:
                bin.append(70)
            elif row > 40:
                bin.append(50)
            elif row > 20:
                bin.append(30)
            else:
                bin.append(10)
        df['bin'] = bin

# categorise datasets
def f(row):
    val = []
    if row['ds'] == 'pr_cmorph' or row['ds'] == 'pr_per_ccs' or row['ds'] == 'pr_per_ccscdr' or row['ds'] == 'pr_per_cdr' or row['ds'] == 'pr_per_now' or row['ds'] == 'pr_per_per' or row['ds'] == 'pr_chirps' or row['ds'] == 'pr_gpcp' or row['ds'] == 'pr_gpm' or row['ds'] == 'pr_trmm':
        val = 'Satellite'
    elif row['ds'] == 'pr_cpc' or row['ds'] == 'pr_cru' or row['ds'] == 'pr_gpcc' or row['ds'] == 'pr_udel':
        val = 'Station'
    elif row['ds'] == row['ds'] == 'pr_era' or row['ds'] == 'pr_merra' or row['ds'] == 'pr_ncep' or row['ds'] == 'pr_noaa' or row['ds'] == 'pr_jra':
        val = 'Reanalysis'
    return val

# read in analysis output as well as cc then plot cc binned cc on x and dP on y
p_path = 'file_path'
cc_path = 'file_path'
out_path = 'file_path'

# read in shapefile
shp_path = 'file_path'
# clip areas
amazon_shp = gpd.read_file(shp_path + 'Amazon_biome/amazonia.shp')
congo_shp = gpd.read_file(shp_path + 'congo_shp_sept21.shp')
sea_shp = gpd.read_file(shp_path + 'sea_with_png.shp')

# bin width
width = '20'


cc = xr.open_dataset(cc_path + 'treeCoverChange_tropics_2.0_2017_v1.9_bilinear.nc')
cc = cc.rename({'longitude': 'lon','latitude': 'lat'})
cc = cc.sel(lat=slice(-30, 30))
cc = cc.to_dataframe()
cc = cc.drop(columns=['time'])
cc = cc['tree'] * -1

ds_p = pd.read_csv(p_path + 'diff_2.0_5yr_3x3_03_17.csv').set_index(['lat', 'lon'])

# concat the ds
df = pd.concat([ds_p, cc], axis=1).dropna().reset_index()

# convert pandas df to geopandas
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)

# clip geopandas to regions
gdf_amazon = gdf.clip(amazon_shp)
gdf_congo = gdf.clip(congo_shp)
gdf_sea = gdf.clip(sea_shp)

# concat the regions together and plot as one
gdf['Region'] = 'Tropics'
gdf_amazon['Region'] = 'Amazon'
gdf_congo['Region'] = 'Congo'
gdf_sea['Region'] = 'SEA'


df_all = [gdf, gdf_amazon, gdf_congo, gdf_sea]
df_list = []

for item in df_all:
    df = item.drop(columns=['lat','lon', 'geometry'])

    # sort df by tree cover values
    df = df.sort_values(by=['tree'])
    # find length of df, to enable splitting into equal groups
    len = df.shape[0]
    # assign new empty column, then select rows based on index and assign a bin value
    df['bin'] = ''
    df.iloc[0:round(len*0.2), df.columns.get_loc('bin')] = 1
    df.iloc[round(len*0.2):round(len*0.4), df.columns.get_loc('bin')] = 2
    df.iloc[round(len*0.4):round(len*0.6), df.columns.get_loc('bin')] = 3
    df.iloc[round(len*0.6):round(len*0.8), df.columns.get_loc('bin')] = 4
    df.iloc[round(len*0.8):, df.columns.get_loc('bin')] = 5

    df_list.append(df)

df = pd.concat(df_list)


df = df.set_index(['bin', 'Region', 'tree']).stack().reset_index().rename(columns={'level_3':'ds', 0:'P'})
df['Cat'] = df.apply(f, axis=1)
df_group = df.groupby(['bin', 'Region'])['tree'].median()
df_dict = df_group.to_dict()
df['mapped'] = df.set_index(['bin', 'Region']).index.map(df_dict.get)

# select only satellite ds
df = df.loc[df.Cat == 'Satellite']
df = df.loc[df.Region == 'Tropics']

df = df.drop(columns=['bin', 'Region', 'tree', 'ds', 'Cat'])

plotting_var = df.groupby(['Res', 'mapped'])['P'].median().reset_index()
plotting_err = df.groupby(['Res', 'mapped'])['P'].sem().reset_index()

var_2 = plotting_var.loc[plotting_var.Res == '2.0']
err_2 = plotting_err.loc[plotting_err.Res == '2.0']

x = var_2['mapped'].values
y = var_2['P'].values


fig = plt.figure(figsize=(6,6))
fig.add_subplot(1,1,1)
plt.errorbar(var_2['mapped'], var_2['P'], yerr=err_2['P'], fmt="o")
plt.xlabel('Forest cover change (%)')
plt.ylabel('∆P (mm month$^{-1}$)')
plt.axhline(y=0, ls='-', c='k', alpha=0.4)

fname = '/fname.jpg'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
