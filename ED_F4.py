#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read in dPdef, dPcont, dPdiff
Plot a 2x3 of 0.05 and 2 degrees for deforested, control and difference
"""

# read in packages
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

dp_def_005 = pd.read_csv(data_path + 'def_cont_0.05_5yr_3x3_03_17.csv')
dp_def_2 = pd.read_csv(data_path + 'def_cont_2.0_5yr_3x3_03_17.csv')

dp_cont_005 = pd.read_csv(data_path + 'def_cont_0.05_5yr_3x3_03_17.csv')
dp_cont_2 = pd.read_csv(data_path + 'def_cont_2.0_5yr_3x3_03_17.csv')

dp_diff_005 = pd.read_csv(data_path + 'diff_0.05_5yr_3x3_03_17.csv').drop(columns=['lat', 'lon']).stack().reset_index().rename(columns={'level_1': 'Dataset', 0:'P'})
dp_diff_2 = pd.read_csv(data_path + 'diff_2.0_5yr_3x3_03_17.csv').drop(columns=['lat', 'lon']).stack().reset_index().rename(columns={'level_1': 'Dataset', 0:'P'})

# deforested
# make the datasets for plotting...
x_005 = np.arange(2, 10, 2)
y_005 = np.arange(3, 11, 2)
x_2 = np.arange(2, 38, 2)
y_2 = np.arange(3, 39, 2)

col_x_005 = []
col_y_005 = []
col_x_2 = []
col_y_2 = []

for column in dp_def_005.columns[x_005]:
    col_x_005.append(column)
for column in dp_def_005.columns[y_005]:
    col_y_005.append(column)
for column in dp_def_2.columns[x_2]:
    col_x_2.append(column)
for column in dp_def_2.columns[y_2]:
    col_y_2.append(column)

# apply column list to dfs, stack, rename etc
dp_def_005 = ((dp_def_005[col_y_005]).stack().reset_index().rename(columns={'level_1': "Dataset", 0: "P"}))
dp_def_005.Dataset = dp_def_005.Dataset.str.rstrip("_defP")
dp_def_2 = ((dp_def_2[col_y_2]).stack().reset_index().rename(columns={'level_1': "Dataset", 0: "P"}))
dp_def_2.Dataset = dp_def_2.Dataset.str.rstrip("_defP")

# apply cat fn
dp_def_005['Cat'] = dp_def_005.apply(f, axis=1)
dp_def_2['Cat'] = dp_def_2.apply(f, axis=1)

# select only satellite
dp_def_005 = dp_def_005.loc[dp_def_005['Cat'] == 'Satellite']
dp_def_2 = dp_def_2.loc[dp_def_2['Cat'] == 'Satellite']


# control
# make the datasets for plotting...

# apply column list to dfs, stack, rename etc
dp_cont_005 = ((dp_cont_005[col_x_005]).stack().reset_index().rename(columns={'level_1': "Dataset", 0: "P"}))
dp_cont_005.Dataset = dp_cont_005.Dataset.str.rstrip("_contP")
dp_cont_2 = ((dp_cont_2[col_x_2]).stack().reset_index().rename(columns={'level_1': "Dataset", 0: "P"}))
dp_cont_2.Dataset = dp_cont_2.Dataset.str.rstrip("_contP")

# apply cat fn
dp_cont_005['Cat'] = dp_cont_005.apply(f, axis=1)
dp_cont_2['Cat'] = dp_cont_2.apply(f, axis=1)

# select only satellite
dp_cont_005 = dp_cont_005.loc[dp_cont_005['Cat'] == 'Satellite']
dp_cont_2 = dp_cont_2.loc[dp_cont_2['Cat'] == 'Satellite']


# differnce
# make the datasets for plotting...

# apply cat fn
dp_diff_005['Cat'] = dp_diff_005.apply(f, axis=1)
dp_diff_2['Cat'] = dp_diff_2.apply(f, axis=1)

# select only satellite
dp_diff_005 = dp_diff_005.loc[dp_diff_005['Cat'] == 'Satellite']
dp_diff_2 = dp_diff_2.loc[dp_diff_2['Cat'] == 'Satellite']


# plotting
print('plotting...')
fig = plt.figure(figsize=(6,6))
sns.reset_orig()

g = fig.add_subplot(3, 2, 1)
g = sns.histplot(data=dp_def_005, x="P")
g.set(xlabel=None)
g.set(ylabel=None)
g.set(xlim=(-100,100))
g.set(xticklabels=[])
# g.set_title('0.05 deg', fontsize=8)
g.annotate('a)', xy=(0.9, 0.9), xycoords="axes fraction")
plt.axvline(x=0, c='k', linestyle='--')

g = fig.add_subplot(3, 2, 2)
g = sns.histplot(data=dp_def_2, x="P")
g.set(ylabel=None)
g.set(xlabel=None)
g.set(xlim=(-100,100))
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_title('0.1 deg', fontsize =  8)
g.annotate('b)', xy=(0.9, 0.9), xycoords="axes fraction")
g.annotate('$ΔP_{def}$', xy=(1.1, 0.4), xycoords="axes fraction", rotation=90, size=12)
plt.axvline(x=0, c='k', linestyle='--')


g = fig.add_subplot(3, 2, 3)
g = sns.histplot(data=dp_cont_005, x="P")
g.set(xlabel=None)
g.set(ylabel=None)
g.set(xlim=(-100,100))
g.set(xticklabels=[])
# g.set_title('0.25 deg', fontsize =  8)
g.annotate('c)', xy=(0.9, 0.9), xycoords="axes fraction")
plt.axvline(x=0, c='k', linestyle='--')


g = fig.add_subplot(3, 2, 4)
g = sns.histplot(data=dp_cont_2, x="P")
g.set(xlabel=None)
g.set(ylabel=None)
g.set(xlim=(-100,100))
g.set(xticklabels=[])
g.set(yticklabels=[])
# g.set_title('0.5 deg', fontsize =  8)
g.annotate('d)', xy=(0.9, 0.9), xycoords="axes fraction")
g.annotate('$ΔP_{cont}$', xy=(1.1, 0.4), xycoords="axes fraction", rotation=90, size=12)
plt.axvline(x=0, c='k', linestyle='--')


g = fig.add_subplot(3, 2, 5)
g = sns.histplot(data=dp_diff_005, x="P")
# g.set(xlabel='Change in P')
g.set(ylabel=None)
g.set(xlabel=None)
g.set(xlim=(-100,100))
# g.set_title('1.0 deg', fontsize =  8)
g.annotate('e)', xy=(0.9, 0.9), xycoords="axes fraction")
plt.axvline(x=0, c='k', linestyle='--')


g = fig.add_subplot(3, 2, 6)
g = sns.histplot(data=dp_diff_2, x="P")
# g.set(xlabel='Change in P')
g.set(ylabel=None)
g.set(xlabel=None)
g.set(xlim=(-100,100))
g.set(yticklabels=[])
# g.set_title('2.0 deg', fontsize =  8)
g.annotate('f)', xy=(0.9, 0.9), xycoords="axes fraction")
g.annotate('$ΔP_{def} - ΔP_{cont}$', xy=(1.1, 0.2), xycoords="axes fraction", rotation=90, size=12)
plt.axvline(x=0, c='k', linestyle='--')

fig.supxlabel('ΔP (mm month$^{-1}$)')
fig.supylabel('Count')

plt.tight_layout()
print('saving hist')
out_path = 'file_path'
fname = 'distplot_combined_ED.jpg'
plt.savefig(out_path+fname, dpi=600, bbox_inches='tight')
