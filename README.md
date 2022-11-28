# tropical_precipitation
Code to analyse the change in precipitation due tropical forest loss. This analysis forms the publication "Tropical forest loss causes large reductions in observed precipitation", which has been accepted for publication in Nature. Addiontially, source data for supplementary figures.

The files in this repository take freely available precipitation, land cover and forest loss datasets and analyse the impact of tropical forest loss on these precipitation datasets. Firstly the precipitation, land cover and forest loss datasets are processed and regridded to a range of spatial scales (regrid_all_datasets.py). The datasets are then analysed to observe the changes in precipitation over time and to compare deforested regions with nearby forest (analysis_script.py). Lastly these results are plotted in a series of ways to explore these changes (F*plotting.py and ED*.py).

Source data is available via the publication for the main and extended data figures, whilst supplementary source data is available here only.
