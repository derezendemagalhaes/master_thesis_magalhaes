from dateutil.relativedelta import *
from datetime import datetime
import dask.dataframe as dd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

import ee
import sys
import os
from shapely.geometry import Polygon, Point

def merge_df_nmonths(directory, start_date, nmonths:int, file_pattern):
    
    # Select nmonths for time series
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = start_date+relativedelta(months=+nmonths)
    
    df_list = []

    # Create dask dataframe for data between desired nmonths
    df_list = map(lambda file_name: dd.read_hdf(f'{directory}{file_name}*', '*').loc[lambda df: df.Date.between(f'{start_date}', f'{end_date}')], file_pattern)
   
    # Compute the results using multiple processes
    df_list = [df.compute(scheduler='processes') for df in df_list]

    return (df_list)


def process_dataframes(dataframes, start_value, end_value, start_value_, end_value_):
    subset_dfs = []
    for df in dataframes:
        # select the rows of the DataFrame where the values in 'col' are within the current pair of values
        df_subset = df[(df['easting_sq'] >= start_value) & (df['easting_sq'] < end_value)]
        df_subset = df_subset[(df_subset['northing_sq'] >= start_value_) & (df_subset['northing_sq'] < end_value_)]
        df_subset = df_subset.loc[:, ~df_subset.columns.duplicated(keep='first')]
        subset_dfs.append(df_subset)

    return subset_dfs


def Plot_df_x_nmonths_window(directory, start_date, end_date, nmonths:int, file_pattern, window_size:int):
    
    # Parse the start date as a datetime object
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Add the number of months to the start date until the new date is before the end date
    while start_date < end_date:
        start_date += relativedelta(months=nmonths)
        # Convert the datetime object to a string in the specified format
        date_str = start_date.strftime('%Y-%m-%d')
        #function to merge all data into df and select nmonths required
        df_Grass_raw, df_TOC_raw = merge_df_nmonths(directory,
                                                    start_date=date_str,
                                                    nmonths=nmonths,
                                                    file_pattern=file_pattern)
        
        if not df_Grass_raw.empty:
            # Find the minimum and maximum values of easting_sq and northing_sq
            min_easting_sq = df_Grass_raw['easting_sq'].min()
            max_easting_sq = df_Grass_raw['easting_sq'].max()
            min_northing_sq = df_Grass_raw['northing_sq'].min()
            max_northing_sq = df_Grass_raw['northing_sq'].max()

            east = list(range(int(min_easting_sq), int(max_easting_sq+1), window_size))
            north = list(range(min_northing_sq, max_northing_sq+1, window_size))

            for e in range(len(east)-1):
                for n in range(len(north)-1):

                    # extract the current pair of values from the list
                    start_value = east[e]
                    end_value = east[e+1]
                    start_value_ = north[n]
                    end_value_ = north[n+1]

                    dataframes = [df_Grass_raw, df_TOC_raw]
                    df_Grass, df_TOC = process_dataframes(dataframes,
                                                          start_value,
                                                          end_value, 
                                                          start_value_, 
                                                          end_value_)

                    df_TOC_mean = df_TOC.groupby('Date')['TOC_Height'].mean().reset_index()
                    df_Grass_mean = df_Grass.groupby('Date')['Grass_Height'].mean().reset_index()

                    if len(df_TOC_mean) > 2:
                        sns.set(style="darkgrid")
                        fig, axs = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [0.8, 3]})

                        # Boxplot
                        bp = sns.boxplot(data=[df_TOC['TOC_Height'], df_Grass['Grass_Height']], ax=axs[0], width=0.3)
                        axs[0].grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
                        axs[0].tick_params(axis='both', which='major', labelsize=12)
                        axs[0].set_ylabel('Height', fontsize=12)
                        bp.set_xticklabels(['TOC', 'Grass'])
                        ylim = bp.get_ylim() 

                        # Time-serie
                        sns.lineplot(data=df_TOC_mean, x='Date', y='TOC_Height', ax=axs[1], label='TOC', ci='sd')
                        sns.lineplot(data=df_Grass_mean, x='Date', y='Grass_Height', ax=axs[1], label='Grass', ci='sd')
                        sns.scatterplot(data=df_TOC_mean, x='Date', y='TOC_Height', ax=axs[1], s=30)
                        sns.scatterplot(data=df_Grass_mean, x='Date', y='Grass_Height', ax=axs[1], s=30)
                        axs[1].grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
                        axs[1].tick_params(axis='both', which='major', labelsize=12)
                        axs[1].set_xlabel('Date', fontsize=12)
                        axs[1].set_ylabel('Height', fontsize=12)
                        axs[1].legend(fontsize=12)
                        axs[1].set_ylim(ylim)

                        fig.suptitle(f'Top of Canopy and Grass Height from: {start_date} +{nmonths} months. Easting: {start_value}000-{end_value}000, Northing: {start_value_}000-{end_value_}000', fontsize=14)
                        plt.show()
                        
#                         print(f"Subset between {start_value} and {end_value}/{start_value_} and {end_value_}: {len(df_TOC_mean)} rows")

def rescale(image):
    scale_factor = 1
    date = image.get('system:time_start')
    return image.multiply(scale_factor).set('system:time_start', date)


def createTS(image):
    date = image.get('system:time_start')
    value = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=ROI).get(var)
    std = image.reduceRegion(reducer=ee.Reducer.stdDev(), geometry=ROI).get(var)
    ft = ee.Feature(None, {'date': ee.Date(date).format('Y/M/d'), var: value, 'STD': std})
    return ft


def TS_to_pandas(TS):
    dump = TS.getInfo()
    fts = dump['features']
    out_vals = np.empty((len(fts)))
    out_dates = []
    out_std = np.empty((len(fts)))
    
    for i, f in enumerate(fts):
        props = f['properties']
        date = props['date']
        val = props[var]
        std = props['STD']
        out_vals[i] = val
        out_std[i] = std
        out_dates.append(pd.Timestamp(date))
    
    ser = pd.Series(out_vals, index=out_dates)
    return ser, out_std


def gee_geometry_from_shapely(geom, crs='epsg:4326'):
    """ 
    Simple helper function to take a shapely geometry and a coordinate system and convert them to a 
    Google Earth Engine Geometry.
    """
    from shapely.geometry import mapping
    ty = geom.type
    if ty == 'Polygon':
        return ee.Geometry.Polygon(ee.List(mapping(geom)['coordinates']), proj=crs, evenOdd=False)
    elif ty == 'Point':
        return ee.Geometry.Point(ee.List(mapping(geom)['coordinates']), proj=crs, evenOdd=False)    
    elif ty == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(ee.List(mapping(geom)['coordinates']), proj=crs, evenOdd=False)
