from dateutil.relativedelta import *
from shapely.geometry import Point
from datetime import datetime
import dask.dataframe as dd
import geopandas as gpd
import itertools
import rasterio
from rasterio.features import rasterize
from scipy.spatial import cKDTree

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


def convert_to_geodataframe(dataframes):
    """
    Function to convert dataframes list into geodataframes.
    """
    geodataframes = []
    
    for df in dataframes:
        geometry = [Point(xy) for xy in zip(df['Easting'], df['Northing'])]
        crs = 'epsg:32733'
        geodf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        geodataframes.append(geodf)
        
    return geodataframes



def convert_gdf_to_raster(dataframes, output_filename, cell_size, value_column):
    # Define the resolution and extent of the output raster
    xmin, ymin, xmax, ymax = dataframes[0].total_bounds
    width = int((xmax - xmin) / cell_size)
    height = int((ymax - ymin) / cell_size)
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    for df in dataframes:
        # Get the values to assign to the raster
        values = df[value_column]

        # Rasterize the point data with the specified values
        features = ((geom, value) for geom, value in zip(df['geometry'], values))
        raster = rasterize(shapes=features, out_shape=(height, width), transform=transform)

        # Write the raster to a file
        with rasterio.open(output_filename, 'w', driver='GTiff', width=width, height=height,
                           count=1, dtype=rasterio.float32, nodata=0, transform=transform) as dst:
            dst.write(raster, indexes=1)



def rescale(image):
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
    
    


def kdtree_comparison(point_gdf, polygon_gdf, polygon_column_name, new_column_name):
    # Create a k-d tree from the polygons in the polygon GeoDataFrame
    polygon_tree = cKDTree(np.array([np.array(x.centroid.coords)[0] for x in polygon_gdf.geometry]))

    # Find the nearest polygon to each point in the point GeoDataFrame
    distances, indices = polygon_tree.query(np.array([np.array(x.coords)[0] for x in point_gdf.geometry]), k=1)

    # Add a new column to the point GeoDataFrame with the value from the nearest polygon
    point_gdf[new_column_name] = polygon_gdf.iloc[indices][polygon_column_name].values

    # Return the updated point GeoDataFrame
    return point_gdf


    
