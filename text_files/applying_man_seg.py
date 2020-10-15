import numpy as np

import numba
import pandas

import skimage
import skimage.io
import skimage.filters
import skimage.segmentation
import skimage.measure
import skimage.morphology
import skimage.feature
import bi1x

import pandas as pd

import imageio

import os
import colorcet

import osgeo
from osgeo import ogr

import bokeh.io
import bokeh.plotting
import bokeh.palettes as bp
import bokeh.models
import sys

bokeh.io.output_notebook()

notebook_url = 'localhost:8888'

import max_int_projection
import auto_segmentation







def read_csv_coords(input_path):
    '''Takes an input path where a csv is located and returns a dataframe containing the 
    contents of the csv. Takes the column names of the csv and converts them into coordinates.
    
    Returns a dataframe of all the coordinates.'''
    
    # Taking data from column names, saving it, and renaming the columns x and y
    csv_df = pd.read_csv(input_path)
    x_coord1_str = csv_df.columns[0]
    x_coord1_float = float(x_coord1_str)
    y_coord1_str = csv_df.columns[1]
    y_coord1_float = float(y_coord1_str)
    csv_df_updated = csv_df.rename(columns={x_coord1_str: 'x', y_coord1_str: 'y'})
    
    # Reindexing the dataframe
    index_lst = []
    for i in range(csv_df_updated.shape[0]):
        index_lst.append(i+1)
    index_df = pd.DataFrame({'index': index_lst}, dtype=np.int8)
    csv_df_updated = pd.concat([index_df, csv_df_updated], axis=1)
    csv_df_updated = csv_df_updated.set_index('index')
    
    # Creating a dataframe from the first coordinate values
    coord_dct = {'x': x_coord1_float, 'y': y_coord1_float}
    coord_df = pd.DataFrame(coord_dct, index=[0])
    merged_df = pd.concat([coord_df, csv_df_updated])
    
    return merged_df






def convert_to_coord_array(df):
    '''Takes a pandas dataframe of coordinates with the columns x and y
    and converts it into a numpy array of coordinates.'''
    coord = []
    coord_lst = []
    x_df = df['x']
    y_df = df['y']
    for i in range(df.shape[0]-1):
        coord = [x_df[i], y_df[i]]
        coord_lst.append(coord)
    return np.array(coord_lst)






def produce_roi(coord_array, im_array):
    '''Takes in a pandas dataframe of coordinates with the columns x and y.
    
    Uses a package written by Justin Bois for Bi1x
    
    Connects the coordinates in order to form a polygon. Returns a ring 
    connecting all of the points.'''
    shape = im_array.shape
    roi, _, _ = bi1x.image.verts_to_roi(coord_array, shape[0], shape[1])
    return roi*1






def add_roi_vals(im_array, roi_array, val):
    '''Takes in an input of an image and roi array and a value. 
    Assigns the value to coordinates where the roi exists to differentiate
    rois from each other. Applies this to the third layer of an RGB image.
    Removes manually segmented cells from second layer of RGB image.'''
    shape = roi_array.shape
    im_array = im_array.astype('int32')
    for x in range(shape[0]):
        for y in range(shape[1]):
            if roi_array[x][y] == 1:
                im_array[x][y] = val
            else:
                im_array[x][y] = 0
    return im_array






def make_empty_array_2d(shape):
    '''Makes and empty numpy array of integers of a specified shape. Works for 3D arrays'''
    empty_array = np.empty(shape, dtype=np.int16)
    for x in range(shape[0]):
        for y in range(shape[1]):
            empty_array[x][y] = 0
            empty_array[x][y] = 0
            empty_array[x][y] = 0
    return empty_array





def make_empty_array_3d(shape):
    '''Makes and empty numpy array of integers of a specified shape. Works for 3D arrays'''
    empty_array = np.empty(shape, dtype=np.int16)
    for x in range(shape[0]):
        for y in range(shape[1]):
            empty_array[x][y][0] = 0
            empty_array[x][y][1] = 0
            empty_array[x][y][2] = 0
    return empty_array






def convert_to_rgb(im_2d):
    '''Takes an input of a 2-dimensional image and converts it into a 3-dimensional image with
    the third dimension being RGB values for the individual pixel. Pixel values will be stored
    in the R layer of the RGB structure and preserved from the original image.
    
    Returns a numpy array of an image in a 3-dimensional structure. The x and y dimensions are 
    the same as the input image array.'''
    print(im_2d.shape)
    (x_shape, y_shape) = im_2d.shape
    rgb_shape = (x_shape, y_shape, 3)
    empty_array = make_empty_array_3d(rgb_shape)
    for x in range(x_shape):
        for y in range(y_shape):
            pixel_val = im_2d[x][y]
            if pixel_val != 0:
                empty_array[x][y][0] += pixel_val
    return empty_array





def remove_small_objects(seg_im, min_area):
    '''Removes objects below a certain input minimum area in an image array.
    Returns an image array with the small objects removed.'''
    labeled_im = skimage.measure.label(seg_im)
    
    for region in skimage.measure.regionprops(labeled_im):
        for coord in region.coords:
            (x,y,z) = coord
            if region.area < min_area:
                labeled_im[x][y][0] = 0
                labeled_im[x][y][1] = 0
                labeled_im[x][y][2] = 0
            else:
                labeled_im[x][y][0] = seg_im[x][y][0]
                labeled_im[x][y][1] = seg_im[x][y][1]
                labeled_im[x][y][2] = 0
    return labeled_im






def apply_man_seg(im_input_path,
                  csv_input_path,
                  output_path, 
                  save_ims=True,
                  min_area=2000):
    '''Takes an string of an input path of segmented images only containing automatically segmented images
    and a string of an input path of csv files with coordinates created when manually segmenting cell clusters
    
    Uses the coordinates to produce a polygon and adds it to the image array with only automatically segmented images.
    Returns an image array with the automatically and manually segmented cells. Saves the images to the designated 
    output path.'''
    im_with_rois_dct = {}
    for im in os.listdir(im_input_path):
        im_path_name = im.split('.')
        
        if len(im_path_name) >= 2:
            # Check if image is a tiff and splitting name to get image details
            if im_path_name[1] == 'tiff' or im_path_name[1] == 'tif':
                im_array = skimage.io.imread(im_input_path + '/' + im)
                im_path_lst = im_path_name[0].split('/')
                im_name_ind_lst = im_path_lst[len(im_path_lst)-1].split('_')
                im_name_ind = im_name_ind_lst[len(im_name_ind_lst)-1]

                # Initiating array dictionary and array to combine manually segmented cells
                im_roi_dct = {}
                all_man_cells_array = make_empty_array_2d(im_array.shape)


                # Finding matching CSVs for the image
                i = 1
                for csv in os.listdir(csv_input_path):
                    csv_path_name = csv.split('.')
                    if csv_path_name[1] == 'csv':
                        if im_name_ind in csv:
                            csv_df = read_csv_coords(csv_input_path + '/' + csv)
                            coord_array = convert_to_coord_array(csv_df)
                            roi = produce_roi(coord_array, im_array)
                            im_with_rois = add_roi_vals(im_array, roi, i)
                            im_small_objects_removed = remove_small_objects(im_with_rois, min_area=min_area)
                            i += 1
                            im_name = im_path_lst[len(im_path_lst)-1]
                            cell_im_name = 'cell{}_'.format(i) + im_name
                            im_roi_dct[cell_im_name] = im_small_objects_removed
                for key in im_roi_dct:
                    all_man_cells_array += im_roi_dct[key]
                im_with_rois_dct[im_name] = all_man_cells_array
                if save_ims:
                        im_dir = output_path + '/' + im_name + '.tiff'
                        auto_segmentation.save_images(im_dir, all_man_cells_array)
    return im_with_rois_dct
