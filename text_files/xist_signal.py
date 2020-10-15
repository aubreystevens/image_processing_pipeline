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

import imageio

import os
import colorcet

import bokeh.io
import bokeh.plotting
import bokeh.palettes as bp
import bokeh.models

bokeh.io.output_notebook()

notebook_url = 'localhost:8888'

import max_int_projection
import auto_segmentation
import applying_man_seg







def segment_xist(im, channel, connectivity=10, mode='inner', buffer_size=10):
    '''Takes in an input of an image array as a numpy array for the xist channel
    and the name of the channel in the format c#. 
    
    Returns an image array as a numpy array that has been median filtered, thresholded,
    and segmented.'''
    seg_im = auto_segmentation.mark_boundaries(im, 
                                               connectivity=connectivity, 
                                               mode=mode,
                                               buffer_size=buffer_size)
    return seg_im






def make_xist_2d(labeled_im_xist):
    '''Takes an input of a 3d labeled image of XIST using imageprops and returns a 2d version.'''
    for region in skimage.measure.regionprops(labeled_im_xist):
        for coord in region.coords:
            (x,y,z) = coord
            labeled_im_xist[x][y][0] = 0
            labeled_im_xist[x][y][1] = 1
            labeled_im_xist[x][y][2] = 0
    return labeled_im_xist




def make_empty_array_3d(shape):
    '''Makes and empty numpy array of integers of a specified shape. Works for 3D arrays'''
    empty_array = np.empty(shape, dtype=np.int16)
    for x in range(shape[0]):
        for y in range(shape[1]):
            empty_array[x][y][0] = 0
            empty_array[x][y][1] = 0
            empty_array[x][y][2] = 0
    return empty_array





def combine_xist_dapi(im_xist, im_dapi, min_area):
    '''Takes an input of an image array as a numpy array segmented in the 
    DAPI channel and a numpy array segmented in the XIST channel and removes
    XIST signal not within cells. Also compresses the DAPI channel into 2D 
    and merges the DAPI and XIST images into different layers of an RGB image.
    
    Returns a tuple of a numpy array of cells positive for XIST signal, an array of cells
    negative for XIST, and a tuple of the number of xist positive cells in the image and 
    the number of xist negative cells in the image.'''
    labeled_im_xist = skimage.measure.label(im_xist)
    labeled_im_dapi = skimage.measure.label(im_dapi)
    
    # Initialize dictionary for area counts
    xist_cell_size_ratio_dct = {}
    num_xist_pos = 0
    num_xist_neg = 0
    
    # Making XIST im 2D and create image array to store cells negative for XIST
    labeled_im_xist = make_xist_2d(labeled_im_xist)
    shape = im_xist.shape
    xist_false_array = make_empty_array_3d(shape)
            
    # Make DAPI 2D and add Xist onto layer of RGB image
    for region in skimage.measure.regionprops(labeled_im_dapi):
        
        # Initialize area counts for XIST in a cell, restarts with each new cell
        xist_area_count = 0
        dapi_area_count = region.area
        for coord in region.coords:
            (x, y, z) = coord
            if labeled_im_xist[x][y][1] == 1:
                xist_area_count += 1
                
        # Checking if cells have a XIST signal and separating cells into separate images based on that
        if xist_area_count / dapi_area_count < min_area:
            num_xist_neg += 1
            for coord in region.coords:
                (x, y, z) = coord
                labeled_im_dapi[x][y][0] = 0
                labeled_im_dapi[x][y][1] = 0
                labeled_im_dapi[x][y][2] = 0
                xist_false_array[x][y][0] = im_dapi[x][y][0]
                if labeled_im_xist[x][y][1] == 1:
                    xist_false_array[x][y][1] = 1
                else:
                    xist_false_array[x][y][1] = 0
        else:
            num_xist_pos += 1
            for coord in region.coords:
                (x, y, z) = coord
                labeled_im_dapi[x][y][0] = im_dapi[x][y][0]
                labeled_im_dapi[x][y][2] = 0
                if labeled_im_xist[x][y][1] == 1:
                    labeled_im_dapi[x][y][1] = 1
                else:
                    labeled_im_dapi[x][y][1] = 0
    return (labeled_im_dapi, xist_false_array, (num_xist_pos, num_xist_neg))









def apply_xist_signal(input_path, 
                      output_path,
                      xist_channel,
                      dapi_channel,
                      path_type='folder',
                      save_ims=True,
                      min_area=0.05,
                      connectivity=10,
                      mode='inner', 
                      buffer_size=10,
                      min_cell_area=2000):
    '''Inputs:
    -input_path is the path to where your file is. Input as a string
    -output_path is where you want outputs to be sent
    -channel is the channel XIST is in in the form "c#"
    -path_type is the form of whether the file you are calling is an "image" or a "folder" of images,
    default is "folder"
    -save_ims is whether you want the arrays to be saved as images to the output path, default is True
    -min_area_ratio is the minimum area XIST needs to cover compared to the cell size to be considered a XIST cloud
     default is 0.1
    -connectivity, mode, and buffer are all characteristics used for the segmentation functions. See the segmentation
     file for further detail.
    
    This either takes an image or iterates through a folder of images and compresses the XIST and DAPI images into
    a single RGB channel (DAPI in channel 0 and XIST in channel 2). Then, combines the images, removes 
    non-cellular XIST, determines if there is enough XIST in a cell to be considered a signal, and 
    separates cells with a positive XIST signal and those with a negative signal into two separate images.
    
    Outputs:
    -A dictionary of image names as keys and tuple of image arrays for both the positive and negative XIST signal
    as values
    -Saves images to the output path if save_ims is True
    '''
    pos_im_dct = {}
    neg_im_dct = {}
    xist_counts = {}
    
    # If you want to iterate through a folder of images
    if path_type == 'folder':
        directory = os.listdir(input_path)
        
        # Iterates through images in a folder
        for xist_im in directory:
            xist_path_name = xist_im.split('.')
            if len(xist_path_name) >= 2:
                
                # Checking that file is a .tiff and producing an image name
                if xist_path_name[1] == 'tiff' or xist_path_name[1] == 'tif':
                    xist_im_name_lst = xist_im.split('/')
                    xist_im_name = xist_im_name_lst[len(xist_im_name_lst)-1]
                    xist_im_split_lst = xist_im_name.split('_')
                    xist_im_ind = xist_im_split_lst[len(xist_im_split_lst)-1]

                    # Taking the image in the xist channel
                    if xist_channel in xist_im_name:
                        for dapi_im in directory:
                            dapi_path_name = dapi_im.split('.')
                            if len(dapi_path_name) >= 2:
                                
                                if dapi_path_name[1] == 'tiff' or dapi_path_name[1] == 'tif':
                                    dapi_im_name_lst = dapi_im.split('/')
                                    dapi_im_name = dapi_im_name_lst[len(dapi_im_name_lst)-1]

                                    # Checking image in DAPI channel and same image as XIST image
                                    if dapi_channel in dapi_im_name:
                                        if xist_im_ind in dapi_im_name:       
                                            im_xist = skimage.io.imread(input_path + '/' + xist_path_name[0] + '.' + xist_path_name[1])
                                            xist_seg_im = segment_xist(im_xist, 
                                                                  channel=xist_channel, 
                                                                  connectivity=connectivity,
                                                                  mode=mode,
                                                                  buffer_size=buffer_size)
                                            im_dapi = skimage.io.imread(input_path + '/' + dapi_path_name[0] + '.' + dapi_path_name[1])
                                            im_dapi = applying_man_seg.remove_small_objects(im_dapi, min_area=min_cell_area)
                                            xist_signal_array_tuple = combine_xist_dapi(xist_seg_im, im_dapi, min_area=min_area)
                                            pos_im_dct[xist_im_name] = xist_signal_array_tuple[0]
                                            neg_im_dct[xist_im_name] = xist_signal_array_tuple[1]
                                            xist_counts[xist_im_name] = xist_signal_array_tuple[2]
                                            if save_ims:
                                                pos_path = output_path + '/' 'pos_' + xist_im_name
                                                neg_path = output_path + '/' 'neg_' + xist_im_name
                                                auto_segmentation.save_images(pos_path, xist_signal_array_tuple[0])
                                                auto_segmentation.save_images(neg_path, xist_signal_array_tuple[1])
    return (pos_im_dct, neg_im_dct, xist_counts)