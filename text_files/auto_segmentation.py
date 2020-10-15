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





def med_filter(im):
    '''Takes in an input of a numpy array of an image and returns a median
    filtered numpy array.'''
    return skimage.filters.median(im)
    
    
    
    
    
def threshold(im):
    '''Takes in an input of an image numpy array and returns a thresholded numpy array'''
    im_med_filt = med_filter(im)
    otsu_thresh = skimage.filters.threshold_otsu(im_med_filt)
    return im > otsu_thresh
    
    
    
    
def clear_partial_cells(im, buffer_size=10):
    '''Takes an input of a numpy array of an image and a buffer size
    for how many pixels around the edges. Default is buffer_size=10.
    Returns a numpy array with partial cells on the perimeter of the 
    image removed.'''
    im_thresh = threshold(im)
    return skimage.segmentation.clear_border(im_thresh, buffer_size=buffer_size)
    
    
    
    
    
def watershed_segmentation(im, 
                           indices=False, 
                           footprint=np.ones((3,3)), 
                          buffer_size=10):
    '''Takes an input of a numpy array of an image,
    indices to put into the peak local maxes, default is indices=False,
    a footprint, default is footprint=np.ones((3,3)), 
    and a buffer size for the clear edges function, default is 10.
    
    Returns a numpy array of a segmented image.
    '''
    image = clear_partial_cells(im, buffer_size=buffer_size)
    distance = ndimage.distance_transform_edt(image)
    local_maxi = skimage.feature.peak_local_max(distance, indices=indices, footprint=footprint, labels=image)
    markers = skimage.morphology.label(local_maxi)
    return skimage.morphology.watershed(-distance, markers, mask=image)
    
    
    
    
def random_walker_segmentation(im, indices=False, 
                               footprint=np.ones((3,3)),
                               buffer_size=10):
    '''Takes an input of a numpy array of an image,
    the indices, default is False,
    the footprint, default is np.ones((3,3)), and
    a buffer size for the clear edges function. 
    
    Returns a segmented numpy array.'''
    image = clear_partial_cells(im, buffer_size=buffer_size)
    im_to_invert = image.astype(np.int)
    distance = ndimage.distance_transform_edt(image)
    local_maxi = skimage.feature.peak_local_max(distance, indices=indices, footprint=footprint, labels=image)
    markers = skimage.morphology.label(local_maxi)
    markers[~im_to_invert] = -1
    return skimage.segmentation.random_walker(im_to_invert, markers)
    
    
    
    
def image_props_segmentation(im, min_area, min_eccentricity, buffer_size=10):
    '''Takes in a numpy array of an image,
    the minimum area cells need to be,
    the minimum eccentricity of cells, and 
    a buffer size for the clear partial cells function, default is 10.
    
    Returns a segmented array of the image.
    
    This method uses code from Justin Bois. Taught during Bi1X.'''
    image = clear_partial_cells(im, buffer_size=buffer_size)
    
    # Produce binary image with cells=1 and background=0 and gives each cell a unique identifier
    im_labeled, n_labels = skimage.measure.label(image, background=0, return_num=True)
    
    # Extract the cells from background
    im_props = skimage.measure.regionprops(im_labeled)
    
    # Loop through image properties and delete small objects and objects that aren't circular enough
    n_regions = 0
    for prop in im_props:
        area = prop.area
        if prop.area < min_area:
            image[im_labeled == prop.label] = False
        else:
            n_regions += 1
    return image, n_regions

        
        
        
        
def chan_vese_segmentation(im, mu=.75, buffer_size=10):
    '''Takes in an input of a numpy array of an image,
    a mu value for segmentation, default is .75, and
    a buffer size, default is 10.'''''
    image = clear_partial_cells(im, buffer_size=buffer_size).astype(int)
    return skimage.segmentation.chan_vese(image, mu=mu)
    
    
    
    
def feltzenszwalb_segmentation(im, scale=2.0, multichannel=False, buffer_size=10):
    '''Takes an input of a numpy array of an image,
    a scale to perform segmentation in, default is 2.0,
    whether the image is multichannel, default is False,
    and the buffer size, default is 10
    
    Returns a numpy array of a segmented image'''
    image = clear_partial_cells(im, buffer_size=buffer_size)
    return skimage.segmentation.felzenszwalb(image, scale=scale, multichannel=multichannel)
    
    
    
    
def find_boundaries(im, connectivity=10, mode='inner', buffer_size=10):
    '''Takes an input of a numpy array of an image, 
    the connectivity of the boundaries, default is 10,
    the mode, default is "inner", and 
    the buffer size, default is 10.
    
    Returns a numpy array of boundaries of cells.'''
    image = clear_partial_cells(im, buffer_size=buffer_size)
    return skimage.segmentation.find_boundaries(image, connectivity=10, mode='inner')
    
    
    
    
def mark_boundaries(im, connectivity=10, mode='inner', buffer_size=10):
    '''Takes an input of a numpy array of an image, 
    the connectivity of the boundaries, default is 10,
    the mode, default is "inner", and 
    the buffer size, default is 10.
    
    Returns a numpy array of boundaries of cells bound to the 
    original image.'''
    image = clear_partial_cells(im, buffer_size=buffer_size)
    boundaries = find_boundaries(im, connectivity=10, mode='inner', buffer_size=10) 
    return skimage.segmentation.mark_boundaries(image, boundaries)
    
    
    
    
def im_arrays(directory):
    '''Iterates through an input directory (if folder) and produces a dictionary 
    of image names as keys and numpy arrays of the images as values.'''
    im_dct = {}
    for im in os.listdir(directory):
        im_name = im.split('.')
        if len(im_name) > 1:
            if im_name[1] == 'tiff' or im_name[1] == 'tif':
                value = skimage.io.imread(directory + '/' + im)
                im_dct[im] = value
    return im_dct
    
    
    
    
def save_images(im_path, im_array):
    '''Saves an image for a given path and image array as a tiff. 
    
    Note that this creates a user warning about a low contrast image, 
    but this is fine since this is simply a binary image'''
    
    int_array = 1 * im_array
    return skimage.io.imsave(im_path, int_array, check_contrast=False)
    
    
    
    
def save_as_csv(data, input_path, segmentation, output_path):
    '''Takes an input of of cell data as a tuple of two dictionaries
    and saves it in a csv file for a specified output path.
    
    Returns a pandas dataframe of the input data.'''
    
    cell_type_lst = []
    
    # Separating the tuple of dictionaries and then merging them and converting to dataframe
    auto_cells_dct = data[0]
    auto_cells_dct.update({'automated or manual segmentation': 'automated'})
    manual_cells_dct = data[1]
    manual_cells_dct.update({'automated or manual segmentation': 'manual'})

    auto_df = pandas.DataFrame(auto_cells_dct, index = [0])
    manual_df = pandas.DataFrame(manual_cells_dct, index = [0])
    df = pandas.concat([auto_df, manual_df], join='inner', ignore_index=True)
    
    
    '''# Producing indexing for auto vs manually segmented cells
    for i in range(len(auto_cells_dct)):
        cell_type_lst.append('# auto segmented cells')
    for i in range(len(manual_cells_dct)):
        cell_type_lst.append('# cells to manually segment')
    auto_vs_man_dct = {'# auto or man cells': cell_type_lst}
    auto_vs_man_df = pandas.DataFrame(auto_vs_man_dct, index = [0])
    df.merge(auto_vs_man_df)'''
    
    # Producing directory and name for the csv file for specified output path
    input_lst = input_path.split('/')
    csv_file_name = input_lst[len(input_lst)-1] + '.csv'
    directory = output_path + '/' + segmentation + '_' + csv_file_name
    
    # Saving as a csv file
    df.to_csv(directory)
    return df
    
    
    
    
def make_empty_array(im):
    '''Takes in the shape of an image and returns an empty rgb numpy array 
    based on the shape of the original image.'''
    shape = im.shape
    shape_lst = []
    if shape[2] != 3:
        shape_lst.append(shape[0]).append(shape[1]).append(3)
        shape = tuple(shape_lst)
    return np.empty(shape, dtype=np.int8)
    
    
    
    
def check_area_and_count_cells(im_segmented, im_name, min_area, max_area):
    '''Takes in a segmented image, ensures each segmented cell
    meets the minimum and maximum areas of a cell. Asks for user input for cells measured to be too large to be a cell.
    Counts the number of labeled regions within the image that were automatically counted.
    Changes RGB values to distinguish cells that need to be manually segmented from those that were automatically segmented.
    
    Manually segment=blue and automatically segment is yellow.'''
    n_cells = 0
    manual_cells = 0
    labeled_im = skimage.measure.label(im_segmented)
    
    for region in skimage.measure.regionprops(labeled_im):
        for coord in region.coords:
            (x,y,z) = coord
            if region.area < min_area:
                labeled_im[x][y][z] = 0
            elif region.area > max_area:
                labeled_im[x][y][0] = 1
                labeled_im[x][y][1] = 0
                labeled_im[x][y][2] = 0
            else:
                labeled_im[x][y][0] = 0
                labeled_im[x][y][1] = 0
                labeled_im[x][y][2] = 1
                
    return (labeled_im, n_cells, manual_cells)
    
    
    
    
def segmentation_arrays(input_path, 
                        output_path,
                        channel, 
                        segmentation,
                        save_ims=True,
                        save_csv=True,
                        connectivity=10,
                        mode='inner',
                        buffer_size=0,
                        min_area=2000,
                        max_area = 6000,
                        min_eccentricity=0.5,
                        indices=False,
                        footprint=np.ones((3,3)),
                        mu=.75,
                        scale=2.0,
                        multichannel=False
                       ):
    '''Input channel as a string in the form c# that you would like to segment in
    
    Segmentations: 
    "image_props" = image prop segementation
    "watershed" = watershed segmentation
    "random_walker" = random walker segmentation
    "chan_vese" = Chan Vese Segmentation
    "feltzenszwalb" = Feltzenszwalb Segmentation
    "boundaries" = find and mark boundaries segmentation
    
    
    Returns a dictionary of image arrays in that channel that have been segmented.'''
    # Initialize dictionary for segmentation arrays
    segmented_dct = {}
    n_cells_dct = {}
    manual_cells_dct = {}
    
    # Produce a dictionary from input of max intensity files
    # This will need some additional tests
    im_dct = im_arrays(input_path)
    for key in im_dct:
        
        # Ensure that only segmenting in the DAPI channel or specified channel
        split_key = key.split('/')
        im_name = split_key[len(split_key)-1]
        if len(im_name) > 1:
            key_channel = im_name.split('_')[1]


            if key_channel == channel:
                im = im_dct[key]
                seg_im_name = output_path + '/' + segmentation + '_' + key

                # Median filtering, thresholding, and clearing boundaries are all built into segmentation
                # If segmentation is find boundaries
                if segmentation == 'boundaries':
                    im_segmented = mark_boundaries(im, 
                                                   mode=mode,
                                                   connectivity=connectivity,
                                                   buffer_size=buffer_size,
                                                  )
                    checked_tuple = check_area_and_count_cells(im_segmented, 
                                                               key,
                                                               min_area=min_area,
                                                               max_area=max_area)
                    # Calls function to save segmented cells as images
                    if save_ims:
                        save_images(seg_im_name, checked_tuple[0])
                        
                # Need to integrate to check area with other methods
                elif segmentation == 'image_props':
                    im_props_tuple = image_props_segmentation(im, 
                                                            min_area, 
                                                            min_eccentricity, 
                                                            buffer_size=buffer_size)
                    checked_tuple = im_props_tuple
                    if save_ims:
                        save_images(seg_im_name, checked_tuple[0])
                
                # Might not be working properly
                elif segmentation == 'watershed':
                    im_segmented = watershed_segmentation(im, 
                                                          indices=indices, 
                                                          footprint=footprint, 
                                                          buffer_size=buffer_size)
                    checked_tuple = check_area_and_count_cells(im_segmented, 
                                                               key,
                                                               min_area=min_area,
                                                               max_area=max_area)
                    if save_ims:
                        save_images(seg_im_name, checked_tuple[0])
                elif segmentation == 'random_walker':
                    im_segmented = random_walker_segmentation(im, indices=indices, 
                                                              footprint=footprint,
                                                              buffer_size=buffer_size)
                    checked_tuple = check_area_and_count_cells(im_segmented, 
                                                               key,
                                                               min_area=min_area,
                                                               max_area=max_area)
                    if save_ims:
                        save_images(seg_im_name, checked_tuple[0])
                elif segmentation == 'chan_vese':
                    im_segmented = chan_vese_segmentation(im, 
                                                          mu=mu, 
                                                          buffer_size=buffer_size)
                    checked_tuple = check_area_and_count_cells(im_segmented, 
                                                               key,
                                                               min_area=min_area,
                                                               max_area=max_area)
                    if save_ims:
                        save_images(seg_im_name, checked_tuple[0])
                elif segmentation == 'feltzenszwalb':
                    im_segmented = feltzenszwalb_segmentation(im, 
                                                              scale=scale, 
                                                              multichannel=multichannel, 
                                                              buffer_size=buffer_size)
                    checked_tuple = check_area_and_count_cells(im_segmented, 
                                                               key,
                                                               min_area=min_area,
                                                               max_area=max_area)
                    if save_ims:
                        save_images(seg_im_name, checked_tuple[0])
                else:
                    print('Please input a valid segmentation type listed in the docstring.')
                segmented_dct[key] = checked_tuple[0]
                n_cells_dct[key] = checked_tuple[1]
                manual_cells_dct[key] = checked_tuple[2]
    if save_csv:
        df = save_as_csv((n_cells_dct, manual_cells_dct), 
                        input_path=input_path,
                        segmentation=segmentation,
                        output_path=output_path)
    return (segmented_dct, n_cells_dct)
