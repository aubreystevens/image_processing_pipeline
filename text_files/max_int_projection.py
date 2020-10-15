import numpy as np
import scipy.misc
from scipy import ndimage

import skimage
import skimage.io

import bi1x

from czifile import CziFile
import czifile
import imageio

import os




def czi_array(path):
    '''Takes in the path of a czi file and returns an array of the file and its shape as a tuple'''
    array = czifile.imread(path)
    return (array, array.shape)





def mod_array(path):
    '''Removes the first three dimensions of a czi array. Returns a modified
    czi array of 5 dimensions and array shape tuple as a tuple.'''
    array = czi_array(path)[0]
    shape = czi_array(path)[1]
    for i in range(3):
        if shape[i] != 1:
            print('''There are more than 4 dimensions in this array. Important
            information may have been removed from these layers''')
        array = array[0]
    new_shape = array.shape
    '''if new_shape < 4:
        raise ValueError("Cannot have an array with less than 4 dimensions")'''
    return (array, new_shape)





def max_int_projection(path):
    '''Takes in a path and returns a list of maximum intensity arrays
    for each channel.'''

    # Geting modified array and array shape and initializing list of arrays
    array = mod_array(path)[0]
    shape = mod_array(path)[1]
    channel_arrays = []
    x_ind = 0
    y_ind = 0
    z_ind = 0
    # For each channel, produce a new array of maximum intensities

    # Check to ensure that correct axis identified as x, y, and z
    if shape[2] == shape[3]:
        x_ind = 2
        y_ind = 3
        z_ind = 1
    elif shape[1] == shape[2]:
        x_ind = 1
        y_ind = 2
        z_ind = 3
    else:
        print('Shape unidentified. X and Y axes have different number of pixels.')


    for channel in range(shape[0]):
        max_int_array = np.empty((shape[x_ind], shape[y_ind]))

        # Iterate through x and y coordinates of pixels
        for x in range(shape[x_ind]):
            for y in range(shape[y_ind]):
                coord_val = 0

                # Slice the czi based on z slice
                for z in range(shape[z_ind]):
                    new_val = array[channel][z][x][y][0]

                    # Compare pixel values at a specific x and y coordinate for each slice
                    if new_val > coord_val:
                        coord_val = new_val

    # Store the maximums in an array, append it to a list, and return the list
                max_int_array[x][y] = coord_val
        channel_arrays.append(max_int_array)
    return channel_arrays





def loop_through_files(input_path, output_path, path_type='folder'):
    '''path_type="folder" will iterate through all the czi files in a folder
    and find it's maximum intensity projection.

    path_type = "image", will find the maximum intensity projection of a
    single image.

    input_path is a string of the path that the folder or image is in.

    output_path is the path that you would like the maximum intensity
    projections stored in. This should be a folder path. They are stored as tiff files

    Returns a dictionary of input image names as keys and their
    maximum intensity arrays as values.'''
    max_int_dct = {}

    # Iterate through images in a folder
    if path_type == 'folder':
        directory = os.listdir(input_path)
        for im in directory:
            im_name = im.split('.')
            if im_name[1] == 'czi':

                # Use the number of channels to save a max intensity image of each channel for each image
                array_shape = mod_array(input_path + '/' + im)[1]
                for i in range(array_shape[0]):

                    # Naming format for the images, stores in output_path, uses max_c#_imagename
                    im_tiff_name = im.split('.')
                    im_tiff_name = output_path + '/' + 'max_c{}_'.format(i+1) + im_tiff_name[0] + '.tiff'
                    max_int_array = max_int_projection(input_path + '/' + im)[i]
                    max_int_dct[im_tiff_name] = max_int_array
                    imageio.imwrite(im_tiff_name, max_int_array)

    # Take the image path, isolate the image name
    elif path_type=='image':

        # Use the number of channels to save a max intensity image of each channel for each image
        array_shape = mod_array(input_path)[1]
        for i in range(array_shape[0]):

            # Make into a .tiff with format max_c#_imagename
            im_tiff_name = input_path.split('/')
            index = len(im_tiff_name)
            im_tiff_name = im_tiff_name[index-1].split('.')
            im_tiff_name = im_tiff_name[0] + '.tiff'
            im_tiff_name = output_path + '/' + 'max_c{}_'.format(i+1) + im_tiff_name
            max_int_array = max_int_projection(input_path)[i]
            max_int_dct[im_tiff_name] = max_int_array
            imageio.imwrite(im_tiff_name, max_int_array)
    return max_int_dct
'''else:
        raise ValueError(path_type is invalid. Use either path_type="folder"
        or path_type="image")'''
