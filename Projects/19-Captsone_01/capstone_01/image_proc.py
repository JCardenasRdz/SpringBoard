import dicom
import os
import numpy as np
from PIL import Image
import dhash as _dhash

def get_full_path(origin, seriesID):
    '''
    Returns the full path to the directory to a dicm seriesID
    ---------------------------------------------------------
    inputs:
    origin = string with the directory where the os.walk starts
    seriesID = string for the series ID to be found
    ---------------------------------------------------------
    Example:
    series_ID = ''1.3.6.1.4.1.14519.5.2.1.7695.1700.933316195746120155903339740103'
    home_dir  =  './raw_data/MRI/'
    full_path_to_images=   get_full_path(origin, seriesID)
    ---------------------------------------------------------
    '''
    full_path = ''  # create an empty list
    for root, dirs, fileList in os.walk(origin):
        if len(dirs) > 0 :
            if seriesID in dirs:
                full_path = os.path.join(root,seriesID)
    if len(full_path) < 1:
        print( "That series is not under the origin directory" )

    return full_path

def file_list(full_path):
    '''
    Returns a list of all .dcm files under the directory at the end of full_path
    '''
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(full_path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    return lstFilesDCM

def load_dicom(origin, seriesID):
    '''
    Returns all images in a seriesID directory  as a singel matrix
    ---------------------------------------------------------
    inputs:
    origin = string with the directory where the os.walk starts
    seriesID = string for the series ID to be found
    ---------------------------------------------------------
    Example:
    series_ID = ''1.3.6.1.4.1.14519.5.2.1.7695.1700.933316195746120155903339740103'
    home_dir  =  './raw_data/MRI/'
    dicom_images=   get_full_path(origin, seriesID)
    ---------------------------------------------------------
    '''
    SeriesID_path = get_full_path( origin, seriesID)
    file_names = file_list(SeriesID_path)

    dicom_dict = dicom.read_file(file_names[0])
    rows = dicom_dict.Rows
    cols = dicom_dict.Columns
    num_images = len(file_names)
    Images = np.zeros( (num_images, rows, cols))

    for idx,file in enumerate(file_names):
        dcm = dicom.read_file(file)
        Images[dcm.InstanceNumber-1, :, :] =  dcm.pixel_array
    return Images

def _array_to_PIL(image_array):
    return Image.fromarray(np.uint8( 255 * image_array / np.max(image_array)))

def _matrix_2_list(matrixformat):
    newlist =  list()
    for element in matrixformat.split():
        for number in element:
            newlist.append(number)
    return newlist

def image_to_Dhash(image_array, size=8):
    image_PIL = _array_to_PIL(image_array)
    # hash as an integer
    row, col = _dhash.dhash_row_col(image_PIL)

    # transform to binary
    row_binary = _matrix_2_list( _dhash.format_matrix(row,size = size) )
    col_binary = _matrix_2_list( _dhash.format_matrix(col,size = size) )

    return row_binary, col_binary
