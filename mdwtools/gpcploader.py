# -*- coding: utf-8 -*-

"""
Script Name: gpcploader.py
Version Date: 2016-04-14
Author: Matthew Woelfle (mdwoelfle@gmail.com)

Functions for loading GPCP data directly from binary files to ndarrays
See example in main block for usage

Based on code written by some random dude on the internet:
    https://mattijnvanhoek.wordpress.com/2014/01/20/read-gpcp-1dd-data-using-python/
    originally: [read_1dd_num_months; byte_swap_1DD_structure; read_1DD;
                 read_1DD_header]

Notes on 1D GPCP (daily):
    First point center = (89.5N, 0.5E)
    Second point center = (89.5N, 1.5E)
    Last point center = (89.5S, 0.5W)
    Units = mm/d
    Directory = 'internet?'
    num_lon = 360
    num_lat = 180

Notes on 2D GPCP (monthly):
    First point center = (88.75N, 1.25E)
    Second point center = (88.75N, 3.75E)
    Last point center = (88.75S, 1.25W)
    Units = mm/d
    Directory = '/home/disk/eos9/woelfle/datasets/GPCP/monthly/'
    num_lon = 144
    num_lat = 72

"""

# %% Import packages as needed

import numpy as np
import sys

# %% Define functions


def get_gpcp_latlon(grid='2DD'):
    """
    Return lat and lon for GPCP dataset

    Args:
        grid - grid for GPCP data: 2DD for 2.5 degree, 1DD for 1 degree

    Returns:
        lat - latitude of box centers
        lon - longitude of box centers

    """

    if grid.lower() == '2dd':
        lat = np.arange(88.75, -90, -2.5)
        lon = np.arange(1.25, 360, 2.5)
    elif grid.lower() == '1dd':
        lat = np.arange(89.5, -90, -1)
        lon = np.arange(0.5, 360, 1)
    else:
        raise Exception('Unknown grid requested for GPCP data')

    return (lat, lon)


def get_gpcp_gridsize(filepath,
                      grid=None):
    """
    Return grid value for given filepath

    Args:
        filepath - full file path for gpcp file of interest
        grid - grid for GPCP data: '2DD' for 2.5 degree, '1DD' for 1 degree;
            automatically selected from filepath if not provided

    Returns:
        nlat - number of latitudes in file
        nlon - number of longitudes in file

    Notes:
        Determines grid size from filename

    """

    # If grid is not provided, set to empty string
    if grid is None:
        grid = ''

    # Determine size of grid for various data
    if ('v2' in filepath) | (grid.lower() == '2dd'):
        nlat = 72
        nlon = 144

    elif ('v1' in filepath) | (grid.lower() == '1dd'):
        nlat = 180
        nlon = 360

    else:
        raise Exception('Unknown grid for GPCP file: \n' + filepath)

    return (nlat, nlon)


def read_gpcp_num_times(filepath,
                        grid=None):
    """
    The "times" metadata is [times]=1-nn, where nn is the number
    of [times] in the year, which is what we want for the 3rd
    dimension. Hence the len([times]).
    [times] = {'1DD': 'days=',
               '2DD': 'months='}[grid]
        where grid is determined from the filepath

    Args:
        filepath - full path for gpcp file to be loaded
        grid - grid for GPCP data: '2DD' for 2.5 degree, '1DD' for 1 degree;
            automatically selected from filepath if not provided

    Returns:
        num_times - number of time steps in file
        header - header information for file

    """

    # If grid is not provided, set to empty string
    if grid is None:
        grid = ''

    num_lon = get_gpcp_gridsize(filepath, grid=grid)[1]

    with open(filepath, 'rb') as filein:
        # filein = open(filepath, 'rb')
        header = filein.read(num_lon*4)

    # Remove blank space at end of header
    header = header.rstrip()

    # Find index for number of time steps
    if ('v2' in filepath) | (grid.lower() == '2dd'):
        jTimes = header.find(b'months=') + len('months=1-')
    elif ('v1' in filepath) | (grid.lower() == '1dd'):
        jTimes = header.find(b'days=') + len('days=1-')
    else:
        raise Exception('Unknown grid for GPCP file: \n' + filepath)

    # Pull number of time steps
    num_times = int(header[jTimes:jTimes+2])

    # Return number of time steps and header
    return num_times, header


def byte_swap_gpcp_struct(filepath,
                          num_times,
                          header,
                          grid=None):
    """
    Uses output of read_1dd_num_day to determine if byte-swap is needed

    Args:
        filepath - full path from which to load data
        grid - grid for GPCP data: '2DD' for 2.5 degree, '1DD' for 1 degree;
            automatically selected from filepath if not provided
    """

    # Get size of data arrays
    num_lat, num_lon = get_gpcp_gridsize(filepath, grid=grid)  # cols, rows

    # see if file is written in big-endian (indicated by Silicon machine)
    if b'Silicon' in header:
        file_byte_order = 'big'
    else:
        file_byte_order = 'little'

    if sys.byteorder == 'little' and file_byte_order == 'big':
        # open file using big endian dtype and select all values that
        #   correspond with to the metadata
        data = np.fromfile(filepath,
                           dtype='>f')[-(num_lon*num_lat*num_times)::]
    else:
        data = np.fromfile(filepath,
                           dtype='f')[-(num_lon*num_lat*num_times)::]

    data = data.reshape((num_times, num_lat, num_lon))

    return data


def read_gpcp(filepath,
              grid=None,
              header_flag=False):
    """
    The main procedure; read both header and data, and swap bytes if needed.

    Args:
        filepath - full path to gpcp file to be loaded
        grid - grid for GPCP data: '2DD' for 2.5 degree, '1DD' for 1 degree;
            automatically selected from filepath if not provided
        header_flag - flag determining if header is included in output

    Returns:
        if header_flag = False (default):
            data as numpy array
        if header_flag = True:
            data as numpy array
            header as string
    """

    # Throw error if files still gzipped
    if filepath[-3:] == '.gz':
        raise Exception('Please gunzip GPCP files prior to loading.')

    # Determine number of time steps in file and get header info
    num_times, header = read_gpcp_num_times(filepath, grid=grid)

    # Load data from file
    data = byte_swap_gpcp_struct(filepath, num_times, header, grid=grid)

    # Prepare to return output
    if header_flag is False:
        return data
    elif header_flag is True:
        return (data, header)


def read_multigpcp(dataDir,
                   fileBase,
                   timeList,
                   grid=None,
                   header_flag=False):
    """
    Load multiple gpcp files to single ndarray

    Args:
        dataDir - directory in which GPCP files are stored
        fileBase - base for file name
            e.g. 'gpcp_v2.2_psg.' or 'gpcp_1dd_v1.2_p1d.'
        timeList - list of times for loading
            e.g. ['1981', '1982'] or ['198101', '198102']
        grid - resolution to be loaded
            e.g. '2DD' or '1DD'
        header_flag - true to also return list of header information from each
            file

    Returns:
        data - ndarray with dimensions [time, lat, lon] for gpcp data loaded
        (headerList) - list of header information for each file loaded
            (only returned if header_flag==True)

    Notes:
        For examples given first items is for 2DD, second for 1DD.

    """

    # Create empty list to hold data as it is loaded
    dataList = []

    # Load each file's data to list
    for loadTime in timeList:
        dataList.append(read_gpcp(dataDir + fileBase + loadTime,
                                  grid=None,
                                  header_flag=False))

    # Concatenate to single ndarray
    data = np.concatenate(dataList, axis=0)

    # Load header information if requested
    if header_flag:
        headerList = []
        for loadTime in timeList:
            headerList.append(read_gpcp_header(dataDir + fileBase +
                                               loadTime))

    if header_flag:
        return data, headerList
    else:
        return data


def read_gpcp_header(filepath,
                     grid=None):
    """
    Just read the header
    Args:
    filepath - full path from which to read gpcp data
    grid - grid for GPCP data: 2DD for 2.5 degree, 1DD for 1 degree;
        automatically selected from filepath if not provided
    """

    # Determine number of longitudes in file
    num_lon = get_gpcp_gridsize(filepath,
                                grid=grid)[1]

    # Open file and read header information
    with open(filepath, 'rb') as filein:
        # filein = open(filepath, 'rb')
        header = filein.read(num_lon*4)

    # Strip blank space from end
    header = header.rstrip()

    # Return header
    return header

# %% Example

if __name__ == '__main__':

    # Need to unzip files prior to loading

    # Set directory for data archive
    dataDir = '/home/disk/eos9/woelfle/dataset/GPCP/monthly/'

    # Set base name for file
    fileBase = 'gpcp_v2.2_psg.'

    # Set list of time periods to load
    loadTimes = ['1981', '1982']

    # Load data from file
    data = read_multigpcp(dataDir, fileBase, loadTimes)

    # Load lat/lon info
    gpcpLat, gpcpLon = get_gpcp_latlon(grid='2DD')
