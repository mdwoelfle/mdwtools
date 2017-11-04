# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:41:41 2015
Version Date: 2017-06-13

@author: woelfle
"""
import numpy as np
import netCDF4 as nc4
from math import pi
from socket import gethostname
import glob
import os                        # import operating system functions
from scipy import interpolate    # interpolation functions
import mdwtools.gpcploader as gpcploader  # load functions for loading gpcp
import matplotlib.pyplot as plt  # for plotting
import xarray as xr              # For managing data

"""
List of Functions:
    setlatlonedges - define all box edges for a lat/lon grid
    getplanetradius - return radius of planet [m]
    getplanetrotationrate - return rotation rate of planet [s^-1]
    calcregarea - compute area in given rectangular area in lat/lon space
    calcareaweight - compute relative contribution of each grid box in a given
        area to the total area in that region
    calcallregmeans - compute regional, regional zonal, and regional meridional
        means
    calcregmean - compute regional mean
    calcregzonmean - compute regional zonal mean
    calcregmeridmean - compute regional meridional mean
    calceqseascycle - compute seasonal cycle of a given mean field after given
        spinup time
    convertsigmatopres - comvert sigma vertical coordinates to given pressures
    findlast - return index of last instance of a substring in a bigger string
    gethybsigcoeff - get coefficients for converting from hybrid sigma to
        pressure coordinates
    loadnetcdf2class - load variable to workspace as nclatlonvar
    loadcase2dict - placeholder for loading dictionary of nclatlonvars
    loadhybtop - load variables from hybrid sigma coords to pressure levels
    getgridsize - return nx x ny for a given grid
    regridcesm2d - workhorse for regridding any cesm data using NCAR methods
    regridcesmnd - wrapper for regridcesm2d for 2d, 3d, and 4d variables
    getstdunit- return standard string for a given unit
    convertunit - convert one unit to another (works only for precip)
    rmse - compute root mean squared error between two arrays

List of Classes:
    ncgenvar - define generic netCDF wrapper
    nclatlonvar - define netCDF wrapper for lat-lon datasets to speed analysis

"""


def setlatlonedges(lat=None, lon=None, latb=None, lonb=None, datasrc='cesm'):
    """
    Modify latitude and longitude box edges to conform to a standard with all
        interior and exerior edges defined. May operate from given interior
        edges (i.e. latb, lonb provided as in CESM output) or may assume box
        centers given with no edge information and compute all edges from there
        (e.g. CORE dataset)

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-07
    """

    # Update if from cesm
    if datasrc.lower() in ['cesm']:
        if latb[0] != -90:
            latb = np.insert(latb, 0, -90)
        if latb[-1] != 90:
            latb = np.append(latb, 90)
        if (lonb[0] % 360) != (lonb[-1] % 360):
            lonb = np.append(lonb, lonb[0])
        return latb, lonb

    # Update if from CORE dataset on regular grid (no latb/lonb available)
    elif datasrc.lower() in ['core', 'core1d']:
        if latb is None:
            dlatb = (lat[1:] - lat[:-1])/2.
            if all(dlatb == dlatb[0]):
                latb = np.append(lat - dlatb[0], 90)
            else:
                latb = np.append(np.insert(lat[:-1] + dlatb, 0, -90), 90)
        if lonb is None:
            dlonb = (lon[1:] - lon[:-1])/2.
            if all(dlonb == dlonb[0]):
                lonb = np.append(lon - dlonb[0], 360)
            else:
                latb = np.append(np.insert(lon[:-1] + dlonb, 0, 0), 360)
        return latb, lonb


def getplanetradius(planet='Earth'):
    """
    Set standard value for radii of planets
    Version Date: 2015-03-30
    """

    # Default planet is earth
    if planet is None:
        planet = "earth"

    # Return radius of planet [m]
    try:
        return {'venus': 6.052e6,
                'earth': 6.371e6,
                'mars': 3.390e6
                }[planet.lower()]
    except KeyError:
        raise KeyError('Unknown planet.')


def getplanetrotationrate(planet='Earth'):
    """
    Set standard value for rotation rate of planets
    Version Date: 2016-03-30
    """

    # Return rotation rate of planet [s^-1]
    try:
        return{'venus': 2.9927e-7,
               'earth': 7.2921e-5,
               'mars': 7.0882e-5}[planet.lower()]
    except KeyError:
        raise KeyError('Unknown planet.')


def calcregarea(latLim=None,
                lonLim=None,
                planet=None
                ):
    """
    Calculates area between given latitude and longitude limits on a spherical
        planet

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-06-25

    Args:
        latLim - latitude limits for subset region (default = [-90, 90])
        lonLim - longitude limits for subset region (default - [0, 360])
        planet - planet used to compute areas (default = "earth")

    Returns:
        regArea - area of requested subset region in m^2

    Raises:
        none

    Notes:
        Assumes latitudes run S to N, e.g. -90 to 90
        Assumes longitudes run E to W, e.g. 0 to 360
        Assumes rectangular area
    """

    # Assign region limits if not provided by user
    if latLim is None:
        latLim = np.array([-90, 90])
    if lonLim is None:
        lonLim - np.array([0, 360])

    # Force longitude limits to be increasing (allows for wrapping of 0)
    while lonLim[1] < lonLim[0]:
        lonLim[1] = lonLim[1] + 360

    # Obtain radius of planet
    re = getplanetradius(planet=planet)

    # Compute area of defined region
    regArea = (re**2 * (np.deg2rad(lonLim[1]) - np.deg2rad(lonLim[0])) *
               (np.sin(np.deg2rad(latLim[1])) -
                np.sin(np.deg2rad(latLim[0])))
               )

    # Return region area in m^2
    return regArea


def calcareaweight(lat, lon,
                   latb=None,
                   lonb=None,
                   latLim=None,
                   lonLim=None
                   ):
    """
    Creates a weight mask for subsetting data on a lat/lon grid to a specific
        region

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-10-19

    Args:
        lat - latitude for grid box centers
        lon - longitude for grid box centers

    Kwargs:
        latb - latitude for grid box edges
        lonb - longitude for grid box edges
        latLim - latitude limits for subset region (default = [-90, 90])
        lonLim - longitude limits for subset region (default = [0 360])

    Returns:
        areaFilt - boolean array masked to grid cells which overlap the subset
            area; output is axis[-1]=lon, axis[-2]=lat
        areaWeight - array with relative contribution of each grid cell to the
            total subset area; output is axis[-1]=lon, axis[-2]=lat

    Raises:
        none

    Notes:
        Assumes latitudes run S to N, e.g. -90 to 90
        Assumes longitudes run E to W, e.g. 0 to 360
        Cannot handle lon limits wrapping 0 degrees
    """

    # Assign region limits if not provided by user
    if latLim is None:
        latLim = np.array([-90, 90])
    if lonLim is None:
        lonLim = np.array([0, 360])

    # Check inputs to handle common usage errors
    if lat[1] <= lat[0]:
        raise ValueError("ERROR: Latitudes must run S to N, e.g. -90 to 90")

    if lon[1] <= lon[0]:
        raise ValueError("ERROR: Longitudes must run E to W, e.g. 0 to 360 degE")

    if latLim[1] <= latLim[0]:
        raise ValueError("Latitude limits are not increasing.")

    if lonLim[1] <= lonLim[0]:
        raise ValueError("ERROR: Longitude limits are not increasing or wrap across 0.")

    # Correct longitudes to run 0 to 360

    # Force longitude limits to be between 0 and 360
    if (lonLim[0] % 360) != 0:
        lonLim[0] = lonLim[0] % 360
    if (lonLim[-1] % 360) != 0:
        lonLim[-1] = lonLim[-1] % 360

    # Force longitudes to monotonically increase
    lonb[lonb > 360] = lonb[lonb > 360] % 360
    while any(lonb[1:] < lonb[:-1]):
        lonInd = np.asarray(np.where(lonb[1:] < lonb[:-1])) + 1
        try:
            lonb[lonInd:] = lonb[lonInd:] + 360
        except TypeError:
            # Included as original indexing broke as of 2017-07-10
            lonInd = lonInd[0][0]
            lonb[lonInd:] = lonb[lonInd:] + 360

    # Filter to area of interest

    # Filter latitude to area of interest based on box edges
    #   Finds boxes where interior edge is within limits
    latbFilt = (latb > latLim[0]) & (latb < latLim[1])

    # Filter longitudes based on box edges
    #   Finds boxes where interio edge is within limits
    if (np.diff(lonLim) % 360) == 0:                    # All longitudes
                                                        #   included
        lonbFilt = np.ones(lonb.shape, dtype=bool)
    elif lonLim[1] > lonLim[0]:                         # Cases where region
                                                        #   does not cross 0
        lonbFilt = (lonb > lonLim[0]) & (lonb < lonLim[1])

    # Use box edge filter to locate boxes to include in subsetted region
    latFilt = latbFilt[1:] | latbFilt[:-1]
    lonFilt = lonbFilt[1:] | lonbFilt[:-1]

    # Compute latitude weighting for each area of interest

    # Pull interior lat edges for filtered latitudes for computing weights
    latb4w = latb[latbFilt]

    # Add latitude limits as exterior of latitudes pulled for weighting
    latb4w = np.insert(latb4w, 0, latLim[0])
    latb4w = np.append(latb4w, latLim[-1])

    # Compute regional latitude weights
    latW = np.sin(latb4w[1:]/180.*pi) - np.sin(latb4w[:-1]/180.*pi)
    latW = latW/np.sum(latW)

    # Create array of weights for all latitudes
    latWall = np.zeros_like(latFilt, dtype=float)
    latWall[latFilt] = latW

    # Compute longitude weighting for each area of interest

    # Pull interior lon edges for filtered latitudes for computing weights
    lonb4w = lonb[lonbFilt]

    if (np.diff(lonLim) % 360) != 0:                    # Not all longitudes
        # Add box edges if not global
        lonb4w = np.insert(lonb4w, 0, lonLim[0])
        lonb4w = np.append(lonb4w, lonLim[-1])

    # Compute regional longitude weights
    lonW = lonb4w[1:] - lonb4w[:-1]
    lonW = lonW/float(np.sum(lonW))

    # Create array of weights for all longitudes
    lonWall = np.zeros_like(lonFilt, dtype=float)
    lonWall[lonFilt] = lonW

    # Combine latitude and longitude weights to create full region weights

    # Cast weights from 1D vectors to matrices
    latWall = latWall.reshape([latWall.size, 1])    # row
    lonWall = lonWall.reshape([1, lonWall.size])    # col

    # Combine longitude and latitude weighting
    areaWeight = latWall*lonWall                    # nLon x nLat

    # Cast filters from 1D vectors to matrices
    latFilt = latFilt.reshape([latFilt.size, 1])
    lonFilt = lonFilt.reshape([1, lonFilt.size])

    # Combine latitude and longitude
    areaFilt = latFilt*lonFilt

    # Return areaFilt and areaWeight
    return (areaFilt, areaWeight)


def calcallregmeans(dataIn, lat, lon, latb, lonb,
                    latLim=None,
                    lonLim=None,
                    planet=None
                    ):
    """
    Compute regional mean, regional zonal mean, and regional meridional mean
    2015-06-29
    """

    # Compute regional mean
    regMean = calcregmean(dataIn, lat, lon, latb, lonb,
                          latLim=latLim,
                          lonLim=lonLim,
                          planet=planet
                          )[0]

    # Compute zonal mean
    zonMean = calcregzonmean(dataIn, lat, lon, latb, lonb,
                             latLim=latLim,
                             lonLim=lonLim,
                             planet=planet
                             )

    # Computer meridional mean
    meridMean = calcregmeridmean(dataIn, lat, lon, latb, lonb,
                                 latLim=latLim,
                                 lonLim=lonLim,
                                 planet=planet
                                 )

    # Return all means
    return (regMean, zonMean, meridMean)


def calcregmean(dataIn, lat, lon,
                latb=None,
                lonb=None,
                latLim=None,
                lonLim=None,
                planet=None
                ):
    """
    Computes the regional mean over a given area on a planet.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-05

    Args:
        dataIn - [time x] lat x lon array of data for which mean is computed
        lat - latitude for grid box centers
        lon - longitude for grid box centers

    Kwargs:
        latb - latitude for grid box edges
        lonb - longitude for grid box edges
        latLim - latitude limits for subset region (default = [-90, 90])
        lonLim - longitude limits for subset region (default = [0, 360])
        planet - planet for which computation is run (default = "earth")

    Returns:
        regMean - vector of regional means for each time step in dataIn
        nonNanArea - total area used to compute regional mean, excluding
            regions where dataIn is NaN; most useful for computing fluxes
        regArea - total area within the reigon defined by latLim and lonLim

    Raises:
        none

    Notes:
        Assumes latitudes run S to N, e.g. -90 to 90
        Assumes longitudes run E to W, e.g. 0 to 360
        Cannot handle lon limits wrapping 0 degrees
        Now sets nans to zeros when summing values prior to dividing by area
            Should prevent outputting an array of nans without use of nan fns.
    """

    # Set default values for limits if not provided by user
    if latLim is None:
        latLim = np.array([-90, 90])
    if lonLim is None:
        lonLim = np.array([0, 360])

    # Find area weights for region
    areaFilt, areaWeight = calcareaweight(lat, lon, latb, lonb,
                                          latLim=latLim,
                                          lonLim=lonLim
                                          )

    # Find total area occupied by data (exclude nans) at each time step
    nonNanArea = (areaWeight*~np.isnan(dataIn)).sum(axis=-2).sum(axis=-1)

    # Replace input nans with 0s for summing
    dataInNonan = dataIn.copy()
    dataInNonan[np.isnan(dataIn)] = 0

    # Weight data by areaWeight and sum over region
    dataInW = (dataInNonan*areaWeight).sum(axis=-2).sum(axis=-1)

    # Compute regional mean of data
    regMean = dataInW / nonNanArea

    # Compute area of subset region (includes points without data)
    regArea = calcregarea(latLim=latLim,
                          lonLim=lonLim
                          )

    # Return values
    return (regMean, nonNanArea, regArea)


def calcregzonmean(dataIn, lat, lon, latb, lonb,
                   latLim=None,
                   lonLim=None,
                   planet=None
                   ):
    """
    Computes the zonal mean over a given region on a planet.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-06-25

    Args:
        dataIn - [time x] lat x lon array of data for which mean is computed
        lat - latitude for grid box centers
        lon - longitude for grid box centers
        latb - latitude for grid box edges
        lonb - longitude for grid box edges
        latLim - latitude limits for subset region (default = [-90, 90])
        lonLim - longitude limits for subset region (default = [0, 360])
        planet - planet for which computation is run (default = "earth")

    Returns:
        zonMean - zonal means at each timestep axis[-1]=lat[, axis[-2]=time]

    Raises:
        none

    Notes:
        Assumes latitudes run S to N, e.g. -90 to 90
        Assumes longitudes run E to W, e.g. 0 to 360
        Cannot handle lon limits wrapping 0 degrees
    """
    # Assign default values to region limits if none provided by user
    if latLim is None:
        latLim = np.array([-90, 90])
    if lonLim is None:
        lonLim = np.array([0, 360])

    # Obtain area weights
    areaFilt, areaWeight = calcareaweight(lat, lon, latb, lonb,
                                          latLim=latLim,
                                          lonLim=lonLim
                                          )

    # Weight data by areaWeight
    # dataInW = (areaWeight*dataIn).sum(axis=-1)
    dataInW = np.nansum(areaWeight*dataIn, axis=-1)

    # Compute area occupied by data at each time step
    nonNanArea = (areaWeight*~np.isnan(dataIn)).sum(axis=-1)

    # Compute zonal means over specifed region
    zonMean = dataInW / nonNanArea

    # Return zonal mean
    return zonMean


def calczonregmean_g16(data,
                       latLim,
                       lonLim,
                       lat=None,
                       lon=None,
                       latb=None,
                       lonb=None):
    """
    Compute zonal mean on g16 (POP 1 deg) grid

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        Unknown

    Args:
        data - np array of values
        latLim - latitude limits for computing zonal mean
        lonLim - longitude limits for computing zonal mean
        lat - new latitude array
        lon - new longitude array
        latb - new latitude edge array
        lonb - new longitude edge array

    Returns:
        zonMean - np array of zonal mean values ([time, lat])

    Notes:
        Reload dimensions each time, may be problematic...
    """

    # Check if need to load dimensions
    loadDim_flag = all([lat is None, lon is None,
                        latb is None, lonb is None])

    # Regrid to atmosphere grid
    f16Var = regridcesmnd(data,
                          'g16',
                          'f19')

    # Load new dimension information from file
    if loadDim_flag:
        # Set dimension file name
        dimFile = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                   'cesm.f19.dimensions.nc')
        loadDims = ['lat', 'lon', 'slat', 'slon']
        dimNames = ['lat', 'lon', 'latb', 'lonb']
        dims = dict()
        with nc4.Dataset(dimFile, 'r') as ncDataset:
            for jDim in range(len(loadDims)):
                dims[dimNames[jDim]] = ncDataset.variables[loadDims[jDim]][:]

        # Clean up latitude edges
        dims['latb'], dims['lonb'] = setlatlonedges(lat=dims['lat'],
                                                    lon=dims['lon'],
                                                    latb=dims['latb'],
                                                    lonb=dims['lonb'])

        # Copy to new variable names from dictionary
        lat = dims['lat']
        lon = dims['lon']
        latb = dims['latb']
        lonb = dims['lonb']

    # Copmute regional, zonal mean
    zonMean = calcregzonmean(f16Var,
                             lat,
                             lon,
                             latb,
                             lonb,
                             latLim=latLim,
                             lonLim=lonLim,
                             )
    return zonMean


def calcregmeridmean(dataIn, lat, lon, latb, lonb,
                     latLim=None,
                     lonLim=None,
                     planet=None
                     ):
    """
    Computes the merdional mean over a given region on a planet.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-06-25

    Args:
        dataIn - [time x] lat x lon array of data for which mean is computed
        lat - latitude for grid box centers
        lon - longitude for grid box centers
        latb - latitude for grid box edges
        lonb - longitude for grid box edges
        latLim - latitude limits for subset region (default = [-90, 90])
        lonLim - longitude limits for subset region (default = [0, 360])
        planet - planet for which computation is run (default = "earth")

    Returns:
        meridMean - meridional means at each timestep axis[-1]=lat[, axis[-2]=time]

    Raises:
        none

    Notes:
        Assumes latitudes run S to N, e.g. -90 to 90
        Assumes longitudes run E to W, e.g. 0 to 360
        Cannot handle lon limits wrapping 0 degrees
    """

    # Assign default values to region limits if none provided by user
    if latLim is None:
        latLim = [-90, 90]
    if lonLim is None:
        lonLim = [0, 360]

    # Obtain area weights
    areaFilt, areaWeight = calcareaweight(lat, lon, latb, lonb,
                                          latLim=latLim,
                                          lonLim=lonLim
                                          )

    # Weight data by areaWeight
    dataInW = np.nansum(areaWeight*dataIn, axis=-2)

    # Compute area occupied by data at each time step
    nonNanArea = (areaWeight*~np.isnan(dataIn)).sum(axis=-2)

    # Compute meridional means over specifed region
    meridMean = dataInW / nonNanArea

    # Return meridional mean
    return meridMean


def calceqseascycle(tEq, dataIn):
    """
    Compute equilibrium seasonal cycle of given mean field (tested on up to 2D)
    2015-06-29
    """

    # Determine time steps for averaging "equilibrium"
    tMean = np.arange(tEq, dataIn.shape[0])

    # Pull dimension sizes and compute mean seasonal cycle
    if dataIn.ndim == 1:
        nt = dataIn[tMean].shape[0]
        nX = 1
        dataInEq = np.mean(np.reshape(dataIn[tMean], [nt/12, 12*nX]), axis=0)
    elif dataIn.ndim == 2:
        nt, nX = dataIn[tMean, :].shape
        dataInEq = np.reshape(np.mean(np.reshape(dataIn[tMean, :],
                                                 [nt/12, 12*nX]),
                                      axis=0),
                              [12, nX])

    # Return average seasonal cycle
    return dataInEq


def calcddlat_m(data,
                lat, lon
                ):
    """
    Calculate meridional derivative of given field on a map (lat/lon coords)

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-03-25

    Args:
        data - data for which derivative is to be computed [(time), lat, lon]
        lat - latitudes of size [nlat, ]
        lonGrid - longitudes of size [nlon, ]

    Returns:
        ddatady - meridional deerivative of data

    Notes:
        N/A
    """

    # Set constants
    Re = getplanetradius('Earth')  # Radius of Earth (m)

    # For testing purposes
    # u = -cases[jCase]['TAUX'].data  # [t, :, :]
    # v = -cases[jCase]['TAUY'].data  # [t, :, :]

    # Create lat/lon grids from vectors
    lonG, latG = np.meshgrid(lon, lat)

    # Compute 2dy for each grid point (dy at N/S edges)
    dlatG = np.roll(latG, -1, axis=-2) - np.roll(latG, 1, axis=-2)
    dlatG[0, :] = latG[1, :] - latG[0, :]
    dlatG[-1, :] = latG[-1, :] - latG[-2, :]
    dy = dlatG * Re * np.pi/180.

    # Compute data(j+1) - data(j-1)
    # j = y index
    ddata = np.roll(data, -1, axis=-2) - np.roll(data, 1, axis=-2)
    ddata[:, 0, :] = data[:, 1, :] - data[:, 0, :]
    ddata[:, -1, :] = data[:, -1, :] - data[:, -2, :]

    # Compute meridional derivative (d/dy)
    ddatady = ddata/dy

    return ddatady


def calcddlon_m(data,
                lat, lon
                ):
    """
    Calculate zonal derivative of given field on a map (lat/lon coords)

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-03-30

    Args:
        data - data for which derivative is to be computed [(time), lat, lon]
        lat - latitudes of size [nlat, ]
        lonGrid - longitudes of size [nlon, ]

    Returns:
        ddatadx - zonal deerivative of data

    Notes:
        N/A
    """

    # Set constants
    Re = getplanetradius('Earth')  # Radius of Earth (m)

    # For testing purposes
    # u = -cases[jCase]['TAUX'].data  # [t, :, :]
    # v = -cases[jCase]['TAUY'].data  # [t, :, :]

    # Create lat/lon grids from vectors
    lonG, latG = np.meshgrid(lon, lat)

    # Compute 2dx for each grid point
    dlonG = np.roll(lonG, -1, axis=-1) - np.roll(lonG, 1, axis=-1)
    dlonG[:, 0] = dlonG[:, 0] + 360
    dlonG[:, -1] = dlonG[:, -1] + 360
    dx = dlonG * Re * latG * np.pi/180.

    # Compute data(i+1) - data(i-1)
    # i = x index
    ddata = np.roll(data, -1, axis=-1) - np.roll(data, 1, axis=-1)

    # Compute zonal derivative (d/dx)
    ddatadx = ddata/dx

    return ddatadx


def calccurl_m(u, v,
               lat, lon,
               cyclon_flag=False):
    """
    Calculate vorticity from a given u, v vector field
        on a map (lat/lon coords)

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-03-25

    Args:
        u - u vector
        v - v vector
        lat - latitudes of size [nlat, ]
        lonGrid - longitudes of size [nlon, ]
        cyclon_flag - True to return vorticity as cyclonic positive rather
            than counterclockwise positive

    Returns:
        curl of given vector field as either
        if cyclon_flag:
            cyclonic positive (-curl(lat <0))
        else:
            counterclockwise positive

    Notes:
        N/A
    """

    # Create lat/lon grids from vectors
    lonG, latG = np.meshgrid(lon, lat)

    # Compute derivatives
    dvdx = calcddlon_m(v, lat, lon)
    dudy = calcddlat_m(u, lat, lon)

    # Compute vorticity (dv/dx - du/dy)
    curl = dvdx - dudy

    # Multiply by sign(latitude) to make "cyclonic" rather than
    #   "counterclockwise" positive
    cycl = curl * np.sign(latG)

    # Return requested version of vorticity
    if cyclon_flag:
        return cycl
    else:
        return curl


def calcdiv_m(u, v,
              lat, lon):
    """
    Calculate divergence from a given u, v vector field
        on a map (lat/lon coords)

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-03-30

    Args:
        u - u vector
        v - v vector
        lat - latitudes of size [nlat, ]
        lonGrid - longitudes of size [nlon, ]

    Returns:
        divergence of given vector field

    Notes:
        N/A
    """

    # Compute derivatives of vector components
    dudx = calcddlon_m(u, lat, lon)
    dvdy = calcddlat_m(v, lat, lon)

    # Compute divergence (du/dx + dv/dy)
    div = dudx + dvdy

    return div


def calcensmean(cases,
                ensBases,
                caseDim=None,
                verbose_flag=False,
                ):
    """
    Compute mean from ensemble set in case structure

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-07-07

    Args:
        cases - dictionary of simulation dictionaries of variables
            e.g. cases.keys() = ['ctrl0', 'ctrl1', 'nodc0', 'nodc1']
                 cases['ctrl0'].keys() = ['PRECT', 'TAUX']
        ensBases - base ids for ensembles
            e.g. ['ctrl', 'nodc']
        caseDim - dictionary of cases dictionaries of dimensions
        verbose_flag - True to print out status as means are being computed

    Returns:
        cases - case dictionary now including ensemble means

    Notes:
        N/A

    """

    # Loop through ensembles
    for ensBase in ensBases:
        ensCases = [cases.keys()[j]
                    for j in range(len(cases.keys()))
                    if all([(ensBase in cases.keys()[j][:len(ensBase)]),
                            ('mean' not in cases.keys()[j]),
                            strisint(cases.keys()[j][len(ensBase):])
                            ])]
        if verbose_flag:
            print(ensCases)

        # Skip to next ensBase if there are no members for the current one
        #   or ensemble mean already exists
        if ((not ensCases) | ((ensBase + 'mean') in cases.keys())):
            continue

        # Create new case for ensemble means
        cases[ensBase + 'mean'] = dict()

        # Loop through variables
        for var in cases[ensCases[0]].keys():
            if verbose_flag:
                print('Computing ' + var + ' ens. mean')
            # if var == 'RHO':
            #    continue
            # Compute ensemble mean, preserving masking if present
            if np.ma.is_masked(cases[ensCases[0]][var].data):
                meanData = np.ma.mean([cases[j][var].data
                                       for j in ensCases], axis=0)
            else:
                meanData = np.mean([cases[j][var].data
                                    for j in ensCases], axis=0)

            # Store as new case
            cases[ensBase + 'mean'][var] = \
                likenclatlonvar(meanData,
                                name=var,
                                units=cases[ensCases[0]][var].units,
                                src=cases[ensCases[0]][var].src,
                                srcid=(ensBase + '_mean')
                                )

            # Add dimensional information to ensemble mean
            if caseDim is not None:
                for dimName in caseDim[ensCases[0]].keys():
                    setattr(cases[ensBase + 'mean'][var], dimName,
                            caseDim[ensCases[0]][dimName])
            for attr in cases[ensCases[0]][var].__dict__.keys():
                if attr not in cases[ensBase + 'mean'][var].__dict__.keys():
                    setattr(cases[ensBase + 'mean'][var], attr,
                            getattr(cases[ensCases[0]][var], attr))

    return cases


def calcensstd(cases,
               ensBases,
               caseDim=None
               ):
    """
    Compute standard deviation from ensemble set in case structure

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-03-29

    Args:
        cases - dictionary of simulation dictionaries of variables
            e.g. cases.keys() = ['ctrl0', 'ctrl1', 'nodc0', 'nodc1']
                 cases['ctrl0'].keys() = ['PRECT', 'TAUX']
        ensBases - base ids for ensembles
            e.g. ['ctrl', 'nodc']
        caseDim - dictionary of cases dictionaries of dimensions
    Returns:
        cases - case dictionary now including ensemble standard deviations

    Notes:
        N/A

    """

    # Loop through ensembles
    for ensBase in ensBases:
        ensCases = [cases.keys()[j]
                    for j in range(len(cases.keys()))
                    if ((ensBase in cases.keys()[j]) &
                        ('mean' not in cases.keys()[j]))]

        # Skip to next ensBase if there are no members for the current one
        if ((not ensCases) | ((ensBase + 'std') in cases.keys())):
            continue

        # Create new case for ensemble means
        cases[ensBase + 'std'] = dict()

        # Loop through variables
        for var in cases[ensCases[0]].keys():

            # Compute ensemble mean
            stdData = np.std([cases[j][var].data
                              for j in ensCases], axis=0)
            # Store as new case
            cases[ensBase + 'std'][var] = \
                likenclatlonvar(stdData,
                                name=var,
                                units=cases[ensCases[0]][var].units,
                                src=cases[ensCases[0]][var].src,
                                srcid=(ensBase + '_std')
                                )

            # Add dimensional information to ensemble mean
            if caseDim is not None:
                for dimName in caseDim[ensCases[0]].keys():
                    setattr(cases[ensBase + 'std'][var], dimName,
                            caseDim[ensCases[0]][dimName])
    return cases


def calcprectda(ds):
    """
    Create PRECT dataArray in ds

    Notes:
        PRECT = PRECC + PRECL
    """

    # Ensure PRECC and PRECL have the same units
    if ds['PRECC'].units != ds['PRECL'].units:
        raise Exception('Units for PRECC and PRECL do not match')

    # Create dataArray for PRECT as the sum of PRECC and PRECL
    prectDa = xr.DataArray(ds['PRECC'] + ds['PRECL'],
                           attrs=ds['PRECC'].attrs
                           )

    # Update long_name for PRECT
    prectDa.attrs['long_name'] = ('Total precipitation rate (liq + ice)')

    return prectDa


def calcdaregmean(da,
                  gwDa=None,
                  landFracDa=None,
                  latLim=np.array([-90, 90]),
                  lonLim=np.array([0, 360]),
                  ocnOnly_flag=False,
                  qc_flag=False,
                  stdUnits_flag=True,
                  ):
    """
    Compute regional mean for a given dataarray

    Version Date:
        2017-10-25

    Args:
        da - data array for which regional mean is to be computed

    Kwargs:
        gwDs - gaussian weights data array (for CESM output) for weighting by
            latitude
        landFracDs - land fraction dataset for selecting only ocean points
            defined as landFrac == 0
        latLim - np array of latitude limits
        lonLim - np array of longitude limits
        ocnOnly_flag - True to use landFracDs to omit all land points from
            regional mean computation
        qc_flag - True to make extra plots for quality control purposes
        stdUnits_flag - True to convert to standard units before taking
            regional mean
    """

    # Convert ds to standard units
    if stdUnits_flag:
        (da.values,
         da.attrs['units']) = convertunit(
            da.values,
            da.units,
            getstandardunits(da.name)
            )

    # Ensure latitude limits and latitudes in same direction
    if any([(da.lat[0] > da.lat[1]) and (latLim[0] < latLim[1]),
            (da.lat[0] < da.lat[1]) and (latLim[0] > latLim[1])]):
        latLim = latLim[::-1]
        print('flipping latLim')

    # Filter to ocean only if requested
    if ocnOnly_flag:
        # Create ocean filter as True wherever the is no land
        ocnFilt = (landFracDa.values == 0)
        # Set values in ds to nan wherever there is some land
        da.values[~ocnFilt] = np.nan

    # Pull area weights
    if gwDa is not None:
        areaWeights = gwDa.loc[dict(lat=slice(latLim[0], latLim[-1]))]
        if areaWeights.shape[-1] == 0:
            raise ValueError('Invalide latitude limits')

        # Pull data for region of interest
        regDa = da.loc[dict(lat=slice(latLim[0], latLim[-1]),
                            lon=slice(lonLim[0], lonLim[1]))]

        # Plot data to ensure right region selected
        if qc_flag:
            plt.figure()
            plt.contourf(regDa.lon, regDa.lat, regDa[0, :, :])
            plt.colorbar()
            plt.title('Subset data')

        # Compute sum of weighted data
        regWDa = regDa * areaWeights
        regWDa = regWDa.sum(dim=('lat', 'lon'))

        # Create array for grid weights as grid (add dummy dim to do 3D x 2D)
        areaWeightsGrid = xr.full_like(regDa, 1)*areaWeights

        # Retrieve weights only wherever nan is not found in regDs
        areaWeightsGrid = areaWeightsGrid.where(~np.isnan(regDa.values))

        if qc_flag:
            plt.figure()
            plt.contourf(regDa.lon, regDa.lat, np.isnan(regDa.values)[0, :, :])
            plt.figure()
            plt.contourf(regDa.lon, regDa.lat, areaWeightsGrid[0, :, :])
            plt.colorbar()
            plt.title('Area Weights')

        # Sum area weights over region
        sumAreaWeights = areaWeightsGrid.sum(dim=('lon', 'lat'))

        # Compute regional mean
        regMeanDa = regWDa/sumAreaWeights

        # Add units back to regMeanDs
        regMeanDa.attrs['units'] = da.units
        regMeanDa.attrs['long_name'] = da.long_name
        regMeanDa = regMeanDa.rename(da.name)
    else:
        # Pull data for region of interest
        regDa = da.loc[dict(lat=slice(latLim[0], latLim[-1]),
                            lon=slice(lonLim[0], lonLim[1]))]

        # Create areaWeights as function of latitude
        areaWeights = np.cos(np.deg2rad(regDa.lat))

        # Compute sum of weighted data (xarray defaults to nansum)
        regWDa = regDa * areaWeights
        regWDa = regWDa.sum(dim=('lat', 'lon'))

        # Create array for grid weights as grid
        areaWeightsGrid = xr.full_like(regDa, 1)*areaWeights
        # Retreive weights only where regDa is not nan
        # areaWeightsGrid.values[np.isnan(regDa.values)] = np.nan
        areaWeightsGrid = areaWeightsGrid.where(~np.isnan(regDa.values))
        # Sum area weights over region
        sumAreaWeights = areaWeightsGrid.sum(dim=('lon', 'lat'))

        # Compute regional mean
        regMeanDa = regWDa/sumAreaWeights.values

        # Add units back to regMeanDs
        regMeanDa.attrs['units'] = da.units
        regMeanDa.attrs['long_name'] = da.long_name
        regMeanDa = regMeanDa.rename(da.name)

    return regMeanDa


def calcdaregzonmean(da,
                     gwDa=None,
                     landFracDa=None,
                     latLim=np.array([-90, 90]),
                     lonLim=np.array([0, 360]),
                     ocnOnly_flag=False,
                     qc_flag=False,
                     stdUnits_flag=True,
                     ):
    """
    Compute regional, zonal mean for a given dataarray

    Version Date:
        2017-10-31

    Args:
        da - data array for which regional zonal mean is to be computed

    Kwargs:
        gwDs - gaussian weights data array (for CESM output) for weighting by
            latitude
        landFracDs - land fraction dataset for selecting only ocean points
            defined as landFrac == 0
        latLim - np array of latitude limits
        lonLim - np array of longitude limits
        ocnOnly_flag - True to use landFracDs to omit all land points from
            regional mean computation
        qc_flag - True to make extra plots for quality control purposes
        stdUnits_flag - True to convert to standard units before taking
            regional mean
    """

    # Convert ds to standard units
    if stdUnits_flag:
        (da.values,
         da.attrs['units']) = convertunit(
            da.values,
            da.units,
            getstandardunits(da.name)
            )

    # Ensure latitude limits and latitudes in same direction
    if any([(da.lat[0] > da.lat[1]) and (latLim[0] < latLim[1]),
            (da.lat[0] < da.lat[1]) and (latLim[0] > latLim[1])]):
        latLim = latLim[::-1]
        print('flipping latLim')

    # Filter to ocean only if requested
    if ocnOnly_flag:
        # Create ocean filter as True wherever the is no land
        ocnFilt = (landFracDa.values == 0)
        # Set values in ds to nan wherever there is some land
        da.values[~ocnFilt] = np.nan

    # Pull area weights
    if gwDa is not None:
        areaWeights = gwDa.loc[dict(lat=slice(latLim[0], latLim[-1]))]
        if areaWeights.shape[-1] == 0:
            raise ValueError('Invalide latitude limits')

        # Pull data for region of interest
        regDa = da.loc[dict(lat=slice(latLim[0], latLim[-1]),
                            lon=slice(lonLim[0], lonLim[1]))]

        # Plot data to ensure right region selected
        if qc_flag:
            plt.figure()
            plt.contourf(regDa.lon, regDa.lat, regDa[0, :, :])
            plt.colorbar()
            plt.title('Subset data')

        # Compute sum of weighted data
        regWDa = regDa * areaWeights
        regWDa = regWDa.sum(dim=('lon'))

        # Create array for grid weights as grid (add dummy dim to do 3D x 2D)
        areaWeightsGrid = xr.full_like(regDa, 1)*areaWeights

        # Retrieve weights only wherever nan is not found in regDs
        areaWeightsGrid = areaWeightsGrid.where(~np.isnan(regDa.values))

        if qc_flag:
            plt.figure()
            plt.contourf(regDa.lon, regDa.lat, np.isnan(regDa.values)[0, :, :])
            plt.figure()
            plt.contourf(regDa.lon, regDa.lat, areaWeightsGrid[0, :, :])
            plt.colorbar()
            plt.title('Area Weights')

        # Sum area weights over region
        sumAreaWeights = areaWeightsGrid.sum(dim=('lon'))

        # Compute regional mean
        regMeanDa = regWDa/sumAreaWeights

        # Add units back to regMeanDs
        regMeanDa.attrs['units'] = da.units
        regMeanDa.attrs['long_name'] = da.long_name
        regMeanDa = regMeanDa.rename(da.name)
    else:
        # Pull data for region of interest
        regDa = da.loc[dict(lat=slice(latLim[0], latLim[-1]),
                            lon=slice(lonLim[0], lonLim[1]))]

        # Create areaWeights as function of latitude
        areaWeights = np.cos(np.deg2rad(regDa.lat))

        # Compute sum of weighted data (xarray defaults to nansum)
        regWDa = regDa * areaWeights
        regWDa = regWDa.sum(dim=('lon'))

        # Create array for grid weights as grid
        areaWeightsGrid = xr.full_like(regDa, 1)*areaWeights
        # Retreive weights only where regDa is not nan
        # areaWeightsGrid.values[np.isnan(regDa.values)] = np.nan
        areaWeightsGrid = areaWeightsGrid.where(~np.isnan(regDa.values))
        # Sum area weights over region
        sumAreaWeights = areaWeightsGrid.sum(dim=('lon'))

        # Compute regional mean
        regMeanDa = regWDa/sumAreaWeights.values

        # Add units back to regMeanDs
        regMeanDa.attrs['units'] = da.units
        regMeanDa.attrs['long_name'] = da.long_name
        regMeanDa = regMeanDa.rename(da.name)

    return regMeanDa


def calcdsctindex(ds,
                  indexType='Woelfleetal2017',
                  sstVar='TS',
                  ):
    """
    Compute cold tongue index for a given dataset and return as dataArray
    """

    if indexType.lower() in ['woelfle', 'woelfleetal2017']:
        # Compute regional mean through time along equator
        regMeanDs = calcdsregmean(ds[sstVar],
                                  gwDa=(ds['gw']
                                        if 'gw' in ds
                                        else None),
                                  latLim=np.array([-3, 3]),
                                  lonLim=np.array([180, 220]),
                                  ocnOnly_flag=(True
                                                if 'LANDFRAC' in ds
                                                else False),
                                  landFracDa=(ds['LANDFRAC']
                                              if 'LANDFRAC' in ds
                                              else None),
                                  stdUnits_flag=True,
                                  )

        # Compute reference regional mean over greater tropical Pacific
        refRegMeanDs = calcdsregmean(ds[sstVar],
                                     gwDa=(ds['gw']
                                           if 'gw' in ds
                                           else None),
                                     latLim=np.array([-20, 20]),
                                     lonLim=np.array([150, 250]),
                                     ocnOnly_flag=(True
                                                   if 'LANDFRAC' in ds
                                                   else False),
                                     landFracDa=(ds['LANDFRAC']
                                                 if 'LANDFRAC' in ds
                                                 else None),
                                     stdUnits_flag=True,
                                     )

        # Compute CTI
        ctDs = regMeanDs - refRegMeanDs
    else:
        raise ValueError('Unknown indexType, {:s}, '.format(indexType) +
                         'for computing cold tongue index')

    return ctDs


def calcdsditczindex(ds,
                     indexType='Bellucci2010',
                     precipVar='PRECT',
                     ):
    """
    Compute double-ITCZ index for a given dataset and return as data array
   """

    if indexType.lower() in ['bellucci2010', 'bellucci10']:
        regMeanDs = calcdsregmean(ds[precipVar],
                                  gwDa=(ds['gw']
                                        if 'gw' in ds
                                        else None),
                                  latLim=np.array([-20, 0]),
                                  lonLim=np.array([210, 260]),
                                  ocnOnly_flag=False,
                                  stdUnits_flag=True,
                                  )
    else:
        raise ValueError('Unknown indexType, {:s}, '.format(indexType) +
                         'for computing double-ITCZ index')

    return regMeanDs


def convertsigmatopres(inData,
                       ps,
                       newlevs,
                       hCoeffs=None,
                       modelid='cesm',
                       verbose_flag=False):
    """
    Convert data from hybrid sigma vertical coordinate to pressure vertical
    coordinate.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)
        (adapted from MATLAB code written by Dan Vimont:
            http://www.aos.wisc.edu/~dvimont/matlab/)

    Version Date:
        2017-06-13

    Args:
        in - data to be converted to pressure levels from hybrid sigma levels
        ps - surface pressure data
            (CAREFUL:  we will attempt to convert this to hPa from Pa).
        newlevs - pressure levels to which the data is to be interpolated (hPa)
        hCoeffs - dictionary of hybrid sigma coefficients; loaded from file
            if not provided.
        modelid - source model ('am2', 'cesm')
        verbose_flag - True to output which levels are being computed

    Returns:
        outData - inData on pressure levels instead of hybrid sigma levels

    Conventions are:
        1.  Assume ps is in Pa (we will convert this to hPa)
        2.  Assume newlev is in hPa
        3.  [ntim, nlev, nlat, nlon] = size(in);
        4.  [ntim, nlat, nlon] = size(ps);
        5.  nlev2 = length(newlevs);
        6.  [ntim, nlev2, nlat, nlon] = size(out);
        Note that for the above, singleton dimensions should be preserved.
    """

    # Make sure newlevs are in Pa
    if np.round(np.log10(np.mean(newlevs[:]))) == 3:
        newlevs = newlevs*100

    # True if coefficients are for interfaces not box centers
    int_flag = False

    # Get model level coefficients
    if hCoeffs is None:
        hyam, hybm = gethybsigcoeff(modelid=modelid)[2:]
    else:
        try:
            hyam = hCoeffs['hyam']
            hybm = hCoeffs['hybm']
        except KeyError:
            raise KeyError('Can''t find hyam/hybm in dictionary')

    # Set reference pressure value
    # pnot = 1000;

    # Allocate arrays for levels (old and new)
    nlev = hyam.size
    newlevs = newlevs[:]
    nlev2 = newlevs.size

    # Go through argument check.
    dims_ps = np.ndim(ps)
    dims_in = np.ndim(inData)
    # shape_ps = ps.shape
    shape_in = inData.shape

    if dims_ps != (dims_in - 1):
        # print('Variable ''inData'' should have one more dimension than' +
        #       ' ''ps''.')
        raise ValueError('Variable ''inData'' should have one more dimension' +
                         ' than ''ps''.')
        # return

    if shape_in[1] == (nlev - 1):
        int_flag = True
        nlevi = nlev
        nlev = nlevi - 1
    elif shape_in[1] != nlev:
        # print('Either ''inData'' has an inconsistent level dimension, or ' +
        #       'the order (ntim, nlev, nlat, nlon) is wrong.')
        raise ValueError('Either ''inData'' has an inconsistent level ' +
                         'dimension, or the order (ntim, nlev, nlat, nlon) ' +
                         'is wrong.')
        # return

    #  Reorder dimensions so level is the first dimension (move time to end)
    inData = np.rollaxis(inData, 0, 4)

    # Reshape for some reason...
    inData = inData.reshape([shape_in[1], np.prod(shape_in)/nlev])

    # Reshape surface pressure field to match reshaped inData
    ps = np.rollaxis(ps, 0, 3)
    ps = ps.reshape([1, ps.size])

    #  Check for size consistency between inData and ps
    if ps.shape[1] != inData.shape[1]:
        # print('Variables ''in'' and ''ps'' have inconsistent dimensions')
        raise ValueError('Variables ''inData'' and ''ps'' have inconsistent ' +
                         'dimensions')
        return

    # Determinenumber of points (columns) to regrid
    npts = np.prod(shape_in)/nlev

    #  Now that ps is 2D, make sure units are in Pa
    if np.round(np.log10(np.mean(np.mean(ps)))) == 3:
        ps = ps*100

    #  Now, do interpolation
    outData = np.empty([nlev2, np.prod(shape_in)/shape_in[1]])

    # Get size of hyam, hybm
    nHyam = hyam.size

    # Matrix multiplication
    plevs = (np.dot(hyam.reshape([nHyam, 1]), np.ones_like(ps)) +
             np.dot(hybm.reshape([nHyam, 1]), ps))

    # Convert from interface pressures to box center pressures if needed. Use
    #   linear interpolation between interface (aka: use mean of interfaces
    #   as pressure for box center.)
    if int_flag:
        plevs = (plevs[1:, :] + plevs[:-1, :])/2.

    # Loop through each new level to which data is to be interpolated
    for jNlev2 in range(nlev2):
        if verbose_flag:
            print('Computing values for {:0.0f}'.format(newlevs[jNlev2]/100.) +
                  ' hPa')
        for jNPts in range(npts):

            # Find the index of the level just below that to be interpolated
            xup = np.sum(plevs[:, jNPts] < newlevs[jNlev2]) - 1

            # linearly interpolate to new pressure level for each column
            # Xnew = X0 + (ln(Pnew) - ln(P0)) * (X1 - X0) / (ln(P1) - ln(P0))
            if xup < (nlev-1):
                outData[jNlev2, jNPts] = \
                    (inData[xup, jNPts] +
                     (np.log(newlevs[jNlev2]) - np.log(plevs[xup, jNPts])) *
                     (inData[xup + 1, jNPts] - inData[xup, jNPts]) /
                     (np.log(plevs[xup + 1, jNPts]) -
                      np.log(plevs[xup, jNPts])
                      )
                     )

    #  Reshape back to [height, lat, lon, time]
    outData = outData.reshape([nlev2, shape_in[2], shape_in[3], shape_in[0]])

    # Reorder dimensions back to [time, height, lat, lon]
    outData = np.rollaxis(outData, 3, 0)

    # Return data interpolated to defined pressure levels
    return outData


def convertsigmatopres_mp(inTuple):
    """
    Wrapper for multiprocessing of convertsigmatopres

    Version Date: 2017-09-06
    """

    return convertsigmatopres(inTuple[0],
                              inTuple[1],
                              inTuple[2],
                              modelid=inTuple[3],
                              hCoeffs=inTuple[4],
                              )


def convertunit(inData, inUnit, outUnit,
                verbose_flag=False):
    """
    Convert units for geophysical data

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-10-17

    Args:
        inData - ndarray of data to be converted
        inUnit - string of current data unit
        outUnit - string of unit to which data is to be converted

    Kwargs:
        verbose_flag - True for extra status outputs

    Returns:
        outData - data in outUnits
        outUnit - string of unit to which data was converted
    """

    # Convert inUnit and outUnit to standard values for lookup
    inUnit = getstandardunitstring(inUnit)
    outUnit = getstandardunitstring(outUnit)

    # Return data if inUnit == outUnit, i.e. no conversion necessary
    if inUnit == outUnit:
        return (inData, inUnit)

    # Define constants
    rhow = 1000.     # kg/m3
    lv = 2.5e6       # J/kg

    # Define basic conversion factors
    cmperm = 100.    # cm/m
    dynepern = 1e5   # dyne/N
    mmperm = 1000.   # mm/m
    mgperkg = 1e6    # mg/kg
    paperhpa = 100.  # Pa/hPa
    sperd = 86400.   # s/d

    # Attempt to perform multiplicative conversions
    try:
        conversionFactor = {('kg/m2/s', 'mm/d'): sperd*mmperm/rhow,
                            ('kg/m2/s', 'W/m2'): lv,
                            ('kg/m2/s', 'm/s'): 1./rhow,
                            ('mm/d', 'kg/m2/s'): rhow/sperd/mmperm,
                            ('mm/d', 'W/m2'): lv*rhow/sperd/mmperm,
                            ('mm/d', 'm/s'): 1./mmperm/sperd,
                            ('m/s', 'kg/m2/s'): rhow,
                            ('m/s', 'mm/d'): mmperm*sperd,
                            ('m/s', 'W/m2'): rhow*lv,
                            ('W/m2', 'kg/m2/s'): 1./lv,
                            ('W/m2', 'mm/d'): mmperm*sperd/rhow/lv,
                            ('W/m2', 'm/s'): 1./rhow/lv,
                            ('mg/m2/s', 'mm/d'): sperd*mmperm/rhow/mgperkg,
                            ('Pa', 'hPa'): 1./paperhpa,
                            ('dyne/centimeter^2', 'N/m2'): (cmperm**2 *
                                                            1/dynepern),
                            ('dyne/centimeter^2', 'N m$^-2$'): (cmperm**2 *
                                                                1/dynepern),
                            }[(inUnit, outUnit)]

        # Convert unit and return new values
        if verbose_flag:
            print('Converting ' + inUnit + ' to ' + outUnit)
        return (inData*conversionFactor, outUnit)

    # Perform other conversions
    except KeyError:
        try:
            conversionOffset = {('$^\circ$C', 'K'): 273.15,
                                ('deg. C', 'K'): 273.15,
                                ('degC', 'K'): 273.15,
                                ('C', 'K'): 273.15,
                                ('K', 'degC'): -273.15,
                                ('K', 'deg. C'): -273.15,
                                ('K', 'C'): -273.15,
                                ('K', '$^\circ$C'): -273.15,
                                }[(inUnit, outUnit)]

            # Convert unit and return new values
            if verbose_flag:
                print('Converting ' + inUnit + ' to ' + outUnit)
            return (inData + conversionOffset, outUnit)

        except KeyError:
            # Return inData if units are unchanged
            if ((inUnit in ['$^\circ$C', 'deg. C', 'degC', 'C']) and
                    (outUnit in ['$^\circ$C', 'deg. C', 'degC', 'C'])):
                if verbose_flag:
                    print('No unit conversion performed')
                return (inData, inUnit)
            else:
                raise KeyError('Could not convert ' +
                               inUnit + ' to ' + outUnit)


def findlast(inString, subString):
    """
    Returns index of last instance of subString in inString
    """
    lastInd = (len(inString) - inString[::-1].find(subString[::-1]) -
               (len(subString)))
    return lastInd


def gethybsigcoeff(modelid='cesm', coordFile=None):
    """
    Obtain hybrid sigma coefficients for finding pressure levels of model
        interfaces and model box centers
    2015-08-27
    """
    # If coordFile not provided, find file for loading coefficients based on
    #    server
    if coordFile is None:
        if gethostname() in ['stable', 'challenger', 'p', 'fog']:
            coordFile = ('/home/disk/p/woelfle/MATLAB/common/' + modelid +
                         '.hybsigcoeffs.nc')
        elif gethostname() == 'woelfle-laptop':
            print('No coordinate file defined for woelfle-laptop')
            return
        elif gethostname()[0:5] in ['yslogi', 'geyser']:
            print('No coordinate file defined for ncar machines')

    # Set variable names for hybrid sigma coefficients
    if modelid == 'cesm':
        hybcoeffs = ['hyai', 'hybi', 'hyam', 'hybm']
    elif modelid == 'am2':
        hybcoeffs = ['pk', 'bk']

    # Load coefficients from file
    with nc4.Dataset(coordFile, 'r') as ncDataset:
        # Interface levels
        hyai = ncDataset.variables[hybcoeffs[0]][:]
        hybi = ncDataset.variables[hybcoeffs[1]][:]

        # Box center levels
        if modelid == 'cesm':
            hyam = ncDataset.variables[hybcoeffs[2]][:]
            hybm = ncDataset.variables[hybcoeffs[3]][:]
        else:
            hyam = None
            hybm = None

    return (hyai, hybi, hyam, hybm)


class ncgenvar:
    """
    Define generic netCDF wrapper with no methods.
    2015-07-06
    """
    def __init__(self, ncVarObj, path=None, srcid=None):
        self.name = ncVarObj._name
        self.data = ncVarObj[:]
        if hasattr(ncVarObj, 'units'):
            self.units = ncVarObj.units
        else:
            self.units = None
        self.src = path
        self.srcid = srcid

    pass


class nclatlonvar:
    """
    Define netCDF wrapper for lat-lon datasets to allow easier analysis.

    Args:
        ncVarObj - variable from netCDF4 dataset
            e.g. nc4.Dataset(filepath).variables[varName]
        path - full path of source file
        srcid - short ID for source case
            e.g. 'CTRL', 'NoDC', 'GPCP', 'SODA'

    Version Date:
        2015-07-06
    """

    # Initialize the class
    def __init__(self,
                 ncVarObj,
                 path=None,
                 srcid=None,
                 tMax=None):
        self.name = ncVarObj._name
        self.data = ncVarObj[:tMax, ...]
        if hasattr(ncVarObj, 'units'):
            self.units = ncVarObj.units
        else:
            self.units = None
        self.src = path
        self.srcid = srcid

    # Define mean taking functions
    # Global mean
    def calcglobalmean(self, planet='Earth'):
        """
        Call calcregmean on this variable for all lat, lon, and time
        """
        return calcregmean(self.data, self.lat, self.lon,
                           self.latb, self.lonb,
                           latLim=[-90, 90],
                           lonLim=[0, 360],
                           planet=planet)

    # Global zonal mean
    def calcglobalzonmean(self, planet='Earth'):
        """
        Call calcglobalzonmean on this variable for all lat, lon, and time
        """
        return calcregzonmean(self.data, self.lat, self.lon,
                              self.latb, self.lonb,
                              latLim=[-90, 90],
                              lonLim=[0, 360],
                              planet=planet)

    # Regional mean
    def calcregmean(self, **kwargs):
        """
        Call calcregmean on this variable for the specified lat/lon range, and
            all time
        """
        return calcregmean(self.data, self.lat, self.lon,
                           self.latb, self.lonb,
                           **kwargs)

    # Regional meridional mean
    def calcregmeridmean(self, **kwargs):
        """
        Call calcregmeridmean on this variable for the specified lat/lon range
            and all time
        """
        return calcregmeridmean(self.data, self.lat, self.lon,
                                self.latb, self.lonb,
                                **kwargs)

    # Regional zonal mean
    def calcregzonmean(self, **kwargs):
        """
        Call calcregzonalmean on this variable for the specified lat/lon range
            and all time
        """
        return calcregzonmean(self.data, self.lat, self.lon,
                              self.latb, self.lonb,
                              **kwargs)

    # Area, meridional, and zonal means at once
    def calcallregmeans(self, **kwargs):
        """
        Call calcallregmeans on this variable for the specified lat/lon range
            and all time
        """
        return calcallregmeans(self.data, self.lat, self.lon,
                               self.latb, self.lonb,
                               **kwargs)

    # Compute equilibrium seasonal cycle of zonal mean
    def calceqzoncycle(self, tEq, **kwargs):
        """
        Compute equilibrium seasonal cycle of variable from time tEq to end.

        Args:
            **kwargs - optional inputs into calcregzonmean
        """
        # Compute zonal mean using optional inputs
        regZonMean = self.calcregzonmean(**kwargs)

        # Return equilibrium zonal mean
        return calceqseascycle(tEq, regZonMean)

    # Return attributes as dictionary keys
    def keys(self):
        """
        Return attributes as dictionary keys
        """
        return self.__dict__.keys()


class likenclatlonvar:
    """
    Define class like nclatlonvar without having to load from netcdf

    Args:
        ncVarObj - variable from netCDF4 dataset
            e.g. nc4.Dataset(filepath).variables[varName]
        path - full path of source file
        srcid - short ID for source case
            e.g. 'CTRL', 'NoDC', 'GPCP', 'SODA'

    Version Date:
        2015-07-06
    """

    # Initialize the class
    def __init__(self, data,
                 name=None,
                 src=None,
                 srcid=None,
                 units=None,
                 ):
        self.name = name
        self.data = data
        self.units = units
        self.src = src
        self.srcid = srcid

    # Define mean taking functions
    # Global mean
    def calcglobalmean(self, planet='Earth'):
        """
        Call calcregmean on this variable for all lat, lon, and time
        """
        return calcregmean(self.data, self.lat, self.lon,
                           self.latb, self.lonb,
                           latLim=[-90, 90],
                           lonLim=[0, 360],
                           planet=planet)

    # Global zonal mean
    def calcglobalzonmean(self, planet='Earth'):
        """
        Call calcglobalzonmean on this variable for all lat, lon, and time
        """
        return calcregzonmean(self.data, self.lat, self.lon,
                              self.latb, self.lonb,
                              latLim=[-90, 90],
                              lonLim=[0, 360],
                              planet=planet)

    # Regional mean
    def calcregmean(self, **kwargs):
        """
        Call calcregmean on this variable for the specified lat/lon range, and
            all time
        """
        return calcregmean(self.data, self.lat, self.lon,
                           self.latb, self.lonb,
                           **kwargs)

    # Regional meridional mean
    def calcregmeridmean(self, **kwargs):
        """
        Call calcregmeridmean on this variable for the specified lat/lon range
            and all time
        """
        return calcregmeridmean(self.data, self.lat, self.lon,
                                self.latb, self.lonb,
                                **kwargs)

    # Regional zonal mean
    def calcregzonmean(self, **kwargs):
        """
        Call calcregzonalmean on this variable for the specified lat/lon range
            and all time
        """
        return calcregzonmean(self.data, self.lat, self.lon,
                              self.latb, self.lonb,
                              **kwargs)

    # Area, meridional, and zonal means at once
    def calcallregmeans(self, **kwargs):
        """
        Call calcallregmeans on this variable for the specified lat/lon range
            and all time
        """
        return calcallregmeans(self.data, self.lat, self.lon,
                               self.latb, self.lonb,
                               **kwargs)

    # Compute equilibrium seasonal cycle of zonal mean
    def calceqzoncycle(self, tEq, **kwargs):
        """
        Compute equilibrium seasonal cycle of variable from time tEq to end.

        Args:
            **kwargs - optional inputs into calcregzonmean
        """
        # Compute zonal mean using optional inputs
        regZonMean = self.calcregzonmean(**kwargs)

        # Return equilibrium zonal mean
        return calceqseascycle(tEq, regZonMean)

    # Return attributes as dictionary keys
    def keys(self):
        """
        Return attributes as dictionary keys
        """
        return self.__dict__.keys()


def calcregrmse(data1, data2, lat, lon, latb, lonb,
                latLim=None,
                lonLim=None,
                planet=None
                ):
    """
    Computes the regional root mean squared error over a given area on a
        planet.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-11

    Args:
        data1 - [time x] lat x lon array of data for which rmse is computed
        data2 - [time x] lat x lon array of data for which rmse is computed
        lat - latitude for grid box centers
        lon - longitude for grid box centers
        latb - latitude for grid box edges
        lonb - longitude for grid box edges
        latLim - latitude limits for subset region (default = [-90, 90])
        lonLim - longitude limits for subset region (default = [0, 360])
        planet - planet for which computation is run (default = "earth")

    Returns:
        regRmse - vector of regional rmses for each time step in dataIn

    Raises:
        none

    Notes:
        Assumes latitudes run S to N, e.g. -90 to 90
        Assumes longitudes run E to W, e.g. 0 to 360
        Cannot handle lon limits wrapping 0 degrees
        Now sets nans to zeros when summing values prior to dividing by area
            Should prevent outputting an array of nans without use of nan fns.
    """
    # return np.nan

    # Set default values for limits if not provided by user
    if latLim is None:
        latLim = np.array([-90, 90])
    if lonLim is None:
        lonLim = np.array([0, 360])

    # Find area weights for region
    areaFilt, areaWeight = calcareaweight(lat, lon, latb, lonb,
                                          latLim=latLim,
                                          lonLim=lonLim
                                          )

    # Find total area occupied by data (exclude nans) at each time step
    nonNanArea = (areaWeight*~np.isnan(data1)).sum(axis=-2).sum(axis=-1)

    # Replace input nans with 0s for summing
    nanfilt1 = np.isnan(data1)
    nanfilt2 = np.isnan(data2)
    data1Nonan = data1
    data1Nonan[nanfilt1 | nanfilt2] = 0
    data2Nonan = data2
    data2Nonan[nanfilt1 | nanfilt2] = 0

    # Weight data by areaWeight and sum over region
    # Compute area weighted squared area for each grid cell
    errW = ((data1Nonan - data2Nonan)**2*areaWeight).sum(axis=-2).sum(axis=-1)

    # Compute regional mean of data
    regRmse = np.sqrt(errW / nonNanArea)

    # Return values
    return regRmse


def cgstomks(inData, inUnit,
             verbose_flag=False):
    """
    convert values from cgs system to mks system

    Returns:
        (np array in mks, new units as string)
    """
    # Define basic conversions
    cmperm = 1.e2           # cm/m
    gperkg = 1.e3           # g/kg
    ergperj = 1.e7          # Erg/J
    dynepercm2perpa = 1.e1  # (dyne/cm^2)/Pa

    # Define conversion factors
    # {(inUnit): conversion_factor_to_mks}
    conversionFactor = {'cm': 1./cmperm,
                        'centimeters': 1./cmperm,
                        'centimeter': 1./cmperm,
                        'centimeter/s': 1./cmperm,
                        'centimeter/s^2': 1./cmperm,
                        'centimeter^2': 1/cmperm**2,
                        'centimeter^2/s': 1./cmperm**2.,
                        'cm degC/s': 1/cmperm,
                        'erg/g': gperkg/ergperj,
                        'erg/g/K': gperkg/ergperj,
                        'gram/centimeter^3': cmperm**3./gperkg,
                        'dyne/centimeter^2': 1./dynepercm2perpa,
                        'degC cm/s': 1./cmperm
                        }

    # Convert inData to new units
    try:
        if verbose_flag:
            print('Old units: ' + inUnit + '\n'
                  'Multiplying by ' + str(conversionFactor[inUnit]))
        outData = inData*conversionFactor[inUnit]
    except KeyError:
        outData = inData

    # Define new unit
    newUnit = {'cm': 'm',
               'centimeters': 'm',
               'centimeter': 'm',
               'centimeter/s': 'm/s',
               'centimeter/s^2': 'm/s^2',
               'centimeter^2': 'm^2',
               'centimeter^2/s': 'm^2/s',
               'cm degC/s': 'm degC/s',
               'erg/g': 'J/kg',
               'erg/g/K': 'J/kg/K',
               'gram/centimeter^3': 'kg/m^3',
               'dyne/centimeter^2': 'Pa',
               'degC cm/s': 'degC m/s'
               }

    # Convert inUnit to new units
    try:
        outUnit = newUnit[inUnit]
    except KeyError:
        outUnit = inUnit

    # Return data in new unit system
    return (outData, outUnit)


def getenscaseids(cases,
                  prefix,
                  caseKey=None):
    """
    Obtain keys for ensemble members from cases dictionary
    Returns: keys as list
    """
    if caseKey is None:
        caseKey = prefix + 'mean'

    # Pull keys for ensemble cases
    ensCaseIds = [cases.keys()[j] for j in range(len(cases.keys()))
                  if ((prefix == cases.keys()[j][:len(prefix)]) and
                      (cases.keys()[j] != caseKey))]

    # Return caseids as a list
    return ensCaseIds


def getgridsize(gridName, src='cesm'):
    """
    Return dimensions of a given CESM grid
    """
    if src == 'cesm':
        # Create dictionary of acceptable grids
        gridSize = {'g16':        np.array([384, 320]),
                    'gx1v6':      np.array([384, 320]),
                    'f09':        np.array([192, 288]),
                    'fv0.9x1.25': np.array([192, 288]),
                    'f19':        np.array([96, 144]),
                    'fv1.9x2.5':  np.array([96, 144])}

        return gridSize[gridName]
    else:
        print('unknown source')


def getstandardunits(varName):
    """
    Get standard units for a given variable
    """
    try:
        stdUnits = {'FLDS': 'W/m2',
                    'FLNS': 'W/m2',
                    'FSDS': 'W/m2',
                    'FSNS': 'W/m2',
                    'LHFLX': 'W/m2',
                    'PRECC': 'mm/d',
                    'PRECL': 'mm/d',
                    'PRECT': 'mm/d',
                    'precip': 'mm/d',
                    'PS': 'hPa',
                    'SHFLX': 'W/m2',
                    'sp': 'hPa',
                    'sst': 'K',
                    'TAUX': 'N/m2',
                    'TAUY': 'N/m2',
                    'TREFHT': 'K',
                    'TS': 'K'}[varName]
    except KeyError:
        raise KeyError('Cannot find standard units for ' + varName)

    return stdUnits


def getstandardunitstring(unitString,
                          prependSpace_flag=False):
    """
    Change unit to standard string.
    """
    if prependSpace_flag:
        prependValue = ' '
    else:
        prependValue = ''

    if unitString in ['kg/s/m^2', 'kg/s/m2', 'kg/m2/s', 'kg m^-2 s^-1',
                      'kg/m^2/s']:
        return prependValue + 'kg/m2/s'
    elif unitString in ['m/s', 'm s^-1']:
        return prependValue + 'm/s'
    elif unitString in ['mm/d', 'mm d^-1', 'mm/day']:
        return prependValue + 'mm/d'
    elif unitString in ['W/m2', 'W/m^2', 'W m^-2', 'watt/m^2']:
        return prependValue + 'W/m2'
    elif unitString in ['mg/m2/s', 'mg m^-2 s^-1']:
        return prependValue + 'mg/m2/s'
    elif unitString in ['centimeter/s', 'cm/s', 'cm s^-1']:
        return prependValue + 'cm/s'
    elif unitString in ['meter/s', 'm/s', 'm s^-1', 'm/sec']:
        return prependValue + 'm/s'
    elif unitString in ['deg. C', 'degC', 'C']:
        return r'$^\circ$C'
    elif unitString in ['N/m^2', 'N/m2']:
        return 'N/m2'
    else:
        if unitString is None:
            return ''
        else:
            return unitString


def loadcase2dict(ncPath, loadVars,
                  srcid=None):
    """
    Load list of variables from a given CESM case
    (Unfinished...)
    """

    # Return dictionary of variables loaded for given cesm case
    return 2


def loadnetcdf2class(ncPath, loadVar,
                     srcid=None,
                     tMax=None):
    """
    Load a variable and associated dimensions from netCDF to workspace

    Returns:
        ncObj - nclatlonvar with information about loaded variable
        ncDimDict - dictionary of dimension values
    2016-02-26
    """

    try:
        with nc4.Dataset(ncPath, 'r') as ncDataset:
            dimDict = dict()
            # ncObjDim = ncvar()

            # Pull dimension information

            # Pull names of dimensions from dataset
            dims = ncDataset.dimensions.keys()

            # Load dimensions from netcdf file to dictionary
            for dimName in dims:
                try:
                    dimDict[dimName] = ncDataset.variables[dimName][:]
                except:
                    dimDict[dimName] = None

            # Pull variable information
            try:
                ncObj = nclatlonvar(ncDataset.variables[loadVar],
                                    path=ncPath,
                                    srcid=srcid,
                                    tMax=tMax,
                                    )
            except KeyError as e:
                raise KeyError(e.message + ':\n' + ncPath)

    except RuntimeError as e:
        if e.message == 'No such file or directory':
            raise IOError(e.message + ':\n' + ncPath)
        else:
            raise

    return ncObj, dimDict  # ncObjDim


def loadcmap(years,
             tMax=None,
             verbose=False):
    """
    Load CMAP data from text file

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-09-29

    Args:
        years - years of CMAP precip data to load
        tMax - maximum number of timesteps to load
        verbose - True to write out years and files being loaded

    Returns:
        cmap - dictionary of dimensions and values as follows
            year - year of each time point
            month - month of each time point
            lat - latitudes (deg. E)
            lon - longitues (deg. N)
            rain1 - precipitation rate in mm/d (uses model + sat + gauge)
            error1 - error in rain1 in precetage of total
            rain2 - precipitation rate in mm/d (uses only sat + gauge)
            error2 - error in rain2 in precetage of total

    Notes:
        2016-02-17: Updated to read v1602 of dataset
        2016-09-29: Updated to include tMax

    """

    # Loop through and load each year from file
    for year in years:
        cmapDir = '/home/disk/eos9/woelfle/dataset/CMAP/'
        cmapFile = ('cmap_mon_v1602_{:02.0f}.txt'
                    .format(year - 100 * (year/100)))
        if verbose:
            print(year + ': ' + cmapFile)

        # Open year file and read in all data in one go
        fid = open(cmapDir + cmapFile, 'r')
        a = fid.readlines()
        fid.close()

        # Parse a into one list for each field
        yearIn = np.array([a[j][0:4] for j in range(len(a))],
                          dtype=np.int)
        monIn = np.array([a[j][4:8] for j in range(len(a))],
                         dtype=np.int)
        latIn = np.array([a[j][8:16] for j in range(len(a))],
                         dtype=np.float32)
        lonIn = np.array([a[j][16:24] for j in range(len(a))],
                         dtype=np.float32)
        rain1In = np.array([a[j][24:32] for j in range(len(a))],
                           dtype=np.float32)
        error1In = np.array([a[j][32:40] for j in range(len(a))],
                            dtype=np.float32)
        rain2In = np.array([a[j][40:48] for j in range(len(a))],
                           dtype=np.float32)
        error2In = np.array([a[j][48:56] for j in range(len(a))],
                            dtype=np.float32)

        # Determine lengths for dimensions
        npts = len(a)
        nyear = len(set(yearIn))
        nmon = len(set(monIn))
        nlon = len(set(lonIn))
        nlat = len(set(latIn))

        # Pull unique dimension info
        yearIn = yearIn[np.arange(0, npts, nmon * nlat * nlon)]
        monIn = monIn[np.arange(0, npts / nyear, nlat * nlon)]
        latIn = latIn[np.arange(0, npts / nyear / nmon, nlon)]
        lonIn = lonIn[np.arange(0, npts / nyear / nmon / nlat)]

        # Pull data
        rain1In = rain1In.reshape([nyear * nmon, nlat, nlon])
        error1In = error1In.reshape([nyear * nmon, nlat, nlon])
        rain2In = rain2In.reshape([nyear * nmon, nlat, nlon])
        error2In = error2In.reshape([nyear * nmon, nlat, nlon])

        # Concatenate years
        if year == years[0]:
            yearOut = yearIn
            monOut = monIn
            latOut = latIn
            lonOut = lonIn
            rain1Out = rain1In
            error1Out = error1In
            rain2Out = rain2In
            error2Out = error2In
        else:
            yearOut = np.concatenate([yearOut, yearIn], axis=0)
            monOut = np.concatenate([monOut, monIn], axis=0)
            rain1Out = np.concatenate([rain1Out, rain1In], axis=0)
            error1Out = np.concatenate([error1Out, error1In], axis=0)
            rain2Out = np.concatenate([rain2Out, rain2In], axis=0)
            error2Out = np.concatenate([error2Out, error2In], axis=0)

    # Convert rain to masked array based on missing value in documentation
    missingval = -999.
    if tMax is None:
        tMax = rain1Out.shape[0]
    rain1Out = np.ma.array(rain1Out, mask=(rain1Out == missingval))[:tMax, :, :]
    error1Out = np.ma.array(error1Out, mask=(error1Out == missingval))[:tMax, :, :]
    rain2Out = np.ma.array(rain2Out, mask=(rain2Out == missingval))[:tMax, :, :]
    error2Out = np.ma.array(error2Out, mask=(error2Out == missingval))[:tMax, :, :]

    cmap = {'year': yearOut,
            'mon': monOut,
            'lat': latOut,
            'lon': lonOut,
            'rain1': rain1Out,
            'error1': error1Out,
            'rain2': rain2Out,
            'error1': error2Out}

    return cmap


def loadcore(loadVars,
             yr1,
             newlat=None,
             newlon=None,
             regrid_flag=False,
             tMax=None):
    """
    Load CORE data from text file

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-09-29

    Args:
        loadVars - variables to be loaded (as CESM style)
        yr1 - first year to be loaded (1981, 1986, or 1991) as string
        newlat - latitude for new grid
        newlon - longitude for new grid
        regrid_flag - true to regrid to CESM grid

    Returns:
        core - core data
        coreDims - associated dimensions

    Notes:
        Time referenced to start of run, not generally meaningful

    """
    # Dictionary to translate requested CESM variables to SODA variables
    cesm2coreDict = {'FLDS': 'Q_lwdn',
                     'FLNS': 'Q_lwnet',
                     'FLUS': 'Q_lwup',
                     'FNS': 'Q_net',
                     'FSNS': 'Q_swnet',
                     'LHFLX': 'Q_lat',
                     'PRECT': 'F_prec',
                     'SHFLX': 'Q_sen',
                     'TAUX': 'taux',
                     'TAUY': 'tauy'
                     }

    # Initialize dictionaries to hold CORE variables
    core = dict()
    coreDir = '/home/disk/eos9/woelfle/dataset/COREV2/'
    coreVars = [cesm2coreDict[j] for j in loadVars]

    # Construct full history file paths and load output from netcdfs
    for (jVar, loadVar) in enumerate(coreVars):
        print('Loading ' + loadVar + '...')
        ncFileCase = ('corev2.' + yr1 + '.' + loadVar + '.nc')
        corePath = (coreDir + ncFileCase)
        core[loadVars[jVar]], coreDim = (
            loadnetcdf2class(corePath, loadVar,
                             srcid='CORE',
                             tMax=tMax))

        if loadVars[jVar] in ['PRECT']:
            core[loadVars[jVar]].data = \
                convertunit(core[loadVars[jVar]].data,
                            core[loadVars[jVar]].units,
                            'mm/d'
                            )
            core[loadVars[jVar]].units = 'mm/d'

    # Update dimension dictionaries to include latitude and longitude edge
    #   info
    dLatb = coreDim['lat'][1] - coreDim['lat'][0]
    latbMin = coreDim['lat'][0] - dLatb/2.
    latbMax = coreDim['lat'][-1] + dLatb/2.
    coreDim['latb'] = np.arange(latbMin, latbMax+dLatb/2., dLatb)

    dLonb = coreDim['lon'][1] - coreDim['lon'][0]
    lonbMin = coreDim['lon'][0] - dLonb/2.
    lonbMax = coreDim['lon'][-1] + dLonb/2.
    coreDim['lonb'] = np.arange(lonbMin, lonbMax+dLonb/2., dLonb)

    # Add dimension info to each variable
    for loadVar in core.keys():
        for dimName in coreDim.keys():
            setattr(core[loadVar], dimName, coreDim[dimName])

    # REGRID TO CESM GRID
    if regrid_flag:
        # Assume regridding to CESM f19 grid if no grid given
        if newlat is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlat = ncDataset.variables['lat'][:]
        if newlon is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlon = ncDataset.variables['lon'][:]

        for regridVar in core.keys():
            # Regrid variables to new grid using linear interpolation
            core[regridVar].data = \
                np.array([interpolate.interp2d(
                    coreDim['lon'],
                    coreDim['lat'],
                    core[regridVar].data[j, :, :],
                    kind='linear',
                    copy=True,
                    bounds_error=False,
                    fill_value=np.nan)(newlon, newlat)
                    for j in range(core[regridVar].data.shape[0])])

            # Assign new dimensions
            setattr(core[regridVar], 'lat', newlat)
            setattr(core[regridVar], 'lon', newlon)
            setattr(core[regridVar], 'latb', None)
            setattr(core[regridVar], 'lonb', None)

    return (core, coreDim)


def loaderai(daNewGrid=None,
             kind='linear',
             loadClimo_flag=False,
             newGridFile=None,
             newGridName=None,
             newLat=None,
             newLon=None,
             qc_flag=False,
             regrid_flag=False,
             whichErai='monmean',
             ):
    """
    Load ERAI data from netcdf file

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-11-02

    Args:
        N/A

    Kwargs:
        daNewGrid - dataArray with new grid for regridding
        kind - method to use for regridding
        loadClimo_flag - True to only load climatology
        newGridFile - file with new grid
        newGridName - (str) name of new grid for regridding
            available grids: '0.9x1.25'
        qc_flag - True to make plots for quality control of regridding
        regrid_flag - True to regrid to given new grid
        whichErai - shortform for which file to load
            'monsum' - sum of daily forecasts for monthly means
            # others coming at some point in the future

    Returns:
        eraiDs - xr dataset with fields from ERAI
         (or)
        eraiDsRg - xr dataset with regridded ERAI fields

    Notes:
        Now works with xarray!
        Available fields -
            monmean - ci, sst, sp, msl, tcc, t2m, d2m, lcc, mcc, hcc, si10, skt
                (sea ice cover, SST, Psfc, MSLP, CLOUDFRAC, T2m, TD2m,
                 CLDLOW, CLDMED, CLDHIGH, U10, Tskin)
            monmean.fc - iews, inss, ishf, ie (instantaneous fluxes)
                (taux, tauy, SHF, evap)

    """
    # Set directories for ERAI
    eraiDir = '/home/disk/eos9/woelfle/dataset/ERAI/'
    if whichErai == 'monsum':
        eraiSubDir = 'monsum/'
        if loadClimo_flag:
            print('Not ready yet.')
        else:
            eraiFileList = ['ERAI.1979-1989.monsum.nc',
                            'ERAI.1990-1999.monsum.nc',
                            'ERAI.2000-2009.monsum.nc',
                            # 'ERAI.2010-2010.monsum.nc'
                            ]
    elif whichErai == 'monmean':
        eraiSubDir = 'monmean/'
        if loadClimo_flag:
            eraiFile = 'ERAI.197901-201012.allmonmean.monclimo.nc'
        else:
            eraiFileList = ['ERAI.1979-1989.monmean.nc',
                            'ERAI.1990-1999.monmean.nc',
                            'ERAI.2000-2009.monmean.nc'
                            'ERAI.2010-2010.monmean.nc']

    # Load dataset to file
    if loadClimo_flag:
        eraiDs = xr.open_dataset(eraiDir + eraiSubDir + eraiFile)
    else:
        eraiDs = xr.open_mfdataset([eraiDir + eraiSubDir + eraiFileList[j]
                                    for j in range(len(eraiFileList))])

    # Rename dimensions to match CESM standard
    eraiDs.rename({'latitude': 'lat',
                   'longitude': 'lon'},
                  inplace=True
                  )

    # Add id to dataset
    eraiDs.attrs['id'] = 'ERAI'

    # Regrid if requested
    if regrid_flag:
        print('Cannot regrid yet')
        eraiDsRG = eraiDs
        return eraiDsRG
    else:
        return eraiDs

    # Dictionary to translate requested CESM variables to SODA variables
    cesm2eraiDict = {'PS': 'sp',
                     'PSL': 'msl',
                     'TS': 'sst',
                     'U': 'u',
                     'V': 'v',
                     }

    # Initialize dictionaries to hold CORE variables
    erai = dict()
    eraiDir = '/home/disk/eos9/woelfle/dataset/ERAI/singlevar/'
    eraiVars = [cesm2eraiDict[j] for j in loadVars]

    # Construct full history file paths and load output from netcdfs
    for (jVar, loadVar) in enumerate(eraiVars):
        print('Loading ' + loadVar + '...')
        ncFileCase = ('ERAI.' + yr1 + '.' + loadVar + '.nc')
        eraiPath = (eraiDir + ncFileCase)
        erai[loadVars[jVar]], eraiDim = (
            loadnetcdf2class(eraiPath, loadVar,
                             srcid='ERAI',
                             tMax=tMax))

        # Flip lats from N->S to S->N
        erai[loadVars[jVar]].data = np.flip(erai[loadVars[jVar]].data,
                                            axis=-2)

    # Flip latitude order from N->S to S->N
    eraiDim['latitude'] = np.flip(eraiDim['latitude'],
                                  axis=0)

    # Convert unit names to CESM convention
    eraiDim['lat'] = eraiDim['latitude']
    eraiDim['lon'] = eraiDim['longitude']

    # Update dimension dictionaries to include latitude and longitude edge
    #   info
    dLatb = eraiDim['lat'][1] - eraiDim['lat'][0]
    latbMin = eraiDim['lat'][0] - dLatb/2.
    latbMax = eraiDim['lat'][-1] + dLatb/2.
    eraiDim['latb'] = np.arange(latbMin, latbMax+dLatb/2., dLatb)

    dLonb = eraiDim['lon'][1] - eraiDim['lon'][0]
    lonbMin = eraiDim['lon'][0] - dLonb/2.
    lonbMax = eraiDim['lon'][-1] + dLonb/2.
    eraiDim['lonb'] = np.arange(lonbMin, lonbMax+dLonb/2., dLonb)

    # Add dimension info to each variable
    for loadVar in erai.keys():
        for dimName in eraiDim.keys():
            setattr(erai[loadVar], dimName, eraiDim[dimName])

    # REGRID TO CESM GRID
    if regrid_flag:
        # Assume regridding to CESM f19 grid if no grid given
        if newlat is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlat = ncDataset.variables['lat'][:]
        if newlon is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlon = ncDataset.variables['lon'][:]

        for regridVar in erai.keys():
            if np.ndim(erai[regridVar].data) != 3:
                continue

            # Regrid variables to new grid using linear interpolation
            erai[regridVar].data = \
                np.array([interpolate.interp2d(
                    eraiDim['lon'],
                    eraiDim['lat'],
                    erai[regridVar].data[j, :, :],
                    kind='linear',
                    copy=True,
                    bounds_error=False,
                    fill_value=np.nan)(newlon, newlat)
                    for j in range(erai[regridVar].data.shape[0])])

            # Assign new dimensions
            setattr(erai[regridVar], 'lat', newlat)
            setattr(erai[regridVar], 'lon', newlon)
            setattr(erai[regridVar], 'latb', None)
            setattr(erai[regridVar], 'lonb', None)

    return (erai, eraiDim)


def loadgpcp(loadYrs,
             newlat=None,
             newlon=None,
             regrid_flag=False,
             tMax=None):
    """
    Load 2.5deg GPCP data from binary file

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-07-12

    Args:
        loadYrs - years for which to load data
        newlat - latitudes to which to regrid
        newlon - longitudes to which to regrid
        regrid_flag - true to regrid to CESM grid
        tMax - maximum number of time steps to load

    Returns:
        gpcp - dictionary of gpcp data
        gpcpDim - associated dimensions

    Notes:
        Time referenced to start of run, not generally meaningful
    """
    # Convert input list of years to strings if needed
    try:
        loadYrs = [str(loadYrs[j]) for j in range(len(loadYrs))]
    except:
        raise Exception('Cannot parse loadYrs')

    # Initialize dictionary to hold GPCP data
    gpcpDir = '/home/disk/eos9/woelfle/dataset/GPCP/monthly/'
    gpcpVar = 'PRECT'
    gpcpBase = 'gpcp_v2.2_psg.'

    # Obtain grid information
    lat, lon = gpcploader.get_gpcp_latlon('2DD')

    # Get header information
    header = gpcploader.read_gpcp_header(gpcpDir + gpcpBase + loadYrs[0])

    # Get data collection method
    jTechStart = header.index('technique') + len('technique=')
    jTechEnd = header[jTechStart:].index(' ') + jTechStart
    technique = header[jTechStart:jTechEnd]

    # Load gpcp data from file
    gpcpData = gpcploader.read_multigpcp(gpcpDir, gpcpBase, loadYrs)

    # Subset number of time steps if requested
    if tMax is not None:
        gpcpData = gpcpData[:tMax, :, :]

    # Initialize gpcp as dictionary
    gpcp = dict()

    # Load gpcp data to dictionary in personal netcdf variable format
    gpcp[gpcpVar] = likenclatlonvar(gpcpData,
                                    name='PRECT',
                                    units='mm/d',
                                    src='GPCP 2DD',
                                    srcid='GPCP')
    gpcp[gpcpVar].lat = lat
    gpcp[gpcpVar].lon = lon
    gpcp[gpcpVar].missing_value = -99999
    gpcp[gpcpVar].method = technique
    gpcp[gpcpVar].time = np.arange(0, gpcp[gpcpVar].data.shape[0])

    # REGRID TO CESM GRID
    if regrid_flag:
        # Assume regridding to CESM f19 grid if no grid given
        if newlat is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlat = ncDataset.variables['lat'][:]
        if newlon is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlon = ncDataset.variables['lon'][:]

        gpcpRegridVars = ['PRECT']
        for regridVar in gpcp.keys():
            if regridVar in gpcpRegridVars:

                # Regrid variables to new grid using linear interpolation
                if gpcp[regridVar].data.ndim == 3:
                    gpcp[regridVar].data = \
                        np.array([interpolate.interp2d(
                            gpcp[regridVar].lon,
                            gpcp[regridVar].lat,
                            gpcp[regridVar].data[j, :, :],
                            kind='linear',
                            copy=True,
                            bounds_error=False,
                            fill_value=np.nan)(newlon, newlat)
                            for j in range(gpcp[regridVar].data.shape[0])])

                # Assign new dimensions
                setattr(gpcp[regridVar], 'lat', newlat)
                setattr(gpcp[regridVar], 'lon', newlon)
                setattr(gpcp[regridVar], 'latb', None)
                setattr(gpcp[regridVar], 'lonb', None)

            else:
                print('Can''t regrid ' + regridVar)

    return gpcp


def loadhadisst(daNewGrid=None,
                kind='linear',
                newGridFile=None,
                newGridName=None,
                newLat=None,
                newLon=None,
                qc_flag=False,
                regrid_flag=False,
                whichHad='pd_monclimo',
                ):
    """
    Load HadISST data from netCDF file

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-10-24

    Args:
        N/A

    Kwargs:
        daNewGrid - dataArray with new grid for regridding
        kind - method to use for regridding
        newGridFile - file with new grid
        newGridName - (str) name of new grid for regridding
            > available grid: '0.9x1.25'
        qc_flag - True to make plots for quality control of regridding
        regrid_flag - True to regrid to given new grid
        whichHad - shortform for which file to load
            'pd_monclimo' - present day monthly climatology
                climo/HadISST_197901-201012_monclimo.nc
            'all' - load all available timesteps
                HadISST_sst.nc

    Returns:
        hadIsstDs - xr dataset with sst from HadISST
          (or)
        hadIsstDsRG - xr dataset with regridded sst from HadISST

    Notes:
        Now works with xarray!

    """

    # Set directories for HadISST
    hadIsstDir = '/home/disk/eos9/woelfle/dataset/HADISST/'
    if whichHad == 'pd_monclimo':
        hadIsstFile = 'climo/HadISST_197901-201012_monclimo.nc'
    elif whichHad == 'all':
        hadIsstFile = 'HadISST_sst.nc'

    # Load HadISST monthly output file
    hadIsstDs = xr.open_dataset(hadIsstDir + hadIsstFile)

    # Rename dimensions to match CESM standard
    hadIsstDs.rename({'latitude': 'lat',
                      'longitude': 'lon'},
                     inplace=True
                     )

    # Add id to dataset
    hadIsstDs.attrs['id'] = 'HadISST'

    # Roll data from 180W-180E to 0-360E
    if hadIsstDs.lon.values.min() < 0:
        # Determine lenght of roll required
        rollDist = np.sum(hadIsstDs.lon.values < 0)

        # Roll entire dataset
        hadIsstDs = hadIsstDs.roll(lon=rollDist)

        # Update longitudes to be positive definite
        hadIsstDs['lon'].values = np.mod(hadIsstDs['lon'] + 360, 360)

    if regrid_flag:
        hadIsstDsRG = xr.Dataset({
            'sst': roughregrid(
            hadIsstDs['sst'],
            daNewGrid=daNewGrid,
            kind=kind,
            newGridFile=newGridFile,
            newGridName=newGridName,
            newLat=newLat,
            newLon=newLon)
            })
        hadIsstDsRG.attrs['id'] = 'HadISST_rg'

        # Make plots to quickly check regridding of data
        if qc_flag:

            # Create figure for plotting
            plt.figure()

            # Make loop to shorten code for plotting
            for jDs, hadDs in enumerate([hadIsstDs, hadIsstDsRG]):
                plt.subplot(2, 1, jDs+1)

                # Plot some fields for comparison
                plt.imshow(hadDs['sst'].values[0, :, :],
                           cmap='RdBu_r',
                           vmin=290,
                           vmax=305)

        return hadIsstDsRG
    else:
        return hadIsstDs


def loadhybtop(histDir, caseName,
               loadVars=None,
               modelid='cesm',
               ncSubDir='',
               plevs=None,
               srcid='modelrun'):
    """
    Load 3d (+ time) variables to standard pressure levels
    2015-08-27
    """
    if loadVars is None:
        loadVars = ['T', 'V', 'Q']

    if 'PS' not in loadVars:
        loadVars = loadVars + ['PS']

    if plevs is None:
        plevs = np.array([1000, 925, 850, 700, 600, 500, 400, 300,
                          250, 200, 150, 100, 70, 50, 30, 20, 10])

    # Construct full history file path and load
    ncDict = dict()
    for loadVar in loadVars:
        # Set file names for loading
        ncFile = (caseName + '.cam.h0.' + loadVar + '.nc')

        # Set full file paths for loading
        ncPath = (histDir + caseName + os.sep + ncSubDir + ncFile)

        # Load data from netcdf file
        ncDict[loadVar], ncDim = loadnetcdf2class(ncPath, loadVar, srcid=srcid)

    # Update ncDim to include new pressure levels as a dimension
    ncDim['P'] = likenclatlonvar(plevs,
                                 name='P',
                                 src='User defined',
                                 srcid=srcid,
                                 units='hPa')

    # Convert to pressure levels from hybrid sigma (4d vars only)
    for var in loadVars[:-1]:
        if np.ndim(ncDict[var].data) == (np.ndim(ncDict['PS'].data) + 1):
            print('Converting ' + var)
            ncDict[var].datap = convertsigmatopres(ncDict[var].data,
                                                   ncDict['PS'].data,
                                                   plevs,
                                                   modelid)

    # Return interpolated variables as dictionary
    return (ncDict, ncDim)


def loadsoda(loadVars,
             yr1,
             cesmDim_flag=False,
             newlat=None,
             newlon=None,
             regrid_flag=False,
             tMax=None,
             version='2'):
    """
    Load SODA data from file

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-07-10

    Args:
        loadVars - variables to be loaded (as CESM style)
        yr1 - first year to be loaded (1981, 1986, or 1991) as string
        cesmDim_flag - True to make dimensions conform to CESM conventions
        newlat - latitude for new grid
        newlon - longitude for new grid
        regrid_flag - true to regrid to CESM grid
        tMax - maximum number of timesteps to load (to save on RAM)
        version - version of SODA dataset to load
            '2' for version 2.2.4
            '3' for verison 3.3.1

    Returns:
        soda - soda data
        sodaDim - associated dimensions

    Notes:
        Time referenced to start of run, not generally meaningful

    """
    # Set long version name based on short version
    longVersion = {'2': 'SODA_2.2.4',
                   '3': 'soda3.3.1',
                   }[version]

    # Dictionary to translate requested CESM variables to SODA variables
    if version == '2':
        cesm2sodaDict = {'SST': 'temp',
                         'TAUX': 'taux',
                         'TAUY': 'tauy',
                         'TEMP': 'temp',
                         'TS': 'temp',
                         'UVEL': 'u',
                         'VVEL': 'v',
                         'WVEL': 'w'
                         }
    elif version == '3':
        cesm2sodaDict = {'SST': 'temp',
                         'TAUX': 'taux',
                         'TAUY': 'tauy',
                         'TEMP': 'temp',
                         'TS': 'temp',
                         'UVEL': 'u',
                         'VVEL': 'v',
                         'WVEL': 'wt'
                         }

    # Initialize dictionaries to hold SODA variables
    soda = dict()
    sodaDir = {'2': '/home/disk/eos9/woelfle/dataset/SODA/rawdata/',
               '3': '/home/disk/eos9/woelfle/dataset/SODA3/data/singlevar/',
               }[version]
    sodaVars = [cesm2sodaDict[j] for j in loadVars]

    # Construct full history file paths and load output from netcdfs
    for (jVar, loadVar) in enumerate(sodaVars):
        print('Loading ' + loadVar + '...')
        ncFileCase = (longVersion + '.' + yr1 +
                      '.' + loadVar + '.nc')
        sodaPath = (sodaDir + ncFileCase)
        soda[loadVars[jVar]], sodaDim = (
            loadnetcdf2class(sodaPath, loadVar,
                             srcid='SODA' + version,
                             tMax=tMax))

        # Only retain topmost layer for SST
        if loadVars[jVar] in ['SST', 'TS']:
            soda[loadVars[jVar]].data = \
                soda[loadVars[jVar]].data[:, 0, :, :]

        # Convert 'latitude' and 'longitude' to 'lat' and 'lon' if needed
        if 'latitude' in sodaDim.keys():
            sodaDim['lat'] = sodaDim['latitude'].copy()
            del sodaDim['latitude']

        if 'longitude' in sodaDim.keys():
            sodaDim['lon'] = sodaDim['longitude'].copy()
            del sodaDim['longitude']

    # Update dimension dictionaries to include latitude and longitude edge
    #   info
    dLatb = sodaDim['lat'][1] - sodaDim['lat'][0]
    latbMin = sodaDim['lat'][0] - dLatb/2.
    latbMax = sodaDim['lat'][-1] + dLatb/2.
    sodaDim['latb'] = np.arange(latbMin, latbMax+dLatb/2., dLatb)

    dLonb = sodaDim['lon'][1] - sodaDim['lon'][0]
    lonbMin = sodaDim['lon'][0] - dLonb/2.
    lonbMax = sodaDim['lon'][-1] + dLonb/2.
    sodaDim['lonb'] = np.arange(lonbMin, lonbMax+dLonb/2., dLonb)

    # Add dimension info to each variable
    for loadVar in soda.keys():
        for dimName in sodaDim.keys():
            setattr(soda[loadVar], dimName, sodaDim[dimName])

        # Add dimensions to be consistent with CESM output
        if cesmDim_flag:
            if (np.ndim(soda[loadVar].data) == 4):
                # Include conversion from m to cm to match CESM
                setattr(soda[loadVar], 'z_t',
                        sodaDim['depth']*100.)
                setattr(soda[loadVar], 'dz',
                        midpttothickness(sodaDim['depth'])*100.)

    # REGRID TO CESM GRID
    if regrid_flag:
        # Assume regridding to CESM f19 grid if no grid given
        if newlat is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlat = ncDataset.variables['lat'][:]
                newlatb = ncDataset.variables['slat'][:]
        if newlon is None:
            f19path = ('/home/disk/p/woelfle/cesm/nobackup/hist/' +
                       'cesm.f19.dimensions.nc')
            with nc4.Dataset(f19path, 'r') as ncDataset:
                newlon = ncDataset.variables['lon'][:]
                newlonb = ncDataset.variables['slon'][:]

        sodaRegridVars = ['SST', 'TAUX', 'TAUY', 'TS']
        for regridVar in soda.keys():
            if regridVar in sodaRegridVars:

                # Regrid variables to new grid using linear interpolation
                if soda[regridVar].data.ndim == 3:
                    soda[regridVar].data = \
                        np.array([interpolate.interp2d(
                            sodaDim['lon'],
                            sodaDim['lat'],
                            soda[regridVar].data[j, :, :],
                            kind='linear',
                            copy=True,
                            bounds_error=False,
                            fill_value=np.nan)(newlon, newlat)
                            for j in range(soda[regridVar].data.shape[0])])

                # Assign new dimensions
                setattr(soda[regridVar], 'lat', newlat)
                setattr(soda[regridVar], 'lon', newlon)
                try:
                    setattr(soda[regridVar], 'latb', newlatb)
                except NameError:
                    setattr(soda[regridVar], 'latb', None)
                try:
                    setattr(soda[regridVar], 'lonb', newlonb)
                except NameError:
                    setattr(soda[regridVar], 'lonb', None)

            else:
                print('Can''t regrid ' + regridVar)

    return (soda, sodaDim)


def midpttothickness(x,
                     x0=0):
    """
    Compute thickness of a cells denoted by midpoints at x

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-04-29

    Args:
        x - 1d array of midpts.
        x0 - location of first edge, assumed to be at x = 0

    Returns:
        dx - thickness of each cell/layer/region

    Notes:
        N/A

    """
    dx = np.zeros_like(x)
    dx[0] = (x[0] - x0) + (x[1] - x[0])/2.
    dx[1:-1] = (x[2:] - x[:-2])/2.
    dx[-1] = dx[-2]

    return dx


def printdoneblock():
    """
    Print block of text to stdout denoting a thing has finished

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-08-11

    Args:
        None

    Returns:
        None

    """

    print('\n' + '#'*50 + '\n' +
          '#'*10 + ' '*30 + '#'*10 + '\n' +
          '#'*10 + ' '*13 + 'DONE' + ' '*13 + '#'*10 + '\n' +
          '#'*10 + ' '*30 + '#'*10 + '\n' +
          '#'*50)


def pullnumsfromstr(inString):
    """
    Pull and return (as int) numerals from within a string

    Example:
        inString = 'abcd123' returns outNum = 123
    """

    outString = int(''.join([inString[j]
                             for j in range(len(inString))
                             if inString[j].isdigit()]))

    return outString


def regridcesm2d(inData, inGridName, outGridName,
                 regridFile=None,
                 cs_flag=True,
                 loop_flag=False,
                 missingValOut=np.nan):
    """
    Regrids output from one cesm grid to another using area conservative method

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2016-04-06

    Args:
        inData - data to be converted to new grid on old grid as 2D array with
            dimensions of [nlon, nlat]
        inGridName - string indicating grid of inData
        outGridName - string indicating grid to which data is to be converted
        regridFile - full path of regrid file to be used
        cs_flag - True to use cumulative sum method for regridding; setting to
            false forces use of loop method (which is much slower)
        loop_flag - True to force use of loop in place of cum. sum method;
            **overrides cs_flag
        missingValOut - value to use for missing values in outData

    Returns:
        outData - data after being converted to new grid; returned as double

    Included grids [shorthand] (can add more):
        fv0.9x1.25 [f09]
        fv1.9x2.5  [f19]
        gx1v6      [g16]

    Raises:
        none

    Notes:
        - Script inherited from M. Lague via A. Ordonez via CC Bitz
        - Script greatly changed to correct computations (MDW 2015-06-05)
            - All prior versions are incorrect and should not be used.
        - Script verified versus NCL conversion and works well. Difference of
            O(10^-5) for a field of O(10^2) from the NCL file. The cumulative
            sum method has a maximum absolute difference of O(10^-10) from the
            loop method.
        - Script updated to use cumulative summing for speed (MDW 2015-06-11)
            if possible
        * Always check output between loop and cumulative sum method the first
            time you try to use a new dataset to ensure errors from summing
            tons of points remains small, though it should now that the inData
            is converted to a double prior to summing.
        * Finding the regrid file is the bottleneck for the cumulative sum
            method. Thus the function can be sped up by 0.2-0.3 s if the
            'regridFile' optional input is used.
        - May work for bilinear intep, but would need to update regridMethod
            and check against NCL to make sure before running too far with it.
        - Now works with masked and unmasked arrays (2016-04-06)
    """
    # Set method for regridding. Need to update regridDir if going to bilin
    regridMethod = 'aave'

    # Set defaults for options
    #  parfor_flag = false;  % True to use parfor if looping

    # Create dictionary of acceptable grids
    gridLongName = {'g16':        'gx1v6',
                    'gx1v6':      'gx1v6',
                    'f09':        'fv0.9x1.25',
                    'fv0.9x1.25': 'fv0.9x1.25',
                    'f19':        'fv1.9x2.5',
                    'fv1.9x2.5':  'fv1.9x2.5'}

    # Parse input grid name to full name
    inGridName = gridLongName[inGridName]

    # Parse output grid name to full name and get shape of output grid
    outGridName = gridLongName[outGridName]
    outGridShape = getgridsize(outGridName)

    # Find weight file for regridding if regridFile not provided as part of
    #   optional inputs
    #   **Only works if running on yellowstone/geyser/etc. or if run by woelfle
    #       on UW server.
    #   If don't want to mess with this, always prescribe the restart file when
    #       calling the regrid function.
    ncarMachs = ['yslogi', 'geyser']
    uwMachs = ['challenger', 'p', 'stable', 'fog']
    if regridFile is None:
        # Set directory to search for weighting file
        # Get home directory from environment variables
        # (only tested on UNIX, maybe works for mac?)
        #    homeDir = getenv('HOME');
        if gethostname()[0:5] in ncarMachs:
            regridDir = '/glade/p/cesmdata/cseg/inputdata/cpl/cpl6/'
        elif gethostname() in uwMachs:
            # Change following line to be wherever you store your regrid weight
            #   files
            regridDir = '/home/disk/p/woelfle/cesm/regrid/'

        # Try to find regrid weighting file
        regridFile = glob.glob(regridDir + 'map_' + inGridName + '*' +
                               outGridName + '*' + regridMethod + '*')

        if len(regridFile) == 0:
            print('Cannot find weighting file for ' + inGridName + ' to ' +
                  outGridName + ' with grid method ' + regridMethod + '. ' +
                  'Check that weight file exists.')
            return
        elif len(regridFile) > 1:
            print('Found too many weighting files for ' + inGridName +
                  ' to ' + outGridName + ' with grid method ' + regridMethod +
                  '. Check to ensure only 1 version available.')
            return
        else:
            regridFile = regridFile[0]

    # Regrid data from inGrid to outGrid

    # Best analogy I've got is that it uses the weights as kind of a map for
    #   fracturing the old (input) grid and reassembling the pieces into a
    #   new (output) grid

    # Set value to be used internally for missing values. These points will be
    #   replaced by missingValOut before function returns output
    missingVal = -9e9

    with nc4.Dataset(regridFile, 'r') as weightFile:
        # Get index into weighting array for inData
        #   (- 1 to match python indexing)
        inGridIndex2Wts = weightFile.variables['col'][:] - 1

        # Get index into weighting array for outData
        #   (- 1 to match python indexing)
        outGridIndex2Wts = weightFile.variables['row'][:] - 1

        # Get vector containing all possible overlapping regions between the
        #   grids and how much each box fromt the input grid contributes to
        #   that portion of the output grid
        mapWt = weightFile.variables['S'][:]

    # Convert input data into a column vector and a double to reduce
    #   computational errors which may arise if inDataVec is a single.
    if isinstance(inData, np.ma.core.MaskedArray):
        inDataVec = np.float64(inData.data)

        # Replace masked data points with missingVal if needed
        inDataVec[inData.mask] = missingVal
    else:
        inDataVec = np.float64(inData)
    inDataVec = np.hstack(inDataVec)

#    inDataVec = np.float64(inData.data)
#
#    # Replace masked data points with missingVal if needed
#    # try:
#    inDataVec[inData.mask] = missingVal
#    # except AttributeError:
#    #     print('do be do')
#
#    inDataVec = np.hstack(inDataVec)

    # Compute weighted contribution of the data on the input grid to the output
    #   grid. Essentially force the input data through a screen lined by the
    #   output grid and get the many resulting "fractured" pieces of the input
    #   data
    wtDataIn = mapWt*inDataVec[inGridIndex2Wts]

    # Sum pieces of "fractured" input data to construct output boxes
    #
    #  % if outGridIndex2Wts is non-decreasing, use cumulative summing method
    if cs_flag and all(np.diff(outGridIndex2Wts, n=1, axis=0) >= 0):

        # FOLLOWING COMMENTARY WAS WRITTEN FOR MATLAB. COPIED HERE,
        #   **NOT TRANSLATED TO PYTHON**
        #   Example of how the indexing in this section works as it is somewhat
        #     novel:
        #
        #     Suppose we have weighted input data vector a which is 7x1, and we
        #       want to add the values in a in a given way to make a new vector
        #       b which is 3x1. We want b(1) to be sum(a(1:3)),
        #       b(2) to be sum(a(4:5)) and b(3) to be sum(a(6:7)). We could do
        #       this by looping through each element of b and pulling the
        #       corresponding elements of a to sum, or we can do the following
        #       which is much faster particularly when b is large. In this
        #       analogy, a = wtDataIn, and b = outDataCS.
        #
        #     vv CODE ANALOG vv          % vv DESCRIPTION [related_var]      vv
        #
        #     a = 1:7;                   % - Weighted and "fractured" input
        #                                %   data [wtDataIn]
        #                                %
        #     ind = [1 1 1 2 2 3 3];     % - Indices indicating the element of
        #                                %   b to which each element of a
        #                                %   contributes its weight. ind must
        #                                %   be non-decreasing for this method
        #                                %   to work. [outGridIndex2Wts]
        #                                %
        #     sumA = cumsum(a);          % 1 Cumulatively sum weighted input
        #                                %   data [csWtDataIn]
        #                                %
        #     bcs = ones(3,1)*missingVal;% 2 Initialize output array to
        #                                %   missingVal [outDataCS]
        #                                %
        #     bcs(ind) = sumA;           % 3 Assign cumulative sums to output
        #                                %   array. This type of indexing
        #                                %   operates similar to the
        #                                %   following operation:
        #                                %     bTemp = bcs;
        #                                %     for jInd = 1:numel(ind)
        #                                %       bTemp(ind(jInd)) = sumA(jInd);
        #                                %     end
        #                                %     bcs = bTemp;
        #                                %   where bTemp is not accessible to
        #                                %   the user,
        #                                %   i.e.   b(ind) = b(ind) + a;
        #                                %   yields b = [ 3 5  7]
        #                                %   not    b = [ 6 9 13]
        #                                %   (presuming missingVal = 0)
        #                                %   [outDataCS]
        #                                %
        #      b = [bcs(1);diff(bcs(:))]; % 4 Difference cumulative sums to
        #                                %   recover discrete weights [outData]
        #                                %   ** code is a bit more detailed
        #                                %   here to account for locations in b
        #                                %   which are not assinged a value in
        #                                %   step 3 but the general principle
        #                                %   is the same
        # Step 1: Cumulatively sum weighted data to prep for computing new grid
        #   values
        csWtDataIn = wtDataIn.cumsum()

        # Step 2: Create vector for holding cumulativley summed output data.
        #   Non-filled cells will equal missingVal
        outDataCS = np.ones(outGridShape[0]*outGridShape[1])*missingVal

        # Step 3: Pull values from cumulative sum which corresponds to last
        #   requested position from monotonically increasing outGridIndex2Wts
        outDataCS[outGridIndex2Wts] = csWtDataIn

        # Step 4: Difference cumulative sums to recover actual weighted data
        #   for each index
        #   **care must be taken when differencing to account for values not
        #   set in step 3, e.g. land pts. when going to an ocean grid

        # Step 4a: Pull only points whose value was assigned in Step 3
        outDataTemp = outDataCS[outDataCS != missingVal]

        # Step 4b: Create array to hold final output data
        outDataVec = np.ones(outGridShape[0]*outGridShape[1])*missingValOut

        # Step 4c: Un-cumulatively sum weights to new grid and assign to
        #   appropriate location in the outData vector
        outDataVec[outDataCS != missingVal] = np.insert(np.diff(outDataTemp),
                                                        0, outDataTemp[0])

    else:
        # If index is not non-decreasing, must use loop to sum pieces for
        #   outData. Will also use this section if cs_flag == false.

        # Preallocate space for output data vector
        outData = np.ones(outGridShape[0]*outGridShape[1])*missingValOut

        # Loop through indices, collect weighted data to be summer for that
        #   index, sum it, and add it to the output data vector. May be very
        #   slow for output grids with large numbers of elements.
        for jInd in range(outGridIndex2Wts.size):
            outData[jInd] = np.sum(wtDataIn[outGridIndex2Wts == jInd])

    # Convert output data vector to 2d array [nlon x nlat]
    outData = outDataVec.reshape(outGridShape)

    # Retrun regridded output array
    return outData


def regridcesmnd(inData, inGridName, outGridName, **kwargs):
    """
    Run regridcesm2d function over all other dimesions of input data

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-05

    Args:
        inData - data field to be converted to new grid
        inGridName - string indicating grid of inData
        outGridName - string indicating grid to which data is to be converted
        **kwargs - see regridcesm2d for details

    Returns:
        outData - data after being converted to new grid

    """

    # For lat x lon
    if np.ndim(inData) == 2:
        # Regrid data
        outData = regridcesm2d(inData, inGridName, outGridName, **kwargs)

    # For time x lat x lon
    elif np.ndim(inData) == 3:
        # Add location to store regridded data
        outData = np.empty([inData.shape[0],
                            getgridsize(outGridName)[0],
                            getgridsize(outGridName)[1]])

        # Regrid data
        for jt in range(inData.shape[0]):
            outData[jt, :, :] = regridcesm2d(inData[jt, :, :],
                                             inGridName, outGridName,
                                             **kwargs)

    # For time x depth x lat x lon
    elif np.ndim(inData) == 4:
        # Add location to store regridded data
        outData = np.empty([inData.shape[0], inData.shape[1],
                            getgridsize(outGridName)[0],
                            getgridsize(outGridName)[1]])

        # Regrid data
        for jt in range(inData.shape[0]):
            for jz in range(inData.shape[1]):
                outData[jt, jz, :, :] = \
                    regridcesm2d(inData[jt, jz, :, :],
                                 inGridName, outGridName,
                                 **kwargs)

    return outData


def rmallsubstring(fullString,
                   subString):
    """
    Removes all instances of subString from fullString
    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)
    Version Date:
        2017-03-21
    Args:
        fullString- (str) string from which substrings are to be removed
        subString - (str) string to be removed from fullString
    Returns:
        fullString - (str) fullString with all instances of subString removed
    """
    # Get length of subString
    lenSubString = len(subString)

    # Remove all instances of subString from fullString
    while subString in fullString:
        fullString = (fullString[0:fullString.index(subString)] +
                      fullString[fullString.index(subString)+lenSubString:])

    # Return new string
    return fullString


def rmse(data1, data2):
    """
    Compute root mean squared error

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-11
    """
    # Compute number of non-NaN entries
    nData = data1.size - np.sum(np.isnan(data1+data2))

    # Set value to 0 if NaN in either array (data1 or data2)
    nanFilt1 = np.isnan(data1)
    nanFilt2 = np.isnan(data2)
    data1[nanFilt1 | nanFilt2] = 0
    data2[nanFilt1 | nanFilt2] = 0

    return np.sqrt(1./nData*np.sum((data1-data2)**2))


def roughregrid(daIn,
                daNewGrid=None,
                kind='linear',
                newGridFile=None,
                newGridName=None,
                newLat=None,
                newLon=None,
                ):
    """
    Regrid dataArray, daIn, to new grid given by daNewGrid

    Author:
        Matthew Woelfle

    Version Date:
        2017-10-24

    Notes:
        Very rough right now. Need to make better.
    """

    valuesIn = daIn.values

    # Retrieve new grid from file if not provided
    if any([newLat is None, newLon is None]):
        # If new grid is not provided
        if daNewGrid is None:
            # If file for loading new grid is not provided
            if newGridFile is None:
                # Get name of file for loading standard grids
                newGridFile = {
                    '0.9x1.25': ('/home/disk/eos9/woelfle/cesm/' +
                                 'nobackup/cesm1to2/' +
                                 'b.e15.B1850G.f09_g16.pi_control.01/' +
                                 '0.9x1.25/b.e15.B1850G.f09_g16.pi_control.' +
                                 '01_01_climo.nc'
                                 ),
                    '1.9x2.5': ('coming soon'),
                    }[newGridName]

            # Load new grid from file
            daNewGrid = xr.open_dataset(newGridFile)

        # Get new grid lats and lons
        newLat = daNewGrid.lat
        newLon = daNewGrid.lon

    # Replace nans with extreme missingVal if needed to avoid all nans
    valuesIn[np.isnan(valuesIn)] = -9e9

    # Regrid variables to new grid using linear interpolation (poor, I know)
    if valuesIn.ndim == 3:
        valuesOut = \
            np.array([interpolate.interp2d(
                daIn.lon,
                daIn.lat,
                valuesIn[j, :, :],
                kind=kind,
                copy=True,
                bounds_error=False,
                fill_value=-9e9)(newLon, newLat)
                for j in range(valuesIn.shape[0])
                ])

    # Set super low values back to nan
    valuesOut[valuesOut < -1e3] = np.nan

    # Reset nans in valuesIn
    valuesIn[valuesIn == -9e9] = np.nan

    # Create dataArray with newly regridded values
    if valuesIn.ndim == 3:
        daOut = xr.DataArray(valuesOut,
                             coords={'time': daIn.time,
                                     'lat': newLat,
                                     'lon': newLon},
                             dims=('time', 'lat', 'lon'),
                             attrs=daIn.attrs)

    # Return regridded dataArray
    return daOut


def shiftlons(data,
              lon,
              lonAxis=None,
              newLonLims=np.array([0, 360])):
    """
    Shift (roll) lat-lon data from -180,180 longitude to 0,360 longitude
        or viceversa

    Version Date:
        2017-05-31

     Args:
         data - data array to be shifted; can be multiple dimensions beyond
             pure latitude longitude, e.g. [time, lat, lon]
         lon - longitude array; assumed to be 1d for now
         lonAxis - axis of data which corresponds to longitude;
             will atttempt to deduce using axis lengths if not provided
         newLonLims - 1x2 np.arrray of new longitude limits desired

    Returns:
        data - shifted data array
        lon - shifted longitude array
    """

    # Define axis along which to roll data
    #   (implicitly assumes time is the first dimension)
    if lonAxis is None:
        lonAxis = [j
                   for j in range(np.ndim(data))
                   if (np.shape(data)[j] == np.size(lon))
                   ][-1]

    # Determine numper of points outside of new western boundary
    excessPtsWest = np.sum(lon < newLonLims.min())
    excessPtsEast = np.sum(lon > newLonLims.max())

    # Roll data nd longitudes to take data from west and add to east end
    if (excessPtsWest > 0) and (excessPtsEast == 0):
        # Roll data field
        data = np.roll(data, -excessPtsWest, axis=lonAxis)

        # Roll longitudes
        lon = np.roll(lon, -excessPtsWest)
        lon[lon < newLonLims.min()] = \
            lon[lon < newLonLims.min()] + 360

        # Return reordered data and longtidues
        return data, lon

    elif (excessPtsWest == 0) and (excessPtsEast > 0):
        # Roll data field
        data = np.roll(data, excessPtsEast, axis=lonAxis)

        # Roll longitudes
        lon = np.roll(lon, excessPtsEast)
        lon[lon > newLonLims.max()] = \
            lon[lon > newLonLims.max()] - 360

        # Return reordered data and longitudes
        return data, lon


def srcidfromcasenames(caseNames,
                       extraIdString):
    """
    Create srcid strings from casename strings. Appends extraIdString and moves
        ensemble number to end.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-05-04

    Inputs:
        caseNames - dictionary with short case identifier as key and longform
            identifier (such as used for file name) as value
        extraIdString - string of extra identifier to be added to shortform id
            from caseNames keys

    Returns:
        srcIds - dictionary with same keys as caseNames. Values are now more
            descriptive shortform identifiers

    Notes:
        Typically used to add year to output (i.e. CTRL0 -> CTRL1991_0)
    """

    srcIds = {caseNames.keys()[j]:
              ((caseNames.keys()[j][:-1] + extraIdString +
                '_' + caseNames.keys()[j][-1])
               if caseNames.keys()[j][-1].isdigit()
               else caseNames.keys()[j] + extraIdString)
              for j in range(len(caseNames))}

    return srcIds


def strisint(inString):
    """
    Check if given string contains an integer

    Args:
        inString - string to be checked

    Returns:
        True if inString contains only an integer else False

    Notes:
        Based on stack exchange code (https://stackoverflow.com/a/1267145)
    """
    try:
        int(inString)
        return True
    except ValueError:
        return False
