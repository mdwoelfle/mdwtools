# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:43:25 2015

Last Updated: 2017-05-16

NOTE: SLOWLY CONVERTING THIS SCRIPT TO WORK WITH XARRAY RATHER THAN
    NCLATLONVAR

@author: woelfle
"""
import numpy as np               # import numerical python, i.e. MATLAB-like
import matplotlib.pyplot as plt  # import plotting functionality
import matplotlib.gridspec as gridspec  # for pretty subplots
from math import pi, log10       # import constants
import mdwtools.mdwfunctions as mwfn      # import my functions
import os                        # import operating system commands
from mpl_toolkits.basemap import Basemap  # import tool for lat/lon plotting
from datetime import datetime    # for working with dates and stuff
import sys
from scipy import interpolate    # interpolation functions

# from mpl_toolkits import basemap as bmp

"""
List of Functions:
    find_nearest - find nearest value in numpy array
    regularconts - define regular contours for plotting
    subsetregion - placeholder for something...
    plotglobalmap - plot lon x lat map for all lons and lats
    plotzonalmeanline - plot line of zonal mean over a given set of time steps
    plotmeridmean - plot Hovmoller of meridional mean
    plotmeridslice - plot lat x depth slice across given lons and times
    compmeridslice - subplot plotmeridslice multiple times
    plotzonmean - plot Hovmoller of zonal mean
    plotzonmeanannual - plot zonal mean line averaged over each year
    getmaxxlim - find maximum x limits over all subplots
    getmaxylim - find maximum y limits over all subplots
    savefig - save figure using user defined settings
"""


def compmeridslice(dataIn, plot_t,
                   dataV=None,
                   dataW=None,
                   compcont=None,
                   conts=None,
                   depthLine_flag=False,
                   depthLineColor='w',
                   depthLineDepth=100.,
                   depthLineStyle='--',
                   depthLineWidth=1,
                   layout='onecol',
                   latLim=np.array([-90., 90.]),
                   lonLim=np.array([0, 360]),
                   plotRefCase_flag=False,
                   refCont_flag=False,
                   refContColor='w',
                   refData=None,
                   sinLat_flag=True,
                   vecRefUnits='[m/s, mm/s]',
                   vecRefLen=0.01,
                   wScale=1000,
                   **kwargs
                   ):
    """
    Plot multiple instances of plotmeridslice on one figure.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-05-16
    Args:
        dataIn - list containing latlonvar instances for plotting
        plot_t - times to average over for plotting
        compcont - contours to plot in bold for emphasis/reference
        conts - contour values to use when plotting
        depthLine_flag - true to plot reference line at constant depth
        depthLineColor - color for constant depth line
        depthLineDepth - [m] depth at which to plot line of constant depth
        deptLineStyle - line style for line of constant depth
        depthLineWidth - line width for line of constant depth
        latLim - latitude limits for plot area and zonally averaging
        lonLim - longitude limits for zonally averaging
        refCase_flag - True to plot bold contour from refData on all plots
        refData - case on different grid from dataIn to plot for reference
        sinLat_flag - true to plot versus sin(latitude) instead of versus
            straight latitude (little difference in tropics)
        **kwargs - options passed directly to plotmeridslice():
            > cMap, grid_flag, maxConts, sym_flag, tickDir, zLim

    Returns:
        n/a

    Raises:
        none

    Notes:
        n/a
    """

    # Set font size
    plt.rc('font', size=18)

    # Pull vertical coordinate
    if 'z_t' in dataIn[0].__dict__.keys():
        z = dataIn[0].z_t/100
    elif 'z_w_top' in dataIn[0].__dict__.keys():
        z = dataIn[0].z_w_top/100

    if refData is not None:
        if 'depth' in refData.__dict__.keys():
            zRef = refData.depth

    nSubs = len(dataIn) + 1 - (refData is None)

    # Create subplot grid
    if layout.lower() in ['twocols', 'twocol']:
        nRows = (int(plotRefCase_flag) +
                 int(np.ceil(len(dataIn)/2.)) +
                 1)
        gs = gridspec.GridSpec(nRows,
                               4,
                               left=0.07,
                               right=0.99,
                               bottom=0.05,
                               top=0.96,
                               hspace=0.4,
                               height_ratios=[5]*(int(nRows) - 1) + [1],
                               wspace=0.3,
                               )

        # Set indices for gridSpec
        refColInds = [1, 3]
        refRowInd = 0
        plotColInds = [[0, 2], [2, 4]]*int(np.ceil(len(dataIn)/2.))
        # plotRowInds = [j, j]*int(np.ceil(len(dataIn)/2.))
        plotRowInds = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        cbColInd = 0
        cbRowInd = nRows - 1

    # Initialize subplot counter
    jSub = 0

    # Plot reference case if provided
    if plotRefCase_flag:
        # Compute slices to be plotted
        refSlice = refData.calcregzonmean(latLim=latLim,
                                          lonLim=lonLim)

        # Plot ocean only output
        if layout.lower() in ['onecol']:
            ha = plt.subplot2grid((nSubs, 5), (jSub, 0), colspan=4)
        else:
            ha = plt.subplot(gs[refRowInd, refColInds[0]:refColInds[1]])
        cset = plotmeridslice(refSlice, refData.lat, zRef,
                              plot_t,
                              annotate_flag=False,
                              compcont=compcont,
                              cbar_flag=False,
                              conts=conts,
                              dataId=refData.srcid,
                              latLim=latLim,
                              sinLat_flag=sinLat_flag,
                              varName=refData.name,
                              varUnits=refData.units,
                              **kwargs
                              )
        if refCont_flag:
            # Compute lats for reference line
            refXVec = (np.sin(refData.lat[:]*pi/180.)
                       if sinLat_flag
                       else refData.lat)

            # Print common reference line
            plt.contour(refXVec, zRef, refSlice[plot_t, :, :].mean(axis=0),
                        np.array([compcont]),
                        colors=refContColor,
                        linewidths=2,
                        hold='on'
                        )

        if depthLine_flag:
            # Print line of constant depth for reference
            plt.plot([-100, 100],
                     [depthLineDepth, depthLineDepth],
                     color=depthLineColor,
                     linestyle=depthLineStyle,
                     linewidth=depthLineWidth,
                     )

        # Add subplot number
        ha.annotate('(' + chr(ord('a')) + ')',
                    # xy=(-0.12, 1.09),
                    xy=(-0.11, 1.01),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontweight='bold',
                    )

        # Add source name to subplot
        if refData.srcid[-5:] == '_mean':
            caseString = refData.srcid[:-5]
        else:
            caseString = refData.srcid
        plt.annotate(caseString,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        if jSub == 0:
            # Add lat/lon limits to first subplot
            plt.annotate(r'$\theta$=[{:0.0f}'.format(lonLim[0]) +
                         ', {:0.0f}]'.format(lonLim[1]) +
                         r' $\phi$=[{:0.0f}'.format(latLim[0]) +
                         ', {:0.0f}]'.format(latLim[1]),
                         xy=(0.5, 1),
                         xycoords='axes fraction',
                         horizontalalignment='center',
                         verticalalignment='bottom'
                         )

            # Add timesteps used to first subplot
            plt.annotate('t = [{t1:0.0f}, {t2:0.0f}]'.format(t1=plot_t[0],
                                                             t2=plot_t[-1]),
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )

            # Remove x axis label
            plt.xlabel('')

        # Increment subplot counter
        jSub = jSub + 1

    # Loop through latlonvars provided in dataIn
    for jData in range(len(dataIn)):

        # Compute slices to be plotted
        dataSlice = dataIn[jData].calcregzonmean(latLim=latLim,
                                                 lonLim=lonLim)

        # Compute zonal mean V over a given region returns [t, z, lat]
        if dataV is not None:
            dataVSlice = dataV[jData].calcregzonmean(latLim=latLim,
                                                     lonLim=lonLim)
        else:
            dataVSlice = None

        # Compute zonal mean W over a given region returns [t, z, lat]
        if dataW is not None:
            dataWSlice = dataW[jData].calcregzonmean(latLim=latLim,
                                                     lonLim=lonLim)*wScale
        else:
            dataWSlice = None

        # Plot ocean only output
        if layout.lower() in ['onecol']:
            ha = plt.subplot2grid((nSubs, 5), (jSub, 0), colspan=4)
        else:
            ha = plt.subplot(gs[plotRowInds[jData],
                                plotColInds[jData][0]:plotColInds[jData][1]])
        # ha = plt.subplot2grid((nSubs, 5), (jSub, 0), colspan=4)
        cset = plotmeridslice(dataSlice, dataIn[jData].lat, z,
                              plot_t,
                              annotate_flag=False,
                              compcont=compcont,
                              cbar_flag=False,
                              conts=conts,
                              dataId=dataIn[jData].srcid,
                              latLim=latLim,
                              sinLat_flag=sinLat_flag,
                              varName=dataIn[jData].name,
                              varUnits=dataIn[jData].units,
                              vVel=dataVSlice,
                              vecRefLen=vecRefLen,
                              vecRefName='Velocity',
                              vecRefUnits=vecRefUnits,
                              wVel=dataWSlice,
                              **kwargs
                              )

        if refCont_flag:
            # Compute slices to be plotted
            refSlice = refData.calcregzonmean(latLim=latLim,
                                              lonLim=lonLim)

            # Compute lats for reference line
            refXVec = (np.sin(refData.lat[:]*pi/180.)
                       if sinLat_flag
                       else refData.lat)

            # Print common reference line
            plt.contour(refXVec, zRef, refSlice[plot_t, :, :].mean(axis=0),
                        np.array([compcont]),
                        colors=refContColor,
                        linewidths=2,
                        hold='on'
                        )

        if depthLine_flag:
            # Print line at constant depth for reference
            plt.plot([-100, 100],
                     [depthLineDepth, depthLineDepth],
                     color=depthLineColor,
                     linestyle=depthLineStyle,
                     linewidth=depthLineWidth,
                     )

        ha.annotate('(' + chr(ord('a') + jSub) + ')',
                    # xy=(-0.12, 1.09),
                    xy=(-0.11, 1.01),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontweight='bold',
                    )


        # Remove x axis labels if not bottom plot
        if layout in ['onecol']:
            if jSub != (nSubs - 1):
                ha.set_xlabel('')
        else:
            if plotRowInds[jData] != (nRows - 2):
                ha.set_xlabel('')
            if plotColInds[jData][0] != 0:
                ha.set_ylabel('')

        # Set to match contours between plots
        conts = cset.levels

        # Add source name to subplot
        # Set caseString for plotting
        if dataIn[jData].srcid[-5:] == '_mean':
            caseString = dataIn[jData].srcid[:-5]
        elif dataIn[jData].srcid[-4:].isdigit():
            caseString = dataIn[jData].srcid[:-4]
        else:
            caseString = dataIn[jData].srcid
        plt.annotate(caseString,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        if jSub == 0:
            # Add lat/lon limits to first subplot
            plt.annotate(r'$\theta$=[{:0.0f}'.format(lonLim[0]) +
                         ', {:0.0f}]'.format(lonLim[1]) +
                         r' $\phi$=[{:0.0f}'.format(latLim[0]) +
                         ', {:0.0f}]'.format(latLim[1]),
                         xy=(0.5, 1),
                         xycoords='axes fraction',
                         horizontalalignment='center',
                         verticalalignment='bottom'
                         )

            # Add timesteps used to first subplot
            plt.annotate('t = [{t1:0.0f}, {t2:0.0f}]'.format(t1=plot_t[0],
                                                             t2=plot_t[-1]),
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )

        # Increment subplot counter
        jSub = jSub + 1

    # Clean up layout to fit all labels
    try:
        plt.tight_layout()
    except ValueError:
        pass

    # Add common colorbar
    if layout in ['onecol']:
        cbar_ax = plt.subplot2grid((1, 5), (0, 4))
    else:
        cbar_ax = plt.subplot(gs[nRows - 1, 0:])
    hcb = plt.colorbar(cset, cax=cbar_ax,
                       orientation=('vertical'
                                    if layout in ['onecol']
                                    else 'horizontal')
                       )
    pcb = cbar_ax.get_position()
    if layout in ['onecol']:
        cbar_ax.set_position([pcb.x0+0.02, pcb.y0 + pcb.height/4.,
                              0.02, pcb.height/2.])
        cbar_ax.set_ylabel(getplotvarstring(dataIn[0].name) + ' (' +
                           mwfn.getstandardunitstring(dataIn[0].units) + ')')
    else:
        cbar_ax.set_position([pcb.x0 + pcb.width/4., pcb.y0 + 0.015,
                              pcb.width/2., 0.02])
        cbar_ax.set_xlabel(getplotvarstring(dataIn[0].name) + ' (' +
                           mwfn.getstandardunitstring(dataIn[0].units) + ')' )
    # Add colorbar ticks and ensure compcont is labeled
    hcb.set_ticks(cset.levels[::2]
                  if np.array([compcont]) in cset.levels[::2]
                  else cset.levels[1::2])
    # print()

    # Plot bold contour on colorbar for reference
    if compcont is not None:
        boldLoc = (compcont - conts[0]) / (conts[-1] - conts[0])
        if layout in ['onecol']:
            cbar_ax.hlines(boldLoc, 0, 1, colors='k', linewidth=2)
        else:
            cbar_ax.vlines(boldLoc, 0, 1, colors='k', linewidth=2)

    plt.show()


def compzonslice(dataIn, plot_t,
                 compcont=None,
                 conts=None,
                 latLim=np.array([-90., 90.]),
                 lonLim=np.array([0, 360]),
                 **kwargs
                 ):
    """
    Plot multiple instances of plotzonslice on one figure.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-10-30

    Args:
        dataIn - list containing latlonvar instances for plotting
        plot_t - times to average over for plotting
        compcont - contour to plot in bold for reference
        conts - contour values to use when plotting
        latLim - latitude limits for plot area and zonally averaging
        lonLim - longitude limits for zonally averaging
        **kwargs - options passed directly to plotmeridslice():
            > cMap, grid_flag, maxConts, sym_flag, tickDir, zLim

    Returns:
        n/a

    Raises:
        none

    Notes:
        n/a
    """

    # Loop through latlonvars provided in dataIn
    for jData in range(len(dataIn)):

        # Pull vertical coordinate
        if 'z_t' in dataIn[0].__dict__.keys():
            z = dataIn[jData].z_t/100.
        elif 'z_w_top' in dataIn[0].__dict__.keys():
            z = dataIn[jData].z_w_top/100.

        # Compute slices to be plotted
        dataSlice = dataIn[jData].calcregmeridmean(latLim=latLim,
                                                   lonLim=lonLim)

        # Plot ocean only output
        ha = plt.subplot2grid((len(dataIn), 5), (jData, 0), colspan=4)
        cset = plotzonslice(dataSlice, dataIn[jData].lon, z,
                            plot_t,
                            annotate_flag=False,
                            compcont=compcont,
                            cbar_flag=False,
                            conts=conts,
                            dataId=dataIn[jData].srcid,
                            lonLim=lonLim,
                            varName=dataIn[jData].name,
                            varUnits=dataIn[jData].units,
                            **kwargs
                            )

        # Dress plot
        if jData != (len(dataIn) - 1):
            ha.set_xlabel('')

        # Set to match contours between plots
        conts = cset.levels

        # Add source name to subplot
        plt.annotate(dataIn[jData].srcid,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        if jData == 0:
            # Add lats/lons used to first subplot
            plt.annotate(r'$\theta$=[{:0.0f}'.format(lonLim[0]) +
                         ', {:0.0f}]'.format(lonLim[1]) +
                         r' $\phi$=[{:0.0f}'.format(latLim[0]) +
                         ', {:0.0f}]'.format(latLim[1]),
                         xy=(0.5, 1),
                         xycoords='axes fraction',
                         horizontalalignment='center',
                         verticalalignment='bottom'
                         )

            # Add timesteps used to first subplot
            plt.annotate('t = [{t1:0.0f}, {t2:0.0f}]'.format(t1=plot_t[0],
                                                             t2=plot_t[-1]),
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )

    # Clean up layout to fit all labels
    plt.tight_layout()

    # Add common colorbar
    cbar_ax = plt.subplot2grid((2, 5), (0, 4), rowspan=2)
    hcb = plt.colorbar(cset, cax=cbar_ax, format='%0.2g')  # , orientation='horizontal')
    pcb = cbar_ax.get_position()
    cbar_ax.set_position([pcb.x0+0.02, pcb.y0, 0.02, pcb.height])
    cbar_ax.set_ylabel(dataIn[0].name + ' (' +
                       mwfn.getstandardunitstring(dataIn[0].units) + ')')
    hcb.set_ticks(cset.levels)

    # Plot bold contour on colorbar for reference
    if compcont is not None:
        boldLoc = (compcont - conts[0]) / (conts[-1] - conts[0])
        cbar_ax.hlines(boldLoc, 0, 1, colors='k', linewidth=2)

    plt.show()


def diffmeridslice(ctrl, data, plot_t,
                   ctrlV=None,
                   ctrlW=None,
                   dataV=None,
                   dataW=None,
                   compcont=None,
                   conts=None,
                   latLim=np.array([-90., 90.]),
                   lonLim=np.array([0, 360]),
                   vecRefUnits='[m/s, mm/s]',
                   vecRefLen=0.01,
                   wScale=1000,
                   **kwargs
                   ):
    """
    Plot multiple instances of plotmeridslice on one figure
        as difference from provided field.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-11-23

    Args:
        ctrl - latlonvar to be subtracted from other inputs prior to plotting
        data - list containing latlonvar instances for plotting as
            difference from ctrl
        plot_t - times to average over for plotting
        compcont - contours to plot in bold for emphasis/reference
        conts - contour values to use when plotting
        latLim - latitude limits for plot area and zonally averaging
        lonLim - longitude limits for zonally averaging
        **kwargs - options passed directly to plotmeridslice():
            > cMap, extend, grid_flag, maxConts, sym_flag, tickDir, zLim, etc.

    Returns:
        n/a

    Raises:
        none

    Notes:
        n/a
    """

    # Pull vertical coordinate
    if 'z_t' in data[0].__dict__.keys():
        z = data[0].z_t/100
    elif 'z_w_top' in data[0].__dict__.keys():
        z = data[0].z_w_top/100

    # Determine control field to be subtracted from other plots (1st input)
    ctrlSlice = ctrl.calcregzonmean(latLim=latLim,
                                    lonLim=lonLim)
    if ctrlV is not None:
        ctrlVSlice = ctrlV.calcregzonmean(latLim=latLim,
                                          lonLim=lonLim)
    else:
        ctrlVSlice = None
    if ctrlW is not None:
        ctrlWSlice = ctrlW.calcregzonmean(latLim=latLim,
                                          lonLim=lonLim)
    else:
        ctrlWSlice = None

    # Loop through latlonvars provided in data
    for jData in range(len(data)):

        # Compute slices to be plotted
        dataSlice = data[jData].calcregzonmean(latLim=latLim,
                                               lonLim=lonLim)
        # Force times to align
        ctrlSlice = ctrlSlice[range(np.min([ctrlSlice.shape[0],
                                            dataSlice.shape[0]])), :, :]
        dataSlice = dataSlice[range(np.min([ctrlSlice.shape[0],
                                            dataSlice.shape[0]])), :, :]

        if dataV is not None:
            dataVSlice = dataV[jData].calcregzonmean(latLim=latLim,
                                                     lonLim=lonLim)
            dataVSlice = dataVSlice[range(np.min([ctrlVSlice.shape[0],
                                                  dataVSlice.shape[0]])), :, :]
            ctrlVSlice = ctrlVSlice[range(np.min([ctrlVSlice.shape[0],
                                                  dataVSlice.shape[0]])), :, :]
            dVSlice = dataVSlice - ctrlVSlice
        else:
            dVSlice = None

        if dataW is not None:
            dataWSlice = dataW[jData].calcregzonmean(latLim=latLim,
                                                     lonLim=lonLim)
            dataWSlice = dataWSlice[range(np.min([ctrlWSlice.shape[0],
                                                  dataWSlice.shape[0]])), :, :]
            ctrlWSlice = ctrlWSlice[range(np.min([ctrlWSlice.shape[0],
                                                  dataWSlice.shape[0]])), :, :]
            dWSlice = (dataWSlice - ctrlWSlice)*wScale
        else:
            dWSlice = None

        # Plot ocean only output
        ha = plt.subplot2grid((len(data), 5), (jData, 0), colspan=4)
        cset = plotmeridslice(dataSlice - ctrlSlice,
                              data[jData].lat, z,
                              plot_t,
                              annotate_flag=False,
                              compcont=compcont,
                              cbar_flag=False,
                              conts=conts,
                              dataId=data[jData].srcid,
                              latLim=latLim,
                              varName=data[jData].name,
                              varUnits=data[jData].units,
                              vVel=dVSlice,
                              vecRefLen=vecRefLen,
                              vecRefName='Velocity',
                              vecRefUnits=vecRefUnits,
                              wVel=dWSlice,
                              **kwargs
                              )

        # Dress plot
        if jData != (len(data) - 1):
            ha.set_xlabel('')

        # Set to match contours between plots
        conts = cset.levels

        # Add source name to subplot
        plt.annotate(data[jData].srcid + '-' + ctrl.srcid,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        if jData == 0:
            # Add lats/lons used to first subplot
            plt.annotate(r'$\theta$=[{:0.0f}'.format(lonLim[0]) +
                         ', {:0.0f}]'.format(lonLim[1]) +
                         r' $\phi$=[{:0.0f}'.format(latLim[0]) +
                         ', {:0.0f}]'.format(latLim[1]),
                         xy=(0.5, 1),
                         xycoords='axes fraction',
                         horizontalalignment='center',
                         verticalalignment='bottom'
                         )

            # Add timesteps used to first subplot
            plt.annotate('t = [{t1:0.0f}, {t2:0.0f}]'.format(t1=plot_t[0],
                                                             t2=plot_t[-1]),
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )

    # Clean up layout to fit all labels
    try:
        plt.tight_layout()
    except ValueError:
        pass

    # Add common colorbar
    cbar_ax = plt.subplot2grid((2, 5), (0, 4), rowspan=2)
    hcb = plt.colorbar(cset, cax=cbar_ax)  # , orientation='horizontal')
    pcb = cbar_ax.get_position()
    cbar_ax.set_position([pcb.x0+0.02, pcb.y0, 0.02, pcb.height])
    cbar_ax.set_ylabel(r'$\Delta$' + data[0].name + ' (' +
                       mwfn.getstandardunitstring(data[0].units) + ')')
    hcb.set_ticks(cset.levels)

    # Plot bold contour on colorbar for reference
    if compcont is not None:
        boldLoc = (compcont - conts[0]) / (conts[-1] - conts[0])
        cbar_ax.hlines(boldLoc, 0, 1, colors='k', linewidth=2)

    plt.show()


def diffzonslice(ctrlIn, dataIn, plot_t,
                 compcont=None,
                 conts=None,
                 ctrlLineConts_flag=False,
                 ctrlLineConts=None,
                 interpKind='linear',
                 latLim=np.array([-90., 90.]),
                 lonLim=np.array([0, 360]),
                 **kwargs
                 ):
    """
    Plot multiple instances of plotzonslice on one figure
        as differences from provided field

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-03-22

    Args:
        ctrlIn - list containing latlonvar instances for subtracting from
            instances found in dataIn
        dataIn - list containing latlonvar instances for plotting
        plot_t - times to average over for plotting
        compcont - contour to plot in bold for reference
        conts - contour values to use when plotting
        ctrlLineConts_flag - true to contour control values over colors from
            differences
        ctrlLineConts - values for contouring control
        interpKind - kind of interpolation to use if obs-data grids don't match
        latLim - latitude limits for plot area and zonally averaging
        lonLim - longitude limits for zonally averaging
        **kwargs - options passed directly to plotmeridslice():
            > cMap, extend, grid_flag, maxConts, sym_flag, tickDir, zLim

    Returns:
        n/a

    Raises:
        none

    Notes:
        n/a
    """
    # Pull vertical coordinate. Raise exception if cannot find first case
    try:
        if 'z_t' in dataIn[0].__dict__.keys():
            z = dataIn[0].z_t/100
        elif 'z_w_top' in dataIn[0].__dict__.keys():
            z = dataIn[0].z_w_top/100
    except IndexError:
        raise IndexError('No plotable cases found.')

    # Loop through latlonvars provided in dataIn
    for jData in range(len(dataIn)):

        # Determine control field to be subtracted from other plots (1st input)
        ctrlSlice = ctrlIn[jData].calcregmeridmean(latLim=latLim,
                                                   lonLim=lonLim)

        # Compute slices to be plotted
        dataSlice = dataIn[jData].calcregmeridmean(latLim=latLim,
                                                   lonLim=lonLim)

        # Compute difference from reference slice
        try:
            diffSlice = dataSlice - ctrlSlice
        except ValueError:
            print('Regridding reference slice with ' + interpKind +
                  ' interpolation')
            ctrlSlice = np.array(
                [interpolate.interp2d(
                 ctrlIn[jData].lon,
                 ctrlIn[jData].z_t,
                 ctrlSlice[j, :, :],
                 kind=interpKind,
                 copy=True,
                 bounds_error=False,
                 fill_value=np.nan)(dataIn[jData].lon,
                                    dataIn[jData].z_t)
                 for j in range(ctrlSlice.shape[0])
                 ]
                )
            diffSlice = dataSlice - ctrlSlice

        # Plot ocean output
        ha = plt.subplot2grid((len(dataIn), 5), (jData, 0), colspan=4)
        cset = plotzonslice(diffSlice,
                            dataIn[jData].lon, z,
                            plot_t,
                            annotate_flag=False,
                            compcont=compcont,
                            cbar_flag=False,
                            conts=conts,
                            ctrlValues=ctrlSlice,
                            ctrlLineConts_flag=ctrlLineConts_flag,
                            ctrlLineConts=ctrlLineConts,
                            dataId=dataIn[jData].srcid,
                            lonLim=lonLim,
                            varName=dataIn[jData].name,
                            varUnits=dataIn[jData].units,
                            **kwargs
                            )

        # Dress plot
        if jData != (len(dataIn) - 1):
            ha.set_xlabel('')

        # Set to match contours between plots
        conts = cset.levels

        # Add source name to subplot
        plt.annotate(dataIn[jData].srcid + '-' +
                     ctrlIn[jData].srcid,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        if jData == 0:
            # Add lats/lons used to first subplot
            plt.annotate(r'$\theta$=[{:0.0f}'.format(lonLim[0]) +
                         ', {:0.0f}]'.format(lonLim[1]) +
                         r' $\phi$=[{:0.0f}'.format(latLim[0]) +
                         ', {:0.0f}]'.format(latLim[1]),
                         xy=(0.5, 1),
                         xycoords='axes fraction',
                         horizontalalignment='center',
                         verticalalignment='bottom'
                         )

            # Add timesteps used to first subplot
            plt.annotate('t = [{t1:0.0f}, {t2:0.0f}]'.format(t1=plot_t[0],
                                                             t2=plot_t[-1]),
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )
            # Add contour values
            if ctrlLineConts_flag:
                # annotate_ax = plt.subplot2grid((2, 5), (0, 4))
                plt.annotate(' Bold Cont = {:0.2g}'.format(compcont) +
                             mwfn.getstandardunitstring(dataIn[0].units, True) +
                             '\n' + r' $\delta$' +
                             'Cont = {:0.2g}'.format(np.diff(ctrlLineConts).max()) +
                             mwfn.getstandardunitstring(dataIn[0].units, True),
                             xy=(1, 1),
                             xycoords='axes fraction',
                             horizontalalignment='left',
                             verticalalignment='top',
                             )
    # Clean up layout to fit all labels
    plt.tight_layout()

    # Add common colorbar
    if ctrlLineConts_flag:
        cbar_ax = plt.subplot2grid((7, 5), (1, 4), rowspan=6)

    else:
        cbar_ax = plt.subplot2grid((2, 5), (0, 4), rowspan=2)
    hcb = plt.colorbar(cset, cax=cbar_ax)  # , orientation='horizontal')
    pcb = cbar_ax.get_position()
    cbar_ax.set_position([pcb.x0+0.02, pcb.y0, 0.02, pcb.height])
    cbar_ax.set_ylabel(dataIn[0].name + ' (' +
                       mwfn.getstandardunitstring(dataIn[0].units) + ')')
    hcb.set_ticks(cset.levels)

    # Plot bold contour on colorbar for reference
    if (compcont is not None) and (not ctrlLineConts_flag):
        print('oogity')
        boldLoc = (compcont - conts[0]) / (conts[-1] - conts[0])
        cbar_ax.hlines(boldLoc, 0, 1, colors='k', linewidth=2)

    plt.show()

    return


def find_nearest(array, value):
    """
    from:
    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-
        array
    2015-07-01
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def getcmap(plotVar,
            diff_flag=False):
    """
    Assign colormap for each plotVar

    Author:
        Matthew Woelfle

    Version Date:
        2017-10-17

    Args:
        plotVar - name of variable (in CESM parlance) for which cmap is to be
            retrieved

    Kwargs:
        diff_flag - true if plotting difference in variable
    """
    if not diff_flag:
        try:
            cMap = {'FLNS': 'Blues',
                    'FNS': 'RdBu_r',
                    'FSNS': 'Reds',
                    'LHFLX': 'Blues',
                    'OMEGA500': 'RdBu_r',
                    'OMEGA850': 'RdBu_r',
                    'PRECC': 'Purples',
                    'PRECL': 'Purples',
                    'PRECT': 'Purples',
                    'precip': 'Purples',
                    'SHFLX': 'Blues',
                    'TAUX': 'RdBu_r',
                    'TAUY': 'RdBu_r',
                    'TS': 'Reds',
                    'curlTau': 'RdBu_r',
                    'curlTau_y': 'RdBu_r',
                    'divTau': 'RdBu_r',
                    'ekmanx': 'RdBu_r',
                    'ekmany': 'RdBu_r',
                    'sst': 'Reds',
                    'sverdrupx': 'RdBu_r',
                    'MGx': 'RdBu_r'
                    }[plotVar]
        except KeyError:
            cMap = 'Greys'
    else:
        try:
            cMap = {'FLNS': 'RdBu',
                    'FNS': 'RdBu',
                    'FSNS': 'RdBu_r',
                    'LHFLX': 'RdBu',
                    'OMEGA500': 'RdBu_r',
                    'OMEGA850': 'RdBu_r',
                    'PRECC': 'PuOr',  # 'RdBu',
                    'PRECL': 'PuOr',  # 'RdBu',
                    'PRECT': 'PuOr',  # 'RdBu',
                    'PS': 'RdBu_r',
                    'SHFLX': 'RdBu',
                    'TAUX': 'RdBu_r',
                    'TAUY': 'RdBu_r',
                    'TS': 'RdBu_r',
                    'curlTau': 'RdBu_r',
                    'curlTau_y': 'RdBu_r',
                    'divTau': 'RdBu_r',
                    'ekmanx': 'RdBu_r',
                    'ekmany': 'RdBu_r',
                    'sverdrupx': 'RdBu_r',
                    'MGx': 'RdBu_r'
                    }[plotVar]
        except KeyError:
            cMap = 'RdBu_r'

    return cMap


def getlatlbls(latLim):
    """
    Get nicely spaced latitudes for labeling
    """
    # Compute latitude labels from limits
    latLbls = np.arange(np.ceil(latLim[0]), np.floor(latLim[-1]) + 0.01,
                        (np.floor(latLim[-1]) - np.ceil(latLim[0])) / 6)
    # Prevent wonky/long labels
    if any(latLbls*2 != np.floor(latLbls*2)):
        latLbls = np.arange(np.ceil(latLim[0]), np.floor(latLim[-1]) + 0.01,
                            (np.floor(latLim[-1]) - np.ceil(latLim[0])) / 4)

    # Return latitude labels
    return latLbls


def getlonlbls(lonLim):
    """
    Get nicely spaced longitudes for labeling
    """
    # Compute longitude labels from limits
    lonLbls = np.arange(np.ceil(lonLim[0]), np.floor(lonLim[-1]) + 0.01,
                        (np.floor(lonLim[-1]) - np.ceil(lonLim[0])) / 5)

    # Prevent wonky/long labels
    if any(lonLbls*2 != np.floor(lonLbls*2)):
        lonLbls = np.arange(np.ceil(lonLim[0]), np.floor(lonLim[-1]) + 0.01,
                            (np.floor(lonLim[-1]) - np.ceil(lonLim[0])) / 4)

    # Return longtitude labels
    return lonLbls


def getlatlimstring(latLim,
                    joinStr='-',
                    join_flag=True,
                    ):
    """
    Get string for latitude limits
    """

    # Create latlimString
    latLimStr = ['{:02.0f}'.format(np.abs(latLim[x])) +
                 'S'*int(latLim[x] < 0) +
                 'N'*int(latLim[x] > 0)
                 for x in range(latLim.size)]
    if join_flag:
        return joinStr.join(latLimStr)
    else:
        return latLimStr


def getlonlimstring(lonLim,
                    joinStr='-',
                    join_flag=True,
                    lonFormat='E',
                    ):
    """
    Get string for latitude limits

    Notes:
        Currently tailored to Pacific limits which don't wrap 0.
    """
    if lonFormat == 'EW':
        lonLim = np.array([lonLim[j] if (lonLim[j] < 180) else lonLim[j] - 360
                           for j in range(len(lonLim))])
        if all(lonLim < 0) and (lonLim[-1] > lonLim[0]):
            lonLim = lonLim[::-1]
        lonLimStr = ['{:2.0f}'.format(np.abs(lonLim[x])) +
                     'W'*int(lonLim[x] < 0) +
                     'E'*int(lonLim[x] > 0)
                     for x in range(len(lonLim))]
    elif lonFormat == 'E':
        lonLimStr = ['{:2.0f}'.format(np.abs(lonLim[x])) +
                     'W'*(lonLim[x] < 0) +
                     'E'*(lonLim[x] > 0)
                     for x in range(len(lonLim))]

    if join_flag:
        return joinStr.join(lonLimStr)
    else:
        return lonLimStr


def getmaxxlim(hf=None):
    """
    Return maximum x limit for a set of subplots

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-10
    """
    # Pull figure handle if not specified
    if hf is None:
        hf = plt.gcf()

    # Pull list of axes
    ax_list = hf.get_axes()

    # Loop through axes and find min and max limits for x axis
    for jAx in range(len(ax_list)):
        if jAx == 0:
            xLim = np.array(ax_list[jAx].get_xlim())
        else:
            xLim = np.array([np.min([xLim[0],
                                     np.array(ax_list[jAx].get_xlim())[0]]),
                             np.max([xLim[1],
                                     np.array(ax_list[jAx].get_xlim())[1]])
                             ]
                            )

    # Return broadest necessary x limts
    return xLim


def getmaxylim(hf=None):
    """
    Return maximum y limit for a set of subplots

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-10
    """
    # Pull figure handle if not specified
    if hf is None:
        hf = plt.gcf()

    # Pull list of axes
    ax_list = hf.get_axes()

    # Loop through axes and find min and max limits for y axis
    for jAx in range(len(ax_list)):
        if jAx == 0:
            yLim = np.array(ax_list[jAx].get_ylim())
        else:
            yLim = np.array([np.min([yLim[0],
                                     np.array(ax_list[jAx].get_ylim())[0]]),
                             np.max([yLim[1],
                                     np.array(ax_list[jAx].get_ylim())[0]])
                             ]
                            )

    # Return broadest necessary y limts
    return yLim


def getplotunitstring(unitIn):
    """
   Get prettier name of a unit for plotting purposes

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-09-06

    Args:
        unitIn - name of unit to prettify

    Returns:
        plotUnit - prettier unit for plotting

    Notes:
        n/a
    """
    try:
        plotUnit = {'N/m2': '$\mathregular{N/m^{2}}$',
                    'N/m^2': '$\mathregular{N/m^{2}}$',
                    }[unitIn]
    except KeyError:
        plotUnit = unitIn

    return plotUnit


def getplotvarstring(varName):
    """
   Get prettier name of a variable for plotting purposes

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-10-17

    Args:
        varName - name of variable as read from file

    Returns:
        plotName - prettier name for plotting

    Notes:
        n/a
    """

    try:
        plotName = {'ADVT3D': '$\mathregular{[\dot{Q}_{adv}]}$',
                    'dEdt': r'$\frac{d}{dt}$OHC',
                    # 'delQx': r'$-u\frac{\partial}{\partial x}OHC$',
                    # 'delQy': r'$-v\frac{\partial}{\partial y}OHC$',
                    # 'delQz': r'$-w\frac{\partial}{\partial z}OHC$',
                    'delQx': '$\mathregular{[\dot{Q}_{adv,x}]}$',
                    'delQy': '$\mathregular{[\dot{Q}_{adv,y}]}$',
                    'delQz': '$\mathregular{[\dot{Q}_{adv,z}]}$',
                    'delQxyz': r'$-\nabla_{3D}\cdot(V*OHC)$',
                    'delQxyzRef': r'$-OHC\cdot\nabla_{3D}V$',
                    'dOHCZ': (r'$\frac{\mathregular{d}}{\mathregular{dt}}' +
                              r'\mathregular{OHC_{100}}$'),
                    'DIA_IMPVF_TEMP': 'Diabatic\nmixing',
                    'FLNS': r'$F_{LW,sfc}',
                    'FNS': r'$F_{net,sfc}$',
                    'FSNS': r'$F_{SW,sfc}$',
                    'HDIFT3D': r'Horiz. Diff.',
                    'KPP_SRC_TEMP': r'KPP',
                    'LHFLX': r'$H_L$',
                    'PRECC': 'Conv. Precip. Rate',
                    'PRECL': 'Large Scale Precip. Rate',
                    'PRECT': 'Precipitation Rate',
                    'SHF': '$\mathregular{\dot{Q}_{sfc}}$',
                    'SHFLX': r'$H_S$',
                    'TAUX': (r'$\tau_x$'),
                    'TS': 'Surface Temperature',
                    'TEMP': 'Temperature',
                    'udT': (r'$-\overline{u}\frac{\partial}{\partial x}' +
                            '\overline{OHC}$'),
                    'vdT': (r'$-\overline{v}\frac{\partial}{\partial y}' +
                            '\overline{OHC}$'),
                    'wdT': (r'$-\overline{w}\frac{\partial}{\partial z}' +
                            '\overline{OHC}$'),
                    }[varName]
    except KeyError:
        plotName = varName

    return plotName


def lonaxislblstoew(ax=None,
                    axis='x'
                    ):
    """
    Convert axis labels on plot to be degE and degW instead of just degE

    Args:
        ax - axis handle
        axis - axis for which labels are to be updated
    """

    # Get axis handle if not provided
    if ax is None:
        ax = plt.gca()

    # Draw canvas to ensure labels are defined
    plt.gcf().canvas.draw()

    # Get current axis labels
    if axis == 'x':
        labels = ax.get_xticklabels()
    else:
        labels = ax.get_yticklabels()

    # Define new labels as degE and degW rather than degE
    newLabels = []
    degSign = u'\N{DEGREE SIGN}'
    for label in labels:
        try:
            label = int(label.get_text())
            newLabel = ('{:d}'.format(label) + degSign + 'E'
                        if label < 180
                        else ('{:d}'.format(360 - label) + degSign + 'W'
                              if label > 180
                              else '180' + degSign)
                        )
        except ValueError:
            newLabel = ''
        newLabels.append(newLabel)

    # Set new axis labels on requested axis
    if axis == 'x':
        ax.set_xticklabels(newLabels)
    elif axis == 'y':
        ax.set_yticklabels(newLabels)

    return


def plotg16map(lon, lat, plotData, tStep,
               box_flag=False,
               boxColor=None,
               boxLat=np.array([-20, 0]),
               boxLon=np.array([210, 260]),
               boxLineWidth=3,
               casename_flag=False,
               caseString=None,
               cbar_flag=True,
               cbar_tickLabels=None,
               cMap='RdBu_r',
               compcont=None,
               extend='both',
               fill_color=[1, 1, 0],
               fontsize=10,
               latlbls=None,
               latLim=np.array([-90, 90]),
               levels=None,
               lonlbls=None,
               lonLim=np.array([0, 360]),
               plotcoast_flag=True,
               projection='cea',
               resolution='l',
               time_flag=False,
               timeString=None,
               varName='',
               varUnits=''
               ):
    """
    Plot POP data on its native grid using contourf. pcolormesh does not work
        properly.
    """
    # Mean data if more than 1 time step provided
    if tStep.size > 1:
        plotData = plotData[tStep, :, :].mean(axis=0)
    else:
        if np.ndim(plotData) == 3:
            plotData = plotData[tStep, :, :].squeeze()

    # make longitudes monotonically increasing.
    lon = np.where(np.greater_equal(lon, min(lon[:, 0])), lon-360, lon)

    # stack grids side-by-side (in longitiudinal direction), so
    # any range of longitudes may be plotted on a world map.
    lon = np.concatenate((lon, lon + 360), -1)
    lat = np.concatenate((lat, lat), -1)
    pData = np.ma.concatenate((plotData, plotData), -1)

    # subplot 1 just shows POP grid cells.
    if projection in ['merc']:
        m = Basemap(projection=projection,
                    lat_ts=20,
                    llcrnrlon=lonLim[0],
                    urcrnrlon=lonLim[1],
                    llcrnrlat=latLim[0],
                    urcrnrlat=latLim[1],
                    resolution=resolution)
        if plotcoast_flag:
            m.drawcoastlines()
            m.fillcontinents(color=fill_color)

        x, y = m(lon, lat)

        im1 = m.contourf(x, y, pData,
                         levels=levels,
                         cmap=cMap)
        plt.colorbar(im1)

    elif projection in ['moll', 'robin', 'cea']:
        if projection in ['moll', 'robin']:
            m = Basemap(projection=projection,
                        lon_0=lonLim.mean(),
                        resolution=resolution)
        elif projection in ['cea']:
            m = Basemap(projection=projection,
                        lon_0=lonLim.mean(),
                        llcrnrlon=lonLim[0],
                        urcrnrlon=lonLim[1],
                        llcrnrlat=latLim[0],
                        urcrnrlat=latLim[1],
                        resolution=resolution)

        if plotcoast_flag:
            m.drawcoastlines()
            m.fillcontinents(color=fill_color)

        x, y = m(lon, lat)

        pVar = pData

        # Remove cross over top of map
        pVar1 = np.where(np.concatenate([np.zeros([1, pVar.shape[1]],
                                                  dtype=bool),
                                         np.diff(lat, axis=0) < 0],
                                        axis=0),
                         np.nan, pVar)
        # Remove cross over side of map
        pVar1 = np.where(np.less(lon, lonLim[0]), np.nan, pVar1)
        pVar1 = np.where(np.greater(lon, lonLim[1]), np.nan, pVar1)

        # Collect missing piece
        # Pull cross over top of map
        pVar2 = np.where(np.concatenate([np.zeros([1, pVar.shape[1]],
                                                  dtype=bool),
                                         np.diff(lat, axis=0) < 0],
                                        axis=0),
                         pVar, np.nan)
        pVar2 = np.where(np.less(lon, lonLim[0]), np.nan, pVar2)
        pVar2 = np.where(np.greater(lon, lonLim[1]), np.nan, pVar2)

        # Reapply land mask
        pVar1[pVar.mask] = np.nan
        pVar2[pVar.mask] = np.nan

        # Plot using contours (pcolormesh not working)
        im1 = m.contourf(x, y, pVar1,
                         levels=levels,
                         antialiased=False,
                         cmap=cMap,
                         extend=extend)
        plt.hold(True)
        m.contourf(x, y, pVar2,
                   levels=im1.levels,
                   antialiased=False,
                   cmap=cMap,
                   extend=extend)
        # im1.set_clim(-500, 500)
    # Add meridions and parallels
    if latlbls is not None:
        m.drawparallels(latlbls,
                        labels=[1, 0, 0, 0],
                        fontsize=fontsize)
    if lonlbls is not None:
        m.drawmeridians(lonlbls,
                        labels=[0, 0, 0, 1],
                        fontsize=fontsize)

    # Add case label and time steps
    ax = plt.gca()
    if casename_flag:
        ax.annotate(caseString,
                    xy=(0, 1),
                    xycoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom'
                    )
    if time_flag:
        if timeString is None:
            timeString = 't = %d,%d' % (tStep[0], tStep[-1])
        ax.annotate(timeString,
                    xy=(1, 1),
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom'
                    )

    # Add reference box if requested
    if box_flag:
        if boxColor is None:
            boxColor = 'k'
        if isinstance(boxLon, list):
            if not isinstance(boxColor[0], list):
                boxColor = [list(boxColor) for j in range(len(boxLon))]
            if isinstance(boxLineWidth, int):
                boxLineWidth = [boxLineWidth for j in range(len(boxLon))]
            for j in range(len(boxLon)):
                print('box' + str(j))
                xs = [boxLon[j][0], boxLon[j][1],
                      boxLon[j][1], boxLon[j][0],
                      boxLon[j][0]]
                ys = [boxLat[j][0], boxLat[j][0],
                      boxLat[j][1], boxLat[j][1],
                      boxLat[j][0]]
                print(xs)
                print(ys)
                print(boxColor[j])
                m.plot(xs, ys,
                       color=boxColor[j],
                       linewidth=boxLineWidth[j],
                       latlon=True)
        else:
            xs = [boxLon[0], boxLon[1],
                  boxLon[1], boxLon[0],
                  boxLon[0]]
            ys = [boxLat[0], boxLat[0],
                  boxLat[1], boxLat[1],
                  boxLat[0]]
            print(xs)
            print(ys)
            m.plot(xs, ys,
                   color=boxColor,
                   linewidth=boxLineWidth,
                   latlon=True)

    # Add colorbar
    if cbar_flag:
        cb = plt.colorbar(
            im1,
            orientation='horizontal',
            label=(varName + ' (' +
                   mwfn.getstandardunitstring(varUnits) + ')'))
        if cbar_tickLabels is not None:
            cb.set_ticklabels(cbar_tickLabels)

    plt.show()

    # return nothing (placeholder to mark end of function)
    return


def plotglobalmap(lon, lat, plotData, tStep,
                  cMap='RdBu_r',
                  varName=None,
                  varUnits=None
                  ):
    """
    Plot a global map of a given data field. Input plotData should have all
    time steps, or tStep should be np.array([0]).
    Uses pcolormesh for now, may go to contourf at some point
    """
    # Mean data if more than 1 time step provided
    if tStep.size > 1:
        plotData = np.mean(plotData[tStep, :, :],
                           axis=0)
    else:
        plotData = np.squeeze(plotData[tStep, :, :])

    # Rearrange fields for plotting
    pData = np.column_stack((plotData[:, :],
                             plotData[:, 1]))

    # Set x axis info
    xVec = np.append(lon[:], lon[-1]+lon[1]-lon[0])
    xTicks = np.array([0, 60, 120, 180, 240, 300, 360])

    # Set y axis info
    yVec = np.sin(lat[:]*pi/180)
    yTicks = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90])

    # Plot figure
    # plt.figure()

    # Grab axis handle
    ha = plt.gca()

    # Set colormap
    # cMap = cbmap.RdBu_9.mpl_colormap

    # Plot data with pcolor
    hp = plt.pcolormesh(xVec, yVec, pData)
    # plt.contourf(xVec, yVec, pData)

    # Update colormap, use default or user provided map
    if cMap:
        hp.set_cmap(cMap)
    else:
        hp.set_cmap('RdBu_r')

    # Add title of variable with units
    varString = ""
    if varName:
        varString = varString + varName
    if varUnits:
        varString = varString + " (" + varUnits + ")"
    plt.annotate(varString,
                 xy=(0, 1),
                 xycoords='axes fraction',
                 horizontalalignment='left',
                 verticalalignment='bottom'
                 )

    # Add timesteps to plot annotations
    plt.annotate("t = [" + str(np.min(tStep)) + ", " +
                 str(np.max(tStep)) + "]",
                 xy=(1, 1),
                 xycoords='axes fraction',
                 horizontalalignment='right',
                 verticalalignment='bottom'
                 )

    # Format x axis
    plt.xlim([np.min(xVec), np.max(xVec)])
    ha.set_xticks((xTicks))
    ha.set_xticklabels((xTicks))

    # Format y axis
    plt.ylim([np.min(yVec), np.max(yVec)])
    ha.set_yticks((np.sin(yTicks*pi/180)))
    ha.set_yticklabels((yTicks))

    # Add colorbar
    plt.colorbar()

    # Show plot
    plt.show()

    # Output plotData
    return plotData


def plotmap(lon, lat, plotData,
            tStep=np.array([0]),
            tStepLabel_flag=True,
            box_flag=False,
            boxLat=np.array([-20, 0]),
            boxLon=np.array([210, 260]),
            boxLineStyle='-',
            caseString=None,
            cbar_flag=True,
            cbar_downshift=0.05,
            cbar_dy=0.1,
            cbar_height=0.03,
            cbar_ticks=None,
            cbar_tickLabels=None,
            cMap=None,
            compcont=None,
            extend='both',
            fill_color=[1, 1, 0],
            fontsize=12,
            labelValueSize=10,
            latlbls=None,
            latLim=np.array([-90, 90]),
            levels=None,
            lonlbls=None,
            lonLim=np.array([0, 360]),
            plotcoast_flag=True,
            plottype='contourf',
            projection='cea',
            quiver_flag=False,
            quiverScale=0.4,
            quiverUnits='inches',
            returnM_flag=False,
            resolution='l',
            rmzonalmean_flag=False,
            subSamp=5,
            U=None,
            Uname=None,
            Uref=None,
            Uunits=None,
            V=None,
            varName=None,
            varUnits=None,
            ):
    """
    Plot a global map of a given data field.
    Uses contourf by default, pcolormesh also working.
    """
    # Change default colorbar shift if run from command line
    try:
        if os.isatty(sys.stdout.fileno()) & (cbar_downshift == 0.05):
            cbar_downshift = 0.12
    except:
        pass

    # Get default colormap if none provided
    if cMap is None:
        cMap = getcmap(varName)

    if latlbls is None:
        latlbls = np.arange(latLim[0], latLim[1]+1, 15)

    if lonlbls is None:
        lonlbls = np.arange(lonLim[0], lonLim[1]+1, 30)

    # Compress data to [lon, lat] by averaging over time(s) (if needed)
    if np.ndim(plotData) == 3:
        plotData = plotData[tStep, :, :].mean(axis=0)
    if np.ndim(U) == 3:
        U = U[tStep, :, :].mean(axis=0)
    if np.ndim(V) == 3:
        V = V[tStep, :, :].mean(axis=0)

    # Prep data, lats, and lons for plotting
    if np.ndim(lon) == 1 & np.ndim(lat) == 1:
        # Rearrange fields to repeat across lon edge
        pData = np.column_stack((plotData[:, :],
                                 plotData[:, 1]))  # <- should this be 0?

        if U is not None:
            U = np.column_stack((U[:, :],
                                 U[:, 1]))
        if V is not None:
            V = np.column_stack((V[:, :],
                                 V[:, 1]))

        # Set x axis info to repeat across lon edge
        lon = np.append(lon[:], lon[-1]+lon[1]-lon[0])

        # Create lat/lon meshed grids
        lonG, latG = np.meshgrid(lon, lat)
    elif np.ndim(lon) == 2 & np.ndim(lat) == 2:
        pData = plotData
        lonG = lon
        latG = lat

    # Plot figure
    # fig = plt.figure()
    ax = plt.gca()  # fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Set focal points for map
    lat_0 = latLim.mean()
    lon_0 = lonLim.mean()

    # Create axis for plotting
    if projection in ['cea']:
        m = Basemap(projection='cea',
                    llcrnrlat=latLim[0],
                    llcrnrlon=lonLim[0],
                    urcrnrlat=latLim[1],
                    urcrnrlon=lonLim[1],
                    resolution=resolution)  # resolution)
    elif projection.lower() in ['merc', 'mercator']:
        m = Basemap(projection='merc',
                    lat_ts=20,
                    llcrnrlon=lonLim[0],
                    urcrnrlon=lonLim[1],
                    llcrnrlat=np.max([latLim[0], -84]),
                    urcrnrlat=np.min([latLim[1], 84]),
                    resolution=resolution)
    elif projection.lower() in ['robin', 'robinson']:
        # pData, lonG = bmp.shiftgrid(360, pData, lonG, start=False)
        m = Basemap(projection=projection,
                    lat_0=lat_0,
                    lon_0=lon_0,
                    resolution=resolution)

    # Draw line around outer edge of map and set background color
    m.drawmapboundary(fill_color=fill_color)

    # Remove zonal mean if requested
    if rmzonalmean_flag:
        pData = (pData.transpose() -
                 pData.mean(axis=1).transpose()
                 ).transpose()

    # Plot with prescribed plotting method
    if plottype in ['contf', 'contourf']:
        im1 = m.contourf(lonG, latG, pData,
                         levels=levels,
                         cmap=cMap,
                         extend=extend,
                         latlon=True
                         )
    elif plottype in ['pcolor', 'pcolormesh']:
        im1 = m.pcolormesh(lonG, latG, pData,
                           shading='faceted',
                           antialiased=True,
                           cmap=cMap,
                           extend=extend,
                           latlon=True)

    # Plot vector field on top of map if requested
    if quiver_flag:

        # Rearrange fields to repeat across lon edge
        q1 = m.quiver(lonG[::subSamp, ::subSamp],
                      latG[::subSamp, ::subSamp],
                      U[::subSamp, ::subSamp],
                      V[::subSamp, ::subSamp],
                      latlon=True,
                      pivot='tail',
                      units=quiverUnits,
                      scale=quiverScale)
        plt.quiverkey(q1, 0.5, 1.05, Uref,
                      Uname + ' (' +
                      '{:0.5f}'.format(Uref).rstrip('0').rstrip('.') + ' ' +
                      mwfn.getstandardunitstring(Uunits) + ')',
                      coordinates='axes',
                      labelpos='E')
        # return (U, V, q1)

    # Plot black contour for reference
    if compcont is not None:
        m.contour(lonG, latG, pData,
                  levels=compcont,
                  colors='k',
                  latlon=True)

    # Add coast lines, meridions, and parallels
    if plotcoast_flag:
        m.drawcoastlines(linewidth=1)
    m.drawparallels(latlbls,
                    labels=[1, 0, 0, 0],
                    fontsize=labelValueSize)
    m.drawmeridians(lonlbls,
                    labels=[0, 0, 0, 1],
                    fontsize=labelValueSize)

    # Add reference box if requested
    if box_flag:
        if any([isinstance(boxLon[0], list),
                isinstance(boxLon[0], np.ndarray)]):
            if isinstance(boxLineStyle, str):
                boxLineStyle = [boxLineStyle]*len(boxLon)
            for jBox in range(len(boxLon)):
                xs = [boxLon[jBox][0], boxLon[jBox][1],
                      boxLon[jBox][1], boxLon[jBox][0],
                      boxLon[jBox][0]]
                ys = [boxLat[jBox][0], boxLat[jBox][0],
                      boxLat[jBox][1], boxLat[jBox][1],
                      boxLat[jBox][0]]
                m.plot(xs, ys,
                       color='k',
                       linewidth=3,
                       lineStyle=boxLineStyle[jBox],
                       latlon=True)
        else:
            xs = [boxLon[0], boxLon[1],
                  boxLon[1], boxLon[0],
                  boxLon[0]]
            ys = [boxLat[0], boxLat[0],
                  boxLat[1], boxLat[1],
                  boxLat[0]]
            m.plot(xs, ys,
                   color='k',
                   linestyle=boxLineStyle,
                   linewidth=3,
                   latlon=True)

    # Update axes locations
    plt.draw()

    #  add colorbar
    if cbar_flag:
        posax = ax.get_position()
        poscb = [posax.x0, posax.y0 - cbar_dy - cbar_downshift,
                 posax.width, cbar_height]
        cbar_ax = plt.gcf().add_axes(poscb)
        cb = plt.colorbar(im1,
                          cax=cbar_ax,
                          orientation='horizontal')
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        if cbar_tickLabels is not None:
            cb.set_ticklabels(cbar_tickLabels)
        cb.ax.set_xlabel(varName + ' (' +
                         mwfn.getstandardunitstring(varUnits) + ')')
        # for label in cb.ax.xaxis.get_ticklabels()[::2]:
        #     label.set_visible(False)

    # Plot reference contour on colorbar for reference
    if (compcont is not None) & cbar_flag:
        boldLoc = (compcont - im1.levels[0]) / float(im1.levels[-1] -
                                                     im1.levels[0])
        cbar_ax.vlines(boldLoc, 0, 1, colors='k', linewidth=1)

    # Add case label and time steps
    ax.annotate(caseString,
                xy=(0, 1),
                xycoords='axes fraction',
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=fontsize,
                )
    if tStepLabel_flag:
        ax.annotate('t = %d,%d' % (tStep[0], tStep[-1]),
                    xy=(1, 1),
                    xycoords='axes fraction',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=fontsize,
                    )

    # output plotted data field
    if returnM_flag:
        return (im1, ax, m)
    else:
        return (im1, ax)


def plotmeridmean(meridMean, lon, timeVec,
                  cbar_ticks=None,
                  cbarLabel_flag=True,
                  cMap=None,
                  conts=None,
                  compcont=None,
                  dataId=None,
                  extend='both',
                  grid_flag=False,
                  lonLim=np.array([-90, 90]),
                  maxConts=10,
                  sym_flag=False,
                  tickDir='out',
                  timeLim=None,
                  varName=None,
                  varUnits=None
                  ):
    """
    Contours the meridional mean over a given region on a planet. Meridional
        mean should come from output from mdwfunctions.calcregmeridmean

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-06-29

    Args:
        meridMean - time x lon numpy array of meridional means for contouring
        lon - longitudes corresponding to meridMean
        timeVec - times corresponding to meridMean
        cbar_ticks - ticks for colorbar
        cbarLabel_flag - true to label colorbar with variable name and units
        cMap - colormap to use when plotting
        conts - contour values to use when plotting
        dataId - identifier for data to be printed in top right corner of plot
        grid_flag - True to plot grid atop contours for reference
        lonLim - longitude limits for plot area
        maxConts - maximum number of contours to use
        sym_flag - True to make color limits symmetric about 0
        tickDir - direction ticks point from axis ('in' or 'out')
        timeLim - time limits for plot area
        varName - name of variable being plotted
        varUnits - units of variable being plotted

    Returns:
        cset1.levels- values for contour levels

    Raises:
        none

    Notes:
        n/a
    """

    # Set x axis info
    xVec = lon
    if (lonLim == [0, 360]).all():
        xTicks = np.arange(0, 361, 60)
    else:
        xTicks = (np.linspace(lonLim[0], lonLim[1], num=7).round()).astype(int)

    # Get axis handle
    ha = plt.gca()

    # Plot contour plot
    if varName is None:
        # Plot data
        plt.contourf(xVec, timeVec, meridMean)

    else:
        if varName.lower() == 'prect':
            if cMap is None:
                if sym_flag:
                    cMap = 'RdBu'
                else:
                    cMap = 'Blues'
            if ~(varUnits is None):
                # Convert precipitation to mm/d from m/s
                if varUnits.lower() == 'm/s':
                    meridMean = meridMean*8.64*10**7
                    varUnits = 'mm/d'
        elif varName.lower() == 'ts':
            if cMap is None:
                if sym_flag:
                    cMap = 'RdBu_r'
                else:
                    cMap = 'Reds'

        if conts is None:
            conts = regularconts(meridMean,
                                 maxConts=maxConts,
                                 sym_flag=sym_flag
                                 )

        if cMap is None:
            if sym_flag:
                cMap = 'RdBu_r'

        # Plot filled contours
        cset1 = plt.contourf(xVec, timeVec, meridMean,
                             levels=conts,
                             cmap=cMap,
                             extend=extend)

        # Plot contour edges
        if compcont is not None:
            plt.contour(xVec, timeVec, meridMean,
                        levels=compcont,
                        colors='k',
                        hold='on'
                        )

    # Add colorbar to figure
    if cbarLabel_flag:
        varString = ''
        if varName:
            varString = varString + varName
        if varUnits:
            varString = varString + ' (' + varUnits + ')'
    else:
        varString = ''
    hcb = plt.colorbar(cset1,
                       label=varString)
    if cbar_ticks is not None:
        hcb.set_ticks(cbar_ticks)
    if compcont is not None:
        boldLoc = ((compcont - cset1.levels[0]) /
                   (cset1.levels[-1] - cset1.levels[0]))
        hcb.ax.hlines(boldLoc, 0, 1, colors='k', linewidth=1)

    # Format x axis
    ax = plt.gca()
    if lonLim is None:
        plt.xlim([np.min(xVec), np.max(xVec)])
    else:
        plt.xlim(lonLim)
    ha.set_xticks(xTicks)
    ha.set_xticklabels((xTicks))
    plt.xlabel('Longitude')
    ax.tick_params(axis='x', direction=tickDir)

    # Format y axis
    if ~(timeLim is None):
        plt.ylim(timeLim)
    plt.ylabel('Time (mon.)')
    ax.tick_params(axis='y', direction=tickDir)

    # Add grid lines for readability
    if grid_flag:
        plt.grid(b=True, which='major', color=[0.5, 0.5, 0.5], linestyle='--')

    # Add title of variable wth units
#    varString = ""
#    if varName:
#        varString = varString + varName
#    if varUnits:
#        varString = varString + " (" + varUnits + ")"
#    plt.annotate(varString,
#                 xy=(0, 1),
#                 xycoords='axes fraction',
#                 horizontalalignment='left',
#                 verticalalignment='bottom'
#                 )

    # Add data source
    if ~(dataId is None):
        plt.annotate(dataId,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

    # Make plot appear
    plt.show()

    # Return contour levels
    return cset1.levels


def plotmeridslice(zonMean, lat, z, plot_t,
                   annotate_flag=True,
                   cbar_flag=False,
                   cMap=None,
                   compcont=None,
                   conts=None,
                   dataId=None,
                   extend='both',
                   grid_flag=False,
                   latLim=np.array([-90, 90]),
                   maxConts=10,
                   quiver_flag=False,
                   sinLat_flag=True,
                   subSamp=1,
                   sym_flag=False,
                   tickDir='out',
                   varName=None,
                   varUnits=None,
                   vVel=None,
                   vecRef_flag=True,
                   vecRefLen=None,
                   vecRefName=None,
                   vecRefUnits=None,
                   wVel=None,
                   zLim=None
                   ):
    """
    Contours a meridional slice of a field over a given region and time period.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-14

    Args:
        zonMean - time x depth x lat numpy array of zonal means for
            contouring
        lat - latitudes for dataIn
        z - depths for dataIn
        plot_t - times to average over for plotting
        annotate_flag - true to anotate the plot with variable and id info
        cbar_flag - true to plot colorbar
        cMap - colormap to use when plotting
        compcont - contour(s) to highlight for comparing across axes
        conts - contour values to use when plotting
        dataId - identifier for data to be printed in top right corner of plot
        extend - extend colorbar beyond color limits?
            options = ['neither', 'min', 'max', 'both']
        grid_flag - True to plot grid atop contours for reference
        latLim - latitude limits for plot area
        maxConts - maximum number of contours to use
        sym_flag - True to make color limits symmetric about 0
        tickDir - direction ticks point from axis ('in' or 'out')
        varName - name of variable being plotted
        varUnits - units of variable being plotted

    Returns:
        cset1.levels - values for contour levels

    Raises:
        none

    Notes:
        n/a
    """

    # Set x axis info
    if sinLat_flag:
        xVec = np.sin(lat[:]*pi/180.)
    else:
        xVec = lat[:]
    if (latLim == [-90, 90]).all():
        xTicks = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90])
    else:
        xTicks = (np.linspace(latLim[0], latLim[1], num=9).round()).astype(int)

    # Get axis handle
    ha = plt.gca()

    # Plot contour plot
    if varName is None:
        # Plot data
        plt.contourf(xVec, z, zonMean[plot_t, :, :].mean(axis=0),
                     cmap=cMap)

    else:
        if varName in ['UVEL', 'VVEL', 'WVEL']:
            if cMap is None:
                cMap = 'RdBu_r'
            sym_flag = True
        elif varName in ['T', 'TEMP']:
            if cMap is None:
                cMap = 'Reds'
            sym_flag = False

        # Set contour values
        if conts is None:
            conts = regularconts(zonMean,
                                 maxConts=maxConts,
                                 sym_flag=sym_flag
                                 )

        # Set colormap
        if cMap is None:
            if sym_flag:
                cMap = 'RdBu_r'
            else:
                cMap = 'Greys'

        # Plot filled contours
        cset1 = plt.contourf(xVec, z, zonMean[plot_t, :, :].mean(axis=0),
                             conts,
                             cmap=cMap,
                             extend=extend)

    # Plot bold contour if requested
    if compcont is not None:
        plt.contour(xVec, z, zonMean[plot_t, :, :].mean(axis=0),
                    np.array([compcont]),
                    colors='k',
                    linewidths=2,
                    hold='on'
                    )

    # Add velocity vectors to plot if requested
    if (quiver_flag & (vVel is not None) & (wVel is not None)):
        q1 = plt.quiver(xVec, z,
                        vVel[plot_t, :, :].mean(axis=0)[::1, ::1],
                        wVel[plot_t, :, :].mean(axis=0)[::1, ::1],
                        pivot='tail')
        if vecRef_flag:
            plt.quiverkey(q1, 0.75, -0.15, vecRefLen,
                          vecRefName + ' (' +
                          '{:0.5f}'.format(vecRefLen).rstrip('0').rstrip('.') +
                          ' ' + mwfn.getstandardunitstring(vecRefUnits) + ')',
                          coordinates='axes',
                          labelpos='E')
    # Add colorbar to figure
    if cbar_flag:
        plt.colorbar(cset1)

    # Format x axis
    ax = plt.gca()
    if latLim is None:
        plt.xlim([np.min(xVec), np.max(xVec)])
    else:
        plt.xlim(np.sin(latLim*pi/180.))
    plt.xlabel('Latitude ($^\circ$N)')
    ax.tick_params(axis='x', direction=tickDir)
    if sinLat_flag:
        ha.set_xticks((np.sin(xTicks.astype(float)*pi/180.)))
    else:
        ha.set_xticks(xTicks)
    ha.set_xticklabels((xTicks))

    # Format y axis
    if zLim is None:
        plt.ylim([np.min(z), np.max(z)])
    else:
        plt.ylim(zLim)
    plt.ylabel('Depth (m)')
    ax.tick_params(axis='y', direction=tickDir)
    ha.invert_yaxis()

    # Add grid lines for readability
    if grid_flag:
        plt.grid(b=True, which='major', color=[0.5, 0.5, 0.5], linestyle='--')

    # Add annotations to plot
    if annotate_flag:
        # Add title of variable wth units
        varString = ""
        if varName:
            varString = varString + varName
        if varUnits:
            varString = varString + " (" + mwfn.getstandardunitstring(varUnits) + ")"
        plt.annotate(varString,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        # Add data source/ID
        if ~(dataId is None):
            plt.annotate(dataId,
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )

    # Make plot appear
    plt.show()

    # Return contour levels
    return cset1


def plotzonalmeanline(lat, lon, plotData,
                      tStep=None,
                      varName=None,
                      varUnits=None
                      ):
    """
    Plot zonal mean of a given variable meaned(?) over tStep
    """
    if (tStep is None):
        tStep = plotData.shape[0]

    # Mean data if more than 1 time step provided
    if tStep.size > 1:
        plotData = np.mean(plotData[tStep, :, :],
                           axis=0)
    else:
        plotData = np.squeeze(plotData[tStep, :, :])

    # Compute zonal mean of plotData
    meanPlotData = np.mean(plotData, axis=1)

    # Set y axis info
    yVec = np.sin(lat[:]*pi/180)
    yTicks = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90])

    # Plot figure
    # plt.figure()

    # Grab axis handle
    ha = plt.gca()

    # Plot zonal mean
    plt.plot(meanPlotData, yVec)

    # Add title of variable wth units
    varString = ""
    if varName:
        varString = varString + varName
    if varUnits:
        varString = varString + " (" + varUnits + ")"
    plt.annotate(varString,
                 xy=(0, 1),
                 xycoords='axes fraction',
                 horizontalalignment='left',
                 verticalalignment='bottom'
                 )

    # Add timesteps to plot annotations
    plt.annotate("t = [" + str(np.min(tStep)) + ", " +
                 str(np.max(tStep)) + "]",
                 xy=(1, 1),
                 xycoords='axes fraction',
                 horizontalalignment='right',
                 verticalalignment='bottom'
                 )

    # Format y axis
    plt.ylim([np.min(yVec), np.max(yVec)])
    ha.set_yticks((np.sin(yTicks*pi/180)))
    ha.set_yticklabels((yTicks))

    # Show plot
    plt.show()

    # return output
    return meanPlotData


def plotzonmean(zonMean, lat, timeVec,
                cbar_flag=True,
                cbar_ticks=None,
                cbarLabel_flag=True,
                cMap=None,
                conts=None,
                compcont=None,
                dataId=None,
                extend='both',
                grid_flag=False,
                latLim=np.array([-90, 90]),
                maxConts=10,
                sym_flag=False,
                tickDir='out',
                timeLim=None,
                varName=None,
                varUnits=None,
                yTickLabelFreq=2
                ):
    """
    Contours the zonal mean over a given region on a planet. Zonal mean should
        come from output from mdwfunctions.calcregzonmean

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-10
    Args:
        zonMean - time x lat numpy array of zonal means for contouring
        lat - latitudes corresponding to meridMean
        timeVec - times corresponding to meridMean
        ...begin optional imputs...
        cbar_flag - true to plot colorbar
        cbar_ticks - ticks for colorbar
        cbarLabel_flag - true to label colorbar
        cMap - colormap to use when plotting
        conts - contour values to use when plotting
        compcont - contour(s) to highlight for comparing across axes
        dataId - identifier for data to be printed in top right corner of plot
        grid_flag - True to plot grid atop contours for reference
        latLim - latitude limits for plot area
        maxConts - maximum number of contours to use
        sym_flag - True to make color limits symmetric about 0
        tickDir - direction ticks point from axis ('in' or 'out')
        timeLim - time limits for plot area
        varName - name of variable being plotted
        varUnits - units of variable being plotted
        yTickLabelFreq - Factor by which to reduce # of y ticks (2 = every other)

    Returns:
        cset1.levels() - values for contour levels

    Raises:
        none

    Notes:
        n/a
    """

    # Set y axis info
    yVec = np.sin(lat[:]*pi/180)
    if (latLim == [-90, 90]).all():
        yTicks = np.array([-90, -60, -30, -15, 0, 15, 30, 60, 90])
    else:
        yTicks = (np.linspace(latLim[0], latLim[1], num=9).round()).astype(int)

    # Get axis handle
    ha = plt.gca()

    # Plot contour plot
    if varName is None:
        # Plot data
        cset1 = plt.contourf(timeVec, yVec, zonMean.transpose())

    else:
        if varName.lower() == 'prect':
            if cMap is None:
                if sym_flag:
                    cMap = 'RdBu'
                else:
                    cMap = 'Blues'
            if ~(varUnits is None):
                if varUnits.lower() == 'm/s':
                    zonMean = zonMean*8.64*10**7
                    varUnits = 'mm/d'
        elif varName.lower() == 'ts':
            if cMap is None:
                if sym_flag:
                    cMap = 'RdBu_r'
                else:
                    cMap = 'Reds'
        # Set contour values
        if conts is None:
            conts = regularconts(zonMean,
                                 maxConts=maxConts,
                                 sym_flag=sym_flag
                                 )

        # Set colormap
        if cMap is None:
            if sym_flag:
                cMap = 'RdBu_r'
            else:
                cMap = 'Greys'

        # Plot filled contours
        cset1 = plt.contourf(timeVec, yVec, zonMean.transpose(),
                             levels=conts,
                             cmap=cMap,
                             extend=extend)

        # Plot reference contour
        if compcont is not None:
            plt.contour(timeVec, yVec, zonMean.transpose(),
                        levels=compcont,
                        colors='k',
                        hold='on'
                        )

    # Add colorbar to figure
    if cbar_flag:
        if cbarLabel_flag:
            varString = ''
            if varName:
                varString = varString + varName
            if varUnits:
                varString = varString + ' (' + varUnits + ')'
        else:
            varString = ''
        hcb = plt.colorbar(cset1,
                           label=varString)
        if cbar_ticks is not None:
            hcb.set_ticks(cbar_ticks)

        if compcont is not None:
            boldLoc = ((compcont - cset1.levels[0]) /
                       (cset1.levels[-1] - cset1.levels[0]))
            hcb.ax.hlines(boldLoc, 0, 1, colors='k', linewidth=1)

    # Format x axis
    ax = plt.gca()
    if ~(timeLim is None):
        plt.xlim(timeLim)
    plt.xlabel('Time (d)')
    ax.tick_params(axis='x', direction=tickDir)

    # Format y axis
    if latLim is None:
        plt.ylim([np.min(yVec), np.max(yVec)])
    else:
        plt.ylim(np.sin(latLim*pi/180.))
    ha.set_yticks((np.sin(yTicks.astype(float)*pi/180.)))
    # print(yTicks)
    # print(yTickLabelFreq)
    ha.set_yticklabels([str(yTicks[j])
                        if j/float(yTickLabelFreq) == j/int(yTickLabelFreq)
                        else ''
                        for j in range(len(yTicks))])
    plt.ylabel('Latitude')
    ax.tick_params(axis='y', direction=tickDir)
    # rcParams['ytick.direction'] = tickDir

    # Add grid lines for readability
    if grid_flag:
        plt.grid(b=True, which='major', color=[0.5, 0.5, 0.5], linestyle='--')

    # Add title of variable wth units
    if dataId is not None:
        plt.annotate(dataId,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

    # Make plot appear
    plt.show()

    # Return contour levels
    return cset1.levels, cset1


def plotzonmeanannual(ncDataList,
                      latLim=None,
                      lonLim=None,
                      lenYr=None):
    """
    Plot subplots of annual mean lines

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-08-10
    """

    # Set defaults
    if latLim is None:
        latLim = np.array([-90, 90])
    if lonLim is None:
        lonLim = np.array([0, 360])
    if lenYr is None:
        if ncDataList[0].data.shape[0] > 1000:
            lenYr = 365
        else:
            lenYr = 12

    # Define axis tick positions
    yTicks = np.round(regularconts(latLim, maxConts=8)).astype(int)

    # Create figure for plotting
    hf = plt.figure()
    hf.set_size_inches(12, 5, forward=True)

    # Prep for figure plotting
    nRow = 1
    nCol = ncDataList[0].data.shape[0]/lenYr + 1

    # Create subplot distribution
    gs = gridspec.GridSpec(nRow, nCol)

    # Preallocate array to hold line handles and labels
    hl = range(len(ncDataList))
    lName = range(len(ncDataList))

    # Loop through each year and plot separately
    for jyr in range(ncDataList[0].data.shape[0]/lenYr):

        for jVar in range(len(ncDataList)):

            # Create subplot for control plot
            ax = plt.subplot(gs[0, jyr])

            # Set time indices over which to take mean
            tInd = np.arange(lenYr) + lenYr*jyr
            if ncDataList[jVar].data.shape[0] % lenYr != 0:
                if jyr == 0:
                    tInd = tInd[:-1]
                else:
                    tInd = tInd - 1

            # Plot regional zonal mean
            hl[jVar], = plt.plot(ncDataList[jVar]
                                 .calcregzonmean(latLim=latLim,
                                                 lonLim=lonLim)[tInd, :]
                                 .mean(axis=0),
                                 np.sin(np.deg2rad(ncDataList[jVar].lat)),
                                 linewidth=2
                                 )
            plt.hold

            # Pull variable source for label
            lName[jVar] = ncDataList[jVar].srcid

        # Add text to identify the year
        ax.text(0.95, 0.02, 'Yr. ' + str(jyr+1),
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='bottom'
                )

        # Set axes ticks outward
        ax.tick_params(axis='x', direction='out')
        ax.tick_params(axis='y', direction='out')

        # Add grid to plot
        plt.grid(b=True, which='major', color=[0.8, 0.8, 0.8],
                 linestyle='-')

        # Set plot y limts
        ax.set_ylim(np.sin(np.deg2rad(latLim)))
        ax.set_yticks(np.sin(np.deg2rad(yTicks)))
        ax.set_yticklabels(yTicks)

    # Plot legend
    plt.legend(hl, lName,
               bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=1.)

    # Add text showing lat/lon limits
    ax.text(1.1, 1,
            'lat = [' + '{:3.0f}'.format(latLim[0]) + '$^\circ$, ' +
            '{:3.0f}'.format(latLim[1]) + '$^\circ$]\n' +
            'lon = [' + '{:3.0f}'.format(lonLim[0]) + '$^\circ$, ' +
            '{:3.0f}'.format(lonLim[1]) + '$^\circ$]',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top'
            )

    # Unify plot x limits and x ticks
    maxXlim = getmaxxlim()
    for jp in range(len(plt.gcf().get_axes())):
        plt.subplot(gs[0, jp]).set_xlim(maxXlim)
        plt.xticks(regularconts(maxXlim, maxConts=3))

    # Label plot
    ha = hf.get_axes()
    ha[0].set_ylabel('Latitude ($^\circ$N)')
    ha[2].set_xlabel(ncDataList[0].name + ' (' + ncDataList[0].units + ')')

    # Switch to tight layout
    plt.tight_layout()

    # Show figure
    plt.show()


def plotzonslice(merMean, lon, z, plot_t,
                 annotate_flag=True,
                 compcont=None,
                 cbar_flag=False,
                 cMap=None,
                 conts=None,
                 ctrlLineConts_flag=False,
                 ctrlLineConts=None,
                 ctrlValues=None,
                 dataId=None,
                 extend='both',
                 grid_flag=False,
                 lineConts=None,
                 lonLim=np.array([-90, 90]),
                 maxConts=10,
                 sym_flag=False,
                 tickDir='out',
                 varName=None,
                 varUnits=None,
                 verbose_flag=False,
                 zLim=None
                 ):
    """
    Contours a zonal slice of a field over a given region and time period.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2017-07-10

    Args:
        merMean - time x depth x lon numpy array of meridional means for
            contouring
        lon - longitudes for dataIn
        z - depths for dataIn
        plot_t - times to average over for plotting
        annotate_flag - true to anotate the plot with variable and id info
        compcont - contour to plot bolder than others for emphasis
        cbar_flag - true to plot colorbar
        cMap - colormap to use when plotting
        conts - contour values to use when plotting
        dataId - identifier for data to be printed in top right corner of plot
        extend - extend colorbar beyond color limits?
            options = ['neither', 'min', 'max', 'both']
        grid_flag - True to plot grid atop contours for reference
        lineConts - contour values for lined contours
        lonLim - longitude limits for plot area
        maxConts - maximum number of contours to use
        sym_flag - True to make color limits symmetric about 0
        tickDir - direction ticks point from axis ('in' or 'out')
        varName - name of variable being plotted
        varUnits - units of variable being plotted
        verbose_flag - True to ouput status updates to command line

    Returns:
        cset1.levels- values for contour levels

    Raises:
        none

    Notes:
        n/a
    """
    # Set x axis info
    xVec = lon
    if (lonLim == [0, 360]).all():
        xTicks = np.arange(0, 360.1, 120)
    else:
        xTicks = (np.linspace(lonLim[0], lonLim[1], num=11).round()
                  ).astype(int)

    # Get axis handle
    ha = plt.gca()

    # Plot contour plot
    if varName is None:
        # Plot data
        plt.contourf(xVec, z, merMean[plot_t, :, :].mean(axis=0),
                     cmap=cMap)

    else:
        if varName in ['UVEL', 'VVEL', 'WVEL']:
            cMap = 'RdBu_r'
            sym_flag = True
        elif varName in ['T']:
            cMap = 'Reds'
            sym_flag = False

        # Set contour values
        if conts is None:
            conts = regularconts(merMean,
                                 maxConts=maxConts,
                                 sym_flag=sym_flag
                                 )

        # Set colormap
        if cMap is None:
            if sym_flag:
                cMap = 'RdBu_r'
            else:
                cMap = 'Greys'

        # Plot filled contours
        cset1 = plt.contourf(xVec, z, merMean[plot_t, :, :].mean(axis=0),
                             conts,
                             cmap=cMap,
                             extend=extend)

        # Plot contour edges
        if ctrlLineConts_flag:
            plt.contour(xVec, z, ctrlValues[plot_t, :, :].mean(axis=0),
                        ctrlLineConts,
                        colors='k',
                        hold='on'
                        )
        else:
            if lineConts is None:
                if verbose_flag:
                    print('Creating lineConts')
                lineConts = cset1.levels[:]
            plt.contour(xVec, z, merMean[plot_t, :, :].mean(axis=0),
                        lineConts,  # cset1.levels,
                        colors='k',
                        hold='on'
                        )

    # Plot bold contour if requested
    if compcont is not None:
        if ctrlLineConts_flag:
            plt.contour(xVec, z, ctrlValues[plot_t, :, :].mean(axis=0),
                        np.array([compcont]),
                        colors='k',
                        linewidths=2,
                        hold='on'
                        )
        else:
            plt.contour(xVec, z, merMean[plot_t, :, :].mean(axis=0),
                        np.array([compcont]),
                        colors='k',
                        linewidths=2,
                        hold='on'
                        )

    # Add colorbar to figure
    if cbar_flag:
        plt.colorbar(cset1)

    # Format x axis
    ax = plt.gca()
    if lonLim is None:
        plt.xlim([np.min(xVec), np.max(xVec)])
    else:
        plt.xlim(lonLim)
    plt.xlabel('Longitude ($^\circ$N)')
    ax.tick_params(axis='x', direction=tickDir)
    ha.set_xticks(xTicks.astype(float))
    ha.set_xticklabels((xTicks))

    # Format y axis
    if zLim is None:
        plt.ylim([np.min(z), np.max(z)])
    else:
        plt.ylim(zLim)
    plt.ylabel('Depth (m)')
    ax.tick_params(axis='y', direction=tickDir)
    ha.invert_yaxis()

    # Add grid lines for readability
    if grid_flag:
        plt.grid(b=True, which='major', color=[0.5, 0.5, 0.5], linestyle='--')

    # Add annotations to plot
    if annotate_flag:
        # Add title of variable wth units
        varString = ""
        if varName:
            varString = varString + varName
        if varUnits:
            varString = varString + " (" + mwfn.getstandardunitstring(varUnits) + ")"
        plt.annotate(varString,
                     xy=(0, 1),
                     xycoords='axes fraction',
                     horizontalalignment='left',
                     verticalalignment='bottom'
                     )

        # Add data source/ID
        if ~(dataId is None):
            plt.annotate(dataId,
                         xy=(1, 1),
                         xycoords='axes fraction',
                         horizontalalignment='right',
                         verticalalignment='bottom'
                         )

    # Make plot appear
    plt.show()

    # Return contour levels
    return cset1


def regularconts(dataIn, maxConts=10, sym_flag=False):
    """
    Return contours at integer values
    Needs tweaking to handle non-interger contouring

    maxConts - maximum number of contours to return
    sym_flag - True to make contours symmetric about 0
    2015-08-10
    """

    # Determine data extrema
    contUpperLim = np.ceil(np.nanmax(dataIn))
    contLowerLim = np.floor(np.nanmin(dataIn))

    # If requested, make limits symmetric about 0
    if sym_flag:
        contUpperLim = np.abs([contUpperLim, contLowerLim]).max()
        contLowerLim = -contUpperLim

    # Compute contour interval to give reasonable values for contours
    dCont = (contUpperLim - contLowerLim)/(maxConts + 1)
    if dCont > 1:
        dCont = np.ceil(dCont)
    elif dCont < 1:
        dContExp = int(np.floor(log10(dCont)))
        dContCoeff = dCont/(10**dContExp)
        dConts = np.array([1, 2, 2.5, 5, 10])
        dContCoeff = find_nearest(dConts, dContCoeff)
        dCont = dContCoeff*(10**dContExp)

    # Determine minum contour value
    if sym_flag:
        conts = np.arange(0, np.ceil(contUpperLim+0.0001), dCont)
        conts = np.append(-np.fliplr([conts])[0][:-1], conts)
    else:
        if (dataIn[:] > 0).all() or (dataIn < 0).all():
            contMin = np.floor(np.nanmin(dataIn))
            contMax = np.nanmax(dataIn)
        else:
            contMin = np.nanmin(dCont*np.floor(dataIn/dCont))
            contMax = np.nanmax(dCont*np.ceil(dataIn/dCont))
        # else:
        #    contMin = np.nanmax(dCont)

        # Determine value of contours
        conts = np.arange(contMin, contMax+dCont, dCont)

    # Return contour values
    return conts


def savefig(saveFile,
            imageFormat='png',
            dpi=100,
            figHandle=None,
            hf=None,
            mkdir_flag=False,
            noclobber=False,
            shape=None,
            units='inches'
            ):
    """
   Generic saving function. Checks if directory exists, and can create it if
       need be.

    Author:
        Matthew Woelfle (mdwoelfle@gmail.com)

    Version Date:
        2015-12-04

    Args:
        saveFile - full path (minus file suffix) of file to be saved
        ...begin optional imputs...
        imageFormat - format of image to be saved
        dpi - dots per linear inch for raster images (300 for "prod. quality")
        figHandle - handle for figure frame
        hf - same as figHandle
        mkdir_flag - True to make all necessary subdirectories to save file to
            the requested location
        noclobber - if True, will not save over existing files
        shape - size of saved image [x, y]
        units - units of shape ('inches', 'cm', 'pixels')

    Returns:
        n/a

    Raises:
        none

    Notes:
        n/a
    """
    """
    Generic functon to save figures to files
    DO NOT include file suffix in saveFile. This will be automatically appended
    2015-07-02
    """
    # Define dictionary of good file formats and suffixes to append
    goodFormats = {'png': '.png',
                   'pdf': '.pdf',
                   'eps': '.eps',
                   'ps': '.ps'
                   }

    # Get handle of figure to save if not provided by user
    if (hf is not None):
        figHandle = hf
    elif (figHandle is None):
        figHandle = plt.gcf()

    # Get shape of figure if not provided by user
    if shape is None:
        shape = figHandle.get_size_inches()

    # Convert shape to inches
    if units == 'cm':
        shape = shape/2.54
    elif units == 'pixels':
        shape = shape/dpi

    # Check if file output format is acceptable
    if imageFormat not in goodFormats.keys():
        print("ERROR: Unrecognized file format " + imageFormat + ". Please" +
              " select one of the following: ")
        print(goodFormats)
        return

    # Determine if save directory exists[, create it if it doesn't]
    saveDir = saveFile[::-1].split(os.sep, 1)[1][::-1]
    # Following may have a race condition (?) error at times, we'll see...
    if not os.path.isdir(saveDir):
        if mkdir_flag:
            # make directory and all necessary subdirectories
            os.path.makedirs(saveDir)
        else:
            print('ERROR: Intended save directory does not exist.')

    # Append appropriate file ending to saveFile
    saveFile = saveFile + goodFormats[imageFormat]

    # Check for existing file prior to saving
    if noclobber:
        if os.path.exists(saveFile):
            print('ERROR: File exists. Use \'noclobber=True\' to overwrite')

    # Set figure size
    figHandle.set_size_inches(shape[0], shape[1], forward=True)

    # Save figure
    figHandle.savefig(saveFile, dpi=dpi)


def stampdate(hf=None, x=0.95, y=0.05):
    """
    Stamp current date on a figure

    hf - figure onto which to stamp current date
    x - axis relative x location for bottom right of date stamp
    y - axis relative y location for bottom right of date stamp
    2015-11-23
    """
    if hf is None:
        hf = plt.gcf()
    datenow = ('{:04.0f}'.format(datetime.now().year) +
               '{:02.0f}'.format(datetime.now().month) +
               '{:02.0f}'.format(datetime.now().day))
    hf.text(x, y, datenow, ha='right', va='baseline')
