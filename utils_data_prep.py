#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:07:11 2021

@author: Github: jonaszubindu
"""

import os

from astropy import constants as c
from copy import deepcopy

from irisreader import observation
from irisreader.utils.date import to_epoch

from irispy.utils import get_interpolated_effective_area

import astropy.units as u
from astropy.time import Time
from sunpy.time import parse_time
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import astroscrappy
from sunpy.coordinates.sun import earth_distance

from datetime import datetime
from warnings import warn
import h5py
import gc
from astropy import constants as c

import utils_features as uf

DN2PHOT = {'NUV':18/(u.pixel), 'FUV': 4/(u.pixel)} # for calibration part


# Those are some of the values that I used to play around with the Mg 2832 data:


#######################################################################################################################

"""Data preperation helper functions"""

def clean_pre_interpol(X, threshold, mode='zeros'):
    '''
    X must have shape (some value, n_breaks)
    clean profiles which meet the following criterion:
        - Negative values -> condition = profile with any value <= -100
        - Noisy -> condition = profile with maximum <= 10 before normalizing
        - Overexposure -> condition = more than 3 points == max


    input  - X: matrix (example, feature)
           - modes:
               1) 'zeros' -> replaces bad profiles with a vector of zeros
               add other modes
    '''

    # Clean profiles with negative
    neg = np.min(X, axis=-1)
    #Clean profiles with too little signal
    small = np.max(X, axis=-1)
    # Clean overexposed profiles
    maxx = np.max(X, axis=-1).reshape(small.shape + (1,))
    diff = (X-maxx)
    w = np.where(diff == 0)
    dump = np.zeros(X.shape)
    dump[w] = 1
    s =  np.sum(dump, axis = -1)

    if mode == 'zeros':

        X[np.where(np.all(X  < threshold, axis=-1))] = np.full((1, X.shape[-1]), fill_value = 0)

        X[np.logical_or(neg <= -100, s >=3)] = np.full((1, X.shape[-1]), fill_value = 0)

    return X


def clean_aft_interpol(X, mode='zeros'):

    """
    This cleaning process is used to reject spectra with cosmic rays and abnormal peak ratio. This needs to be updated when other
    lines will be processed.

    input  - X: matrix (example, feature)
           - modes:
               1) 'ones' -> replaces bad profiles with a vector of ones
               2) 'nans' -> replaces bad profiles with a vector of nan's

    """
    try:
        X = X/X.unit
    except Exception:
        pass

    # Reject spectra outside of peak ratio [1:1, 2:1]
    df_features = extract_features_MgIIk(X, save_path=None)
    inds = df_features.index[(df_features['kh_ratio'] < .8) | (df_features['kh_ratio'] > 2) ] #.7 to efficiently remove noisy peaks, remove nan trip emission (only few) | (np.isnan(df_features['trip_emiss']))

    # Find maximum and compare to the peak window to check if there are spikes outside of the peak window
    maxx_ind = [np.argmax(prof) for prof in X]
    inds_m = np.array([n for n, max_ind in enumerate(maxx_ind) if (max_ind not in np.arange(kl,kr+1,1) and max_ind not in
                                                                   np.arange(hl,hr+1,1))])
    #delete spectra with extreamly high pseudocontinium probably from limb obs
    high_continuum = np.squeeze(np.argwhere(np.mean(X[:,460:500], axis=-1) > .6))


    if mode == 'zeros':

        if high_continuum.size != 0:
            print("cleaned by high continuum: ", high_continuum.size)
            X[high_continuum] = np.full((1, X.shape[-1]), fill_value = 0)

        if inds_m.size != 0:
            print("cleaned by cosmic ray in continuum: ", inds_m.size)
            X[inds_m] = np.full((1, X.shape[-1]), fill_value = 0)


        if inds.size != 0:
            print("cleaned by peak ratio outside of theory: ", inds.size)
            X[inds] = np.full((1, X.shape[-1]), fill_value = 0)

    if mode == 'del':

        del_inds = np.hstack([low_continuum, inds_m, inds])

        X = np.delete(X, del_inds.astype(int), axis=0)

    return X


def normalize( X ):
    '''
    normalize each profile by its maximum
    '''
    maxx_u = np.max( X, axis=-1 )
    maxx_u = maxx_u.reshape(maxx_u.shape + (1,))

    try:
        X = X/X.unit
    except Exception:
        pass
    maxx = np.max( X, axis=-1 ).reshape(maxx_u.shape)

    X[np.where(np.all(X == 0, axis=-1))] = np.full((1, X.shape[-1]), fill_value = 0)
    X[np.where(np.any(X != 0, axis=-1))] = X[np.where(np.any(X != 0, axis=-1))]/maxx[np.where(np.any(X != 0, axis=-1))]


    return X, maxx_u

def minmaxnorm(X, min_, max_): # apply after standardization on all data
    """mapping the intensity levels onto a [0,1] in a log shape"""
    return (X-min_)/(max_-min_)


def spectra_quick_look(spectra, lambda_min=None, lambda_max=None, n_breaks=None, dim=5, ind012 = None):
    '''
    plot a random sample of spectra
    '''
    if lambda_min:
        lambda_units = lambda_min + np.arange(0,n_breaks)*(lambda_max-lambda_min)/n_breaks
    else:
        lambda_units = np.arange(0,spectra.shape[-1])

    if spectra.shape[0]<=0 or spectra.shape[1]<=0:
        warn(f'spectra can not be rendered because shapes are {spectra.shape[0]} and {spectra.shape[1]}')
        return None
    else:
        if len(spectra.shape)>4:
            spectra = np.vstack(np.vstack(np.vstack(spectra)))
        elif len(spectra.shape)>3:
            spectra = np.vstack(np.vstack(spectra))
        elif len(spectra.shape)>2:
            spectra = np.vstack(spectra)

        if ind012:
            spectra = spectra[ind012, :]
        else:
            ind_select = np.random.randint(0, spectra.shape[0], size=dim*dim) # low >= high outputted
            spectra = spectra[ind_select,:]

        fig = plt.figure(figsize=(36,16))
        gs = fig.add_gridspec(dim, dim, wspace=0, hspace=0)
        for i in range(dim):
            for j in range(dim):
                ind = (i*dim)+j
                ax = fig.add_subplot(gs[i, j])
                ax.grid(False)
                try:
                    plt.plot(lambda_units, spectra[ind], linewidth=1, linestyle='-', color='snow')
                except Exception:
                    plt.plot(spectra[ind], linewidth=1, linestyle='-', color='snow')
                ax.set_facecolor('xkcd:black')
                plt.xticks(fontsize=16, color='red')
                plt.yticks(fontsize=16, color='red')

        plt.show()

    return ind012

##########################################################################################



def interpolate_spectra(raster, lambda_min, lambda_max, field, line, n_breaks, threshold, calib = True):
    """
    This function interpolates all spectra onto the same predefined wavelength grid. Before interpolation, cosmic rays are
    removed as good as possible with the cosmic ray removal algorithm astroscrappy.detect_cosmics. Afterwards, the spectra are
    interpolated and can also be calibrated and normalized. The default is to calibrate and normalize the spectra. Print statements let
    the user track how many spectra are cleaned out in each step. This can also be tracked in a dictionary if desired, then the user should uncomment the
    lines with spectra_stats_single_obs.
    """


    hdrs = raster.headers

    lambda_min_mes = hdrs[0]['WAVEMIN'] # get the minimum measured wavelength
    lambda_max_mes = hdrs[0]['WAVEMAX'] # get the maximum measured wavelength
    delta_lambda = hdrs[0]['CDELT1'] # get the resolution in Å
    num_points = hdrs[0]['NAXIS1'] # get the number of sample points for the measured wavelength points
    start_obs = Time(hdrs[0]['STARTOBS']) # start date and time of the observation
    pix_size = hdrs[0]['SUMSPAT']*0.166*u.arcsec # for calibration
    slit_width = 0.33*u.arcsec # for calibration

    sol_rad = 696340*u.km # for calibration

    d = delta_lambda*u.Angstrom/u.pixel # wavelength bin width, for calibration
    dist = earth_distance(start_obs) # for calibration
    omega = slit_width*pix_size*(sol_rad/((sol_rad/dist.to(u.km))*u.radian).to(u.arcsec))**2/(1.496E+8*u.km)**2

    n_min = int(np.floor((lambda_min - lambda_min_mes)/delta_lambda))-1
    n_max = int(np.ceil((lambda_max - lambda_min_mes)/delta_lambda))+1 # to make sure the transformation to the desired grid is
                                                                       # possible
    lambda_units = lambda_min_mes + np.arange(0, num_points)*delta_lambda
    lambda_units = lambda_units[n_min:n_max] # set the wavelength grid to the desired wavelength grid for all observations

    obs_wavelength = np.linspace( lambda_min, lambda_max, num=n_breaks ) # set up the desired wavelength grid to interpolate
                                                                         # spectra onto.

    effA = get_interpolated_effective_area(start_obs, response_version=6, detector_type=field,
                                                   obs_wavelength=obs_wavelength*u.Angstrom) # get effective area of IRIS for
                                                                                             # calibration. If the response
                                                                                             # version 6 does not work, try 5 or
                                                                                             # 4.
    effA = effA.to(u.cm*u.cm)

    if field == "NUV":
        dn2phot = DN2PHOT['NUV']
        exptime = 'EXPTIMEN'
    else:
        dn2phot = DN2PHOT['FUV']
        exptime = 'EXPTIMEF'

    exp_times = np.asarray([hdrs[n][exptime] for n in range(len(hdrs))])

    raster_aft_exp = raster[:,:,:]/exp_times.reshape(exp_times.shape[0], 1, 1) # devide by exposure time

    raster_cut = raster_aft_exp[:,:,n_min:n_max] # cut to desired wavelength window

    print("Original number of spectra : ",
      raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape)

#     spectra_stats_single_obs['Original'] = raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape

    del raster


#######################################################################################################################

    #automatic cosmic ray removal, not thoroughly tested so far,
    #readnoise was determined with noisy spectra standard deviation.
    #sigclip and sigfrac was determined with trial and error on one observation in Si IV, detect cosmic is optimized for parallel computing.
    for n in range(raster_cut.shape[0]):

        crmask, _ = astroscrappy.detect_cosmics(raster_cut[n,:,:], sigclip=4.5, sigfrac=0.1, objlim=2, readnoise=2)
        raster_cut[(n,) + np.where(np.any(crmask, axis=-1))] = np.full((1, raster_cut.shape[-1]), fill_value = 0)

    print("remaining spectra after removing cosmic rays : ",
      raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape)

# spectra_stats_single_obs['remaining cosmic'] = raster_cut[np.where(np.any(raster_cut!=0, axis=-1))].shape


#######################################################################################################################

    interpolated_slice = clean_pre_interpol(raster_cut, threshold=threshold, mode='zeros')
    print("remaining spectra after cleaning with pre interpol : ",
          interpolated_slice[np.where(np.any(interpolated_slice!=0, axis=-1))].shape)

#     spectra_stats_single_obs['cleaned pre interpol'] = interpolated_slice[np.where(np.any(interpolated_slice!=0, axis=-1))].shape

    interpol_f_cut = interp1d(lambda_units, interpolated_slice, kind="linear", axis = -1 ) # interpolate the data

    obs_wavelength = np.linspace( lambda_min, lambda_max, num=n_breaks ) # set up obs_wavelength newly w/o units

    interpolated_slice = interpol_f_cut( obs_wavelength ) # interpolate onto grid
    interpolated_slice[np.where(np.any(np.isnan(interpolated_slice), axis= -1))] = np.full((1, interpolated_slice.shape[-1]), fill_value = 0) # set nan's to 0

    if calib:
        # calibrate spectra here
        interpolated_slice_calibrated = ((interpolated_slice/u.second)/((obs_wavelength*u.Angstrom).to(u.cm)*effA*d*omega))*dn2phot*(c.h).to(u.erg*u.second)*(c.c).to(u.cm/u.second)/u.steradian
    else:
        interpolated_slice_calibrated = deepcopy(interpolated_slice)

    interpolated_slice_calibrated, norm_vals = normalize(interpolated_slice_calibrated) # normalize spectra (if needed)

    interpolated_slice_calibrated = interpolated_slice_calibrated.reshape(raster_cut.shape[:-1] + (n_breaks,))
    norm_vals = norm_vals.reshape(raster_cut.shape[:-1] + (1,))


    return interpolated_slice_calibrated, norm_vals#, spectra_stats_single_obs

#######################################################################################################################

# From here on comes code for my customized observation classes that were used in the study: Which is the best spectral line for
# solar flare prediction.

def transform_arrays(times, image_arr=None, norm_vals_arr=None, num_of_raster_pos=0, forward=True):

    """

    DON'T LOOK AT THIS FUNCTION EXCEPT YOU WANT A HEADACHE but it might come in handy at a later stage.

    transforms between image_arrays in shape (n,t,y,lambda) -> (n*t,y,lambda)
    and time arrays from shape (n,t) -> (n*t,)

    or

    the reverse operation

    """

    if not np.any(image_arr) and forward:

        image_arr = np.zeros(times.shape + (1,)+ (1,))
        norm_vals_arr = np.zeros(times.shape + (1,)+ (1,))

    elif not np.any(image_arr) and not forward:
        image_arr = np.zeros((times.shape[1],) + (1,)+ (1,))

    if forward:

#     if num_of_raster_pos == 1:
#         image_arr_r = image_arr
#         times_r = times
#         norm_vals_r = norm_vals_arr
#     else:
        image_arr_r = image_arr.reshape(image_arr.shape[0]*image_arr.shape[1], image_arr.shape[2], image_arr.shape[3],
                                        order='F')
        times_r = times.reshape(times.shape[0]*times.shape[1], 1, order='F')
        norm_vals_r = norm_vals_arr.reshape(norm_vals_arr.shape[0]*norm_vals_arr.shape[1], norm_vals_arr.shape[2], 1, order='F')

        if np.any(times_r != sorted(times_r)):
            print(times_r[np.where(times_r != sorted(times_r))])
            if num_of_raster_pos == 1:
                sorted_inds = np.squeeze(np.argsort(times_r,axis=0))
#                 print(sorted_inds)
                times_r = times_r[sorted_inds]
                image_arr_r = image_arr_r[sorted_inds,:,:]
                norm_vals_r = norm_vals_r[sorted_inds,:,:]

            else:
                raise Warning('could not reshape times, times are not sequential and number of raster steps is: '
                              , num_of_raster_pos)

        complete_raster_step = times.shape[1]
        raster_pos = np.hstack([np.arange(0,num_of_raster_pos) for n in range(complete_raster_step)])
        times_r = np.vstack([raster_pos,times_r.T])

    else: # reverse

        if num_of_raster_pos > image_arr.shape[0]:

            start_ind = None
            stop_ind = None
            try:
                start_ind = np.where(times[0]==0)[0][0]
            except Exception:
                stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

            image_arr_new = np.zeros([num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]])
            times_new = np.zeros([num_of_raster_pos])
            norm_vals_new = np.zeros([num_of_raster_pos, image_arr.shape[1], 1])

            if start_ind:
                image_arr_new[start_ind:] = image_arr
            elif stop_ind:
                image_arr_new[:stop_ind] = image_arr
            else:
                image_arr_new[int(times[0][0]):int(times[0][-1])+1] = image_arr


            print(f"number of raster positions > timesteps, {num_of_raster_pos} > {image_arr.shape[0]}")

            times_origin = deepcopy(times)

            image_arr = image_arr_new
            times = times_new
            norm_vals_arr = norm_vals_new

            image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

            times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

            norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

        elif num_of_raster_pos == 1:

            image_arr_r = image_arr.reshape(1,image_arr.shape[0],image_arr.shape[1],image_arr.shape[2])
            times_r = times[1].reshape(1,times.shape[1])
            norm_vals_r = norm_vals_arr.reshape(1,norm_vals_arr.shape[0],norm_vals_arr.shape[1],1)

        else:


            start_ind = None
            stop_ind = None

            start_ind = np.where(times[0]==0)[0][0]
            stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

            if stop_ind < start_ind:

                try:
                    start_ind = np.where(times[0]==0)[0][0]
                except Exception:
                    stop_ind = np.where(times[0]==(num_of_raster_pos-1))[0][-1]

                image_arr_new = np.zeros([num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]])
                times_new = np.zeros([num_of_raster_pos])
                norm_vals_new = np.zeros([num_of_raster_pos, image_arr.shape[1], 1])

                if start_ind:
                    image_arr_new[start_ind:] = image_arr
                elif stop_ind:
                    image_arr_new[:stop_ind] = image_arr
                else:
                    image_arr_new[int(times[0][0]):int(times[0][-1])+1] = image_arr


                print(f"number of raster positions > timesteps, {num_of_raster_pos} > {image_arr.shape[0]}")

                times_origin = deepcopy(times)

                image_arr = image_arr_new
                times = times_new
                norm_vals_arr = norm_vals_new

                image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

                times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

                norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

            else:

                num_of_cycles = int(np.floor((stop_ind - start_ind)/(num_of_raster_pos-1)))


                if start_ind == stop_ind+1:
                    raise ValueError("cannot reshape, start_ind and stop_ind for symmetric reshaping is the same")

                image_arr_new = np.zeros([num_of_raster_pos*(num_of_cycles+2), image_arr.shape[1], image_arr.shape[2]])
                times_new = np.zeros([num_of_raster_pos*(num_of_cycles+2)])
                norm_vals_new = np.zeros([num_of_raster_pos*(num_of_cycles+2), norm_vals_arr.shape[1], 1])

                image_arr_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = image_arr
                times_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = times[1,:]
                norm_vals_new[int(times[0][0]):int(times.shape[1]+times[0][0])] = norm_vals_arr.reshape(norm_vals_arr.shape[0], norm_vals_arr.shape[1],1)

                times_origin = deepcopy(times)

                image_arr = image_arr_new
                times = times_new
                norm_vals_arr = norm_vals_new

                image_arr_r = image_arr.reshape(int(image_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, image_arr.shape[1], image_arr.shape[2]).transpose(1,0,2,3).reshape(num_of_raster_pos, int(image_arr.shape[0]/num_of_raster_pos), image_arr.shape[1], image_arr.shape[2])

                times_r = times.reshape(int(times.shape[0]/num_of_raster_pos), num_of_raster_pos).transpose(1,0).reshape(num_of_raster_pos, int(times.shape[0]/num_of_raster_pos))

                norm_vals_r = norm_vals_arr.reshape(int(norm_vals_arr.shape[0]/num_of_raster_pos), num_of_raster_pos, norm_vals_arr.shape[1],1).transpose(1,0,2,3).reshape(num_of_raster_pos, int(norm_vals_arr.shape[0]/num_of_raster_pos), norm_vals_arr.shape[1],1)

    return times_r, image_arr_r, norm_vals_r




##########################################################################################


def clean_SAA_cls(obs_cls):
        """
        Cleans the parts of observations when IRIS crossed the southern atlantic anomaly SAA and puts them everywhere to zero.

        input:

        im_arr_slit: array containing the spectra ordered either by global time steps or just one slit.

        """
        path_to_SAA = '/sml/jannaschk/Other_data/'
        df_SAA = pd.read_csv(f'{path_to_SAA}saa_results.csv')

        times = obs_cls.times_global[1,:]

        for start, end in zip(df_SAA['start'], df_SAA['end']):
            start = datetime.fromisoformat((start.split('.')[0]).split(' ')[-1])
            end = datetime.fromisoformat((end.split('.')[0]).split(' ')[-1])

            start_e = to_epoch(start)
            end_e = to_epoch(end)

            if ((start_e > times[0]) and (start_e < times[-1])) or ((end_e > times[0]) and (end_e < times[-1])):

                diff = times - start_e
                try:
                    start_ind = np.argmin(np.abs(diff[diff<0]))
                except Exception:
                    start_ind = 0


                diff = times - end_e

                try:
                    end_ind = np.argmin(np.abs(diff[diff>0])) + np.argmin(np.abs(diff[diff<0])) + 1 # only take steps outside of SAA

                except Exception:
                    end_ind = len(times)

                obs_cls.im_arr_global[start_ind:end_ind, :, :] = np.full((obs_cls.im_arr_global[start_ind:end_ind, :, :].shape[0],
                                                                       obs_cls.im_arr_global.shape[1],
                                                                       obs_cls.im_arr_global.shape[2]),
                                                                       fill_value = 0)
                # Times remains unchanged to keep the structure of the arrays intact.
        print("remaining spectra after removing SAA : ",
              obs_cls.im_arr_global[np.where(np.any(obs_cls.im_arr_global!=0, axis=-1))].shape)

        obs_cls.spectra_stats_single_obs['cleaned SAA'] = obs_cls.im_arr_global[np.where(np.any(obs_cls.im_arr_global!=0, axis=-1))].shape


        try:
            delattr(obs_cls, 'times')
            delattr(obs_cls, 'im_arr')
            delattr(obs_cls, 'norm_vals')
        except Exception:
            pass





class Obs_raw_data:

    """

    Class structure to store all the important information for later training and testing of prediction models, train and clean
    with VAE's or overplot SJI images.

        Parameters:
        -------------------
        obs_id : string
            IRIS observation ID
        num_of_raster_pos : int
            Number of raster positions (not times steps)
        times_global : numpy ndarray
            first row contains raster position, second row contains time in unix
        im_arr_global : numpy ndarray
            contains all spectra like (T,y,lambda)
        norm_vals_global : numpy ndarray
            contains the normalization values for each spectrum like (T,y,1)
        n_breaks : int
            interpolation points
        lambda_min : int or float
            lower wavelength limit
        lambda_max : int or float
            upper wavelength limit
        field : string
            FUV or NUV
        line : string
            spectral line
        threshold : int
            lower limit in DN/s at which spectra were cleaned
        hdrs : pandas DataFrame
            containing all headers from raster headers

    class methods :

        __init__ : initializes a new instance of Obs_raw_data

        time_clipping : clips the observation in time: times_global, im_arr_global, norm_vals_global

            start : datetime
            end : datetime

        save_arr : saves the Obs_raw_data instance according to a specific frame, adjust path in save_arr.

            filename : filename to store the instance as
            line : spectralline
            typ : type of observation : QS, SS, AR, PF

    global methods: check each method for necessary args and kwargs

        clean_SAA_cls(obs_cls) : cleans the given instance for SAA by setting SAA parts to 0

        transform arrays : transforms array between (n,t,y,lambda) <-> (T,y,lambda)
                           CAUTION : timeclipping destroys the equivalence between the two arrays. The function automatically
                           accounts for that by using the first and last complete raster steps.

        spectra_quick_look : allows the user to have a peek at some random spectra

        load_obs_data : allows the user to load a stored Obs_raw_data instance. args/kwargs are the same as in save_arr. Adjust
                        path if necessary


    """

    def __init__(self, obs_id=None, raster=None, lambda_min=None, lambda_max=None, n_breaks=None, field=None, line=None,
                 threshold=None, load_dict=None):

        spectra_stats_single_obs = {}

        if load_dict:

            filename, line, typ = load_dict.values()
            try:
                with h5py.File(f'/sml/jannaschk/IRIS_30_obs_prepared/{line}/{typ}/{filename}/arrays.h5', 'r') as f:
                    im_arr_global = f["im_arr_global"][:,:,:]
                    times_global = f["times_global"][:,:]
                    norm_vals_global = f["norm_vals_global"][:,:,:]

                init_dict = np.load(f'/sml/jannaschk/IRIS_30_obs_prepared/{line}/{typ}/{filename}/dict.npz', allow_pickle=True)['arr_0'][()]

                for key, value in init_dict.items():

                    setattr(self, key, value)

                self.im_arr_global = im_arr_global[:,:,:]
                self.times_global = times_global[:,:]
                self.norm_vals_global = norm_vals_global[:,:]

            except Exception as exc:
                print(exc)


        else:

            self.obs_id = obs_id

            self.num_of_raster_pos = raster.n_raster_pos

            times = [raster.get_timestamps(n) for n in range(raster.n_raster_pos)]
            times = np.vstack(times)

            self.lambda_min = lambda_min
            self.lambda_max = lambda_max
            self.n_breaks = n_breaks
            self.field = field
            self.line = line
            self.threshold = threshold
            self.spectra_stats_single_obs = {}

            interpolated_image_clean_norm, norm_val_image = interpolate_spectra(raster, lambda_min, lambda_max, field, line, n_breaks, threshold, calib = True) #spectra_stats_single_obs

            # self.spectra_stats_single_obs = spectra_stats_single_obs

            self.hdrs = pd.DataFrame(list(raster.headers))


            # transforms image to global raster step
            times_global, _, _ = transform_arrays(times, num_of_raster_pos=raster.n_raster_pos, forward = True)

            self.times_global = times_global # contains raster position data and times for later processing.
            self.im_arr_global = interpolated_image_clean_norm
            self.norm_vals_global = norm_val_image


            #clean out edges
            self.im_arr_global[np.where((np.argmax(self.im_arr_global, axis=-1) > (self.im_arr_global.shape[-1]*0.95)) |
                                        (np.argmax(self.im_arr_global, axis=-1) < (self.im_arr_global.shape[-1]*0.05)))] = 0

            clean_SAA_cls(self)




    def time_clipping(self, start, end):

        """
        Clip image_array and time_array according to start datetime and end datetime, only works with global time steps

        """
        '''
        start_e = to_epoch(start)
        end_e = to_epoch(end)
    
        '''
        #my attempt to fix that, without using the to_epoch fct from irisreader:
        # convert datetime object to unix timestamp

        start_e = int(start.timestamp())
        end_e = int(end.timestamp())

        times = self.times_global[1,:]


        if not (((start_e > times[0]) and (start_e < times[-1])) or ((end_e > times[0]) and (end_e < times[-1]))):
            start_t = parse_time(self.times_global[1,0], format='unix').to_datetime()
            end_t = parse_time(self.times_global[1,-1], format='unix').to_datetime()
            raise Warning(f'in {self.obs_id}: start {start} or end {end} is outside of times: {start_t}, {end_t}')

        start = start_e
        end = end_e

        diff = times - start

        try:
            start_ind = np.argmin(np.abs(diff[diff<0]))
        except ValueError:
            start_ind = 0

        diff = times - end

        end_ind = np.argmin(np.abs(diff[diff<0])) # only take last step before end

        self.times_global = self.times_global[:,start_ind:end_ind] # first dimension contains raster position information
        self.im_arr_global = self.im_arr_global[start_ind:end_ind, :, :] # first dimension contains raster position information
        self.norm_vals_global = self.norm_vals_global[start_ind:end_ind, :]

        # Quick visualization of the selected data ###############################################
#         try:
#             nprof = self.im_arr_global*self.norm_vals_global#.reshape(self.norm_vals_global.shape + (1,))
#         except ValueError:
#             nprof = self.im_arr_global*self.norm_vals_global.reshape(self.norm_vals_global.shape + (1,))


#         nprof = nprof.reshape(nprof.shape[0]*nprof.shape[1], self.n_breaks)

#         nprof = nprof[np.where(~np.all(((nprof == 0) | (nprof == 1)), axis=-1))]

        # self.spectra_stats_single_obs['remaining spectra after time clipping'] = self.im_arr_global[np.where(np.any(self.im_arr_global!=0, axis=-1))].shape

#         try:
#             spectra_quick_look(nprof, self.lambda_min, self.lambda_max, self.n_breaks)
#             spectra_quick_look(self.im_arr_global, self.lambda_min, self.lambda_max, self.n_breaks)

#             fig = plt.figure(figsize=(20,20))
#             ax = fig.add_subplot(projection='3d')
#             x = np.arange(nprof.shape[0])
#             y = np.arange(nprof.shape[1])
#             xs, ys = np.meshgrid(x, y)

#             ax.plot_surface(ys.T, xs.T, nprof, cmap=plt.cm.Blues, linewidth=1, alpha=0.9)#, vmin=-5, vmax=+10)

#             ax.axes.set_zlim3d(bottom=-1, top=2000000)
#             ax.view_init(10, -95)
#             plt.show()

#         except Exception as exc:
#             print(exc)
#             print('no data in this time-window')

        ##########################################################################################


    def save_arr(self, filename, line, typ, highest_directory = 'IRIS_30_obs_prepared'):

        print('save arr fct was called')

        filename = filename.split('.')[0]
        path=f'/sml/jannaschk/{highest_directory}/{line}/{typ}/{filename}/'
        
        if os.path.exists(path):
            pass
        else:
            # make directory for save file to keep it all together.
            os.makedirs(path)
        

        filename = filename.split('.')[0]

        save_dict = {'obs_id': self.obs_id,
                    'num_of_raster_pos': self.num_of_raster_pos,
                    'lambda_min': self.lambda_min,
                    'lambda_max': self.lambda_max,
                    'n_breaks': self.n_breaks,
                    'field' : self.field,
                    'line' : self.line,
                    'threshold' : self.threshold,
                    'hdrs' : self.hdrs,
                    'stats' : self.spectra_stats_single_obs
                   }

        np.savez(f'/sml/jannaschk/{highest_directory}/{line}/{typ}/{filename}/dict.npz', save_dict)



        try:

            f = h5py.File(f'/sml/jannaschk/{highest_directory}/{line}/{typ}/{filename}/arrays.h5', 'w')
            dataset = f.create_dataset("im_arr_global", data=self.im_arr_global)
            dataset = f.create_dataset("times_global", data=self.times_global)
            dataset = f.create_dataset("norm_vals_global", data=self.norm_vals_global)
            f.close()
            del dataset

        except Exception as exc:

            print(exc)
            f.close()


##########################################################################################

def load_obs_data(highest_directory, filename, line, typ, only_im_arr = False):

    filename = filename.split('.')[0]

    if only_im_arr:

        try:

            f = h5py.File(f'{highest_directory}/{line}/{typ}/{filename}/array.h5', 'r')
            im_arr_global = f["im_arr_global"]
            f.close()
            return im_arr_global

        except Exception as exc:

            print(exc)
            f.close()

            return None

    else:

        load_dict = {
                     'filename' : filename,
                     'line' : line,
                     'typ' : typ
                    }

        return Obs_raw_data(load_dict=load_dict)













if __name__ == "__main__":

    # define parameter for interpolation grid and data cleaning
    MgIIk = {'lambda_min' : 2794,
         'lambda_max' : 2806,
         'n_breaks' : 960,
         'line' : "Mg II k",
         'field' : "NUV",
         'threshold' : 10
        }

    obs_ids = ['20220118_165527_4204700138', '20210908_161659_3603013104'] # place the observations you want to process here

    # Any preprocessing of IRIS obs

    spectra_stats = {}

    for obs_id in tqdm(['20150311_224758_3860107071']): # replace the list with the obs_ids you want to process
        gc.collect()
        year = obs_id[:4]
        month = obs_id[4:6]
        day = obs_id[6:8]
        pth = f'/iris/{year}/{month}/{day}/{obs_id}'
        obs = observation( pth, keep_null=True )

        #load raster with irispy
        raster_MgIIk = obs.raster("Mg II k")

        #prepare obs data with my pipeline
        obs_MgIIk = utils.Obs_raw_data(obs_id, raster_MgIIk, **MgIIk)

        obs.close()

        # save data as struct
        obs_MgIIk.save_arr(f'{obs_id}', 'MgIIk', 'PF_')

        # load data as struct
        obs_cls.load_obs_data(f'{obs_id}', 'MgIIk', 'PF_')


        #prepare wavelength grid for plotting
        obs_wavelength=np.linspace(2794,2806,960)

        #plot example spectra
        plt.plot(obs_wavelength, obs_cls.im_arr_global[120,200,:].T)
        plt.show()

        #prepare some data array to extract features
        im_arr_test = obs_cls.im_arr_global.reshape(-1,960)

        #extract features
        df_features = extract_features_MgIIk(im_arr_test, save_path=None)

        #plot Mg II k trip emission map (x is time, y is y pixel position for all raster steps. You will see a jumpy pattern.)
        trip_arr = df_features['trip_emiss'].values
        trip_map = trip_arr.reshape(obs_cls.im_arr_global.shape[0:2])

        plt.imshow(trip_map[0].T)
        plt.show()


        # transform from (n*t) to (n,t)
        times, im_arr, norm_vals = utils.transform_arrays(obs_cls.times_global, obs_cls.im_arr_global, obs_cls.norm_vals_global, obs_cls.num_of_raster_pos, forward=False)

        #extract features
        df_features = uf.extract_features_MgIIk(im_arr_test, save_path=None)

        #plot Mg II k trip emission map (x is time, y is y pixel position for a single raster step)
        trip_arr = df_features['trip_emiss'].values
        trip_map = trip_arr.reshape(im_arr.shape[0:3])

        plt.imshow(trip_map[0].T)
        plt.show()




