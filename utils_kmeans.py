from irisreader.utils.date import from_Tformat
from irisreader.utils.date import to_epoch
import matplotlib.pyplot as plt
import numpy as np
import datetime

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

from irisreader.data.mg2k_centroids import assign_mg2k_centroids, LAMBDA_MIN, LAMBDA_MAX
from irisreader.coalignment import find_closest_raster
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

import utils_data_prep as utils
from irisreader import observation
import gc
import tqdm
import os
import utils_data_prep
import utils_kmeans

# first part are my own methods
# it is indicated later on, which functions were taken from Brandon Panos' code.



def elbow_method(group_to_subsample, obs_id, weighed_centroids, max_n_subclusters = 5):
    '''
    this fct will do kmeans with many n_subclusters clusters, and make an elbow plot.
    This means it will plot kmeans inertia vs n_subclusters.
    '''


    # get the spectral data for that obs_id:
    directory = '/sml/jannaschk/IRIS_30_obs_prepared'
    obs_cls = utils_data_prep.load_obs_data(highest_directory = directory,
                                            filename = f'{obs_id}',
                                            line = 'MgIIk',
                                            typ = 'PF')
    Data = obs_cls.im_arr_global[:,:,:].reshape(-1,960)
    total_spectra_data_unweighed = utils_kmeans.filter_out_zero_spectra(Data)

    # find the spectra of only that group:
    total_spectra_data_weighed = weigh_the_triplet(total_spectra_data_unweighed, global_triple_weight_factor)
    labels = assign_mg2k_centroids(X = total_spectra_data_weighed, centroids = weighed_centroids)
    spectral_data_of_that_group = total_spectra_data_weighed[labels == group_to_subsample]

    # weigh the data for the triplet:
    spectral_data_of_that_group_weighed = weigh_the_triplet(spectral_data_of_that_group, global_triple_weight_factor)

    print('Done with all the stuff before kmeans.')

    list_of_inertias = []
    for n_subclusters in range(1, max_n_subclusters + 1):
        # run kmeans on those spectra of that group:
        _, _, subsampled_inertia, _ = mini_batch_k_means(X = spectral_data_of_that_group_weighed,
                                                n_clusters=n_subclusters,
                                                batch_size=1000,
                                                n_init=10, SOM = False, verbose=0)
        list_of_inertias.append(subsampled_inertia)
        print(f'calc for n_subclusters = {n_subclusters} done.')
    
    # plot the elbow plot:
    plt.plot(range(1, max_n_subclusters + 1), list_of_inertias, linestyle = '-', marker = 'd', color = 'darkorchid')
    plt.title(f'Elbow plot for the subsampling of group {group_to_subsample}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    return list_of_inertias

def find_closest_mediods(centroids, Data):
    '''
    This fct finds the closest mediods to the centroids in the data set.
    Ie the closest real spectra to the centroids.
    The Data are all spectra of one group, or of the whole obs. And then it gives you the closest spectra to the centroids.
    '''

    closest_mediods = []
    n_clusters = len(centroids)
    closest, _ = pairwise_distances_argmin_min(centroids, Data)
    for n in range(n_clusters):
        closest_mediods.append(Data[closest[n]])
    return closest_mediods

def find_closest_mediods_big_data(obs_list, centroids):
    '''
    this fct does the same as the fct find_closest_mediods, but for a list of obs_ids.
    It finds the closest mediods to the centroids in the data set.
    The fct exists because the data is too big to be loaded at once and compared to find the best mediods.
    So you do it incrementally, and update the closest mediods for each obs_id, if they need to be updated.
    '''

    # Find the paths:
    directory = 'IRIS_30_obs_prepared'
    not_found_list = []
    matching_file_paths = []
    for search_string in obs_list:
        string_was_found = False
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                #for file in files:
                    #print(f'{file}')
                    #print(f'{dir}')
                    if search_string in dir:
                        matching_file_paths.append(os.path.join(root, dir))
                        string_was_found = True
                        #print(matching_file_paths)
        if not string_was_found:
            not_found_list.append(search_string)
    if len(matching_file_paths) == 0:
        print('No matching files found')
        return None

    list_of_currently_closest_spectra = []
    # Loop over the obs_list:
    counting_index = 0
    for path in tqdm(matching_file_paths, desc = 'Finding closest mediods'):
        obs_cls = utils_data_prep.load_obs_data(highest_directory = directory,
                                                filename = f'{path[-26:]}',
                                                line = 'MgIIk',
                                                typ = f'{path[-29:-27]}')
        current_data = obs_cls.im_arr_global[:,:,:].reshape(-1,960)
        current_data = utils_kmeans.filter_out_zero_spectra(current_data)

        temporary_list = find_closest_mediods(centroids = centroids, Data = current_data)

        if counting_index == 0:
            list_of_currently_closest_spectra = temporary_list

        for k in range(len(centroids)):
            distance_current_k = np.linalg.norm(centroids[k] - temporary_list[k])
            best_current_distance_k = np.linalg.norm(centroids[k] - list_of_currently_closest_spectra[k])

            if distance_current_k < best_current_distance_k:
                list_of_currently_closest_spectra[k] = temporary_list[k]

        counting_index += 1



    list_of_mediods = list_of_currently_closest_spectra
    # the follwoing paths were found:
    print(f'the following obs_ids were found: {matching_file_paths}')
    print('')
    if len(not_found_list) > 0:
        print(f'The following obs_ids were not found: {not_found_list}')
        print('')

    return list_of_mediods


def filter_out_zero_spectra(spectral_data, also_return_thrown_out_indices = False , verbose = 0):
    '''
    after cleaning with Jonas data_prep_utils, there could still be null spectra left.
    We don't want a useless group found by the kmeans algo, so we exclude them.
    '''

    # Example spectral data: 20 spectra with 960 data points each
    #num_spectra = 20
    #num_data_points = 960
    #spectral_data = np.random.rand(num_spectra, num_data_points)  # Replace with your actual spectral data

    # Check if the maximum value of each spectrum is less than 0.5
    max_values = np.max(spectral_data, axis=1)
    indices_to_keep = max_values >= 0.5

    # Discard spectra where the maximum value is less than 0.5
    filtered_spectral_data = spectral_data[indices_to_keep]

    # If you also need to keep track of the indices of spectra that are discarded:
    discarded_indices = np.where(indices_to_keep == False)[0]
    if verbose == 1:
        print('Filtered the spectral data: No more zero spectra left.')

    if also_return_thrown_out_indices:
        return filtered_spectral_data, discarded_indices
    else:
        return filtered_spectral_data
    

def clean_dict_of_obs(obs_dict) -> None:

    '''
    if cleans, but cleans multiple obs one after another. can be used in a screen of a .py script
    See clean_obs_in_screen.py for this.
    '''

    for key, value in obs_dict.items():
        print(f"Key: {key}, Value: {value}")

        if key == 'QS':
            list_to_clean = value
            QS_list = value
            print(f'Cleaning the following QS observations: {list_to_clean}')
            clean_list_of_obs(list_to_clean, obs_type = key)
        elif key == 'AR':
            list_to_clean = value
            AR_list = value
            print(f'Cleaning the following AR observations: {list_to_clean}')
            clean_list_of_obs(list_to_clean, obs_type = key)
        elif key == 'PF':
            list_to_clean = value
            PF_list = value
            print(f'Cleaning the following PF observations: {list_to_clean}')
            clean_list_of_obs(list_to_clean, obs_type = key)
    
    cleaned_obs_list = QS_list + AR_list + PF_list

    print('Cleaned everything in the following list:')
    for obs_id in cleaned_obs_list:
        print(obs_id)

    return None


def clean_list_of_obs(list_to_clean, obs_type = 'QS', highest_directory = 'IRIS_30_obs_prepared') -> None :
    
    '''
    this fct is used in clean_dict_of_obs, where it is calle multiple times. It cleans multiple obs one after another.
    '''

    for obs_id in list_to_clean:

        print(f'Cleaning at the moment: {obs_id}', end='\n\n')

        gc.collect()
        year = obs_id[:4]
        month = obs_id[4:6]
        day = obs_id[6:8]
        pth = f'/sml/iris/{year}/{month}/{day}/{obs_id}'
        #pth = f'/sml/jannaschk/IRIS_raw_data/{obs_id}'
        obs = observation( pth, keep_null=True )

        raster_MgIIk = obs.raster("Mg II k")

        MgIIk = {'lambda_min' : 2794,
            'lambda_max' : 2806,
            'n_breaks' : 960, #do not modify it! it has to be 960.
            'line' : "Mg II k",
            'field' : "NUV",
            'threshold' : 10
        }

        #prepare obs data with Jonas pipeline:

        obs_MgIIk = utils.Obs_raw_data(obs_id, raster_MgIIk, **MgIIk)
        obs.close()

        # save data as struct

        # here in the parameter obs_type in the fct save_array
        # should be either QS, AR or PF

        obs_MgIIk.save_arr(f'{obs_id}', 'MgIIk', obs_type, highest_directory)
        
        print('')
        print(f'Done cleaning: {obs_id}')
        print('')

    return None



def compute_cluster_var(cluster_data):
    '''
    this fct computes the variance of a cluster. it is used in the fct compute_all_inter_cluster_variances.
    and also in centroid_summary

    There are many choices for variance calculating, here we used this one:
    calculate the variance of the each point of a single wavelength point.
    Then add up all these 960 variances, and you get a number between 0.5 and 3 more or less.

    '''
    variance = np.var(cluster_data, axis = 0, ddof=0)

    summed_variance = np.sum(variance)
    
    return (summed_variance)



def plot_kmeans_results( X, labels, n_clusters, ax=None, **kwargs ): 

    '''
    I didnt use this fct yet, but it is a super simple version of centroid_summary.
    '''


    ax = ax or plt.gca()
    colors = ['#FF9C34', 'black', 'gold', 'aquamarine', '#4EACC5', 
              '#FF9C34', '#4E9A06', 'pink', 'blue', 'grey', 'red']
    for k, clr in zip(range(n_clusters), colors):
        my_members = labels == k
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=clr, marker='X', label = f'Group {k+1}', markersize=5)
        ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_title('K-means results') 
    plt.grid(True)
    return None



def plot_closest_to_centroids(centroids_dict, closest_centroids_dict,  n_clusters, plot_centroids_as_well = True, Zoom = True,   ax=None): 
    #ax = ax or plt.gca()
    '''
    this fct is also a version of a simple centroid_summary. 
    
    '''

    fig, ax = plt.subplots(figsize=(20, 15))  # width = 10 inches, height = 6 inches
    obs_wavelength=np.linspace(2794,2806,960)


    for n in range(n_clusters):
        ax.plot(obs_wavelength, closest_centroids_dict[tuple(f'closest_centroids_{n}')], label=f'Closest Spectrum to centroid {n+1}', linestyle='-')

        if plot_centroids_as_well:
            ax.plot(obs_wavelength, centroids_dict[f'centroids_{n}'], label=f'Centroid {n+1}', linestyle='--')
            title = 'K-means results: Centroids and closest real spectra'
        else:
            title = 'K-means results: Closest real spectra'

        ax.legend(loc='best', shadow=False, scatterpoints=1)

    ax.set_title(title) 
    plt.grid(False)

    # Create a new figure
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Copy data from the original Axes to the new Axes
    for line in ax.get_lines():
        ax2.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())

    # Set the same labels and title
    ax2.set_title(ax.get_title() + 'Zoom around Mg K Line')
    ax2.set_xlabel(ax.get_xlabel())
    ax2.set_ylabel(ax.get_ylabel())

    # Set specific x and y limits
    ax2.set_xlim(2795.5, 2797)
    ax2.set_ylim(0.3, 1.1)

    # Show the zoomed-in plot
    #plt.show()


    return None




def compute_all_inter_cluster_variances(data, centroids, labels, n_clusters):
    '''
    this function computes the inter-cluster variance for each cluster
    and returns a dictionary with the results, in order of increasing variance.

    '''

    inter_cluster_variances = {}
    
    for n in range(n_clusters):


        centroid_point = centroids[n]

        cluster_points = data[labels == n]

        cluster_variance = sum_squared_distances(cluster_points, centroid_point)

        inter_cluster_variances[n] = cluster_variance

    sorted_dict = {k: v for k, v in sorted(inter_cluster_variances.items(), key=lambda item: item[1])}

    
    return sorted_dict #, inter_cluster_variances (uncomment if you want to include the unsorted dictionary as well)



def sum_squared_distances(array, data_point):
    """
    Computes the sum of squared distances from each point in an array to a single data point.

    Parameters:
    array (numpy.ndarray): A 2D array where each row is a data point.
    data_point (numpy.ndarray): A 1D array representing the single data point.

    Returns:
    float: The sum of squared distances.
    """
    # Ensure array and data_point are numpy arrays
    array = np.asarray(array)
    data_point = np.asarray(data_point)
    
    # Compute the squared distances
    squared_distances = np.sum((array - data_point) ** 2, axis=1)
    
    # Sum the squared distances
    total_squared_distance = np.sum(squared_distances)
    
    return total_squared_distance









# Here, I fit piece by piece each obs to find the centroids. Also, the centroids get SOM-ordered.


'''
cpu_percentage = 30
total_cpus = multiprocessing.cpu_count()
num_cpus = int(total_cpus * (cpu_percentage / 100))
os.environ["OMP_NUM_THREADS"] = str(num_cpus)

# params:
n_clusters = 80
batch_size = 1000
n_init = 10
verbose = 0

mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                            n_init=n_init, max_no_improvement=10, verbose=verbose)

base_dir = "IRIS_30_obs_prepared"

for root, dirs, files in os.walk(base_dir):
    for directory in dirs:
        if directory.startswith("AR"):
            ar_dir = os.path.join(root, directory)
            for directory_obs in os.listdir(ar_dir):
                obs_id = directory_obs
                obs_cls = utils_data_prep.load_obs_data(highest_directory = base_dir,
                                                        filename = f'{obs_id}',
                                                        line = 'MgIIk',
                                                        typ = f'{ar_dir[-2:]}')
                Data = obs_cls.im_arr_global[:,:,:].reshape(-1,960)
                Data = utils_kmeans.filter_out_zero_spectra(Data)
                #Big_data.append(Data)
                print(f'{obs_id} loaded')
                mbk.partial_fit(Data)
                print('Centroids were updated')
            print('AR done')
            print('')
        elif directory.startswith("PF"):
            pf_dir = os.path.join(root, directory)
            for directory_obs in os.listdir(pf_dir):
                obs_id = directory_obs
                obs_cls = utils_data_prep.load_obs_data(highest_directory = base_dir,
                                                        filename = f'{obs_id}',
                                                        line = 'MgIIk',
                                                        typ = f'{pf_dir[-2:]}')
                Data = obs_cls.im_arr_global[:,:,:].reshape(-1,960)
                Data = utils_kmeans.filter_out_zero_spectra(Data)
                #Big_data.append(Data)
                print(f'{obs_id} loaded')
                mbk.partial_fit(Data)
                print('Centroids were updated')
            print('PF done')
            print('')
        elif directory.startswith("QS"):
            qs_dir = os.path.join(root, directory)
            for directory_obs in os.listdir(qs_dir):
                obs_id = directory_obs
                obs_cls = utils_data_prep.load_obs_data(highest_directory = base_dir,
                                                        filename = f'{obs_id}',
                                                        line = 'MgIIk',
                                                        typ = f'{qs_dir[-2:]}')
                Data = obs_cls.im_arr_global[:,:,:].reshape(-1,960)
                Data = utils_kmeans.filter_out_zero_spectra(Data)
                #Big_data.append(Data)
                print(f'{obs_id} loaded')
                mbk.partial_fit(Data)
                print('Centroids were updated')
            print('QS done')
            print('')

del directory, ar_dir, directory_obs, obs_id, obs_cls
gc.collect()

centroids = mbk.cluster_centers_
centroids = som.obtain_som_ordered_data(centroids_to_be_ordered = centroids)
labels = mbk.labels_
inertia = mbk.inertia_



np.save(f'centroids_n_clusters_{n_clusters}_SOM.npy', centroids)
'''













'''

# brandon Panos' fcts:


# def preprocess_obs( obs, obs_type, lambda_min, lambda_max, n_bins, start_stop_inds=None, length_minutes=25, flare_margin_minutes=1, savedir="level_2A/", data_format="hdf5", verbosity_level=1 ):
"""
verbosity level : int
    0: no output
    1: print shapes
    2: print flare diagnostics
"""

raster = obs.raster("Mg II k")

if verbosity_level == 1:
    print("\n---------------------------------------------------------------- Flare Events -------------------------------------------------------------------------------------------------------------------\n")
    # plot flare locations
    obs.goes.events.plot_flares()
    plt.show()
    
    # get closest flare
    flares = obs.goes.events.get_flares()
    mx_flares = obs.goes.events.get_flares( classes="MX" )
    
    if len( flares ) > 0:
        display(HTML(flares.to_html()))
        if len( mx_flares ) > 0:
            closest_flare = mx_flares.iloc[0]
            print( "Start time of closest M/X flare: {} (distance {} arcsec)".format( closest_flare['event_starttime'], np.round( closest_flare['dist_arcsec'], 2 ) ) )
        else:
            print( "No M/X flares within range." )
    else:
        print( "No flares within range." )       
    
    print("\n---------------------------------------------------------------- Proposed time cut -------------------------------------------------------------------------------------------------------------\n")

if start_stop_inds is None:
    start, stop, delta_t = time_cut( obs, flare=(obs_type=="PF"), length_minutes=length_minutes, flare_margin_minutes=flare_margin_minutes )
else:
    start = start_stop_inds[0]
    stop = start_stop_inds[1]
    delta_t = ( from_Tformat( raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )[stop-1]['DATE_OBS'] ) - from_Tformat( raster.get_raster_pos_headers( raster_pos=0 )[start]['DATE_OBS'] ) ).seconds/60

if verbosity_level >= 2:
    plot_goes_flux( obs, start, stop )
    
if verbosity_level >= 1:
    print("index interval for raster position 0: {}-{} (totalling {} exposures, {} minutes)".format(start, stop, stop-start, np.round(delta_t, 1 ) ) )

raster.cut( raster.get_global_raster_step( raster_pos=0, raster_step=start ), raster.get_global_raster_step( raster_pos=0, raster_step=stop ) )

if verbosity_level >= 2:
    display( animate( obs ) )
    print("HEK URL:")
    display( obs.get_hek_url() )

data = get_interpolated_raster_data( raster, lambda_min, lambda_max, n_bins )
    
if verbosity_level >= 1:
    full_obs_timedelta = np.round( ( from_Tformat( raster.time_specific_headers[-1]['DATE_OBS'] ) - from_Tformat( raster.time_specific_headers[0]['DATE_OBS'] ) ).seconds/60, 1 )
    print("Raster is {} minutes long after cut (should be equal to {} minutes estimated above)".format( full_obs_timedelta, delta_t, 2 ) )
    print( "data Shape: {}, raster shape: {}, n_raster_pos: {}".format( data.shape, raster.shape, raster.n_raster_pos ) )

try:
    sji = obs.sji("Mg II h/k 2796")
except:
    sji = obs.sji("Si")
    
return data, raster, sji
        
# def extract_flare( obs, lambda_min, lambda_max, n_bins, length_minutes=25, flare_margin_minutes=1, savedir="level_2A/", data_format="hdf5", verbosity_level=1 ):
    
    raster = obs.raster("Mg II k")
    
    # get event info of closest flare
    mx_flares = obs.goes.events.get_flares( classes="MX" )
    closest_flare = mx_flares.iloc[0]
    
    # extract raster_pos=0 indices of flare start and stop
    start_time = from_Tformat( closest_flare['event_starttime'] )
    stop_time = from_Tformat( closest_flare['event_endtime'] )
    ts = np.array( raster.get_timestamps( raster_pos=0 ) )
    
    start = np.argmin( np.abs(ts-to_epoch(start_time) ) )
    stop = np.argmin( np.abs(ts-to_epoch(stop_time) ) )
    
    eff_start_time = from_Tformat( raster.get_raster_pos_headers( raster_pos=0 )[start]['DATE_OBS'] )
    eff_stop_time = from_Tformat( raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )[stop]['DATE_OBS'] )

    # plot GOES curve
    if verbosity_level >= 2:
        plot_goes_flux( obs, start, stop )
    
    if verbosity_level >= 1:
        print( "Cutting raster from indices {}-{} --> {:.1f} minutes (of {:.1f} minutes total flare duration)".format( start, stop, (eff_stop_time-eff_start_time).seconds/60, (stop_time-start_time).seconds/60 ) )
    
    # cut the raster
    raster.cut( raster.get_global_raster_step( raster_pos=0, raster_step=start ), raster.get_global_raster_step( raster_pos=0, raster_step=stop+1 ) )
    
    # show animation
    if verbosity_level >= 2:
        display( animate( obs ) )
        print("HEK URL:")
        display( obs.get_hek_url() )
    
    # save the data
#         print( "Saving data.." )
    data = get_interpolated_raster_data( raster, lambda_min, lambda_max, n_bins )
#         save_data( data, raster.headers, savedir, "{}_{}".format( "FL", obs.full_obsid ) )
    
    if verbosity_level >= 1:
        full_obs_timedelta = np.round( ( from_Tformat( raster.time_specific_headers[-1]['DATE_OBS'] ) - from_Tformat( raster.time_specific_headers[0]['DATE_OBS'] ) ).seconds/60, 1 )
        print("Raster is {} minutes long after cut".format( full_obs_timedelta ) )
        print( "data Shape: {}, raster shape: {}, n_raster_pos: {}".format( data.shape, raster.shape, raster.n_raster_pos ) )
        
    try:
        sji = obs.sji("Mg II h/k 2796")
    except:
        sji = obs.sji("Si")
    
    return data, raster, sji

# --------


'''





# fcts from Brandon:

def plot_goes_flux( obs, i_start, i_stop ):

    raster = obs.raster("Mg II k")
    start_date = obs.start_date
    end_date = obs.end_date
    th = raster.get_raster_pos_headers( raster_pos=0 )
    th_n = raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )
    start_cut_date = from_Tformat( th[i_start]['DATE_OBS'] )
    stop_cut_date = from_Tformat( th_n[i_stop]['DATE_OBS'] )
    
    # put time cut into green area
    ax = obs.goes.xrs.data.plot( y='B_FLUX', logy=True, label="GOES X-ray Flux", figsize=(24,5), lw=2 )
    ax.axvspan( start_cut_date, stop_cut_date, alpha=0.2, color='green' )
    ax.axvline( x=start_cut_date, color='green', linestyle='--', linewidth=3.0 )
    ax.axvline( x=stop_cut_date, color='green', linestyle='--', linewidth=3.0 )
    ax.set_xlim([start_date, end_date])
    plt.text( start_cut_date, 1e-1, i_start, fontsize=14, color="green", ha="center" )
    plt.text( stop_cut_date, 1e-1, i_stop+1, fontsize=14, color="green", ha="center" )

    ax.axhline( y=1e-4, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-5, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-6, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-7, color='black', linestyle='--', linewidth=1.0 )
    ax.axhline( y=1e-8, color='black', linestyle='--', linewidth=1.0 )
    
    steps = obs.raster("Mg II k").get_raster_pos_steps( raster_pos=0 )
    if steps<=250:
        gridstep=1
    elif steps<=2500:
        gridstep=10
    else:
        gridstep=100
        
    for i in range( 0, len(th), gridstep ):
        ax.axvline( x=th[i]['DATE_OBS'], color='black', linestyle='--', linewidth=1.0 )
    
    ax.set_ylim([1e-9, 1e-2])
    ax.set_ylabel(r'Watts / m$^2$')
    ax.set_xlabel("Universal Time")
    
    ax2 = ax.twinx()
    ax2.set_yscale( 'log' )
    ax2.set_ylim( ax.get_ylim() )
    ax2.set_yticks([3e-8, 3e-7, 3e-6, 3e-5, 3e-4])
    ax2.set_yticklabels(['A', 'B', 'C', 'M', 'X'])
    ax2.minorticks_off()
    ax2.tick_params( right=False )
    
    ax3 = ax.twiny()
    ax3.set_xlim( ax.get_xlim() )
    xticks = [from_Tformat(th[i]['DATE_OBS']) for i in np.arange( 0, len(th), 10*gridstep )]
    ax3.set_xticks( xticks )
    ax3.set_xticklabels( np.arange( 0, len(th), 10*gridstep )  )
    ax3.set_xlabel("Exposure")
    plt.show()
    
def animate( obs ):
    ts = np.array( obs.sji[0].get_timestamps() )
    start_date = obs.raster("Mg II k").time_specific_headers[0]['DATE_OBS']
    stop_date = obs.raster("Mg II k").time_specific_headers[-1]['DATE_OBS']
    i_start_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( start_date ) ) ) )
    i_stop_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( stop_date ) ) ) )
    return obs.sji[0].animate( index_start=i_start_sji, index_stop=i_stop_sji, cutoff_percentile=99.0 )

def get_interpolated_raster_data( raster, lambda_min, lambda_max, n_bins ):
    """
    Returns data in the shape [n_raster_pos, n_steps_per_raster, n_y_pix, n_lambda_pix]
    
    Parameters
    ----------
        raster : irisreader.raster_cube
            already time cut raster cube object
        lambda_min : float
            minimum wavelength for interpolation
        lambda_max : float
            maximum wavelength for interpolation
        n_bins : int
            number of bins for interpolation
    """
    
    # empty array with the desired shape
    X = np.empty( shape=( raster.n_raster_pos, raster.shape[0], raster.shape[1], n_bins ) )

    for raster_pos in range( raster.n_raster_pos ):
        for step in range( raster.get_raster_pos_steps( raster_pos ) ):
            X[ raster_pos, step, :, :] = raster.get_interpolated_image_step( 
                step, raster_pos=raster_pos, lambda_min=lambda_min, lambda_max=lambda_max, n_breaks=n_bins
        )
    
    return X

def time_cut( obs, flare=False, length_minutes=25, flare_margin_minutes=1 ):
    
    raster = obs.raster("Mg II k")
    ts = np.array( raster.get_timestamps( raster_pos=0 ) )
    
    if flare:
        # find end of preflare in terms of raster_pos=0 raster that happens _before_ the flare (taken care of in find_preflare_end)
        # stop = find_preflare_end( obs, margin_minutes=flare_margin_minutes )
        
        # find closest raster_pos=0 raster to suggested start time
        start_time = ts[stop] - length_minutes*60
        start = np.argmin( np.abs( ts - start_time ) )
        
    else:
        # set start to first raster sweep
        start = 0
        
        # find closest raster_pos=0 raster to start time at first raster sweep
        start_time = ts[0]
        stop = np.argmin( np.abs( ts - (start_time + length_minutes*60) ) )
    
    start_date = from_Tformat( raster.get_raster_pos_headers( raster_pos=0 )[start]['DATE_OBS'] )
    stop_date = from_Tformat( raster.get_raster_pos_headers( raster_pos=raster.n_raster_pos-1 )[stop-1]['DATE_OBS'] )
    delta_t = stop_date - start_date
    return [start, stop, np.round( delta_t.seconds/60, 1 )]

def find_sji_inds( obs ):
    ts = np.array( obs.sji[0].get_timestamps() )
    start_date = obs.raster("Mg II k").time_specific_headers[0]['DATE_OBS']
    stop_date = obs.raster("Mg II k").time_specific_headers[-1]['DATE_OBS']
    i_start_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( start_date ) ) ) )
    i_stop_sji = np.argmin( np.abs( ts - to_epoch( from_Tformat( stop_date ) ) ) )
    return i_start_sji, i_stop_sji



# brandon's original animate_mask function:
'''

THIS BELOW is Brandon's original animate_mask function.

def animate_mask(obs, obs_type, centroids, clr_dic):
    
    """
    Returns an animation of the observation with a centroid mask overlay 
    
    Parameters
    ----------
        obs : irisreader.observation
        obs_type : str
            allows us to set specific parameters for each observation
        centroids : float
            centroids found by k-means [n_centroids, lambda]
        clr_dic : {int:str}
            dictionary to color code each centroid 

    """

    sji = obs.sji[1]
    
    # extract only the most interesting part of an observation with the cut method
    if obs_type == 'qs':
        try: sji.cut( 0, 10 )
        except: pass
        gamma = .1 # control contrast

    if obs_type == 'f1':
        try: sji.cut( 1086, 1244 )
        except: pass
        gamma = .4
        
    if obs_type == 'f2':
        try: sji.cut( 0, 50 )
        except: pass
        gamma = .1

    raster = obs.raster("Mg II k")
    raster_inds = [find_closest_raster( raster, sji, i )[0] for i in range(sji.n_steps)]
    
    #------------ fucntion for labeling profiles with there closest centroids --------------
    
    def assign_mg2k_centroids( X, centroids ):

        centroid_ids = list( range( centroids.shape[0] ) )

        # check whether X comes in the correct dimensions
        if not X.shape[1] == centroids.shape[1]:
            raise ValueError( "Expecting X to have shape (_,{}). Please interpolate accordingly (More information with 'help(assign_mg2k_centroids)').".format( centroids.shape[1] ) )

        # create nearest centroid finder instance and fit it
        knc = NearestCentroid()
        knc.fit( X=centroids, y=centroid_ids )

        # predict nearest centroids for the supplied spectra
        # (making sure that X is normalized)
        assigned_mg2k_centroids = knc.predict( normalize(X) )

        # return vector of assigned centroids

        return assigned_mg2k_centroids 

    #---------------------------- label data ---------------------------

    labels = []

    for i in raster_inds:
        X = raster.get_interpolated_image_step( 
                    step = i, 
                    lambda_min = LAMBDA_MIN, 
                    lambda_max = LAMBDA_MAX, 
                    n_breaks = 216  
        )

        labels.append( assign_mg2k_centroids( X, centroids ) )

    #---------------------------- set up animation ---------------------------
    n = sji.shape[0]

    interval_ms = 100
    N = sji.shape[1]

    # initialize plot
    image = sji.get_image_step( 0 ).clip(min=0.01)**gamma
    sltxpos = sji.get_slit_pos( 0 )
    sltypos = np.linspace( 0, image.shape[0]-1, N, dtype=np.int )
    sji_flare_mask=labels[0][sltypos]
    clr_mask = [clr_dic.get(sji_flare_mask[i],'grey') for i in range(len(sji_flare_mask))]
    fig = plt.figure( figsize=(10,10) )
    im = plt.imshow( -image, cmap="Greys", origin='lower' )
    scat = plt.scatter( [sltxpos]*N, sltypos, c=clr_mask, marker='s', s=10, alpha=1)

    # do nothing in the initialization function
    def init():
        return im, scat

    # animation function
    def animate(i):
        sltxpos = sji.get_slit_pos( i )
        sji_flare_mask=labels[i][sltypos] # just some random shit
        xcenix = sji.headers[i]['XCENIX']
        ycenix = sji.headers[i]['YCENIX']
        date_obs = sji.headers[i]['DATE_OBS']
        im.axes.set_title( "Frame {}: {}".format( i, date_obs ) )
        im.set_data( -sji.get_image_step( i ).clip(min=0.01)**gamma )
        scat.set_offsets( np.array([ [sltxpos]*N, sltypos ] ).T )
        scat.set_array( sji_flare_mask )
        clr_mask = [clr_dic.get(sji_flare_mask[i],'grey') for i in range(len(sji_flare_mask))]
        scat.set_color( clr_mask )
        return im, scat

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=sji.n_steps, interval=interval_ms, blit=True)

    # Close the plot
    plt.close(anim._fig)

    # Show animation in notebook
    return HTML(anim.to_html5_video())

'''















'''

Brandon's animate_mask fct before i introduced dictionaries merged groups into it: (date: 03.07.24, 19:03)



def animate_mask(obs, centroids, clr_dic, start, end, interval_ms = 100):
    """
    Returns an animation of the observation with a centroid mask overlay
    Parameters
    ----------
        obs : irisreader.observation
        centroids : float
            centroids found by k-means [n_centroids, lambda]
        clr_dic : {int:str}
            dictionary to color code each centroid 
        start : float in unix time
            start of the animated movie
        end : float in unix time
            end of the animated movie
        interval_ms : int
            interval between frames in milliseconds
    """


    try:
        obs.sji('Mg II h/k').line_info
    except:
        print('No MgII k line found in observation')
        return None




    sji = obs.sji[1]  #index 1 means the Mg II h/k line, which is what we want. Index 0 would be the Si IV line.
    #times_in_unix = np.array( obs.sji[1].get_timestamps() )

    times_in_unix = obs.sji[1].get_timestamps()
    times_in_unix = np.array(times_in_unix)
    
    start_index = np.abs(times_in_unix - start).argmin()
    end_index =   np.abs(times_in_unix - end).argmin()

    # Here, we cut it to the desired length

    sji.cut(start_index, end_index)


    raster = obs.raster("Mg II k")
    raster_inds = [find_closest_raster( raster, sji, i )[0] for i in range(sji.n_steps)]    
   

    #---------------------------- label data ---------------------------

    labels = []
    for i in raster_inds:
        X = raster.get_interpolated_image_step( 
                    step = i, 
                    lambda_min = 2794,
                    lambda_max = 2806,
                    n_breaks = 960 #used to be 216 originally
        )

        labels.append( assign_mg2k_centroids( X, centroids ) )

    #---------------------------- set up animation ---------------------------
    #n = sji.shape[0]

    N = sji.shape[1]
    gamma = 0.1

    # initialize plot

    image = sji.get_image_step( 0 ).clip(min=0.01)**gamma

    slit_x_position = sji.get_slit_pos( 0 )
    slit_y_position = np.linspace( 0, image.shape[0]-1, N, dtype=np.int )

    sji_flare_mask=labels[0][slit_y_position]

    clr_mask = [clr_dic.get(sji_flare_mask[i],'grey') for i in range(len(sji_flare_mask))]
    
    fig = plt.figure( figsize=(12,12) )


    im = plt.imshow( -image, cmap="Greys", origin='lower' )

    scat = plt.scatter( [slit_x_position]*N, slit_y_position, c=clr_mask, marker='s', s=10, alpha=1)

    # do nothing in the initialization function
    def init():
        return im, scat

    # animation function
    def animate(i):
        slit_x_position = sji.get_slit_pos( i )
        sji_flare_mask=labels[i][slit_y_position]
        #xcenix = sji.headers[i]['XCENIX']
        #ycenix = sji.headers[i]['YCENIX']
        date_obs = sji.headers[i]['DATE_OBS']
        im.axes.set_title( "Frame {}: {}".format( i, date_obs ) )

        im.set_data( -sji.get_image_step( i ).clip(min=0.01)**gamma )

        scat.set_offsets( np.array([ [slit_x_position]*N, slit_y_position ] ).T )

        scat.set_array( sji_flare_mask )

        clr_mask = [clr_dic.get(sji_flare_mask[i],'grey') for i in range(len(sji_flare_mask))]

        scat.set_color( clr_mask )

        return im, scat

    # Call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=sji.n_steps, interval=interval_ms, blit=True)

    # Close the plot
    plt.close(anim._fig)

    # Show animation in notebook
    return HTML(anim.to_html5_video())





'''
