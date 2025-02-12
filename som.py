# SOM = self organising map

import numpy as np
import pandas as pd

# Code taken from Brandon Panos

'''
SOM if a self organising map that solves the traveling salesman problem (TSP)
It will order the centroid shapes, in an order that they look close visually next to one another.
It is only a visual help.
'''


def obtain_som_ordered_data(centroids_to_be_ordered):

    route = som(iterations=100000, learning_rate=0.8, neuron_weight=8).fit(centroids_to_be_ordered)

    centroids_ordered = [centroids_to_be_ordered[route[i]] for i in range(len(route))]

    return centroids_ordered


class som:
     '''
     Solves the traveling salesman problem (TSP) using a self organising map (SOM)
     Literature for SOM: Teuvo Kohonen (1998) The self-organizing map
                         Lucas Brocki (2010) Kohonen Self-Organizing Map for the Traveling Salesperson Problem
     '''
     def __init__(self, iterations=100000, learning_rate=0.8, neuron_weight=8):

         self.iterations = iterations
         self.learning_rate = learning_rate
         self.neuron_weight = neuron_weight

     global select_closest
     def select_closest(candidates, origin):
         """
         Return the index of the closest candidate to a given point.
         """
         return euclidean_distance(candidates, origin).argmin()

     global euclidean_distance
     def euclidean_distance(a, b):
         """
         Return the array of distances of two numpy arrays of points.
         """
         return np.linalg.norm(a - b, axis=1)

     def fit(self, spectra):

         n_features = spectra.shape[1]
         n_neurons = spectra.shape[0] * self.neuron_weight

         mins = np.min(spectra, axis=0)
         maxs = np.max(spectra, axis=0)
         norm_spectra = (spectra - mins)/(maxs-mins)

         np.random.seed(0)
         network = np.random.rand(n_neurons, n_features)

         print('Network of {} neurons created. Starting the iterations:'.format(n_neurons))

         n = n_neurons
         for i in range(self.iterations):
             # if not i % 100:
                # print('\t> Iteration {}/{}'.format(i, self.iterations), end="\r")
                # print('iterating')

             # select a random spectrum
             spectrum = norm_spectra[np.random.randint(norm_spectra.shape[0], size=1), :]
             winner_idx = np.linalg.norm(network - spectrum, axis=1).argmin()

             # Generate a filter that applies changes to the winner's gaussian (not sure why 10)
             center = winner_idx
             radix = n//10
             domain = network.shape[0]
             # Impose an upper bound on the radix to prevent NaN and blocks
             if radix < 1:
                 radix = 1
             # Compute the circular network distance to the center
             deltas = np.absolute(center - np.arange(domain))
             distances = np.minimum(deltas, domain - deltas)
             # Compute Gaussian distribution around the given center
             gaussian = np.exp(-(distances*distances) / (2*(radix*radix)))

             # Update the network's weights (closer to the city)
             learning_rate = self.learning_rate
             network += gaussian[:,np.newaxis] * learning_rate * (spectrum - network)

             # Decay the variables
             learning_rate = learning_rate * 0.99997
             n = n * 0.9997
             # Check if any parameter has completely decayed.
             if n < 1:
                 print('Radius has completely decayed, finishing execution',
                 'at {} iterations'.format(i))
                 break
             if learning_rate < 0.001:
                 print('Learning rate has completely decayed, finishing execution',
                 'at {} iterations'.format(i))
                 break
         else:
             print('Completed {} iterations.'.format(iterations))

         spectra_df = pd.DataFrame(spectra, columns=None, index=None)
         spectra_df['winner'] = spectra_df.apply(
             lambda c: select_closest(network, c),
             axis=1, raw=True)

         route = spectra_df.sort_values('winner').index.tolist()

         return route
     


'''
Usage:
# from som import *
np.random.seed(0)

route = som(iterations=100000, learning_rate=0.8, neuron_weight=8).fit(centroids_pf)
'''
