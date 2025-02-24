import copy
import numpy as np
from math import inf

def truncnorm(mu=0, sigma=1, bounds=[-inf,inf], n=1):
    values = []
    for i in range(n):
        while True:
            value = np.random.normal(mu,sigma)
            if bounds[0] <= value <= bounds[1]:
                values.append(value)
                break
    return values

def HVCactivity(n:int, s:float, ds:float, t:float, dt:float, Tau:float=1, dTau:float=0, S:float=1, dS:float=0, T:float=1, dT:float=0, jitter:float=None, clock:float=None, seed:int=None):
    ''' Function generates HVC spiking activity as described in [citation needed]
    Input:
        n      - (int) Number of neurons                                                           \n
        s ± ds - (float) Number of spikes in a burst (or Song) per neuron                          \n
        t ± dt - (float) Temporal distance between spikes                                          \n
        Tau ± dTau - (float, optional, deafault = 1, 0) Total time of a single song (max. value)   \n
        S ± dS     - (float, optional, deafault = 1, 0) Number of Song repetitions                \n
        T ± dT     - (float, optional, deafault = 1, 0) Time between the songs                     \n
        clock      - (float, optional, deafault = None) Size of time step used to discretize time  \n
        jitter     - (float, optional, deafault = None) Standard deviation of jitter noise         \n
        seed       - (int,   optional, deafault = None) Seed for the generation of random numbers  \n  
    Output:                                                                                        \n
        Array of Spiketimes and corresponding index array
    '''
    if seed != None:
        np.random.seed(seed)
    if dt < clock:
        raise ValueError('The clock time has to be smaller than the temporal distance between the spikes')
    while True: # Ensure interval length >0
        I = np.random.normal(Tau, dTau)
        if I>0:
            break
    song = []; song_indices = []
    init_spikes = [0]+list(np.random.uniform(0, I, size=n-1))
    for i in range(len(init_spikes)):
        spike = init_spikes[i]
        song.append(spike); song_indices.append(i)
        times = truncnorm(t,dt, bounds=[clock,I], n=int(round(truncnorm(s-1,ds, bounds=[0,inf])[0], 0)))
        for time in times:
            spike += time
            song.append(spike); song_indices.append(i)
    spikes = copy.deepcopy(song)
    indices = copy.deepcopy(song_indices)
    temporal_spacers = []
    if jitter == None:
        for k in range(1,int(round(truncnorm(S,dS, bounds=[0,inf])[0], 0))):
            temporal_spacers.append(truncnorm(T,dT, bounds=[0,inf])[0])
            spikes = spikes + [spike+k*I+sum(temporal_spacers) for spike in song]
            indices = indices + song_indices
    else:
        song = [-inf]+song+[inf]
        for k in range(1,int(round(truncnorm(S,dS, bounds=[0,inf])[0], 0))):
            temporal_spacers.append(truncnorm(T,dT, bounds=[0,inf])[0])
            for i in range(1,len(song)-1):
                spikes.append(k*I+sum(temporal_spacers)+(song[i]+truncnorm(0, jitter, bounds=[-(abs(song[i]-song[i-1])-clock)/2, (abs(song[i+1]-song[i])-clock)/2])[0]))
                indices.append(song_indices[i-1])   ######## Check!
    return np.array(spikes), np.array(indices)