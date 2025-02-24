
############################
# Simulation of Nucleus RA #
#       HPC Version        #
# Last Update: 2024-11-12  #
############################

# External packeges
import os
import h5py
import yaml
import json
import shutil
import numpy as np
from brian2 import *
from pathlib import Path
from datetime import datetime

start = datetime.now()

# Local Files
from HVCSimulator import *

# Create simulation output folder
if not os.path.exists('simulation_output'):
      os.makedirs('simulation_output')

# Get the SLURM job ID
slurm_job_id = os.getenv('SLURM_JOB_ID')

# Determinte the slur job ID or create a random non-existing folder
if slurm_job_id == None:
    slurm_job_id = 0
    while True:
        slurm_job_id += 1
        if not os.path.exists(Path('simulation_output/Simulation {}'.format(slurm_job_id))): 
            break

# Create simulation output folder for the current simulation
so_path = Path('simulation_output/Simulation {}'.format(slurm_job_id)) # simulation output folder path
if not os.path.exists(so_path):
      os.makedirs(so_path)

# Copy the Parameters.yaml file to the simulation output folder and load the parameters
source_file      = 'Parameters.yaml'
destination_file = Path('simulation_output/Simulation {}/Parameters.yaml'.format(slurm_job_id))
shutil.copy(source_file, destination_file)

# Open YAML file with parameters
with open(destination_file, 'r') as file:
    parameters = yaml.safe_load(file)
            
### Simulation Settings ###
scale        = parameters['Simulation_Settings']['scale']  
report       = parameters['Simulation_Settings']['report']
saveSpikes   = parameters['Simulation_Settings']['saveSpikes']
saveVoltages = parameters['Simulation_Settings']['saveVoltages']
saveWeights  = parameters['Simulation_Settings']['saveWeights']
saveRates    = parameters['Simulation_Settings']['saveRates']
saveProtocol = parameters['Simulation_Settings']['saveProtocol']

# Notify user about copied parameters so that new simulation can be started
if report:
      print(f'Copied Parameters.yaml to simulation_output/Simulation {slurm_job_id}/Parameters.yaml \n')

### HVC Parameters ###
s      = parameters['HVC_Parameters']['s']     # Number of spikes in a burst per neuron
ds     = parameters['HVC_Parameters']['ds']
ts     = parameters['HVC_Parameters']['ts']    # Temporal spacing between spikes in a burst im ms
dts    = parameters['HVC_Parameters']['dts']  
Tau    = parameters['HVC_Parameters']['Tau']   # Temporal length of a song in ms
dTau   = parameters['HVC_Parameters']['dTau']
S      = parameters['HVC_Parameters']['S']     # Number of song repetitions
dS     = parameters['HVC_Parameters']['dS']
T      = parameters['HVC_Parameters']['T']     # Temporal spacing between songs
dT     = parameters['HVC_Parameters']['dT']
jitter = parameters['HVC_Parameters']['jitter']

### Simulation Parameters ###
Timescale   = ms 
Buffer_Time = parameters['Simulation_Parameters']['Buffer_Time']       # Buffer time before and after the simulation     
Simtime     = S*Tau + (S-1)*T + Buffer_Time                            # Total Simulation Time
sim_dt      = parameters['Simulation_Parameters']['sim_dt']*Timescale  # Time step of the simulation 
N_HVC_RA    = int(20000*scale/100)   # Number of RA projecting HVC Axons
N_RA_nXIIts = int( 8000*scale/100)   # Number of nXIIts projecting neurons
N_RA_int    = int( 1000*scale/100)   # Number of Inhibitory neurons

### Start Scope ###
start_scope()
prefs.core.default_float_dtype = float32 # Set global preference to single-precision floating-point format (float32)

######################
### Neuronal Model ###
######################

### HVC Activity ###

hvc_spiketimes, indices = HVCactivity(N_HVC_RA, s, ds, ts, dts, Tau, dTau, S, dS, T, dT, jitter=jitter, clock=float(sim_dt/Timescale))
HVC_RA    = SpikeGeneratorGroup(N_HVC_RA, indices, hvc_spiketimes*ms, dt=sim_dt)

### Neuronal Model Initialization ###
neuronal_model = parameters['Dynamic_Neuron_Parameters']['model']

# General Parameters
tau_refractory_int = parameters['Dynamic_Neuron_Parameters']['tau_refractory_int']*ms  # Refractory period
tau_refractory_nXIIts = parameters['Dynamic_Neuron_Parameters']['tau_refractory_nXIIts']*ms  # Refractory period

if neuronal_model == 'LIF':
      # Neuronal Parameters for RA_int Neurons # Inhibitory
      v_thr_int          = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_LIF']['v_thr_int']*mV            # Firing threshold 
      v_rest_int         = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_LIF']['v_rest_int']*mV          # Resting potential
      tau_int            = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_LIF']['tau_int']*ms             # Membrane time constant
      
      # Neuronal Model for RA_int Neurons 
      LIF_int = Equations(''' dv/dt = (v_rest_int - v)/tau_int : volt ''')
      RA_int    = NeuronGroup(N_RA_int,    LIF_int,    threshold='v>v_thr_int',    reset='v=v_rest_int',   method='exact', refractory=tau_refractory_int,    dt=sim_dt)

      # Neuronal Parameters for RA_nXIIts Neurons # Exitatory, Tonically Active
      v_thr_nXIIts          = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_LIF']['v_thr_nXIIts']*mV            # Firing threshold
      v_rest_nXIIts         = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_LIF']['v_rest_nXIIts']*mV          # Resting potential
      I_ext                 = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_LIF']['I_ext']*nA                  # External current
      R_nXIIts              = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_LIF']['R_nXIIts']*Mohm             # Resistance
      tau_nXIIts            = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_LIF']['tau_nXIIts']*ms             # Membrane time constant

      # Neuronal Model for RA_nXIIts Neurons # 
      LIF_nXIIts = Equations(''' dv/dt = ((v_rest_nXIIts - v)+I_ext*R_nXIIts)/tau_nXIIts : volt ''')
      RA_nXIIts = NeuronGroup(N_RA_nXIIts, LIF_nXIIts, threshold='v>v_thr_nXIIts', reset='v=v_rest_nXIIts', method='exact', refractory=tau_refractory_nXIIts, dt=sim_dt)

      # Initiate the membrane potentials "v" at resting potential
      RA_int.v    = 'v_rest_int'
      RA_nXIIts.v = 'v_rest_nXIIts'

elif neuronal_model == 'Izhikevich':

      Izhikevich_model = Equations(""" dv/dt = ((0.04/ms/mV)*v**2 + (5/ms)*v + 140*mV/ms - u + I ) : volt
                                       du/dt = (a*(b*v - u)) : volt/second
                                       I : volt/second
                                       a : hertz
                                       b : hertz
                                       c : volt
                                       d : volt/second     """)

      # Neuronal Model for RA_int Neurons  # Inhibitory
      v_thr_nXIIts = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['v_thr_int']*mV
      RA_int       = NeuronGroup(N_RA_int, Izhikevich_model, threshold="v>=v_thr_nXIIts", reset="v=c; u+=d", method="rk4", refractory=tau_refractory_nXIIts, dt=sim_dt)

      # Neuronal Parameters for RA_int Neurons
      RA_int.a = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['a_int']/ms
      RA_int.b = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['b_int']/ms
      RA_int.c = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['c_int']*mV
      RA_int.d = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['d_int']*mV/ms
      RA_int.I = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['I_int']*mV/ms

      # Neuronal Model for RA_nXIIts Neurons # Exitatory, Tonically Active
      v_thr_nXIIts = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['v_thr_nXIIts']*mV
      RA_nXIIts    = NeuronGroup(N_RA_nXIIts, Izhikevich_model, threshold="v>=v_thr_nXIIts", reset="v=c; u+=d", method="rk4", refractory=tau_refractory_nXIIts, dt=sim_dt)

      # Neuronal Parameters for RA_nXIIts Neurons 
      RA_nXIIts.a = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['a_nXIIts']/ms
      RA_nXIIts.b = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['b_nXIIts']/ms
      RA_nXIIts.c = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['c_nXIIts']*mV
      RA_nXIIts.d = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['d_nXIIts']*mV/ms
      RA_nXIIts.I = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['I_nXIIts']*mV/ms

      # Initiate the membrane potentials "v" at resting potential and u at the corresponding value
      RA_int.v    = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['c_int']*mV
      RA_int.u    = parameters['Dynamic_Neuron_Parameters']['RA_int_Neuron_Parameters_Izhikevich']['u_start_int']*mV/ms
      RA_nXIIts.v = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['c_nXIIts']*mV
      RA_nXIIts.u = parameters['Dynamic_Neuron_Parameters']['RA_nXIIts_Neuron_Parameters_Izhikevich']['u_start_nXIIts']*mV/ms

else:
      raise ValueError('Neuronal model not defined! Please choose between LIF and Izhikevich')
      



#################
###  Synapses ###
#################

### Synaptic Parameters for HVC_RA and RA_nXIIts Neurons
init_weight_ex = parameters['Synaptic_Parameters']['Exitatory_Synaptic_Parameters']['init_weight_ex']*mV  
taupre_ex      = parameters['Synaptic_Parameters']['Exitatory_Synaptic_Parameters']['taupre_ex']*ms
taupost_ex     = parameters['Synaptic_Parameters']['Exitatory_Synaptic_Parameters']['taupost_ex']*ms
wmax_ex        = parameters['Synaptic_Parameters']['Exitatory_Synaptic_Parameters']['wmax_ex']*mV
apre_increment_ex  = float(parameters['Synaptic_Parameters']['Exitatory_Synaptic_Parameters']['apre_increment_ex']) * wmax_ex
apost_increment_ex = float(parameters['Synaptic_Parameters']['Exitatory_Synaptic_Parameters']['apost_increment_ex']) * wmax_ex

synapse_eqs_ex = ''' w : volt
                  dapre/dt  = -apre/taupre_ex   : volt (event-driven)
                  dapost/dt = -apost/taupost_ex : volt (event-driven) '''
stdp_pre_ex    = ''' v_post += w
                  apre += apre_increment_ex
                  w = clip(w + apost, 0*volt, wmax_ex) '''
stdp_post_ex   = ''' apost += apost_increment_ex
                  w = clip(w + apre, 0*volt, wmax_ex) '''

### Synaptic Model for RA_int Neurons
init_weight_in = parameters['Synaptic_Parameters']['Inhibitory_Synaptic_Parameters']['init_weight_in']*mV
taupre_in      = parameters['Synaptic_Parameters']['Inhibitory_Synaptic_Parameters']['taupre_in']*ms
taupost_in     = parameters['Synaptic_Parameters']['Inhibitory_Synaptic_Parameters']['taupost_in']*ms
wmax_in        = parameters['Synaptic_Parameters']['Inhibitory_Synaptic_Parameters']['wmax_in']*mV
apre_increment_in  = float(parameters['Synaptic_Parameters']['Inhibitory_Synaptic_Parameters']['apre_increment_in']) * wmax_in
apost_increment_in = float(parameters['Synaptic_Parameters']['Inhibitory_Synaptic_Parameters']['apost_increment_in']) * wmax_in

synapse_eqs_in = ''' w : volt
                  dapre/dt  = -apre/taupre_in   : volt (event-driven)
                  dapost/dt = -apost/taupost_in : volt (event-driven) '''
stdp_pre_in    = ''' v_post -= w
                  apre += apre_increment_in
                  w = clip(w + apost, 0*volt, wmax_in) '''
stdp_post_in   = ''' apost += apost_increment_in
                  w = clip(w + apre, 0*volt, wmax_in) '''

### Synapse Models ###

S_HVC_RA_RA_nXIIts    = Synapses(HVC_RA, RA_nXIIts, model=synapse_eqs_ex, on_pre=stdp_pre_ex, on_post=stdp_post_ex, dt=sim_dt)     # exitatory

S_HVC_RA_RA_int       = Synapses(HVC_RA, RA_int, model=synapse_eqs_ex, on_pre=stdp_pre_ex, on_post=stdp_post_ex, dt=sim_dt)        # exitatory

S_RA_nXIIts_RA_nXIIts = Synapses(RA_nXIIts, RA_nXIIts, model=synapse_eqs_ex, on_pre=stdp_pre_ex, on_post=stdp_post_ex, dt=sim_dt)  # exitatory

S_RA_nXIIts_RA_int    = Synapses(RA_nXIIts, RA_int, model=synapse_eqs_ex, on_pre=stdp_pre_ex, on_post=stdp_post_ex, dt=sim_dt)     # exitatory

S_RA_int_RA_nXIIts    = Synapses(RA_int, RA_nXIIts, model=synapse_eqs_in, on_pre=stdp_pre_in, on_post=stdp_post_in, dt=sim_dt)     # inhibitory

S_RA_int_RA_int       = Synapses(RA_int, RA_int, model=synapse_eqs_in, on_pre=stdp_pre_in, on_post=stdp_post_in, dt=sim_dt)        # inhibitory

### Connectivity ###

p_HVC_RA_RA_nXIIts      = parameters['Connectivity_Parameters']['p_HVC_RA_RA_nXIIts']
S_HVC_RA_RA_nXIIts.connect(p = p_HVC_RA_RA_nXIIts)
S_HVC_RA_RA_nXIIts.w = init_weight_ex

p_HVC_RA_RA_int         = parameters['Connectivity_Parameters']['p_HVC_RA_RA_int']
S_HVC_RA_RA_int.connect(p = p_HVC_RA_RA_int)
S_HVC_RA_RA_int.w       = init_weight_in

p_RA_nXIIts_RA_nXIIts   = parameters['Connectivity_Parameters']['p_RA_nXIIts_RA_nXIIts']
S_RA_nXIIts_RA_nXIIts.connect(p = p_RA_nXIIts_RA_nXIIts)
S_RA_nXIIts_RA_nXIIts.w = init_weight_ex

p_RA_nXIIts_RA_int      = parameters['Connectivity_Parameters']['p_RA_nXIIts_RA_int']
S_RA_nXIIts_RA_int.connect(p = p_RA_nXIIts_RA_int)
S_RA_nXIIts_RA_int.w    = init_weight_in

p_RA_int_RA_nXIIts      = parameters['Connectivity_Parameters']['p_RA_int_RA_nXIIts']
S_RA_int_RA_nXIIts.connect(p = p_RA_int_RA_nXIIts)
S_RA_int_RA_nXIIts.w    = init_weight_ex

p_RA_int_RA_int         = parameters['Connectivity_Parameters']['p_RA_int_RA_int']
S_RA_int_RA_int.connect(p = p_RA_int_RA_int)
S_RA_int_RA_int.w       = init_weight_in


######################
### Run Simulation ###
######################

### Set Monitors ###
dt_v = parameters['Simulation_Parameters']['dt_v']*Timescale  # Voltage Monitor Clock
dt_r = parameters['Simulation_Parameters']['dt_r']*Timescale  # Rate Monitor Clock
dt_w = parameters['Simulation_Parameters']['dt_w']*Timescale  # Weight Monitor Clock

if saveSpikes:
      SpikeMon_nXIIts = SpikeMonitor(RA_nXIIts, record=True)
      SpikeMon_int =    SpikeMonitor(RA_int,    record=True)
      
if saveRates:
      RateMon_nXIIts = PopulationRateMonitor(RA_nXIIts)
      RateMon_int =    PopulationRateMonitor(RA_int)

if saveVoltages:
      percentage_neurons = parameters['Simulation_Parameters']['pc_n']/100
      random_indices_nXIIts = np.random.choice(np.arange(N_RA_nXIIts), int(np.ceil(N_RA_nXIIts*percentage_neurons)), replace=False)
      random_indices_int    = np.random.choice(np.arange(N_RA_int),    int(np.ceil(N_RA_int*percentage_neurons)),    replace=False)
      VoltMon_nXIIts = StateMonitor(RA_nXIIts, 'v', record=random_indices_nXIIts, dt=dt_v)
      VoltMon_int =    StateMonitor(RA_int,    'v', record=random_indices_int,    dt=dt_v)

if saveWeights:
      # Saving initial weights
      initial_W_HVC_RA_RA_nXIIts      = S_HVC_RA_RA_nXIIts.w[:].copy()
      initial_W_HVC_RA_RA_nXIIts_i    = S_HVC_RA_RA_nXIIts.i[:].copy()
      initial_W_HVC_RA_RA_nXIIts_j    = S_HVC_RA_RA_nXIIts.j[:].copy()
      initial_W_HVC_RA_RA_int         = S_HVC_RA_RA_int.w[:].copy()
      initial_W_HVC_RA_RA_int_i       = S_HVC_RA_RA_int.i[:].copy()
      initial_W_HVC_RA_RA_int_j       = S_HVC_RA_RA_int.j[:].copy()
      initial_W_RA_nXIIts_RA_nXIIts   = S_RA_nXIIts_RA_nXIIts.w[:].copy()
      initial_W_RA_nXIIts_RA_nXIIts_i = S_RA_nXIIts_RA_nXIIts.i[:].copy()
      initial_W_RA_nXIIts_RA_nXIIts_j = S_RA_nXIIts_RA_nXIIts.j[:].copy()
      initial_W_RA_nXIIts_RA_int      = S_RA_nXIIts_RA_int.w[:].copy()
      initial_W_RA_nXIIts_RA_int_i    = S_RA_nXIIts_RA_int.i[:].copy()
      initial_W_RA_nXIIts_RA_int_j    = S_RA_nXIIts_RA_int.j[:].copy()
      initial_W_RA_int_RA_nXIIts      = S_RA_int_RA_nXIIts.w[:].copy()
      initial_W_RA_int_RA_nXIIts_i    = S_RA_int_RA_nXIIts.i[:].copy()
      initial_W_RA_int_RA_nXIIts_j    = S_RA_int_RA_nXIIts.j[:].copy()
      initial_W_RA_int_RA_int         = S_RA_int_RA_int.w[:].copy()
      initial_W_RA_int_RA_int_i       = S_RA_int_RA_int.i[:].copy()
      initial_W_RA_int_RA_int_j       = S_RA_int_RA_int.j[:].copy()

      # Setting up munitor for % of synapses in each simulation
      percentage_synapses = parameters['Simulation_Parameters']['pc_w']/100

      number_initalized_HVC_RA_RA_nXIIts    = len(S_HVC_RA_RA_nXIIts.w)
      random_indices_HVC_RA_RA_nXIIts       = np.random.choice(np.arange(number_initalized_HVC_RA_RA_nXIIts), int(np.ceil(percentage_synapses*number_initalized_HVC_RA_RA_nXIIts)), replace=False)
      W_HVC_RA_RA_nXIIts                    = StateMonitor(S_HVC_RA_RA_nXIIts, ('i','j','w'), record=random_indices_HVC_RA_RA_nXIIts, dt=dt_w)
  
      number_initalized_HVC_RA_RA_int       = len(S_HVC_RA_RA_int.w)
      random_indices_HVC_RA_RA_int          = np.random.choice(np.arange(number_initalized_HVC_RA_RA_int), int(np.ceil(percentage_synapses*number_initalized_HVC_RA_RA_int)), replace=False)
      W_HVC_RA_RA_int                       = StateMonitor(S_HVC_RA_RA_int, ('i','j','w'), record=random_indices_HVC_RA_RA_int, dt=dt_w)

      number_initalized_RA_nXIIts_RA_nXIIts = len(S_RA_nXIIts_RA_nXIIts.w)
      random_indices_RA_nXIIts_RA_nXIIts    = np.random.choice(np.arange(number_initalized_RA_nXIIts_RA_nXIIts), int(np.ceil(percentage_synapses*number_initalized_RA_nXIIts_RA_nXIIts)), replace=False)
      W_RA_nXIIts_RA_nXIIts                 = StateMonitor(S_RA_nXIIts_RA_nXIIts, ('i','j','w'), record=random_indices_RA_nXIIts_RA_nXIIts, dt=dt_w)

      number_initalized_RA_nXIIts_RA_int    = len(S_RA_nXIIts_RA_int.w)
      random_indices_RA_nXIIts_RA_int       = np.random.choice(np.arange(number_initalized_RA_nXIIts_RA_int), int(np.ceil(percentage_synapses*number_initalized_RA_nXIIts_RA_int)), replace=False)
      W_RA_nXIIts_RA_int                    = StateMonitor(S_RA_nXIIts_RA_int, ('i','j','w'), record=random_indices_RA_nXIIts_RA_int, dt=dt_w)

      number_initalized_RA_int_RA_nXIIts    = len(S_RA_int_RA_nXIIts.w)
      random_indices_RA_int_RA_nXIIts       = np.random.choice(np.arange(number_initalized_RA_int_RA_nXIIts), int(np.ceil(percentage_synapses*number_initalized_RA_int_RA_nXIIts)), replace=False)
      W_RA_int_RA_nXIIts                    = StateMonitor(S_RA_int_RA_nXIIts, ('i','j','w'), record=random_indices_RA_int_RA_nXIIts, dt=dt_w)

      number_initalized_RA_int_RA_int       = len(S_RA_int_RA_int.w)
      random_indices_RA_int_RA_int          = np.random.choice(np.arange(number_initalized_RA_int_RA_int), int(np.ceil(percentage_synapses*number_initalized_RA_int_RA_int)), replace=False)
      W_RA_int_RA_int                       = StateMonitor(S_RA_int_RA_int, ('i','j','w'), record=random_indices_RA_int_RA_int, dt=dt_w)

### Run Simulation ###
if report:
      start = datetime.now()
      run(Simtime*Timescale, report='text')
else:
      start = datetime.now()
      run(Simtime*Timescale)

# Save final Weights
if saveWeights:
      final_HVC_RA_RA_nXIIts      = S_HVC_RA_RA_nXIIts.w[:].copy()
      final_HVC_RA_RA_int         = S_HVC_RA_RA_int.w[:].copy()
      final_RA_nXIIts_RA_nXIIts   = S_RA_nXIIts_RA_nXIIts.w[:].copy()
      final_RA_nXIIts_RA_int      = S_RA_nXIIts_RA_int.w[:].copy()
      final_RA_int_RA_nXIIts      = S_RA_int_RA_nXIIts.w[:].copy()
      final_RA_int_RA_int         = S_RA_int_RA_int.w[:].copy()

### Write simulation output ###

if report:
      print('\nSimulation {} took {:f} s'.format(slurm_job_id,(datetime.now()-start).total_seconds()))
      print(f'\nWriting simulation output to {so_path}')

start = datetime.now()

if saveProtocol:
      # Defining the 'to be' json string with information about the process
      simulation_protocol = {
            "Process Information" : 
                  {"SLURM Job ID"          : slurm_job_id,
                        "Time"             : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Scale"            : scale/100,
                        "Runtime"          : f'{(datetime.now() - start).total_seconds() * 10**3} ms',
                        "Total Simulation Time"  : f'{Simtime} {str(Timescale)}'}}
            
      # Addind the Clock values of the Monitors (if they are active)
      protocol_monitors = {
            "Monitor" : 
                  {"Temporal Clock" : str(sim_dt/Timescale)}}
      if saveVoltages:
            protocol_monitors["Monitor"].update({"Voltage Clock" : str(dt_v/Timescale)})
      if saveWeights:
            protocol_monitors["Monitor"].update({"Weight Clock"  : str(dt_w/Timescale)})
      simulation_protocol.update(protocol_monitors)
      
      # Physical Dimensions used in simulation
      protocol_dimensions = {
            "Dimensions" : 
                  {"Timescale" : str(Timescale)}}  
               
      if saveVoltages:
            protocol_dimensions["Dimensions"].update({"Voltage Dim" : str(get_dimensions(VoltMon_nXIIts.v[:]))})
      
      if saveWeights:
            protocol_dimensions["Dimensions"].update({"Weights Dim" : str(get_dimensions(W_HVC_RA_RA_nXIIts.w[:]))})
      
      simulation_protocol.update(protocol_dimensions)  

      if saveVoltages:
            protocol_action_potential = {
                  "Properties of Action Potential" : 
                        {"Peak Amplitude of Action Potential RA_nXIIts" : 
                              {"Minimal Value" : f'{np.min(VoltMon_nXIIts.v[:])} {get_dimensions(VoltMon_nXIIts.v)}',
                                    "Average Value" : f'({np.mean(VoltMon_nXIIts.v[:])} ± {np.std(VoltMon_nXIIts.v[:])}) {get_dimensions(VoltMon_nXIIts.v)}',
                                    "Maximal Value" : f'{np.max(VoltMon_nXIIts.v[:])} {get_dimensions(VoltMon_nXIIts.v)}'},
                        "Peak Amplitude of Action Potential RA_int" : 
                              {"Minimal Value" : f'{np.min(VoltMon_int.v[:])} {get_dimensions(VoltMon_int.v)}',
                                    "Average Value" : f'({np.mean(VoltMon_int.v[:])} ± {np.std(VoltMon_int.v[:])}) {get_dimensions(VoltMon_int.v)}',
                                    "Maximal Value" : f'{np.max(VoltMon_int.v[:])} {get_dimensions(VoltMon_int.v)}'}}}
            simulation_protocol.update(protocol_action_potential)

      if saveRates:
            protocol_rates = {
                  "Firing Rate of Populations" :
                        {"Firing Rate RA_nXIIts" : 
                              {"Minimal Value" : f'{np.min(RateMon_nXIIts.rate_[:])} {get_dimensions(RateMon_int.rate)}',
                                    "Average Value" : f'({np.mean(RateMon_nXIIts.rate_[:])} ± {np.std(RateMon_nXIIts.rate_[:])}) {get_dimensions(RateMon_nXIIts.rate)}',
                                    "Maximal Value" : f'{np.max(RateMon_nXIIts.rate_[:])} {get_dimensions(RateMon_nXIIts.rate)}'},
                        "Firing Rate RA_int" : 
                              {"Minimal Value" : f'{np.min(RateMon_int.rate_[:])} {get_dimensions(RateMon_int.rate)}',
                                    "Average Value" : f'({np.mean(RateMon_int.rate_[:])} ± {np.std(RateMon_int.rate_[:])}) {get_dimensions(RateMon_int.rate)}',
                                    "Maximal Value" : f'{np.max(RateMon_int.rate_[:])} {get_dimensions(RateMon_int.rate)}'}},
                  "Frequency Adaptation Rate of Populations" :
                        {"Firing Rate RA_nXIIts" : 
                              {"Minimal Value" : np.min(np.diff(RateMon_nXIIts.rate_[:])/np.diff(RateMon_nXIIts.t_[:])),
                                    "Average Value" : f'({np.mean(np.diff(RateMon_nXIIts.rate_[:])/np.diff(RateMon_nXIIts.t_[:]))} ± {np.std(np.diff(RateMon_nXIIts.rate_[:])/np.diff(RateMon_nXIIts.t_[:]))})',
                                    "Maximal Value" : np.max(np.diff(RateMon_nXIIts.rate_[:])/np.diff(RateMon_nXIIts.t_[:]))},
                        "Firing Rate RA_int" : 
                              {"Minimal Value" : np.min(np.diff(RateMon_int.rate_[:])/np.diff(RateMon_int.t_[:])),
                                    "Average Value" : f'({np.mean(np.diff(RateMon_int.rate_[:])/np.diff(RateMon_int.t_[:]))} ± {np.std(np.diff(RateMon_int.rate_[:])/np.diff(RateMon_int.t_[:]))})',
                                    "Maximal Value" : np.max(np.diff(RateMon_int.rate_[:])/np.diff(RateMon_int.t_[:]))}}}
            simulation_protocol.update(protocol_rates)


      with open(Path('simulation_output/Simulation {}/Simulation_Protocol.json'.format(slurm_job_id)), 'w') as file:
            json.dump(simulation_protocol, file, indent=2)

if saveWeights:
      File_Weights = h5py.File(Path(f'simulation_output/Simulation {slurm_job_id}/Weights.hdf5'), 'w')
      
      # Saving initial weights
      File_Weights.create_dataset("Initial_Weights_HVC_RA_RA_nXIIts",      data=initial_W_HVC_RA_RA_nXIIts,      dtype='float32')
      File_Weights.create_dataset("Initial_Weights_HVC_RA_RA_nXIIts.i",    data=initial_W_HVC_RA_RA_nXIIts_i,    dtype='float32')
      File_Weights.create_dataset("Initial_Weights_HVC_RA_RA_nXIIts.j",    data=initial_W_HVC_RA_RA_nXIIts_j,    dtype='float32')

      File_Weights.create_dataset("Initial_Weights_HVC_RA_RA_int",         data=initial_W_HVC_RA_RA_int,         dtype='float32')
      File_Weights.create_dataset("Initial_Weights_HVC_RA_RA_int.i",       data=initial_W_HVC_RA_RA_int_i,       dtype='float32')
      File_Weights.create_dataset("Initial_Weights_HVC_RA_RA_int.j",       data=initial_W_HVC_RA_RA_int_j,       dtype='float32')

      File_Weights.create_dataset("Initial_Weights_RA_nXIIts_RA_nXIIts",   data=initial_W_RA_nXIIts_RA_nXIIts,   dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_nXIIts_RA_nXIIts.i", data=initial_W_RA_nXIIts_RA_nXIIts_i, dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_nXIIts_RA_nXIIts.j", data=initial_W_RA_nXIIts_RA_nXIIts_j, dtype='float32')

      File_Weights.create_dataset("Initial_Weights_RA_nXIIts_RA_int",      data=initial_W_RA_nXIIts_RA_int,      dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_nXIIts_RA_int.i",    data=initial_W_RA_nXIIts_RA_int_i,    dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_nXIIts_RA_int.j",    data=initial_W_RA_nXIIts_RA_int_j,    dtype='float32')

      File_Weights.create_dataset("Initial_Weights_RA_int_RA_nXIIts",      data=initial_W_RA_int_RA_nXIIts,      dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_int_RA_nXIIts.i",    data=initial_W_RA_int_RA_nXIIts_i,    dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_int_RA_nXIIts.j",    data=initial_W_RA_int_RA_nXIIts_j,    dtype='float32')

      File_Weights.create_dataset("Initial_Weights_RA_int_RA_int",         data=initial_W_RA_int_RA_int,         dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_int_RA_int.i",       data=initial_W_RA_int_RA_int_i,       dtype='float32')
      File_Weights.create_dataset("Initial_Weights_RA_int_RA_int.j",       data=initial_W_RA_int_RA_int_j,       dtype='float32')

      # Saving final weights
      File_Weights.create_dataset("Weights_Time", data=W_HVC_RA_RA_nXIIts.t[:], dtype='float32')

      File_Weights.create_dataset("Weights_HVC_RA_RA_nXIIts",      data=W_HVC_RA_RA_nXIIts.w[:],    dtype='float32')
      File_Weights.create_dataset("Weights_HVC_RA_RA_nXIIts.i",    data=W_HVC_RA_RA_nXIIts.i[:],    dtype='float32')
      File_Weights.create_dataset("Weights_HVC_RA_RA_nXIIts.j",    data=W_HVC_RA_RA_nXIIts.j[:],    dtype='float32')

      File_Weights.create_dataset("Weights_HVC_RA_RA_int",         data=W_HVC_RA_RA_int.w[:],       dtype='float32')
      File_Weights.create_dataset("Weights_HVC_RA_RA_int.i",       data=W_HVC_RA_RA_int.i[:],       dtype='float32')
      File_Weights.create_dataset("Weights_HVC_RA_RA_int.j",       data=W_HVC_RA_RA_int.j[:],       dtype='float32')

      File_Weights.create_dataset("Weights_RA_nXIIts_RA_nXIIts",   data=W_RA_nXIIts_RA_nXIIts.w[:], dtype='float32')
      File_Weights.create_dataset("Weights_RA_nXIIts_RA_nXIIts.i", data=W_RA_nXIIts_RA_nXIIts.i[:], dtype='float32')
      File_Weights.create_dataset("Weights_RA_nXIIts_RA_nXIIts.j", data=W_RA_nXIIts_RA_nXIIts.j[:], dtype='float32')

      File_Weights.create_dataset("Weights_RA_nXIIts_RA_int",      data=W_RA_nXIIts_RA_int.w[:],    dtype='float32')
      File_Weights.create_dataset("Weights_RA_nXIIts_RA_int.i",    data=W_RA_nXIIts_RA_int.i[:],    dtype='float32')
      File_Weights.create_dataset("Weights_RA_nXIIts_RA_int.j",    data=W_RA_nXIIts_RA_int.j[:],    dtype='float32')

      File_Weights.create_dataset("Weights_RA_int_RA_nXIIts",      data=W_RA_int_RA_nXIIts.w[:],    dtype='float32')
      File_Weights.create_dataset("Weights_RA_int_RA_nXIIts.i",    data=W_RA_int_RA_nXIIts.i[:],    dtype='float32')
      File_Weights.create_dataset("Weights_RA_int_RA_nXIIts.j",    data=W_RA_int_RA_nXIIts.j[:],    dtype='float32')

      File_Weights.create_dataset("Weights_RA_int_RA_int",         data=W_RA_int_RA_int.w[:],       dtype='float32')
      File_Weights.create_dataset("Weights_RA_int_RA_int.i",       data=W_RA_int_RA_int.i[:],       dtype='float32')
      File_Weights.create_dataset("Weights_RA_int_RA_int.j",       data=W_RA_int_RA_int.j[:],       dtype='float32')

      # Saving final weights (Note: Indices are same as initial weights)
      File_Weights.create_dataset("Final_Weights_HVC_RA_RA_nXIIts",      data=final_HVC_RA_RA_nXIIts,      dtype='float32')
      File_Weights.create_dataset("Final_Weights_HVC_RA_RA_int",         data=final_HVC_RA_RA_int,         dtype='float32')
      File_Weights.create_dataset("Final_Weights_RA_nXIIts_RA_nXIIts",   data=final_RA_nXIIts_RA_nXIIts,   dtype='float32')
      File_Weights.create_dataset("Final_Weights_RA_nXIIts_RA_int",      data=final_RA_nXIIts_RA_int,      dtype='float32')
      File_Weights.create_dataset("Final_Weights_RA_int_RA_nXIIts",      data=final_RA_int_RA_nXIIts,      dtype='float32')
      File_Weights.create_dataset("Final_Weights_RA_int_RA_int",         data=final_RA_int_RA_int,         dtype='float32')

      File_Weights.close()

if saveSpikes:
      np.savetxt(Path('simulation_output/Simulation {}/Spiketimes_HVC_RA.txt'.format(slurm_job_id)),          np.vstack((indices, hvc_spiketimes, np.full(len(indices), str(Timescale)))).T, fmt=['%5s', '%.20s', '%-5s'], delimiter=' : ')
      np.savetxt(Path('simulation_output/Simulation {}/Spiketimes_RA_nXIIts.txt'.format(slurm_job_id)),       np.vstack((SpikeMon_nXIIts.i[:], SpikeMon_nXIIts.t_[:], np.full(len(SpikeMon_nXIIts.i[:]), get_dimensions(SpikeMon_nXIIts.t)))).T, fmt=['%i', '%.8f', '%-4s'], delimiter=' : ')
      np.savetxt(Path('simulation_output/Simulation {}/Spiketimes_RA_Interneutons.txt'.format(slurm_job_id)), np.vstack((SpikeMon_int.i[:],    SpikeMon_int.t_[:],    np.full(len(SpikeMon_int.i[:]),    get_dimensions(SpikeMon_int.t)))).T,    fmt=['%i', '%.8f', '%-4s'], delimiter=' : ')

if saveRates:
      File_Rates = h5py.File(Path(f'simulation_output/Simulation {slurm_job_id}/Rates.hdf5'), 'w')
      File_Rates.create_dataset("Rates_time",   data=RateMon_nXIIts.t_[:],    dtype='float32')
      File_Rates.create_dataset("Rates_nXIIts", data=RateMon_nXIIts.rate_[:], dtype='float32')
      File_Rates.create_dataset("Rates_int",    data=RateMon_int.rate_[:],    dtype='float32')
      File_Rates.close()

if saveVoltages:

      File_Voltages = h5py.File(Path(f'simulation_output/Simulation {slurm_job_id}/Voltages.hdf5'), 'w')
      File_Voltages.create_dataset("Voltages_Time",              data=VoltMon_nXIIts.t_[:],  dtype='float32')
      File_Voltages.create_dataset("Voltages_RA_nXIIts_Indices", data=random_indices_nXIIts, dtype='int32')
      File_Voltages.create_dataset("Voltages_RA_int_Indices",    data=random_indices_int,    dtype='int32')
      File_Voltages.create_dataset("Voltages_RA_nXIIts",         data=VoltMon_nXIIts.v_[:],  dtype='float32')
      File_Voltages.create_dataset("Voltages_RA_Interneutons",   data=VoltMon_int.v_[:],     dtype='float32')
      File_Voltages.close()

if report:
      print('\nSaving the data took {:f} s'.format((datetime.now()-start).total_seconds()))
      print('\nFinished!')
