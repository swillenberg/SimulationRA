# Spiking Neural Network Simulation of Nucleus RA

This project implements a computational model of the robust nucleus of the arcopallium (RA) in the songbird brain, focusing on synaptic plasticity and neural dynamics. The simulation uses fixed $HVC_{RA}$​ spiking patterns modeled after measurements by Hahnloser et al. (2002) and Fee et al. (2004) [1,2]. Since the goal of the simulation is only to approximate the true connectivity, the anterior forebrain pathway for learning is omitted.

## Overview

The simulation models the interaction between three different neural populations:
- RA-projecting HVC neurons (exitatory, fixed pattern)
- Interneurons in RA (inhibitory)
- nXIIts-projecting RA neurons (exitatory)

The model uses spiking neural networks with:
- Linear Integrate-and-Fire (LIF) or Izhikevich neuron models
- Spike-Timing Dependent Plasticity (STDP)
- Configurable network connectivity and parameters

## Installation

1. Clone the repository:
```
git clone [repository-url]
```
2. Install dependencies:
```
pip install -r requirements.txt
```
Required packages:
- brian2 (neural simulation)
- h5py (data storage)
- pyyaml (configuration)
- numpy (numerical computations)

## Usage

1. *Optional:* Use the Jupyter notebook `ModelTuning.ipynb` to find fitting parameters for the model.
2. Configure simulation parameters in `Parameters.yaml` 
3. Run the simulation:
```
python RASimulation.py
```
4. Analyze results of the simulation using the Jupyter notebook `Evaluation.ipynb`.

## Configuration

Key parameters can be configured in `Parameters.yaml`:

- Simulation settings (scale, output options)
- Neural parameters (thresholds, time constants)
- Synaptic parameters (weights, STDP)
- Network connectivity

## Output

Simulation results are saved in `simulation_output/` including:
- Spike times
- Membrane potentials
- Synaptic weights
- Population rates

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Bibliography
[1] : Hahnloser, Richard H. R., Alexay A. Kozhevnikov, and Michale S. Fee (2002). “An ultra-sparse code underlies the generation of neural sequences in a songbird.” In: Nature 419.6902, pp. 65–70. doi: [10.1038/nature00974](htpps://doi.org/10.1038/nature00974).
[2] : Fee, Michale S., Alexay A. Kozhevnikov, and Richard H. R. Hahnloser (2004). “Neural mechanisms of vocal sequence generation in the songbird.” In: Annals of the New York Academy of Sciences 1016, pp. 153–170. issn: 0077-8923. doi: [10.1196/annals.1298.022](https://doi.org/10.1196/annals.1298.022).
