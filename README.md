This repository contains both the code for generating our new biological cellular system benchmark dataset and the code of MS-GODE.

# Bio CDL Benchmark
The code for generating the configuration files for the simulations is contained in ```VCell_config_gen.py```. We have restricted the model coefficients to a specific range to ensure the simulations remain stable.
After generating the configuration files, they should be uploaded into the VCell platform for generating the simulations. The specific steps are listed below.

1. Download the VCell client through 'https://vcell.org/run-vcell-software', create a free account and log in.
2. Select the 'BioModel' tab, in the 'Search' box, locate and load the models 'Rule-based_egfr_tutorial' or 'rule-based_Ran_transport' under the 'Tutorials' directory for the EGFR model or Ran model, respectively.
<img src="https://github.com/QueuQ/MS-GODE/blob/master/vcell_models.png"/> 

3. In the upper left part of the window, click the 'Applications', then click the 'Simulations' under the 'network_determ' tab.
<img src="https://github.com/QueuQ/MS-GODE/blob/master/vcell_simulations.png"/>
4. In the main window on the upper right, select the 'Simulations' tab and click the first icon under  the 'Simulations' tab to create a new simulation.
<img src="https://github.com/QueuQ/MS-GODE/blob/master/vcell_batch_simulations.png"/>
5. Click to select the created simulation, then click the icon with a blue arrow and a gear to load the previously generated simulations.
6. After the simulations are done, select all the simulations and click the icon with a green arrow and a gear to export all the simulation results into a specified folder.
7. Specify the ```data_store_path``` and run ```generate_dataset.py``` to obtain the data files that can be loaded and used by the MS-GODE model.

# MS-GODE

MS-GODE is a framework for learning continually over system sequences with diverse dynamics.

This implementation of MS-GODE is based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) API.


## Setup

The mdoel implementation is based on the following packages:

* [Python 3.6.10](https://www.python.org/)

- [Pytorch 1.4.0](https://pytorch.org/)

- [pytorch_geometric 1.4.3](https://pytorch-geometric.readthedocs.io/)

  - torch-cluster==1.5.3
  - torch-scatter==2.0.4
  - torch-sparse==0.6.1

- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

- [numpy 1.16.1](https://numpy.org/)



## Example command for running the experiments with MS-GODE:

```bash
python run_models.py --cut_num 20 \
--nepos 20 \
--device 5 \
--n_iters_to_viz 10 \
--thresholding 'fast' \
--mask True \
--dropout_mask '0.0' \
--batch-size 10 \
--mode "ex" \
--normalizeVCellfeat 'universal' \
--system 'VCell' \
--repeats 5 \
--overwrite_results 'False' \
--fix_random_seed 'False' \
--save_results True
```

## Example scripts for data generation of cellular/physics systems:


```bash
 python ./data/generate_dataset.py --simulation VCell \
  --num-train 3000 \
  --num-test 3000 \
  --n-balls 5
```

The generation of cellular simulation data requires first obtaining simulation data from VCell platform, which is introduced in the Appendix A.1.2 of the submission.

```bash
 python ./data/generate_dataset.py --simulation simulation \
  --num-train 3000 \
  --num-test 3000 \
  --n-balls 5
```


## Using custom data

Please format the custom data into the format of the example data. Specifically, 

1. loc.npy contains the locations of the system objects in the temporal order. The shape of the variable should be ```[number_of_trajectories, number_of_objects, (number_of_timestamps, dim)]```. 'dim' refers to the feature dimension. E.g. for 2D spatial trajectories, dim=2.
2. times.npy contains the timestamps of the system objects. The shape of the variable should be ```[number_of_trajectories, number_of_timestamps]```.
3. edge.npy contains the graph structure of the system. If the structure is unknown, fully connected graph structure could be used. The shape of the variable should be ```[number_of_trajectories, number_of_objects, number_of_objects]```.
4. (optional) vel.npy contains the first derivative of the trajectories, e.g. velocity. This file is optional. The shape of the variable should be ```[number_of_trajectories, number_of_objects, (number_of_timestamps, dim)]```. 

