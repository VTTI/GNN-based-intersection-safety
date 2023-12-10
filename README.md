## Overview
This repository contains code for graph-level predictions on intersection data, represented in the form of graphs. The code expects your dataset to be formatted in a specific structure for proper processing.


### Dependencies
`git clone https://github.com/VTTI/GNN-based-intersection-safety.git`

`cd driver-secondary-action-recognition`

`docker build . -t gnn-safety`

`docker run -it --rm --runtime=nvidia -v {{dataPath}}:/data gnn-safety /bin/bash`


### Dataset Structure
Ensure your dataset follows the structure outlined below:

1. Individual folders named `train` and `test`, each containing a subfolder called `raw`.
2. Inside the `raw` folders, there should be:
   - Folder named `jsongraphs` containing JSON files representing graphs.
   - Text file named `jsonfiles.txt` listing the names of each JSON file in the jsongraphs folder.
   - Labels file named `labels.npy`.


### Code Files
`dataset_featurizer.py`: This file is responsible for extracting node features, edge features, and the adjacency matrix from the JSON graphs corresponding to each frame of the video.

`model.py`: This file contains details about the model architecture. Modify it according to your specific requirements. Note that not all models support both node and edge features. Refer to [PyTorch Geometric GNN Cheat Sheet](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html) to choose an appropriate model based on your data characteristics.
