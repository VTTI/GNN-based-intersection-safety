Graph-level predictions for intersection represented in the form of a graph.

The file `dataset_featurizer.py` obtains the node features, edge features, and adjacency matrix from the jsongraphs corresponding to each frame of the video.
The file `model.py` has the details regarding the model architecture and can be modified as per the required model and layers. (Note: Not all models allow data where you can have bode node and edge features, so choose the appropriate one from https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html)
 
