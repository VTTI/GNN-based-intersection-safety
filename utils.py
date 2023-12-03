import mlflow
import requests
import torch
import random
import numpy as np
import json
import time

mlflow.set_tracking_uri("http://localhost:5000")

def get_model_predictions(payload):
    """
    Get model predictions  
    ENDPOINT = Calls an endpoint to get the predictions
    REGISTRY = Loads model from registry and predicts
    MOCKED = Randomly generated prediction
    """
    option="MOCKED"

    if option == "ENDPOINT":
        # Currently not supported for multi-input models
        DEPLOYED_ENDPOINT = "http://127.0.0.1:5001/invocations"
        headers = {"Content-Type":"application/json"}
        prediction = requests.post(url=DEPLOYED_ENDPOINT, 
                                   data={"inputs": {
                                            "x": payload["x"].numpy(),
                                            "edge_attr": payload["edge_attr"].numpy(),
                                            "edge_index": payload["edge_index"].numpy().astype(np.int32),
                                            "batch_index": np.expand_dims(payload["batch_index"].numpy().astype(np.int32), axis=1)
                                        }}, headers=headers)
        prediction = json.loads(prediction.content.decode("utf-8")) 
    
    if option == "REGISTRY":
        # Currently not supported for multi-input models
        model_name = "GraphTransformer"
        model_version = "2"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")


        prediction = model.predict({
            "x": payload["x"].numpy(),
            "edge_attr": payload["edge_attr"].numpy(),
            "edge_index": payload["edge_index"].numpy().astype(np.int32),
            "batch_index": np.expand_dims(payload["batch_index"].numpy().astype(np.int32), axis=1)
        })
        
    if option == "MOCKED":
        # Fake API call
        time.sleep(2)
        prediction = random.choice([0,1])

    return prediction

    





