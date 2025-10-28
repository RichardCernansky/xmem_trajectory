import json
import pickle

def open_config(path: str):
    with open(path) as f:
        config = json.load(f)
    return config

def open_index(path: str):
    with open(path, "rb") as f:
        rows = pickle.load(f)
    return rows

   