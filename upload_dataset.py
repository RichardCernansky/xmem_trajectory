from azureml.core import Workspace, Datastore

# Connect to workspace
ws = Workspace.from_config(path="data/configs/azure_config.json")

# Get the datastore
datastore = Datastore.get(ws, "workspaceblobstore")  # Change if using a different datastore
# Define target datastore path
datastore_target_path = "datasets/xcernansky_nuscenes"

# Upload local folder to the blob storage
datastore.upload(
    src_dir="",
    target_path=datastore_target_path,
    overwrite=True,
    show_progress=True
)
