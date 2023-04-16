"""
This script fits a neural network to the output of an auditory filterbank.
"""
import datetime
import h5py
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import sys
import time
import torch

import student
import teacher

# Collect command-line arguments
sav_dir = sys.argv[1]
domain = sys.argv[2]
arch = sys.argv[3]
init_id = sys.argv[4]

# Print header
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print(__doc__ + "\n")
print("Command-line arguments:\n" + "\n".join(sys.argv[1:]) + "\n")

# Print version numbers
for module in [h5py, np, pd, pl, torch]:
    print("{} version: {:s}".format(module.__name__, module.__version__))
print("")
sys.stdout.flush()

# Create model directory
model_dir = os.path.join(sav_dir, "models", domain)
model_sav_name = "_".join([domain, arch, "init-" + str(init_id)])
model_sav_path = os.path.join(model_dir, model_sav_name)
os.makedirs(model_sav_path, exist_ok=True)
pred_path = os.path.join(model_sav_path, "predictions.npy")

# Initialize dataset
dataset = teacher.SpectrogramDataModule(sav_dir=sav_dir, domain=domain)
print(str(datetime.datetime.now()) + " Finished initializing dataset")

# Launch computation
if __name__ == "__main__":
    print("Current device: ", torch.cuda.get_device_name(0))
    torch.multiprocessing.set_start_method('spawn')

    # Initialize model
    if arch == "convnet":
        model = student.init_convnet(domain=domain)
    elif arch == "LEAF":
        model = student.init_leaf(domain=domain)
    elif arch == "MuReNN":
        model = student.init_murenn(domain=domain)
    print(str(datetime.datetime.now()) + " Finished initializing model")

    # Setup checkpoints and Tensorboard logger

    # Setup trainer

    # Train

    # Test

# Print elapsed time.
print(str(datetime.datetime.now()) + " Success.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.0
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(
    elapsed_hours, elapsed_minutes, elapsed_seconds
)
print("Total elapsed time: " + elapsed_str + ".")