import argparse
import os

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
args = parser.parse_args()
bpath = args.exp_directory
data_dir = args.data_directory

# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)



