# Basic Usage

##  Setup

### 1. Install Python 3.9

ETSim requires [`Python 3.9.x`](https://www.python.org/downloads/release/python-390/).

### 2. Clone ETSim repository

### 3. Install required Python modules

See [Setup]() documentation for a list of dependencies, or run the included `install.sh` script.

## Configure the Simulator

ETSim uses configuration files to determine the details of each simulation run. There are 3 types of configuration files: the Main Configuration CSV, the Model & Data Builder Module, and the Node Configuration JSON List.

See [Configuration]() documentation for instructions on how to use these files. For now, you may use the included [example files]().

## Run

Run ETSim with Python 3.9:

`python3 etsim.py [main config csv file] [output directory]`

Progress information will be logged to standard out, and output files will be created in the `output directory`. See [Output]() for information about output files.