# ETSim

A distributed deep learning simulator using discrete event simulation, built atop TensorFlow. Focused on edge-based tree-like parameter server hierarchy training. 

Independently developed by Joshua Daley, M.S. Computer Science; under supervision of Dr. Yifan Zhang, Binghamton University Computer Science Department.

# Basic Usage

##  Setup

### 1. Install Python 3.9

ETSim requires [`Python 3.9.x`](https://www.python.org/downloads/release/python-390/).

### 2. Install required Python modules

ETSim requires [TensorFlow 2.6]() and [numpy]().

    $ python3.9 -m pip install numpy
    $ python3.9 -m pip install tensorflow==2.6.0

*To enable GPU use, install [CUDA 11]() and [cudnn]().*

### 3. Clone the ETSim repo

    $ git clone https://github.com/joshua-d/ddl-simulator.git

## Configure the Simulator

ETSim uses configuration files to determine the details of each simulation run. There are 3 types of configuration files: the Main Configuration CSV, the Model & Data Builder Module, and the Node Configuration JSON.

See [Configuration]() documentation for instructions on how to use these files. For now, you may use the included [example files]().

## Run

Run ETSim with Python 3.9:

`python3 etsim.py [main config csv file] [output directory]`

Progress information will be logged to standard out, and output files will be created in the `output directory`. See [Output]() for information about output files.