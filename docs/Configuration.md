# Configuration

ETSim uses configuration files to determine the details of each simulation run. There are 3 types of configuration files: the Main Configuration CSV, the Model & Data Builder Module, and the Node Configuration JSON.

&nbsp;

## Main Configuration CSV

This config file provides the main controls for the simulator. It can also define the topology and attributes of the distributed system.

It is a CSV file whose cells are separated by tab characters (`\t`).

The first row (the **key row**) is made up of keys, each representing a configuration control for the simulator. The following rows (**value rows**) are made up of values for these controls.

Each value row creates **one configuration**. The simulator will run **one simulation** for each configuration, sequentially. The configurations depicted by the rows (and the simulations they configure) are completely independent from one another.

Here is a visual snippet of a config csv that defines 2 different configurations for the simulator to run:

    topology    sync-config     epochs  target_acc_train    ...
    2-2         S-S             10      0.95    
    4-4         A-A             20      0.95    

[Main Configuration Keys]() contains a list of all keys and their documentation.

&nbsp;

## Model & Data Builder Module

The Model & Data Builder (MADB) Module is a Python module that exposes utilities to the simulator that allow it to set up the model and the dataset. The path to the module must be specified in the [Main Configuration CSV](). Examples are provided in the `madb` directory of this repository.

### Requirements

#### Train Dataset Function

The module must expose a function `train_dataset_fn()` that returns a TensorFlow Dataset that holds the training data and labels, like what's returned from `tf.data.Dataset.from_tensor_slices(...)`. The dataset should be shuffled. It will be batched, repeated, and reshuffled automatically later on.

#### Test Dataset Function

The module must expose a function `test_dataset_fn()` that returns a tuple `(x_test, y_test)`, where `x_test` is a tensor containing test data inputs, and `y_test` is a tensor containing test data labels, like what's returned from `tf.keras.datasets.[dataset name].load_data()`.

#### Model Builder Function

The module must expose a function `model_builder()`. This function returns a tuple containing:

`model` - a keras model.

`params` - a dictionary mapping IDs to each of the model's trainable variables. *Currently, must be created by adding elements from `model.trainable_variables` to the dictionary in order. IDs should just be numerical indices.*

`forward_pass` - a function that takes a batch of data, performs a forward pass using the model, and returns the resulting gradients. See [Examples]().

`build_optimizer` - a function that returns a `tf.keras.Optimizer` for performing a backward pass (i.e. applying gradients, optimizing).

`loss_type` - a `tf.keras.Loss` object representing the type of loss measured for this model.

`train_acc_metric` - a `tf.keras.Metric` used for measuring the train accuracy.

`batch_size` &

`learning_rate`.
\
&nbsp;
## Node Configuration JSON

A Node Configuration JSON file, though not required, allows for more control over the attributes of individual nodes in the system. To use one, it must be specified in the [Main Configuration CSV]() with the `node_config_file` key.

This file contains a list of objects, each representing a node. There are 2 types of nodes: parameter servers and workers. [Node Configuration Keys]() contains a list of all required keys for each node and their documentation.

