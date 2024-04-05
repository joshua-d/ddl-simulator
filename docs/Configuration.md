# Configuration

ETSim uses configuration files to determine the details of each simulation run. There are 3 types of configuration files: the Main Configuration CSV, the Model & Data Builder Module, and the Node Configuration JSON.
\
&nbsp;
## Main Configuration CSV

### File Details 

This config file provides the main controls for the simulator. It can also define the topology and attributes of the distributed system.

It is a CSV file whose cells are separated by tab characters (`\t`).

The first row (the **key row**) is made up of keys, each representing a configuration control for the simulator. The following rows (**value rows**) are made up of values for these controls.

Each value row creates **one configuration**. The simulator will run **one simulation** for each configuration, sequentially. The configurations depicted by the rows (and the simulations they configure) are completely independent from one another.

Here is a visual snippet of a config csv that defines 2 different configurations for the simulator to run:

    topology    sync-config     epochs  target_acc_train    ...
    2-2         S-S             10      0.95    
    4-4         A-A             20      0.95    

### Configuration Keys

#### *Always-Required Keys*

**Key**: `madb_file` \
**Value Type**: `str` \
Specifies the [Model & Data Builder Module]().

**Key**: `trainless` \
**Value Type**: `0 or 1` \
Run the simulation in [trainless mode]().

**Key**: `epochs` \
**Value Type**: `float` \
Maximum number of epochs to simulate.

#### *Training Controls*

**Key**: `target_acc_train` \
**Value Type**: `float` \
Target accuracy on the train dataset. Epoch and time at which this is reached will be logged.

**Key**: `target_acc_test` \
**Value Type**: `float` \
Target accuracy on the test dataset. Epoch and time at which this is reached will be logged.

**Key**: `stop_at_target_train` \
**Value Type**: `0 or 1` \
Stop the simulation when target train accuracy is reached.

**Key**: `stop_at_target_test` \
**Value Type**: `0 or 1` \
Stop the simulation when target test accuracy is reached.

#### *Simulator Options*

**Key**: `generate_gantt` \
**Value Type**: `0 or 1` \
**Default Value**: `0` \
Generate the schedule chart for this simulation.

**Key**: `bypass_NI` \
**Value Type**: `0 or 1` \
**Default Value**: `0` \
Bypass the network simulation interface - all transmissions between nodes are sent and received immediately.

**Key**: `n_runs` \
**Value Type**: `int` \
**Default Value**: `1` \
Number of runs of this configuration. Equivalent to having `n_runs ` copies of this row.

#### *System Attributes*

**Key**: `update_type` \
**Value Type**: `'str - 'params' or 'grads'` \
**Default Value**: `params` \
Type of worker update. \
`params` - worker applies backward pass to its own copy of the model params, then sends the new param values up to its parent parameter server. \
`grads` - worker sends gradients (obtained from forward pass) up to its parent parameter server, who applies the backward pass.

**Key**: `network_style` \
**Value Type**: `str - 'hd' or 'fd'` \
**Default Value**: `hd` \
Specifies whether the network is half duplex (`hd`) or full duplex (`fd`). If half duplex, for a node who is sending and receiving data at the same time, transmission rates are halfed.

**Key**: `rb_strat` \
**Value Type**: `str - 'none', 'nbbl', 'sbbl', 'bwbbl', or 'oabbl'` \
**Default Value**: `none` \
Specifies rebalancing strategy. If worker dropout is enabled, and the topology is 2-level with at least 2 subclusters: \
`none` - Don't rebalance. \
`nbbl` - Numerically balanced bottom layer. Rebalance based on number of workers per subcluster. \
`sbbl` - Speed-balanced bottom layer. Rebalance based on worker speed.
`bwbbl` - Bandwidth-balanced bottom layer. Rebalance based on worker bandwidth. \
`oabbl` - Overall-balanced bottom layer. Rebalance based on worker speed and bandwidth.

**Key**: `node_config_file` \
**Value Type**: `str` \
Specifies a [Node Configuration JSON file]().

#### *If `node_config_file` is not specified, then the following keys must be present. If it is specified, all of the following keys will be overridden if present.*

**Key**: `topology` \
**Value Type**: `str` \
Hyphen-separated `int`s representing the number of workers in each subcluster. Supports 1-level and 2-level topologies. Ex: \
`4` : 1-level - 4 workers, 1 parameter server \
`2-2-2` : 2-level - 3 subclusters, each with 2 workers and 1 mid-level parameter server. 1 top-level parameter server.

**Key**: `sync_config` \
**Value Type**: `str` \
Hyphen-separated `S`s and `A`s representing the synchronicity of a level. Supports 1-level and 2-level topologies. Ex: \
`S` : 1-level - parameter server is synchronous \
`A-S` : 2 level, - top-level parameter server is asynchronous, mid-level parameter servers are synchronous.

**Key**: `bw` \
**Value Type**: `float` \
Specifies the inbound and outbound bandwidth of every node in the system in `Mbps`.

**Key**: `w_step_time` \
**Value Type**: `float` \
Specifies the time in `seconds` for a worker to complete 1 step (the time from receiving parameters to having an update ready to send).

**Key**: `w_step_var` \
**Value Type**: `float` \
Specifies a variation (in `seconds`) that will be applied to `w_step_time` to determine actual worker step time. \
Actual worker step time =  `w_step_time` +- U(0, `w_step_var`) \
In other words, a random number between 0 and `w_step_var` (of uniform distribution) will be added to or subtracted from `w_step_time`.

**Key**: `ps_aggr_time` \
**Value Type**: `float` \
If `update_type` is `params`: `ps_aggr_time` specifies the time in `seconds` for a parameter server to aggregate and save params from its children. \
If `update_type` is `grads`: `ps_aggr_time` specifies the time in `seconds` for a `synchronous` parameter server to sum the gradients from its children.

**Key**: `ps_apply_time` \
**Value Type**: `float` \
Only relevant if `update_type` is `grads`. Specifies the time in `seconds` for a parameter server to apply gradients to its parameters (backward pass).

**Key**: `dropout_chance` \
**Value Type**: `float` \
Specifies, for each worker, the probability that it drops out after performing a step. For example, if `dropout_chance` is `0.1`, then each worker has a `10%` chance of dropping out each time it performs a step.
\
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

This file contains a list of objects, each representing a node. There are 2 types of nodes: parameter servers and workers.

### Shared Keys

All node objects must have the following keys:

**Key**: `node_type` \
**Value Type**: `str - 'ps' or 'worker'` \
Specifies if this node is a parameter server (`ps`) or a worker (`worker`)

**Key**: `id` \
**Value Type**: `int` \
Numerical ID of this node. Must be equal to the index of this node in the list.

**Key**: `parent` \
**Value Type**: `int` \
`id` of this node's parent.

**Key**: `inbound_bw` \
**Value Type**: `float` \
Inbound bandwidth of this node in `Mbps`.

**Key**: `outbound_bw` \
**Value Type**: `float` \
Outbound bandwidth of this node in `Mbps`.

### Parameter Server Keys

All node objects of type parameter server must have the following keys:

**Key**: `sync_style` \
**Value Type**: `str - 'sync' or 'async'` \
Specifies whether this PS uses a `sync` or `async` update policy.

**Key**: `aggr_time` \
**Value Type**: `float` \
If `update_type` is `params`: `aggr_time` specifies the time in `seconds` for a parameter server to aggregate and save params from its children. \
If `update_type` is `grads`: `aggr_time` specifies the time in `seconds` for a `synchronous` parameter server to sum the gradients from its children.

**Key**: `apply_time` \
**Value Type**: `float` \
Only relevant if `update_type` is `grads`. Specifies the time in `seconds` for a parameter server to apply gradients to its parameters (backward pass).

### Worker Keys

All node objects of type worker must have the following keys:

**Key**: `step_time` \
**Value Type**: `float` \
Specifies the time in `seconds` for a worker to complete 1 step (the time from receiving parameters to having an update ready to send).

**Key**: `step_var` \
**Value Type**: `float` \
Specifies a variation (in `seconds`) that will be applied to `step_time` to determine actual worker step time. \
Actual worker step time =  `step_time` +- U(0, `step_var`) \
In other words, a random number between 0 and `step_var` (of uniform distribution) will be added to or subtracted from `step_time`.

**Key**: `dropout_chance` \
**Value Type**: `float` \
Specifies the probability that this worker drops out after performing a step. For example, if `dropout_chance` is `0.1`, then this worker has a `10%` chance of dropping out each time it performs a step.