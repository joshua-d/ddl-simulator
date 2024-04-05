# Main Configuration Keys

## Always-Required Keys

### *MADB File*
**Key**: `madb_file` \
**Value Type**: `str`

Specifies the [Model & Data Builder Module]().

### *Trainless Mode*
**Key**: `trainless` \
**Value Type**: `0 or 1`

Run the simulation in [trainless mode]().

### *Epochs*
**Key**: `epochs` \
**Value Type**: `float`

Maximum number of epochs to simulate.

## Training Controls

### *Target Train Accuracy*
**Key**: `target_acc_train` \
**Value Type**: `float`

Target accuracy on the train dataset. Epoch and time at which this is reached will be logged.

### *Target Test Accuracy*
**Key**: `target_acc_test` \
**Value Type**: `float`

Target accuracy on the test dataset. Epoch and time at which this is reached will be logged.

### *Stop At Target Train Accuracy*
**Key**: `stop_at_target_train` \
**Value Type**: `0 or 1`

Stop the simulation when target train accuracy is reached.

### *Stop At Target Test Accuracy*
**Key**: `stop_at_target_test` \
**Value Type**: `0 or 1`

Stop the simulation when target test accuracy is reached.


## Simulator Options

### *Generate Schedule Chart*
**Key**: `generate_gantt` \
**Value Type**: `0 or 1` \
**Default Value**: `0`

Generate the schedule chart for this simulation.

### *Bypass Network Simulation*
**Key**: `bypass_NI` \
**Value Type**: `0 or 1` \
**Default Value**: `0`

Bypass the network simulation interface - all transmissions between nodes are sent and received immediately.

### *Number of Runs*
**Key**: `n_runs` \
**Value Type**: `int` \
**Default Value**: `1`

Number of runs of this configuration. Equivalent to having `n_runs ` copies of this row.

## System Attributes

### *Update Type*
**Key**: `update_type` \
**Value Type**: `'str - 'params' or 'grads'` \
**Default Value**: `params`

Type of worker update. 

`params` - worker applies backward pass to its own copy of the model params, then sends the new param values up to its parent parameter server. \
`grads` - worker sends gradients (obtained from forward pass) up to its parent parameter server, who applies the backward pass.

### *Network Style*
**Key**: `network_style` \
**Value Type**: `str - 'hd' or 'fd'` \
**Default Value**: `hd`

Specifies whether the network is half duplex (`hd`) or full duplex (`fd`). If half duplex, for a node who is sending and receiving data at the same time, transmission rates are halfed.

### *Rebalancing Strategy*
**Key**: `rb_strat` \
**Value Type**: `str - 'none', 'nbbl', 'sbbl', 'bwbbl', or 'oabbl'` \
**Default Value**: `none`

Specifies rebalancing strategy. If worker dropout is enabled, and the topology is 2-level with at least 2 subclusters:

`none` - Don't rebalance. \
`nbbl` - Numerically balanced bottom layer. Rebalance based on number of workers per subcluster. \
`sbbl` - Speed-balanced bottom layer. Rebalance based on worker speed.
`bwbbl` - Bandwidth-balanced bottom layer. Rebalance based on worker bandwidth. \
`oabbl` - Overall-balanced bottom layer. Rebalance based on worker speed and bandwidth.

### *Node Configuration JSON File*
**Key**: `node_config_file` \
**Value Type**: `str`

Specifies a [Node Configuration JSON file]().

### If `node_config_file` is not specified, then the following keys must be present. If it is specified, all of the following keys will be overridden if present.

### *System Topology*
**Key**: `topology` \
**Value Type**: `str`

Hyphen-separated `int`s representing the number of workers in each subcluster. Supports 1-level and 2-level topologies. Ex:

`4` : 1-level - 4 workers, 1 parameter server \
`2-2-2` : 2-level - 3 subclusters, each with 2 workers and 1 mid-level parameter server. 1 top-level parameter server.

### *System Synchronicity Configuration*
**Key**: `sync_config` \
**Value Type**: `str`

Hyphen-separated `S`s and `A`s representing the synchronicity of a level. Supports 1-level and 2-level topologies. Ex:

`S` : 1-level - parameter server is synchronous \
`A-S` : 2 level, - top-level parameter server is asynchronous, mid-level parameter servers are synchronous.

### *Bandwidth*
**Key**: `bw` \
**Value Type**: `float`

Specifies the inbound and outbound bandwidth of every node in the system in `Mbps`.

### *Worker Step Time*
**Key**: `w_step_time` \
**Value Type**: `float`

Specifies the time in `seconds` for a worker to complete 1 step (the time from receiving parameters to having an update ready to send).

### *Worker Step Variation*
**Key**: `w_step_var` \
**Value Type**: `float`

Specifies a variation (in `seconds`) that will be applied to `w_step_time` to determine actual worker step time.

Actual worker step time =  `w_step_time` +- U(0, `w_step_var`)

In other words, a random number between 0 and `w_step_var` (of uniform distribution) will be added to or subtracted from `w_step_time`.

### *PS Aggregation Time*
**Key**: `ps_aggr_time` \
**Value Type**: `float`

If `update_type` is `params`: `ps_aggr_time` specifies the time in `seconds` for a parameter server to aggregate and save params from its children. \
If `update_type` is `grads`: `ps_aggr_time` specifies the time in `seconds` for a `synchronous` parameter server to sum the gradients from its children.

### *PS Gradient Apply Time*
**Key**: `ps_apply_time` \
**Value Type**: `float`

Only relevant if `update_type` is `grads`. Specifies the time in `seconds` for a parameter server to apply gradients to its parameters (backward pass).

### *Worker Dropout Chance*
**Key**: `dropout_chance` \
**Value Type**: `float`

Specifies, for each worker, the probability that it drops out after performing a step. For example, if `dropout_chance` is `0.1`, then each worker has a `10%` chance of dropping out each time it performs a step.