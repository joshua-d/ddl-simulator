# Node Configuration Keys

## Shared Keys

***All node objects must have the following keys:***

### *Node Type*
**Key**: `node_type` \
**Value Type**: `str - 'ps' or 'worker'`

Specifies if this node is a parameter server (`ps`) or a worker (`worker`)

### *Node ID*
**Key**: `id` \
**Value Type**: `int`

Numerical ID of this node. Must be equal to the index of this node in the list.

### *Parent ID*
**Key**: `parent` \
**Value Type**: `int`

`id` of this node's parent.

### *Inbound Bandwidth*
**Key**: `inbound_bw` \
**Value Type**: `float`

Inbound bandwidth of this node in `Mbps`.

### *Outbound Bandwidth*
**Key**: `outbound_bw` \
**Value Type**: `float`

Outbound bandwidth of this node in `Mbps`.

## Parameter Server Keys

***All node objects of type parameter server must have the following keys:***

### *Synchronicity*
**Key**: `sync_style` \
**Value Type**: `str - 'sync' or 'async'`

Specifies whether this PS uses a `sync` or `async` update policy.

### *Parameter Aggregation Time*
**Key**: `aggr_time` \
**Value Type**: `float`

If `update_type` is `params`: `aggr_time` specifies the time in `seconds` for a parameter server to aggregate and save params from its children. \
If `update_type` is `grads`: `aggr_time` specifies the time in `seconds` for a `synchronous` parameter server to sum the gradients from its children.

### *Gradient Apply Time*
**Key**: `apply_time` \
**Value Type**: `float`

Only relevant if `update_type` is `grads`. Specifies the time in `seconds` for a parameter server to apply gradients to its parameters (backward pass).

## Worker Keys

***All node objects of type worker must have the following keys***:***

### *Step Time*
**Key**: `step_time` \
**Value Type**: `float`

Specifies the time in `seconds` for a worker to complete 1 step (the time from receiving parameters to having an update ready to send).

### *Step Variation*
**Key**: `step_var` \
**Value Type**: `float`

Specifies a variation (in `seconds`) that will be applied to `step_time` to determine actual worker step time.

Actual worker step time =  `step_time` +- U(0, `step_var`)

In other words, a random number between 0 and `step_var` (of uniform distribution) will be added to or subtracted from `step_time`.

### *Dropout Chance*
**Key**: `dropout_chance` \
**Value Type**: `float`

Specifies the probability that this worker drops out after performing a step. For example, if `dropout_chance` is `0.1`, then this worker has a `10%` chance of dropping out each time it performs a step.