# Configuration

ETSim uses configuration files to determine the details of each simulation run. There are 3 types of configuration files: the Main Configuration CSV, the Model & Data Builder Module, and the Node Configuration JSON List.


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

<table>
    <tr>
        <th>Key</th><th>Value Type</th><th>Description</th>
    </tr>
<tr>
<td>

`trainless`

</td>
<td>

`0 or 1`

</td>
<td>

Run the simulation in [trainless mode]().

</td>
</tr>
</table>

#### *Always Required Keys*

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

