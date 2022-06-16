from enum import Enum
import threading


class UpdatePolicy(Enum):
    REPLACE = 1
    GRADIENT = 2
    AVERAGE = 3


class Node:

    def __init__(self, id, parents, update_policies, param_locations, ni):
        self.id = id

        # List of node IDs
        self.parents = parents

        # Map of parent node ID to update policy
        self.update_policies = update_policies

        # Map of parent node ID to param IDs that it holds
        self.param_locations = param_locations

        # Network Interface
        self.ni = ni

        # TODO update this documentation
        # List of ParamUpdate objects
        # ParamUpdate:
        #
        #   sender_id: id of sender
        #
        #   apply(params, optimizer)
        #       params: ParameterServer.params
        #
        #       applies update
        self.param_update_queue = []
        self.param_update_queue_cond = threading.Condition()


    # TODO consider building update parent function so it doesn't have to check parent update policy all the time
    # gradients: map of param id to gradient
    # param_values: map of param id to param value
    def update_parents(self, gradients, param_values):

        for parent_id in self.parents:

            update_policy = self.update_policies[parent_id]

            if update_policy == UpdatePolicy.REPLACE:

                # Generate vals_by_param_id
                vals_by_param_id = {}
                for param_id in self.param_locations[parent_id]:
                    vals_by_param_id[param_id] = param_values[param_id]

                # Call NI to send
                self.ni.send_params_replace(parent_id, vals_by_param_id)

            elif update_policy == UpdatePolicy.GRADIENT:

                # Generate grads_by_param_id
                grads_by_param_id = {}
                for param_id in self.param_locations[parent_id]:
                    grads_by_param_id[param_id] = gradients[param_id]

                # Call NI to send
                self.ni.send_params_gradient(parent_id, grads_by_param_id, self.id)

            elif update_policy == UpdatePolicy.AVERAGE:

                # Generate vals_by_param_id
                vals_by_param_id = {}
                for param_id in self.param_locations[parent_id]:
                    vals_by_param_id[param_id] = param_values[param_id]

                # Call NI to send
                self.ni.send_params_average(parent_id, vals_by_param_id, self.id)