import tensorflow as tf
import keras_model
from Worker import Worker
from ParameterServer import ParameterServer
from Cluster import Cluster


def model_builder():
    model = keras_model.build_model()
    params = {
        'K1': model.layers[1].kernel,
        'B1': model.layers[1].bias,
        'K2': model.layers[2].kernel,
        'B2': model.layers[2].bias
    }
    def forward_pass(batch):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        gradients = {
            'K1': grads_list[0],
            'B1': grads_list[1],
            'K2': grads_list[2],
            'B2': grads_list[3]
        }
        return gradients

    return model, params, forward_pass



initial_model = keras_model.build_model()

params = {
    'K1': initial_model.layers[1].kernel,
    'B1': initial_model.layers[1].bias,
    'K2': initial_model.layers[2].kernel,
    'B2': initial_model.layers[2].bias
}




ps = ParameterServer(params, tf.keras.optimizers.RMSprop(learning_rate=0.1))

cl = Cluster()
cl.parameter_servers = {
    'ps1': ps
}

ds = keras_model.mnist_dataset().batch(10)
param_locations = {
    'ps1': ['K1', 'B1', 'K2', 'B2']
}
w1 = Worker(cl, model_builder, iter(ds.take(100)), param_locations)
w2 = Worker(cl, model_builder, iter(ds.skip(100).take(100)), param_locations)

w1.train()
w1.train()
