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


# this just used to create the vars on the parameter server
initial_model = keras_model.build_model()

params_1 = {
    'K1': initial_model.layers[1].kernel,
    'B1': initial_model.layers[1].bias
}

params_2 = {
    'K2': initial_model.layers[2].kernel,
    'B2': initial_model.layers[2].bias
}

ps_1 = ParameterServer(params_1, tf.keras.optimizers.RMSprop(learning_rate=0.1))
ps_2 = ParameterServer(params_2, tf.keras.optimizers.RMSprop(learning_rate=0.1))


cl = Cluster()
cl.parameter_servers = {
    'ps1': ps_1,
    'ps2': ps_2
}

ds = keras_model.mnist_dataset().batch(10)
param_locations = {
    'ps1': ['K1', 'B1'],
    'ps2': ['K2', 'B2']
}
w1 = Worker(cl, model_builder, iter(ds.take(1000)), param_locations)
w2 = Worker(cl, model_builder, iter(ds.skip(1000).take(1000)), param_locations)


def eval_once():
    num_test_samples = 300
    x_test, y_test = keras_model.test_dataset(num_test_samples)
    predictions = initial_model.predict(x_test)

    num_correct = 0
    for prediction, target in zip(predictions, y_test):
        answer = 0
        answer_val = prediction[0]
        for poss_ans_ind in range(len(prediction)):
            if prediction[poss_ans_ind] > answer_val:
                answer = poss_ans_ind
                answer_val = prediction[poss_ans_ind]
        if answer == target:
            num_correct += 1

    test_accuracy = float(num_correct) / num_test_samples
    print('Test accuracy: %f' % test_accuracy)


eval_once()

for _ in range(1000):
    w1.train()
    w2.train()

eval_once()