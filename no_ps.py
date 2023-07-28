from model_and_data_builder import model_builder, dataset_fn
from DatasetIterator import DatasetIterator
import keras_model
import tensorflow as tf
import math
from time import perf_counter


num_train_samples = 50000
num_test_samples = 50000

batch_size = 64
learning_rate = 0.1


def get_next_batch(data_chunk_iterator, batch_idx, data_chunk_size, dataset_iterator):
    batch = next(data_chunk_iterator)
    batch_idx += 1

    if batch_idx == data_chunk_size:
        chunk = next(dataset_iterator)
        data_chunk_size = len(chunk)
        data_chunk_iterator = iter(chunk)
        batch_idx = 0

    return batch, data_chunk_iterator, batch_idx, data_chunk_size


if __name__ == '__main__':
    model, params, forward_pass, build_optimizer = model_builder()
    dataset = dataset_fn(num_train_samples).shuffle(1024)#.batch(batch_size)#.repeat(5)
    di = DatasetIterator(dataset, batch_size, 64)

    optimizer = build_optimizer(learning_rate)

    x_train, y_train = keras_model.train_dataset()
    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        rotation_range=10,
        zoom_range=0.1
    )

    datagen_train.fit(x_train)

    x_test, y_test = keras_model.test_dataset(num_test_samples)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    fit = False

    if fit:
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        model.fit(tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size), validation_data=(x_test, y_test), epochs=150) # datagen_train.flow(x_train, y_train, batch_size=batch_size)

    else:
        print('Beginning training')
        n_batch = 1
        batch_idx = 0
        batches_per_epoch = math.ceil(num_train_samples / batch_size)
        start_time = perf_counter()
        chunk = next(di)
        data_chunk_iterator = iter(chunk)
        data_chunk_size = len(chunk)

        while True:
            batch, data_chunk_iterator, batch_idx, data_chunk_size = get_next_batch(data_chunk_iterator, batch_idx, data_chunk_size, di)
            grads_list, loss = forward_pass(batch)
            optimizer.apply_gradients(zip(grads_list, params.values()))

            if n_batch % batches_per_epoch == 0:
                print(f'Trained {n_batch} batches')
                print(f'{perf_counter() - start_time}s per epoch')
                start_time = perf_counter()

            # if batch % eval_interval == 0:
            #     # Evaluate model
            #     predictions = model.predict(x_test)            

            #     num_correct = 0
            #     for prediction, target in zip(predictions, y_test):
            #         answer = 0
            #         answer_val = prediction[0]
            #         for poss_ans_ind in range(len(prediction)):
            #             if prediction[poss_ans_ind] > answer_val:
            #                 answer = poss_ans_ind
            #                 answer_val = prediction[poss_ans_ind]
            #         if answer == target:
            #             num_correct += 1

            #     test_accuracy = float(num_correct) / num_test_samples
            #     print(f'Trained {batch} batches')
            #     print('Test accuracy: %f' % test_accuracy)

            #     if test_accuracy >= target_acc:
            #         break

            n_batch += 1


    # FROM meas-train
    
    n_batches = 1000
    fp_time = 0
    opt_time = 0

    for i in range(n_batches):
        b = next(di)

        fp_start = perf_counter()
        grads_list = forward_pass(b)
        fp_time += perf_counter() - fp_start

        opt_start = perf_counter()
        optimizer.apply_gradients(zip(grads_list, params.values()))
        opt_time += perf_counter() - opt_start

    fp_time /= n_batches
    opt_time /= n_batches

    print(f'fp: {fp_time}')
    print(f'opt: {opt_time}')


    n_param_sets = [2, 4, 6, 8, 10, 12, 14, 16]
    n_aggs = 1000
    for n_param in n_param_sets:

        param_sets = []

        for i in range(n_param):
            _, params, _, _ = model_builder()
            param_sets.append(params)
            
        agg_time = 0

        for i in range(n_aggs):
            agg_start = perf_counter()

            for param_id in param_sets[0]:
                param_value = 0

                for param_set in param_sets:
                    param_value += param_set[param_id]
                
                param_value /= len(param_sets)

                param_sets[0][param_id].assign(param_value)

            agg_time += perf_counter() - agg_start

        agg_time /= n_aggs
        print(f'{n_param} sets, {n_aggs} aggs: {agg_time}')


        
