from model_and_data_builder import model_builder, dataset_fn
from DatasetIterator import DatasetIterator
import keras_model
import tensorflow as tf


num_train_samples = 50000
num_test_samples = 50000

batch_size = 64
learning_rate = 0.001

eval_interval = 100
target_acc = 0.95


if __name__ == '__main__':
    model, params, forward_pass, build_optimizer = model_builder()
    dataset = dataset_fn(num_train_samples).shuffle(1024).batch(batch_size)
    di = iter(dataset)

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

    fit = True

    if fit:
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        model.fit(datagen_train.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_test, y_test), epochs=50)

    else:
        print('Beginning training')
        batch = 1
        while True:
            grads_list = forward_pass(next(di))
            optimizer.apply_gradients(zip(grads_list, params.values()))

            if batch % eval_interval == 0:
                # Evaluate model
                predictions = model.predict(x_test)            

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
                print(f'Trained {batch} batches')
                print('Test accuracy: %f' % test_accuracy)

                if test_accuracy >= target_acc:
                    break

            batch += 1