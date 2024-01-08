import tensorflow as tf

# lr 0.045
# bs 96
# lr decay 0.98 per epoch

batch_size = 96
learning_rate = 0.045

optimizer_constructor = tf.keras.optimizers.Adam
loss_constructor = tf.keras.losses.CategoricalCrossentropy
train_acc_metric_constructor = tf.keras.metrics.CategoricalAccuracy


model_seed = 1  # model seed and shuffle seed (in dataset_fn) for consistent tests


def train_dataset():
  (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 255.0
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  return x_train, y_train


def test_dataset():
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_test, y_test


def build_model_with_seed(seed):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None
    )

    # Add custom top layers for CIFAR-10 classification
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    # Combine the base model and custom top layers to create the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    return model


# In dataset-rework, this just gives the master dataset which is automatically "sharded" by thread-safe DatasetIterator
# def dataset_fn(batch_size):
#     x_train, y_train = train_dataset()

#     datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         horizontal_flip=True,
#         vertical_flip=False,
#         rotation_range=10,
#         zoom_range=0.1
#     )

#     datagen_train.fit(x_train)

#     return datagen_train.flow(x_train, y_train, batch_size=batch_size)

def dataset_fn():
    x_train, y_train = train_dataset()
    cifar10_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train))
    dataset = cifar10_dataset.shuffle(len(cifar10_dataset), seed=model_seed, reshuffle_each_iteration=False)
    return dataset


def model_builder():
    model = build_model_with_seed(model_seed)
    
    p_idx = 0
    params = {}

    for param in model.trainable_variables:
        params[p_idx] = param
        p_idx += 1

    train_acc_metric = train_acc_metric_constructor()

    def forward_pass(batch, acc_metric):
        batch_inputs, batch_targets = batch
        with tf.GradientTape() as tape:
            predictions = model(batch_inputs, training=True)
            loss = loss_constructor()(batch_targets, predictions)

        grads_list = tape.gradient(loss, model.trainable_variables)
        acc_metric.update_state(batch_targets, predictions)
        
        return grads_list, loss

    def build_optimizer(learning_rate):
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     learning_rate,
        #     decay_steps=1562,
        #     decay_rate=0.98,
        #     staircase=True)
        return optimizer_constructor(learning_rate=learning_rate)

    return model, params, forward_pass, build_optimizer, loss_constructor(), train_acc_metric, batch_size, learning_rate
