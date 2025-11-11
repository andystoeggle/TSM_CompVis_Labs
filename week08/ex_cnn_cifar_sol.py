import keras
import keras.datasets
import keras.datasets.cifar100
import keras.utils
import keras.optimizers
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger

os.environ["KERAS_BACKEND"] = "tensorflow"

print(keras.__version__)


def resnet_pre_activ(inputs, filters, regularizer_weight):
    out = keras.layers.BatchNormalization()(inputs)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Conv2D(filters, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Conv2D(filters, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    return out

def resnet_down(inputs, filters, regularizer_weight):
    """
       Downsampling resnet block where the first conv has stride 2, the returned last layer will have half the size
       Args:
           input:
           filters:
           regularizer_weight:
       """
    out = keras.layers.BatchNormalization()(inputs)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Conv2D(filters, 3, padding='same', strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Conv2D(filters, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    return out

def build_res_like_model(regularizer_weight, dropout_rate):
    inp = keras.Input(shape=(32, 32, 3))
    # 2 conv layers at the beginning
    out = keras.layers.Conv2D(32, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(inp)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Conv2D(32, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Conv2D(32, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    # Downsampling to 16x16
    res = resnet_down(out, 64, regularizer_weight)
    out = keras.layers.Conv2D(64, 1, strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.Add()([out, res])

    # 3 blocks
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])

    # downsampling to 8x8
    res = resnet_down(out, 128, regularizer_weight)
    out = keras.layers.Conv2D(128, 1, strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.Add()([out, res])

    # 3 blocks
    res = resnet_pre_activ(out, 128, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 128, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 128, regularizer_weight)
    out = keras.layers.Add()([out, res])

    # downsampling to 4x4
    res = resnet_down(out, 256, regularizer_weight)
    out = keras.layers.Conv2D(256, 1, strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.Add()([out, res])

    # 3 blocks
    res = resnet_pre_activ(out, 256, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 256, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 256, regularizer_weight)
    out = keras.layers.Add()([out, res])

    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(512, activation='relu')(out)
    out = keras.layers.Dropout(rate=dropout_rate)(out)
    out = keras.layers.Dense(512, activation='relu')(out)
    out = keras.layers.Dropout(rate=dropout_rate)(out)
    out = keras.layers.Dense(100, activation='softmax')(out)

    return keras.Model(inp, out)


def build_res_like_model_small(regularizer_weight, dropout_rate):
    inp = keras.Input(shape=(32, 32, 3))
    # 2 conv layers at the beginning


    out = keras.layers.Conv2D(32, 3, padding='same',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(inp)
    # Downsampling to 16x16
    res = resnet_down(out, 64, regularizer_weight)
    out = keras.layers.Conv2D(64, 1, strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.Add()([out, res])

    # 2 blocks
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])

    # downsampling to 8x8
    res = resnet_down(out, 64, regularizer_weight)
    out = keras.layers.Conv2D(64, 1, strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.Add()([out, res])

    # 2 blocks
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])
    res = resnet_pre_activ(out, 64, regularizer_weight)
    out = keras.layers.Add()([out, res])

    # downsampling to 4x4
    res = resnet_down(out, 128, regularizer_weight)
    out = keras.layers.Conv2D(128, 1, strides=(2, 2),
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight))(out)
    out = keras.layers.Add()([out, res])


    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(512, activation='relu')(out)
    out = keras.layers.Dropout(rate=dropout_rate)(out)
    out = keras.layers.Dense(100, activation='softmax')(out)

    return keras.Model(inp, out)

def build_model_from_base():
    base_model = keras.applications.efficientnet.EfficientNetB0(input_shape=(32, 32, 3),
                                             weights='imagenet',
                                             include_top=False,
                                             classes=100)

    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(100, activation='softmax'))


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    batch_size = 128
    epochs = 100
    regularizer_weight = 0.0002
    dropout_rate = 0.5
    learning_rate = 0.0001

    # also an option is to use a LR Scheduling, but the callback below usually works better
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)
    opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model = build_res_like_model(regularizer_weight=regularizer_weight, dropout_rate=dropout_rate)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.000001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    wandb.init(project="cifar-100-2", config={"network"     : "res-like",
                                              "bs"          : batch_size,
                                              "reg_weights" : regularizer_weight,
                                              "dropout_rate": dropout_rate, "learning_rate": learning_rate})
    model.fit(x_train,
              y_train,
              batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[WandbMetricsLogger(), reduce_lr])


if __name__ == '__main__':
    main()