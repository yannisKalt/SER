from tools_no_librosa import *
from tensorflow.keras.callbacks import ModelCheckpoint
def cnn():
    X_train, X_test, Y_train, Y_test = split_dataset(*aesdd_spectrogram(), 0.7)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape = X_train.shape[1:]))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(12, 5, (3,2), activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(12, 5, (3,2), activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Conv2D(32, 5, (3,2), activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(32, 5, (3,2), activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(64, 3, (3,1), activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(5, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])

    filepath = 'weights.best.hdf5'
    #checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', verbose = 1 ,
    #                             save_best_only = True, mode = 'max')
    #callbacks_list = [checkpoint]
    #hist = model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 52,
    #                 callbacks = callbacks_list)
    
    model.load_weights(filepath)
    return model, X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    pass
