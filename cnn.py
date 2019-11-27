from tools_no_librosa import *

def cnn():
    X_train, X_test, Y_train, Y_test = split_dataset(*aesdd_spectrogram(), 0.7)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape = X_train.shape[1:]))


    model.add(keras.layers.Conv2D(12, 3, 2, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Conv2D(12, 3, 2, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Conv2D(12, 3, 2, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Conv2D(12, 3, 2, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Conv2D(12, 3, 2, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Conv2D(12, 3, 2, activation = 'relu'))
    model.add(keras.layers.Dropout(0.3))


    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(32, activation = 'relu'))
    model.add(keras.layers.Dense(5, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, Y_train, validation_data = [X_test, Y_test], epochs = 600)
    return model


if __name__ == '__main__':
    model = cnn()
    print(model.summary())
