from keras import models
from keras import layers
from keras import optimizers

def generate_model(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (6, 6), activation='relu', input_shape=input_size))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # model.summary()
    # exit()
    
    return model