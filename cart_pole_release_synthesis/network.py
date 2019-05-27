from keras import models
from keras import layers
from keras import optimizers

def generate_model(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_size))#, strides=(4,4), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))#, strides=(2,2), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))#, strides=(2,2), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))#, strides=(2,2), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    model.summary()
    exit()

    return model

def generate_model_from_list(architecture, input_size):
    num_layers = len(architecture)
    
    model = models.Sequential()
    model.add(layers.Dense(4, activation='sigmoid', input_shape=input_size))

    for i in range(num_layers):
        num_neurons = architecture[i]
        model.add(layers.Dense(num_neurons, activation='sigmoid'))
    
    model.add(layers.Dense(1, activation='sigmoid'))

    model.trainable = False
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # model.summary()
    # exit()

    return model    
    